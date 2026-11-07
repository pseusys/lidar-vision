"""Item 81: DROW's wheelchairs and walkers stop being trained as background.

**The defect.** 56.4% of the objects DROW annotates in its test split are not pedestrians -- 1,970 wheelchairs
and 581 walkers against 1,976 pedestrians -- and the trainer reads `det_wp` alone. So every one of them is a
knee-height, person-shaped return that the loss teaches as **background** and the metric scores as a **false
positive** when the network finds it anyway. A person in a wheelchair is a person.

**The treatment is an ignore region, not a label.** Those beams leave the classification term entirely --
neither positive nor negative -- which is the standard handling for a class present in the scene and absent from
the labels. Turning them into positives would be a different claim (that the detector reports wheelchairs), and
that is item 84's multi-class head.

**Three things must not move**, and each has a test here: the **vote** term, since a wheelchair beam has no
person centre to regress towards; `pos_weight`'s **positive** count, which is a count of annotated people; and
**FROG**, which has one class, so its mask is empty and every number on record is reproduced bit for bit.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import train as classic  # noqa: E402
import train_takheliped as tak  # noqa: E402

#: A 16-beam fan either side of straight ahead, wide enough that two objects 2 m apart own different beams.
ANGLES = np.linspace(-0.4, 0.4, 16)
SCAN = np.full(16, 5.0, np.float32)


def scan_towards(bearing, distance=5.0):
    """A scan whose every beam reads `distance`, so a label at `(distance, bearing)` sits on the beam at that bearing."""
    return np.full(len(ANGLES), distance, np.float32)


class TestWhichBeamsAreIgnored:
    def test_a_beam_at_a_non_person_object_is_ignored(self):
        mask = tak.ignore_beams(SCAN, ANGLES, [(5.0, 0.0)], radius=0.6, max_range_m=30.0)
        assert mask[len(ANGLES) // 2] or mask[len(ANGLES) // 2 - 1]

    def test_a_beam_far_from_every_object_is_not(self):
        mask = tak.ignore_beams(SCAN, ANGLES, [(5.0, 0.4)], radius=0.6, max_range_m=30.0)
        assert not mask[0]

    def test_no_objects_ignores_nothing(self):
        mask = tak.ignore_beams(SCAN, ANGLES, [], radius=0.6, max_range_m=30.0)
        assert mask.shape == (len(ANGLES),) and not mask.any()

    def test_the_mask_uses_the_same_geometry_as_the_label_it_replaces(self):
        """Built by `make_targets` itself, so a beam is ignored exactly where it would have been a positive."""
        objects = [(5.0, 0.1)]
        mask = tak.ignore_beams(SCAN, ANGLES, objects, radius=0.6, max_range_m=30.0)
        labels, _ = classic.make_targets(tak.anchor_ranges(SCAN, 30.0), ANGLES,
                                         {3: [tuple(o) for o in objects]}, vote_collect_radius=0.6)
        assert mask.tolist() == (labels > 0).tolist()

    def test_a_person_keeps_a_beam_the_two_of_them_share(self):
        """Precedence, and it has to be this way round: a wheelchair beside a person must not silence the person."""
        people, objects = [(5.0, 0.0)], [(5.0, 0.02)]
        labels, _ = classic.make_targets(tak.anchor_ranges(SCAN, 30.0), ANGLES, {3: people}, vote_collect_radius=0.6)
        mask = tak.ignore_beams(SCAN, ANGLES, objects, radius=0.6, max_range_m=30.0, positives=labels > 0)
        assert not (mask & (labels > 0)).any()
        assert mask.any()


class TestTheLossDropsThoseBeamsFromTheClassificationTermOnly:
    def setup_method(self):
        torch.manual_seed(0)
        self.logits = torch.zeros(6, 1, requires_grad=True)
        self.votes = torch.zeros(6, 2)
        self.labels = torch.tensor([3, 0, 0, 0, 0, 0])
        self.vote_targets = torch.zeros(6, 2)

    def test_with_no_mask_it_is_bit_identical_to_what_produced_every_number_on_record(self):
        before = classic.compute_loss(self.logits, self.votes, self.labels, self.vote_targets)
        after = classic.compute_loss(self.logits, self.votes, self.labels, self.vote_targets, ignore=None)
        assert after[0].item() == before[0].item() and after[1] == before[1] and after[2] == before[2]

    def test_an_all_false_mask_is_also_bit_identical(self):
        """So `--ignore-other-classes` on a one-class dataset cannot move a FROG number by a rounding error."""
        before = classic.compute_loss(self.logits, self.votes, self.labels, self.vote_targets)
        after = classic.compute_loss(self.logits, self.votes, self.labels, self.vote_targets,
                                     ignore=torch.zeros(6, dtype=torch.bool))
        assert after[0].item() == pytest.approx(before[0].item(), abs=0.0)

    def test_a_confident_mistake_on_an_ignored_beam_costs_nothing(self):
        ignore = torch.tensor([False, True, False, False, False, False])
        quiet = torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        shouting = torch.tensor([[0.0], [9.0], [0.0], [0.0], [0.0], [0.0]])
        a = classic.compute_loss(quiet, self.votes, self.labels, self.vote_targets, ignore=ignore)
        b = classic.compute_loss(shouting, self.votes, self.labels, self.vote_targets, ignore=ignore)
        assert b[1] == pytest.approx(a[1])

    def test_the_same_mistake_on_a_kept_beam_does_cost(self):
        """The guard on the test above: it must be the mask doing the work and not a dead gradient."""
        ignore = torch.tensor([False, True, False, False, False, False])
        shouting = torch.tensor([[0.0], [0.0], [9.0], [0.0], [0.0], [0.0]])
        a = classic.compute_loss(self.logits, self.votes, self.labels, self.vote_targets, ignore=ignore)
        b = classic.compute_loss(shouting, self.votes, self.labels, self.vote_targets, ignore=ignore)
        assert b[1] > a[1]

    def test_an_ignored_beam_stops_counting_as_a_negative_in_the_class_balance(self):
        """`pos_weight` is negatives over positives, so masking a negative must raise the weight, not leave it."""
        labels = torch.tensor([3, 0, 0, 0])
        logits, votes, targets = torch.zeros(4, 1), torch.zeros(4, 2), torch.zeros(4, 2)
        whole = classic.compute_loss(logits, votes, labels, targets)
        masked = classic.compute_loss(logits, votes, labels, targets,
                                      ignore=torch.tensor([False, True, True, False]))
        assert masked[1] != pytest.approx(whole[1])

    def test_the_vote_term_is_untouched(self):
        """A wheelchair beam has no person centre to regress to, so the mask must not reach localisation."""
        votes = torch.full((6, 2), 0.5)
        ignore = torch.tensor([False, True, False, False, False, False])
        assert (classic.compute_loss(self.logits, votes, self.labels, self.vote_targets, ignore=ignore)[2]
                == pytest.approx(classic.compute_loss(self.logits, votes, self.labels, self.vote_targets)[2]))

    def test_a_soft_target_is_masked_the_same_way(self):
        """The adopted recipe trains on `--label-sigma-m 0.3`, so the soft path is the one that will actually run."""
        soft = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ignore = torch.tensor([False, True, False, False, False, False])
        quiet = torch.zeros(6, 1)
        shouting = torch.tensor([[0.0], [9.0], [0.0], [0.0], [0.0], [0.0]])
        a = classic.compute_loss(quiet, self.votes, self.labels, self.vote_targets, soft_labels=soft, ignore=ignore)
        b = classic.compute_loss(shouting, self.votes, self.labels, self.vote_targets, soft_labels=soft, ignore=ignore)
        assert b[1] == pytest.approx(a[1])


class TestTheNonPersonAnnotationsReachTheLoader:
    def test_drow_names_the_two_annotation_files_the_trainer_never_opened(self):
        assert tak.DATASETS["drow"].other_detections == ("det_wc", "det_wa")

    def test_frog_has_none_because_it_has_one_class(self):
        assert tak.DATASETS["frog"].other_detections == ()
        assert tak.DATASETS["frog"].classes == ("pedestrian",)

    def test_every_named_file_is_a_class_the_spec_already_declared(self):
        """Two places could disagree about how many classes a dataset has; this is the one that notices."""
        for spec in tak.DATASETS.values():
            assert len(spec.other_detections) == len(spec.classes) - 1

    def test_a_split_carries_them_per_annotated_frame_beside_the_people(self):
        ds = SimpleNamespace(
            scans=[np.stack([np.full(8, float(i)) for i in range(4)]).astype(np.float32)],
            odoms=[np.zeros(4, dtype=[("eq", np.uint32), ("t", np.float64), ("xya", np.float32, 3)])],
            scan_time=[np.array([10.0, 10.08, 10.16, 10.24])],
            det_id=[np.array([1, 3])], idet2iscan=[{0: 1, 1: 3}],
            det_wp=[[[(2.0, 0.0)], []]],
            det_wc=[[[], [(3.0, 0.1)]]],
            det_wa=[[[], [(4.0, -0.1)]]],
        )
        split = tak.frames_from_dataset(ds, tak.DATASETS["drow"])
        assert len(split.other_polar) == len(split.people_polar) == 2
        assert len(split.other_polar[0]) == 0
        assert split.other_polar[1][:, 0].tolist() == pytest.approx([3.0, 4.0])

    def test_a_one_class_dataset_gives_an_empty_array_per_frame_and_not_a_missing_one(self):
        """So the training loop indexes it unconditionally instead of asking which dataset it is on."""
        ds = SimpleNamespace(
            scans=[np.stack([np.full(8, float(i)) for i in range(2)]).astype(np.float32)],
            odoms=[np.zeros(2, dtype=[("eq", np.uint32), ("t", np.float64), ("xya", np.float32, 3)])],
            scan_time=[np.array([10.0, 10.025])],
            det_id=[np.array([0])], idet2iscan=[{0: 0}],
            det_wp=[[[(2.0, 0.0)]]],
        )
        split = tak.frames_from_dataset(ds, tak.DATASETS["frog"])
        assert len(split.other_polar) == 1 and split.other_polar[0].shape == (0, 2)

    def test_a_label_no_sensor_of_this_kind_could_have_produced_is_dropped_here_too(self):
        """DROW's four annotations at 316-716 m are in `det_wp`; nothing says the other two files are cleaner."""
        ds = SimpleNamespace(
            scans=[np.full((1, 8), 5.0, np.float32)],
            odoms=[np.zeros(1, dtype=[("eq", np.uint32), ("t", np.float64), ("xya", np.float32, 3)])],
            scan_time=[np.array([10.0])], det_id=[np.array([0])], idet2iscan=[{0: 0}],
            det_wp=[[[]]], det_wc=[[[(400.0, 0.0), (3.0, 0.0)]]], det_wa=[[[]]],
        )
        split = tak.frames_from_dataset(ds, tak.DATASETS["drow"])
        assert split.other_polar[0][:, 0].tolist() == pytest.approx([3.0])


class TestTheFlagAndItsMirror:
    def test_stage_two_takes_it_and_it_is_off_by_default(self):
        args = tak.build_parser().parse_args(["step2"])
        assert args.ignore_other_classes is False

    def test_turning_it_on_is_what_the_drow_arm_does(self):
        args = tak.build_parser().parse_args(["step2", "--dataset", "drow", "--ignore-other-classes"])
        assert args.ignore_other_classes is True

    def test_the_objects_mirror_with_the_scan_or_the_mask_lands_on_the_wrong_side(self):
        objects = np.array([[5.0, 0.3]])
        mirrored = tak.mirror_people(objects, ANGLES)
        assert mirrored[0, 1] == pytest.approx(ANGLES[0] + ANGLES[-1] - 0.3)
