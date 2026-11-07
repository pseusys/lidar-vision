"""Training and scoring a multi-class head: what the targets become, and what must not move when they do.

`tests/test_decision_head.py` pins the head itself. This pins the two places the dataset's classes reach it --
the per-beam targets a batch is built from, and the column a decoder reads back -- and the equivalences that
make the change safe to carry into the FROG record:

* a `binary` run builds the same targets it always did, so every FROG number is reproducible;
* a one-class `sigmoid` network holding a binary checkpoint's folded weights decodes **identical** candidates,
  so the extra columns change what is learned next and nothing about what is already trained.

The class precedence is pinned too, because it is the part that is easy to get backwards: a person standing
beside a wheelchair is a person, and `make_targets` delivers that through DROW's own id order rather than
through anything written here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from takheliped.detectors.takheliped import CalibrationNetwork, load_calibration_weights  # noqa: E402
from train import make_targets  # noqa: E402
from train_takheliped import (DROW, FROG, build_parser, class_annotations, class_columns,  # noqa: E402
                              head_class_count, network_candidates)


def _angles(n: int = 48) -> np.ndarray:
    return np.linspace(-np.pi / 2, np.pi / 2, n, dtype=np.float64)


def net(**kwargs) -> CalibrationNetwork:
    return CalibrationNetwork(torch.from_numpy(_angles().astype(np.float32)), channels=(8, 16, 32), blocks=(1, 1, 1),
                              bottleneck_blocks=1, **kwargs).eval()


def parse(*argv):
    return build_parser().parse_args(["step2", *argv])


class TestTheFlagsExistWhereTheyBelongAndNowhereElse:
    def test_the_default_is_the_head_every_number_on_file_was_measured_with(self):
        args = parse()
        assert args.head == "binary" and args.class_norm == "softmax"

    def test_both_arms_of_the_owners_decision_are_reachable(self):
        for norm in ("softmax", "sigmoid"):
            assert parse("--head", "multiclass", "--class-norm", norm).class_norm == norm

    def test_an_unknown_head_is_refused_by_the_parser(self):
        with pytest.raises(SystemExit):
            parse("--head", "classes")

    def test_stage_three_takes_no_head_flag_because_the_head_comes_with_the_weights(self):
        """`network_shape` rebuilds it from the checkpoint, so a flag here could only disagree with it."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["step3a", "--stage2", "x.pth", "--head", "multiclass"])


class TestTheHeadIsAsWideAsTheDatasetItIsTrainedOn:
    @pytest.mark.parametrize("spec,wide", [(FROG, 1), (DROW, 3)])
    def test_a_multiclass_run_emits_one_column_per_class_the_spec_annotates(self, spec, wide):
        assert head_class_count("multiclass", spec) == wide

    @pytest.mark.parametrize("spec", [FROG, DROW])
    def test_a_binary_run_is_one_column_whatever_the_dataset_annotates(self, spec):
        assert head_class_count("binary", spec) == 1


class TestTheTargetsAreTheDatasetsClassesOnlyWhenTheHeadCanSayThem:
    def _frame(self):
        """One person at bearing 0 and one wheelchair well away from it, both at 4 m."""
        return np.full(48, 4.0), [(4.0, 0.0)], [np.array([[4.0, 0.8]]), np.zeros((0, 2))]

    def test_a_binary_run_annotates_the_people_and_nothing_else(self):
        _, people, others = self._frame()
        assert class_annotations(DROW, people, others, "binary") == {3: [(4.0, 0.0)]}

    def test_a_multiclass_run_annotates_every_class_under_its_own_id(self):
        _, people, others = self._frame()
        gt = class_annotations(DROW, people, others, "multiclass")
        assert gt == {3: [(4.0, 0.0)], 1: [(4.0, 0.8)], 2: []}

    def test_a_one_class_dataset_is_the_same_dict_either_way(self):
        """FROG annotates nothing but people, so the head cannot change its targets."""
        assert class_annotations(FROG, [(4.0, 0.0)], [], "multiclass") == class_annotations(
            FROG, [(4.0, 0.0)], [], "binary") == {3: [(4.0, 0.0)]}

    def test_a_wheelchair_takes_its_own_column_rather_than_the_background(self):
        scan, people, others = self._frame()
        labels, _ = make_targets(scan, _angles(), class_annotations(DROW, people, others, "multiclass"))
        columns = class_columns(labels, DROW.class_ids)
        assert set(columns.tolist()) == {0, 1, 2}, "background, person and wheelchair"
        assert (columns == 1).sum() and (columns == 2).sum()

    def test_the_same_frame_under_a_binary_head_leaves_the_wheelchair_background(self):
        scan, people, others = self._frame()
        labels, _ = make_targets(scan, _angles(), class_annotations(DROW, people, others, "binary"))
        assert set(class_columns(labels, DROW.class_ids).tolist()) == {0, 1}

    def test_a_person_beside_a_wheelchair_is_labelled_a_person(self):
        """The precedence that matters: `make_targets` resolves a shared beam by the higher id, and the person's
        is the highest. A wheelchair winning here would teach the one shape this detector exists to find."""
        scan = np.full(48, 4.0)
        gt = class_annotations(DROW, [(4.0, 0.0)], [np.array([[4.0, 0.02]]), np.zeros((0, 2))], "multiclass")
        labels, _ = make_targets(scan, _angles(), gt)
        middle = class_columns(labels, DROW.class_ids)[24]
        assert middle == 1, "the person's column, not the wheelchair's"

    def test_the_vote_targets_are_shared_rather_than_per_class(self):
        """One vote regression for every class is DROW3's own split; a per-class vote would be our invention."""
        scan, people, others = self._frame()
        shared = net(head="multiclass", class_norm="sigmoid", classes=3)
        assert shared(torch.from_numpy(scan.astype(np.float32))[None])[1].shape[-1] == 2
        _, votes = make_targets(scan, _angles(), class_annotations(DROW, people, others, "multiclass"))
        assert votes.shape == (48, 2)


class TestAFoldedBinaryCheckpointDecodesIdenticalCandidates:
    def test_the_candidates_are_the_same_arrays_not_merely_the_same_count(self):
        torch.manual_seed(0)
        source = net()
        target = net(head="multiclass", class_norm="sigmoid", classes=3)
        load_calibration_weights(target, source.state_dict())
        scans = np.full((3, 48), 4.0, dtype=np.float32)
        scans[:, 20:26] = 2.0
        mine = network_candidates(source, scans, torch.device("cpu"))
        theirs = network_candidates(target, scans, torch.device("cpu"))
        assert len(mine) == len(theirs)
        for (score, xy), (other_score, other_xy) in zip(mine, theirs, strict=True):
            assert np.array_equal(score, other_score) and np.array_equal(xy, other_xy)
