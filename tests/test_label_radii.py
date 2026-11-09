"""Per-class label radii, and the rule that decides a beam inside two classes' disks.

DROW's and DR-SPAAM's own targets give each class its own radius -- wheelchair 0.6, walker 0.4, person 0.35 m -- and
hand a beam to the annotation it is **deepest inside**: distance minus that annotation's radius, argmin against a zero
column for "outside every disk" (`_closest_detection` in DR-SPAAM's `drow_dataset.py`). At one radius for every class
that is exactly the nearest assignment every stage 2 on file trained with, which is what lets the flag exist without
moving a single recorded number. The radius itself is a stage-2 argument, so stage 3's fine-tuning has to read it back
from the checkpoint the way it reads the rest of the objective (`tests/test_joint_targets.py`).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from train import beam_heatmap, make_targets  # noqa: E402
from train_takheliped import DROW, FROG, build_parser, class_radii, stage2_training  # noqa: E402

#: A fan dense enough that a 0.35 m disk at 4 m holds several beams.
ANGLES = np.linspace(-0.6, 0.6, 241)
#: DROW's three radii by DROW's own label ids: 1 wheelchair, 2 walker, 3 person.
PUBLISHED = {1: 0.6, 2: 0.4, 3: 0.35}


def wall(distance: float = 4.0) -> np.ndarray:
    """Every beam ends on a wall `distance` ahead, so a beam's endpoint is `(distance * tan(angle), distance)`."""
    return distance / np.cos(ANGLES)


def endpoints(scan: np.ndarray) -> np.ndarray:
    return np.stack([scan * -np.sin(ANGLES), scan * np.cos(ANGLES)], axis=1)


def polar(x: float, y: float):
    return (math.hypot(x, y), math.atan2(-x, y))


def crowded():
    """A person with a walker in hand and a wheelchair beside them, on the wall, where every disk overlaps another."""
    return {3: [polar(0.0, 4.0)], 2: [polar(0.08, 4.0)], 1: [polar(0.45, 4.0)]}


def deepest(scan: np.ndarray, gt: dict, radii: dict) -> np.ndarray:
    """The labels DR-SPAAM's `_closest_detection` gives, written out independently of `make_targets`."""
    people = [(c, r, phi) for c, anns in sorted(gt.items()) for r, phi in anns]
    centres = np.array([[r * -np.sin(phi), r * np.cos(phi)] for _, r, phi in people])
    depth = np.linalg.norm(endpoints(scan)[:, None] - centres[None], axis=-1) - np.array([radii[c] for c, _, _ in people])
    depth = np.hstack([np.zeros((len(scan), 1)), depth])
    owner = depth.argmin(axis=1)
    return np.array([0] + [c for c, _, _ in people])[owner]


class TestOneRadiusIsTodaysTargets:
    @pytest.mark.parametrize("nearest", [True, False])
    def test_one_radius_given_per_class_is_the_same_radius_given_once(self, nearest):
        scan = wall()
        once = make_targets(scan, ANGLES, crowded(), vote_collect_radius=0.6, nearest=nearest)
        per_class = make_targets(scan, ANGLES, crowded(), vote_collect_radius={1: 0.6, 2: 0.6, 3: 0.6}, nearest=nearest)
        assert np.array_equal(once[0], per_class[0]) and np.array_equal(once[1], per_class[1])

    def test_at_one_radius_deepest_inside_is_nearest_within(self):
        """So the adopted nearest assignment is the published rule's special case rather than a different rule."""
        scan = wall()
        labels, _ = make_targets(scan, ANGLES, crowded(), vote_collect_radius=0.6, nearest=True)
        assert np.array_equal(labels, deepest(scan, crowded(), {1: 0.6, 2: 0.6, 3: 0.6}))


class TestEachClassHasItsOwnDisk:
    @pytest.mark.parametrize("class_id", [1, 2, 3])
    def test_a_lone_object_is_positive_exactly_inside_its_own_class_radius(self, class_id):
        scan, centre = wall(), np.array([0.0, 4.0])
        labels, _ = make_targets(scan, ANGLES, {class_id: [polar(*centre)]}, vote_collect_radius=PUBLISHED, nearest=True)
        inside = np.linalg.norm(endpoints(scan) - centre, axis=1) < PUBLISHED[class_id]
        assert inside.sum() > 0
        assert np.array_equal(labels > 0, inside)

    def test_the_last_listed_assignment_bounds_each_class_by_its_own_radius_too(self):
        scan = wall()
        gt = {1: [polar(-0.8, 4.0)], 3: [polar(0.8, 4.0)]}
        labels, _ = make_targets(scan, ANGLES, gt, vote_collect_radius=PUBLISHED)
        distance = np.linalg.norm(endpoints(scan)[:, None] - np.array([[-0.8, 4.0], [0.8, 4.0]])[None], axis=-1)
        assert np.array_equal(labels == 1, distance[:, 0] < 0.6)
        assert np.array_equal(labels == 3, distance[:, 1] < 0.35)


class TestABeamInsideTwoDisksGoesToTheOneItIsDeepestInside:
    def test_every_label_is_the_published_rule_s(self):
        scan = wall()
        labels, _ = make_targets(scan, ANGLES, crowded(), vote_collect_radius=PUBLISHED, nearest=True)
        assert np.array_equal(labels, deepest(scan, crowded(), PUBLISHED))

    def test_a_beam_nearer_the_person_can_still_belong_to_the_wider_wheelchair(self):
        """0.20 m from the person (0.15 m inside 0.35) and 0.25 m from the wheelchair (0.35 m inside 0.6): the
        wheelchair. A nearest-centre rule would say person, which is what makes this the case that tells them apart."""
        scan = wall()
        gt = {3: [polar(0.0, 4.0)], 1: [polar(0.45, 4.0)]}
        labels, votes = make_targets(scan, ANGLES, gt, vote_collect_radius=PUBLISHED, nearest=True)
        beam = int(np.argmin(np.abs(endpoints(scan)[:, 0] - 0.20)))
        assert labels[beam] == 1
        assert np.allclose(votes[beam], make_targets(scan, ANGLES, {1: gt[1]}, vote_collect_radius=PUBLISHED, nearest=True)[1][beam])

    def test_a_beam_outside_the_nearest_disk_belongs_to_the_one_it_is_inside(self):
        """0.40 m from the person, outside its 0.35 m; 0.50 m from the wheelchair, inside its 0.6 m."""
        scan = wall()
        gt = {3: [polar(0.0, 4.0)], 1: [polar(0.90, 4.0)]}
        labels, _ = make_targets(scan, ANGLES, gt, vote_collect_radius=PUBLISHED, nearest=True)
        beam = int(np.argmin(np.abs(endpoints(scan)[:, 0] - 0.40)))
        assert labels[beam] == 1


class TestTheSoftTargetFollowsThePersonsRadius:
    def test_its_support_is_the_person_positives_at_the_person_radius(self):
        scan, people = wall(), {3: [polar(0.1, 4.0)]}
        labels, _ = make_targets(scan, ANGLES, people, vote_collect_radius=PUBLISHED, nearest=True)
        heat = beam_heatmap(scan, ANGLES, people, sigma_m=0.3, vote_collect_radius=PUBLISHED)
        assert np.array_equal(heat > 0, labels > 0)


class TestTheFlag:
    def parse(self, *argv):
        return build_parser().parse_args(["step2", *argv])

    def test_the_default_is_one_radius_for_every_class(self):
        args = self.parse()
        assert class_radii(DROW, args.label_radius) == {3: 0.6, 1: 0.6, 2: 0.6}
        assert class_radii(FROG, args.label_radius) == {3: 0.6}

    def test_one_value_per_class_follows_the_spec_s_class_order(self):
        """`DatasetSpec.classes` is pedestrian, wheelchair, walker on DROW, so the values land on ids 3, 1, 2."""
        args = self.parse("--label-radius", "0.35", "0.6", "0.4")
        assert class_radii(DROW, args.label_radius) == {3: 0.35, 1: 0.6, 2: 0.4}

    @pytest.mark.parametrize("spec,values", [(DROW, [0.35, 0.6]), (FROG, [0.35, 0.6, 0.4])])
    def test_a_count_that_is_neither_one_nor_the_class_count_is_refused(self, spec, values):
        with pytest.raises(ValueError):
            class_radii(spec, values)


class TestStageThreeReadsThePersonRadiusBack:
    """Step 3b's stage-2 term is person-only, so the person's radius is the whole of what it needs from the flag."""

    @pytest.mark.parametrize("stored,person", [("[0.35, 0.6, 0.4]", 0.35), ("[0.45]", 0.45), ("0.45", 0.45)])
    def test_every_stored_form_gives_the_person_radius(self, stored, person):
        assert stage2_training({"label_radius": stored})["vote_collect_radius"] == person
