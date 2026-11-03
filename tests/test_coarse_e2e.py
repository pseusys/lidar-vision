"""`utils/coarse_e2e.py`: does stage 2's coarse convolution earn anything end to end?

Pinned here: a person-observation's age is the time since its trajectory's first annotation, linked in world coordinates with no
identities; FROG's copied labels (`memory/data-model.md`) are left out of the per-person measures, since a copy would start a
trajectory of its own every frame and could only be matched by a double report; first-second recall counts exactly the observations
under one second old; and the decision rule is the one written down before the runs.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from coarse_e2e import first_second_recall, trajectory_ages, verdict

FRAMES = [10, 11, 12, 13]
TIMES = {10: 0.0, 11: 0.1, 12: 0.2, 13: 0.3}


class TestTrajectoryAges:

    def test_a_person_present_throughout_ages_with_the_clock(self):
        people = [np.array([[1.0, 2.0]])] * 4
        ages = trajectory_ages(people, FRAMES, TIMES)
        assert [ages[(f, 0)] for f in FRAMES] == pytest.approx([0.0, 0.1, 0.2, 0.3])

    def test_a_person_who_appears_later_starts_at_zero(self):
        a, b = np.array([1.0, 2.0]), np.array([5.0, 6.0])
        people = [a[None], a[None], np.stack([a, b]), np.stack([a, b])]
        ages = trajectory_ages(people, FRAMES, TIMES)
        assert ages[(12, 1)] == pytest.approx(0.0) and ages[(13, 1)] == pytest.approx(0.1)
        assert ages[(13, 0)] == pytest.approx(0.3)

    def test_a_copied_label_has_no_age(self):
        a = np.array([1.0, 2.0])
        people = [a[None], np.stack([a, a]), a[None], a[None]]
        ages = trajectory_ages(people, FRAMES, TIMES)
        assert (11, 1) not in ages and ages[(11, 0)] == pytest.approx(0.1)


class TestFirstSecondRecall:

    def test_only_observations_under_a_second_old_count(self):
        ages = {(1, 0): 0.2, (1, 1): 0.9, (2, 0): 1.0, (2, 1): 5.0}
        covered = {1: np.array([True, False]), 2: np.array([False, True])}
        assert first_second_recall(covered, ages) == (1, 2)

    def test_frames_never_scored_are_left_out(self):
        assert first_second_recall({}, {(1, 0): 0.2}) == (0, 0)


class TestVerdict:
    """Drop if the default's mean AP is not above the no-temporal mean by more than 1.5 pp and its first-second recall not
    above by more than 2 pp; keep otherwise."""

    def test_within_both_bars_drops_the_convolution(self):
        assert verdict(ap_gain_pp=1.0, first_second_gain_pp=1.5) == "drop"

    def test_more_ap_keeps_it(self):
        assert verdict(ap_gain_pp=1.6, first_second_gain_pp=0.0) == "keep"

    def test_more_first_second_recall_alone_keeps_it(self):
        assert verdict(ap_gain_pp=0.0, first_second_gain_pp=2.5) == "keep"

    def test_a_loss_drops_it(self):
        assert verdict(ap_gain_pp=-0.8, first_second_gain_pp=-3.0) == "drop"
