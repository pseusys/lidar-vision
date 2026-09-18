"""The role-C bound, asked as a discrimination question (TODO.md A45).

The precision oracle measured trail lengths on **person-free** frames: what
fraction of phantom detections sit on trails shorter than L. It never had to
tell a phantom from a person, because the split contains none -- so it bounds
*removal*, not *discrimination*, and A41 measured what that costs when a real
mechanism must do both (recall 79.3% -> 23.2%).

This is the honest version. Build trails over **every** detection on populated
frames, label each detection by whether it matched an annotated person, and
report both sides of the filter at each horizon: false positives removed *and*
true positives destroyed. A one-sided bound flatters; a two-sided one decides
whether role C is worth building at all.

`trail_curve` is the accounting, separated out because the whole result is a
subtraction and an off-by-one in it would be invisible in the output.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from horizon_sweep import trail_curve


# Each track is (length_in_frames, n_true_positives, n_false_positives).
MIXED = [(1, 0, 1), (1, 1, 0), (5, 3, 2), (50, 0, 50), (100, 80, 20)]


class TestTheAccounting:

    def test_a_filter_at_one_removes_nothing(self):
        fp_rm, tp_lost = trail_curve(MIXED, [1])[0]
        assert fp_rm == 0 and tp_lost == 0

    def test_a_filter_longer_than_everything_removes_everything(self):
        fp_rm, tp_lost = trail_curve(MIXED, [1000])[0]
        assert fp_rm == sum(t[2] for t in MIXED)
        assert tp_lost == sum(t[1] for t in MIXED)

    def test_both_sides_are_monotone_in_the_horizon(self):
        """A longer filter can only remove more of both. If either column ever
        dips, the accounting is wrong rather than the data being interesting."""
        curve = trail_curve(MIXED, [1, 2, 5, 10, 50, 100, 500])
        for (a_fp, a_tp), (b_fp, b_tp) in zip(curve, curve[1:]):
            assert b_fp >= a_fp and b_tp >= a_tp

    def test_a_pure_person_track_costs_only_recall(self):
        assert trail_curve([(5, 5, 0)], [6]) == [(0, 5)]

    def test_a_pure_phantom_track_costs_only_precision(self):
        assert trail_curve([(5, 0, 5)], [6]) == [(5, 0)]

    def test_a_mixed_track_is_all_or_nothing(self):
        """Trails are dropped whole -- a filter cannot keep the person half of
        a trail that merged a person with a phantom. That is a real limit of
        any trail-length rule and the bound must charge for it."""
        assert trail_curve([(5, 3, 2)], [6]) == [(2, 3)]
        assert trail_curve([(5, 3, 2)], [5]) == [(0, 0)]

    def test_the_boundary_is_shorter_than_L(self):
        """`drop every trail shorter than L`, matching the existing precision
        oracle's rule exactly, so the two are comparable."""
        assert trail_curve([(5, 0, 1)], [5]) == [(0, 0)]
        assert trail_curve([(5, 0, 1)], [6]) == [(1, 0)]

    def test_empty_input(self):
        assert trail_curve([], [1, 10]) == [(0, 0), (0, 0)]

    @pytest.mark.parametrize("L", [1, 3, 7, 20])
    def test_nothing_is_double_counted(self, L):
        fp_rm, tp_lost = trail_curve(MIXED, [L])[0]
        kept = [t for t in MIXED if t[0] >= L]
        assert fp_rm + sum(t[2] for t in kept) == sum(t[2] for t in MIXED)
        assert tp_lost + sum(t[1] for t in kept) == sum(t[1] for t in MIXED)
