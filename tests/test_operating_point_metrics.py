"""The merge radius scored on what Tier 0 measured, not on AP (TODO.md A40).

A38 swept the merge radius against AP and found AP wanting *more* merging, out
to 0.45 m and past the value that reproduces the published number -- while Tier
0 (`memory/static-detector-diagnosis.md`) measured over-merging costing ~5 pp of
recall at a fixed operating point. Both are right about their own quantity: a
duplicate is a full false positive, while a close-pair miss costs only one
missed ground-truth person, because the surviving merged detection still
matches *one* of the two. AP therefore under-charges exactly the error Tier 0
found.

These tests pin the four counts that measure it instead, because the whole
result turns on their definitions:

    tp          ground-truth people with a detection assigned one-to-one
    duplicate   a detection near a person, beyond the one assigned to them
    phantom     a detection near no annotated person at all
    close-pair  a MISSED person who nonetheless had a detection within the
                association radius -- it went to a neighbour. This is the
                quantity `range_probe.py` called "stolen", and the one a
                merge radius trades against duplicates.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from nms_sweep import _frame_metrics


ASSOC = 0.5


def _m(dets, gt):
    """-> (tp, dup, phantom, closepair)"""
    return _frame_metrics(np.asarray(dets, dtype=float).reshape(-1, 2),
                          np.asarray(gt, dtype=float).reshape(-1, 2), ASSOC)


class TestTheFourCounts:

    def test_one_person_one_detection(self):
        assert _m([[2.0, 0.0]], [[2.0, 0.0]]) == (1, 0, 0, 0)

    def test_a_second_detection_on_the_same_person_is_a_duplicate(self):
        """Not a second true positive, and not a phantom either."""
        assert _m([[2.0, 0.0], [2.2, 0.0]], [[2.0, 0.0]]) == (1, 1, 0, 0)

    def test_a_detection_near_nobody_is_a_phantom(self):
        assert _m([[2.0, 0.0], [9.0, 9.0]], [[2.0, 0.0]]) == (1, 0, 1, 0)

    def test_a_missed_person_with_no_detection_nearby_is_not_a_close_pair(self):
        """The representation failure case: nothing was there to steal."""
        assert _m([[9.0, 9.0]], [[2.0, 0.0]]) == (0, 0, 1, 0)

    def test_two_people_one_detection_between_them_is_a_close_pair(self):
        """The over-merging case A40 exists to measure.

        One detection, two people 0.4 m apart, both within the 0.5 m gate. The
        assignment can only serve one; the other is missed *with a detection
        0.2 m away*. AP charges this as a single missed GT. Tier 0 charges it
        as the thing the merge radius broke.
        """
        gt = [[2.0, -0.2], [2.0, 0.2]]
        assert _m([[2.0, 0.0]], gt) == (1, 0, 0, 1)

    def test_two_people_two_detections_resolves_both(self):
        gt = [[2.0, -0.2], [2.0, 0.2]]
        dets = [[2.0, -0.2], [2.0, 0.2]]
        assert _m(dets, gt) == (2, 0, 0, 0)

    def test_assignment_is_one_to_one_not_nearest(self):
        """Two detections both nearest to person A must not both match A while
        B is called a phantom-free miss -- the shortcut `min(n_near, n_gt)`
        that was wrong in phantom_analysis.py until 2026-09-11."""
        gt = [[2.0, 0.0], [2.3, 0.0]]
        dets = [[2.05, 0.0], [2.10, 0.0]]
        tp, dup, phantom, close = _m(dets, gt)
        assert tp == 2 and dup == 0 and phantom == 0 and close == 0

    def test_empty_frame_with_detections_is_all_phantom(self):
        assert _m([[1.0, 1.0], [2.0, 2.0]], np.empty((0, 2))) == (0, 0, 2, 0)

    def test_empty_detections(self):
        assert _m(np.empty((0, 2)), [[2.0, 0.0]]) == (0, 0, 0, 0)

    def test_both_empty(self):
        assert _m(np.empty((0, 2)), np.empty((0, 2))) == (0, 0, 0, 0)

    def test_every_detection_is_accounted_for_exactly_once(self):
        """tp + dup + phantom must equal the detection count, always --
        otherwise a radius change can move detections into a category that is
        not being counted and look like an improvement."""
        rs = np.random.default_rng(0)
        for _ in range(200):
            dets = rs.uniform(-5, 5, size=(rs.integers(0, 8), 2))
            gt = rs.uniform(-5, 5, size=(rs.integers(0, 5), 2))
            tp, dup, phantom, _close = _frame_metrics(dets, gt, ASSOC)
            assert tp + dup + phantom == len(dets)

    def test_close_pairs_never_exceed_the_misses(self):
        rs = np.random.default_rng(1)
        for _ in range(200):
            dets = rs.uniform(-2, 2, size=(rs.integers(0, 8), 2))
            gt = rs.uniform(-2, 2, size=(rs.integers(0, 6), 2))
            tp, _dup, _ph, close = _frame_metrics(dets, gt, ASSOC)
            assert close <= len(gt) - tp

    @pytest.mark.parametrize("assoc", [0.3, 0.5])
    def test_the_gate_is_respected_on_both_sides(self, assoc):
        """A detection just outside the gate is a phantom, just inside a TP."""
        inside = _frame_metrics(np.array([[2.0, assoc - 0.01]]),
                                np.array([[2.0, 0.0]]), assoc)
        outside = _frame_metrics(np.array([[2.0, assoc + 0.01]]),
                                 np.array([[2.0, 0.0]]), assoc)
        assert inside == (1, 0, 0, 0)
        assert outside == (0, 0, 1, 0)
