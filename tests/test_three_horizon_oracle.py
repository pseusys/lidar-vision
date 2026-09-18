"""The interpolation bound replayed on our own detector (TODO.md A50 phase 1, item 1).

`three_horizon_oracle.bounds` answers two questions at once over trajectories
whose points carry which of two detectors covered them:

    recoverable   stage-2 misses a perfect temporal model could fill, meaning
                  the same trajectory was detected both *before* and *after*
                  the miss, no further apart than the bridge
    recovered     how many of those stage 3 actually covered

The asymmetry is the point and is easy to get wrong: a miss after the last
detection, or before the first, is *not* recoverable by interpolation however
long the bridge is -- filling those would be extrapolation, which the docstring
of `utils/temporal_oracle.py` explains no real model does reliably.

The bridge is measured in seconds against each trajectory's own rate, so the
same frame gap counts differently at 10 Hz and at 26.2 Hz.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from three_horizon_oracle import bounds

HZ = 10.0


def _track(points, hz=HZ):
    """One trajectory of (frame, covered by stage 2, covered by stage 3) points, as `bounds` takes them."""
    return [(points, hz)]


class TestRecoverable:

    def test_a_miss_between_two_detections_within_the_bridge_is_recoverable(self):
        assert bounds(_track([(0, True, False), (1, False, False), (2, True, True)]), 1.0) == (1, 0)

    def test_a_miss_between_detections_further_apart_than_the_bridge_is_not(self):
        # 100 frames at 10 Hz is 10 s, past a 1 s bridge.
        assert bounds(_track([(0, True, False), (50, False, False), (100, True, True)]), 1.0) == (0, 0)

    def test_the_same_gap_counts_against_the_trajectory_own_rate(self):
        points = [(0, True, False), (7, False, False), (15, True, True)]
        assert bounds([(points, 10.0)], 1.0) == (0, 0)      # 15 frames at 10 Hz is 1.5 s, past a 1 s bridge
        assert bounds([(points, 26.2)], 1.0) == (1, 0)      # the same 15 frames at 26.2 Hz is 0.57 s, inside it

    def test_a_miss_after_the_last_detection_is_never_recoverable(self):
        assert bounds(_track([(0, True, False), (1, False, False)]), 60.0) == (0, 0)

    def test_a_miss_before_the_first_detection_is_never_recoverable(self):
        assert bounds(_track([(0, False, False), (1, True, False)]), 60.0) == (0, 0)

    def test_a_trajectory_stage_2_never_detected_offers_nothing_to_interpolate(self):
        assert bounds(_track([(0, False, True), (1, False, True), (2, False, True)]), 60.0) == (0, 0)


class TestRecovered:

    def test_recovered_counts_only_recoverable_misses_stage_3_covered(self):
        points = [(0, True, True), (1, False, True), (2, False, False), (3, True, True)]
        assert bounds(_track(points), 1.0) == (2, 1)

    def test_stage_3_covering_an_unrecoverable_miss_is_not_counted(self):
        # Covered by stage 3, but past the last stage-2 detection, so outside the bound entirely.
        assert bounds(_track([(0, True, True), (1, False, True)]), 60.0) == (0, 0)

    def test_a_trajectory_without_a_usable_rate_is_skipped(self):
        assert bounds(_track([(0, True, False), (1, False, True), (2, True, True)], hz=float("nan")), 1.0) == (0, 0)


class TestSeveralTrajectories:

    def test_counts_add_over_trajectories(self):
        one = ([(0, True, False), (1, False, True), (2, True, True)], HZ)
        two = ([(0, True, False), (1, False, False), (2, True, True)], HZ)
        assert bounds([one, two], 1.0) == (2, 1)

    def test_no_trajectories_is_zero_rather_than_an_error(self):
        assert bounds([], 2.0) == (0, 0)
