"""One definition of a dataset's frame rate, for every figure quoted per second.

FROG's recorder stamps scans in bunches -- roughly 38 ms, 38 ms, then under 0.1 ms -- so **the median interval reads 26.2 Hz
where the sensor delivers a scan every 25 ms**, and a median was what `evaluate.py` used. Every `FP/s` in the repo was therefore
1.53x low on the real clock, and the "38.2 ms real-time budget" the cost claims were set against is really 25 ms.

The rate is elapsed time over frames, inside each recording. A gap longer than a recording pause is not a frame interval and
must not be averaged in: FROG's loader already splits sessions at 0.5 s, but the helper does not get to assume its caller did.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))

from takheliped.utils.drow_utils import MAX_LINKED_INTERVAL_S, frame_period_s

FROG_NOMINAL_S = 0.025          # the sensor's 25 ms per scan, which the mean over linked intervals recovers
FROG_BUNCHED = (0.038, 0.038, 0.00005)


def _stamps(intervals, start=0.0):
    return np.concatenate([[start], start + np.cumsum(np.asarray(intervals, dtype=np.float64))])


class TestFramePeriod:
    def test_bunched_stamps_give_the_sensors_rate_not_the_medians(self):
        """Three scans per ~76 ms is 25.3 ms each, however the recorder distributed the stamps."""
        times = [_stamps(FROG_BUNCHED * 20)]

        period = frame_period_s(times)

        assert abs(period - FROG_NOMINAL_S) < 0.001
        assert abs(period - np.median(np.diff(times[0]))) > 0.01       # the median would say 38 ms

    def test_a_recording_pause_does_not_set_the_rate(self):
        times = [_stamps([0.025] * 10 + [74.3] + [0.025] * 10)]

        assert frame_period_s(times) == pytest.approx(0.025)

    def test_recordings_pool_by_their_intervals(self):
        times = [_stamps([0.02] * 10), _stamps([0.04] * 10)]

        assert frame_period_s(times) == pytest.approx(0.03)

    def test_a_longer_recording_weighs_more_than_a_short_one(self):
        times = [_stamps([0.02] * 99), _stamps([0.04])]

        assert frame_period_s(times) < 0.021

    def test_a_single_frame_recording_contributes_nothing(self):
        times = [_stamps([0.025] * 4), np.array([17.0])]

        assert frame_period_s(times) == pytest.approx(0.025)

    def test_a_split_with_no_usable_interval_has_no_rate_rather_than_a_made_up_one(self):
        assert frame_period_s([np.array([1.0]), np.array([])]) is None
        assert frame_period_s([]) is None

    def test_a_pause_exactly_at_the_threshold_is_still_a_frame_interval(self):
        times = [_stamps([MAX_LINKED_INTERVAL_S] * 4)]

        assert frame_period_s(times) == pytest.approx(MAX_LINKED_INTERVAL_S)

    def test_a_non_monotonic_stamp_is_dropped_rather_than_subtracted(self):
        """A backwards stamp would otherwise pull the mean down and raise the reported rate."""
        times = [np.array([0.0, 0.025, 0.010, 0.035, 0.060])]

        assert frame_period_s(times) == pytest.approx(0.025)
