"""
Unit tests for the person-free-split metric (TODO.md A31).

On a split with no annotated people, wp-AUC is not "low", it is *undefined*:
precision is 0 at every threshold and recall has no denominator. The metric
that replaces it counts what is provably there -- every detection is a false
positive -- so the properties worth pinning down are all about the denominator
and the threshold, which is where a false-positive rate is easiest to flatter
by accident.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from follow_the_drow.utils.drow_utils import false_positive_rate, FP_THRESHOLDS


FROG_PERIOD = 1.0 / 26.2


class TestCounting:
    def test_every_detection_is_a_false_positive(self):
        stats = false_positive_rate([0.9, 0.8, 0.7], [0, 1, 2], n_frames=3,
                                    thresholds=(0.5,))
        assert stats["per_threshold"][0.5]["fp"] == 3
        assert stats["per_threshold"][0.5]["fp_per_frame"] == 1.0

    def test_frames_that_fired_nothing_stay_in_the_denominator(self):
        """The easy way to turn a terrible result into a good one is to divide
        by the frames that produced a detection instead of the frames that were
        evaluated. 3 detections over 100 frames is 0.03/frame, not 1.0."""
        stats = false_positive_rate([0.9, 0.8, 0.7], [0, 1, 2], n_frames=100,
                                    thresholds=(0.5,))
        assert stats["per_threshold"][0.5]["fp_per_frame"] == pytest.approx(0.03)

    def test_several_detections_in_one_frame_all_count(self):
        stats = false_positive_rate([0.9, 0.8, 0.7], [4, 4, 4], n_frames=10,
                                    thresholds=(0.5,))
        row = stats["per_threshold"][0.5]
        assert row["fp"] == 3
        assert row["frames_with_fp"] == 1
        assert row["frame_fp_rate"] == pytest.approx(0.1)

    def test_no_detections_at_all_is_zero_not_nan(self):
        """A detector that never fires is the best possible result here, and it
        must read as 0.000 rather than as a missing measurement."""
        stats = false_positive_rate([], [], n_frames=50,
                                    frame_period_s=FROG_PERIOD)
        for row in stats["per_threshold"].values():
            assert row["fp"] == 0
            assert row["fp_per_frame"] == 0.0
            assert row["fp_per_s"] == 0.0
            assert row["frames_with_fp"] == 0


class TestThresholding:
    def test_threshold_is_inclusive(self):
        stats = false_positive_rate([0.5], [0], n_frames=1, thresholds=(0.5,))
        assert stats["per_threshold"][0.5]["fp"] == 1

    def test_higher_thresholds_never_count_more(self):
        rng = np.random.RandomState(0)
        scores = rng.uniform(0, 1, 500)
        frames = rng.randint(0, 100, 500)
        stats = false_positive_rate(scores, frames, n_frames=100)
        counts = [stats["per_threshold"][t]["fp"] for t in sorted(FP_THRESHOLDS)]
        assert counts == sorted(counts, reverse=True)

    def test_every_requested_threshold_is_reported(self):
        stats = false_positive_rate([0.9], [0], n_frames=1,
                                    thresholds=(0.1, 0.25, 0.99))
        assert sorted(stats["per_threshold"]) == [0.1, 0.25, 0.99]


class TestRates:
    def test_per_second_uses_frames_times_period_not_the_timestamp_span(self):
        """`transferred`'s person-free frames are interleaved with populated
        ones the split drops, so the span between the first and last scored
        frame includes time the detector was never asked about. Billing that to
        the detector would understate its rate."""
        stats = false_positive_rate([0.9] * 262, list(range(262)),
                                    n_frames=262, frame_period_s=FROG_PERIOD,
                                    thresholds=(0.5,))
        assert stats["duration_s"] == pytest.approx(10.0, abs=0.01)
        assert stats["per_threshold"][0.5]["fp_per_s"] == pytest.approx(26.2, rel=1e-3)

    def test_without_a_frame_period_the_rate_is_nan_not_invented(self):
        stats = false_positive_rate([0.9], [0], n_frames=1, thresholds=(0.5,))
        assert np.isnan(stats["duration_s"])
        assert np.isnan(stats["per_threshold"][0.5]["fp_per_s"])
        # the per-frame figure needs no clock and must still be real
        assert stats["per_threshold"][0.5]["fp_per_frame"] == 1.0


class TestInputValidation:
    def test_mismatched_scores_and_frames_raise(self):
        with pytest.raises(ValueError, match="index the same detections"):
            false_positive_rate([0.9, 0.8], [0], n_frames=1)

    def test_zero_frames_raises_rather_than_dividing(self):
        with pytest.raises(ValueError, match="n_frames must be positive"):
            false_positive_rate([], [], n_frames=0)


class TestSplitGuard:
    """The metric is only meaningful where every detection *is* a false
    positive. Running it on a split that contains people would report misses
    and hits as phantoms, which is precisely the class of mislabelled number
    A31 exists to stop."""

    def _fake_dataset(self, wps):
        class _DS:
            det_wp = wps
        return _DS()

    def test_a_split_with_people_is_refused(self):
        from evaluate import _require_person_free
        ds = self._fake_dataset([[[], [(2.0, 0.1)]]])
        with pytest.raises(SystemExit, match="no annotated people"):
            _require_person_free(ds, "test metric")

    def test_a_person_free_split_passes(self):
        from evaluate import _require_person_free
        ds = self._fake_dataset([[[], []], [[]]])
        _require_person_free(ds, "test metric")   # must not raise

    def test_frame_period_is_the_median_so_pauses_cannot_set_it(self):
        """FROG carries 60 recording pauses of up to 74.3 s. A mean would let
        them set the frame rate; the median reports the ~26.2 Hz a deployment
        actually sees."""
        from evaluate import _frame_period_s

        class _DS:
            scan_time = [np.concatenate([
                np.arange(100) * FROG_PERIOD,
                [100 * FROG_PERIOD + 74.3],
            ])]

        assert _frame_period_s(_DS()) == pytest.approx(FROG_PERIOD, rel=1e-6)


class TestTrackedEvaluationWalksEveryFrame:
    """A tracker is a deployment-time filter, so it has to see every frame the
    sensor produced -- not just the frames the protocol scores.

    Two ways that goes wrong if the walk is the scored set: under
    `--eval-stride N` the tracker's time step silently becomes N times longer
    than the `dt=1.0` it assumes, and on the `transferred` split the scored
    frames are the person-free ones, interleaved in the raw recording with
    populated frames the split drops -- so consecutive scored frames can be
    minutes apart, and joining them into a "track" is meaningless.
    """

    def _dataset(self, n_scans, scored_scans):
        class _DS:
            scans = [np.zeros((n_scans, 720), dtype=np.float32)]
            det_id = [list(range(len(scored_scans)))]
            idet2iscan = [{d: s for d, s in enumerate(scored_scans)}]
        return _DS()

    def test_without_a_tracker_the_walk_is_exactly_the_scored_set(self):
        from evaluate import _lfe_frames
        ds = self._dataset(100, [0, 5, 10, 15])
        walk, scored = _lfe_frames(ds, 0, None)
        assert list(walk) == [0, 5, 10, 15]
        assert set(scored) == {0, 5, 10, 15}

    def test_with_a_tracker_every_raw_frame_is_walked(self):
        from evaluate import _lfe_frames
        ds = self._dataset(100, [0, 5, 10, 15])
        walk, scored = _lfe_frames(ds, 0, {"min_hits": 3})
        assert list(walk) == list(range(100))
        assert set(scored) == {0, 5, 10, 15}, "the scored subset must not widen"

    def test_interleaved_scored_frames_still_walk_contiguously(self):
        """The `transferred` shape: person-free frames scattered through a
        recording whose populated frames the split drops."""
        from evaluate import _lfe_frames
        ds = self._dataset(50, [3, 4, 5, 30, 31])
        walk, _ = _lfe_frames(ds, 0, {"min_hits": 3})
        assert list(walk) == list(range(50))


class TestEgoMotionDeltas:
    """FROG odometry is not zero -- 1.88 cm and up to 0.4 deg per frame -- and
    the claim that it was came from `_load_or_fake_odom()` substituting zeros
    on failure (TODO.md A14). Over a track's lifetime 0.4 deg/frame is a third
    of a metre at 8 m range, most of the tracker's gate."""

    def _odoms(self, xya):
        dtype = np.dtype([("eq", np.uint32), ("t", np.float64), ("xya", np.float32, 3)])
        o = np.zeros(len(xya), dtype=dtype)
        o["xya"] = np.asarray(xya, dtype=np.float32)
        return o

    def test_first_frame_has_no_delta(self):
        from evaluate import _EgoMotion
        ego = _EgoMotion(self._odoms([[1.0, 2.0, 0.5], [1.1, 2.0, 0.5]]))
        dtheta, dxy = ego.step(0)
        assert dtheta == 0.0
        assert dxy == pytest.approx((0.0, 0.0))

    def test_heading_delta_is_the_difference_from_the_previous_frame(self):
        from evaluate import _EgoMotion
        ego = _EgoMotion(self._odoms([[0.0, 0.0, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, 0.75]]))
        ego.step(0)
        assert ego.step(1)[0] == pytest.approx(0.25, abs=1e-6)
        assert ego.step(2)[0] == pytest.approx(0.50, abs=1e-6)

    def test_a_stationary_robot_produces_no_motion(self):
        from evaluate import _EgoMotion
        ego = _EgoMotion(self._odoms([[3.0, 4.0, 1.0]] * 3))
        for i in range(3):
            dtheta, dxy = ego.step(i)
            assert dtheta == pytest.approx(0.0)
            assert dxy == pytest.approx((0.0, 0.0))
