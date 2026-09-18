"""The dataset anomaly checker (`TODO.md` A49, spec approved 2026-09-17).

Its numbers decide which denoising mechanism, if any, a dataset gets (`memory/noise-structure.md`), so every category it counts
is pinned on a scan built to contain exactly one known anomaly. Pinned in particular:

- what counts as invalid: non-finite, non-positive, or at or above a dataset's measured no-return value;
- that a sentinel peak is found against a smooth distribution;
- that an A-B-A spike and an A-B-B step are told apart, and that a triple spanning a recording gap or an invalid reading counts
  as neither;
- that "near a person" means the beam passes within the match radius of the person's centre;
- that the printed report stays ASCII (`AGENTS.md` I3) and the result is JSON-serialisable.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from scan_anomalies import HISTOGRAM_TOP_MM, SENSORS, ScanAnomalies, Segment, format_report, invalid_mask, near_people_mask, run_lengths, sentinel_peaks, triple_kinds

BEAMS = 36
ANGLES = np.linspace(-np.pi / 2, np.pi / 2, BEAMS, endpoint=False)


def _flat(frames: int, value: float = 3.0) -> np.ndarray:
    return np.full((frames, BEAMS), value, dtype=np.float32)


class TestInvalidMask:

    def test_non_finite_and_non_positive_are_invalid(self):
        scan = np.array([[1.0, np.nan, np.inf, -np.inf, 0.0, -1.0]], dtype=np.float32)
        assert invalid_mask(scan, None).tolist() == [[False, True, True, True, True, True]]

    def test_the_no_return_value_and_above_are_invalid(self):
        scan = np.array([[29.9, 29.96, 29.99]], dtype=np.float32)
        assert invalid_mask(scan, 29.95).tolist() == [[False, True, True]]


class TestRunLengths:

    def test_runs_down_each_column_are_measured_separately(self):
        mask = np.array([[1, 0], [1, 1], [0, 1], [1, 1]], dtype=bool)
        assert sorted(run_lengths(mask, axis=0).tolist()) == [1, 2, 3]

    def test_runs_along_a_row(self):
        mask = np.array([[1, 1, 0, 1, 0, 0, 1, 1, 1]], dtype=bool)
        assert sorted(run_lengths(mask, axis=1).tolist()) == [1, 2, 3]

    def test_no_runs(self):
        assert run_lengths(np.zeros((3, 4), dtype=bool), axis=0).size == 0


class TestSentinelPeaks:

    def test_a_planted_value_stands_out_of_a_smooth_distribution(self):
        rng = np.random.default_rng(0)
        values = np.concatenate([rng.uniform(0.5, 30.0, 200_000), np.full(5_000, 29.96)]).astype(np.float32)
        histogram = np.bincount(np.round(values * 1000).astype(np.int64), minlength=HISTOGRAM_TOP_MM + 2)
        peaks = sentinel_peaks(histogram)
        assert peaks and abs(peaks[0]["value_m"] - 29.96) < 1e-6

    def test_readings_rounded_to_centimetres_are_not_all_peaks(self):
        """DROW's ranges are rounded to 1 cm, so nine in ten millimetre bins are empty; a common centimetre value is not a sentinel."""
        rng = np.random.default_rng(2)
        histogram = np.zeros(HISTOGRAM_TOP_MM + 2, dtype=np.int64)
        centimetres = np.arange(500, 29_000, 10)
        histogram[centimetres] = rng.integers(40_000, 60_000, len(centimetres))
        histogram[29_960] = 25_000_000
        peaks = sentinel_peaks(histogram)
        assert [round(p["value_m"], 3) for p in peaks] == [29.96]

    def test_a_sentinel_above_sixty_metres_is_found(self):
        """FROG's train/val file writes no return as 61.0 m, above the UTM-30LX's 60 m maximum."""
        rng = np.random.default_rng(3)
        values = np.concatenate([rng.uniform(0.5, 30.0, 200_000), np.full(5_000, 61.0)])
        histogram = np.bincount(np.round(values * 1000).astype(np.int64), minlength=HISTOGRAM_TOP_MM + 2)
        assert [p["value_m"] for p in sentinel_peaks(histogram)] == [61.0]

    def test_a_smooth_distribution_has_no_peak(self):
        rng = np.random.default_rng(1)
        values = rng.uniform(0.5, 30.0, 200_000)
        histogram = np.bincount(np.round(values * 1000).astype(np.int64), minlength=HISTOGRAM_TOP_MM + 2)
        assert sentinel_peaks(histogram) == []


class TestTripleKinds:

    def _kinds(self, a: float, b: float, c: float, invalid: np.ndarray | None = None, linked: np.ndarray | None = None):
        scans = np.array([[a], [b], [c]], dtype=np.float32)
        invalid = np.zeros_like(scans, dtype=bool) if invalid is None else invalid
        linked = np.ones(2, dtype=bool) if linked is None else linked
        spike, step, valid = triple_kinds(scans, invalid, linked, jump_m=0.3, agree_m=0.1)
        return bool(spike[0, 0]), bool(step[0, 0]), bool(valid[0, 0])

    def test_a_b_a_is_a_spike(self):
        assert self._kinds(3.0, 1.0, 3.05) == (True, False, True)

    def test_a_b_b_is_a_step_not_a_spike(self):
        assert self._kinds(3.0, 1.0, 1.05) == (False, True, True)

    def test_the_return_from_a_spike_is_not_a_step(self):
        scans = np.array([[3.0], [1.0], [3.0], [3.0]], dtype=np.float32)
        spike, step, _ = triple_kinds(scans, np.zeros_like(scans, dtype=bool), np.ones(3, dtype=bool), jump_m=0.3, agree_m=0.1)
        assert spike[:, 0].tolist() == [True, False]
        assert step[:, 0].tolist() == [False, False]

    def test_jitter_is_neither(self):
        assert self._kinds(3.0, 3.02, 2.99) == (False, False, True)

    def test_a_triple_with_an_invalid_reading_is_not_counted(self):
        invalid = np.array([[False], [True], [False]])
        assert self._kinds(3.0, 1.0, 3.0, invalid=invalid) == (False, False, False)

    def test_a_triple_across_a_gap_is_not_counted(self):
        assert self._kinds(3.0, 1.0, 3.0, linked=np.array([True, False])) == (False, False, False)


class TestNearPeople:

    def test_beams_passing_within_the_radius_of_a_person(self):
        near = near_people_mask([np.array([[5.0, 0.0]])], ANGLES, radius_m=0.5)
        half_width = np.arcsin(0.5 / 5.0)
        assert near[0].tolist() == (np.abs(ANGLES) <= half_width).tolist()
        assert near[0].any()

    def test_a_person_closer_than_the_radius_covers_every_beam(self):
        assert near_people_mask([np.array([[0.3, 0.0]])], ANGLES, radius_m=0.5)[0].all()

    def test_frames_without_people(self):
        assert not near_people_mask([np.zeros((0, 2))], ANGLES, radius_m=0.5).any()


class TestScanAnomalies:

    def _checker(self) -> ScanAnomalies:
        return ScanAnomalies(SENSORS["UTM-30LX"], ANGLES, no_return_m=None)

    def test_counts_rate_and_categories_on_a_built_recording(self):
        frames = 40
        scans = _flat(frames)
        scans[5, 3] = np.nan
        scans[6, 3] = np.inf
        scans[10, 7] = 0.0
        scans[frames - 1, 8] = 45.0    # finite, beyond the rated 30 m; last frame, so no triple is centred on it
        scans[20, 10] = 1.0            # A-B-A spike
        scans[30:, 12] = 1.5           # A-B-B step
        times = np.arange(frames) / 25.0
        people = [np.array([[3.0, float(ANGLES[10])]]) for _ in range(frames)]
        checker = self._checker()
        checker.add(Segment("built", scans, times, people))
        result = checker.result()

        assert result["readings"] == frames * BEAMS
        assert result["values"]["nan"] == 1
        assert result["values"]["posinf"] == 1
        assert result["values"]["zero"] == 1
        assert result["values"]["above_rated_range"] == 1
        assert result["values"]["largest_finite_m"] == 45.0
        assert result["invalid"]["temporal_runs"]["2"] == 1
        assert result["spikes"]["count"] == 1
        assert result["steps"]["count"] == 1
        assert result["spikes"]["near_people"] == 1
        assert abs(result["rate_hz"] - 25.0) < 1e-6
        json.dumps(result)

    def test_the_rate_ignores_gaps_between_segments(self):
        checker = self._checker()
        checker.add(Segment("a", _flat(11), np.arange(11) / 20.0, None))
        checker.add(Segment("b", _flat(11), 1000.0 + np.arange(11) / 20.0, None))
        assert abs(checker.result()["rate_hz"] - 20.0) < 1e-6

    def test_bunched_timestamps_are_reported(self):
        """FROG's scans are 25 ms apart but stamped in bunches (38 ms, 38 ms, ~0 ms): the median says 26 Hz, the mean 40 Hz,
        and anything dividing by the stamped interval sees a near-zero step on a quarter of frames."""
        times = np.cumsum(np.tile([0.0381, 0.0368, 0.00005], 20))
        checker = self._checker()
        checker.add(Segment("bunched", _flat(len(times)), times, None))
        result = checker.result()
        intervals = np.diff(times)
        assert abs(result["rate_hz"] - 1.0 / intervals.mean()) < 1e-6
        assert abs(result["timestamps"]["median_rate_hz"] - 1.0 / np.median(intervals)) < 0.5
        assert abs(result["timestamps"]["share_under_quarter_period"] - np.mean(intervals < 0.25 * intervals.mean())) < 1e-6
        assert result["timestamps"]["median_rate_hz"] < 0.7 * result["rate_hz"]

    def test_the_report_is_ascii(self):
        checker = self._checker()
        scans = _flat(10)
        scans[4, 2] = np.nan
        checker.add(Segment("a", scans, np.arange(10) / 25.0, None))
        report = format_report("test", checker.result())
        report.encode("ascii")
        assert "Missing returns" in report
