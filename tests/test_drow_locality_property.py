"""The pieces of item 102's measurement: which property of DROW makes locality matter.

Our stage 2 fits DROW's training recordings and does not carry it to the test ones, where a cutout -- a local,
depth-normalised window per beam -- does. The script bins DROW's test frames by each candidate property and scores both
models per bin; these tests pin the binning, the scene-novelty distance it bins by, and the recording types.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from drow_locality_property import (ap_by_bin, ap_by_range, beam_span, confuser_bins, exact_twins, nearest_distance,  # noqa: E402
                                    partition_verdict, person_ranges, person_score, range_band_frames, ranking_summary, recording_type,
                                    reference_bins, scan_descriptor, scan_levels, scan_statistics, sister_recordings, top_k_hits,
                                    transfer_summary)
from train_takheliped import score_detections  # noqa: E402

SPACING_DEG = 0.5


class TestTheSceneDescriptor:
    def test_one_value_per_sector_of_the_requested_width(self):
        d = scan_descriptor(np.full((3, 450), 4.0, np.float32), SPACING_DEG, sector_deg=2.5, clip_m=10.0)
        assert d.shape == (3, 90) and np.allclose(d, 4.0)

    def test_far_and_missing_returns_read_as_the_clip(self):
        scans = np.array([[3.0, 29.99, np.inf, np.nan, 61.0, 3.0]], np.float32)
        assert np.allclose(scan_descriptor(scans, SPACING_DEG, sector_deg=0.5, clip_m=10.0), [[3.0, 10, 10, 10, 10, 3.0]])

    def test_a_person_a_few_beams_wide_barely_moves_its_sector(self):
        """The median keeps the room's shape: two beams on a leg inside a five-beam sector change nothing."""
        scan = np.full((1, 450), 6.0, np.float32)
        scan[0, 101:103] = 2.0
        assert np.allclose(scan_descriptor(scan, SPACING_DEG, sector_deg=2.5, clip_m=10.0), 6.0)


class TestNearestDistance:
    def _bank(self):
        rng = np.random.default_rng(0)
        return rng.uniform(1.0, 10.0, (50, 40)).astype(np.float32)

    def test_a_scene_in_the_bank_is_at_distance_zero(self):
        bank = self._bank()
        assert nearest_distance(bank[[7, 30]], bank, max_shift=0, device="cpu") == pytest.approx([0.0, 0.0], abs=1e-6)

    def test_a_rotated_copy_is_found_within_the_shift_and_missed_without_it(self):
        bank = self._bank()
        rotated = np.roll(bank[[5]], 2, axis=1)
        assert nearest_distance(rotated, bank, max_shift=2, device="cpu")[0] == pytest.approx(0.0, abs=1e-6)
        assert nearest_distance(rotated, bank, max_shift=0, device="cpu")[0] > 0.5

    def test_it_takes_the_closest_entry_in_mean_absolute_metres(self):
        bank = np.stack([np.full(40, 5.0), np.full(40, 9.0)]).astype(np.float32)
        query = np.full((1, 40), 6.0, np.float32)
        assert nearest_distance(query, bank, max_shift=1, device="cpu")[0] == pytest.approx(1.0)

    def test_chunking_does_not_change_the_answer(self):
        """Blocks over the queries and over the bank both, against a brute-force reference."""
        bank, query = self._bank(), self._bank()[:9] + 0.3
        s = query.shape[1]
        reference = np.array([min(np.abs(q[k:] - b[:s - k]).mean() if k >= 0 else np.abs(q[:s + k] - b[-k:]).mean()
                                  for b in bank for k in range(-2, 3)) for q in query])
        assert np.allclose(nearest_distance(query, bank, max_shift=2, device="cpu", chunk=7), reference, atol=1e-5)
        assert np.allclose(nearest_distance(query, bank, max_shift=2, device="cpu", chunk=4), reference, atol=1e-5)


class TestConfusers:
    def test_three_bins_by_how_close_a_wheelchair_or_walker_is_to_a_person(self):
        people = [np.array([[0.0, 4.0]]), np.array([[0.0, 4.0]]), np.zeros((0, 2)), np.array([[0.0, 4.0]])]
        others = [np.array([[0.8, 4.0]]), np.array([[3.0, 4.0]]), np.array([[1.0, 2.0]]), np.zeros((0, 2))]
        assert confuser_bins(people, others, radius=1.0).tolist() == [0, 1, 1, 2]


class TestApByBin:
    def test_each_bin_is_scored_exactly_as_its_own_frames_alone(self):
        rng = np.random.default_rng(1)
        frames = [(rng.uniform(0, 1, 3).astype(np.float32), rng.uniform(-3, 3, (3, 2)).astype(np.float32),
                   rng.uniform(-3, 3, (2, 2))) for _ in range(12)]
        bins = np.array([0, 1] * 6)
        result = ap_by_bin(frames, bins, ("even", "odd"))
        assert result["even"]["ap"] == pytest.approx(score_detections(frames[0::2])["ap"])
        assert result["odd"]["frames"] == 6


class TestRecordingTypes:
    @pytest.mark.parametrize("name,kind", [("run_2015-11-25-14-58-49-a.bag", "driving"), ("run_t_2015-11-26-11-22-03.bag", "driving"),
                                           ("lunch_2015-11-26-12-04-23.bag", "lunch"), ("rot-1w_2015-11-24.bag", "turning"),
                                           ("n-1r_2015-11-24-18-08-31.bag", "stationary"), ("s2w2p_2015-11-24.bag", "stationary")])
    def test_the_four_kinds_of_drow_recording(self, name, kind):
        assert recording_type(name) == kind


class TestWithinFrameRanking:
    """Pooled AP ranks every detection of every frame on one scale; `top_k_hits` asks only whether each frame's own
    `k` best detections are its `k` people, so a score offset that differs from scene to scene cannot move it."""

    @staticmethod
    def _frame(scores, xy, people):
        return np.asarray(scores, np.float32), np.asarray(xy, np.float32).reshape(-1, 2), np.asarray(people, np.float64).reshape(-1, 2)

    def test_the_k_best_detections_of_a_frame_with_k_people(self):
        right = self._frame([0.9, 0.8, 0.1], [[0, 2], [1, 3], [5, 5]], [[0, 2], [1, 3]])
        wrong_first = self._frame([0.9, 0.8, 0.7], [[5, 5], [0, 2], [1, 3]], [[0, 2], [1, 3]])
        assert top_k_hits([right]) == pytest.approx(1.0)
        assert top_k_hits([wrong_first]) == pytest.approx(0.5)
        assert top_k_hits([right, wrong_first]) == pytest.approx(0.75)

    def test_two_detections_on_one_person_count_once(self):
        assert top_k_hits([self._frame([0.9, 0.8], [[0, 2], [0.1, 2]], [[0, 2], [3, 3]])]) == pytest.approx(0.5)

    def test_a_detection_beyond_the_radius_is_a_miss(self):
        assert top_k_hits([self._frame([0.9], [[0, 2.6]], [[0, 2]])], radius=0.5) == 0.0

    def test_frames_without_people_are_not_counted(self):
        empty = self._frame([0.99], [[4, 4]], np.zeros((0, 2)))
        assert top_k_hits([empty, self._frame([0.2], [[0, 2]], [[0, 2]])]) == pytest.approx(1.0)

    def test_a_per_frame_score_offset_moves_pooled_ap_and_not_this(self):
        rng = np.random.default_rng(3)
        frames = []
        for _ in range(20):
            people = rng.uniform(-4, 4, (2, 2))
            xy = np.concatenate([people + rng.normal(0, 0.05, people.shape), rng.uniform(-4, 4, (3, 2))])
            frames.append(self._frame(rng.uniform(0, 1, 5), xy, people))
        shifted = [(s * rng.uniform(0.05, 1.0), xy, gt) for s, xy, gt in frames]
        assert top_k_hits(shifted) == pytest.approx(top_k_hits(frames))
        assert score_detections(shifted)["ap"] != pytest.approx(score_detections(frames)["ap"])


class TestRankingSummary:
    """Three readings of one pass: what separates pooled AP from AP over populated frames is confident detections on
    scans with nobody there; what separates that from `top_k_hits` is how scores compare between populated scans."""

    @staticmethod
    def _frames():
        f = TestWithinFrameRanking._frame
        return [f([0.9, 0.2], [[0, 2], [4, 4]], [[0, 2]]), f([0.6], [[1, 3]], [[1, 3]]),
                f([0.95, 0.8], [[2, 2], [3, 1]], np.zeros((0, 2)))]

    def test_populated_ap_is_pooled_ap_over_the_frames_with_people(self):
        frames = self._frames()
        r = ranking_summary(frames)
        assert r["populated_ap"] == pytest.approx(score_detections(frames[:2])["ap"])
        assert r["pooled_ap"] == pytest.approx(score_detections(frames)["ap"])
        assert r["populated_ap"] > r["pooled_ap"]

    def test_it_counts_the_person_free_frames_and_carries_top_k_hits(self):
        r = ranking_summary(self._frames())
        assert (r["frames"], r["person_free"]) == (3, 1)
        assert r["top_k_hits"] == pytest.approx(top_k_hits(self._frames()))


class TestScanLevels:
    """Per scan: the mean score of the detections matched one to one to its people, and the highest score of the rest."""

    def test_true_positive_mean_and_top_false_positive(self):
        f = TestWithinFrameRanking._frame
        frames = [f([0.9, 0.7, 0.4, 0.2], [[0, 2], [0.1, 2], [3, 3], [1, 3]], [[0, 2], [1, 3]])]
        tp, fp = scan_levels(frames)
        assert tp[0] == pytest.approx((0.9 + 0.2) / 2)
        assert fp[0] == pytest.approx(0.7), "the second detection on a taken person is a false positive"

    def test_undefined_where_there_is_nothing_to_average(self):
        f = TestWithinFrameRanking._frame
        tp, fp = scan_levels([f([0.8], [[0, 2]], [[0, 2]]), f([0.5], [[5, 5]], [[0, 2]])])
        assert np.isnan(fp[0]) and tp[0] == pytest.approx(0.8)
        assert np.isnan(tp[1]) and fp[1] == pytest.approx(0.5)


class TestScanStatistics:
    def test_returns_within_range_their_median_and_jumps_per_ten_degrees(self):
        scan = np.full((1, 40), 4.0, np.float32)          # 20 degrees at 0.5 deg
        scan[0, 10:14] = 2.0                               # one object: two jumps of 2 m
        scan[0, 30:] = np.inf                              # ten beams with no return
        s = scan_statistics(scan, [np.array([[0.0, 2.0], [3.0, 4.0]])], spacing_deg=0.5)
        assert s["returns_within_10m"][0] == pytest.approx(0.75)
        assert s["median_range_m"][0] == pytest.approx(4.0)
        assert s["jumps_per_10deg"][0] == pytest.approx(2 / 2.0)
        assert s["people"][0] == 2
        assert s["people_range_m"][0] == pytest.approx((2.0 + 5.0) / 2)

    def test_a_jump_into_a_missing_return_is_not_clutter(self):
        scan = np.full((1, 20), 3.0, np.float32)
        scan[0, 5:8] = np.nan
        assert scan_statistics(scan, [np.zeros((0, 2))], spacing_deg=0.5)["jumps_per_10deg"][0] == 0.0


class TestReferenceBins:
    def test_bins_by_the_reference_distribution_s_percentiles(self):
        reference = np.arange(1.0, 101.0)                  # p10 = 10.9, p25 = 25.75
        values = np.array([5.0, 10.9, 11.0, 25.0, 26.0, 500.0])
        assert reference_bins(values, reference, (10, 25)).tolist() == [0, 1, 1, 1, 2, 2]

    def test_an_undefined_value_reads_as_the_open_end(self):
        """A scan with no return within range has no median range: it is as open as a scan can be."""
        assert reference_bins(np.array([np.nan]), np.arange(1.0, 11.0), (10, 25)).tolist() == [2]


class TestSisterRecordings:
    def test_a_segment_whose_twin_is_in_train(self):
        train = ["run_2015-11-26-14-07-15-b", "lunch_2015-11-26-12-04-23", "run_2015-11-25-14-58-49-a"]
        val = ["run_2015-11-26-14-07-15-a", "run_2015-11-25-14-58-49-a", "run_2015-11-27-10-00-00-a", "run_2015-11-28-10-00-00"]
        assert sister_recordings(val, train).tolist() == [True, True, False, False]


class TestTransferSummary:
    @staticmethod
    def frames(n, hit=True):
        f = TestWithinFrameRanking._frame
        return [f([0.9, 0.3], [[0, 2] if hit else [3, 3], [4, 4]], [[0, 2]]) for _ in range(n)]

    def test_the_ratio_is_train_ap_over_test_ap(self):
        train, test = self.frames(4), self.frames(2) + self.frames(2, hit=False)
        r = transfer_summary(train, test, np.array([True, False, True, False]))
        assert r["train_ap"] == pytest.approx(score_detections(train)["ap"])
        assert r["ratio"] == pytest.approx(r["train_ap"] / r["test_ap"])

    def test_the_enclosed_reading_is_the_ranking_over_the_enclosed_test_scans_alone(self):
        test = self.frames(2) + self.frames(2, hit=False)
        r = transfer_summary(self.frames(2), test, np.array([True, True, False, False]))
        alone = ranking_summary(test[:2])
        assert r["enclosed"]["comparability"] == pytest.approx(alone["populated_ap"] / alone["top_k_hits"])
        assert r["enclosed"]["scans"] == 2 and r["typical"]["scans"] == 2
        assert r["all"]["scans"] == 4 and r["all"]["frames"] == 4


class TestPartitionVerdict:
    """The rule written into `TODO.md` row 106 before any run, applied to the means over seeds."""

    @staticmethod
    def rows(ratio, comparability):
        return [dict(ratio=ratio, enclosed=dict(comparability=comparability))] * 2

    @pytest.mark.parametrize("balanced,shifted,verdict", [
        ((1.3, 0.85), (2.0, 0.65), "geometry is the trigger"),
        ((1.9, 0.62), (2.0, 0.65), "geometry is not the trigger"),
        ((1.3, 0.85), (1.3, 0.85), "the construction, not the geometry"),
        ((1.5, 0.85), (2.0, 0.65), "geometry is not the trigger"),
    ])
    def test_each_outcome(self, balanced, shifted, verdict):
        assert partition_verdict(self.rows(*balanced), self.rows(*shifted))["verdict"] == verdict


class TestRangeBands:
    """Item 107's first test: AP and within-scan hits with people and detections both kept inside one range band."""

    @staticmethod
    def frames():
        f = TestWithinFrameRanking._frame
        return [f([0.9, 0.8, 0.2], [[0, 1.5], [0, 6.0], [0, 3.0]], [[0, 1.5], [0, 6.0]]),
                f([0.7, 0.6], [[3.9, 0], [0, 6.2]], [[0, 1.6], [0, 6.2]])]

    def test_a_band_keeps_only_its_own_people_and_detections(self):
        near = range_band_frames(self.frames(), 1.0, 2.5)
        assert [len(s) for s, _, _ in near] == [1, 0]
        assert [len(g) for _, _, g in near] == [1, 1]

    def test_each_band_is_scored_alone(self):
        bands = ap_by_range(self.frames(), (1.0, 2.5, 5.0, 10.0))
        assert bands["1.0-2.5 m"]["people"] == 2 and bands["5.0-10.0 m"]["people"] == 2
        assert bands["5.0-10.0 m"]["top_k_hits"] == pytest.approx(1.0)
        assert bands["1.0-2.5 m"]["top_k_hits"] == pytest.approx(0.5)

    def test_person_ranges_are_every_annotated_person_s_distance(self):
        assert person_ranges([np.array([[3.0, 4.0]]), np.zeros((0, 2)), np.array([[0.0, 2.0], [6.0, 8.0]])]).tolist() == [5.0, 2.0, 10.0]


class TestContextProbe:
    """Item 107's probe (d): a person's scan with everything outside an arc around them blanked, and the person's own
    score read before and after."""

    ANGLES = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)

    def test_the_arc_is_wider_in_beams_for_a_near_person(self):
        near, far = beam_span(self.ANGLES, 0.0, 2.0, 0.75), beam_span(self.ANGLES, 0.0, 8.0, 0.75)
        assert near[1] - near[0] > far[1] - far[0] > 0
        assert near[0] < 180 < near[1]

    def test_the_span_stays_inside_the_scan(self):
        lo, hi = beam_span(self.ANGLES, self.ANGLES[1], 0.5, 0.75)
        assert lo == 0 and hi <= 360

    def test_a_person_s_score_is_the_best_beam_on_them(self):
        logits = np.zeros(360)
        logits[178:183] = [1.0, 3.0, 2.0, 0.5, -1.0]
        logits[100] = 9.0                                 # a better beam elsewhere is not this person's
        assert person_score(logits, self.ANGLES, 0.0, 4.0, 0.35) == pytest.approx(3.0)


class TestExactTwins:
    def test_a_query_scan_found_byte_for_byte_in_the_bank(self):
        bank = np.arange(12, dtype=np.float32).reshape(4, 3)
        query = np.stack([bank[2], bank[2] + np.float32(1e-3), bank[0]])
        assert exact_twins(query, bank).tolist() == [True, False, True]


def test_the_distance_runs_on_the_given_device():
    bank = np.ones((4, 8), np.float32)
    assert nearest_distance(bank[:1], bank, max_shift=1, device=torch.device("cpu")).dtype == np.float64
