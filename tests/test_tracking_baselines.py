"""The tracking-literature baselines and metrics of `TODO.md` A56 (`memory/experiments/2026-09-tracking-claims.md`).

GOSPA, an exact fast AP and its block bootstrap, the rule scorer that puts CBMOT's score update on the memory's own bookkeeping, the
tracker replay, and the per-person strata (occlusion, first-detection latency).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from takheliped.detectors.lfe_detector import _merge_nearby
from takheliped.detectors.takheliped import SlotRules
from takheliped.utils.tracking import ByteTracker
from train_takheliped import SequenceCandidates, pack, replay_sort, score_detections
from tracking_baselines import (
    PrecisionRecall, block_ids, double_report_anatomy, drop_duplicate_annotations, first_detection_delays, gospa, gospa_frames,
    merge_detections, occlusion_share, paired_bootstrap, parse_arms, replay_rule_scorer, replay_tracker,
    report_focus, stride_frames, summarise,
)


def _seq(per_frame, poses=None, annotated=None, people=None, times=None) -> SequenceCandidates:
    """One recording from per-frame candidate lists `[(score, x, y), ...]` and annotated people `[(x, y), ...]`."""
    n, p = len(per_frame), max(1, max(len(c) for c in per_frame))
    people = people if people is not None else [[] for _ in range(n)]
    g = max(1, max(len(q) for q in people))
    seq = SequenceCandidates(
        xy=np.zeros((n, p, 2), np.float32), score=np.zeros((n, p), np.float32), valid=np.zeros((n, p), bool), target=np.zeros((n, p), bool),
        pose=np.zeros((n, 3), np.float32) if poses is None else np.asarray(poses, np.float32),
        time=np.arange(n, dtype=np.float64) * 0.025 if times is None else np.asarray(times, np.float64),
        annotated=np.ones(n, bool) if annotated is None else np.asarray(annotated, bool),
        gt_xy=np.zeros((n, g, 2), np.float32), gt_valid=np.zeros((n, g), bool), gt_covered=np.zeros((n, g), bool),
        features=np.zeros((n, p, 0), np.float16), free_range=np.full((n, 90), 10.0, np.float16))
    for f, cands in enumerate(per_frame):
        for k, (sc, x, y) in enumerate(cands):
            seq.score[f, k], seq.xy[f, k], seq.valid[f, k] = sc, (x, y), True
        for k, (x, y) in enumerate(people[f]):
            seq.gt_xy[f, k], seq.gt_valid[f, k] = (x, y), True
    return seq


def _split(per_frame, **kwargs):
    return pack([_seq(per_frame, **kwargs)])


def _random_frames(rng, n=40):
    frames = []
    for _ in range(n):
        gt = rng.uniform(-3, 3, (rng.integers(0, 4), 2))
        near = gt[rng.random(len(gt)) > 0.3]
        near = near + rng.normal(0, 0.2, near.shape)
        clutter = rng.uniform(-3, 3, (rng.integers(0, 5), 2))
        xy = np.concatenate([near, clutter]).astype(np.float32)
        scores = rng.uniform(0, 1, len(xy)).astype(np.float32)
        frames.append((scores, xy, gt.astype(np.float32)))
    return frames


class TestGospa:
    """GOSPA with alpha = 2 (Rahmathullah, Garcia-Fernandez & Svensson, FUSION 2017): localisation for assigned pairs within the
    cut-off, plus `c^p / 2` for every missed and every false target."""

    def test_perfect_detections_cost_nothing(self):
        gt = np.array([[0.0, 1.0], [2.0, 2.0]])
        assert gospa(gt.copy(), gt, c=0.5, p=1) == (0.0, 0.0, 0, 0)

    def test_every_missed_and_false_target_costs_half_the_cutoff(self):
        total, loc, missed, false = gospa(np.array([[5.0, 5.0]]), np.array([[0.0, 1.0], [2.0, 2.0]]), c=0.5, p=1)
        assert (loc, missed, false) == (0.0, 2, 1)
        assert total == pytest.approx(0.25 * 3)

    def test_a_close_pair_costs_its_distance(self):
        total, loc, missed, false = gospa(np.array([[0.0, 1.1]]), np.array([[0.0, 1.0]]), c=0.5, p=1)
        assert (missed, false) == (0, 0)
        assert total == pytest.approx(0.1) and loc == pytest.approx(0.1)

    def test_a_pair_beyond_the_cutoff_is_a_miss_and_a_false_target(self):
        total, _, missed, false = gospa(np.array([[0.0, 1.7]]), np.array([[0.0, 1.0]]), c=0.5, p=1)
        assert (missed, false) == (1, 1) and total == pytest.approx(0.5)

    def test_p_2_takes_the_root(self):
        total, loc, _, _ = gospa(np.array([[0.0, 1.3], [5.0, 5.0]]), np.array([[0.0, 1.0]]), c=0.5, p=2)
        assert loc == pytest.approx(0.09)
        assert total == pytest.approx(math.sqrt(0.09 + 0.25 / 2))

    def test_symmetric_and_satisfies_the_triangle_inequality(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            a, b, c = (rng.uniform(-1, 1, (rng.integers(0, 4), 2)) for _ in range(3))
            assert gospa(a, b, 0.5, 1)[0] == pytest.approx(gospa(b, a, 0.5, 1)[0])
            assert gospa(a, c, 0.5, 1)[0] <= gospa(a, b, 0.5, 1)[0] + gospa(b, c, 0.5, 1)[0] + 1e-9

    def test_empty_sets(self):
        assert gospa(np.zeros((0, 2)), np.zeros((0, 2)), 0.5, 1) == (0.0, 0.0, 0, 0)

    def test_the_frame_average_splits_into_its_parts(self):
        frames = _random_frames(np.random.default_rng(1))
        r = gospa_frames(frames, threshold=0.3, c=0.5, p=1)
        assert r["gospa"] == pytest.approx(r["localisation"] + 0.25 * (r["missed"] + r["false"]))


class TestPrecisionRecall:
    """AP from per-detection true-positive increments, exactly as `score_detections` computes it, so that a block bootstrap can
    re-weight frames without re-running the matching."""

    def test_reproduces_score_detections_exactly(self):
        rng = np.random.default_rng(2)
        for _ in range(5):
            frames = _random_frames(rng)
            assert PrecisionRecall(frames).ap() == score_detections(frames)["ap"]

    def test_unit_weights_change_nothing(self):
        frames = _random_frames(np.random.default_rng(3))
        pr = PrecisionRecall(frames)
        assert pr.ap(np.ones(len(frames))) == pytest.approx(pr.ap(), abs=1e-12)

    def test_doubling_every_frame_changes_nothing(self):
        frames = _random_frames(np.random.default_rng(4))
        pr = PrecisionRecall(frames)
        assert pr.ap(np.full(len(frames), 2.0)) == pytest.approx(pr.ap(), abs=1e-9)


class TestBootstrap:

    def test_blocks_never_cross_a_recording(self):
        ids = block_ids(np.array([0, 0, 0, 1, 1]), np.array([0.0, 10.0, 31.0, 0.0, 5.0]), 30.0)
        assert ids[0] == ids[1] != ids[2]
        assert len({ids[0], ids[2]} & {ids[3], ids[4]}) == 0

    def test_identical_inputs_give_a_zero_interval(self):
        frames = _random_frames(np.random.default_rng(5))
        r = paired_bootstrap(frames, frames, np.arange(len(frames)) // 5, resamples=50, seed=0)
        assert r["difference"] == 0.0 and r["low"] == 0.0 and r["high"] == 0.0

    def test_the_same_seed_gives_the_same_interval(self):
        rng = np.random.default_rng(6)
        a, b = _random_frames(rng), _random_frames(rng)
        blocks = np.arange(len(a)) // 5
        assert paired_bootstrap(a, b, blocks, 50, seed=3) == paired_bootstrap(a, b, blocks, 50, seed=3)

    def test_a_better_detector_has_an_interval_above_zero(self):
        rng = np.random.default_rng(7)
        worse = _random_frames(rng, 200)
        better = [(np.concatenate([s, np.ones(len(g), np.float32)]), np.concatenate([xy, g]), g) for s, xy, g in worse]
        r = paired_bootstrap(better, worse, np.arange(200) // 10, 200, seed=0)
        assert r["low"] > 0


class TestStrideFrames:

    def test_every_strideth_annotated_frame_per_recording(self):
        split = pack([_seq([[]] * 7, annotated=[1, 0, 1, 1, 0, 1, 1]), _seq([[]] * 4, annotated=[1, 1, 1, 1])])
        assert stride_frames(split, 2).tolist() == [0, 3, 6, 7, 9]


class TestRuleScorer:
    """CBMOT's score update on the memory's own bookkeeping: with `gain = 1` a matched slot's new value is `1 - (1 - v)(1 - s)`."""

    def test_a_matched_candidate_takes_cbmot_s_multiplication(self):
        split = _split([[(0.5, 0.0, 1.0)], [(0.6, 0.0, 1.05)]], people=[[(0.0, 1.0)], [(0.0, 1.0)]])
        frames, _ = replay_rule_scorer(split, "cpu", streams=1, rules=SlotRules(gain=1.0), unmatched_scale=1.0, coast_floor=None)
        assert frames[0][0].tolist() == pytest.approx([0.5]), "a new slot takes the candidate's own score"
        assert frames[1][0].tolist() == pytest.approx([1 - 0.5 * 0.4])

    def test_coasting_reports_an_unmatched_slot_at_its_faded_value(self):
        split = _split([[(0.9, 0.0, 1.0)], []], people=[[(0.0, 1.0)], [(0.0, 1.0)]])
        frames, _ = replay_rule_scorer(split, "cpu", 1, SlotRules(gain=1.0, fade_time_s=10.0), 1.0, coast_floor=0.5)
        scores, xy, _ = frames[1]
        assert len(scores) == 1 and 0.5 < scores[0] < 0.9
        assert xy[0] == pytest.approx([0.0, 1.0], abs=1e-5)

    def test_unmatched_candidates_are_scaled(self):
        split = _split([[(0.9, 0.0, 1.0)], [(0.4, 3.0, 3.0), (0.9, 0.0, 1.0)]])
        frames, _ = replay_rule_scorer(split, "cpu", 1, SlotRules(gain=1.0), unmatched_scale=0.5, coast_floor=None)
        assert sorted(frames[1][0].tolist()) == pytest.approx([0.2, 1 - 0.1 * 0.1]), "the new one halved, the matched one lifted"


class TestReplayTracker:

    def test_the_single_stage_tracker_replays_like_replay_sort(self):
        rng = np.random.default_rng(8)
        per_frame = [[(float(rng.uniform(0.01, 1)), float(rng.uniform(-2, 2)), float(rng.uniform(1, 5))) for _ in range(rng.integers(0, 5))]
                     for _ in range(60)]
        poses = np.cumsum(rng.normal(0, 0.01, (60, 3)), axis=0)
        split = _split(per_frame, poses=poses, people=[[(0.0, 2.0)]] * 60)
        frames, _ = replay_tracker(split, lambda: ByteTracker(two_stage=False, high_thresh=0.0, low_thresh=0.0, new_track_thresh=0.0,
                                                               match_radius=0.5, min_hits=3, max_age=1), output="tracks")
        reference = replay_sort(split, min_hits=3)
        assert len(frames) == len(reference)
        for (s, xy, _), (rs, rxy, _) in zip(frames, reference):
            assert s.tolist() == rs.tolist() and xy.tolist() == rxy.tolist()


class TestStrata:

    def test_a_return_in_front_of_a_person_occludes_them(self):
        angles = np.linspace(-np.pi / 2, np.pi / 2, 721)[:-1]
        ranges = np.full(720, 9.0)
        person = np.array([0.0, 4.0])                       # straight ahead, phi = 0
        assert occlusion_share(ranges, angles, person) == 0.0
        centre = int(np.argmin(np.abs(angles)))
        ranges[centre - 2:centre + 3] = 2.0                 # something at 2 m covering part of the person's arc
        share = occlusion_share(ranges, angles, person)
        assert 0.0 < share < 1.0
        ranges[:] = 2.0
        assert occlusion_share(ranges, angles, person) == 1.0

    def test_missing_returns_do_not_occlude(self):
        angles = np.linspace(-np.pi / 2, np.pi / 2, 721)[:-1]
        assert occlusion_share(np.full(720, np.inf), angles, np.array([1.0, 3.0])) == 0.0

    def test_first_detection_delay_counts_frames_to_the_first_covered_point(self):
        assert first_detection_delays([[(10, False), (11, False), (12, True), (13, True)], [(5, True)], [(1, False), (2, False)]]) == [2, 0, None]


class TestMergeDetections:
    """A58: the LFE detectors' greedy merge, applied to an output that has none."""

    def test_keeps_what_lfe_merge_nearby_keeps(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            n = int(rng.integers(0, 12))
            xy = rng.uniform(-2, 2, (n, 2))
            scores = np.round(rng.uniform(0, 1, n), 1)          # ties, so the order among equal scores is pinned too
            for radius in (0.0, 0.2, 0.5, 1.0):
                kept = merge_detections(scores, xy, radius)
                expected = _merge_nearby([(float(s), float(x), float(y)) for s, (x, y) in zip(scores, xy, strict=True)], radius)
                assert sorted(zip(scores[kept].tolist(), xy[kept, 0].tolist(), xy[kept, 1].tolist(), strict=True)) == sorted(expected)

    def test_the_higher_score_of_a_close_pair_survives(self):
        kept = merge_detections(np.array([0.4, 0.9, 0.7]), np.array([[0.0, 1.0], [0.1, 1.0], [3.0, 3.0]]), 0.3)
        assert kept.tolist() == [False, True, True]

    def test_a_suppressed_detection_suppresses_nothing(self):
        # 0.25 m apart in a chain: the middle one goes, so the third stays although it is within reach of the second
        kept = merge_detections(np.array([0.9, 0.8, 0.7]), np.array([[0.0, 0.0], [0.25, 0.0], [0.5, 0.0]]), 0.3)
        assert kept.tolist() == [True, False, True]

    def test_no_radius_keeps_everything(self):
        kept = merge_detections(np.array([0.5, 0.5]), np.zeros((2, 2)), None)
        assert kept.tolist() == [True, True]
        assert merge_detections(np.zeros(0), np.zeros((0, 2)), 0.3).tolist() == []


class TestDuplicateAnnotations:

    def test_drops_exact_copies_only(self):
        gt = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.001], [3.0, 4.0], [1.0, 2.0]], np.float32)
        np.testing.assert_array_equal(drop_duplicate_annotations(gt), gt[[0, 2, 3]])

    def test_empty(self):
        assert drop_duplicate_annotations(np.zeros((0, 2))).shape == (0, 2)


class TestDoubleReportAnatomy:

    def test_classifies_each_double_report_by_where_its_detections_came_from(self):
        people = np.array([[0.0, 2.0], [3.0, 2.0], [-3.0, 2.0], [6.0, 2.0], [6.0, 2.0]], np.float32)
        candidates = (np.array([0.9, 0.8, 0.85, 0.7, 0.95], np.float32),
                      np.array([[0.05, 2.0], [3.0, 2.1], [3.1, 2.0], [-3.0, 2.0], [6.0, 2.0]], np.float32), people)
        # the memory keeps every candidate at its own position, rescored, then appends its coasting slots
        memory = (np.array([0.9, 0.8, 0.85, 0.7, 0.95, 0.6, 0.65, 0.1, 0.9], np.float32),
                  np.array([[0.05, 2.0], [3.0, 2.1], [3.1, 2.0], [-3.0, 2.0], [6.0, 2.0],
                            [0.0, 2.1], [-3.1, 2.0], [-2.9, 2.0], [6.1, 2.0]], np.float32), people)
        result = double_report_anatomy([memory], [candidates], threshold=0.5)
        # person 0: a candidate and a slot; person 1: two candidates; person 2: a candidate and one slot above threshold, the other
        # below; the copied person at (6, 2) is left out, whatever surrounds it
        assert result["people"] == 3
        assert result["doubled"] == 3
        assert result["kinds"] == {"candidate+candidate": 1, "candidate+slot": 2, "slot+slot": 0}

    def test_a_single_detection_is_not_a_double_report(self):
        people = np.array([[0.0, 2.0]], np.float32)
        frame = (np.array([0.9], np.float32), np.array([[0.0, 2.0]], np.float32), people)
        result = double_report_anatomy([frame], [frame], threshold=0.5)
        assert result["people"] == 1 and result["doubled"] == 0


class TestArmsOption:
    """`--arms` replays memory checkpoints other than A56's own, e.g. one seed per stage-2 seed on the adopted recipe (`TODO.md` A61)."""

    def test_runs_are_grouped_by_arm_in_order(self):
        assert parse_arms(["E13:a", "B0:b", "E13:c"]) == {"E13": ["a", "c"], "B0": ["b"]}

    def test_a_run_without_its_arm_is_refused(self):
        with pytest.raises(ValueError, match="ARM:RUN"):
            parse_arms(["a60_e13_static_s0"])


class TestSummary:
    """A61 replays each stage-2 seed on its own, so one arm's seeds sit in three reports under three run names; the summary lines
    them up by arm."""

    @staticmethod
    def _report(seed, e13, e14, low):
        run = f"E13:a60_e13_static_s{seed}"
        return dict(ap={"candidates": dict(ap=0.82, matched_fp_recall=0.85), "bytetrack_cbmot": dict(ap=e14, matched_fp_recall=0.86),
                        run: dict(ap=e13, matched_fp_recall=0.88)},
                    bootstrap={f"{run} - bytetrack_cbmot": dict(difference=e13 - e14, low=low, high=low + 0.02)},
                    best_classical="bytetrack_cbmot",
                    gospa={run: dict(at_val_best=dict(gospa=0.5))},
                    latency={run: dict(first_frame=0.8, within_0_5_s=0.9)},
                    strata={"occlusion": {"occluded": {run: dict(recall=0.7, people=10)}}})

    def test_one_arm_s_seeds_line_up_across_reports(self):
        summary = summarise([self._report(0, 0.86, 0.84, 0.005), self._report(1, 0.85, 0.84, -0.001)])
        assert summary["ap"]["E13"]["seeds"] == [0.86, 0.85]
        assert summary["ap"]["E13"]["mean"] == pytest.approx(0.855)
        assert summary["ap"]["candidates"]["seeds"] == [0.82, 0.82]
        assert summary["matched_fp_recall"]["E13"]["mean"] == pytest.approx(0.88)
        assert summary["best_classical"] == ["bytetrack_cbmot", "bytetrack_cbmot"]

    def test_bootstrap_intervals_are_counted_above_zero(self):
        summary = summarise([self._report(0, 0.86, 0.84, 0.005), self._report(1, 0.85, 0.84, -0.001)])
        pair = summary["bootstrap"]["E13 - bytetrack_cbmot"]
        assert pair["difference"] == pytest.approx([0.02, 0.01]) and pair["above_zero"] == 1

    def test_gospa_latency_and_strata_are_averaged_by_arm(self):
        summary = summarise([self._report(0, 0.86, 0.84, 0.005), self._report(1, 0.85, 0.84, -0.001)])
        assert summary["gospa"]["E13"]["mean"] == pytest.approx(0.5)
        assert summary["first_frame"]["E13"]["mean"] == pytest.approx(0.8)
        assert summary["strata"]["occlusion"]["occluded"]["E13"]["mean"] == pytest.approx(0.7)
class TestReportFocus:
    """Which methods the per-person tables -- the strata and the first-detection latency -- actually compare."""

    def test_the_candidates_and_the_strongest_classical_pipeline_are_always_compared(self):
        assert report_focus({"candidates": {}, "sort_m1": {}}, "sort_m1") == ["candidates", "sort_m1"]

    def test_every_arm_is_included_whatever_its_label(self):
        """A hardcoded `E13:` dropped item 56's `FRZ:` arm from the strata and latency tables altogether."""
        table = {"candidates": {}, "bytetrack_cbmot": {}, "FRZ:a71_freeze_s0": {}, "E13:static": {}}
        assert report_focus(table, "bytetrack_cbmot") == ["candidates", "bytetrack_cbmot", "FRZ:a71_freeze_s0", "E13:static"]

    def test_a_report_with_no_classical_pipeline_still_compares_the_candidates_and_the_arm(self):
        assert report_focus({"candidates": {}, "FRZ:run": {}}, None) == ["candidates", "FRZ:run"]
