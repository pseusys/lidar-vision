"""Training harness of the three-horizon detector, step 1: stage 3 on saved candidates (`docs/PROPOSAL.md` §8, §10; `TODO.md` A43).

Targets, the candidate cache, the stream sampler for truncated backpropagation, early stopping and the exchange rate.
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from follow_the_drow.detectors.three_horizon import MAX_RANGE_M, CalibrationNetwork, ObjectMemory, SlotRules
from train_three_horizon import (
    BestCheckpoint, EarlyStopping, PeriodicCheckpoint, StreamSampler, build_candidates, calibration_network, decode_candidates, evaluate_calibration, candidate_pooling_weights, candidate_targets, clip_frames, clip_length, coast_targets, decode_votes, evaluate, evaluation_segments,
    FrameSplit, SequenceCandidates, exchange_rate, first_recordings, frames_from_dataset, joint_frame_loss, joint_step, network_candidates,
    DT_AVERAGE_FRAMES, MIN_DT_S, TOP_P, coasting_hits, load_resume, memory_schedule, non_finite_summary, rng_restore, rng_snapshot, save_resume, training_fault, object_memory, frame_periods, gather, mean_frame_period, memory_ms_per_frame, pack, plateau_schedule, recording_bounds, replay_sort, score_detections,
)


class TestCandidateTargets:

    def test_the_nearest_candidate_within_the_radius_is_the_target(self):
        target, covered = candidate_targets(np.array([[0, 1.0], [0, 1.3], [3, 3]]), np.ones(3, bool), np.array([[0, 1.1]]), 0.5)
        assert target.tolist() == [True, False, False]
        assert covered.tolist() == [True]

    def test_one_candidate_per_person_and_one_person_per_candidate(self):
        target, covered = candidate_targets(np.array([[0, 1.0]]), np.ones(1, bool), np.array([[0, 1.1], [0, 0.9]]), 0.5)
        assert target.tolist() == [True]
        assert covered.sum() == 1

    def test_candidates_beyond_the_radius_or_invalid_are_never_targets(self):
        target, covered = candidate_targets(np.array([[0, 1.6], [0, 1.0]]), np.array([True, False]), np.array([[0, 1.0]]), 0.5)
        assert target.tolist() == [False, False]
        assert covered.tolist() == [False]

    def test_no_people_means_no_targets(self):
        target, covered = candidate_targets(np.array([[0, 1.0]]), np.ones(1, bool), np.empty((0, 2)), 0.5)
        assert target.tolist() == [False] and covered.shape == (0,)


class _FakeDetector:
    """Returns fixed (score, x, y) candidates per scan, looked up by the scan's first range."""

    def __init__(self, by_marker):
        self.by_marker = by_marker

    def detect(self, scan, angles):
        return self.by_marker[float(scan[0])]


def _fake_dataset():
    odoms = np.zeros(4, dtype=[("eq", np.uint32), ("t", np.float64), ("xya", np.float32, 3)])
    odoms["xya"][:, 0] = [0.0, 0.1, 0.2, 0.3]
    scans = np.stack([np.full(8, float(i)) for i in range(4)]).astype(np.float32)
    # as FROG_Dataset builds them: det_id are scan ids, idet2iscan maps annotation index -> scan index as a dict,
    # and annotations need not start at scan 0 (train and val drop each recording's first scans)
    return SimpleNamespace(
        scans=[scans], odoms=[odoms], scan_time=[np.array([10.0, 10.04, 10.08, 10.12])],
        det_id=[np.array([1, 3])], idet2iscan=[{0: 1, 1: 3}],
        det_wp=[[[(2.0, 0.0)], []]],
    )


class TestBuildCandidates:

    def test_keeps_the_best_candidates_padded_with_annotations_on_annotated_frames(self):
        det = _FakeDetector({0.0: [], 1.0: [(0.2, 0.0, 2.1), (0.9, 1.0, 1.0), (0.5, 0.0, 2.0)], 2.0: [(0.4, 0.0, 1.0)], 3.0: []})
        features = {0.0: [], 1.0: [[1, 1], [2, 2], [3, 3]], 2.0: [[4, 4]], 3.0: []}
        ds = _fake_dataset()

        def candidates_of(s):
            return [(np.asarray(det.detect(scan, None), float).reshape(-1, 3), np.asarray(features[float(scan[0])], float).reshape(-1, 2))
                    for scan in ds.scans[s]]

        seq = build_candidates(ds, candidates_of, top_p=2, radius=0.5)[0]
        assert seq.score.shape == (4, 2) and seq.xy.shape == (4, 2, 2)
        assert seq.score[1].tolist() == pytest.approx([0.9, 0.5])
        assert seq.xy[1, 1].tolist() == pytest.approx([0.0, 2.0])
        assert seq.valid.tolist() == [[False, False], [True, True], [True, False], [False, False]]
        assert seq.annotated.tolist() == [False, True, False, True]
        assert seq.gt_xy[1, 0].tolist() == pytest.approx([0.0, 2.0])          # r = 2 straight ahead: x right, y forward
        assert seq.gt_valid[1].tolist() == [True] and seq.gt_valid[3].tolist() == [False]
        assert seq.target[1].tolist() == [False, True]                       # (0, 2.0) is the person, not the 0.9 at (1, 1)
        assert seq.pose[:, 0].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])
        assert seq.features[1].tolist() == [[2, 2], [3, 3]]              # features follow their candidates through the cut
        assert seq.features.dtype == np.float16


class TestNetworkCandidates:

    def test_every_scan_gets_candidates_with_one_feature_row_each(self):
        from follow_the_drow.detectors.three_horizon import CalibrationNetwork
        torch.manual_seed(0)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = CalibrationNetwork(torch.from_numpy(angles), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1).eval()
        scans = (np.random.default_rng(0).random((6, 48)) * 4 + 1).astype(np.float32)
        per_frame = network_candidates(net, scans, np.zeros((6, 3), np.float32), "cpu")
        assert len(per_frame) == 6
        for cands, feats in per_frame:
            assert cands.shape[1] == 3 and feats.shape == (len(cands), 8)

    def test_decoding_anchors_votes_on_the_ranges_the_network_read(self):
        """Votes are offsets from a beam's endpoint, so decoding must use the sanitised range the network saw: a missing return and
        a reading at the range limit are the same scan to the whole detector, not only to its input layer."""
        from follow_the_drow.detectors.three_horizon import CalibrationNetwork
        torch.manual_seed(0)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = CalibrationNetwork(torch.from_numpy(angles), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1).eval()
        at_limit = (np.random.default_rng(0).random((4, 48)) * 4 + 1).astype(np.float32)
        at_limit[:, 10:14] = net.max_range_m
        missing = at_limit.copy()
        missing[:, 10:12] = np.inf
        missing[:, 12:14] = 61.0
        poses = np.zeros((4, 3), np.float32)
        for (want, _), (got, _) in zip(network_candidates(net, at_limit, poses, "cpu"), network_candidates(net, missing, poses, "cpu")):
            assert np.allclose(want, got)

    def test_evaluation_scores_a_missing_return_like_a_reading_at_the_limit(self):
        from follow_the_drow.detectors.three_horizon import CalibrationNetwork
        torch.manual_seed(0)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = CalibrationNetwork(torch.from_numpy(angles), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1, coarse_lags=(0,))
        with torch.no_grad():                            # every beam confidently votes for its own endpoint, so a dropped beam changes the detections
            net.head.weight.zero_()
            net.head.bias.copy_(torch.tensor([3.0, 0.0, 0.0]))
        at_limit = (np.random.default_rng(1).random((3, 48)) * 4 + 1).astype(np.float32)
        at_limit[:, 0:10] = 3.0                          # a wall the annotated person stands on, so there is a true positive
        at_limit[:, 20:30] = net.max_range_m
        missing = at_limit.copy()
        missing[:, 20:30] = np.inf
        phi = float(angles[5])

        def split(scans):
            people = [np.array([[3.0, phi]])] * 3
            xy = [np.array([[-3.0 * np.sin(phi), 3.0 * np.cos(phi)]])] * 3
            return FrameSplit(scans, np.zeros((3, 3), np.float32), np.zeros(3, np.int64), np.arange(3), people, xy, np.arange(3) / 40.0)

        want = evaluate_calibration(net, split(at_limit), np.arange(3), 1, "cpu", 3)
        got = evaluate_calibration(net, split(missing), np.arange(3), 1, "cpu", 3)
        assert want["fp_per_frame"] > 0
        assert got == pytest.approx(want)

    def test_beams_at_the_range_limit_cast_no_vote(self):
        """The default decoding rule (`TODO.md` A50, owner's call 2026-09-17): a beam with nothing within range has no surface for
        a person to stand on, so it casts no vote. It raised test AP by +0.28 to +0.39 pp on all six seed-matched checkpoints."""
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        prob, votes = np.full(48, 0.95, np.float32), np.zeros((48, 2), np.float32)
        wall = np.full(48, 3.0, np.float32)
        nothing = np.full(48, np.inf, np.float32)
        at_limit = np.full(48, 10.0, np.float32)
        assert len(decode_candidates(wall, angles, prob, votes, TOP_P, 10.0)[0]) > 0
        assert len(decode_candidates(nothing, angles, prob, votes, TOP_P, 10.0)[0]) == 0
        assert len(decode_candidates(at_limit, angles, prob, votes, TOP_P, 10.0)[0]) == 0
        assert len(decode_candidates(at_limit, angles, prob, votes, TOP_P, 20.0)[0]) > 0      # the limit, not the value, decides

    def test_evaluation_applies_the_same_rule(self):
        from follow_the_drow.detectors.three_horizon import CalibrationNetwork
        torch.manual_seed(0)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = CalibrationNetwork(torch.from_numpy(angles), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1, coarse_lags=(0,))
        with torch.no_grad():
            net.head.weight.zero_()
            net.head.bias.copy_(torch.tensor([3.0, 0.0, 0.0]))
        phi = float(angles[24])
        scans = np.stack([np.full(48, 3.0), np.full(48, np.inf)]).astype(np.float32)      # a wall, then nothing within range
        people = [np.array([[3.0, phi]]), np.array([[net.max_range_m, phi]])]
        xy = [np.array([[-r * np.sin(phi), r * np.cos(phi)]]) for r in (3.0, net.max_range_m)]
        split = FrameSplit(scans, np.zeros((2, 3), np.float32), np.zeros(2, np.int64), np.arange(2), people, xy, np.arange(2) / 40.0)
        assert evaluate_calibration(net, split, np.arange(2), 1, "cpu", 2)["recall"] == pytest.approx(0.5)     # only the wall's person is found


class TestFramesFromDataset:

    def test_annotations_land_on_their_own_scans_when_they_do_not_start_at_scan_zero(self):
        frames = frames_from_dataset(_fake_dataset())
        assert frames.annotated.tolist() == [1, 3]
        assert frames.people_polar[0].tolist() == [[2.0, 0.0]]
        assert frames.people_xy[0].tolist() == [pytest.approx([0.0, 2.0])]
        assert frames.people_xy[1].shape == (0, 2)
        assert frames.scans[3, 0] == 3.0 and frames.start.tolist() == [0, 0, 0, 0]
        assert frames.time.tolist() == pytest.approx([10.0, 10.04, 10.08, 10.12])


class TestStreamSampler:

    def test_streams_walk_whole_recordings_resetting_at_each_start(self):
        sampler = StreamSampler([5, 3, 4], batch=2, rng=np.random.default_rng(0))
        previous = None
        for _ in range(6):
            seq, frame, reset = sampler.next_chunk(4, reset_prob=0.0)
            assert (reset == (frame == 0)).all()
            for b in range(2):
                for t in range(4):
                    if not reset[b, t]:
                        before = (seq[b, t - 1], frame[b, t - 1]) if t else previous[b]
                        assert (seq[b, t], frame[b, t]) == (before[0], before[1] + 1)
            previous = [(seq[b, -1], frame[b, -1]) for b in range(2)]

    def test_a_chunk_start_resets_with_the_given_probability(self):
        sampler = StreamSampler([50, 60], batch=3, rng=np.random.default_rng(0))
        for _ in range(5):
            _, _, reset = sampler.next_chunk(7, reset_prob=1.0)
            assert reset[:, 0].all()

    def test_every_recording_is_visited_within_one_pass_of_frames(self):
        lengths = [7, 3, 9, 4, 6]
        sampler = StreamSampler(lengths, batch=2, rng=np.random.default_rng(1))
        seen = set()
        for _ in range(math.ceil(sum(lengths) / (2 * 5)) + 1):
            seq, _, _ = sampler.next_chunk(5, reset_prob=0.0)
            seen.update(seq.ravel().tolist())
        assert seen == set(range(len(lengths)))


class TestEarlyStopping:

    def test_stops_after_patience_evaluations_without_improvement(self):
        stop = EarlyStopping(patience_evals=2, min_epochs=0, max_epochs=100, max_hours=100)
        assert [stop.update(m) for m in [0.5, 0.6, 0.55, 0.58]] == [True, True, False, False]
        assert stop.should_stop(epoch=4, hours=0)
        assert stop.best == pytest.approx(0.6)

    def test_never_stops_before_the_minimum_epochs(self):
        stop = EarlyStopping(patience_evals=1, min_epochs=5, max_epochs=100, max_hours=100)
        for m in [0.5, 0.4, 0.3]:
            stop.update(m)
        assert not stop.should_stop(epoch=3, hours=0)
        assert stop.should_stop(epoch=5, hours=0)

    def test_stops_at_the_epoch_and_time_caps(self):
        stop = EarlyStopping(patience_evals=10, min_epochs=0, max_epochs=3, max_hours=1.0)
        stop.update(0.5)
        assert not stop.should_stop(epoch=2, hours=0.5)
        assert stop.should_stop(epoch=3, hours=0.5)
        assert stop.should_stop(epoch=1, hours=1.0)


class TestBestCheckpoint:

    def test_the_best_weights_survive_any_later_epoch_or_stage(self, tmp_path):
        keeper = BestCheckpoint(tmp_path / "best.pth")
        assert keeper.offer(0.688, {"w": 1})
        # the next chunk length's first epoch, just after the optimizer restarts, is worse and must not replace it
        assert not keeper.offer(0.666, {"w": 2})
        assert torch.load(tmp_path / "best.pth")["w"] == 1
        assert keeper.offer(0.700, {"w": 3})
        assert torch.load(tmp_path / "best.pth")["w"] == 3
        assert keeper.best == pytest.approx(0.700)

    def test_it_reports_whether_anything_was_ever_saved(self, tmp_path):
        keeper = BestCheckpoint(tmp_path / "best.pth")
        assert not keeper.saved
        keeper.offer(float("nan"), {"w": 1})                 # a run whose every evaluation was undefined saves nothing
        assert not keeper.saved
        keeper.offer(0.5, {"w": 2})
        assert keeper.saved

    def test_an_undefined_metric_never_replaces_the_best(self, tmp_path):
        keeper = BestCheckpoint(tmp_path / "best.pth")
        keeper.offer(0.5, {"w": 1})
        assert not keeper.offer(float("nan"), {"w": 2})
        assert torch.load(tmp_path / "best.pth")["w"] == 1


class TestClipLength:

    def test_a_clip_spans_the_largest_coarse_lag_and_the_current_frame(self):
        assert clip_length((0, 8, 16, 24, 32)) == 33
        assert clip_length((0,)) == 1


class TestCalibrationNetworkFor:

    def test_the_range_limit_and_lags_come_from_the_checkpoint_args(self):
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = CalibrationNetwork(torch.from_numpy(angles), coarse_lags=(0, 4), max_range_m=15.0)
        checkpoint = dict(model=net.state_dict(), args={"coarse_lags": "[0, 4]", "max_range_m": "15.0"})
        rebuilt = calibration_network(checkpoint, angles)
        assert rebuilt.max_range_m == 15.0 and rebuilt.coarse.lags == [0, 4]

    def test_a_checkpoint_from_before_the_flags_uses_the_defaults(self):
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = CalibrationNetwork(torch.from_numpy(angles))
        rebuilt = calibration_network(dict(model=net.state_dict(), args={}), angles)
        assert rebuilt.max_range_m == MAX_RANGE_M and rebuilt.coarse.lags == [0, 8, 16, 24, 32]


class TestObjectMemoryFor:
    """A stage-3 checkpoint carries the rescoring arms it was trained with (`TODO.md` A51 T1, O1), so anything rebuilding it from
    disk must read them rather than assume the defaults -- a T1 checkpoint has no `rescore_gate` parameter at all."""

    def test_the_arms_and_slot_count_come_from_the_checkpoint_args(self):
        trained = ObjectMemory(SlotRules(capacity=8), dim=32, heads=4, feature_dim=4, rescore_gate=False, rescore_limit=3.0)
        checkpoint = dict(model=trained.state_dict(), args={"slots": "8", "rescore_gate": "False", "rescore_limit": "3.0"})
        rebuilt = object_memory(checkpoint, feature_dim=4, dim=32, heads=4)
        assert rebuilt.rescore_gate is None and rebuilt.rescore_limit == 3.0 and rebuilt.rules.capacity == 8
        assert set(rebuilt.state_dict()) == set(checkpoint["model"])

    def test_the_slot_ablations_survive_a_round_trip(self):
        trained = ObjectMemory(SlotRules(capacity=8, latest_value=True), dim=32, heads=4, feature_dim=4, room_frame=False,
                               fixed_decay_s=8.0, learn_value=True)
        checkpoint = dict(model=trained.state_dict(),
                          args={"slots": "8", "latest_value": "True", "room_frame": "False", "fixed_decay_s": "8.0", "learn_value": "True"})
        rebuilt = object_memory(checkpoint, feature_dim=4, dim=32, heads=4)
        assert not rebuilt.room_frame and rebuilt.rules.latest_value and rebuilt.fixed_decay_s == 8.0 and rebuilt.decay is None
        assert set(rebuilt.state_dict()) == set(checkpoint["model"])

    def test_a_capacity_override_keeps_the_checkpoint_own_slot_rules(self):
        # the slot-count ablation rebuilds a trained memory at another capacity; it must not silently drop how that memory holds slots
        trained = ObjectMemory(SlotRules(capacity=8, latest_value=True), dim=32, heads=4, feature_dim=4)
        checkpoint = dict(model=trained.state_dict(), args={"slots": "8", "latest_value": "True"})
        rebuilt = object_memory(checkpoint, feature_dim=4, capacity=16, dim=32, heads=4)
        assert rebuilt.rules.capacity == 16 and rebuilt.rules.latest_value

    def test_a_checkpoint_from_before_the_flags_keeps_the_gate(self):
        trained = ObjectMemory(SlotRules(capacity=8), dim=32, heads=4, feature_dim=4)
        checkpoint = dict(model=trained.state_dict(), args={"slots": "8"})
        rebuilt = object_memory(checkpoint, feature_dim=4, dim=32, heads=4)
        assert rebuilt.rescore_gate is not None and rebuilt.rescore_limit == 0.0
        assert set(rebuilt.state_dict()) == set(checkpoint["model"])


class TestPeriodicCheckpoint:

    def test_it_keeps_one_checkpoint_per_interval(self, tmp_path):
        keeper = PeriodicCheckpoint(tmp_path, "step2_calibration", 0.5)
        assert keeper.offer(0.25, {"w": 1}) is None
        assert keeper.offer(0.50, {"w": 2}) is not None
        assert keeper.offer(0.75, {"w": 3}) is None
        assert keeper.offer(1.00, {"w": 4}) is not None
        assert [p.name for p in keeper.saved] == ["step2_calibration.epoch0.50.pth", "step2_calibration.epoch1.00.pth"]

    def test_zero_epochs_keeps_nothing(self, tmp_path):
        keeper = PeriodicCheckpoint(tmp_path, "step2_calibration", 0.0)
        assert keeper.offer(1.0, {"w": 1}) is None
        assert not list(tmp_path.iterdir())

    def test_an_evaluation_landing_a_hair_early_still_counts(self, tmp_path):
        # step 2's evaluations land at 0.2499, 0.4998, ...: its iterations do not divide an epoch exactly
        keeper = PeriodicCheckpoint(tmp_path, "s", 0.25)
        assert keeper.offer(0.2499248120300752, {"w": 1}) is not None
        assert keeper.offer(0.4998496240601504, {"w": 2}) is not None

    def test_intervals_that_pass_together_save_once_rather_than_once_each(self, tmp_path):
        keeper = PeriodicCheckpoint(tmp_path, "s", 0.25)
        assert keeper.offer(1.0, {"w": 1}) is not None      # four intervals went by between evaluations
        assert keeper.offer(1.1, {"w": 2}) is None          # the next falls due at 1.25, not immediately
        assert keeper.offer(1.25, {"w": 3}) is not None

    def test_the_saved_checkpoint_round_trips(self, tmp_path):
        path = PeriodicCheckpoint(tmp_path, "s", 1.0).offer(1.0, {"w": 7})
        assert torch.load(path)["w"] == 7


class TestCoastTargets:

    def test_a_coasting_slot_on_a_missed_person_is_positive_and_only_one_per_person(self):
        slot_xy = torch.tensor([[[0.0, 2.0], [0.1, 2.0], [0.0, 5.0], [3.0, 3.0]]])
        coasting = torch.tensor([[True, True, True, False]])
        gt_xy = torch.tensor([[[0.05, 2.0], [0.0, 5.1], [3.0, 3.0]]])
        missed = torch.tensor([[True, False, True]])
        target = coast_targets(slot_xy, coasting, gt_xy, missed, 0.5)
        # slots 0 and 1 compete for the missed person at (0.05, 2): exactly one wins; slot 2 sits on a covered person;
        # slot 3 is on a missed person but is not coasting
        assert target[0, :2].sum() == 1
        assert target[0, 2:].tolist() == [False, False]


class TestExchangeRate:

    def test_false_positives_removed_per_false_negative_added(self):
        assert exchange_rate(fp_base=1.0, fn_base=0.5, fp_new=0.6, fn_new=0.6) == pytest.approx(4.0)

    def test_removing_false_positives_without_adding_false_negatives_is_unbounded(self):
        assert exchange_rate(fp_base=1.0, fn_base=0.5, fp_new=0.6, fn_new=0.5) == math.inf


class TestEvaluationSegments:

    def test_whole_recordings_when_no_windows_are_asked_for(self):
        split = SimpleNamespace(lengths=np.array([100, 40]))
        assert evaluation_segments(split, windows=0, window_frames=10, warmup_frames=5) == [(0, 0, 100, 0), (1, 0, 40, 0)]

    def test_windows_with_their_warm_up_fit_inside_long_enough_recordings(self):
        split = SimpleNamespace(lengths=np.array([1000, 30, 500]))
        segments = evaluation_segments(split, windows=6, window_frames=50, warmup_frames=20)
        assert 5 <= len(segments) <= 7
        for recording, first, frames, warmup in segments:
            assert recording != 1 and frames == 70 and warmup == 20
            assert 0 <= first and first + frames <= split.lengths[recording]


class TestCandidatePoolingWeights:

    def test_features_are_the_probability_weighted_mean_of_the_beams_voting_near_the_candidate(self):
        det_xy = np.array([[0.0, 2.0], [5.0, 5.0]])
        vote_xy = np.array([[0.0, 2.1], [0.1, 2.0], [0.0, 3.0], [5.0, 5.0]])
        prob = np.array([0.9, 0.3, 0.9, 0.0])
        features = np.array([[1.0, 0.0], [0.0, 1.0], [7.0, 7.0], [9.0, 9.0]])
        pooled = candidate_pooling_weights(det_xy, vote_xy, prob, radius=0.5) @ features
        assert pooled[0].tolist() == pytest.approx([0.75, 0.25])     # beams 0 and 1 (weights 0.9 and 0.3); beam 2 is 1 m away
        assert pooled[1].tolist() == [0.0, 0.0]                      # its only nearby beam has zero probability


class TestClipFrames:

    def test_history_is_clamped_at_the_recording_start(self):
        assert clip_frames(3, 0, 5).tolist() == [0, 0, 1, 2, 3]
        assert clip_frames(101, 100, 4).tolist() == [100, 100, 100, 101]

    def test_a_clip_away_from_the_start_is_the_frames_before_the_target(self):
        assert clip_frames(107, 100, 5).tolist() == [103, 104, 105, 106, 107]


class TestDecodeVotes:

    def test_votes_built_from_a_person_decode_back_onto_the_person(self):
        from train import make_targets
        angles = np.linspace(-np.pi / 2, np.pi / 2, 720, endpoint=False).astype(np.float32)
        px, py, radius = 0.5, 3.0, 0.15                  # detection frame: x right, y forward
        dx, dy = -np.sin(angles.astype(np.float64)), np.cos(angles.astype(np.float64))
        b = -(dx * px + dy * py)
        disc = b ** 2 - (px ** 2 + py ** 2 - radius ** 2)
        scan = np.where(disc >= 0, -b - np.sqrt(np.clip(disc, 0, None)), 20.0).astype(np.float32)
        r_gt, phi_gt = math.hypot(px, py), math.atan2(-px, py)
        labels, votes = make_targets(scan, angles, {3: [(r_gt, phi_gt)]})
        prob = np.where(labels == 3, 0.9, 0.01).astype(np.float32)
        detections = decode_votes(scan, angles, prob, votes)
        best = max(detections)
        assert best[0] > 0.5
        assert math.hypot(best[1] - px, best[2] - py) < 0.1


class TestJointStep:
    """Stage 2 and stage 3 run together on one frame, with the memory drawn back into stage 2 (`docs/PROPOSAL.md` §5.5, phase C)."""

    def _parts(self):
        from follow_the_drow.detectors.three_horizon import CalibrationNetwork, ObjectMemory, SlotRules
        torch.manual_seed(0)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = CalibrationNetwork(torch.from_numpy(angles), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1, prior_channels=2).eval()
        memory = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, feature_dim=8).eval()
        scans = torch.from_numpy((np.random.default_rng(0).random((1, 6, 48)) * 4 + 1).astype(np.float32))
        return net, memory, scans, torch.zeros(1, 6, 3), torch.full((1,), 0.04)

    def test_with_feedback_and_rescoring_still_silent_it_is_stage_2_then_stage_3(self):
        net, memory, scans, poses, dt = self._parts()
        state, calibration, slots = None, None, memory.initial_state(1)
        with torch.no_grad():
            for t in range(6):
                joint, state = joint_step(net, memory, scans[:, t], poses[:, t], dt, state)
                _, _, _, calibration = net.step(scans[:, t], poses[:, t], calibration, torch.zeros(1, 2, 6))
                alone = memory.step(joint.cand_xy, joint.cand_score, joint.cand_valid, poses[:, t], dt, slots, cand_features=joint.cand_features)
                slots = alone.slots
                assert torch.allclose(joint.memory.candidate_logit, alone.candidate_logit, atol=1e-5), t
                assert torch.equal(joint.memory.slots.alive, alone.slots.alive), t

    def test_the_memory_is_drawn_into_stage_2_and_a_fresh_start_draws_nothing(self):
        net, memory, scans, poses, dt = self._parts()
        with torch.no_grad():
            first, state = joint_step(net, memory, scans[:, 0], poses[:, 0], dt, None)
            second, state = joint_step(net, memory, scans[:, 1], poses[:, 1], dt, state)
            fresh, _ = joint_step(net, memory, scans[:, 2], poses[:, 2], dt, None)
        assert first.prior.abs().sum() == 0 and fresh.prior.abs().sum() == 0
        assert state.slots.alive.any() and second.prior.abs().sum() > 0

    def test_the_memory_loss_reaches_stage_2_through_the_candidate_features(self):
        net, memory, scans, poses, dt = self._parts()
        net.train(), memory.train()
        with torch.no_grad():
            memory.rescore_gate.fill_(1.0)
        joint, _ = joint_step(net, memory, scans[:, 0], poses[:, 0], dt, None)
        joint.memory.candidate_logit[joint.cand_valid].sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


class TestRecordingBounds:

    def test_offsets_and_lengths_come_from_each_frames_recording_start(self):
        offsets, lengths = recording_bounds(np.array([0, 0, 0, 3, 3, 5, 5, 5, 5]))
        assert offsets.tolist() == [0, 3, 5] and lengths.tolist() == [3, 2, 4]


class TestFramePeriods:
    """Stage 3's time step never trusts one stamped interval (`TODO.md` A51 0b): FROG stamps 40 Hz scans in bunches of 38 ms, 38 ms and
    under 0.1 ms, and DROW rounds its 12.7 Hz stamps to 0.05 s. A running average of the intervals is one number per stream."""

    @staticmethod
    def _bunched(n):
        return np.concatenate([[100.0], 100.0 + np.cumsum(np.tile([0.038, 0.038, 0.00005], n // 3 + 1)[:n - 1])])

    def test_bunched_stamps_settle_on_the_true_period(self):
        time = self._bunched(400)
        dt = frame_periods(time, np.array([0]), np.array([400]), 32)
        assert np.all(np.abs(dt[100:] - 0.02535) / 0.02535 < 0.05)

    def test_rounded_stamps_settle_on_the_true_period(self):
        true = np.arange(600) / 12.7
        time = np.round(true / 0.05) * 0.05
        dt = frame_periods(time, np.array([0]), np.array([600]), 32)
        assert np.all(np.abs(dt[100:] - 1 / 12.7) / (1 / 12.7) < 0.15)

    def test_it_is_the_plain_mean_of_the_intervals_until_the_window_fills(self):
        time = np.array([0.0, 0.01, 0.03, 0.06])
        dt = frame_periods(time, np.array([0]), np.array([4]), 32)
        assert dt[1:].tolist() == pytest.approx([0.01, 0.015, 0.02])

    def test_it_is_causal(self):
        time = self._bunched(200)
        later = time.copy()
        later[150:] += 0.3
        a = frame_periods(time, np.array([0]), np.array([200]), 32)
        b = frame_periods(later, np.array([0]), np.array([200]), 32)
        assert np.array_equal(a[:150], b[:150]) and not np.array_equal(a[150:], b[150:])

    def test_recordings_are_independent_and_their_first_frame_has_no_elapsed_time(self):
        first, second = self._bunched(90), np.arange(50) * 0.1
        dt = frame_periods(np.concatenate([first, second]), np.array([0, 90]), np.array([90, 50]), 32)
        alone = frame_periods(second, np.array([0]), np.array([50]), 32)
        assert np.array_equal(dt[90:], alone)
        assert dt[0] == dt[90] == MIN_DT_S

    def test_the_mean_period_pools_elapsed_time_over_frames(self):
        time = np.concatenate([np.arange(11) * 0.025, 50.0 + np.arange(21) * 0.1])
        assert mean_frame_period(time, np.array([0, 11]), np.array([11, 21])) == pytest.approx((0.25 + 2.0) / 30)

    def test_gather_hands_stage_3_the_running_period(self):
        n = 60
        seq = SequenceCandidates(
            xy=np.zeros((n, 1, 2), np.float32), score=np.zeros((n, 1), np.float32), valid=np.zeros((n, 1), bool), target=np.zeros((n, 1), bool),
            pose=np.zeros((n, 3), np.float32), time=self._bunched(n), annotated=np.ones(n, bool), gt_xy=np.zeros((n, 1, 2), np.float32),
            gt_valid=np.zeros((n, 1), bool), gt_covered=np.zeros((n, 1), bool), features=np.zeros((n, 1, 0), np.float16))
        split = pack([seq])
        got = gather(split, np.zeros((1, n), dtype=int), np.arange(n)[None], "cpu")["dt"][0].numpy()
        assert got == pytest.approx(frame_periods(seq.time, np.array([0]), np.array([n]), DT_AVERAGE_FRAMES).astype(np.float32))


class TestJointFrameLoss:

    def _frame(self):
        parts = TestJointStep()._parts()
        net, memory, scans, poses, dt = parts
        net.train(), memory.train()
        joint, _ = joint_step(net, memory, scans[:, 0], poses[:, 0], dt, None)
        return net, joint, scans[:, 0].numpy()

    def test_an_unannotated_frame_contributes_nothing(self):
        net, joint, scans = self._frame()
        loss = joint_frame_loss(joint, scans, net.angles.numpy(), [None], [None])
        assert loss.item() == 0.0

    def test_an_annotated_frame_trains_stage_2_and_stage_3(self):
        net, joint, scans = self._frame()
        polar = np.array([[2.0, 0.0]])
        loss = joint_frame_loss(joint, scans, net.angles.numpy(), [polar], [np.array([[0.0, 2.0]])])
        assert torch.isfinite(loss) and loss.item() > 0
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


class TestScoreDetections:

    def test_no_scored_frames_gives_undefined_metrics_instead_of_crashing(self):
        # a validation window can contain no annotated frame; its AP is undefined, and undefined never counts as a new best
        result = score_detections([])
        assert math.isnan(result["ap"]) and result["frames"] == 0
        assert not BestCheckpoint(Path("unused.pth")).offer(result["ap"], {})

    def test_precision_and_recall_at_the_operating_point_are_reported(self):
        # frame 0: the detection sits on the person; frame 1: it is metres away, so that frame holds one false positive and one miss
        person = np.array([[0.0, 1.0]], np.float32)
        frames = [(np.array([0.9], np.float32), np.array([[0.0, 1.0]], np.float32), person),
                  (np.array([0.8], np.float32), np.array([[5.0, 5.0]], np.float32), person)]
        result = score_detections(frames)
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(0.5)
        assert result["fp_per_frame"] == pytest.approx(0.5)
        assert result["fn_per_frame"] == pytest.approx(0.5)

    def test_the_match_radius_decides_whether_a_nearby_detection_counts(self):
        person = np.array([[0.0, 1.0]], np.float32)
        frames = [(np.array([0.9], np.float32), np.array([[0.4, 1.0]], np.float32), person)]
        assert score_detections(frames, radius=0.5)["recall"] == pytest.approx(1.0)
        assert score_detections(frames, radius=0.3)["recall"] == pytest.approx(0.0)

    def test_the_own_operating_point_is_the_lowest_threshold_within_the_false_positive_budget(self):
        # two frames, one person each; scores 0.9 hit, 0.25 hit, 0.2 miss, 0.1 miss: a budget of 0 false positives stops at 0.25
        person = np.array([[0.0, 1.0]], np.float32)
        on, off = np.array([[0.0, 1.0]], np.float32), np.array([[5.0, 5.0]], np.float32)
        frames = [(np.array([0.9, 0.2], np.float32), np.concatenate([on, off]), person),
                  (np.array([0.25, 0.1], np.float32), np.concatenate([on, off]), person)]
        result = score_detections(frames, fp_budget=0.0)
        assert result["own_threshold"] == pytest.approx(0.25)
        assert result["own_recall"] == pytest.approx(1.0)
        assert result["own_fp_per_frame"] == pytest.approx(0.0)
        assert result["own_fn_per_frame"] == pytest.approx(0.0)


class TestCoastingHits:
    """A coasting report is only worth something on a person no kept candidate already covers (`TODO.md` A51 0c)."""

    def test_a_report_on_an_uncovered_person_is_a_hit_and_on_a_covered_one_is_not(self):
        gt = np.array([[0.0, 1.0], [3.0, 3.0]], np.float32)
        candidates = np.array([[0.0, 1.05]], np.float32)
        reports = np.array([[0.05, 1.0], [3.0, 3.1]], np.float32)
        assert coasting_hits(reports, candidates, gt, 0.5) == 1

    def test_one_hit_per_person(self):
        gt = np.array([[3.0, 3.0]], np.float32)
        reports = np.array([[3.0, 3.1], [3.1, 3.0]], np.float32)
        assert coasting_hits(reports, np.zeros((0, 2), np.float32), gt, 0.5) == 1

    def test_no_reports_or_no_people_hit_nothing(self):
        assert coasting_hits(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.array([[1.0, 1.0]], np.float32), 0.5) == 0
        assert coasting_hits(np.array([[1.0, 1.0]], np.float32), np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), 0.5) == 0


class TestPlateauSchedule:

    def test_the_rate_is_cut_after_evaluations_without_improvement_down_to_a_floor(self):
        optimizer = torch.optim.AdamW(torch.nn.Linear(1, 1).parameters(), lr=1e-3)
        schedule = plateau_schedule(optimizer, factor=0.3, patience_evals=2, min_lr=1e-4)
        rates = []
        for ap in [0.70, 0.72, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]:
            schedule.step(ap)
            rates.append(optimizer.param_groups[0]["lr"])
        assert rates[:4] == pytest.approx([1e-3] * 4)       # a new best, then two stalled evaluations are tolerated
        assert rates[4] == pytest.approx(3e-4)             # the third stalled evaluation cuts the rate
        assert min(rates) == pytest.approx(1e-4)           # and it never goes below the floor


class TestResume:
    """A driver bugcheck took a healthy 3.5 h run with it on 2026-09-22 (`memory/gotchas.md`), and every interruption this week cost
    a whole run. A resumed run is not bit-identical to an uninterrupted one -- GPU nondeterminism already means a seed does not
    reproduce a run -- but it continues from the last evaluation instead of from nothing."""

    def test_the_random_state_round_trips(self):
        rng = np.random.default_rng(3)
        torch.manual_seed(3)
        snapshot = rng_snapshot(rng)
        first = (torch.rand(4), rng.random(4))
        rng_restore(snapshot, rng)
        second = (torch.rand(4), rng.random(4))
        assert torch.equal(first[0], second[0]) and np.array_equal(first[1], second[1])

    def test_a_run_state_round_trips_through_disk(self, tmp_path):
        path = tmp_path / "resume.pth"
        save_resume(path, stage=2, epoch=7, history=[{"val_ap": 0.5}])
        state = load_resume(path, resume=True)
        assert state["stage"] == 2 and state["epoch"] == 7 and state["history"] == [{"val_ap": 0.5}]

    def test_nothing_to_resume_from(self, tmp_path):
        assert load_resume(tmp_path / "absent.pth", resume=True) is None
        save_resume(tmp_path / "present.pth", stage=0)
        assert load_resume(tmp_path / "present.pth", resume=False) is None       # the flag decides, not the file

    def test_a_truncated_file_is_ignored_rather_than_crashing_the_run(self, tmp_path):
        path = tmp_path / "resume.pth"
        save_resume(path, stage=1)
        path.write_bytes(path.read_bytes()[: 32])                                # a crash during the save itself
        assert load_resume(path, resume=True) is None

    def test_the_stopper_keeps_its_patience_across_a_resume(self):
        stopper = EarlyStopping(patience_evals=2, min_epochs=0, max_epochs=99, max_hours=99)
        for metric in (0.5, 0.4):
            stopper.update(metric)
        restored = EarlyStopping(patience_evals=2, min_epochs=0, max_epochs=99, max_hours=99)
        restored.__dict__.update(stopper.state())
        assert restored.state() == stopper.state()
        assert not restored.should_stop(epoch=1, hours=0)
        restored.update(0.3)
        assert restored.should_stop(epoch=1, hours=0)                            # the second bad evaluation, not a fresh count

    def test_the_best_checkpoint_keeps_its_bar_across_a_resume(self, tmp_path):
        best = BestCheckpoint(tmp_path / "best.pth")
        best.offer(0.8, {"model": 1})
        restored = BestCheckpoint(tmp_path / "best.pth")
        restored.__dict__.update(best.state())
        assert not restored.offer(0.7, {"model": 2})                             # a worse epoch after the crash cannot overwrite it
        assert restored.offer(0.9, {"model": 3})


class TestNonFiniteSummary:
    """When a stage-3 run goes NaN, the epoch-level guard fires up to 151 iterations later and its traceback points at the guard
    itself. This names what was already non-finite at the iteration that broke (`TODO.md` A51, O1's two failures)."""

    def test_it_names_only_the_non_finite_entries_with_their_kind(self):
        named = {"good": torch.ones(3), "nan": torch.tensor([1.0, float("nan")]), "inf": torch.tensor([float("inf")])}
        out = non_finite_summary(named)
        assert "good" not in out
        assert "nan" in out and "inf" in out

    def test_everything_finite_gives_nothing(self):
        assert non_finite_summary({"a": torch.zeros(2), "b": torch.ones(2)}) == ""

    def test_it_survives_an_empty_or_non_float_tensor(self):
        assert non_finite_summary({"empty": torch.zeros(0), "ints": torch.tensor([1, 2])}) == ""


class TestTrainingFault:
    """A stage-3 run on this machine has twice had its device state go bad mid-run (`memory/gotchas.md`): the candidates' own
    validation AP, which is read from the cache and cannot depend on the model, fell from 72.85% to 3.40% while the loss went to
    exactly zero, and six more epochs ran before the process died. The guard stops at the first such evaluation."""

    def test_a_steady_run_reports_no_fault(self):
        assert training_fault(0.7285, 0.7290, 0.83) is None

    def test_candidate_ap_moving_means_the_cached_data_is_corrupt(self):
        fault = training_fault(0.7285, 0.0340, 0.0)
        assert fault is not None and "candidate" in fault.lower()

    def test_a_non_finite_or_dead_loss_is_a_fault(self):
        assert training_fault(0.7285, 0.7285, float("nan")) is not None
        assert training_fault(0.7285, 0.7285, 0.0) is not None

    def test_an_undefined_candidate_ap_is_not_read_as_corruption(self):
        # a validation window with nothing annotated gives NaN; that is a known case, not a device fault
        assert training_fault(float("nan"), float("nan"), 0.83) is None


class TestMemorySchedule:
    """Stage 3's learning rate (`TODO.md` A51 T2). Its cosine schedule spans `--max-epochs` per chunk length, so in the 9-37 epochs
    a stage actually runs it barely decays; the plateau schedule steps once per evaluation on val AP instead, as step 2 does."""

    def _optimizer(self):
        return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)

    def test_the_cosine_schedule_steps_every_iteration(self):
        schedule, per_iteration = memory_schedule(self._optimizer(), "cosine", max_epochs=10, iterations=5, factor=0.3, patience=2, min_lr=1e-5)
        assert per_iteration and isinstance(schedule, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_the_plateau_schedule_steps_on_the_evaluation_metric_and_cuts_the_rate(self):
        optimizer = self._optimizer()
        schedule, per_iteration = memory_schedule(optimizer, "plateau", max_epochs=10, iterations=5, factor=0.5, patience=1, min_lr=1e-3)
        assert not per_iteration
        for metric in (0.5, 0.4, 0.4, 0.4, 0.4):
            schedule.step(metric)
        assert optimizer.param_groups[0]["lr"] < 1.0


class TestFirstRecordings:

    def test_keeps_whole_leading_recordings_with_only_their_annotations(self):
        split = FrameSplit(scans=np.arange(10, dtype=np.float32).reshape(5, 2), poses=np.zeros((5, 3), np.float32), start=np.array([0, 0, 2, 2, 2]),
                           annotated=np.array([1, 3, 4]), people_polar=[np.ones((1, 2)), np.zeros((0, 2)), np.ones((2, 2))],
                           people_xy=[np.ones((1, 2)), np.zeros((0, 2)), np.ones((2, 2))], time=np.arange(5) * 0.04)
        first = first_recordings(split, 1)
        assert first.scans.shape == (2, 2) and first.start.tolist() == [0, 0] and first.time.tolist() == pytest.approx([0.0, 0.04])
        assert first.annotated.tolist() == [1] and len(first.people_polar) == 1 and len(first.people_xy) == 1


class TestEvaluate:
    """Whole recordings are played in time slices, so memory stays bounded by streams x slice length (the test split's recordings
    reach 6,796 frames; gathering 32 of them at once needed 1.66 GB for the candidate features alone)."""

    def _split(self):
        rng = np.random.default_rng(0)

        def recording(n):
            present = rng.random((n, 2)) < 0.7
            return SequenceCandidates(
                xy=rng.uniform(-3, 3, (n, 6, 2)).astype(np.float32), score=rng.random((n, 6)).astype(np.float32), valid=rng.random((n, 6)) < 0.8,
                target=np.zeros((n, 6), bool), pose=np.zeros((n, 3), np.float32), time=np.arange(n) * 0.04, annotated=np.ones(n, bool),
                gt_xy=rng.uniform(-3, 3, (n, 2, 2)).astype(np.float32), gt_valid=present, gt_covered=present.copy(),
                features=rng.standard_normal((n, 6, 8)).astype(np.float16))
        return pack([recording(23), recording(17)])

    def _model(self):
        from follow_the_drow.detectors.three_horizon import ObjectMemory, SlotRules
        torch.manual_seed(0)
        return ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, feature_dim=8).eval()

    def test_playing_whole_recordings_in_slices_changes_nothing(self):
        split, model = self._split(), self._model()
        segments = evaluation_segments(split, 0, 0, 0)
        whole = evaluate(model, split, torch.device("cpu"), 1, 2, segments)
        sliced = evaluate(model, split, torch.device("cpu"), 1, 2, segments, slice_frames=5)
        for part in ("candidates", "memory"):
            for key in ("ap", "fp_per_frame", "fn_per_frame", "frames", "people"):
                assert sliced[part][key] == pytest.approx(whole[part][key], nan_ok=True)

    def test_an_untrained_memory_reports_no_rescoring_shift_and_no_coasting(self):
        # the rescore gate starts closed and the coasting head silent, so stage 3 starts as exactly stage 2
        split, model = self._split(), self._model()
        split.data.target[:] = split.data.valid & (np.arange(6) < 3)
        result = evaluate(model, split, torch.device("cpu"), 1, 2, evaluation_segments(split, 0, 0, 0))
        memory = result["memory"]
        assert memory["rescore_shift_person"] == pytest.approx(0.0, abs=1e-4)
        assert memory["rescore_shift_other"] == pytest.approx(0.0, abs=1e-4)
        assert memory["coast_per_frame"] == 0.0
        # identical scores, so the own operating point spends the same false-positive budget and cannot recall less
        assert memory["own_fp_per_frame"] <= result["candidates"]["fp_per_frame"] + 1e-9
        assert memory["own_recall"] >= memory["recall"] - 1e-9

    def test_extra_match_radii_are_scored_for_both_stages(self):
        split, model = self._split(), self._model()
        result = evaluate(model, split, torch.device("cpu"), 1, 2, evaluation_segments(split, 0, 0, 0), extra_radii=(0.3,))
        for part in ("candidates", "memory"):
            assert 0.0 <= result[part]["ap_0.3m"] <= 1.0


class TestMemoryMsPerFrame:

    @pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
    def test_it_times_single_stream_steps_for_either_form_of_device(self, device):
        # `detect_device` returns a string, not a `torch.device`, and timing it crashed the first A51 smoke run
        ms = memory_ms_per_frame(TestEvaluate()._model(), TestEvaluate()._split(), device, frames=10)
        assert 0.0 < ms < 10_000.0


class TestReplaySort:

    def test_a_new_track_is_only_reported_after_consecutive_matches_once_the_grace_period_is_over(self):
        n = 10
        present = np.arange(n) >= 4                      # nobody for 4 frames (past the grace period), then a person straight ahead
        seq = SequenceCandidates(
            xy=np.tile(np.array([[[0.0, 2.0]]], np.float32), (n, 1, 1)), score=np.full((n, 1), 0.9, np.float32), valid=present[:, None].copy(),
            target=np.zeros((n, 1), bool), pose=np.zeros((n, 3), np.float32), time=np.arange(n) * 0.04, annotated=np.ones(n, bool),
            gt_xy=np.tile(np.array([[[0.0, 2.0]]], np.float32), (n, 1, 1)), gt_valid=present[:, None].copy(), gt_covered=present[:, None].copy(),
            features=np.zeros((n, 1, 0), np.float16))
        reported = [len(scores) for scores, _, _ in replay_sort(pack([seq]), min_hits=3)]
        assert reported[:5] == [0, 0, 0, 0, 0]           # its first frame is not confirmed
        assert reported[-1] == 1
