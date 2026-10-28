"""Training harness of the three-horizon detector, step 1: stage 3 on saved candidates (`docs/PROPOSAL.md` §8, §10; `TODO.md` A43).

Targets, the candidate cache, the stream sampler for truncated backpropagation, early stopping and the exchange rate.
"""
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from takheliped.detectors.takheliped import MAX_RANGE_M, CalibrationNetwork, ObjectMemory, SlotRules
from takheliped.utils.drow_utils import _win2global, swap_vote_frame
from train import make_targets
from train_takheliped import (
    BestCheckpoint, EarlyStopping, PeriodicCheckpoint, StreamSampler, build_candidates, calibration_network, candidate_cache_path, decode_candidates, evaluate_calibration, candidate_pooling_weights, candidate_targets, clip_state_grad, coast_targets, decode_votes, evaluate, evaluation_segments,
    FrameSplit, SequenceCandidates, exchange_rate, first_recordings, frames_from_dataset, joint_frame_loss, joint_step, network_candidates, oracle_prior, evaluate_joint, PRIOR_MODES, sample_drift, chunk_elapsed,
    DT_AVERAGE_FRAMES, MIN_DT_S, TOP_P, GRAD_SKIP_LIMIT, build_parser, matched_person_xy, coasting_hits, exponential_schedule, network_shape, select_test_weights, mirror_people, mirror_scans, gradient_verdict, load_resume, periodic_reset, perturb_poses, memory_schedule, non_finite_summary, rng_restore, rng_snapshot, save_resume, training_fault, object_memory, frame_periods, gather, mean_frame_period, memory_ms_per_frame, pack, plateau_schedule, recording_bounds, replay_sort, score_detections,
    union_decode,
)


def _small_calibration():
    """A calibration network small enough for a unit test, with no prior channels -- the shape E12a's control arm trains."""
    torch.manual_seed(0)
    angles = torch.from_numpy(np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32))
    return CalibrationNetwork(angles, channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1)


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
        from takheliped.detectors.takheliped import CalibrationNetwork
        torch.manual_seed(0)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = CalibrationNetwork(torch.from_numpy(angles), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1).eval()
        scans = (np.random.default_rng(0).random((6, 48)) * 4 + 1).astype(np.float32)
        per_frame = network_candidates(net, scans, "cpu")
        assert len(per_frame) == 6
        for cands, feats in per_frame:
            assert cands.shape[1] == 3 and feats.shape == (len(cands), 8)

    def test_decoding_anchors_votes_on_the_ranges_the_network_read(self):
        """Votes are offsets from a beam's endpoint, so decoding must use the sanitised range the network saw: a missing return and
        a reading at the range limit are the same scan to the whole detector, not only to its input layer."""
        from takheliped.detectors.takheliped import CalibrationNetwork
        torch.manual_seed(0)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = CalibrationNetwork(torch.from_numpy(angles), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1).eval()
        at_limit = (np.random.default_rng(0).random((4, 48)) * 4 + 1).astype(np.float32)
        at_limit[:, 10:14] = net.max_range_m
        missing = at_limit.copy()
        missing[:, 10:12] = np.inf
        missing[:, 12:14] = 61.0
        for (want, _), (got, _) in zip(network_candidates(net, at_limit, "cpu"), network_candidates(net, missing, "cpu"), strict=True):
            assert np.allclose(want, got)

    def test_evaluation_scores_a_missing_return_like_a_reading_at_the_limit(self):
        from takheliped.detectors.takheliped import CalibrationNetwork
        torch.manual_seed(0)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = CalibrationNetwork(torch.from_numpy(angles), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1)
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

        want = evaluate_calibration(net, split(at_limit), np.arange(3), "cpu", 3)
        got = evaluate_calibration(net, split(missing), np.arange(3), "cpu", 3)
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
        from takheliped.detectors.takheliped import CalibrationNetwork
        torch.manual_seed(0)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = CalibrationNetwork(torch.from_numpy(angles), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1)
        with torch.no_grad():
            net.head.weight.zero_()
            net.head.bias.copy_(torch.tensor([3.0, 0.0, 0.0]))
        phi = float(angles[24])
        scans = np.stack([np.full(48, 3.0), np.full(48, np.inf)]).astype(np.float32)      # a wall, then nothing within range
        people = [np.array([[3.0, phi]]), np.array([[net.max_range_m, phi]])]
        xy = [np.array([[-r * np.sin(phi), r * np.cos(phi)]]) for r in (3.0, net.max_range_m)]
        split = FrameSplit(scans, np.zeros((2, 3), np.float32), np.zeros(2, np.int64), np.arange(2), people, xy, np.arange(2) / 40.0)
        assert evaluate_calibration(net, split, np.arange(2), "cpu", 2)["recall"] == pytest.approx(0.5)     # only the wall's person is found


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
        stop = EarlyStopping(patience_evals=2, min_epochs=0, max_epochs=100)
        assert [stop.update(m) for m in [0.5, 0.6, 0.55, 0.58]] == [True, True, False, False]
        assert stop.should_stop(epoch=4)
        assert stop.best == pytest.approx(0.6)

    def test_never_stops_before_the_minimum_epochs(self):
        stop = EarlyStopping(patience_evals=1, min_epochs=5, max_epochs=100)
        for m in [0.5, 0.4, 0.3]:
            stop.update(m)
        assert not stop.should_stop(epoch=3)
        assert stop.should_stop(epoch=5)

    def test_stops_at_the_epoch_cap(self):
        stop = EarlyStopping(patience_evals=10, min_epochs=0, max_epochs=3)
        stop.update(0.5)
        assert not stop.should_stop(epoch=2)
        assert stop.should_stop(epoch=3)

    def test_no_amount_of_elapsed_time_stops_a_run_that_is_still_improving(self):
        """A51's first full-curriculum run was ended by a 6 h cap inside its first stage; budgets are epochs and patience now."""
        stop = EarlyStopping(patience_evals=10, min_epochs=0, max_epochs=100)
        for epoch, metric in enumerate([0.5, 0.6, 0.7], start=1):
            stop.update(metric)
            assert not stop.should_stop(epoch=epoch)


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


class TestCalibrationNetworkFor:

    def test_the_range_limit_comes_from_the_checkpoint_args(self):
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = CalibrationNetwork(torch.from_numpy(angles), max_range_m=15.0)
        rebuilt = calibration_network(dict(model=net.state_dict(), args={"max_range_m": "15.0"}), angles)
        assert rebuilt.max_range_m == 15.0

    def test_a_checkpoint_from_before_the_flags_uses_the_defaults(self):
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = CalibrationNetwork(torch.from_numpy(angles))
        rebuilt = calibration_network(dict(model=net.state_dict(), args={}), angles)
        assert rebuilt.max_range_m == MAX_RANGE_M


class TestAttributionArmShapes:
    """The attribution arms F1 and W (`TODO.md` A59) change the network's input and widths. Whatever `step2` trained must be rebuilt
    from the checkpoint's args (stored as strings) by every script that loads it -- the averaging, the probe, stage 3's caches."""

    ANGLES = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)

    @staticmethod
    def _stored(argv):
        return {k: str(v) for k, v in vars(build_parser().parse_args(["step2", *argv])).items()}

    def test_step2_defaults_describe_today_s_network(self):
        """Exact equality on purpose: a new key here changes what every loader rebuilds, so adding one is a decision."""
        assert network_shape(self._stored([])) == dict(max_range_m=MAX_RANGE_M, input_features="beam", channels=(32, 64, 128),
                                                       gather_width_m=0.0, gather_samples=9, vote_frame="beam", block="convnext", aggregator=False,
                                                       kernel_size=7, expansion=4, blocks=(2, 2, 2), bottleneck_blocks=2)

    def test_the_arms_flags_describe_their_network(self):
        shape = network_shape(self._stored(["--input-features", "lfe", "--channels", "8", "16", "32"]))
        assert shape == dict(max_range_m=MAX_RANGE_M, input_features="lfe", channels=(8, 16, 32),
                             gather_width_m=0.0, gather_samples=9, vote_frame="beam", block="convnext", aggregator=False,
                                                       kernel_size=7, expansion=4, blocks=(2, 2, 2), bottleneck_blocks=2)

    def test_the_gather_flag_describes_its_network(self):
        """Item 15: the metric window is part of the shape, so a checkpoint trained with it rebuilds with it."""
        shape = network_shape(self._stored(["--gather-width-m", "1.66", "--gather-samples", "11"]))

        assert (shape["gather_width_m"], shape["gather_samples"]) == (1.66, 11)

    def test_item_45_s_shape_knobs_describe_their_network(self):
        """Four numbers chosen once and never varied; each has to reach the network and come back from a checkpoint."""
        shape = network_shape(self._stored(["--kernel-size", "5", "--expansion", "2",
                                            "--blocks", "1", "2", "3", "--bottleneck-blocks", "4"]))

        assert (shape["kernel_size"], shape["expansion"]) == (5, 2)
        assert (shape["blocks"], shape["bottleneck_blocks"]) == ((1, 2, 3), 4)

    def test_a_checkpoint_written_before_item_45_s_knobs_keeps_the_adopted_shape(self):
        shape = network_shape({"max_range_m": "10.0"})

        assert (shape["kernel_size"], shape["expansion"], shape["blocks"], shape["bottleneck_blocks"]) == (7, 4, (2, 2, 2), 2)

    def test_the_aggregator_describes_its_network(self):
        """Item 40: ablating it changes which weights exist, so it is part of the shape a checkpoint rebuilds from."""
        assert network_shape(self._stored(["--no-aggregator"]))["aggregator"] is False
        assert network_shape(self._stored([]))["aggregator"] is False       # item 52: the CLI default is the adopted arch
        assert network_shape(self._stored(["--aggregator"]))["aggregator"] is True
        # **The line that matters most here.** A checkpoint written before the flag existed must still rebuild *with*
        # the aggregator, from the CONSTRUCTOR default -- which is why item 52 moved the CLI default and left that one
        # alone. If this ever flips, every stage-2 checkpoint older than 2026-10-25 silently refuses its own weights.
        assert network_shape({"max_range_m": "10.0"})["aggregator"] is True

    def test_the_vote_frame_and_the_block_describe_their_network(self):
        """Items 18 and 19: both change what the weights mean, so both are read back rather than defaulted."""
        shape = network_shape(self._stored(["--vote-frame", "sensor", "--block", "residual"]))

        assert (shape["vote_frame"], shape["block"]) == ("sensor", "residual")

    def test_a_checkpoint_written_before_either_flag_existed_reads_as_the_adopted_network(self):
        """Every checkpoint trained so far says nothing about them, and must keep rebuilding exactly as it did."""
        shape = network_shape({"max_range_m": "10.0", "input_features": "beam"})

        assert (shape["vote_frame"], shape["block"]) == ("beam", "convnext")

    def test_a_sensor_frame_checkpoint_rebuilds_knowing_which_frame_it_votes_in(self):
        """The decoders read `net.vote_frame`, so losing it would silently rotate every detection."""
        torch.manual_seed(0)
        net = CalibrationNetwork(torch.from_numpy(self.ANGLES), channels=(8, 16, 32), vote_frame="sensor", block="residual",
                                 aggregator=False)      # item 52: what the stored args below now describe
        stored = self._stored(["--vote-frame", "sensor", "--block", "residual", "--channels", "8", "16", "32"])

        rebuilt = calibration_network(dict(model=net.state_dict(), args=stored), self.ANGLES)

        assert (rebuilt.vote_frame, rebuilt.block) == ("sensor", "residual")

    def test_a_checkpoint_rebuilds_into_the_network_it_was_trained_as(self):
        torch.manual_seed(0)
        net = CalibrationNetwork(torch.from_numpy(self.ANGLES), channels=(8, 16, 32), input_features="lfe",
                                 aggregator=False).eval()      # item 52: what the stored args below now describe
        stored = self._stored(["--input-features", "lfe", "--channels", "8", "16", "32"])
        rebuilt = calibration_network(dict(model=net.state_dict(), args=stored), self.ANGLES).eval()
        scans = torch.rand(2, 48) * 8
        with torch.no_grad():
            for a, b in zip(net(scans), rebuilt(scans), strict=True):
                assert torch.equal(a, b)


class TestExponentialSchedule:
    """DR-SPAAM's learning rate, for the attribution arm R (`TODO.md` A59): geometric decay from the initial rate to a floor over
    the run, stepped every iteration, then held."""

    @staticmethod
    def _optimizer():
        return torch.optim.AdamW(torch.nn.Linear(1, 1).parameters(), lr=1e-3)

    def test_the_rate_decays_geometrically_to_the_floor_and_stays_there(self):
        optimizer = self._optimizer()
        schedule = exponential_schedule(optimizer, min_lr=1e-6, iterations=10)
        rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(12):
            optimizer.step()
            schedule.step()
            rates.append(optimizer.param_groups[0]["lr"])
        assert rates[0] == pytest.approx(1e-3)
        assert rates[5] == pytest.approx(math.sqrt(1e-3 * 1e-6))
        assert rates[10] == pytest.approx(1e-6) and rates[12] == pytest.approx(1e-6)

    def test_a_resumed_schedule_continues_where_it_stopped(self):
        optimizer = self._optimizer()
        schedule = exponential_schedule(optimizer, min_lr=1e-6, iterations=10)
        for _ in range(4):
            optimizer.step()
            schedule.step()
        saved_optimizer, saved_schedule = optimizer.state_dict(), schedule.state_dict()
        optimizer.step()
        schedule.step()
        resumed = self._optimizer()
        again = exponential_schedule(resumed, min_lr=1e-6, iterations=10)
        resumed.load_state_dict(saved_optimizer)
        again.load_state_dict(saved_schedule)
        resumed.step()
        again.step()
        assert resumed.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])


class TestSelectTestWeights:
    """`step2 --select last` is DR-SPAAM's protocol (`TODO.md` A59 R): the test scores the weights training ended on, not the
    best on val, and both are kept so the probe can score each."""

    @staticmethod
    def _best(tmp_path):
        other = _small_calibration()
        with torch.no_grad():
            for p in other.parameters():
                p.add_(1.0)
        best = BestCheckpoint(tmp_path / "step2_calibration.best.pth")
        best.offer(0.5, dict(model=other.state_dict()))
        return best, other

    def test_last_keeps_the_weights_training_ended_on_and_saves_them_beside_the_best(self, tmp_path):
        net = _small_calibration()
        before = {k: v.clone() for k, v in net.state_dict().items()}
        best, _ = self._best(tmp_path)
        select_test_weights(net, best, "last", dict(epoch=5.0, args={"select": "last"}), "cpu")
        assert all(torch.equal(v, before[k]) for k, v in net.state_dict().items())
        saved = torch.load(tmp_path / "step2_calibration.last.pth", weights_only=False)
        assert saved["epoch"] == 5.0 and saved["args"] == {"select": "last"}
        assert all(torch.equal(v, before[k]) for k, v in saved["model"].items())

    def test_best_loads_the_run_s_best_on_val(self, tmp_path):
        net = _small_calibration()
        best, other = self._best(tmp_path)
        select_test_weights(net, best, "best", dict(epoch=5.0, args={}), "cpu")
        assert all(torch.equal(v, other.state_dict()[k]) for k, v in net.state_dict().items())
        assert not (tmp_path / "step2_calibration.last.pth").exists()


class TestCandidateCachePath:
    """Stage-3 candidate caches are keyed by the stage-2 checkpoint they come from. Every run writes the same file names
    (`step2_calibration.best.pth`, `.averaged.pth`), so the run's directory must be in the key: two runs whose checkpoints were
    written in the same second would otherwise share one cache (`TODO.md` A60)."""

    def test_two_runs_with_identically_named_checkpoints_get_different_caches(self, tmp_path):
        a, b = tmp_path / "run_a" / "step2_calibration.averaged.pth", tmp_path / "run_b" / "step2_calibration.averaged.pth"
        for p in (a, b):
            p.parent.mkdir()
            p.write_bytes(b"")
            os.utime(p, (1_000_000_000, 1_000_000_000))
        assert candidate_cache_path("test", 0, a) != candidate_cache_path("test", 0, b)

    def test_the_official_cache_name_is_unchanged_so_a_1_gb_cache_is_not_rebuilt(self):
        assert candidate_cache_path("val", 0, None).name.startswith("lfe-peaks_official_val_")

    def test_another_frog_mode_gets_its_own_cache(self):
        """`transferred`'s test split is the person-free complement of `official`'s, and shares its name (`TODO.md` A50)."""
        assert candidate_cache_path("test", 0, None) != candidate_cache_path("test", 0, None, mode="transferred")

    def test_lfe_candidates_need_no_checkpoint(self):
        assert "lfe-peaks" in candidate_cache_path("val", 0, None).name


class TestStage2Recipe:
    """Stage 2's recipe, decided by the owner 2026-10-06 (`TODO.md` A59): nearest-person labels, mirror augmentation and dropout 0.1 by default,
    each still switchable off for an ablation, and candidates decoded by `nms-mean` -- a detection scored by its best beam and placed
    at the probability-weighted mean of the votes it suppresses."""

    ANGLES = np.linspace(-np.pi / 2, np.pi / 2, 720, endpoint=False)

    def test_step2_defaults_to_the_adopted_recipe(self):
        args = build_parser().parse_args(["step2"])
        assert args.label_assign == "nearest" and args.mirror is True and args.dropout == 0.1

    def test_the_attribution_arms_are_off_by_default(self):
        args = build_parser().parse_args(["step2"])
        assert (args.input_features, args.channels, args.select, args.schedule) == ("beam", None, "best", "plateau")

    def test_r_is_expressible_as_flags(self):
        args = build_parser().parse_args(["step2", "--batch", "8", "--max-epochs", "5", "--min-epochs", "5", "--schedule", "exponential",
                                          "--lr", "1e-3", "--min-lr", "1e-6", "--weight-decay", "0", "--select", "last"])
        assert (args.batch, args.schedule, args.min_lr, args.weight_decay, args.select) == (8, "exponential", 1e-6, 0.0, "last")

    def test_each_part_can_still_be_switched_off_for_an_ablation(self):
        args = build_parser().parse_args(["step2", "--label-assign", "last", "--no-mirror"])
        assert args.label_assign == "last" and args.mirror is False

    def test_a_candidate_is_scored_by_its_best_beam_and_placed_at_its_votes_weighted_mean(self):
        scan, prob, votes = np.full(720, 8.0), np.zeros(720), np.zeros((720, 2))
        beams = np.arange(355, 365)
        prob[beams] = np.linspace(0.1, 0.9, len(beams))
        target = np.array([0.5, 4.0])
        r_gt, phi_gt = np.hypot(*target), np.arctan2(-target[0], target[1])
        dphi = phi_gt - self.ANGLES[beams]
        votes[beams, 0], votes[beams, 1] = r_gt * np.sin(dphi), r_gt * np.cos(dphi) - scan[beams]
        candidates, _ = decode_candidates(scan, self.ANGLES, prob, votes, TOP_P, MAX_RANGE_M)
        assert candidates[0, 0] == pytest.approx(0.9)            # the vote grid would score the voters' mean, ~0.5
        assert np.allclose(candidates[0, 1:], target, atol=1e-6)
        assert len(candidates) == 1


class TestMirror:
    """`step2 --mirror` (`TODO.md` A59 arm M): a scan reversed beam for beam, with its people's bearings mirrored to match, is a
    valid new training sample only if the targets built from it are exactly the mirror of the original targets."""

    ANGLES = np.linspace(-np.pi / 2, np.pi / 2, 720, endpoint=False)      # FROG's grid: not symmetric about 0, by one beam

    def test_mirrored_targets_are_the_mirror_of_the_targets(self):
        from train import make_targets
        rng = np.random.default_rng(0)
        scan = rng.uniform(1.0, 8.0, 720)
        people = np.array([[3.0, 0.4], [5.0, -0.9], [2.0, 1.2]])
        labels, votes = make_targets(scan, self.ANGLES, {3: [tuple(p) for p in people]}, nearest=True)
        m_labels, m_votes = make_targets(np.ascontiguousarray(scan[::-1]), self.ANGLES, {3: [tuple(p) for p in mirror_people(people, self.ANGLES)]}, nearest=True)
        assert (m_labels == labels[::-1]).all()
        assert np.allclose(m_votes[:, 0], -votes[::-1, 0], atol=1e-5) and np.allclose(m_votes[:, 1], votes[::-1, 1], atol=1e-5)

    def test_a_mirrored_bearing_lands_on_the_mirrored_beam(self):
        people = np.array([[4.0, self.ANGLES[100]]])
        assert mirror_people(people, self.ANGLES)[0, 1] == pytest.approx(self.ANGLES[719 - 100])

    def test_only_the_flagged_scans_are_mirrored(self):
        ranges = torch.arange(2 * 4, dtype=torch.float32).view(2, 4)
        mirrored = mirror_scans(ranges, np.array([False, True]))
        assert torch.equal(mirrored[0], ranges[0]) and torch.equal(mirrored[1], ranges[1].flip(-1))

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
    """Validation windows. A window whose scored span holds no annotated frame costs its warm-up and returns nothing: 6 of 64
    did on the val cache of 2026-10-09, which is why placement reads the annotations (`TODO.md` A50 phase 1 item 3)."""

    def _split(self, lengths, annotated=None):
        lengths = np.asarray(lengths)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        flags = np.ones(int(lengths.sum()), dtype=bool) if annotated is None else np.asarray(annotated, dtype=bool)
        return SimpleNamespace(lengths=lengths, offsets=offsets, data=SimpleNamespace(annotated=flags))

    def test_whole_recordings_when_no_windows_are_asked_for(self):
        split = self._split([100, 40])
        assert evaluation_segments(split, windows=0, window_frames=10, warmup_frames=5) == [(0, 0, 100, 0), (1, 0, 40, 0)]

    def test_windows_with_their_warm_up_fit_inside_long_enough_recordings(self):
        split = self._split([1000, 30, 500])
        segments = evaluation_segments(split, windows=6, window_frames=50, warmup_frames=20)
        assert 5 <= len(segments) <= 7
        for recording, first, frames, warmup in segments:
            assert recording != 1 and frames == 70 and warmup == 20
            assert 0 <= first and first + frames <= split.lengths[recording]

    def test_the_spacing_is_unchanged_where_every_frame_is_annotated(self):
        segments = evaluation_segments(self._split([1000]), windows=3, window_frames=50, warmup_frames=20)
        assert [first for _, first, _, _ in segments] == [0, 465, 930]

    def test_every_window_scores_at_least_one_annotated_frame(self):
        annotated = np.zeros(1000, dtype=bool)
        annotated[100:260] = True                                                # one annotated stretch, early in the recording
        segments = evaluation_segments(self._split([1000], annotated), windows=6, window_frames=50, warmup_frames=20)
        assert segments
        for _, first, frames, warmup in segments:
            assert annotated[first + warmup:first + frames].any()

    def test_the_window_count_is_kept_when_the_annotations_allow_it(self):
        annotated = np.zeros(1000, dtype=bool)
        annotated[:600] = True
        segments = evaluation_segments(self._split([1000], annotated), windows=6, window_frames=50, warmup_frames=20)
        assert len(segments) == 6

    def test_a_recording_with_no_annotations_contributes_no_windows(self):
        annotated = np.concatenate([np.ones(1000, dtype=bool), np.zeros(1000, dtype=bool)])
        segments = evaluation_segments(self._split([1000, 1000], annotated), windows=6, window_frames=50, warmup_frames=20)
        assert {recording for recording, _, _, _ in segments} == {0}

    def test_windows_are_placed_by_annotation_even_when_nothing_is_annotated_anywhere(self):
        """A split with no annotations at all still yields windows, so a smoke run does not silently score zero frames."""
        segments = evaluation_segments(self._split([1000], np.zeros(1000, dtype=bool)), windows=3, window_frames=50, warmup_frames=20)
        assert len(segments) == 3


class TestCandidatePoolingWeights:

    def test_features_are_the_probability_weighted_mean_of_the_beams_voting_near_the_candidate(self):
        det_xy = np.array([[0.0, 2.0], [5.0, 5.0]])
        vote_xy = np.array([[0.0, 2.1], [0.1, 2.0], [0.0, 3.0], [5.0, 5.0]])
        prob = np.array([0.9, 0.3, 0.9, 0.0])
        features = np.array([[1.0, 0.0], [0.0, 1.0], [7.0, 7.0], [9.0, 9.0]])
        pooled = candidate_pooling_weights(det_xy, vote_xy, prob, radius=0.5) @ features
        assert pooled[0].tolist() == pytest.approx([0.75, 0.25])     # beams 0 and 1 (weights 0.9 and 0.3); beam 2 is 1 m away
        assert pooled[1].tolist() == [0.0, 0.0]                      # its only nearby beam has zero probability


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


class TestNearestLabelAssignment:
    """`make_targets(nearest=True)` gives each beam the label and vote of the nearest person within the radius, as the official
    DR-SPAAM code's `_closest_detection` does; the default lets the last-listed person within the radius win, which on FROG train
    sends 3.3% of positive beams to someone other than their nearest (`TODO.md` A59)."""

    ANGLES = np.linspace(-np.pi / 2, np.pi / 2, 720, endpoint=False)
    PEOPLE_XY = ((0.25, 3.0), (-0.25, 3.0))          # shoulder to shoulder, 0.5 m apart

    def _scan(self):
        """A wall at 3.2 m behind both people, every beam a surface the 0.6 m radius can reach from one of them or both."""
        return np.full(720, 3.2)

    def _polar(self):
        return [(math.hypot(x, y), math.atan2(-x, y)) for x, y in self.PEOPLE_XY]

    def _voted(self, scan, votes):
        r, phi = _win2global(scan, self.ANGLES, votes[:, 0], votes[:, 1])
        return np.stack([r * -np.sin(phi), r * np.cos(phi)], axis=1)

    def test_every_positive_beam_votes_for_its_nearest_person(self):
        from train import make_targets
        scan = self._scan()
        labels, votes = make_targets(scan, self.ANGLES, {3: self._polar()}, vote_collect_radius=0.6, nearest=True)
        endpoints = np.stack([scan * -np.sin(self.ANGLES), scan * np.cos(self.ANGLES)], axis=1)
        nearest = np.argmin(np.linalg.norm(endpoints[:, None] - np.array(self.PEOPLE_XY)[None], axis=-1), axis=1)
        positive = labels == 3
        assert positive.sum() > 0
        assert np.allclose(self._voted(scan, votes)[positive], np.array(self.PEOPLE_XY)[nearest[positive]], atol=1e-6)

    def test_the_default_still_lets_the_last_listed_person_win(self):
        from train import make_targets
        scan = self._scan()
        labels, votes = make_targets(scan, self.ANGLES, {3: self._polar()}, vote_collect_radius=0.6)
        endpoints = np.stack([scan * -np.sin(self.ANGLES), scan * np.cos(self.ANGLES)], axis=1)
        in_both = (np.linalg.norm(endpoints[:, None] - np.array(self.PEOPLE_XY)[None], axis=-1) < 0.6).all(axis=1)
        assert in_both.sum() > 0
        assert np.allclose(self._voted(scan, votes)[in_both], self.PEOPLE_XY[1], atol=1e-6)

    def test_the_radius_still_bounds_who_is_positive(self):
        from train import make_targets
        scan = self._scan()
        wide, _ = make_targets(scan, self.ANGLES, {3: self._polar()}, vote_collect_radius=0.6, nearest=True)
        narrow, _ = make_targets(scan, self.ANGLES, {3: self._polar()}, vote_collect_radius=0.4, nearest=True)
        assert (narrow == 3).sum() < (wide == 3).sum()
        assert ((narrow == 3) <= (wide == 3)).all()

    def test_one_person_alone_gets_the_same_targets_either_way(self):
        from train import make_targets
        scan = self._scan()
        last = make_targets(scan, self.ANGLES, {3: self._polar()[:1]})
        nearest = make_targets(scan, self.ANGLES, {3: self._polar()[:1]}, nearest=True)
        assert (last[0] == nearest[0]).all() and np.allclose(last[1], nearest[1])


class TestJointStep:
    """Stage 2 and stage 3 run together on one frame, with the memory drawn back into stage 2 (`docs/PROPOSAL.md` §5.5, phase C)."""

    def _parts(self):
        from takheliped.detectors.takheliped import CalibrationNetwork, ObjectMemory, SlotRules
        torch.manual_seed(0)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = CalibrationNetwork(torch.from_numpy(angles), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1, prior_channels=2).eval()
        memory = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, feature_dim=8).eval()
        scans = torch.from_numpy((np.random.default_rng(0).random((1, 6, 48)) * 4 + 1).astype(np.float32))
        return net, memory, scans, torch.zeros(1, 6, 3), torch.full((1,), 0.04)

    def test_with_feedback_and_rescoring_still_silent_it_is_stage_2_then_stage_3(self):
        net, memory, scans, poses, dt = self._parts()
        state, slots = None, memory.initial_state(1)
        with torch.no_grad():
            for t in range(6):
                joint, state = joint_step(net, memory, scans[:, t], poses[:, t], dt, state)
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


class TestOraclePrior:
    """A51 E12b: the prior a perfect memory would draw, to put a ceiling on what any feedback could buy."""

    def _angles(self):
        return torch.from_numpy(np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32))

    def test_every_person_marks_its_own_sector_at_full_value(self):
        # 6 sectors of 30 degrees from -90; straight ahead is sector 3, hard left is sector 5
        prior = oracle_prior([np.array([[0.0, 2.0], [-2.0, 0.1]], np.float32)], self._angles(), 6, torch.device("cpu"))
        assert prior.shape == (1, 2, 6)
        assert prior[0, 0].tolist() == pytest.approx([0, 0, 0, 1, 0, 1])
        assert torch.equal(prior[0, 0], prior[0, 1])       # a perfect memory is certain of both value and person

    def test_an_unannotated_frame_draws_nothing_rather_than_something_stale(self):
        # the decision recorded in TODO A51: absent beats stale, so the oracle is never wrong, only silent
        prior = oracle_prior([np.zeros((0, 2), np.float32)], self._angles(), 6, torch.device("cpu"))
        assert prior.shape == (1, 2, 6) and prior.abs().sum() == 0

    def test_streams_are_padded_independently(self):
        people = [np.array([[0.0, 2.0]], np.float32), np.zeros((0, 2), np.float32), np.array([[-2.0, 0.1], [0.0, 2.0]], np.float32)]
        prior = oracle_prior(people, self._angles(), 6, torch.device("cpu"))
        assert prior.shape == (3, 2, 6)
        assert prior[0, 0].tolist() == pytest.approx([0, 0, 0, 1, 0, 0])
        assert prior[1, 0].abs().sum() == 0
        assert prior[2, 0].tolist() == pytest.approx([0, 0, 0, 1, 0, 1])

    def test_a_person_outside_the_field_of_view_draws_nothing(self):
        prior = oracle_prior([np.array([[0.0, -2.0]], np.float32)], self._angles(), 6, torch.device("cpu"))
        assert prior.abs().sum() == 0


class TestJointStepPriorOverride:
    """A51 E12b: an explicit prior replaces the one the slots would have drawn, which is how the oracle is fed in."""

    def test_the_given_prior_is_the_one_that_reaches_stage_2(self):
        net, memory, scans, poses, dt = TestJointStep()._parts()
        given = torch.full((1, 2, 6), 0.5)
        with torch.no_grad():
            joint, _ = joint_step(net, memory, scans[:, 0], poses[:, 0], dt, None, prior=given)
        assert torch.equal(joint.prior, given)

    def test_it_overrides_a_memory_that_would_have_drawn_its_own(self):
        net, memory, scans, poses, dt = TestJointStep()._parts()
        with torch.no_grad():
            _, state = joint_step(net, memory, scans[:, 0], poses[:, 0], dt, None)
            drawn, _ = joint_step(net, memory, scans[:, 1], poses[:, 1], dt, state)
            given = torch.zeros(1, 2, 6)
            forced, _ = joint_step(net, memory, scans[:, 1], poses[:, 1], dt, state, prior=given)
        assert drawn.prior.abs().sum() > 0                  # the slots had something to say
        assert forced.prior.abs().sum() == 0                # and the override silenced it

    def test_a_different_prior_changes_stage_2_once_the_channels_are_trained(self):
        net, memory, scans, poses, dt = TestJointStep()._parts()
        with torch.no_grad():
            net.aggregate.weight[:, -2:].normal_()          # the prior starts silent, so give it a voice first
            quiet, _ = joint_step(net, memory, scans[:, 0], poses[:, 0], dt, None, prior=torch.zeros(1, 2, 6))
            loud, _ = joint_step(net, memory, scans[:, 0], poses[:, 0], dt, None, prior=torch.ones(1, 2, 6))
        assert not torch.allclose(quiet.logit, loud.logit, atol=1e-5)


class TestPriorModes:
    """A51 E12: which prior stage 2 is fed back is a three-way choice, and a typo must not silently pick the default."""

    def test_an_unknown_mode_is_refused_before_any_work(self):
        with pytest.raises(ValueError, match="unknown prior_mode"):
            evaluate_joint(None, None, None, [], 1, torch.device("cpu"), 1, "oracel")

    def test_the_three_known_modes_are_spelled_the_way_the_flag_sets_them(self):
        assert PRIOR_MODES == ("slots", "oracle", "zero")


class TestCalibrationNetworkWithoutPrior:
    """A51 E12a's control arm: the same run with no feedback channels at all."""

    def test_a_network_built_without_prior_channels_runs_and_ignores_one(self):
        checkpoint = dict(model=_small_calibration().state_dict(), args={})
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        net = calibration_network(checkpoint, angles, prior_channels=0, channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1)
        assert net.prior_channels == 0
        assert net.aggregate.weight.shape[1] == 2 * 32
        ranges = torch.from_numpy((np.random.default_rng(0).random((1, 48)) * 4 + 1).astype(np.float32))
        with torch.no_grad():
            logit, _, _ = net(ranges, torch.ones(1, 2, 6))
            plain, _, _ = net(ranges)
        assert logit.shape == (1, 48) and torch.isfinite(logit).all() and torch.equal(logit, plain)


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
            gt_valid=np.zeros((n, 1), bool), gt_covered=np.zeros((n, 1), bool), features=np.zeros((n, 1, 0), np.float16),
            free_range=np.full((n, 8), MAX_RANGE_M, np.float16))
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


class TestPeriodicReset:
    """E1, the memory-horizon curve (`TODO.md` A51 phase 5): emptying the memory every `interval_s` at evaluation and scoring the
    result says how long a memory the model actually uses. The project measured the convolution's horizon and never the slots'."""

    def test_no_interval_never_resets(self):
        due, since = periodic_reset(np.zeros(2), np.full(2, 0.5), None)
        assert not due.any() and since.tolist() == [0.5, 0.5]

    def test_a_stream_resets_once_its_interval_has_elapsed_and_starts_counting_again(self):
        since = np.zeros(1)
        seen = []
        for _ in range(6):
            due, since = periodic_reset(since, np.full(1, 0.4), 1.0)
            seen.append(bool(due[0]))
        assert seen == [False, False, True, False, False, True]      # 0.4, 0.8, 1.2 -> reset, 0.4, 0.8, 1.2 -> reset

    def test_streams_are_independent(self):
        due, since = periodic_reset(np.array([0.9, 0.1]), np.full(2, 0.2), 1.0)
        assert due.tolist() == [True, False]
        assert since.tolist() == pytest.approx([0.0, 0.3])


class TestPerturbPoses:
    """E2, odometry robustness: slots live in room coordinates, so the design's central claim is bought with odometry. A robot's
    odometry drifts, and D1 so far answers only the perfect-odometry case."""

    def test_no_noise_leaves_the_pose_alone(self):
        pose = np.array([[1.0, 2.0, 0.5]], np.float32)
        assert np.allclose(perturb_poses(pose, np.zeros(1), 0.0, 0.0), pose)

    def test_drift_grows_with_elapsed_time_and_is_the_same_for_one_stream(self):
        pose = np.zeros((1, 3), np.float32)
        near = perturb_poses(pose, np.array([1.0]), 0.1, 0.0)
        far = perturb_poses(pose, np.array([10.0]), 0.1, 0.0)
        assert np.linalg.norm(far[0, :2]) > np.linalg.norm(near[0, :2])
        assert np.allclose(perturb_poses(pose, np.array([1.0]), 0.1, 0.0), near)      # deterministic in elapsed time

    def test_heading_drift_turns_the_pose(self):
        pose = np.zeros((1, 3), np.float32)
        assert abs(float(perturb_poses(pose, np.array([10.0]), 0.0, 1.0)[0, 2])) > 0.01


class TestSampleDrift:
    """S3/E11: a drift rate per stream, redrawn each chunk -- domain randomisation over odometry error rather than one rate."""

    def test_every_stream_gets_its_own_rate_inside_the_band(self):
        drift, heading = sample_drift(np.random.default_rng(0), 8, 0.05, 2.0)
        assert drift.shape == (8,) and heading.shape == (8,)
        assert (drift >= 0).all() and (drift <= 0.05).all()
        assert (heading >= 0).all() and (heading <= 2.0).all()
        assert len(np.unique(drift)) > 1               # a band, not one rate applied eight times

    def test_successive_chunks_draw_different_rates(self):
        rng = np.random.default_rng(0)
        assert not np.allclose(sample_drift(rng, 4, 0.05, 2.0)[0], sample_drift(rng, 4, 0.05, 2.0)[0])

    def test_a_zero_band_asks_for_no_drift_at_all(self):
        drift, heading = sample_drift(np.random.default_rng(0), 4, 0.0, 0.0)
        assert not drift.any() and not heading.any()


class TestPerturbPosesPerStream:
    """S3/E11: the same drift model, but each stream on its own rate."""

    def test_each_stream_drifts_at_its_own_rate(self):
        pose = np.zeros((2, 3), np.float32)
        out = perturb_poses(pose, np.array([10.0, 10.0]), np.array([0.0, 0.1]), 0.0)
        assert np.allclose(out[0, :2], 0.0)                       # the stream given no drift is untouched
        assert np.linalg.norm(out[1, :2]) > 0.1

    def test_an_all_zero_array_is_the_same_as_no_drift(self):
        pose = np.array([[1.0, 2.0, 0.5], [3.0, 4.0, 1.0]], np.float32)
        assert np.allclose(perturb_poses(pose, np.array([5.0, 5.0]), np.zeros(2), np.zeros(2)), pose)

    def test_a_scalar_still_works_as_it_always_did(self):
        pose = np.zeros((2, 3), np.float32)
        array = perturb_poses(pose, np.array([4.0, 4.0]), np.array([0.1, 0.1]), 0.0)
        assert np.allclose(perturb_poses(pose, np.array([4.0, 4.0]), 0.1, 0.0), array)


class TestChunkElapsed:
    """S3/E11: drift accumulates from the frame the slots were created in, so the clock restarts wherever memory does."""

    def test_it_accumulates_across_the_chunk(self):
        dt = np.full((1, 4), 0.025)
        elapsed, carry = chunk_elapsed(dt, np.zeros((1, 4), bool), np.zeros(1))
        assert elapsed[0].tolist() == pytest.approx([0.025, 0.05, 0.075, 0.1])
        assert carry[0] == pytest.approx(0.1)

    def test_a_reset_restarts_the_clock_for_that_stream_only(self):
        dt = np.full((2, 3), 1.0)
        reset = np.array([[False, True, False], [False, False, False]])
        elapsed, _ = chunk_elapsed(dt, reset, np.zeros(2))
        assert elapsed[0].tolist() == pytest.approx([1.0, 0.0, 1.0])
        assert elapsed[1].tolist() == pytest.approx([1.0, 2.0, 3.0])

    def test_it_carries_on_from_the_previous_chunk(self):
        dt = np.full((1, 2), 1.0)
        elapsed, carry = chunk_elapsed(dt, np.zeros((1, 2), bool), np.array([10.0]))
        assert elapsed[0].tolist() == pytest.approx([11.0, 12.0])
        assert carry[0] == pytest.approx(12.0)


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
        stopper = EarlyStopping(patience_evals=2, min_epochs=0, max_epochs=99)
        for metric in (0.5, 0.4):
            stopper.update(metric)
        restored = EarlyStopping(patience_evals=2, min_epochs=0, max_epochs=99)
        restored.__dict__.update(stopper.state())
        assert restored.state() == stopper.state()
        assert not restored.should_stop(epoch=1)
        restored.update(0.3)
        assert restored.should_stop(epoch=1)                            # the second bad evaluation, not a fresh count

    def test_the_best_checkpoint_keeps_its_bar_across_a_resume(self, tmp_path):
        best = BestCheckpoint(tmp_path / "best.pth")
        best.offer(0.8, {"model": 1})
        restored = BestCheckpoint(tmp_path / "best.pth")
        restored.__dict__.update(best.state())
        assert not restored.offer(0.7, {"model": 2})                             # a worse epoch after the crash cannot overwrite it
        assert restored.offer(0.9, {"model": 3})


class TestGradientVerdict:
    """What to do with one bad iteration. A gradient norm that overflows float32 is survivable -- `clip_grad_norm_` zeroes the
    update and training continues -- and treating it as fatal cost a healthy 5 h run at its best epoch (`TODO.md` A51 E6,
    2026-09-27). A non-finite loss is not survivable, and neither is an endless stream of overflowing steps."""

    def test_a_healthy_iteration_takes_its_step(self):
        assert gradient_verdict(0.8, 1.2, skipped=0) == "take"

    def test_an_overflowing_gradient_norm_skips_the_step(self):
        assert gradient_verdict(0.8, float("inf"), skipped=0) == "skip"
        assert gradient_verdict(0.8, float("nan"), skipped=3) == "skip"

    def test_too_many_skips_in_one_epoch_stop_the_run(self):
        assert gradient_verdict(0.8, float("inf"), skipped=GRAD_SKIP_LIMIT) == "stop"

    def test_a_non_finite_loss_always_stops(self):
        assert gradient_verdict(float("nan"), 1.0, skipped=0) == "stop"
        assert gradient_verdict(float("inf"), 1.0, skipped=0) == "stop"


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
                features=rng.standard_normal((n, 6, 8)).astype(np.float16),
                free_range=np.full((n, 8), MAX_RANGE_M, np.float16))
        return pack([recording(23), recording(17)])

    def _model(self):
        from takheliped.detectors.takheliped import ObjectMemory, SlotRules
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

    def test_the_scored_frames_can_be_handed_back_with_their_global_index(self):
        """`TODO.md` A56 scores the memory's detections again -- GOSPA, a block bootstrap, per-person strata -- so `evaluate` can
        return the frames it scored, each with the packed frame it came from."""
        split, model = self._split(), self._model()
        frames = {}
        result = evaluate(model, split, torch.device("cpu"), 2, 2, evaluation_segments(split, 0, 0, 0), frames_out=frames)
        assert len(frames["candidates"]) == len(frames["memory"]) == len(frames["index"]) == result["memory"]["frames"]
        assert score_detections(frames["memory"])["ap"] == pytest.approx(result["memory"]["ap"])
        expected = sorted(int(o) + i for o, n in zip(split.offsets, split.lengths) for i in range(0, int(n), 2))
        assert sorted(frames["index"]) == expected
        for f, (scores, xy, gt) in zip(frames["index"], frames["candidates"]):
            assert np.array_equal(gt, split.data.gt_xy[f][split.data.gt_valid[f]])

    def test_scoring_can_be_skipped_when_only_the_frames_are_wanted(self):
        """Scoring every annotated test frame costs ~20 min per checkpoint in `score_detections`; A56 rescores the frames itself."""
        split, model = self._split(), self._model()
        scored, unscored = {}, {}
        evaluate(model, split, torch.device("cpu"), 1, 2, evaluation_segments(split, 0, 0, 0), frames_out=scored)
        result = evaluate(model, split, torch.device("cpu"), 1, 2, evaluation_segments(split, 0, 0, 0), frames_out=unscored, compute_metrics=False)
        assert result == dict(candidates={}, memory={})
        assert unscored["index"] == scored["index"]
        for (a, xa, _), (b, xb, _) in zip(unscored["memory"], scored["memory"]):
            assert np.array_equal(a, b) and np.array_equal(xa, xb)


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
            features=np.zeros((n, 1, 0), np.float16), free_range=np.full((n, 8), MAX_RANGE_M, np.float16))
        reported = [len(scores) for scores, _, _ in replay_sort(pack([seq]), min_hits=3)]
        assert reported[:5] == [0, 0, 0, 0, 0]           # its first frame is not confirmed
        assert reported[-1] == 1


class TestUnionDecode:
    """Item 34: both candidate decoders at once, deduplicated, for stage 3 to weigh.

    Measured 2026-10-15 before any of this was written: `nms-mean` and the vote grid each cover about **900 people of
    30,283 that the other misses**, so their union reaches 91.8-92.5% where the better alone reaches 88.9-89.7%. The
    union is the only way to put those people in front of the memory at all.

    Two properties carry the whole idea, and both are easy to lose:

    * **Dedup must not cost coverage.** Both decoders usually fire on the same person, and feeding two candidates for one
      person breaks the memory's one-to-one slot matching -- the second spawns a duplicate slot. So coincident pairs
      collapse, but at a radius **below** the 0.5 m match radius, or a collapse could swallow a genuinely separate person
      and give back the headroom the union was built for.
    * **A person either decoder finds must survive**, which is the headroom itself.
    """

    @staticmethod
    def _angles(n=720):
        return np.linspace(-np.pi / 2, np.pi / 2, n)

    def _one_person(self, at=(0.0, 4.0), beams=None):
        """A scan whose beams vote a person at `at`, which both decoders should find."""
        angles = self._angles()
        scan = np.full(len(angles), 8.0)
        prob = np.zeros(len(angles))
        beams = np.arange(350, 362) if beams is None else beams
        prob[beams] = 0.9
        target = np.asarray(at, dtype=float)
        r_gt, phi_gt = np.hypot(*target), np.arctan2(-target[0], target[1])
        votes = np.zeros((len(angles), 2))
        dphi = phi_gt - angles[beams]
        votes[beams, 0] = r_gt * np.sin(dphi)
        votes[beams, 1] = r_gt * np.cos(dphi) - scan[beams]
        return scan, angles, prob, votes

    def test_one_person_found_by_both_decoders_yields_one_candidate(self):
        scan, angles, prob, votes = self._one_person()

        out = union_decode(scan, angles, prob, votes, merge_m=0.3)

        assert len(out) == 1

    def test_the_survivor_keeps_the_higher_of_the_two_scores(self):
        scan, angles, prob, votes = self._one_person()
        from train_takheliped import decode_votes, nms_decode

        both = [nms_decode(scan, angles, prob, votes, 0.5, average=True)[:, 0].max(),
                max(d[0] for d in decode_votes(scan, angles, prob, votes))]
        out = union_decode(scan, angles, prob, votes, merge_m=0.3)

        assert out[0, 0] == pytest.approx(max(both))

    def test_two_people_further_apart_than_the_merge_radius_both_survive(self):
        """The merge radius is below the 0.5 m match radius precisely so this cannot collapse."""
        angles = self._angles()
        scan = np.full(len(angles), 8.0)
        prob = np.zeros(len(angles))
        votes = np.zeros((len(angles), 2))
        for target, beams in (((-1.2, 4.0), np.arange(300, 312)), ((1.2, 4.0), np.arange(410, 422))):
            prob[beams] = 0.9
            t = np.asarray(target, dtype=float)
            r_gt, phi_gt = np.hypot(*t), np.arctan2(-t[0], t[1])
            dphi = phi_gt - angles[beams]
            votes[beams, 0], votes[beams, 1] = r_gt * np.sin(dphi), r_gt * np.cos(dphi) - scan[beams]

        out = union_decode(scan, angles, prob, votes, merge_m=0.3)

        assert len(out) == 2

    def test_a_person_only_one_decoder_finds_still_comes_through(self):
        """The headroom itself: whatever either parent produced is in the union unless something coincides with it."""
        scan, angles, prob, votes = self._one_person()
        from train_takheliped import decode_votes, nms_decode

        parents = np.concatenate([nms_decode(scan, angles, prob, votes, 0.5, average=True),
                                  np.asarray(decode_votes(scan, angles, prob, votes), dtype=float).reshape(-1, 3)])
        out = union_decode(scan, angles, prob, votes, merge_m=0.3)

        for parent in parents:
            assert np.min(np.linalg.norm(out[:, 1:] - parent[1:], axis=1)) <= 0.3

    def test_an_empty_scan_decodes_to_nothing(self):
        angles = self._angles()

        out = union_decode(np.full(len(angles), 8.0), angles, np.zeros(len(angles)), np.zeros((len(angles), 2)), merge_m=0.3)

        assert out.shape == (0, 3)

    def test_the_output_is_ordered_by_score_so_the_pool_cap_keeps_the_best(self):
        angles = self._angles()
        scan = np.full(len(angles), 8.0)
        prob = np.zeros(len(angles))
        votes = np.zeros((len(angles), 2))
        for target, beams, p in (((-1.5, 4.0), np.arange(300, 312), 0.4), ((1.5, 4.0), np.arange(410, 422), 0.9)):
            prob[beams] = p
            t = np.asarray(target, dtype=float)
            r_gt, phi_gt = np.hypot(*t), np.arctan2(-t[0], t[1])
            dphi = phi_gt - angles[beams]
            votes[beams, 0], votes[beams, 1] = r_gt * np.sin(dphi), r_gt * np.cos(dphi) - scan[beams]

        out = union_decode(scan, angles, prob, votes, merge_m=0.3)

        assert np.all(np.diff(out[:, 0]) <= 0)


class TestTheCoastingHeadIsGone:
    """The owner's call, 2026-10-16: item 24 measured the coasting head costing nothing, so the architecture drops it.

    Ablated it scored **AP 86.08%** against the matched seed's 86.35% and the three-seed control range 85.49-86.35, with
    every other metric inside the control's own spread. It is 137 parameters of 173,579, so the reason to remove it is
    not size: it fires 0.0012-0.0020 times per populated frame and 2 / 0 / 0 times across 72,533 person-free frames, and
    a write-up should not describe a mechanism that never acts.

    **The flag survives the default.** Every published number to date was produced *with* the head, so reproducing those
    runs has to stay possible -- the same reason the owner kept `SpaceTimeCNNDetector` in item 32 rather than deleting
    the architecture its decisions were made on.
    """

    def test_a_new_memory_has_no_coasting_head(self):
        assert ObjectMemory(SlotRules(capacity=8), dim=16, heads=2).coast is None

    def test_a_checkpoint_written_before_the_flag_existed_still_loads_with_its_head(self):
        """Those checkpoints carry `coast.weight` and say nothing about `coast`, so the weights decide, not the default.

        Falling back to the constructor default -- which is what every other ablation here does -- would now build a
        head-free model and refuse the adopted model's own weights.
        """
        trained = ObjectMemory(coast=True)          # defaults: `object_memory` rebuilds at them when `args` is silent
        checkpoint = {"model": trained.state_dict(), "args": {}}

        restored = object_memory(checkpoint, feature_dim=0)

        assert restored.coast is not None

    def test_a_checkpoint_that_says_it_had_no_head_is_built_without_one(self):
        trained = ObjectMemory(coast=False)
        checkpoint = {"model": trained.state_dict(), "args": {"coast": "False"}}

        assert object_memory(checkpoint, feature_dim=0).coast is None

    def test_the_recorded_flag_wins_over_the_weights(self):
        """`args` is what the run actually asked for; the weights only answer when it is silent."""
        trained = ObjectMemory(coast=True)
        checkpoint = {"model": trained.state_dict(), "args": {"coast": "True"}}

        assert object_memory(checkpoint, feature_dim=0).coast is not None


class TestTheMatchedPersonSPosition:
    """Item 46's L1 target: where the person a candidate matched actually is.

    `candidate_targets` already runs the gated Hungarian match but returns only booleans, and eleven call sites unpack
    its two values -- so rather than churn nine of them, both it and `matched_person_xy` now call one shared
    `_gated_assignment`. These tests pin that the two agree, because a silent divergence between "which candidates are
    people" and "which person each candidate regresses towards" would train the head against the wrong target.
    """

    RADIUS = 0.5

    def test_an_unmatched_candidate_has_no_target(self):
        """NaN rather than zeros: a zero would be a real position near the sensor and the L1 would chase it."""
        xy = np.array([[0.0, 1.0], [3.0, 3.0]])
        out = matched_person_xy(xy, np.ones(2, bool), np.array([[0.0, 1.1]]), self.RADIUS)

        assert np.allclose(out[0], [0.0, 1.1])
        assert np.isnan(out[1]).all()

    def test_it_returns_the_person_the_same_assignment_picked(self):
        """Two candidates, two people, crossed so a greedy rule would pair them differently from the optimal one."""
        xy = np.array([[0.0, 1.0], [0.0, 1.4]])
        gt = np.array([[0.0, 1.45], [0.0, 1.05]])

        target, _ = candidate_targets(xy, np.ones(2, bool), gt, self.RADIUS)
        out = matched_person_xy(xy, np.ones(2, bool), gt, self.RADIUS)

        assert target.tolist() == [True, True]
        assert np.allclose(out[0], [0.0, 1.05]) and np.allclose(out[1], [0.0, 1.45])

    def test_the_two_functions_agree_on_which_candidates_matched(self):
        """The invariant that lets them share a matching rule rather than drift apart."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            xy = rng.normal(scale=2.0, size=(6, 2))
            gt = rng.normal(scale=2.0, size=(4, 2))
            valid = rng.random(6) > 0.3

            target, _ = candidate_targets(xy, valid, gt, self.RADIUS)
            out = matched_person_xy(xy, valid, gt, self.RADIUS)

            assert np.array_equal(target, ~np.isnan(out[:, 0]))

    def test_an_invalid_candidate_never_gets_a_target(self):
        xy = np.array([[0.0, 1.0], [0.0, 1.05]])
        out = matched_person_xy(xy, np.array([False, True]), np.array([[0.0, 1.0]]), self.RADIUS)

        assert np.isnan(out[0]).all() and np.allclose(out[1], [0.0, 1.0])

    def test_a_frame_with_nobody_in_it_gives_every_candidate_nothing(self):
        out = matched_person_xy(np.array([[0.0, 1.0]]), np.ones(1, bool), np.empty((0, 2)), self.RADIUS)

        assert out.shape == (1, 2) and np.isnan(out).all()

    def test_nothing_beyond_the_radius_is_matched(self):
        out = matched_person_xy(np.array([[0.0, 1.0]]), np.ones(1, bool), np.array([[0.0, 9.0]]), self.RADIUS)

        assert np.isnan(out).all()


class TestEveryStage3FlagReachesTheModel:
    """A flag that is parsed, stored and restored but never *passed* trains the control and reports it as the arm.

    That happened on 2026-10-22: `--global-state` reached the parser and `object_memory`'s restore table, but not the
    `ObjectMemory(...)` call in `train_memory`, so a smoke run trained the adopted model and printed the adopted
    parameter count. `ObjectMemory` has fifteen-odd such switches, so this checks the wiring by introspection rather
    than by one more pair of eyes.
    """

    #: Constructor parameters that deliberately do not come from a same-named CLI flag.
    NOT_FLAGS = {"rules", "sigma_m", "decay_init_s", "feature_dim", "rescore"}

    def test_every_constructor_switch_with_a_matching_flag_is_passed_at_the_construction_site(self):
        import inspect
        from inspect import signature

        from train_takheliped import train_memory

        source = inspect.getsource(train_memory)
        call = source[source.index("ObjectMemory("):]
        call = call[:call.index(").to(device)")]
        flags = {action.dest for action in build_parser()._subparsers._group_actions[0].choices["step3a"]._actions}
        shared = {name for name in signature(ObjectMemory).parameters if name not in self.NOT_FLAGS} & flags

        missing = sorted(name for name in shared if f"{name}=args.{name}" not in call)

        assert not missing, f"parsed but never passed to ObjectMemory: {', '.join(missing)}"

    #: `SlotRules` fields that deliberately do not come from a same-named flag -- `capacity` comes from `--slots`, and
    #: the rest are constants A51's S1 swept and found unbinding.
    NOT_RULE_FLAGS = {"capacity", "floor", "margin", "fade_time_s", "gain", "gate_m", "velocity_smoothing"}

    def test_every_slot_rule_with_a_matching_flag_is_passed_too(self):
        """The same bug class one layer in: `SlotRules` is built beside `ObjectMemory` and is just as easy to forget."""
        import inspect
        from inspect import signature

        from takheliped.detectors.takheliped import SlotRules
        from train_takheliped import train_memory

        source = inspect.getsource(train_memory)
        call = source[source.index("SlotRules("):]
        call = call[:call.index(")")]
        flags = {action.dest for action in build_parser()._subparsers._group_actions[0].choices["step3a"]._actions}
        shared = {name for name in signature(SlotRules).parameters if name not in self.NOT_RULE_FLAGS} & flags

        missing = sorted(name for name in shared if f"{name}=args.{name}" not in call)

        assert not missing, f"parsed but never passed to SlotRules: {', '.join(missing)}"


class TestTheSensorFrameTargets:
    """Item 18's training side: `make_targets` in the frame the head regresses.

    The geometry itself -- that the two parameterisations name one point, and that the rotation is its own inverse --
    is pinned in `test_takheliped.py`. What belongs here is what the arm trains on, because `make_targets` is
    `utils/train.py`'s and is shared with every other detector, so a default that moved would retarget all of them.
    """

    ANGLES = np.deg2rad(np.linspace(-90.0, 90.0, 91))

    @staticmethod
    def _endpoints(scan, angles):
        return np.stack([scan * -np.sin(angles), scan * np.cos(angles)], axis=-1)

    def test_the_sensor_frame_target_is_the_person_minus_the_beam_endpoint(self):
        """What the arm actually trains on: one shared frame, so the same person is the same target for every beam."""
        scan = np.full(len(self.ANGLES), 4.0)
        person = (4.0, 0.0)
        labels, sensor = make_targets(scan, self.ANGLES, {3: [person]}, frame="sensor")
        seen = labels > 0
        centre = np.array([person[0] * -np.sin(person[1]), person[0] * np.cos(person[1])])

        assert seen.any()
        assert np.allclose(sensor[seen], centre[None, :] - self._endpoints(scan, self.ANGLES)[seen], atol=1e-6)

    def test_the_two_target_frames_describe_the_same_people(self):
        scan = np.linspace(2.0, 8.0, len(self.ANGLES))
        people = [(3.0, -0.4), (6.0, 0.5)]
        beam_labels, beam = make_targets(scan, self.ANGLES, {3: people})
        sensor_labels, sensor = make_targets(scan, self.ANGLES, {3: people}, frame="sensor")

        assert np.array_equal(beam_labels, sensor_labels)
        assert np.allclose(swap_vote_frame(sensor, self.ANGLES), beam, atol=1e-6)

    def test_an_unlabelled_beam_votes_zero_in_either_frame(self):
        """A zero target must stay zero: the loss is taken over every beam, so a rotated nothing would be something."""
        scan = np.full(len(self.ANGLES), 4.0)
        labels, sensor = make_targets(scan, self.ANGLES, {3: [(4.0, 0.0)]}, frame="sensor")

        assert (sensor[labels == 0] == 0).all()

    def test_the_beam_frame_is_the_default_so_every_other_detector_is_untouched(self):
        scan = np.linspace(2.0, 8.0, len(self.ANGLES))
        people = [(3.0, -0.4)]

        assert np.array_equal(make_targets(scan, self.ANGLES, {3: people})[1],
                              make_targets(scan, self.ANGLES, {3: people}, frame="beam")[1])

    def test_an_unknown_target_frame_is_refused(self):
        with pytest.raises(ValueError, match="frame"):
            make_targets(np.full(8, 4.0), self.ANGLES[:8], {3: []}, frame="world")


class TestStage3SShapeFlags:
    """`TODO.md` item 49: `--dim` and `--heads`, the last shape knobs in the project never varied.

    Item 45 swept stage 2's four and deferred these two in as many words -- "`dim`/`heads` are stage-3 and wait on the
    cheap half". The cheap half is done, so the adopted 128/4 is a choice nobody has priced, and `dim` is the one knob
    that could cut CPU cost: it sets the width of every slot's state, the component item 36 proved load-bearing.
    """

    def test_the_flags_exist_and_default_to_the_adopted_shape(self):
        flags = {a.dest: a for a in build_parser()._subparsers._group_actions[0].choices["step3a"]._actions}

        assert flags["dim"].default == 64      # item 49, adopted 2026-10-27
        assert flags["heads"].default == 4

    def test_a_checkpoint_round_trips_its_own_width(self):
        """The trap this guards: without a shape key a resumed `--dim 64` arm rebuilds at 128 and refuses its own weights."""
        narrow = ObjectMemory(SlotRules(capacity=8), dim=64, heads=8, feature_dim=3)
        checkpoint = {"model": narrow.state_dict(), "args": {"dim": "64", "heads": "8", "slots": "8"}}

        rebuilt = object_memory(checkpoint, feature_dim=3)

        assert rebuilt.dim == 64
        assert rebuilt.recall.heads == 8

    def test_a_checkpoint_predating_the_flags_still_rebuilds_at_the_adopted_shape(self):
        """Every stage-3 checkpoint on disk was written before these flags existed and must keep loading."""
        wide = ObjectMemory(SlotRules(capacity=8), dim=128, heads=4, feature_dim=3)
        checkpoint = {"model": wide.state_dict(), "args": {"slots": "8"}}

        rebuilt = object_memory(checkpoint, feature_dim=3)

        assert rebuilt.dim == 128
        assert rebuilt.recall.heads == 4


class TestTheSoftBeamTargetReachesStage2STraining:
    """`TODO.md` item 44(a): the Gaussian beam target is wired into `step2`, or the flag is decoration.

    This is the bug class that cost item 36 a whole run -- `--global-state` was parsed, stored and restored but never
    passed, so the arm trained the control and reported it as the arm. `--label-sigma-m` is the same shape of risk one
    stage earlier, and the loss is the only place it can act.
    """

    def test_the_flag_exists_and_defaults_to_the_adopted_width(self):
        """Adopted 2026-10-27 after replicating at three seeds; `--label-sigma-m 0` restores the hard label."""
        flags = {a.dest: a for a in build_parser()._subparsers._group_actions[0].choices["step2"]._actions}

        assert flags["label_sigma_m"].default == 0.3

    def test_the_training_loop_builds_the_target_and_hands_it_to_the_loss(self):
        import inspect

        from train_takheliped import train_calibration

        source = inspect.getsource(train_calibration)

        assert "beam_heatmap(" in source, "the flag cannot act without building the target"
        assert "soft_labels=" in source, "the target cannot act without reaching compute_loss"
        assert "args.label_sigma_m" in source, "the target must be built from the flag, not a constant"


class TestTheCliDefaultsAreTheAdoptedArchitecture:
    """`TODO.md` item 52: running `step2`/`step3a` with no flags must train the architecture the project proposes.

    **CLI defaults only -- the `ObjectMemory`/`CalibrationNetwork` constructor defaults deliberately do NOT move.**
    `network_shape` and `object_memory` fall back to a *constructor* default for any flag a checkpoint predates, so
    flipping those would make every older checkpoint rebuild in the new shape and refuse its own weights. The two
    layers answer different questions: the constructor default is "what did a checkpoint that never recorded this
    mean?", the CLI default is "what should a new run do?".
    """

    def _flags(self, mode):
        return {a.dest: a for a in build_parser()._subparsers._group_actions[0].choices[mode]._actions}

    def test_stage3_defaults_to_the_adopted_recipe(self):
        f = self._flags("step3a")

        assert f["recurrence"].default == "mingru"      # item 38
        assert f["exchange"].default is False           # A51 E6
        assert f["learned_match"].default is False      # A51 E7
        assert f["coast"].default is False              # item 24
        assert f["dim"].default == 64                   # item 49
        assert f["chunk_s"].default == [10.0]           # item 41, no curriculum

    def test_stage2_defaults_to_the_adopted_recipe(self):
        f = self._flags("step2")

        assert f["aggregator"].default is False         # item 40
        assert f["label_sigma_m"].default == 0.3        # item 44(a)

    def test_the_constructor_defaults_have_not_moved(self):
        """The guard on the above: old checkpoints load against these, not against the CLI."""
        from inspect import signature

        from takheliped.detectors.takheliped import ObjectMemory

        p = signature(ObjectMemory).parameters

        assert p["dim"].default == 128, "an old checkpoint with no recorded dim is a 128-wide one"
        assert p["recurrence"].default == "selective"
        assert p["exchange"].default is True
        assert p["learned_match"].default is True


class TestTheCarriedStateGradientClip:
    """`TODO.md` item 53: one frame's gradient at the carried slot state, bounded so a long chunk cannot compound.

    **What these do not prove.** Reaching an actual float32 overflow needs ~300 frames of the real recurrence on real
    candidates, which is minutes, not a unit test; that case is covered by the scratchpad replay of the batch
    `a70_slot_attention_s0/step3a_nan.pth` saved, reported in `CHANGELOG.md`. What is testable cheaply is the bound
    itself -- that it holds per frame, that it does nothing when the gradient is already small, and that compounding
    is what it removes.
    """

    def test_a_gradient_above_the_bound_is_scaled_to_it(self):
        grad = torch.tensor([[[3.0, 4.0]]])          # norm 5
        out = clip_state_grad(grad, 1.0)
        assert out.norm().item() == pytest.approx(1.0)
        assert torch.allclose(out, grad / 5.0)       # direction preserved

    def test_a_gradient_below_the_bound_is_untouched(self):
        grad = torch.tensor([[[0.3, 0.4]]])          # norm 0.5
        assert torch.equal(clip_state_grad(grad, 10.0), grad)

    def test_a_non_finite_gradient_becomes_zero(self):
        grad = torch.tensor([[[float("inf"), float("nan"), 2.0]]])
        out = clip_state_grad(grad, 1.0)
        assert torch.isfinite(out).all()
        assert out.abs().sum().item() == 0.0

    def test_without_the_hook_a_long_recurrence_compounds_and_with_it_it_does_not(self):
        """The same residual recurrence, run twice: the gradient reaching the first frame grows without the bound.

        The gain is kept mild on purpose. A first version used 2.3x over 60 frames, which overflowed the *forward* pass,
        so it tested how an infinite loss is handled rather than whether compounding is removed -- the first thing this
        bound has to do. Overflow has its own test above.
        """
        def run(bound):
            torch.manual_seed(0)
            weight = torch.eye(4) * 0.5
            weight.requires_grad_(True)
            state = torch.ones(1, 1, 4)
            seen = []
            loss = torch.zeros(())
            for _ in range(30):
                state = state + state @ weight
                if bound:
                    state.register_hook(lambda g: clip_state_grad(g, bound))
                state.register_hook(lambda g: seen.append(float(g.norm())) or g)
                loss = loss + state.pow(2).mean()
            loss.backward()
            return max(seen)

        unbounded = run(None)
        bounded = run(2.0)
        assert unbounded > 1e5, f"the recurrence did not compound, so the test proves nothing: {unbounded}"
        assert bounded <= 2.0 + 1e-5, f"the bound did not hold: {bounded}"

    def test_the_clip_is_off_by_default_so_every_arm_already_measured_is_unaffected(self):
        """Turning this on by default would silently change the gradient estimator of `a74_fixed256_s0`'s 85.73%."""
        args = build_parser().parse_args(["step3a", "--stage2", "x.pth"])
        assert args.state_grad_clip == 0.0

    def test_the_flag_is_accepted_and_lands_in_the_stored_args(self):
        args = build_parser().parse_args(["step3a", "--stage2", "x.pth", "--state-grad-clip", "10"])
        assert args.state_grad_clip == 10.0
        assert vars(args)["state_grad_clip"] == 10.0     # this dict is what the checkpoint stores
