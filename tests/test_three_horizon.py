"""Step 0 of the three-horizon detector: input features, sector alignment, causal temporal convolution.

`docs/PROPOSAL.md` §5.3-§5.4 and §10 step 0. Conventions pinned here:

    beam angle   0 = forward, counterclockwise positive, beams in increasing angle
    pose         world odometry (x, y, theta): x forward at theta = 0, y left, theta CCW
    features     [range, range - local median,
                  along(-2), across(-2), along(-1), across(-1), along(+1), across(+1), along(+2), across(+2)]
                 where along/across are the neighbour's endpoint in the beam's own frame, relative to the beam's endpoint
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))

from follow_the_drow.detectors.three_horizon import (
    N_BEAM_FEATURES, CalibrationNetwork, CausalTemporalConv, ObjectMemory, SlotRules, align_sectors, beam_features, fold_legacy_calibration,
    initial_slots, load_calibration_weights, manage_slots, render_prior, sanitize_ranges, sensor_to_world, world_to_sensor,
)


def _angles(n: int, fov_deg: float = 180.0) -> torch.Tensor:
    """Beams from -fov/2 upward in steps of fov/n, the FROG layout."""
    return torch.tensor(np.deg2rad(-fov_deg / 2 + np.arange(n) * fov_deg / n), dtype=torch.float32)


def _raycast(pose, angles, circles, room_radius):
    """Ranges and world endpoints of rays from `pose` against solid circles inside a circular room."""
    x0, y0, th = pose
    phi = th + angles.double().numpy()
    d = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    o = np.array([x0, y0])
    best = np.full(len(phi), np.inf)
    for cx, cy, rad in circles:
        oc = o - np.array([cx, cy])
        b = d @ oc
        disc = b ** 2 - (oc @ oc - rad ** 2)
        t = -b - np.sqrt(np.where(disc >= 0, disc, np.nan))
        hit = (disc >= 0) & (t > 0)
        best = np.where(hit & (t < best), t, best)
    b = d @ o
    wall = -b + np.sqrt(b ** 2 - (o @ o - room_radius ** 2))
    best = np.minimum(best, wall)
    return best, o + best[:, None] * d


class TestBeamFeatures:

    def test_constant_range_matches_hand_geometry(self):
        angles = _angles(180)                      # 1 degree per beam
        f = beam_features(torch.full((180,), 2.0), angles)
        d = math.radians(1.0)
        want = [2.0, 0.0,
                2 * math.cos(2 * d) - 2, -2 * math.sin(2 * d),
                2 * math.cos(d) - 2, -2 * math.sin(d),
                2 * math.cos(d) - 2, 2 * math.sin(d),
                2 * math.cos(2 * d) - 2, 2 * math.sin(2 * d)]
        assert f.shape == (10, 180)
        assert f[:, 90].tolist() == pytest.approx(want, abs=1e-5)

    def test_a_leg_gives_the_same_numbers_at_any_bearing(self):
        angles = _angles(720)                      # FROG: 0.25 degree per beam
        centre = {}
        for name, bearing_deg, beam in (("right", -40.0, 200), ("left", 25.0, 460)):
            b = math.radians(bearing_deg)
            ranges, _ = _raycast((0.0, 0.0, 0.0), angles, [(3.0 * math.cos(b), 3.0 * math.sin(b), 0.06)], room_radius=1e4)
            ranges = np.where(ranges > 1e3, np.inf, ranges)
            f = beam_features(torch.tensor(ranges, dtype=torch.float32), angles)
            centre[name] = f[:, beam - 2:beam + 3]
        assert torch.allclose(centre["right"][[0, 2, 3, 4, 5, 6, 7, 8, 9]], centre["left"][[0, 2, 3, 4, 5, 6, 7, 8, 9]], atol=1e-4)

    def test_missing_and_out_of_range_returns_all_read_as_the_range_limit(self):
        """One rule for every dataset's encoding (`memory/noise-structure.md`): FROG's +inf and 61.0 m, DROW's 29.96 m, a real
        wall beyond the limit, and a non-positive reading all mean "nothing within range", and read as the limit itself."""
        angles = _angles(8)
        f = beam_features(torch.tensor([float("inf"), float("nan"), 61.0, 29.96, 15.0, 0.0, -1.0, 3.0]), angles)
        assert f[0].tolist() == pytest.approx([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 3.0])
        assert torch.isfinite(f).all()

    def test_the_range_limit_is_a_parameter(self):
        angles = _angles(4)
        f = beam_features(torch.tensor([float("inf"), 12.0, 25.0, 3.0]), angles, max_range_m=20.0)
        assert f[0].tolist() == pytest.approx([20.0, 12.0, 20.0, 3.0])

    def test_sanitize_ranges(self):
        r = sanitize_ranges(torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0, -2.0, 7.5, 12.0]), 10.0)
        assert r.tolist() == pytest.approx([10.0, 10.0, 10.0, 10.0, 10.0, 7.5, 10.0])

    def test_range_minus_median_marks_a_return_in_front_of_its_surroundings(self):
        angles = _angles(180)
        ranges = torch.full((180,), 5.0)
        ranges[90] = 1.0
        f = beam_features(ranges, angles)
        assert f[1, 88:93].tolist() == pytest.approx([0.0, 0.0, -4.0, 0.0, 0.0])

    def test_leading_dimensions_are_kept(self):
        angles = _angles(36)
        ranges = torch.rand(2, 3, 36) * 5 + 0.5
        f = beam_features(ranges, angles)
        assert f.shape == (2, 3, 10, 36)
        assert torch.allclose(f[1, 2], beam_features(ranges[1, 2], angles))


def _pose(x, y, th):
    return torch.tensor([[x, y, th]], dtype=torch.float32)


class TestAlignSectors:

    def test_rotation_by_whole_sectors_is_an_index_shift_with_empty_sectors_zero(self):
        angles = _angles(720)                      # 90 sectors of 2 degrees
        feats = torch.arange(90, dtype=torch.float32).view(1, 1, 90)
        out = align_sectors(feats, _pose(0, 0, 0), _pose(0, 0, math.radians(10)), angles)
        assert out.shape == (1, 4, 90)
        assert out[0, 0, 0].item() == 5.0
        assert out[0, 0, 84].item() == 89.0
        assert (out[0, 0, 85:] == 0).all()

    def test_pose_channels_hold_the_past_pose_in_the_current_frame(self):
        angles = _angles(720)
        out = align_sectors(torch.zeros(1, 1, 90), _pose(1, 0, 0), _pose(0, 0, math.pi / 2), angles)
        # past pose is 1 m ahead in world x; the robot now faces world y, so that point is 1 m to its right
        assert out[0, 1:, 0].tolist() == pytest.approx([0.0, -1.0, -math.pi / 2], abs=1e-6)
        assert torch.allclose(out[0, 1:], out[0, 1:, :1].expand(-1, 90))


def _walk(batch, steps, seed):
    g = torch.Generator().manual_seed(seed)
    moves = torch.randn(batch, steps, 3, generator=g) * torch.tensor([0.02, 0.01, 0.01])
    return moves.cumsum(dim=1)


class TestCausalTemporalConv:
    """The coarse convolution over bottleneck sectors; the per-beam fine convolution was removed 2026-09-17 (`TODO.md` A50)."""

    def _setup(self, steps=40):
        torch.manual_seed(0)
        conv = CausalTemporalConv(4, 6, [0, 8, 16, 24, 32], _angles(36), kernel_size=3)
        return conv, torch.randn(2, steps, 4, 12), _walk(2, steps, seed=1)

    def test_streaming_reproduces_the_clip_frame_by_frame(self):
        conv, x, poses = self._setup()
        clip = conv(x, poses)
        cache = None
        for t in range(x.shape[1]):
            y, cache = conv.step(x[:, t], poses[:, t], cache)
            assert torch.allclose(y, clip[:, t], atol=1e-5), t

    def test_frames_before_the_clip_read_its_first_frame(self):
        conv, x, poses = self._setup(steps=1)
        repeated = conv(x.expand(-1, 3, -1, -1), poses.expand(-1, 3, -1))
        assert torch.allclose(conv(x, poses)[:, 0], repeated[:, 2], atol=1e-5)

    def test_a_future_frame_never_changes_an_earlier_output(self):
        conv, x, poses = self._setup(steps=10)
        before = conv(x, poses)
        x2 = x.clone()
        x2[:, 7] += 1.0
        after = conv(x2, poses)
        assert torch.allclose(before[:, :7], after[:, :7])
        assert not torch.allclose(before[:, 7], after[:, 7])

    def test_an_empty_cache_starts_over_exactly(self):
        conv, x, poses = self._setup(steps=20)
        cache = None
        for t in range(10):
            _, cache = conv.step(x[:, t], poses[:, t], cache)
        cache = None
        fresh = None
        for t in range(10, 20):
            y, cache = conv.step(x[:, t], poses[:, t], cache)
            y_fresh, fresh = conv.step(x[:, t], poses[:, t], fresh)
            assert torch.equal(y, y_fresh)


class TestFrames:

    def test_sensor_to_world_uses_the_project_detection_frame(self):
        # detection frame: x right, y forward; robot at (1, 2) facing world +y
        pose = _pose(1.0, 2.0, math.pi / 2)
        pts = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        assert sensor_to_world(pts, pose)[0].tolist() == [pytest.approx([1.0, 3.0], abs=1e-6), pytest.approx([2.0, 2.0], abs=1e-6)]

    def test_world_to_sensor_inverts_it(self):
        pose = _pose(0.4, -1.2, 2.3)
        pts = torch.tensor([[[0.3, 2.0], [-1.5, 0.7]]])
        assert torch.allclose(world_to_sensor(sensor_to_world(pts, pose), pose), pts, atol=1e-5)


# ---------------------------------------------------------------- slot bookkeeping: the rules of PROPOSAL.md §5.5

def _rules(**kw):
    args = dict(capacity=4, floor=0.05, margin=0.1, fade_time_s=10.0, gain=0.1, gate_m=0.6, velocity_smoothing=0.1)
    args.update(kw)
    return SlotRules(**args)


def _step(slots, rules, pts=(), scores=(), dt=0.04):
    p = max(len(pts), 1)
    xy, s, valid = torch.zeros(1, p, 2), torch.zeros(1, p), torch.zeros(1, p, dtype=torch.bool)
    for i, (pt, sc) in enumerate(zip(pts, scores)):
        xy[0, i], s[0, i], valid[0, i] = torch.tensor(pt), sc, True
    return manage_slots(slots, xy, s, valid, torch.tensor([dt]), rules)


def _alive_values(slots):
    return sorted(slots.value[0][slots.alive[0]].tolist())


def _decay(dt, rules):
    return math.exp(-dt * math.log(1 / rules.floor) / rules.fade_time_s)


class TestSlotRules:

    def test_candidates_below_the_floor_never_spawn(self):
        r = _rules()
        slots, _ = _step(initial_slots(1, r, 8), r, [(0, 0)], [0.04])
        assert slots.alive.sum() == 0

    def test_free_slots_go_to_the_strongest_candidates_first(self):
        r = _rules(capacity=2)
        slots, events = _step(initial_slots(1, r, 8), r, [(0, 0), (5, 0), (10, 0)], [0.2, 0.9, 0.5])
        assert _alive_values(slots) == pytest.approx([0.5, 0.9])
        assert sorted(events.source[0][events.source[0] >= 0].tolist()) == [1, 2]

    def test_a_candidate_inside_the_gate_updates_rather_than_spawns(self):
        r = _rules()
        slots, _ = _step(initial_slots(1, r, 8), r, [(0, 0)], [0.5])
        slots, events = _step(slots, r, [(0.3, 0)], [0.8])
        k = int(slots.alive[0].nonzero()[0])
        assert slots.alive.sum() == 1 and bool(events.matched[0, k])
        assert slots.value[0, k].item() == pytest.approx(0.5 + 0.1 * (1 - 0.5) * 0.8)
        assert slots.position[0, k].tolist() == pytest.approx([0.3, 0.0])

    def test_a_candidate_matches_at_most_one_slot(self):
        r = _rules()
        slots, _ = _step(initial_slots(1, r, 8), r, [(0, 0), (1.0, 0)], [0.5, 0.5])
        slots, _ = _step(slots, r, [(0.5, 0)], [0.5])
        assert slots.alive.sum() == 2
        assert _alive_values(slots)[0] < 0.5

    def test_an_unmatched_slot_decays_exponentially(self):
        r = _rules()
        slots, _ = _step(initial_slots(1, r, 8), r, [(0, 0)], [0.5])
        slots, _ = _step(slots, r, dt=1.0)
        assert _alive_values(slots) == pytest.approx([0.5 * _decay(1.0, r)], rel=1e-5)

    def test_full_certainty_fades_to_the_floor_in_exactly_the_fade_time(self):
        r = _rules()
        slots, _ = _step(initial_slots(1, r, 8), r, [(0, 0)], [1.0])
        slots, _ = _step(slots, r, dt=9.0)
        assert slots.alive.sum() == 1
        slots, _ = _step(slots, r, dt=2.0)
        assert slots.alive.sum() == 0

    def _full(self):
        r = _rules(capacity=2)
        slots, _ = _step(initial_slots(1, r, 8), r, [(0, 0), (5, 0)], [0.3, 0.6])
        return slots, r

    def test_a_clearly_stronger_candidate_replaces_the_weakest_slot(self):
        slots, r = self._full()
        slots, events = _step(slots, r, [(20, 0)], [0.9])
        assert _alive_values(slots) == pytest.approx([0.6 * _decay(0.04, r), 0.9], rel=1e-5)
        assert int((events.source[0] == 0).sum()) == 1

    def test_no_replacement_within_the_margin(self):
        slots, r = self._full()
        slots, _ = _step(slots, r, [(20, 0)], [0.35])
        assert _alive_values(slots) == pytest.approx([0.3 * _decay(0.04, r), 0.6 * _decay(0.04, r)], rel=1e-5)

    def test_a_slot_matched_this_step_is_never_replaced(self):
        slots, r = self._full()
        slots, _ = _step(slots, r, [(0.1, 0), (20, 0)], [0.06, 0.95])
        assert _alive_values(slots) == pytest.approx([0.3 + 0.1 * 0.7 * 0.06, 0.95], rel=1e-5)

    def test_a_long_tracked_person_survives_a_phantom_flicker(self):
        r = _rules(capacity=1)
        slots = initial_slots(1, r, 8)
        for _ in range(60):
            slots, _ = _step(slots, r, [(0, 0)], [0.9])
        slots, _ = _step(slots, r, dt=0.5)
        slots, _ = _step(slots, r, [(10, 0)], [0.3])
        assert slots.position[0, 0].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)

    def test_velocity_is_learned_from_matches_and_used_to_predict(self):
        r = _rules()
        slots = initial_slots(1, r, 8)
        for i in range(60):
            slots, _ = _step(slots, r, [(0.04 * i, 0)], [0.9])          # 1 m/s along x
        assert slots.velocity[0, 0].tolist() == pytest.approx([1.0, 0.0], abs=0.02)
        x_last = slots.position[0, 0, 0].item()
        slots, _ = _step(slots, r, dt=0.5)
        assert slots.position[0, 0, 0].item() == pytest.approx(x_last + 0.5, abs=0.02)

    def test_streams_in_a_batch_do_not_interact(self):
        r = _rules(capacity=3)
        g = torch.Generator().manual_seed(3)
        both = initial_slots(2, r, 8)
        alone = [initial_slots(1, r, 8), initial_slots(1, r, 8)]
        for _ in range(30):
            xy = torch.rand(2, 5, 2, generator=g) * 4
            s = torch.rand(2, 5, generator=g)
            valid = torch.rand(2, 5, generator=g) > 0.3
            dt = torch.full((2,), 0.04)
            both, _ = manage_slots(both, xy, s, valid, dt, r)
            for b in range(2):
                alone[b], _ = manage_slots(alone[b], xy[b:b + 1], s[b:b + 1], valid[b:b + 1], dt[b:b + 1], r)
        for b in range(2):
            assert torch.equal(both.alive[b], alone[b].alive[0])
            assert torch.allclose(both.value[b], alone[b].value[0])
            assert torch.allclose(both.position[b], alone[b].position[0])


# ---------------------------------------------------------------- the learned object memory

def _memory_inputs(g, batch=2, p=6):
    xy = torch.rand(batch, p, 2, generator=g) * torch.tensor([6.0, 6.0]) - torch.tensor([3.0, 0.0])
    score = torch.rand(batch, p, generator=g) * 0.98 + 0.01
    valid = torch.rand(batch, p, generator=g) > 0.2
    return xy, score, valid


class TestObjectMemory:

    def _run(self, model, steps, seed=0, batch=2):
        g = torch.Generator().manual_seed(seed)
        slots = model.initial_state(batch)
        outs = []
        for t in range(steps):
            xy, score, valid = _memory_inputs(g, batch)
            out = model.step(xy, score, valid, torch.tensor([[0.02 * t, 0.0, 0.01 * t]] * batch), torch.full((batch,), 0.04), slots)
            slots = out.slots
            outs.append((out, score, valid))
        return outs

    def test_untrained_rescoring_reproduces_the_detector_scores(self):
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4)
        for out, score, valid in self._run(model, 12):
            assert torch.allclose(torch.sigmoid(out.candidate_logit)[valid], score[valid], atol=1e-5)

    def test_untrained_slots_report_nobody(self):
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4)
        for out, _, _ in self._run(model, 12):
            assert (torch.sigmoid(out.slot_logit)[out.coasting] < 1e-3).all()

    def test_a_reset_clears_one_stream_and_leaves_the_other(self):
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4)
        g = torch.Generator().manual_seed(1)
        slots = model.initial_state(2)
        for _ in range(5):
            xy, score, valid = _memory_inputs(g)
            xy, score, valid = xy[:1].expand(2, -1, -1), score[:1].expand(2, -1), valid[:1].expand(2, -1)
            slots = model.step(xy, score, valid, torch.zeros(2, 3), torch.full((2,), 0.04), slots).slots
        xy, score, valid = _memory_inputs(g)
        xy, score, valid = xy[:1].expand(2, -1, -1), score[:1].expand(2, -1), valid[:1].expand(2, -1)
        pose, dt = torch.zeros(2, 3), torch.full((2,), 0.04)
        reset = model.step(xy, score, valid, pose, dt, slots, reset=torch.tensor([True, False]))
        kept = model.step(xy, score, valid, pose, dt, slots)
        fresh = model.step(xy[:1], score[:1], valid[:1], pose[:1], dt[:1], model.initial_state(1))
        assert torch.equal(reset.slots.alive[0], fresh.slots.alive[0])
        assert torch.allclose(reset.slots.state[0], fresh.slots.state[0], atol=1e-6)
        assert torch.equal(reset.slots.alive[1], kept.slots.alive[1])
        assert torch.allclose(reset.slots.state[1], kept.slots.state[1], atol=1e-6)

    def test_gradients_reach_matching_exchange_and_recurrence_once_the_gates_open(self):
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4)
        with torch.no_grad():
            model.rescore_gate.fill_(1.0)
        loss = sum(out.candidate_logit[valid].sum() + out.slot_logit.sum() for out, _, valid in self._run(model, 6))
        loss.backward()
        grads = {name: p.grad for name, p in model.named_parameters()}
        assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
        for part in ("match", "exchange", "decay"):
            assert any(g.abs().sum() > 0 for name, g in grads.items() if part in name), part

    def test_candidate_features_reach_the_rescoring_once_the_gate_opens(self):
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, feature_dim=6)
        g = torch.Generator().manual_seed(2)
        xy, score, valid = _memory_inputs(g)
        pose, dt = torch.zeros(2, 3), torch.full((2,), 0.04)
        a = torch.randn(2, 6, 6, generator=g)
        closed = model.step(xy, score, valid, pose, dt, model.initial_state(2), cand_features=a)
        assert torch.allclose(torch.sigmoid(closed.candidate_logit)[valid], score[valid], atol=1e-5)
        with torch.no_grad():
            model.rescore_gate.fill_(1.0)
        with_a = model.step(xy, score, valid, pose, dt, model.initial_state(2), cand_features=a)
        with_b = model.step(xy, score, valid, pose, dt, model.initial_state(2), cand_features=a + 1.0)
        assert not torch.allclose(with_a.candidate_logit, with_b.candidate_logit)

    def test_decay_times_start_spread_over_the_short_and_long_horizons(self):
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, decay_init_s=(1.0, 60.0))
        times = 1.0 / torch.nn.functional.softplus(model.decay_bias)
        assert times.min().item() == pytest.approx(1.0, rel=1e-3)
        assert times.max().item() == pytest.approx(60.0, rel=1e-3)


# ---------------------------------------------------------------- stages 1-2: the calibration network

def _small_network():
    torch.manual_seed(0)
    return CalibrationNetwork(_angles(48), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1).eval()


def _scans(steps, n=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, steps, n, generator=g) * 4 + 1, _walk(1, steps, seed=seed + 1)


class TestCalibrationNetwork:

    def test_streaming_reproduces_the_clip_frame_by_frame(self):
        net = _small_network()
        ranges, poses = _scans(40)
        state, streamed = None, []
        with torch.no_grad():
            for t in range(40):
                logit, votes, features, state = net.step(ranges[:, t], poses[:, t], state)
                streamed.append((logit, votes, features))
            for t in (0, 1, 5, 20, 39):
                for got, want in zip(net(ranges[:, :t + 1], poses[:, :t + 1]), streamed[t]):
                    assert torch.allclose(got, want, atol=1e-4), t

    def test_the_clip_reads_exactly_the_frames_its_taps_need(self):
        # last frame 39; coarse taps 39, 31, 23, 15, 7, each reading only its own frame
        net = _small_network()
        ranges, poses = _scans(40)
        with torch.no_grad():
            base = net(ranges, poses)[0]
            for frame, used in ((35, False), (4, False), (30, False), (5, False), (31, True), (7, True)):
                changed = ranges.clone()
                changed[:, frame] += 1.0
                assert torch.allclose(net(changed, poses)[0], base) != used, frame

    def test_readings_beyond_the_range_limit_are_the_same_as_no_return(self):
        """`max_range_m` is the model's configurable range: past it, a real return and a missing one are indistinguishable."""
        ranges, poses = _scans(3)
        far, missing = ranges.clone(), ranges.clone()
        far[:, :, 10] = 14.0
        missing[:, :, 10] = float("inf")
        with torch.no_grad():
            short = _small_network()
            assert torch.allclose(short(far, poses)[0], short(missing, poses)[0])
            torch.manual_seed(0)
            long = CalibrationNetwork(_angles(48), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1, max_range_m=20.0).eval()
            assert not torch.allclose(long(far, poses)[0], long(missing, poses)[0])

    def test_output_has_a_score_a_vote_and_decoder_features_per_beam(self):
        net = _small_network()
        ranges, poses = _scans(3)
        logit, votes, features = net(ranges.expand(2, -1, -1), poses.expand(2, -1, -1))
        assert logit.shape == (2, 48) and votes.shape == (2, 48, 2) and features.shape == (2, 8, 48)


# ---------------------------------------------------------------- feedback: memory drawn into stage 2's bottleneck

class TestTemporalAblations:
    """`TODO.md` A50 phase 2 removes the coarse convolution by passing `0` as its whole lag list. The conv then spans only the
    current frame, so the test is not that the network still builds but that it has genuinely stopped reading its history."""

    def _net(self, coarse):
        torch.manual_seed(0)
        return CalibrationNetwork(_angles(48), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1, coarse_lags=coarse).eval()

    def _same_present_different_past(self):
        """Two clips sharing their last frame and differing in every earlier one."""
        ranges, poses = _scans(6)
        other, _ = _scans(6, seed=7)
        return ranges, torch.cat([other[:, :-1], ranges[:, -1:]], dim=1), poses

    def test_lag_zero_alone_makes_the_network_ignore_its_history(self):
        net = self._net((0,))
        ranges, rewritten, poses = self._same_present_different_past()
        with torch.no_grad():
            assert torch.allclose(net(ranges, poses)[0], net(rewritten, poses)[0], atol=1e-5)

    def test_the_coarse_convolution_reads_history(self):
        net = self._net((0, 2, 4))
        ranges, rewritten, poses = self._same_present_different_past()
        with torch.no_grad():
            assert not torch.allclose(net(ranges, poses)[0], net(rewritten, poses)[0], atol=1e-5)

    def test_repeating_lag_zero_keeps_the_width_without_the_history(self):
        """The capacity-matched control `TODO.md` A50 phase 2 recommends for the coarse ablation: tapping the current frame five
        times leaves the convolution as wide as five real lags, so a fall in AP cannot be blamed on the lost capacity instead of
        the lost time span."""
        matched = self._net((0, 0, 0, 0, 0))
        single = self._net((0,))
        spanning = self._net((0, 8, 16, 24, 32))
        assert matched.coarse.conv.in_channels == 5 * single.coarse.conv.in_channels
        assert matched.coarse.conv.in_channels < spanning.coarse.conv.in_channels      # the real lags also carry pose channels
        ranges, rewritten, poses = self._same_present_different_past()
        with torch.no_grad():
            assert torch.allclose(matched(ranges, poses)[0], matched(rewritten, poses)[0], atol=1e-5)


class TestLegacyCheckpoints:
    """Checkpoints trained before 2026-09-17 carry an 11-channel stem, whose third input was the validity channel, followed by a
    1x1 fine convolution. With one fine tap (`--fine-lags 0`: the no-fine and static arms) both are linear 1x1 convolutions with
    nothing between them, and on FROG's clamped input the validity channel was always 1, so the pair folds exactly into the new
    10-channel stem. More than one tap read past frames and cannot be folded."""

    def _legacy_state(self, taps):
        net = _small_network()
        torch.manual_seed(3)
        state = {k: v.clone() for k, v in net.state_dict().items()}
        c0 = state["stem.weight"].shape[0]
        state["stem.weight"] = torch.randn(c0, N_BEAM_FEATURES + 1, 1)
        state["stem.bias"] = torch.randn(c0)
        state["fine.conv.weight"] = torch.randn(c0, c0 * taps, 1)
        state["fine.conv.bias"] = torch.randn(c0)
        return net, state

    def test_a_one_tap_checkpoint_folds_exactly(self):
        net, state = self._legacy_state(taps=1)
        features = beam_features(_scans(1)[0][:, 0], _angles(48))                     # [1, 10, 48]
        legacy_input = torch.cat([features[:, :2], torch.ones_like(features[:, :1]), features[:, 2:]], dim=1)
        stem = torch.nn.functional.conv1d(legacy_input, state["stem.weight"], state["stem.bias"])
        want = torch.nn.functional.conv1d(stem, state["fine.conv.weight"], state["fine.conv.bias"])
        load_calibration_weights(net, state)
        with torch.no_grad():
            assert torch.allclose(net.stem(features), want, atol=1e-4)
        assert not any(k.startswith("fine.") for k in fold_legacy_calibration(state))

    def test_a_multi_tap_checkpoint_is_refused(self):
        net, state = self._legacy_state(taps=3)
        with pytest.raises(ValueError, match="fine"):
            load_calibration_weights(net, state)

    def test_a_current_checkpoint_is_untouched(self):
        state = _small_network().state_dict()
        assert fold_legacy_calibration(state) is state


class TestCalibrationDropout:
    """Dropout is opt-in (`TODO.md` A50 phase 1 item 4): off by default, on the head's input only, and parameter-free."""

    def _net(self, dropout):
        torch.manual_seed(0)
        return CalibrationNetwork(_angles(48), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1, dropout=dropout)

    def test_the_default_leaves_train_and_eval_identical(self):
        net, (ranges, poses) = self._net(0.0), _scans(6)
        with torch.no_grad():
            net.train()
            trained = net(ranges, poses)
            net.eval()
            evaluated = net(ranges, poses)
        for got, want in zip(trained, evaluated):
            assert torch.allclose(got, want, atol=1e-6)

    def test_it_perturbs_the_logits_in_train_mode_and_never_in_eval(self):
        net, (ranges, poses) = self._net(0.5), _scans(6)
        with torch.no_grad():
            net.train()
            first, second = net(ranges, poses)[0], net(ranges, poses)[0]
            net.eval()
            once, twice = net(ranges, poses)[0], net(ranges, poses)[0]
        assert not torch.allclose(first, second)
        assert torch.allclose(once, twice)

    def test_the_decoder_features_stage_3_pools_are_never_dropped(self):
        net, (ranges, poses) = self._net(0.5), _scans(6)
        net.train()
        with torch.no_grad():
            assert torch.allclose(net(ranges, poses)[2], net(ranges, poses)[2])

    def test_it_adds_no_parameters_so_existing_checkpoints_still_load(self):
        assert set(self._net(0.5).state_dict()) == set(self._net(0.0).state_dict())
        self._net(0.5).load_state_dict(_small_network().state_dict())


class TestRenderPrior:

    def test_live_slots_mark_their_sector_with_value_and_person_probability(self):
        angles = _angles(48)                        # 6 sectors of 30 degrees, from -90
        xy = torch.tensor([[[0.0, 2.0], [0.0, 3.0], [0.0, -2.0], [-2.0, 0.1], [0.0, 2.5]]])
        # ahead; ahead again (weaker value, stronger person); behind the sensor; far left; ahead but dead
        value = torch.tensor([[0.8, 0.3, 0.9, 0.5, 1.0]])
        person = torch.tensor([[0.6, 0.7, 0.9, 0.4, 1.0]])
        alive = torch.tensor([[True, True, True, True, False]])
        prior = render_prior(xy, value, person, alive, angles, sectors=6)
        assert prior.shape == (1, 2, 6)
        assert prior[0, 0].tolist() == pytest.approx([0, 0, 0, 0.8, 0, 0.5])      # the strongest slot per sector
        assert prior[0, 1].tolist() == pytest.approx([0, 0, 0, 0.7, 0, 0.4])


class TestCalibrationPrior:

    def _with_prior(self):
        torch.manual_seed(1)
        return CalibrationNetwork(_angles(48), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1, prior_channels=2).eval()

    def test_a_trained_network_loaded_with_prior_channels_is_unchanged_until_they_are_trained(self):
        base = _small_network()
        net = self._with_prior()
        load_calibration_weights(net, base.state_dict())
        ranges, poses = _scans(10)
        prior = torch.rand(1, 10, 2, 6)
        with torch.no_grad():
            for got, want in zip(net(ranges, poses, prior), base(ranges, poses)):
                assert torch.allclose(got, want, atol=1e-6)

    def test_streaming_with_a_prior_reproduces_the_clip(self):
        net = self._with_prior()
        with torch.no_grad():
            net.aggregate.weight[:, -2:].normal_()
        ranges, poses = _scans(20)
        prior = torch.rand(1, 20, 2, 6)
        state, streamed = None, []
        with torch.no_grad():
            for t in range(20):
                logit, votes, features, state = net.step(ranges[:, t], poses[:, t], state, prior[:, t])
                streamed.append((logit, votes, features))
            for t in (0, 9, 19):
                for got, want in zip(net(ranges[:, :t + 1], poses[:, :t + 1], prior[:, :t + 1]), streamed[t]):
                    assert torch.allclose(got, want, atol=1e-4), t
