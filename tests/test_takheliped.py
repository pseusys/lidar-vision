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

from takheliped.detectors.takheliped import (
    N_BEAM_FEATURES, CalibrationNetwork, MetricGather, ObjectMemory, SlotRules, beam_features,
    fold_legacy_calibration, gather_beams, metric_offsets,
    free_range_profile, initial_slots, lfe_features, load_calibration_weights, manage_slots, render_prior, sanitize_ranges,
    sensor_to_world, visibility_feature, world_to_sensor,
)
from takheliped.detectors.lfe_detector import _normalize_scan


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

    def test_a_step_says_which_slot_backs_each_candidate(self):
        """`TODO.md` item 14 needs the association observable: a rescored candidate's score rests on the slot matched to
        it, so diagnosing the memory's phantoms means reading that slot's age -- and item 27 established that 99.997% of
        the person-free phantoms are rescored candidates rather than coasting reports, so every one of them has one."""
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4)

        for out, _, valid in self._run(model, 8):
            assert out.events.match_index.shape == out.slots.alive.shape
            matched = out.events.matched
            assert (out.events.match_index[matched] >= 0).all()
            assert (out.events.match_index[matched] < valid.shape[1]).all()
            assert not matched[~out.slots.alive].any()      # a dead slot matches nothing

    def test_the_slot_backing_a_candidate_carries_its_age(self):
        """Age is the quantity a creation or confirmation rule is written against."""
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4)

        outs = self._run(model, 10)
        out = outs[-1][0]
        backed = out.events.matched & out.slots.alive

        assert out.slots.age_s[backed].min() >= 0.0
        assert out.slots.age_s[backed].max() > 0.0          # by frame 10 some slot has aged

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

    def test_without_the_scalar_gate_rescoring_still_starts_silent(self):
        # T1 (`TODO.md` A51): the same "untrained stage 3 is exactly stage 2" property, from a zero-initialised last layer
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, rescore_gate=False)
        assert model.rescore_gate is None
        for out, score, valid in self._run(model, 8):
            assert torch.allclose(torch.sigmoid(out.candidate_logit)[valid], score[valid], atol=1e-5)

    def test_the_scalar_gate_starves_the_rescoring_head_of_gradient_and_dropping_it_does_not(self):
        """Why T1 exists: with `delta = gate * mlp(...)` and `gate` starting at zero, the head's own gradient is scaled by zero, so
        the run's first steps train the scalar alone -- it stayed at +0.03 to +0.05 through step 3a."""
        def head_gradient(**kwargs):
            torch.manual_seed(0)
            model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, **kwargs)
            sum(out.candidate_logit[valid].sum() for out, _, valid in self._run(model, 3)).backward()
            return sum(p.grad.abs().sum().item() for p in model.rescore.parameters())

        assert head_gradient() == pytest.approx(0.0, abs=1e-9)
        assert head_gradient(rescore_gate=False) > 0.0

    def test_a_rescoring_limit_bounds_the_correction_both_ways(self):
        # O1 (`TODO.md` A51): the signed correction is what shifted the whole score distribution down
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, rescore_gate=False, rescore_limit=3.0)
        for out, score, valid in self._run(model, 4):
            assert torch.allclose(torch.sigmoid(out.candidate_logit)[valid], score[valid], atol=1e-5)
        with torch.no_grad():                                   # a trained head, with weights far larger than it would reach
            for p in model.rescore.parameters():
                p.copy_(torch.randn_like(p) * 20.0)
        for out, score, valid in self._run(model, 4):
            shift = (out.candidate_logit - torch.logit(score.clamp(1e-6, 1 - 1e-6)))[valid]
            assert shift.abs().max() <= 3.0 + 1e-5
            assert shift.abs().max() > 1.0                      # and the limit binds rather than silencing the head

    def test_in_sensor_coordinates_a_static_object_drifts_as_the_robot_moves(self):
        """D1 (`TODO.md` A51 phase 3b): room coordinates are the design's central claim -- a chair keeps zero velocity because the
        slot stays put while the robot moves. The ablation keeps slots in the sensor frame, where the same chair appears to move."""
        # the robot drives 0.2 m forward in 0.1 s, so the same static object is 0.2 m nearer in its own frame -- small enough that
        # association still holds either way, which is what isolates the frame from the gate
        score, valid = torch.full((1, 1), 0.9), torch.ones(1, 1, dtype=torch.bool)
        frames = ((torch.tensor([[[0.0, 4.0]]]), torch.zeros(1, 3)), (torch.tensor([[[0.0, 3.8]]]), torch.tensor([[0.2, 0.0, 0.0]])))
        out = {}
        for name, room in (("room", True), ("sensor", False)):
            torch.manual_seed(0)
            model = ObjectMemory(SlotRules(capacity=8), dim=32, heads=4, room_frame=room)
            slots = model.initial_state(1)
            for xy, pose in frames:
                slots = model.step(xy, score, valid, pose, torch.full((1,), 0.1), slots).slots
            out[name] = slots.velocity.norm(dim=-1).max().item()
        assert out["room"] == pytest.approx(0.0, abs=1e-6)    # the object stands still in the room
        assert out["sensor"] > 0.1                            # and appears to move in the sensor frame

    def test_the_latest_score_rule_replaces_the_accumulated_value(self):
        """D2: a slot is held by accumulated value, so a person hidden for a moment keeps their slot; the ablation ranks on the
        latest score instead."""
        xy, score, valid = torch.tensor([[[0.0, 4.0]]]), torch.full((1, 1), 0.8), torch.ones(1, 1, dtype=torch.bool)
        values = {}
        for name, latest in (("accumulate", False), ("latest", True)):
            torch.manual_seed(0)
            model = ObjectMemory(SlotRules(capacity=8, latest_value=latest), dim=32, heads=4)
            slots = model.initial_state(1)
            for _ in range(3):
                slots = model.step(xy, score, valid, torch.zeros(1, 3), torch.full((1,), 0.04), slots).slots
            values[name] = slots.value.max().item()
        # a slot is created at the candidate's score either way; from then on the rule decides
        assert values["latest"] == pytest.approx(0.8, abs=1e-5)
        assert values["accumulate"] > 0.8                    # it climbs towards certainty rather than copying the score

    def test_a_fixed_decay_removes_the_learned_one(self):
        """D3: the selective recurrence decays each slot at a learned, input-dependent rate; the ablation gives every slot one
        fixed time constant, which is verified by the parameters it removes."""
        torch.manual_seed(0)
        learned = ObjectMemory(SlotRules(capacity=8), dim=32, heads=4)
        torch.manual_seed(0)
        fixed = ObjectMemory(SlotRules(capacity=8), dim=32, heads=4, fixed_decay_s=8.0)
        assert fixed.decay is None and learned.decay is not None
        removed = sum(p.numel() for p in learned.decay.parameters()) + learned.decay_bias.numel()
        assert sum(p.numel() for p in learned.parameters()) - sum(p.numel() for p in fixed.parameters()) == removed
        for out, score, valid in TestObjectMemory()._run(fixed, 6):
            assert torch.allclose(torch.sigmoid(out.candidate_logit)[valid], score[valid], atol=1e-5)

    def test_learning_the_value_rule_gives_its_constants_a_gradient(self):
        """D4 (owner's question 2026-09-20): the bookkeeping is a fixed rule under `no_grad`, so the loss cannot shape it. With
        `learn_value` the two constants of the value rule become parameters that the loss can reach."""
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=8), dim=32, heads=4, learn_value=True, rescore_gate=False)
        names = dict(model.named_parameters())
        assert "value_gain" in names and "value_fade" in names
        with torch.no_grad():                                # a trained rescoring head: at init its last layer is zero, which stops
            model.rescore[-1].weight.normal_(0.0, 0.1)       # every gradient from reaching what the head reads, value included
        # one candidate held still, then absent for a frame, then back: the slot is created, matched (gain), faded (fade), matched
        here, gone = torch.tensor([[[0.0, 4.0]]]), torch.tensor([[[0.0, 4.0]]])
        present, absent = torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.bool)
        slots, loss = model.initial_state(1), 0.0
        for xy, valid in ((here, present), (here, present), (gone, absent), (here, present)):
            out = model.step(xy, torch.full((1, 1), 0.8), valid, torch.zeros(1, 3), torch.full((1,), 0.2), slots)
            slots, loss = out.slots, loss + out.candidate_logit.sum()
        loss.backward()
        assert names["value_gain"].grad is not None and names["value_gain"].grad.abs().sum() > 0
        assert names["value_fade"].grad is not None and names["value_fade"].grad.abs().sum() > 0

    def test_the_value_rule_is_a_fixed_rule_by_default(self):
        model = ObjectMemory(SlotRules(capacity=8), dim=32, heads=4)
        assert not any(name.startswith("value_") for name, _ in model.named_parameters())

    def test_without_the_exchange_its_parameters_are_gone_and_rescoring_still_starts_silent(self):
        """E6 (`TODO.md` A51 phase 5): the slot-to-slot self-attention is one of the three learned mechanisms, has never been
        ablated, and is the design's only O(K^2) term."""
        torch.manual_seed(0)
        full = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4)
        torch.manual_seed(0)
        without = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, exchange=False)
        assert without.exchange is None and without.exchange_norm is None
        removed = sum(p.numel() for p in full.exchange.parameters()) + sum(p.numel() for p in full.exchange_norm.parameters())
        assert sum(p.numel() for p in full.parameters()) - sum(p.numel() for p in without.parameters()) == removed
        for out, score, valid in self._run(without, 6):
            assert torch.allclose(torch.sigmoid(out.candidate_logit)[valid], score[valid], atol=1e-5)

    def test_without_learned_matching_the_bookkeeping_still_carries_candidates_into_the_slots(self):
        """E7: with the cross-attention gone, a slot reads only the candidate the hand-coded association gave it. The point is that
        candidate information must still reach the slot state, or the ablation would be testing something else."""
        torch.manual_seed(0)
        full = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, feature_dim=6)
        torch.manual_seed(0)
        without = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, feature_dim=6, learned_match=False)
        assert without.match is None
        removed = sum(p.numel() for p in full.match.parameters())
        assert sum(p.numel() for p in full.parameters()) - sum(p.numel() for p in without.parameters()) == removed

        g = torch.Generator().manual_seed(5)
        xy, score, valid = _memory_inputs(g)
        pose, dt = torch.zeros(2, 3), torch.full((2,), 0.04)
        features = torch.randn(2, 6, 6, generator=g)
        first = without.step(xy, score, valid, pose, dt, without.initial_state(2), cand_features=features)
        second = without.step(xy, score, valid, pose, dt, without.initial_state(2), cand_features=features + 2.0)
        assert not torch.allclose(first.slots.state, second.slots.state)

    def test_without_recurrence_only_the_bookkeeping_reaches_the_rescoring(self):
        """E10 (`TODO.md` A51): with no slot state, the head sees a candidate and the statistics of the slot the hand-coded
        association gave it -- value, age, time since match, displacement, speed, hit rate. If that matches the full model, the
        memory is a tracker with a learned scorer rather than a learned memory."""
        torch.manual_seed(0)
        plain = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, recurrent=False)
        for name in ("spawn", "match", "exchange", "exchange_norm", "update", "decay", "recall"):
            assert getattr(plain, name) is None, name
        assert plain.decay_bias is None
        torch.manual_seed(0)
        full = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4)
        assert sum(p.numel() for p in plain.parameters()) < sum(p.numel() for p in full.parameters()) / 2
        for out, score, valid in self._run(plain, 6):
            assert torch.allclose(torch.sigmoid(out.candidate_logit)[valid], score[valid], atol=1e-5)

    def test_without_recurrence_the_slot_statistics_still_move_the_scores(self):
        """The ablation must still be a memory of something: a slot that has been tracked for a while has to score differently
        from one just created, or the arm would be testing an empty head."""
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, recurrent=False)
        with torch.no_grad():                                  # a trained head, so the gate is not what silences it
            model.rescore[-1].weight.normal_(0.0, 0.5)
            model.rescore_gate.fill_(1.0)
        xy, score = torch.tensor([[[0.0, 4.0]]]), torch.full((1, 1), 0.8)
        valid, pose, dt = torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 3), torch.full((1,), 0.2)
        slots = model.initial_state(1)
        first = model.step(xy, score, valid, pose, dt, slots)   # the slot is created this frame
        slots = first.slots
        for _ in range(8):                                     # the same person, tracked for eight more frames
            out = model.step(xy, score, valid, pose, dt, slots)
            slots = out.slots
        assert not torch.allclose(first.candidate_logit, out.candidate_logit, atol=1e-4)

    def test_decay_times_start_spread_over_the_short_and_long_horizons(self):
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, decay_init_s=(1.0, 60.0))
        times = 1.0 / torch.nn.functional.softplus(model.decay_bias)
        assert times.min().item() == pytest.approx(1.0, rel=1e-3)
        assert times.max().item() == pytest.approx(60.0, rel=1e-3)


# ---------------------------------------------------------------- stages 1-2: the calibration network

def _small_network():
    torch.manual_seed(0)
    return CalibrationNetwork(_angles(48), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1).eval()


def _scans(frames, n=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(frames, n, generator=g) * 4 + 1


class TestCalibrationNetwork:

    def test_readings_beyond_the_range_limit_are_the_same_as_no_return(self):
        """`max_range_m` is the model's configurable range: past it, a real return and a missing one are indistinguishable."""
        ranges = _scans(3)
        far, missing = ranges.clone(), ranges.clone()
        far[:, 10] = 14.0
        missing[:, 10] = float("inf")
        with torch.no_grad():
            short = _small_network()
            assert torch.allclose(short(far)[0], short(missing)[0])
            torch.manual_seed(0)
            long = CalibrationNetwork(_angles(48), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1, max_range_m=20.0).eval()
            assert not torch.allclose(long(far)[0], long(missing)[0])

    def test_frames_in_a_batch_do_not_see_each_other(self):
        ranges = _scans(4)
        with torch.no_grad():
            together = _small_network()(ranges)
            alone = [_small_network()(ranges[i:i + 1]) for i in range(4)]
        for i, single in enumerate(alone):
            for got, want in zip(single, together, strict=True):
                assert torch.allclose(got[0], want[i], atol=1e-5)


# ---------------------------------------------------------------- feedback: memory drawn into stage 2's bottleneck

class TestStage2WithoutTime:
    """The coarse temporal convolution was removed 2026-10-08 (`TODO.md` A60: end to end it lost 0.87 pp). Every stage-3 result
    rests on no-temporal checkpoints, whose one-lag `coarse.conv` was a plain 3-tap convolution across bottleneck sectors; it is now
    `context`, and such a checkpoint must load into today's network and compute exactly what it did. The fixture was written by the
    code at commit e76aa94 (`tests/fixtures/make_no_temporal_stage2.py`). A checkpoint whose convolution read past frames cannot be
    carried over, and the error says where the code that can still run it lives."""

    FIXTURE = Path(__file__).parent / "fixtures" / "no_temporal_stage2.pt"

    def _net(self, saved, prior_channels):
        net = CalibrationNetwork(saved["angles"], channels=(4, 8, 16), blocks=(1, 1, 1), bottleneck_blocks=1, expansion=2, prior_channels=prior_channels).eval()
        load_calibration_weights(net, saved["plain" if not prior_channels else "with_prior"]["state"])
        return net

    def test_a_no_temporal_checkpoint_from_before_the_removal_computes_exactly_what_it_did(self):
        saved = torch.load(self.FIXTURE, weights_only=False)
        with torch.no_grad():
            got = self._net(saved, 0)(saved["ranges"])
        for g, want in zip(got, (saved["plain"]["logit"], saved["plain"]["votes"], saved["plain"]["features"]), strict=True):
            assert torch.equal(g, want)

    def test_so_does_one_with_a_trained_memory_prior(self):
        saved = torch.load(self.FIXTURE, weights_only=False)
        with torch.no_grad():
            got = self._net(saved, 2)(saved["ranges"], saved["prior"])
        for g, want in zip(got, (saved["with_prior"]["logit"], saved["with_prior"]["votes"], saved["with_prior"]["features"]), strict=True):
            assert torch.equal(g, want)

    def test_a_checkpoint_whose_convolution_read_past_frames_is_refused_with_where_to_run_it(self):
        state = {k: v.clone() for k, v in _small_network().state_dict().items()}
        c2 = state["context.weight"].shape[0]
        del state["context.weight"], state["context.bias"]
        state["coarse.conv.weight"], state["coarse.conv.bias"] = torch.randn(c2, 5 * c2 + 4 * 3, 3), torch.randn(c2)
        with pytest.raises(ValueError, match="e76aa94"):
            load_calibration_weights(_small_network(), state)

    def test_one_scan_in_needs_no_pose_and_no_state(self):
        logit, votes, features = _small_network()(_scans(2))
        assert logit.shape == (2, 48) and votes.shape == (2, 48, 2) and features.shape == (2, 8, 48)


class TestAttributionArms:
    """Two of the arms that test why stage 2 beats LFE (`TODO.md` A59, F1 and W): LFE's input alone, and LFE's widths. Each is a
    constructor option; the default network is pinned by the fixture above."""

    def test_the_lfe_input_is_lfe_s_own_normalisation(self):
        ranges = torch.tensor([[0.05, 0.2, 0.7, 4.0, 9.99, 10.0, 61.0, math.inf]])
        got = lfe_features(ranges, 10.0)
        assert got.shape == (1, 1, 8)
        assert np.allclose(got[:, 0].numpy(), _normalize_scan(ranges.numpy()), atol=1e-7)

    def test_a_missing_return_reads_as_far_whatever_its_encoding(self):
        assert torch.equal(lfe_features(torch.tensor([0.0, -1.0, math.nan, math.inf, 61.0]), 10.0), torch.zeros(1, 5))

    def test_the_lfe_input_network_reads_one_channel(self):
        net = CalibrationNetwork(_angles(48), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1, input_features="lfe").eval()
        assert net.stem.in_channels == 1 and _small_network().stem.in_channels == N_BEAM_FEATURES
        logit, votes, _ = net(_scans(2))
        assert logit.shape == (2, 48) and votes.shape == (2, 48, 2)

    def test_an_unknown_input_is_refused(self):
        with pytest.raises(ValueError, match="input_features"):
            CalibrationNetwork(_angles(48), input_features="raw")

    def test_lfe_s_widths_give_the_parameter_count_the_arm_was_planned_at(self):
        assert sum(p.numel() for p in CalibrationNetwork(_angles(720)).parameters()) == 976_707
        assert sum(p.numel() for p in CalibrationNetwork(_angles(720), channels=(8, 16, 32)).parameters()) == 63_891


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
        features = beam_features(_scans(1), _angles(48))                              # [1, 10, 48]
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
        net, ranges = self._net(0.0), _scans(6)
        with torch.no_grad():
            net.train()
            trained = net(ranges)
            net.eval()
            evaluated = net(ranges)
        for got, want in zip(trained, evaluated):
            assert torch.allclose(got, want, atol=1e-6)

    def test_it_perturbs_the_logits_in_train_mode_and_never_in_eval(self):
        net, ranges = self._net(0.5), _scans(6)
        with torch.no_grad():
            net.train()
            first, second = net(ranges)[0], net(ranges)[0]
            net.eval()
            once, twice = net(ranges)[0], net(ranges)[0]
        assert not torch.allclose(first, second)
        assert torch.allclose(once, twice)

    def test_the_decoder_features_stage_3_pools_are_never_dropped(self):
        net, ranges = self._net(0.5), _scans(6)
        net.train()
        with torch.no_grad():
            assert torch.allclose(net(ranges)[2], net(ranges)[2])

    def test_it_adds_no_parameters_so_existing_checkpoints_still_load(self):
        assert set(self._net(0.5).state_dict()) == set(self._net(0.0).state_dict())
        self._net(0.5).load_state_dict(_small_network().state_dict())


class TestFreeRangeProfile:
    """S2: the nearest thing the laser actually saw, per angular sector -- the evidence a coasting slot is or is not occluded."""

    def test_it_keeps_the_nearest_return_in_each_sector(self):
        ranges = torch.tensor([[3.0, 2.0, 5.0, 4.0, 9.0, 8.0]])           # three sectors of two beams
        assert free_range_profile(ranges, sectors=3, max_range_m=30.0)[0].tolist() == pytest.approx([2.0, 4.0, 8.0])

    def test_a_missing_return_counts_as_nothing_there_rather_than_as_an_obstacle(self):
        # FROG writes +inf and 61.0 m for "no return"; read literally, a minimum would make them the nearest obstacle
        ranges = torch.tensor([[float("inf"), 61.0, 0.0, -1.0]])
        assert free_range_profile(ranges, sectors=2, max_range_m=30.0)[0].tolist() == pytest.approx([30.0, 30.0])

    def test_one_real_return_beside_missing_ones_still_wins_its_sector(self):
        ranges = torch.tensor([[float("inf"), 4.0, 61.0, 61.0]])
        assert free_range_profile(ranges, sectors=2, max_range_m=30.0)[0].tolist() == pytest.approx([4.0, 30.0])

    def test_every_sector_is_covered_once(self):
        ranges = torch.arange(1, 13, dtype=torch.float32)[None]
        assert free_range_profile(ranges, sectors=4, max_range_m=30.0).shape == (1, 4)


class TestVisibilityFeature:
    """S2: is a coasting slot hidden behind something, or standing in space the laser saw straight through?"""

    def test_a_slot_behind_a_nearer_return_reads_as_occluded(self):
        # slot 5 m straight ahead, something at 2 m in the same sector: the person could plausibly still be there
        free = torch.full((1, 6), 30.0)
        free[0, 3] = 2.0
        assert visibility_feature(torch.tensor([[[0.0, 5.0]]]), free, _angles(48), sectors=6, scale=2.0)[0, 0].item() < 0

    def test_a_slot_the_laser_saw_past_reads_as_free_space(self):
        # the beam reached 9 m through the slot's own position, so nobody is standing there
        assert visibility_feature(torch.tensor([[[0.0, 5.0]]]), torch.full((1, 6), 9.0), _angles(48), sectors=6, scale=2.0)[0, 0].item() > 0

    def test_it_saturates_rather_than_growing_without_bound(self):
        near = visibility_feature(torch.tensor([[[0.0, 5.0]]]), torch.full((1, 6), 0.01), _angles(48), 6, 2.0)
        far = visibility_feature(torch.tensor([[[0.0, 5.0]]]), torch.full((1, 6), 300.0), _angles(48), 6, 2.0)
        assert near[0, 0].item() == pytest.approx(-1.0) and far[0, 0].item() == pytest.approx(1.0)

    def test_a_slot_outside_the_field_of_view_reads_as_nothing_known(self):
        behind = torch.tensor([[[0.0, -4.0]]])
        assert visibility_feature(behind, torch.full((1, 6), 30.0), _angles(48), sectors=6, scale=2.0)[0, 0].item() == 0.0


class TestMemoryWithVisibility:
    """S2: the feature reaches the heads, and switching it off leaves the model exactly as it was."""

    def _model(self, **kw):
        torch.manual_seed(0)
        return ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, **kw)

    def test_the_statistics_vector_is_one_wider_and_the_heads_follow(self):
        # The coasting head reads the statistics, so it is the probe -- opt in, since it left the architecture in item 24.
        on, off = self._model(visibility=True, coast=True), self._model(coast=True)
        assert on.coast.in_features == off.coast.in_features + 1
        assert on.statistics.in_features == off.statistics.in_features + 1

    def test_it_refuses_to_run_without_the_free_range_it_needs(self):
        model = self._model(visibility=True)
        with pytest.raises(ValueError, match="free_range"):
            TestObjectMemory()._run(model, 1)

    def test_the_default_model_is_untouched(self):
        # the flag must not perturb any existing checkpoint's shape or numbers
        off = self._model(coast=True)
        assert off.coast.in_features == 32 + 8
        for out, score, valid in TestObjectMemory()._run(off, 4):
            assert torch.allclose(torch.sigmoid(out.candidate_logit)[valid], score[valid], atol=1e-5)


class TestRescoringOff:
    """A51 E12a: with the correction gone, the memory's only route to the loss is the prior it draws into stage 2."""

    def _model(self, **kw):
        torch.manual_seed(0)
        return ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, **kw)

    def test_the_candidate_keeps_its_own_logit_exactly(self):
        # not "close to": with no correction term there is nothing to add, so the equality is exact bar the logit round trip
        model = self._model(rescore=False)
        for out, score, valid in TestObjectMemory()._run(model, 8):
            assert torch.allclose(torch.sigmoid(out.candidate_logit)[valid], score[valid], atol=1e-6)

    def test_it_holds_no_rescoring_parameters(self):
        off, on = self._model(rescore=False), self._model()
        assert off.rescore is None and off.rescore_gate is None
        assert not [name for name, _ in off.named_parameters() if name.startswith("rescore")]
        assert [name for name, _ in on.named_parameters() if name.startswith("rescore")]
        assert sum(p.numel() for p in off.parameters()) < sum(p.numel() for p in on.parameters())

    def test_the_slots_and_the_coasting_head_are_untouched(self):
        # only the rescoring path goes; everything the memory is for still has to run
        off = self._model(rescore=False)
        outs = TestObjectMemory()._run(off, 8)
        assert any(out.slots.alive.any() for out, _, _ in outs)
        assert all(torch.isfinite(out.slot_logit).all() for out, _, _ in outs)

    def test_a_gate_is_never_created_even_when_one_is_asked_for(self):
        assert self._model(rescore=False, rescore_gate=True).rescore_gate is None


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
        ranges = _scans(10)
        prior = torch.rand(10, 2, 6)
        with torch.no_grad():
            for got, want in zip(net(ranges, prior), base(ranges), strict=True):
                assert torch.allclose(got, want, atol=1e-6)


class TestCoastingAblation:
    """Item 24: does the coasting head earn its parameters?

    It is the mechanism that lets a slot report a person no candidate covers, and the measurements say it almost never
    does: **0.0012 / 0.0000 / 0.0020 times per scored frame** on the populated split -- one seed emits nothing at all --
    and item 27 found **2 / 0 / 0** reports across all 72,533 person-free frames at the operating threshold, with zero at
    0.5 and 0.9. A paper should not describe a mechanism that never acts, so either removing it costs nothing, which is a
    result, or it costs something, which is the defence.

    Ablating it means removing the head's parameters *and* the reports, not merely zeroing its loss: a weight of zero
    stops it training while leaving it free to fire at inference.
    """

    def test_the_ablated_memory_does_not_carry_the_heads_parameters(self):
        torch.manual_seed(0)
        with_head = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, coast=True)
        without = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4)

        assert sum(p.numel() for p in without.parameters()) < sum(p.numel() for p in with_head.parameters())
        assert not any("coast" in name for name, _ in without.named_parameters())

    def test_no_slot_ever_reports_a_person_no_candidate_covers(self):
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, coast=False)

        for out, _, _ in TestObjectMemory()._run(model, 12):
            assert not out.coasting.any()

    def test_the_slot_logit_stays_silent_so_nothing_could_be_reported(self):
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, coast=False)

        for out, _, _ in TestObjectMemory()._run(model, 8):
            assert (torch.sigmoid(out.slot_logit) < 1e-3).all()

    def test_rescoring_is_untouched_by_the_ablation(self):
        """The ablation must remove the coasting head and nothing else -- the rescoring is the memory's other output."""
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, coast=False)

        for out, score, valid in TestObjectMemory()._run(model, 12):
            assert torch.allclose(torch.sigmoid(out.candidate_logit)[valid], score[valid], atol=1e-5)

    def test_asking_for_the_head_restores_it(self):
        """Removed from the architecture, kept reachable: every number published before 2026-10-16 was produced with it."""
        torch.manual_seed(0)
        model = ObjectMemory(SlotRules(capacity=16), dim=32, heads=4, coast=True)

        assert any("coast" in name for name, _ in model.named_parameters())


class TestMetricOffsets:
    """Item 15: a first layer that reads a fixed width **in metres** instead of a fixed number of beams.

    A person 0.8 m wide subtends 42 beams at 2 m and 9 at 9 m, so a fixed 7-beam kernel is correct at exactly one range
    and wrong everywhere else -- which is what Tier 0's 1-NN probe measured when it priced the adaptive angular window at
    **+2.5 pp**, with the advantage **purely long-range**: +4.9 pp at 8-10 m, zero at 4 m. DROW gets this for free because
    its cutout is resampled per beam; a full-scan network has to be given it.

    `A59`'s F1 arm shifted the prior against input-*feature* changes -- swapping our ten channels for LFE's measured
    nothing. This is not that: it changes which beams the first layer reads, not what each beam says.
    """

    STEP = np.pi / 720          # FROG's angular increment, one beam

    def test_the_window_is_a_fixed_width_in_metres_so_it_shrinks_with_range(self):
        near = metric_offsets(torch.tensor([[2.0]]), self.STEP, width_m=1.66, samples=9)
        far = metric_offsets(torch.tensor([[8.0]]), self.STEP, width_m=1.66, samples=9)

        assert near[0, 0].abs().max() > 3 * far[0, 0].abs().max()

    def test_the_span_matches_the_angle_the_width_subtends(self):
        """Half-width 0.83 m at 4 m subtends atan(0.83/4) radians, which is that many beams at this increment."""
        offsets = metric_offsets(torch.tensor([[4.0]]), self.STEP, width_m=1.66, samples=11)

        assert offsets[0, 0].max().item() == pytest.approx(np.arctan(0.83 / 4.0) / self.STEP, rel=1e-5)
        assert offsets[0, 0].min().item() == pytest.approx(-np.arctan(0.83 / 4.0) / self.STEP, rel=1e-5)

    def test_the_offsets_are_centred_and_evenly_spaced(self):
        offsets = metric_offsets(torch.tensor([[3.0]]), self.STEP, width_m=1.66, samples=7)[0, 0]

        assert offsets[len(offsets) // 2].item() == pytest.approx(0.0, abs=1e-6)
        assert torch.allclose(torch.diff(offsets), torch.diff(offsets)[0].expand(len(offsets) - 1), atol=1e-6)

    def test_one_sample_reads_the_beam_itself(self):
        assert metric_offsets(torch.tensor([[5.0]]), self.STEP, width_m=1.66, samples=1)[0, 0].item() == pytest.approx(0.0)


class TestGatherBeams:
    def test_integer_offsets_read_their_beams_exactly(self):
        x = torch.arange(5, dtype=torch.float32).view(1, 1, 5)
        offsets = torch.tensor([[[-1.0, 0.0, 1.0]] * 5])

        out = gather_beams(x, offsets)

        assert out[0, 0, 2].tolist() == [1.0, 2.0, 3.0]

    def test_a_fractional_offset_interpolates_between_neighbours(self):
        x = torch.tensor([[[0.0, 10.0, 20.0]]])
        offsets = torch.tensor([[[0.0], [0.5], [0.0]]])

        out = gather_beams(x, offsets)

        assert out[0, 0, 1, 0].item() == pytest.approx(15.0)

    def test_the_scan_edges_clamp_rather_than_wrap(self):
        """Beam 0 has no left neighbour; reading past the edge must repeat the edge, not fetch the far side of the scan."""
        x = torch.tensor([[[7.0, 1.0, 2.0, 99.0]]])
        offsets = torch.tensor([[[-3.0], [0.0], [0.0], [3.0]]])

        out = gather_beams(x, offsets)

        assert out[0, 0, 0, 0].item() == 7.0
        assert out[0, 0, 3, 0].item() == 99.0

    def test_the_gathered_axis_is_the_sample_axis(self):
        out = gather_beams(torch.zeros(2, 4, 16), torch.zeros(2, 16, 5))

        assert out.shape == (2, 4, 16, 5)


class TestMetricGather:
    def test_it_returns_the_shape_it_was_given(self):
        gather = MetricGather(channels=8, width_m=1.66, samples=9, step_rad=np.pi / 720)

        out = gather(torch.randn(2, 8, 64), torch.full((2, 64), 4.0))

        assert out.shape == (2, 8, 64)

    def test_gradients_reach_it(self):
        gather = MetricGather(channels=8, width_m=1.66, samples=9, step_rad=np.pi / 720)
        x = torch.randn(1, 8, 32, requires_grad=True)

        gather(x, torch.full((1, 32), 4.0)).sum().backward()

        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_two_ranges_read_different_neighbourhoods(self):
        """The whole point: the same beam pattern seen at 2 m and at 9 m is read through different windows."""
        torch.manual_seed(0)
        gather = MetricGather(channels=4, width_m=1.66, samples=9, step_rad=np.pi / 720)
        x = torch.randn(1, 4, 128)

        near = gather(x, torch.full((1, 128), 2.0))
        far = gather(x, torch.full((1, 128), 9.0))

        assert not torch.allclose(near, far, atol=1e-4)


class TestTheNetworkKeepsItsDefault:
    def test_zero_width_leaves_the_network_exactly_as_it_was(self):
        """Off by default, and bit-identical when off -- every trained checkpoint must keep reading the same."""
        torch.manual_seed(0)
        angles = torch.linspace(-np.pi / 2, np.pi / 2, 128)
        plain = CalibrationNetwork(angles, channels=(8, 16, 32), blocks=(1, 1, 1))
        torch.manual_seed(0)
        same = CalibrationNetwork(angles, channels=(8, 16, 32), blocks=(1, 1, 1), gather_width_m=0.0)
        scan = torch.rand(2, 128) * 8 + 1

        with torch.no_grad():
            a, b = plain(scan), same(scan)

        assert torch.allclose(a[0], b[0], atol=0) and torch.allclose(a[1], b[1], atol=0)

    def test_a_width_adds_the_gather_and_still_runs(self):
        torch.manual_seed(0)
        angles = torch.linspace(-np.pi / 2, np.pi / 2, 128)
        net = CalibrationNetwork(angles, channels=(8, 16, 32), blocks=(1, 1, 1), gather_width_m=1.66)

        logits, votes, features = net(torch.rand(2, 128) * 8 + 1)

        assert logits.shape == (2, 128) and votes.shape == (2, 128, 2)
        assert any("gather" in name for name, _ in net.named_parameters())
