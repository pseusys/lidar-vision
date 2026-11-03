"""one global recurrent state instead of 256 addressable slots -- the untested cell of the 2x2.

    slots + recurrence     the adopted model
    slots - recurrence     `--no-recurrent`, and it costs a lot
    recurrence - slots     **this**, never run

It is the strongest available attack on this project's own contribution: if one scene-level state matches 256
addressable per-object slots, the slots are decoration. That is exactly why it is worth running.

The arm is a fair like-for-like because the frozen memory only **rescores** candidates -- the coasting head was ablated
away -- so neither arm places a detection and both differ only in whether the memory is addressable. `manage_slots` is the
slot mechanism, so this arm does not run it at all: there is no association, no spawn, no eviction and no per-slot decay.

**Attention pooling, not a masked mean** (the owner's call, 2026-10-22): run the strongest variant first to estimate the
ceiling of a swap, because if the strongest global state cannot reach the frozen model's 86% then the weaker variants are
moot and the slots have earned their place.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))

from takheliped.detectors.takheliped import (  # noqa: E402
    CANDIDATE_FEATURES, SILENT_COAST_LOGIT, SLOT_STATISTICS, ObjectMemory, SlotRules,
)

DIM, HEADS, CANDIDATES = 32, 4, 6


def _memory(global_state: bool, recurrence: str = "mingru", **kwargs) -> ObjectMemory:
    torch.manual_seed(0)
    return ObjectMemory(SlotRules(capacity=16), dim=DIM, heads=HEADS, recurrence=recurrence,
                        global_state=global_state, exchange=False, learned_match=False, **kwargs)


def _inputs(batch: int = 2, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(batch, CANDIDATES, 2, generator=g) * 6 - 3
    score = torch.rand(batch, CANDIDATES, generator=g)
    valid = torch.ones(batch, CANDIDATES, dtype=torch.bool)
    pose = torch.zeros(batch, 3)
    return xy, score, valid, pose


def _run(model: ObjectMemory, steps: int = 4, batch: int = 2, dt: float = 0.025, seed: int = 1):
    """Play `steps` frames and return the last output."""
    xy, score, valid, pose = _inputs(batch, seed)
    slots = model.initial_state(batch)
    out = None
    for _ in range(steps):
        out = model.step(xy, score, valid, pose, torch.full((batch,), dt), slots)
        slots = out.slots
    return out


class TestTheDefaultIsUntouched:
    """Every published number comes from the addressable model, so the flag must be off and inert by default."""

    def test_addressable_slots_are_the_default(self):
        assert _memory(False).global_state is False
        assert ObjectMemory(SlotRules(capacity=8), dim=16, heads=2).global_state is False

    def test_a_model_built_without_the_flag_computes_exactly_what_it_did(self):
        torch.manual_seed(0)
        plain = ObjectMemory(SlotRules(capacity=16), dim=DIM, heads=HEADS)
        torch.manual_seed(0)
        same = ObjectMemory(SlotRules(capacity=16), dim=DIM, heads=HEADS, global_state=False)

        with torch.no_grad():
            a, b = _run(plain), _run(same)

        assert torch.allclose(a.candidate_logit, b.candidate_logit, atol=0)


class TestOneStateNotTwoHundredAndFiftySix:
    def test_the_state_is_one_vector_per_stream(self):
        slots = _memory(True).initial_state(3)

        assert slots.state.shape == (3, 1, DIM)

    def test_the_slot_capacity_is_forced_to_one_whatever_was_asked_for(self):
        """A capacity is a count of addressable slots, and this arm has none; honouring 256 would be a lie in the shape."""
        model = ObjectMemory(SlotRules(capacity=256), dim=DIM, heads=HEADS, global_state=True, recurrence="mingru")

        assert model.rules.capacity == 1

    def test_the_addressing_modules_are_gone_and_pooling_takes_their_place(self):
        model = _memory(True)

        assert model.match is None and model.exchange is None and model.recall is None and model.spawn is None
        assert model.pool is not None
        assert any("query" in name for name, _ in model.named_parameters())

    def test_it_holds_fewer_parameters_than_the_addressable_model(self):
        """Dropping the addressing is meant to be a simplification, not a reparameterisation."""
        globals_ = sum(p.numel() for p in _memory(True).parameters())
        slots = sum(p.numel() for p in _memory(False).parameters())

        assert globals_ < slots


class TestItBehavesLikeAMemory:
    def test_it_runs_and_keeps_the_interface(self):
        out = _run(_memory(True))

        assert out.candidate_logit.shape == (2, CANDIDATES)
        assert out.slots.state.shape == (2, 1, DIM)
        assert out.slot_xy.shape == (2, 1, 2)

    def test_an_untrained_global_memory_reproduces_the_detector_scores(self):
        """The rescoring head starts silent in this arm too, so an untrained run must not move the candidates."""
        model = _memory(True)
        xy, score, valid, pose = _inputs()

        with torch.no_grad():
            out = model.step(xy, score, valid, pose, torch.full((2,), 0.025), model.initial_state(2))

        assert torch.allclose(torch.sigmoid(out.candidate_logit), score, atol=1e-6)

    def test_no_slot_ever_reports_a_person_because_there_is_no_slot_to_report_one(self):
        out = _run(_memory(True))

        assert (out.slot_logit == SILENT_COAST_LOGIT).all()
        assert not out.coasting.any()

    def test_streams_in_a_batch_do_not_interact(self):
        model = _memory(True)
        xy, score, valid, pose = _inputs(batch=2)
        dt = torch.full((2,), 0.025)

        with torch.no_grad():
            together = model.step(xy, score, valid, pose, dt, model.initial_state(2))
            alone = model.step(xy[:1], score[:1], valid[:1], pose[:1], dt[:1], model.initial_state(1))

        assert torch.allclose(together.candidate_logit[:1], alone.candidate_logit, atol=1e-6)

    def test_gradients_reach_the_pooling_and_the_recurrence(self):
        model = _memory(True)
        with torch.no_grad():
            model.rescore_gate.fill_(1.0)
        out = _run(model, steps=3)

        out.candidate_logit.sum().backward()

        assert model.query.grad is not None and model.query.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.pool.parameters())


class TestWhatThePoolingMustRespect:
    def test_an_invalid_candidate_cannot_move_the_state(self):
        """Padding rows are everywhere in this pipeline -- `TOP_P` is 64 and a frame offers far fewer."""
        model = _memory(True)
        xy, score, valid, pose = _inputs()
        dt = torch.full((2,), 0.025)
        masked = valid.clone()
        masked[:, -2:] = False
        poisoned = xy.clone()
        poisoned[:, -2:] = 99.0

        with torch.no_grad():
            a = model.step(xy, score, masked, pose, dt, model.initial_state(2))
            b = model.step(poisoned, score, masked, pose, dt, model.initial_state(2))

        assert torch.allclose(a.slots.state, b.slots.state, atol=1e-6)

    def test_elapsed_time_reaches_the_state_even_though_mingru_cannot_read_it(self):
        """With no bookkeeping, the scene statistics are `dt`'s only route -- and the dropped-frame sweep showed that route matters."""
        model = _memory(True, recurrence="mingru")
        xy, score, valid, pose = _inputs()

        with torch.no_grad():
            quick = model.step(xy, score, valid, pose, torch.full((2,), 0.025), model.initial_state(2)).slots.state
            slow = model.step(xy, score, valid, pose, torch.full((2,), 0.500), model.initial_state(2)).slots.state

        assert not torch.allclose(quick, slow, atol=1e-6)

    def test_the_detector_s_own_candidate_features_reach_the_pooling(self):
        """The real arm is built with `feature_dim` from stage 2's pooled decoder features, never 0.

        A unit test at `feature_dim=0` left this branch unexercised and the real run died on the first frame with
        `cand_features` undefined (2026-10-22). The smoke caught it; this keeps it caught.
        """
        model = _memory(True, feature_dim=7)
        xy, score, valid, pose = _inputs()
        extra = torch.randn(2, CANDIDATES, 7)

        with torch.no_grad():
            a = model.step(xy, score, valid, pose, torch.full((2,), 0.025), model.initial_state(2), cand_features=extra)
            b = model.step(xy, score, valid, pose, torch.full((2,), 0.025), model.initial_state(2),
                           cand_features=torch.zeros_like(extra))

        assert model.encode[0].in_features == CANDIDATE_FEATURES + 7
        assert not torch.allclose(a.slots.state, b.slots.state, atol=1e-6)

    def test_the_scene_statistics_are_as_wide_as_the_slot_statistics_they_replace(self):
        """So `_recur` and every gate keep their trained input width and the recurrence family stays orthogonal."""
        model = _memory(True)

        assert model.statistics.in_features == SLOT_STATISTICS
        assert model.encode[0].in_features == CANDIDATE_FEATURES


class TestWhatItRefuses:
    def test_coasting_is_refused_because_a_global_state_has_no_position(self):
        with pytest.raises(ValueError, match="coast"):
            ObjectMemory(SlotRules(capacity=16), dim=DIM, heads=HEADS, global_state=True, coast=True)

    def test_visibility_is_refused_for_the_same_reason(self):
        with pytest.raises(ValueError, match="visibility"):
            ObjectMemory(SlotRules(capacity=16), dim=DIM, heads=HEADS, global_state=True, visibility=True)

    def test_dropping_the_recurrence_too_would_leave_nothing_and_is_refused(self):
        with pytest.raises(ValueError, match="recurrent"):
            ObjectMemory(SlotRules(capacity=16), dim=DIM, heads=HEADS, global_state=True, recurrent=False)

    def test_every_recurrence_still_works_so_the_two_questions_stay_separable(self):
        for name in ("selective", "gru", "mingru"):
            out = _run(_memory(True, recurrence=name))

            assert out.candidate_logit.shape == (2, CANDIDATES), name
            assert torch.isfinite(out.slots.state).all(), name
