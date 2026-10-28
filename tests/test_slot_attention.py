"""Item 42: Slot Attention's own update (Locatello et al., NeurIPS 2020) in place of the hand-coded state update.

`docs/RESEARCH.md` credits Slot Attention for the **abstraction** -- a scene as a set of addressable object states -- and
explicitly not the mechanism, because the association here is a hand-coded one-to-one metric match. Item 36 then proved
the abstraction is what carries the gain, and item 38 established that a citable cell beats a re-implemented one. This is
the conclusion of that logic: if the published update also works, the architecture becomes "Slot Attention over LiDAR
candidates", which is a far easier claim to defend than a bespoke pipeline.

**What it replaces, exactly, because the row could be read more broadly.** `manage_slots` still runs. It has to: it is
what gives a slot its room-frame position, and the room frame is the choice the whole design rests on (`TODO.md` item 36,
`docs/RESEARCH.md` 5.3). What the iterated attention replaces is the association's role in the **state update** -- the
`match_index` gather, or `self.match` where that is on -- and the separate `recall` attention, because Slot Attention's
own attention matrix is already a distribution over slots per candidate and so is the recall.

**The proximity bias stays, and that is a deliberate deviation from the paper.** Published Slot Attention has no spatial
prior; here `_proximity` biases every candidate-slot pair by `-|dx|^2 / 2 sigma^2`. Item 39 measured that bias as
load-bearing with a dose-response curve over `sigma_m`, so dropping it would change two things at once and make a
negative result uninterpretable. The arm asks "does the published *update rule* work here?", not "does the spatial prior
matter?" -- that question already has an answer.
"""
import sys
from dataclasses import replace as replace_fields
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))

from takheliped.detectors.takheliped import (  # noqa: E402
    ObjectMemory, SlotRules, _proximity,
)

DIM, HEADS, CANDIDATES, CAPACITY = 32, 4, 6, 8


def _memory(iterations: int, recurrence: str = "mingru", **kwargs) -> ObjectMemory:
    torch.manual_seed(0)
    return ObjectMemory(SlotRules(capacity=CAPACITY), dim=DIM, heads=HEADS, recurrence=recurrence,
                        exchange=False, learned_match=False, iterations=iterations, **kwargs)


def _inputs(batch: int = 2, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(batch, CANDIDATES, 2, generator=g) * 6 - 3
    score = torch.rand(batch, CANDIDATES, generator=g)
    valid = torch.ones(batch, CANDIDATES, dtype=torch.bool)
    pose = torch.zeros(batch, 3)
    return xy, score, valid, pose


def _run(model: ObjectMemory, steps: int = 4, batch: int = 2, dt: float = 0.025, seed: int = 1):
    xy, score, valid, pose = _inputs(batch, seed)
    slots = model.initial_state(batch)
    out = None
    for _ in range(steps):
        out = model.step(xy, score, valid, pose, torch.full((batch,), dt), slots)
        slots = out.slots
    return out


def _attend(model: ObjectMemory, cand_xy: torch.Tensor, slot_xy: torch.Tensor, valid: torch.Tensor,
            alive: torch.Tensor, items: torch.Tensor = None, state: torch.Tensor = None):
    """Call the real attention module on hand-built geometry: `(update, attention)`."""
    batch, candidates = cand_xy.shape[:2]
    slots = alive.shape[1]
    g = torch.Generator().manual_seed(7)
    items = torch.randn(batch, candidates, DIM, generator=g) if items is None else items
    state = torch.randn(batch, slots, DIM, generator=g) if state is None else state
    return model.attend(state, items, _proximity(cand_xy, slot_xy, model.sigma_m), valid, alive)


def _two_slots_one_candidate():
    """Slots 0 and 1 alive and co-located at the origin, every other slot dead, one candidate on top of them."""
    alive = torch.zeros(1, CAPACITY, dtype=torch.bool)
    alive[0, :2] = True
    slot_xy = torch.zeros(1, CAPACITY, 2)
    slot_xy[0, 2:] = 40.0                       # the dead slots are also far away, so masking is what excludes them
    cand_xy = torch.zeros(1, 1, 2)
    return cand_xy, slot_xy, torch.ones(1, 1, dtype=torch.bool), alive


class TestTheDefaultIsUntouched:
    """Every number this project has published comes from the hand-coded update, so the flag is off and inert by default."""

    def test_the_flag_is_off_by_default(self):
        assert ObjectMemory(SlotRules(capacity=CAPACITY), dim=16, heads=2).iterations == 0

    def test_zero_iterations_computes_exactly_what_the_control_computes(self):
        torch.manual_seed(0)
        control = ObjectMemory(SlotRules(capacity=CAPACITY), dim=DIM, heads=HEADS, recurrence="mingru",
                               exchange=False, learned_match=False)
        off = _memory(0)

        with torch.no_grad():
            a, b = _run(control), _run(off)

        assert torch.equal(a.candidate_logit, b.candidate_logit)
        assert torch.equal(a.slots.state, b.slots.state)

    def test_zero_iterations_builds_no_attention_parameters(self):
        assert _memory(0).attend is None
        assert _memory(2).attend is not None


class TestWhatItReplaces:
    """The arm changes the update rule and the recall, and nothing else -- so the pieces it supersedes must be gone."""

    def test_the_separate_recall_attention_is_replaced_by_the_slot_attention_matrix(self):
        assert _memory(0).recall is not None
        assert _memory(2).recall is None

    def test_the_bookkeeping_still_runs_because_it_is_what_gives_a_slot_its_position(self):
        out = _run(_memory(2))

        assert out.slots.position.shape == (2, CAPACITY, 2)
        assert out.slots.alive.any(), "manage_slots must still spawn slots"
        assert (out.slots.age_s[out.slots.alive] > 0).all()

    @pytest.mark.parametrize("flags", [{"learned_match": True, "exchange": False},
                                       {"learned_match": False, "exchange": True},
                                       {"learned_match": False, "exchange": False, "recurrent": False},
                                       {"learned_match": False, "exchange": False, "global_state": True}])
    def test_it_refuses_the_features_it_would_duplicate_or_cannot_use(self, flags):
        with pytest.raises(ValueError, match="iterations"):
            ObjectMemory(SlotRules(capacity=CAPACITY), dim=DIM, heads=HEADS, iterations=2, **flags)


class TestTheAttentionIsThePaperS:
    """Slot Attention's defining detail is the softmax axis: over **slots**, so slots compete to explain a candidate."""

    def test_each_candidate_distributes_its_whole_weight_over_the_slots_and_the_null(self):
        model = _memory(2)
        cand_xy, slot_xy, valid, alive = _two_slots_one_candidate()

        with torch.no_grad():
            _, attention = _attend(model, cand_xy, slot_xy, valid, alive)

        assert attention.shape == (1, 1, CAPACITY + 1), "one column per slot, plus the null slot"
        assert torch.allclose(attention.sum(dim=-1), torch.ones(1, 1), atol=1e-5)

    def test_a_new_competing_slot_takes_weight_away_from_an_existing_one(self):
        """The test that the softmax axis is slots, and it does not depend on an untrained draw.

        Normalising over slots makes the weights a fixed budget per candidate, so waking a second slot must cost the
        first some of it. Normalising over *candidates* -- a transformer's cross-attention -- would leave slot 0
        untouched, because its own column would be renormalised alone.
        """
        model = _memory(2)
        cand_xy, slot_xy, valid, alive = _two_slots_one_candidate()
        alone = alive.clone()
        alone[0, 1] = False

        with torch.no_grad():
            one = _attend(model, cand_xy, slot_xy, valid, alone)[1][0, 0, 0]
            two = _attend(model, cand_xy, slot_xy, valid, alive)[1][0, 0, 0]

        assert two < one

    def test_a_dead_slot_receives_no_weight(self):
        model = _memory(2)
        cand_xy, slot_xy, valid, alive = _two_slots_one_candidate()

        with torch.no_grad():
            _, attention = _attend(model, cand_xy, slot_xy, valid, alive)

        assert attention[0, 0, 2:CAPACITY].abs().max() == 0.0

    def test_the_proximity_bias_reaches_the_competition(self):
        """A candidate far from every slot falls back on the null slot; one on top of a slot does not."""
        model = _memory(2)
        cand_xy, slot_xy, valid, alive = _two_slots_one_candidate()

        with torch.no_grad():
            near = _attend(model, cand_xy, slot_xy, valid, alive)[1][0, 0, -1]
            far = _attend(model, cand_xy + 50.0, slot_xy, valid, alive)[1][0, 0, -1]

        assert far > 0.99
        assert far > near

    def test_an_invalid_candidate_contributes_nothing_to_a_slot_s_update(self):
        model = _memory(2)
        g = torch.Generator().manual_seed(3)
        cand_xy = torch.zeros(1, 2, 2)
        slot_xy, alive = torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool)
        items = torch.randn(1, 2, DIM, generator=g)
        loud = items.clone()
        loud[0, 1] = 50.0
        valid = torch.tensor([[True, False]])

        with torch.no_grad():
            a = _attend(model, cand_xy, slot_xy, valid, alive, items=items)[0]
            b = _attend(model, cand_xy, slot_xy, valid, alive, items=loud)[0]

        assert torch.equal(a, b)

    def test_no_live_slot_leaves_the_null_holding_everything_and_the_update_finite(self):
        model = _memory(2)
        cand_xy = torch.zeros(1, 2, 2)
        slot_xy, alive = torch.zeros(1, CAPACITY, 2), torch.zeros(1, CAPACITY, dtype=torch.bool)

        with torch.no_grad():
            update, attention = _attend(model, cand_xy, slot_xy, torch.ones(1, 2, dtype=torch.bool), alive)

        assert torch.allclose(attention[..., -1], torch.ones(1, 2), atol=1e-6)
        assert torch.isfinite(update).all()

    def test_no_valid_candidate_leaves_the_update_finite(self):
        model = _memory(2)
        cand_xy = torch.zeros(1, 2, 2)
        slot_xy, alive = torch.zeros(1, CAPACITY, 2), torch.ones(1, CAPACITY, dtype=torch.bool)

        with torch.no_grad():
            update, _ = _attend(model, cand_xy, slot_xy, torch.zeros(1, 2, dtype=torch.bool), alive)

        assert torch.isfinite(update).all()


class TestTheIteration:
    """Iterated refinement is the other half of the mechanism: the update runs, the state changes, the attention re-reads it."""

    def test_more_iterations_change_the_state(self):
        with torch.no_grad():
            one, three = _run(_memory(1)).slots.state, _run(_memory(3)).slots.state

        assert not torch.allclose(one, three, atol=1e-4)

    def test_elapsed_time_is_applied_once_per_frame_not_once_per_iteration(self):
        """`selective` decays over real seconds, so three iterations must not spend `dt` three times."""
        model = _memory(3, recurrence="selective")
        seen = []
        original = model._recur

        def spy(state, u, dt):
            seen.append(dt.clone())
            return original(state, u, dt)

        model._recur = spy
        xy, score, valid, pose = _inputs(batch=1)
        with torch.no_grad():
            model.step(xy, score, valid, pose, torch.full((1,), 0.4), model.initial_state(1))

        assert len(seen) == 3
        assert seen[0].item() == pytest.approx(0.4)
        assert [d.item() for d in seen[1:]] == [0.0, 0.0]

    def test_the_attention_projections_receive_gradient(self):
        """The gate has to be woken first: at its zero initialisation `0 * correction` gives every parameter behind the
        head a grad of exactly zero, so the test would pass on a disconnected graph and measure nothing."""
        model = _memory(2)
        with torch.no_grad():
            model.rescore_gate.fill_(1.0)
        xy, score, valid, pose = _inputs(batch=1)

        out = model.step(xy, score, valid, pose, torch.full((1,), 0.025), model.initial_state(1))
        out.candidate_logit.abs().sum().backward()

        assert model.attend.to_q.weight.grad is not None
        assert model.attend.to_q.weight.grad.abs().sum() > 0
        assert model.attend.to_v.weight.grad.abs().sum() > 0


class TestTheReadout:
    """The same attention matrix is the recall, so what a slot carries has to reach the candidate's score through it."""

    def test_perturbing_only_the_slot_states_changes_the_rescored_logits(self):
        model = _memory(2)
        torch.nn.init.normal_(model.rescore[-1].weight, std=0.5)    # the head starts silent, so wake it for this test
        with torch.no_grad():
            model.rescore_gate.fill_(1.0)
        xy, score, valid, pose = _inputs(batch=1)
        out = _run(model, steps=3, batch=1)
        loud = replace_fields(out.slots, state=out.slots.state + 3.0)

        with torch.no_grad():
            dt = torch.full((1,), 0.025)
            a = model.step(xy, score, valid, pose, dt, out.slots).candidate_logit
            b = model.step(xy, score, valid, pose, dt, loud).candidate_logit

        assert not torch.allclose(a, b, atol=1e-4)

    def test_the_rescoring_head_still_starts_silent(self):
        """Item 46's lesson: an arm whose head is not silent at initialisation measures its own random draw."""
        model = _memory(2)
        xy, score, valid, pose = _inputs()

        with torch.no_grad():
            out = model.step(xy, score, valid, pose, torch.full((2,), 0.025), model.initial_state(2))

        assert torch.allclose(out.candidate_logit, _logit_of(score), atol=1e-5)


class TestTheDegenerateFrames:
    """A softmax over slots has no defined answer when no slot is alive, which is every stream's first frame."""

    def test_an_empty_memory_produces_no_nan(self):
        model = _memory(2)
        xy, score, valid, pose = _inputs()

        with torch.no_grad():
            out = model.step(xy, score, valid, pose, torch.full((2,), 0.025), model.initial_state(2))

        assert torch.isfinite(out.candidate_logit).all()
        assert torch.isfinite(out.slots.state).all()

    def test_a_frame_with_no_valid_candidate_produces_no_nan(self):
        model = _memory(2)
        xy, score, valid, pose = _inputs()
        out = _run(model, steps=2)

        with torch.no_grad():
            empty = model.step(xy, score, torch.zeros_like(valid), pose, torch.full((2,), 0.025), out.slots)

        assert torch.isfinite(empty.candidate_logit).all()
        assert torch.isfinite(empty.slots.state).all()

    def test_a_long_run_stays_finite(self):
        with torch.no_grad():
            out = _run(_memory(3), steps=12)

        assert torch.isfinite(out.slots.state).all()
        assert torch.isfinite(out.candidate_logit).all()


def _logit_of(score: torch.Tensor) -> torch.Tensor:
    p = score.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


class TestTheBackwardSurvivesAnEmptyAttentionRow:
    """Backward-direction invariants for a degenerate attention row. **These are NOT a regression test for item 53.**

    They were written as one and they do not qualify: they pass *before* the fix, because in these synthetic cases the
    weight rows sum to exactly 0 and the gradient through the update is exactly 0 rather than NaN -- even with the
    upstream gradient scaled to 1e12. The real failure needed the magnitudes of a 399-frame chunk with 256 slots and
    64 candidates, which a one-frame unit test does not produce. Item 53's reproduction is the trainer's own saved
    batch replayed under `set_detect_anomaly`, and that is what verifies the fix.

    They are kept because they close the gap that let item 53 through in the first place: the four degenerate-frame
    tests above assert `isfinite` on the **forward** only, and `memory/dos-and-donts.md` now has a rule about exactly
    that. These two assert on `.grad` after a `backward()`.

    What item 53 turned out to be, for the record:

        RuntimeError: Function 'MulBackward0' returned nan values in its 0th output
          takheliped.py  weights = attention[..., :-1] * valid[..., None]

    The gradient *arriving* at that multiply was NaN, so the fault is downstream of it in forward order -- the
    normalisation, whose `clamp(min=1e-6)` denominator makes `1/den` up to 1e6 and `-num/den**2` up to 1e12. The fix
    substitutes 1 for an empty denominator instead of epsilon, which removes both unbounded factors.

    Found by replaying the trainer's own saved batch under `set_detect_anomaly`, which named the line:

        RuntimeError: Function 'MulBackward0' returned nan values in its 0th output
          takheliped.py  weights = attention[..., :-1] * valid[..., None]

    **The fault is an interaction, which is why three earlier probes all came back clean.** `attention` is a softmax
    over slots; where the proximity bias drives real-slot logits to `exp(-large)` the entries underflow to exactly 0.
    Multiplying by `valid` is finite forward. But the normalisation below divided by a denominator `clamp`ed to 1e-6,
    so the *backward* arriving at that multiply carries a ~1e6 factor, and `MulBackward0` then forms `0 * inf` -> NaN.

    Testing the clamp alone showed no gradient amplification -- correctly -- and testing degenerate frames showed a
    finite *forward*. Neither could see the combination. **The fix removes the `0 * huge` term entirely**: substitute
    1 for an empty denominator instead of epsilon, so the numerator's 0 divides by 1 and the backward's denominator is
    never tiny.
    """

    def test_gradients_are_finite_when_a_candidate_contributes_nothing(self):
        model = _memory(2)
        with torch.no_grad():
            model.rescore_gate.fill_(1.0)
        g = torch.Generator().manual_seed(5)
        # One valid candidate and one invalid, and slots far enough away that the proximity bias underflows the
        # softmax for the real slots -- the exact pairing the trainer hit.
        cand_xy = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        slot_xy = torch.full((1, CAPACITY, 2), 60.0)
        valid = torch.tensor([[True, False]])
        alive = torch.ones(1, CAPACITY, dtype=torch.bool)
        items = torch.randn(1, 2, DIM, generator=g, requires_grad=True)
        state = torch.randn(1, CAPACITY, DIM, generator=g, requires_grad=True)

        update, attention = model.attend(state, items, _proximity(cand_xy, slot_xy, model.sigma_m), valid, alive)
        model.attend.read(attention, update).abs().sum().backward()

        assert torch.isfinite(items.grad).all(), "item 53: NaN in the items gradient"
        assert torch.isfinite(state.grad).all(), "item 53: NaN in the state gradient"
        for name, p in model.attend.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"item 53: NaN in {name}"

    def test_an_all_invalid_frame_also_has_a_finite_backward(self):
        """The limit case: every candidate masked out, so every weight row sums to zero."""
        model = _memory(2)
        g = torch.Generator().manual_seed(6)
        cand_xy = torch.zeros(1, 3, 2)
        slot_xy = torch.full((1, CAPACITY, 2), 80.0)
        items = torch.randn(1, 3, DIM, generator=g, requires_grad=True)
        state = torch.randn(1, CAPACITY, DIM, generator=g, requires_grad=True)

        update, attention = model.attend(state, items, _proximity(cand_xy, slot_xy, model.sigma_m),
                                         torch.zeros(1, 3, dtype=torch.bool),
                                         torch.ones(1, CAPACITY, dtype=torch.bool))
        (update.abs().sum() + attention.abs().sum()).backward()

        assert torch.isfinite(items.grad).all()
        assert torch.isfinite(state.grad).all()
