"""Item 46 (A51's O3): slots regress a correction to the candidate's position.

**Why this arm has the best prior left on the list.** Item 35 measured the memory at its own operating point as *net
negative on coverage* -- it recovers 203 person-observations and loses 2,990 -- so its entire +3.9 pp of AP is bought
with ranking rather than recall. Every other arm in the sweep changes how the memory *scores* what stage 2 proposed; a
position correction is the only mechanism on the list that could change *where* a detection is, and therefore the only
one that could convert carried state into coverage.

Two invariants carry the design:

    the head starts silent      its last layer is zero-initialised, so an untrained arm cannot move a detection and
                                the arm's first epochs are the control's. Without this, "refinement" and "a randomly
                                displaced detection" are the same thing.
    the field is always there   `MemoryOutput.candidate_xy` exists whether or not the head does, equal to the input
                                positions when it does not. The evaluation path then reads one field unconditionally
                                rather than branching on a flag -- which is how `--global-state` came to be silently
                                ignored for a whole run (item 36).
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))

from takheliped.detectors.takheliped import ObjectMemory, SlotRules  # noqa: E402

DIM, HEADS, CANDIDATES = 32, 4, 6


def _memory(refine: bool, **kwargs) -> ObjectMemory:
    torch.manual_seed(0)
    return ObjectMemory(SlotRules(capacity=16), dim=DIM, heads=HEADS, recurrence="mingru",
                        exchange=False, learned_match=False, refine=refine, **kwargs)


def _inputs(batch: int = 2, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(batch, CANDIDATES, 2, generator=g) * 6 - 3,
            torch.rand(batch, CANDIDATES, generator=g),
            torch.ones(batch, CANDIDATES, dtype=torch.bool),
            torch.zeros(batch, 3))


def _play(model: ObjectMemory, steps: int = 3, batch: int = 2):
    xy, score, valid, pose = _inputs(batch)
    slots = model.initial_state(batch)
    out = None
    for _ in range(steps):
        out = model.step(xy, score, valid, pose, torch.full((batch,), 0.025), slots)
        slots = out.slots
    return out, xy


class TestTheFieldIsAlwaysThere:
    def test_the_head_is_off_by_default(self):
        assert ObjectMemory(SlotRules(capacity=8), dim=16, heads=2).refine is None

    def test_without_the_head_the_reported_position_is_the_candidate_s_own(self):
        """So `evaluate` can read one field unconditionally instead of branching on a flag."""
        out, xy = _play(_memory(False))

        assert out.candidate_xy.shape == xy.shape
        assert torch.allclose(out.candidate_xy, xy, atol=0)

    def test_the_global_state_arm_reports_it_too(self):
        """Item 36's path builds its own `MemoryOutput`, so it has to carry the field as well."""
        model = _memory(False, global_state=True)
        out, xy = _play(model)

        assert torch.allclose(out.candidate_xy, xy, atol=0)

    def test_without_the_head_no_parameters_are_added_so_old_checkpoints_still_load(self):
        assert set(_memory(False).state_dict()) == set(ObjectMemory(
            SlotRules(capacity=16), dim=DIM, heads=HEADS, recurrence="mingru",
            exchange=False, learned_match=False).state_dict())


class TestItStartsSilent:
    def test_an_untrained_head_does_not_move_a_single_detection(self):
        """The invariant that makes the arm an ablation rather than a perturbation."""
        out, xy = _play(_memory(True))

        assert torch.allclose(out.candidate_xy, xy, atol=1e-6)

    def test_so_an_untrained_arm_scores_exactly_the_control(self):
        with torch.no_grad():
            quiet, _ = _play(_memory(True))
            control, _ = _play(_memory(False))

        assert torch.allclose(quiet.candidate_logit, control.candidate_logit, atol=1e-6)

    def test_the_head_adds_parameters_and_they_are_reachable(self):
        with_head, without = _memory(True), _memory(False)

        assert sum(p.numel() for p in with_head.parameters()) > sum(p.numel() for p in without.parameters())
        assert any("refine" in name for name, _ in with_head.named_parameters())


class TestOnceTrainedItMoves:
    def test_a_trained_head_displaces_the_reported_position(self):
        model = _memory(True)
        with torch.no_grad():                       # stand in for training: break the zero initialisation
            torch.nn.init.normal_(model.refine[-1].weight, std=0.1)
            torch.nn.init.constant_(model.refine[-1].bias, 0.05)

        out, xy = _play(model)

        assert not torch.allclose(out.candidate_xy, xy, atol=1e-4)

    def test_gradients_reach_the_head_through_the_reported_position(self):
        """The L1 loss trains it through `candidate_xy` alone, so that path must carry gradient.

        The target is offset from the candidate on purpose. A first version used the candidate itself, which with the
        zero-initialised head makes the residual *exactly* zero -- and `abs()` has zero subgradient at zero in PyTorch,
        so the test reported no gradient and was measuring its own degenerate construction. The real loss never sits
        there: its target is a person's position, not the candidate's.
        """
        model = _memory(True)
        out, xy = _play(model)

        (out.candidate_xy - (xy + 0.3)).abs().sum().backward()

        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.refine.parameters())

    def test_the_correction_is_added_to_the_candidate_rather_than_replacing_it(self):
        """A displaced detection has to stay near its candidate; a head that predicted absolute positions would start
        at the origin and have to learn the whole scene geometry before it beat the control."""
        model = _memory(True)
        with torch.no_grad():
            torch.nn.init.zeros_(model.refine[-1].weight)
            torch.nn.init.constant_(model.refine[-1].bias, 0.25)

        out, xy = _play(model)

        assert torch.allclose(out.candidate_xy - xy, torch.full_like(xy, 0.25), atol=1e-5)

    def test_it_is_refused_together_with_a_silent_rescoring_head(self):
        """A51 E12a's arm drops `rescore`, which leaves `recalled` with nowhere to go but this head -- a different
        experiment, and not one item 46 is asking for."""
        with pytest.raises(ValueError, match="refine"):
            ObjectMemory(SlotRules(capacity=16), dim=DIM, heads=HEADS, refine=True, rescore=False)
