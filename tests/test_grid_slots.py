"""grid slots -- a slot whose position snaps to a room cell and never tracks the object (the owner's design).

**The question, and why this is the sharp form of it.** The global-state ablation showed that *addressability* is load-bearing: one
global state reaches 82.44% against the slots' 85.88-86.08%. It did **not** show that *object-centric* addressability
beats *spatial* addressability, because nothing spatial was ever run -- and "why slots rather than an occupancy grid?"
is the obvious question about this design. A learned room grid is the classical robotics answer.

**Why grid slots rather than a dense grid.** FROG's venue is 120-250 m across, so a dense person-scale grid over the
room needs ~46,000 cells -- 180x the recall attention's cost, which would make the arm a cost experiment with a
foregone verdict. A 256-cell room grid instead gives ~11 x 17 m cells against a 0.5 m match radius, which is a
strawman. Grid slots take the third way: **keep 256 slots, give each one a quantised room cell, and never instantiate
an empty cell.** Capacity, room frame, recurrence, recall, head and eviction are all unchanged.

**So exactly one thing varies**, which is what makes this an ablation rather than a different model: an adopted slot's
position is continuous and *follows its object*; a grid slot's position is its cell centre and **never moves**. If
object binding is doing real work, this must cost.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))

from takheliped.detectors.takheliped import (  # noqa: E402
    ObjectMemory, SlotRules, initial_slots, manage_slots,
)

CELL_M = 1.0
CAPACITY = 16


def _rules(cell_m: float = CELL_M, **kwargs) -> SlotRules:
    return SlotRules(capacity=CAPACITY, cell_m=cell_m, **kwargs)


def _slots(batch: int = 1, dim: int = 8):
    return initial_slots(batch, _rules(), dim, None)


def _step(slots, xy, score=None, rules=None, dt=0.025):
    xy = torch.as_tensor(xy, dtype=torch.float32)[None] if torch.as_tensor(xy).ndim == 2 else torch.as_tensor(xy)
    n = xy.shape[1]
    score = torch.full((1, n), 0.9) if score is None else torch.as_tensor(score, dtype=torch.float32).reshape(1, n)
    valid = torch.ones(1, n, dtype=torch.bool)
    return manage_slots(slots, xy, score, valid, torch.full((1,), dt), rules or _rules())


def _live(slots):
    return slots.position[0][slots.alive[0]]


class TestACellIsACell:
    """A grid slot's position is a cell centre, and it is the same cell for as long as the slot lives."""

    def test_a_spawned_slot_sits_at_its_cell_centre(self):
        slots, _ = _step(_slots(), [[[0.2, 0.3], [4.8, -2.1]]])
        live = _live(slots)

        assert len(live) == 2
        for x, y in live.tolist():
            assert (x / CELL_M - 0.5) % 1.0 == pytest.approx(0.0, abs=1e-5), x
            assert (y / CELL_M - 0.5) % 1.0 == pytest.approx(0.0, abs=1e-5), y

    def test_a_slot_does_not_follow_its_candidate_across_the_cell(self):
        """The whole experiment: an adopted slot tracks the object, a grid slot does not."""
        slots, _ = _step(_slots(), [[[0.1, 0.1]]])
        first = _live(slots).clone()

        for x in (0.3, 0.6, 0.9):
            slots, _ = _step(slots, [[[x, 0.1]]])

        assert torch.allclose(_live(slots), first, atol=1e-6), "the slot moved inside its own cell"

    def test_a_grid_slot_has_no_velocity(self):
        slots, _ = _step(_slots(), [[[0.1, 0.1]]])
        for x in (0.3, 0.6, 0.9):
            slots, _ = _step(slots, [[[x, 0.1]]])

        assert torch.equal(slots.velocity, torch.zeros_like(slots.velocity))

    def test_crossing_into_the_next_cell_uses_a_different_slot(self):
        slots, _ = _step(_slots(), [[[0.5, 0.5]]])
        before = int(slots.alive.sum())
        slots, _ = _step(slots, [[[1.6, 0.5]]])

        assert int(slots.alive.sum()) == before + 1, "a new cell must take a new slot, not move the old one"


class TestOneSlotPerOccupiedCell:
    def test_two_candidates_in_one_cell_occupy_one_slot(self):
        slots, _ = _step(_slots(), [[[0.2, 0.2], [0.7, 0.7]]])

        assert int(slots.alive.sum()) == 1

    def test_candidates_in_distinct_cells_occupy_distinct_slots(self):
        slots, _ = _step(_slots(), [[[0.5, 0.5], [2.5, 0.5], [0.5, 3.5]]])

        assert int(slots.alive.sum()) == 3
        assert len({tuple(p) for p in _live(slots).tolist()}) == 3

    def test_a_returning_candidate_reuses_its_cell_rather_than_spawning(self):
        slots, _ = _step(_slots(), [[[0.5, 0.5]]])
        slots, events = _step(slots, [[[0.4, 0.6]]])

        assert int(slots.alive.sum()) == 1
        assert bool(events.matched[0].any()), "same cell must match, not spawn"

    def test_empty_cells_are_never_instantiated(self):
        """The point of the sparse design: the room is 150 m across and only occupied cells cost anything."""
        slots, _ = _step(_slots(), [[[40.5, 40.5], [-30.5, 12.5]]])

        assert int(slots.alive.sum()) == 2, "two occupied cells, regardless of how far apart they are"

    def test_capacity_still_bounds_the_live_cells(self):
        g = torch.Generator().manual_seed(0)
        slots = _slots()
        for _ in range(6):
            xy = (torch.rand(1, 12, 2, generator=g) * 40 - 20)
            slots, _ = _step(slots, xy)

        assert int(slots.alive.sum()) <= CAPACITY


class TestTheDefaultIsUntouched:
    """Every published number comes from continuous tracking slots, so the flag is off and inert by default."""

    def test_cell_m_is_zero_by_default(self):
        assert SlotRules().cell_m == 0.0

    def test_with_cell_m_zero_the_bookkeeping_is_what_it_always_was(self):
        g = torch.Generator().manual_seed(1)
        xy = torch.rand(1, 6, 2, generator=g) * 8 - 4
        plain = manage_slots(initial_slots(1, SlotRules(capacity=CAPACITY), 8, None), xy,
                             torch.full((1, 6), 0.9), torch.ones(1, 6, dtype=torch.bool),
                             torch.full((1,), 0.025), SlotRules(capacity=CAPACITY))
        same = manage_slots(initial_slots(1, _rules(cell_m=0.0), 8, None), xy,
                            torch.full((1, 6), 0.9), torch.ones(1, 6, dtype=torch.bool),
                            torch.full((1,), 0.025), _rules(cell_m=0.0))

        assert torch.equal(plain[0].position, same[0].position)
        assert torch.equal(plain[0].alive, same[0].alive)

    def test_a_tracking_slot_does_follow_its_candidate(self):
        """The control for the headline test above -- without it that assertion proves nothing about the mechanism."""
        rules = SlotRules(capacity=CAPACITY)
        slots, _ = _step(initial_slots(1, rules, 8, None), [[[0.1, 0.1]]], rules=rules)
        first = _live(slots).clone()
        for x in (0.3, 0.6, 0.9):
            slots, _ = _step(slots, [[[x, 0.1]]], rules=rules)

        assert not torch.allclose(_live(slots), first, atol=1e-3)


class TestItReachesTheModel:
    def test_the_flag_exists_and_defaults_to_off(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
        from train_takheliped import build_parser

        flags = {a.dest: a for a in build_parser()._subparsers._group_actions[0].choices["step3a"]._actions}

        assert flags["cell_m"].default == 0.0

    def test_a_memory_built_with_cells_keeps_them(self):
        memory = ObjectMemory(_rules(), dim=16, heads=2, recurrence="mingru", exchange=False, learned_match=False)

        assert memory.slot_rules().cell_m == CELL_M

    def test_a_full_step_runs_and_reports_cell_positions(self):
        memory = ObjectMemory(_rules(), dim=16, heads=2, recurrence="mingru", exchange=False, learned_match=False)
        xy = torch.tensor([[[0.2, 0.3], [3.7, 1.1]]])
        out = memory.step(xy, torch.full((1, 2), 0.9), torch.ones(1, 2, dtype=torch.bool),
                          torch.zeros(1, 3), torch.full((1,), 0.025), memory.initial_state(1))

        assert torch.isfinite(out.candidate_logit).all()
        for x, y in out.slots.position[0][out.slots.alive[0]].tolist():
            assert (x / CELL_M - 0.5) % 1.0 == pytest.approx(0.0, abs=1e-5)
