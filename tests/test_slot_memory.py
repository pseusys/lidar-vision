"""Value-ranked slot memory, rule-based stand-in (TODO.md A43, PROPOSAL.md §6.3).

Owner's rule, 2026-09-14: every slot carries one *value* -- its accumulated
certainty -- and a new candidate may take a slot from the weakest one if it is
clearly more certain. The value is accumulated, not the latest score, so a
person hidden for a few frames is not evicted by the first phantom flicker.

    spawn     any unmatched candidate with score >= floor; free slots first,
              strongest candidate first
    match     v <- v + gain * (1 - v) * score               (no decay that step)
    unmatched v <- v * exp(-dt / tau),  tau = retire_s / ln(1 / floor)
              so a slot at full certainty fades to the floor in exactly retire_s
    retire    v < floor
    evict     when full: strongest remaining candidate vs weakest evictable slot,
              replace only if score > value + margin; never a slot matched or
              spawned this step
    oracle    the same, but slots holding an annotated person are evicted only
              when no other slot is evictable -- the reference for the rule
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from slot_budget import PriorityMemory, coverage


def _xy(*pts):
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def _s(*scores):
    return np.asarray(scores, dtype=np.float64)


def _mem(**kw):
    args = dict(capacity=4, floor=0.05, margin=0.1, retire_s=10.0, gain=0.1, gate=0.6)
    args.update(kw)
    return PriorityMemory(**args)


class TestSpawning:

    def test_candidates_below_the_floor_never_spawn(self):
        m = _mem()
        m.step(_xy((0, 0)), _s(0.04), dt=0.04)
        assert m.n_alive == 0

    def test_free_slots_go_to_the_strongest_candidates_first(self):
        m = _mem(capacity=2)
        m.step(_xy((0, 0), (5, 0), (10, 0)), _s(0.2, 0.9, 0.5), dt=0.04)
        assert sorted(m.values.tolist()) == pytest.approx([0.5, 0.9])

    def test_unlimited_capacity_never_evicts(self):
        m = _mem(capacity=None)
        out = m.step(_xy(*[(3 * i, 0) for i in range(50)]), _s(*[0.5] * 50), dt=0.04)
        assert m.n_alive == 50 and out["evicted"] == 0


class TestMatchingAndValue:

    def test_a_candidate_inside_the_gate_updates_rather_than_spawns(self):
        m = _mem()
        m.step(_xy((0, 0)), _s(0.5), dt=0.04)
        m.step(_xy((0.3, 0)), _s(0.8), dt=0.04)
        assert m.n_alive == 1
        assert m.values[0] == pytest.approx(0.5 + 0.1 * (1 - 0.5) * 0.8)
        assert m.positions[0] == pytest.approx([0.3, 0.0])

    def test_a_candidate_matches_at_most_one_slot(self):
        m = _mem()
        m.step(_xy((0, 0), (1.0, 0)), _s(0.5, 0.5), dt=0.04)
        m.step(_xy((0.5, 0)), _s(0.5), dt=0.04)
        assert m.n_alive == 2
        assert sorted(m.values.tolist())[0] < 0.5       # the unmatched one decayed

    def test_an_unmatched_slot_decays_exponentially(self):
        m = _mem()
        m.step(_xy((0, 0)), _s(0.5), dt=0.04)
        m.step(_xy(), _s(), dt=1.0)
        tau = 10.0 / math.log(1 / 0.05)
        assert m.values[0] == pytest.approx(0.5 * math.exp(-1.0 / tau))

    def test_full_certainty_fades_to_the_floor_in_exactly_the_retire_window(self):
        m = _mem()
        m.step(_xy((0, 0)), _s(1.0), dt=0.04)
        m.step(_xy(), _s(), dt=9.0)
        assert m.n_alive == 1
        out = m.step(_xy(), _s(), dt=2.0)
        assert m.n_alive == 0 and out["retired"] == 1


class TestEviction:

    def _full(self, oracle=False):
        m = _mem(capacity=2, oracle=oracle)
        m.step(_xy((0, 0), (5, 0)), _s(0.3, 0.6), dt=0.04,
               is_person=np.array([True, False]))
        return m

    def test_a_clearly_stronger_candidate_evicts_the_weakest_slot(self):
        m = self._full()
        out = m.step(_xy((20, 0)), _s(0.9), dt=0.04, is_person=np.array([False]))
        assert out["evicted"] == 1 and out["evicted_person"] == 1
        assert 0.9 in m.values.tolist()

    def test_no_eviction_within_the_margin(self):
        m = self._full()
        out = m.step(_xy((20, 0)), _s(0.35), dt=0.04, is_person=np.array([False]))
        assert out["evicted"] == 0 and m.n_alive == 2

    def test_a_slot_matched_this_step_is_never_evicted(self):
        m = self._full()
        # the 0.3 slot is re-observed weakly, the 0.6 slot is not
        out = m.step(_xy((0.1, 0), (20, 0)), _s(0.06, 0.95), dt=0.04,
                     is_person=np.array([True, False]))
        assert out["evicted"] == 1 and out["evicted_person"] == 0

    def test_a_long_tracked_person_survives_a_phantom_flicker(self):
        """The case memory exists for: a person tracked at high certainty,
        hidden for half a second, must not lose their slot to a 0.3 flicker."""
        m = _mem(capacity=1)
        for _ in range(60):
            m.step(_xy((0, 0)), _s(0.9), dt=0.04)
        m.step(_xy(), _s(), dt=0.5)
        out = m.step(_xy((10, 0)), _s(0.3), dt=0.04)
        assert out["evicted"] == 0

    def test_the_oracle_evicts_a_non_person_before_a_weaker_person(self):
        m = self._full(oracle=True)
        out = m.step(_xy((20, 0)), _s(0.9), dt=0.04, is_person=np.array([False]))
        assert out["evicted"] == 1 and out["evicted_person"] == 0


class TestCoverage:

    def test_one_slot_covers_at_most_one_person(self):
        assert coverage(_xy((0, 0)), _xy((0.1, 0), (-0.1, 0)), gate=0.5) == 1

    def test_people_beyond_the_gate_are_not_covered(self):
        assert coverage(_xy((0, 0), (5, 5)), _xy((0.2, 0), (9, 9)), gate=0.5) == 1

    def test_nothing_to_cover(self):
        assert coverage(_xy(), _xy((1, 1)), gate=0.5) == 0
        assert coverage(_xy((1, 1)), _xy(), gate=0.5) == 0
