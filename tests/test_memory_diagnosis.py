"""Pairing candidates to annotated people (`TODO.md` A50 phase 1b).

`memory_diagnosis.pairs` is the step the whole "what did stage 3 do to each person" decomposition rests on: it says which
candidate covers which person, one to one and inside the match radius. The two properties worth pinning are the ones the
diagnosis reads meaning into -- a candidate further than the radius covers nobody at all, and only *one* candidate can cover a
person, so a second detection on the same person is counted as covering nobody. That second property is the caveat recorded
against the "+0.853 median delta on candidates covering nobody" figure, so it needs to be true on purpose rather than by luck.

`logit` is pinned too, because the rescoring delta is a difference of logits and a silent infinity there would poison every
median in the summary.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from memory_diagnosis import covered_people, label_coasting_reports, logit, pairs, retirement_gap, windowed_speed


class TestLabelCoastingReports:
    """Which coasting reports would recover a person stage 2 missed, and which would only add a false positive (`TODO.md` A51 O2).
    Splitting them by score is what turns the coasting head's threshold into a curve instead of a guess."""

    def test_a_report_on_a_missed_person_is_a_hit_and_a_second_one_is_not(self):
        hits, false = label_coasting_reports(np.array([[0.0, 1.0], [0.1, 1.0], [5.0, 5.0]]), np.array([0.9, 0.8, 0.7]),
                                             np.array([[0.05, 1.0]]), 0.5)
        assert sorted(hits) == [0.9]
        assert sorted(false) == [0.7, 0.8]

    def test_the_nearest_report_takes_the_person(self):
        hits, false = label_coasting_reports(np.array([[0.4, 1.0], [0.0, 1.0]]), np.array([0.6, 0.9]), np.array([[0.0, 1.0]]), 0.5)
        assert sorted(hits) == [0.9] and sorted(false) == [0.6]

    def test_no_reports_or_nobody_missed(self):
        hits, false = label_coasting_reports(np.zeros((0, 2)), np.zeros(0), np.array([[0.0, 1.0]]), 0.5)
        assert len(hits) == 0 and len(false) == 0
        hits, false = label_coasting_reports(np.array([[0.0, 1.0]]), np.array([0.5]), np.zeros((0, 2)), 0.5)
        assert len(hits) == 0 and sorted(false) == [0.5]


class TestCoveredPeople:
    """How many annotated people have a live slot standing on them: the upper bound on what the memory could recall, whatever the
    learned head then does with it (`TODO.md` A51 S1)."""

    def test_a_person_with_a_live_slot_within_the_radius_is_covered(self):
        slots = np.array([[0.0, 1.0], [9.0, 9.0]])
        assert covered_people(slots, np.array([True, True]), np.array([[0.2, 1.0]]), np.array([True]), 0.5) == 1

    def test_a_dead_slot_covers_nobody(self):
        assert covered_people(np.array([[0.0, 1.0]]), np.array([False]), np.array([[0.0, 1.0]]), np.array([True]), 0.5) == 0

    def test_two_people_under_one_slot_count_once_each_only_if_each_is_inside_the_radius(self):
        slots = np.array([[0.0, 1.0]])
        people = np.array([[0.2, 1.0], [1.2, 1.0]])
        assert covered_people(slots, np.array([True]), people, np.array([True, True]), 0.5) == 1

    def test_padded_people_are_ignored(self):
        people = np.array([[0.0, 1.0], [0.0, 0.0]])
        assert covered_people(np.array([[0.0, 1.0]]), np.array([True]), people, np.array([True, False]), 0.5) == 1


class TestRetirementGap:
    """When a slot is lost and something reappears in the same place, how long was the gap: what state would have to survive for a
    returning person to keep it (`TODO.md` A51 S2)."""

    def test_a_respawn_near_a_lost_slot_reports_its_gap(self):
        gap = retirement_gap(np.array([1.0, 1.0]), np.array([[1.1, 1.0], [8.0, 8.0]]), np.array([100.0, 100.0]), 112.0, 0.5, 60.0)
        assert gap == pytest.approx(12.0)

    def test_the_most_recent_of_several_nearby_losses_wins(self):
        gap = retirement_gap(np.array([1.0, 1.0]), np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([100.0, 110.0]), 112.0, 0.5, 60.0)
        assert gap == pytest.approx(2.0)

    def test_a_respawn_far_from_any_loss_or_past_the_horizon_reports_nothing(self):
        assert np.isnan(retirement_gap(np.array([5.0, 5.0]), np.array([[1.0, 1.0]]), np.array([100.0]), 112.0, 0.5, 60.0))
        assert np.isnan(retirement_gap(np.array([1.0, 1.0]), np.array([[1.0, 1.0]]), np.array([10.0]), 112.0, 0.5, 60.0))
        assert np.isnan(retirement_gap(np.array([1.0, 1.0]), np.zeros((0, 2)), np.zeros(0), 112.0, 0.5, 60.0))


def _xy(points):
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


class TestWindowedSpeed:
    """The reference slot speed for the `dt` check (`TODO.md` A51 0a): displacement over at least `window_s`, never a division by
    one stamped interval, restarted whenever a slot is refilled with a different object."""

    @staticmethod
    def _run(positions, times, refills, window_s=1.0):
        anchor_xy, anchor_t = positions[0].copy(), np.full(positions.shape[1], times[0])
        out = []
        for xy, t, refill in zip(positions, times, refills):
            speed, anchor_xy, anchor_t = windowed_speed(anchor_xy, anchor_t, xy, t, refill, window_s)
            out.append(speed)
        return np.array(out)

    def test_constant_walking_speed_is_recovered_despite_bunched_stamps(self):
        # 40 Hz motion at 1.2 m/s, stamped in FROG's bunches: 38 ms, 38 ms, then under 0.1 ms
        steps = np.tile([0.038, 0.038, 0.00005], 60)
        times = np.concatenate([[0.0], np.cumsum(steps)])
        true_t = np.arange(len(times)) * 0.025
        positions = np.stack([1.2 * true_t, np.zeros_like(true_t)], axis=-1)[:, None, :]
        speeds = self._run(positions, times, np.zeros((len(times), 1), dtype=bool))
        emitted = speeds[np.isfinite(speeds)]
        assert len(emitted) > 0 and np.allclose(emitted, 1.2, rtol=0.15)

    def test_nothing_is_emitted_before_the_window_has_elapsed(self):
        times = np.arange(10) * 0.025
        positions = np.zeros((10, 1, 2))
        assert np.all(np.isnan(self._run(positions, times, np.zeros((10, 1), dtype=bool))))

    def test_a_refill_restarts_the_window_so_a_jump_between_objects_is_not_speed(self):
        times = np.arange(100) * 0.025
        positions = np.zeros((100, 1, 2))
        positions[40:, 0, 0] = 5.0          # a different object 5 m away takes the slot at frame 40
        refills = np.zeros((100, 1), dtype=bool)
        refills[40, 0] = True
        speeds = self._run(positions, times, refills)
        assert np.isfinite(speeds).any() and np.allclose(speeds[np.isfinite(speeds)], 0.0)

    def test_slots_are_independent(self):
        times = np.arange(81) * 0.025
        positions = np.zeros((81, 2, 2))
        positions[:, 1, 1] = 2.0 * times
        speeds = self._run(positions, times, np.zeros((81, 2), dtype=bool))
        assert np.allclose(speeds[np.isfinite(speeds[:, 0]), 0], 0.0)
        assert np.allclose(speeds[np.isfinite(speeds[:, 1]), 1], 2.0)


class TestPairs:

    def test_a_candidate_inside_the_radius_covers_its_person(self):
        assert pairs(_xy([(1.0, 1.0)]), np.array([True]), _xy([(1.1, 1.0)]), 0.5) == {0: 0}

    def test_a_candidate_beyond_the_radius_covers_nobody(self):
        assert pairs(_xy([(1.0, 1.0)]), np.array([True]), _xy([(2.0, 1.0)]), 0.5) == {}

    def test_only_one_candidate_can_cover_a_person(self):
        # the second detection on one person is what the diagnosis counts as "covering nobody"
        got = pairs(_xy([(1.0, 1.0), (1.05, 1.0)]), np.array([True, True]), _xy([(1.02, 1.0)]), 0.5)
        assert list(got) == [0] and got[0] in (0, 1)

    def test_each_person_takes_their_own_nearest_candidate(self):
        got = pairs(_xy([(0.0, 0.0), (5.0, 0.0)]), np.array([True, True]), _xy([(5.05, 0.0), (0.05, 0.0)]), 0.5)
        assert got == {0: 1, 1: 0}

    def test_candidates_marked_unusable_are_ignored(self):
        assert pairs(_xy([(1.0, 1.0)]), np.array([False]), _xy([(1.0, 1.0)]), 0.5) == {}

    def test_no_candidates_or_no_people_pairs_nothing(self):
        assert pairs(_xy([]), np.zeros(0, dtype=bool), _xy([(1.0, 1.0)]), 0.5) == {}
        assert pairs(_xy([(1.0, 1.0)]), np.array([True]), _xy([]), 0.5) == {}


class TestLogit:

    def test_an_even_chance_is_zero_and_the_sign_follows_the_probability(self):
        assert logit(np.array([0.5]))[0] == 0.0
        assert logit(np.array([0.75]))[0] > 0.0 > logit(np.array([0.25]))[0]

    def test_certainty_is_clamped_rather_than_infinite(self):
        assert np.all(np.isfinite(logit(np.array([0.0, 1.0]))))
