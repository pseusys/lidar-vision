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

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from memory_diagnosis import logit, pairs


def _xy(points):
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


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
