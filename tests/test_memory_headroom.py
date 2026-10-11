"""What only the memory can do, and where the shrinking-gain trend has to stop (`TODO.md` §A item 23).

The memory's advantage falls as stage 2 improves: +5.68 pp of AP over candidates scoring 77.66%, +3.68 over 82.2%. Extrapolated
it reaches zero, and a reader is entitled to ask where that stops being true. It stops at whatever a single-frame detector
*cannot* do however good it gets, and the occlusion strata measure exactly that: a person whose returns are blocked is not in
the scan, so no static detector finds them, while a visible person missed by our stage 2 is a person a better stage 2 would
find.

So the memory's recall gain splits into a substitutable part and an irreplaceable one, and the weights are the strata's shares
of the population. That arithmetic is what is pinned here; the measurements it runs on come from A61's reports.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from memory_headroom import decompose, static_ceiling, trend_crosses_zero_at


def _strata(**kwargs):
    """`{name: (people, candidate recall, memory recall)}`."""
    return {name: dict(people=p, candidates=c, memory=m) for name, (p, c, m) in kwargs.items()}


class TestDecompose:
    def test_each_stratum_contributes_its_share_of_its_own_gain(self):
        strata = _strata(visible=(900, 0.90, 0.95), occluded=(100, 0.50, 0.70))

        out = decompose(strata, irreplaceable=("occluded",))

        assert out["gain"] == pytest.approx(0.9 * 0.05 + 0.1 * 0.20)
        assert out["irreplaceable"] == pytest.approx(0.1 * 0.20)
        assert out["substitutable"] == pytest.approx(0.9 * 0.05)

    def test_the_two_parts_sum_to_the_whole(self):
        strata = _strata(a=(500, 0.8, 0.9), b=(300, 0.7, 0.75), c=(200, 0.4, 0.6))

        out = decompose(strata, irreplaceable=("c",))

        assert out["substitutable"] + out["irreplaceable"] == pytest.approx(out["gain"])

    def test_the_irreplaceable_share_is_reported_as_a_fraction_of_the_gain(self):
        strata = _strata(visible=(900, 0.90, 0.90), occluded=(100, 0.50, 0.70))

        out = decompose(strata, irreplaceable=("occluded",))

        assert out["irreplaceable_share"] == pytest.approx(1.0)      # the whole gain came from the blocked people

    def test_a_stratum_the_memory_makes_worse_subtracts(self):
        strata = _strata(visible=(900, 0.95, 0.93), occluded=(100, 0.50, 0.70))

        out = decompose(strata, irreplaceable=("occluded",))

        assert out["substitutable"] < 0
        assert out["gain"] < out["irreplaceable"]

    def test_an_unknown_irreplaceable_name_is_refused_rather_than_ignored(self):
        with pytest.raises(ValueError, match="hidden"):
            decompose(_strata(visible=(10, 0.5, 0.6)), irreplaceable=("hidden",))


class TestStaticCeiling:
    def test_a_perfect_single_frame_detector_reaches_everything_it_can_see(self):
        strata = _strata(visible=(900, 0.9, 0.9), partial=(80, 0.8, 0.85), occluded=(20, 0.5, 0.7))

        assert static_ceiling(strata, irreplaceable=("occluded",)) == pytest.approx(0.98)

    def test_with_nothing_hidden_the_ceiling_is_everything(self):
        assert static_ceiling(_strata(visible=(10, 0.5, 0.6)), irreplaceable=()) == pytest.approx(1.0)


class TestTrend:
    def test_a_falling_gain_crosses_zero_where_the_line_says(self):
        """Two points are enough for a line, and two is what we have; the test pins the arithmetic, not the prediction."""
        assert trend_crosses_zero_at([(77.66, 5.68), (82.19, 3.68)]) == pytest.approx(90.55, abs=0.05)

    def test_a_flat_trend_never_crosses(self):
        assert trend_crosses_zero_at([(70.0, 3.0), (80.0, 3.0)]) is None

    def test_a_rising_trend_never_crosses(self):
        assert trend_crosses_zero_at([(70.0, 2.0), (80.0, 4.0)]) is None

    def test_one_point_is_not_a_trend(self):
        assert trend_crosses_zero_at([(80.0, 3.0)]) is None
