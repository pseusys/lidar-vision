"""The statistics behind `utils/cpu_latency.py` (`TODO.md` §A item 29).

`AGENTS.md` rule 2b makes speed half of every accuracy answer, and the speed table in `memory/performance-log.md` could not
honour it: its CPU rows were taken around 2026-09-10 by a different script than the one that timed this detector, at an unstated
thread count and with no percentile. It satisfied its own stated rule -- one machine -- while breaking the one it needed, one
protocol. So every row is re-timed together, and what gets reported is what a deployment feels: the median, the tail, and the
share of the frame period each consumes.

The timing loop itself is exercised against real processes by the script's own smoke run; what is pinned here is the arithmetic
that turns a list of per-frame durations into a row, because that is where a wrong answer would be quiet.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from cpu_latency import FROG_PERIOD_MS, summarise


class TestSummarise:
    def test_the_median_is_the_middle_not_the_mean(self):
        row = summarise([1.0, 1.0, 1.0, 1.0, 100.0])

        assert row["median"] == 1.0
        assert row["mean"] == pytest.approx(20.8)

    def test_the_tail_is_reported_because_the_median_hides_it(self):
        durations = [10.0] * 95 + [40.0] * 5

        row = summarise(durations)

        assert row["median"] == 10.0
        assert row["p99"] == 40.0

    def test_p99_is_the_value_99_percent_of_frames_come_in_under(self):
        """So a single slow frame in a hundred is the maximum, not the 99th percentile -- it must not set the tail alone."""
        assert summarise([10.0] * 99 + [40.0])["p99"] == 10.0
        assert summarise([10.0] * 90 + [40.0] * 10)["p99"] == 40.0

    def test_the_budget_share_is_against_frogs_real_frame_period(self):
        row = summarise([FROG_PERIOD_MS / 2] * 10)

        assert row["budget_median"] == pytest.approx(0.5)
        assert row["budget_p99"] == pytest.approx(0.5)

    def test_a_detector_over_budget_is_reported_as_over_rather_than_clipped(self):
        row = summarise([FROG_PERIOD_MS * 3] * 10)

        assert row["budget_median"] == pytest.approx(3.0)

    def test_the_frame_count_travels_with_the_row(self):
        assert summarise([1.0, 2.0, 3.0])["frames"] == 3

    def test_no_timings_is_an_empty_row_rather_than_a_crash(self):
        row = summarise([])

        assert row["frames"] == 0
        assert row["median"] is None
        assert row["budget_median"] is None


class TestTheBudgetItself:
    def test_the_period_is_frogs_real_one_not_the_stamped_median(self):
        """40 Hz, settled in item 1: the median stamped interval said 26.2 Hz and made every budget 1.53x too generous."""
        assert FROG_PERIOD_MS == pytest.approx(25.0, abs=0.1)
