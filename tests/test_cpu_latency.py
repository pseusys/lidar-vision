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

from cpu_latency import (FROG_PERIOD_MS, markdown, merge, network_input, pending, per_frame_total,
                         summarise, total_of)


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


class TestWhatTheNetworkIsHanded:
    """Li2Former's `forward_one` is the one that takes a tensor; every other detector's takes numpy.

    The first sweep lost Li2Former's whole row to this -- `AttributeError: 'numpy.ndarray' object has no attribute
    'unsqueeze'`, 2,000 frames into the run and after the other eight had already been timed. The rule is small and the
    failure is silent-ish (one row goes blank), so it is pinned here rather than left to the next person to rediscover.
    """

    def test_li2former_is_handed_the_tensor_it_asks_for(self):
        import torch

        x = torch.zeros(4, 5, 48)

        assert network_input("li2former", x) is x

    def test_every_other_detector_is_handed_numpy(self):
        import numpy as np
        import torch

        handed = network_input("drow", torch.zeros(4, 5, 48))

        assert isinstance(handed, np.ndarray)
        assert handed.shape == (4, 5, 48)


class TestTheTotalIsAFramesCost:
    """A deployment pays all the phases on the *same* frame, so the total's tail is a percentile of per-frame sums.

    Summing each phase's own p99 answers a different question -- how bad could a frame be if every phase had its worst
    frame at once -- and overstates the tail, because the slow preprocess frame and the slow network frame are rarely
    the same frame. On this host that mistake read TAKHeLiPeD's tail as 27.90 ms (112% of the 25 ms budget) where the
    measurement says 25.32 (101%): the difference between 'misses the budget badly' and 'misses it barely'.
    """

    def test_the_tail_is_over_frames_not_over_phases(self):
        # Two frames, and the phases take turns being the slow one: no frame ever costs 10 + 10.
        phases = dict(network=[10.0, 1.0], decode=[1.0, 10.0])

        row = per_frame_total(phases)

        assert row["median"] == 11.0
        assert row["p99"] == 11.0

    def test_the_median_still_adds_when_the_phases_peak_together(self):
        phases = dict(network=[4.0, 4.0, 4.0], decode=[1.0, 1.0, 1.0])

        assert per_frame_total(phases)["median"] == 5.0

    def test_the_budget_share_comes_along(self):
        phases = dict(network=[FROG_PERIOD_MS] * 4)

        assert per_frame_total(phases)["budget_median"] == pytest.approx(1.0)

    def test_phases_of_unequal_length_are_refused_rather_than_zipped_short(self):
        """Truncating would silently drop the frames one phase recorded and the other did not."""
        with pytest.raises(ValueError):
            per_frame_total(dict(network=[1.0, 2.0], decode=[1.0]))

    def test_no_phases_is_an_empty_row(self):
        assert per_frame_total({})["median"] is None


class TestAMeasuredTotalIsNotRecomputed:
    def test_a_row_that_carries_its_own_total_keeps_it(self):
        """TAKHeLiPeD's stages are merged from a run that timed the whole frame end to end; that total is the real one."""
        row = dict(phases=dict(network=dict(median=8.71, p99=11.0)), total=dict(median=20.98, p99=25.32))

        assert total_of(row)["p99"] == 25.32


class TestTheMarkdownTable:
    """The table goes into `memory/performance-log.md` and will be regenerated after the thread sweep (item 30).

    Transcribing seventy cells by hand once is tedious; twice is how a digit changes without anyone deciding it should.
    """

    @staticmethod
    def rows():
        return {
            "drow": dict(phases=dict(preprocess=dict(median=2.26), network=dict(median=318.22), decode=dict(median=1.36)),
                         total=dict(median=321.84, p99=346.92, budget_p99=13.88)),
            "lfe_peaks": dict(phases=dict(network=dict(median=1.11)), total=dict(median=1.11, p99=1.67, budget_p99=0.067)),
            "li2former": dict(phases={}, total=dict(median=None, p99=None, budget_p99=None), failed="AttributeError: no unsqueeze"),
        }

    def test_the_cheapest_detector_comes_first(self):
        lines = [line for line in markdown(self.rows(), threads=4).splitlines() if line.startswith("| ")]

        assert "LFE-Peaks" in lines[2]

    def test_a_phase_a_detector_does_not_have_is_blank_not_zero(self):
        peaks = next(line for line in markdown(self.rows(), threads=4).splitlines() if "LFE-Peaks" in line)

        assert "0.00" not in peaks
        assert peaks.count("--") >= 2        # no preprocess of its own, no separate decode

    def test_a_detector_that_could_not_be_timed_says_so_in_its_row(self):
        row = next(line for line in markdown(self.rows(), threads=4).splitlines() if "Li2Former" in line)

        assert "AttributeError" in row

    def test_the_budget_column_is_the_tails_share_of_the_frame_period(self):
        drow = next(line for line in markdown(self.rows(), threads=4).splitlines() if "DROW" in line)

        assert "1388%" in drow

    def test_the_thread_count_is_stated_because_it_changes_the_answer(self):
        assert "4 threads" in markdown(self.rows(), threads=4)

    def test_rows_carry_the_published_names_not_the_command_line_keys(self):
        table = markdown(self.rows(), threads=4)

        assert "DR-SPAAM" not in table        # it was not among these rows
        assert "lfe_peaks" not in table and "LFE-Peaks" in table


class TestARunThatCanBeResumed:
    """A sweep of nine detectors takes ~110 minutes and used to write nothing until the last one finished.

    Killing it at minute 90 therefore discarded seven completed rows -- which is exactly what happened on 2026-10-11,
    to check a thread-pool anomaly that turned out not to matter. Rows are now written as each detector finishes, and a
    relaunch times only what is missing.
    """

    DONE = {"drow": dict(phases={}, total=dict(median=321.84, p99=346.92)),
            "li2former": dict(phases={}, total=dict(median=None, p99=None), failed="AttributeError")}

    def test_an_empty_output_means_everything_is_pending(self):
        assert pending(["drow", "drspaam"], {}) == ["drow", "drspaam"]

    def test_a_finished_detector_is_not_timed_again(self):
        assert pending(["drow", "drspaam"], self.DONE) == ["drspaam"]

    def test_a_detector_that_failed_is_retried_rather_than_counted_as_done(self):
        """Its row exists but holds no timing, and a harness bug fixed since is the likeliest reason it is there."""
        assert "li2former" in pending(["li2former", "drow"], self.DONE)

    def test_redo_times_everything_again(self):
        assert pending(["drow"], self.DONE, redo=True) == ["drow"]

    def test_the_requested_order_is_kept(self):
        assert pending(["drspaam", "algorithmic", "drow"], self.DONE) == ["drspaam", "algorithmic"]

    def test_rows_from_an_earlier_run_survive_a_narrower_one(self):
        """Re-timing one detector must not delete the other eight: the output is the table, not this invocation's slice."""
        merged = merge(self.DONE, {"drspaam": dict(phases={}, total=dict(median=1197.72))})

        assert set(merged) == {"drow", "drspaam", "li2former"}
        assert merged["drow"]["total"]["median"] == 321.84

    def test_a_fresh_row_replaces_the_stale_one_it_re_times(self):
        merged = merge(self.DONE, {"drow": dict(phases={}, total=dict(median=319.55))})

        assert merged["drow"]["total"]["median"] == 319.55
