"""Item 37: a stream with dropped frames, so the dt-aware design has something to be aware of.

Item 16 found that the three recurrences tie, and the prerequisite check found why that was predictable: `frame_periods`
hands stage 3 an **exponential running average of the stamped intervals, rate 1/32**, clipped to `[MIN_DT_S, MAX_DT_S]`.
That averaging is right -- FROG stamps 40 Hz scans in bunches of 38 ms / 38 ms / under 0.1 ms, and dividing by one of
those sent slot velocities to several m/s (A51 item 0a) -- but it means the model has never seen a per-frame `dt`, so
`exp(-dt * rate)` has been a fixed per-slot decay factor by construction.

Hence two `dt` modes, which answer different questions and are both worth having:

    exact      the gap reaches `dt` as `(k+1) x` the averaged rate. Legitimate because the drops are synthesised, so the
               gap is known exactly rather than read off a bunched stamp. Asks: **is the dt-aware mechanism worth
               anything when it actually gets dt?**
    averaged   the decimated stamps go back through `frame_periods`, which is what the shipped pipeline would do. Asks:
               **does today's pipeline handle dropouts better with the adopted rule?**

The gap between the two answers is itself a result: it prices what the averaging throws away.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from train_takheliped import (  # noqa: E402
    DT_AVERAGE_FRAMES, MAX_DT_S, MIN_DT_S, SequenceCandidates, Split, decimate, frame_periods, pack,
)
from slot_experiments import bernoulli_keep, burst_keep, periodic_keep  # noqa: E402

PERIOD_S = 0.025        # FROG's 40.01 Hz
SECTORS = 90


def _recording(frames: int, people: int = 2, period: float = PERIOD_S, start: float = 0.0) -> SequenceCandidates:
    """One synthetic recording with uniform stamps, enough fields that `pack` and `decimate` have something to carry."""
    t = np.arange(frames, dtype=np.float64) * period + start
    return SequenceCandidates(
        xy=np.zeros((frames, 4, 2), np.float32), score=np.zeros((frames, 4), np.float32),
        valid=np.zeros((frames, 4), bool), target=np.zeros((frames, 4), np.float32),
        pose=np.zeros((frames, 3), np.float32), time=t, annotated=np.ones(frames, bool),
        gt_xy=np.zeros((frames, people, 2), np.float32), gt_valid=np.ones((frames, people), bool),
        gt_covered=np.zeros((frames, people), bool), features=np.zeros((frames, 4, 8), np.float32),
        free_range=np.full((frames, SECTORS), 5.0, np.float32),
    )


def _split(lengths=(40, 25)) -> Split:
    return pack([_recording(n, start=100.0 * i) for i, n in enumerate(lengths)])


def _frames_of(split: Split, recording: int) -> slice:
    o = int(split.offsets[recording])
    return slice(o, o + int(split.lengths[recording]))


class TestKeepEverything:
    """The sweep's zero point has to be the baseline exactly, or every severity is measured against a different model."""

    def test_keeping_every_frame_reproduces_the_split_in_both_modes(self):
        split = _split()
        keep = np.ones(len(split.data.time), bool)

        for mode in ("exact", "averaged"):
            out = decimate(split, keep, mode)

            assert np.array_equal(out.lengths, split.lengths)
            assert np.array_equal(out.offsets, split.offsets)
            assert np.allclose(out.dt, split.dt, atol=0)
            assert np.array_equal(out.data.time, split.data.time)

    def test_it_is_a_new_split_and_leaves_the_original_alone(self):
        split = _split()
        before = split.dt.copy()
        keep = np.ones(len(split.data.time), bool)
        keep[5] = False

        decimate(split, keep, "exact")

        assert np.allclose(split.dt, before, atol=0)
        assert len(split.data.time) == 65


class TestExactGaps:
    """`dt` after dropping k frames must be (k+1) x the period, which is the whole point of the mode."""

    def test_one_dropped_frame_doubles_the_step(self):
        split = _split((40,))
        keep = np.ones(40, bool)
        keep[10] = False

        out = decimate(split, keep, "exact")

        # frame 9 is kept, 10 dropped, 11 kept -> the step onto 11 spans two periods.
        assert out.dt[10] == pytest.approx(2 * split.dt[11], rel=1e-9)
        assert out.dt[9] == pytest.approx(split.dt[9], rel=1e-9)

    def test_a_burst_of_k_gives_k_plus_one_periods(self):
        split = _split((60,))
        for k in (1, 2, 5, 19):
            keep = np.ones(60, bool)
            keep[20:20 + k] = False

            out = decimate(split, keep, "exact")
            step = out.dt[20]        # survivors 0..19 are frames 0..19, so position 20 is the frame after the burst

            assert step == pytest.approx((k + 1) * split.dt[20 + k], rel=1e-9), k

    def test_a_gap_longer_than_the_clamp_is_clipped_rather_than_wrapped(self):
        """MAX_DT_S is 1.0 s, so a burst past 40 frames at 40 Hz saturates -- a sweep must stay under it or say so."""
        split = _split((120,))
        keep = np.ones(120, bool)
        keep[10:70] = False         # 60 frames = 1.5 s, past the clamp

        out = decimate(split, keep, "exact")

        assert out.dt.max() == pytest.approx(MAX_DT_S)

    def test_the_first_surviving_frame_of_a_recording_has_no_elapsed_time(self):
        split = _split()
        keep = np.ones(len(split.data.time), bool)
        keep[[0, 1, 40, 41]] = False        # both recordings lose their opening frames

        out = decimate(split, keep, "exact")

        for r in range(len(out.lengths)):
            assert out.dt[int(out.offsets[r])] == pytest.approx(MIN_DT_S)

    def test_gaps_do_not_leak_across_a_recording_boundary(self):
        """Recordings are packed end to end and their stamps are 100 s apart; a gap must be measured inside one."""
        split = _split((40, 25))
        keep = np.ones(65, bool)
        keep[39] = False                    # the last frame of recording 0

        out = decimate(split, keep, "exact")

        assert out.dt[int(out.offsets[1])] == pytest.approx(MIN_DT_S)
        assert out.dt.max() < 3 * PERIOD_S


class TestAveragedGaps:
    """The shipped pipeline's answer, and the measurement that shows why `exact` has to exist."""

    def test_it_is_what_frame_periods_computes_from_the_surviving_stamps(self):
        split = _split()
        keep = np.ones(len(split.data.time), bool)
        keep[[7, 8, 30]] = False

        out = decimate(split, keep, "averaged")

        assert np.allclose(out.dt, frame_periods(out.data.time, out.offsets, out.lengths, DT_AVERAGE_FRAMES), atol=0)

    def test_a_single_drop_barely_moves_the_averaged_step(self):
        """Under a millisecond, against the 25 ms the exact mode would add: this is why a naive sweep measures nothing."""
        split = _split((200,))
        keep = np.ones(200, bool)
        keep[100] = False

        averaged = decimate(split, keep, "averaged")
        exact = decimate(split, keep, "exact")

        assert averaged.dt[100] - split.dt[100] < 0.001
        assert exact.dt[100] == pytest.approx(2 * PERIOD_S, rel=1e-3)

    def test_a_long_burst_does_reach_the_average_but_smeared_over_many_frames(self):
        """The averaging is not blind to a real outage, it just spreads it -- which is the comparison, not a bug."""
        split = _split((300,))
        keep = np.ones(300, bool)
        keep[100:120] = False       # 0.5 s outage

        averaged = decimate(split, keep, "averaged")
        exact = decimate(split, keep, "exact")

        assert exact.dt[100] == pytest.approx(21 * PERIOD_S, rel=1e-3)
        assert averaged.dt[100] < 0.1 * exact.dt[100]
        assert averaged.dt[100] > split.dt[100]


class TestTheSplitStaysWellFormed:
    def test_every_carried_array_is_subset_in_order(self):
        split = _split((40,))
        split.data.score[:, 0] = np.arange(40, dtype=np.float32)
        keep = np.ones(40, bool)
        keep[[3, 17, 18]] = False

        out = decimate(split, keep, "exact")

        assert np.array_equal(out.data.score[:, 0], np.arange(40, dtype=np.float32)[keep])
        assert np.array_equal(out.data.time, split.data.time[keep])
        assert out.data.free_range.shape == (37, SECTORS)

    def test_lengths_and_offsets_describe_the_survivors_per_recording(self):
        split = _split((40, 25))
        keep = np.ones(65, bool)
        keep[:10] = False           # recording 0 loses ten
        keep[60:] = False           # recording 1 loses five

        out = decimate(split, keep, "exact")

        assert list(out.lengths) == [30, 20]
        assert list(out.offsets) == [0, 30]

    def test_a_recording_that_loses_every_frame_is_dropped_rather_than_kept_empty(self):
        """A zero-length recording would make `evaluation_segments` hand `evaluate` an empty stream."""
        split = _split((40, 25))
        keep = np.ones(65, bool)
        keep[40:] = False

        out = decimate(split, keep, "exact")

        assert list(out.lengths) == [40]
        assert len(out.data.time) == 40

    def test_an_unknown_dt_mode_is_refused(self):
        split = _split((10,))
        with pytest.raises(ValueError, match="dt_mode"):
            decimate(split, np.ones(10, bool), "stamped")

    def test_a_mask_of_the_wrong_length_is_refused(self):
        split = _split((10, 5))
        with pytest.raises(ValueError, match="keep"):
            decimate(split, np.ones(12, bool), "exact")


class TestDropPatterns:
    """Three shapes of outage, each confined to its own recording."""

    def test_periodic_drops_every_nth_frame_of_each_recording(self):
        split = _split((40, 25))

        keep = periodic_keep(split, 5)

        for r in range(2):
            inside = keep[_frames_of(split, r)]
            assert not inside[4::5].any()
            assert inside[[i for i in range(len(inside)) if i % 5 != 4]].all()

    def test_bernoulli_drops_about_the_share_asked_for_and_is_reproducible(self):
        split = _split((2000,))

        a = bernoulli_keep(split, 0.25, seed=0)
        b = bernoulli_keep(split, 0.25, seed=0)

        assert np.array_equal(a, b)
        assert 0.22 < 1 - a.mean() < 0.28
        assert not np.array_equal(a, bernoulli_keep(split, 0.25, seed=1))

    def test_a_burst_drops_consecutive_frames(self):
        split = _split((100,))

        keep = burst_keep(split, length=8, period=40)

        runs = np.diff(np.flatnonzero(np.r_[True, np.diff(keep.astype(int)) != 0, True]))
        assert 8 in runs
        assert keep.sum() == 100 - 2 * 8

    def test_no_pattern_drops_a_recording_s_first_frame(self):
        """The first frame carries no elapsed time anyway, and dropping it only shortens the warm-up."""
        split = _split((40, 25))

        for keep in (periodic_keep(split, 2), burst_keep(split, 5, 10), bernoulli_keep(split, 0.5, seed=3)):
            for r in range(2):
                assert keep[int(split.offsets[r])]

    def test_every_pattern_keeps_something_in_every_recording(self):
        split = _split((40, 25))

        for keep in (periodic_keep(split, 2), burst_keep(split, 5, 10), bernoulli_keep(split, 0.5, seed=3)):
            for r in range(2):
                assert keep[_frames_of(split, r)].sum() > 0
