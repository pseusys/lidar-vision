"""The corrected 0.5 deg chain: one variable differs from the run it replaces, and the old weights survive.

This chain exists to turn the 0.88 pp resolution deficit from an upper bound into a figure, so its whole value
rests on differing from `queue_drow_handling.py`'s `frog_half_*` chain in **exactly one** respect: the averaging
window, which a correct ranking moves from 0.50-1.00 to 0.50-2.00. A second difference -- a seed, a chunk length,
an unfrozen memory -- would make the new number incomparable with the one it is meant to sharpen, which is the
whole reason the comparison is being redone.
"""
import sys
from hashlib import sha256
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import average_checkpoints as avg  # noqa: E402
import queue_drow_handling as old  # noqa: E402
import queue_half_corrected as q  # noqa: E402
from train_takheliped import build_parser as trainer_parser  # noqa: E402


AVERAGED_NAME = "step2_calibration.averaged.pth"

#: The run this chain replaces, whose directory stage 3 must not write over.
SUPERSEDED_STAGE3 = q.CHECKPOINTS / "frog_half_stage3_s0"


def job(name, jobs=None):
    return next(j for j in (jobs if jobs is not None else q.JOBS) if j.name == name)


def parse(name):
    cli = list(job(name).cli)
    parser = {"average_checkpoints.py": avg.build_parser, "train_takheliped.py": trainer_parser}[cli[0]]
    return parser().parse_args(cli[1:])


class TestEveryJobIsACommandItsOwnParserAccepts:
    @pytest.mark.parametrize("name", [j.name for j in q.JOBS])
    def test_it_parses(self, name):
        assert parse(name) is not None

    def test_stage_three_does_not_pass_the_range_limit_because_it_comes_with_the_weights(self):
        assert "--max-range-m" not in job("half_stage3_fixed").cli


class TestExactlyOneThingDiffersFromTheRunThisReplaces:
    """The arm being sharpened is `queue_drow_handling.py`'s `frog_half_stage3`."""

    def test_stage_three_differs_only_in_where_it_writes(self):
        mine = [f for f in job("half_stage3_fixed").cli if "frog_half_stage3" not in str(f)]
        theirs = [f for f in job("frog_half_stage3", old.JOBS).cli if "frog_half_stage3" not in str(f)]
        assert mine == theirs

    def test_the_seed_is_unchanged_because_this_is_not_a_second_seed(self):
        assert parse("half_stage3_fixed").seed == 0

    def test_the_chunk_length_is_unchanged(self):
        assert parse("half_stage3_fixed").chunk_s == [10.0]

    def test_the_memory_stays_frozen_the_way_both_arms_were_measured(self):
        args = parse("half_stage3_fixed")
        assert not args.exchange and not args.learned_match and not args.coast

    def test_both_stages_are_still_decimated(self):
        assert parse("half_avg").beam_stride == 2 and parse("half_stage3_fixed").beam_stride == 2


class TestTheAveragingIsTheOnePassMeantToReplaceTheCheckpoint:
    def test_it_is_not_rank_only_because_the_new_weights_are_the_point(self):
        assert parse("half_avg").rank_only is False

    def test_it_ranks_on_the_runs_own_data_which_is_the_entire_correction(self):
        args = parse("half_avg")
        assert (args.dataset, args.frog_mode, args.beam_stride) == ("frog", "official", 2)

    def test_it_averages_the_run_whose_window_moved(self):
        assert parse("half_avg").runs == [q.HALF_STAGE2]


class TestThePublishedResultsProvenanceSurvives:
    def test_the_weights_the_published_number_was_trained_on_are_kept_aside(self):
        """Re-averaging overwrites `step2_calibration.averaged.pth`, so 85.42%'s input must exist under its own
        name before this chain runs -- otherwise a published figure has no reproducible input."""
        assert (q.HALF_STAGE2 / q.SUPERSEDED).exists()

    def test_the_superseded_name_says_which_window_it_holds(self):
        assert "0.50-1.00" in q.SUPERSEDED

    def test_the_live_averaged_checkpoint_is_never_the_stale_copy(self):
        """The runner skips a job whose declared output already exists. `half_avg` declares the averaged
        checkpoint because replacing it is the point, so while the stale copy sits there the job is skipped and
        stage 3 silently retrains on the **wrong** weights -- a duplicate of the run this chain replaces rather
        than its correction. Caught by `--list` on 2026-11-06, for the second time that day.

        Stated as the invariant rather than as the precondition, because the precondition stopped being true the
        moment the chain ran: absent before the re-average, present and *different* after it. Either passes; the
        stale copy restored over the live name is the one state that must not hold.
        """
        live = q.HALF_STAGE2 / AVERAGED_NAME
        if not live.exists():
            return
        assert sha256(live.read_bytes()).hexdigest() != sha256((q.HALF_STAGE2 / q.SUPERSEDED).read_bytes()).hexdigest(), \
            "the live averaged checkpoint is the superseded window's copy; re-average or the chain is skipped"

    def test_stage_three_writes_beside_the_run_it_replaces_rather_than_over_it(self):
        assert parse("half_stage3_fixed").out_dir == q.HALF_STAGE3_FIXED
        assert parse("half_stage3_fixed").out_dir != SUPERSEDED_STAGE3


class TestEveryJobDeclaresWhatItWritesLast:
    @pytest.mark.parametrize("name", [j.name for j in q.JOBS])
    def test_it_names_at_least_one(self, name):
        assert job(name).outputs

    def test_stage_three_is_proven_done_by_its_results_and_not_by_a_checkpoint(self):
        assert [p.name for p in job("half_stage3_fixed").outputs] == ["step3a_results.json"]


class TestTheGateRunsTheSuiteFirst:
    def test_the_queue_will_not_start_on_a_red_suite(self):
        assert q.GATE.cli[:3] == ("-m", "pytest", "tests")

    def test_the_periodic_checkpoints_the_averaging_needs_are_on_disk(self):
        assert len(list(q.HALF_STAGE2.glob("step2_calibration.epoch*.pth"))) > 1
