"""What the handling queue runs, and that every command in it is one the parser accepts.

The previous DROW chain stopped three times in nine seconds because `step3a` was handed `--max-range-m`, which
is `step2`'s flag alone -- and the test guarding that queue asserted the flag's **presence as a string**, so it
passed while the chain could not start. The acceptance class below is the fix: every job is parsed by
`train_takheliped.py`'s own parser, so a flag that does not exist on the subcommand it is given fails here
rather than after the machine has been held for a night.

The flags a result was produced under are part of the result, so the ones that would silently change what a
number means are pinned too: the dataset, which defaults to `frog`; the range limit, because at FROG's 10 m the
run cannot represent 8% of DROW's labels; and `--beam-stride`, which is what makes the FROG arm a 0.5 deg arm
rather than a second copy of the headline chain.
"""
import sys
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import queue_drow_handling as q  # noqa: E402
from train_takheliped import build_parser, network_shape  # noqa: E402

DROW_JOBS = ["score_10m", "stage2_ignore"]
FROG_JOBS = ["frog_half_stage2", "frog_half_stage3"]
TRAINER_JOBS = DROW_JOBS + FROG_JOBS


def flags(name):
    job = next(j for j in q.JOBS if j.name == name)
    return job, list(job.cli)


def parse(name):
    """The job's command line through the trainer's own parser, which is what no earlier test did."""
    cli = flags(name)[1]
    assert cli[0] == "train_takheliped.py", f"{name} does not run the trainer"
    return build_parser().parse_args(cli[1:])


class TestEveryJobIsACommandTheParserAccepts:
    @pytest.mark.parametrize("name", TRAINER_JOBS)
    def test_it_parses(self, name):
        assert parse(name) is not None

    def test_an_unknown_flag_is_what_this_catches(self):
        """The exact failure that stopped the previous chain, so the guard itself is shown to work."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["step3a", "--stage2", "x.pth", "--max-range-m", "29.99"])


class TestTheDrowJobsReadDrowAtItsOwnSensorLimit:
    @pytest.mark.parametrize("name", DROW_JOBS)
    def test_the_dataset_is_named_because_the_flag_defaults_to_frog(self, name):
        assert parse(name).dataset == "drow"

    def test_the_ignore_arm_keeps_the_sensor_limit_the_headline_was_trained_at(self):
        """An arm at a different range limit would be a different network, not a different target set."""
        assert parse("stage2_ignore").max_range_m == 29.99

    def test_the_ignore_arm_is_the_only_thing_that_differs_from_the_headline_stage_two(self):
        import queue_drow

        headline = next(j for j in queue_drow.JOBS if j.name == "stage2")
        theirs = [f for f in headline.cli if not f.startswith(str(q.CHECKPOINTS))]
        ours = [f for f in flags("stage2_ignore")[1] if not f.startswith(str(q.CHECKPOINTS))]
        assert set(ours) - set(theirs) == {"--ignore-other-classes"}

    def test_the_arm_writes_somewhere_else_so_the_headline_stays_readable_beside_it(self):
        assert parse("stage2_ignore").out_dir != q.DROW_STAGE2


class TestTheTenMetreColumn:
    def test_it_is_a_scoring_job_and_trains_nothing(self):
        assert parse("score_10m").mode == "score"

    def test_it_asks_for_the_restriction_the_decision_named(self):
        assert parse("score_10m").eval_max_range_m == 10.0

    def test_it_scores_the_finished_cell_and_not_a_fresh_one(self):
        args = parse("score_10m")
        assert args.stage2.parent == q.DROW_STAGE2 and args.memory.parent == q.DROW_STAGE3

    def test_it_reruns_both_strides_the_headline_was_reported_at(self):
        """46.01% is stride 1 and 45.36% is stride 5; a column beside only one of them cannot be read."""
        assert parse("score_10m").test_strides == [1, 5]

    def test_it_writes_beside_the_other_results_rather_than_over_them(self):
        assert parse("score_10m").out.name == "drow_within_10m.json"


class TestTheFrogArmIsAtDrowsSpacing:
    @pytest.mark.parametrize("name", FROG_JOBS)
    def test_every_stage_of_the_chain_is_decimated(self, name):
        """Unlike the range limit, this one is not restored from the checkpoint -- `dataset_spec` reads the
        command line -- so a stage that omits it silently trains at 0.25 deg on 0.5 deg weights."""
        assert parse(name).beam_stride == 2

    @pytest.mark.parametrize("name", FROG_JOBS)
    def test_it_stays_on_frog_official_which_is_what_it_is_compared_against(self, name):
        args = parse(name)
        assert args.dataset == "frog" and args.frog_mode == "official"

    def test_the_range_limit_is_not_passed_because_frogs_default_is_already_its_own(self):
        assert "--max-range-m" not in flags("frog_half_stage2")[1]

    def test_stage_three_reads_the_decimated_stage_twos_own_weights(self):
        args = parse("frog_half_stage3")
        assert args.stage2.parent == q.FROG_HALF_STAGE2 and args.stage2.name == q.AVERAGED


class TestTheChainStopsRatherThanRunningOutOfOrder:
    def test_the_averaging_waits_for_the_stage_two_that_feeds_it(self):
        assert flags("frog_half_avg")[0].needs == ("frog_half_stage2",)

    def test_stage_three_waits_for_the_averaged_weights_it_is_given(self):
        assert flags("frog_half_stage3")[0].needs == ("frog_half_avg",)

    def test_the_two_cheap_decisive_jobs_run_before_the_six_hour_one(self):
        """Deliberate, and the whole shape of this queue: the arms that could change what is worth running next."""
        order = [j.name for j in q.JOBS]
        assert order.index("score_10m") < order.index("frog_half_stage2")
        assert order.index("stage2_ignore") < order.index("frog_half_stage2")

    def test_nothing_here_needs_a_checkpoint_the_previous_queue_has_not_written(self):
        assert (q.DROW_STAGE2 / q.AVERAGED).exists() and (q.DROW_STAGE3 / q.MEMORY).exists()

    def test_no_drow_stage_three_is_queued_at_all(self):
        """It waits for the multi-class head, so that one 7 h run buys the better of the two treatments."""
        assert not [j for j in q.JOBS if "step3a" in j.cli and "drow" in str(j.outputs)]


class TestEveryJobDeclaresWhatItWritesLast:
    """The runner waits 600 s after a declared output appears and then kills the child, so a job that names an
    **early** file is a job cut off mid-work -- that silently dropped a whole baseline method once. Each file
    named here is the last thing its subcommand writes: `step2_results.json` and `step3a_results.json` are
    written after the test pass, and `score`'s `--out` after its own."""

    @pytest.mark.parametrize("name", [j.name for j in q.JOBS])
    def test_it_names_at_least_one(self, name):
        assert flags(name)[0].outputs

    @pytest.mark.parametrize("name", ["stage2_ignore", "frog_half_stage2"])
    def test_a_stage_two_is_proven_done_by_its_results_and_not_by_a_checkpoint(self, name):
        """Checkpoints are written every quarter epoch, so one of those would end the job in its first minutes."""
        assert [p.name for p in flags(name)[0].outputs] == ["step2_results.json"]


class TestTheGateRunsTheSuiteFirst:
    def test_the_queue_will_not_start_on_a_red_suite(self):
        assert q.GATE.cli[:3] == ("-m", "pytest", "tests")

    def test_stage_threes_range_limit_still_comes_from_its_stage_two(self):
        """Why `step3a` takes no `--max-range-m`: it is restored with the weights, not passed beside them."""
        assert network_shape({"max_range_m": "29.99"})["max_range_m"] == 29.99
