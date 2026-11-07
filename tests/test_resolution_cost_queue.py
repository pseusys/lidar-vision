"""What the resolution-cost queue runs, and that every command in it is one its own parser accepts.

The DROW chain of 2026-11-05 stopped three times in nine seconds because `step3a` was handed `--max-range-m`,
which is `step2`'s flag alone -- and the test guarding that queue asserted the flag's **presence as a string**, so
it passed while the chain could not start. Every job here is therefore parsed by the parser that will actually
receive it, so a flag that does not exist on the subcommand it is given fails here rather than after the machine
has been held for a night.

The flags a result was produced under are part of the result, so the ones that would silently change what a number
means are pinned too: `--beam-stride`, which is what makes two of these jobs a 0.5 deg measurement rather than a
second copy of the 0.25 deg one; `--dataset`, which defaults to `frog`; and `--rank-only`, whose absence would
replace an averaged checkpoint that a trained stage 3 is already built on.
"""
import sys
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import average_checkpoints as avg  # noqa: E402
import queue_resolution_cost as q  # noqa: E402
import tracking_baselines as tb  # noqa: E402
from train_takheliped import build_parser as trainer_parser  # noqa: E402
from train_takheliped import network_shape  # noqa: E402

LATENCY_JOBS = ["latency_full", "latency_half"]
RERANK_JOBS = ["rerank_half", "rerank_drow", "rerank_balanced"]


def job(name):
    return next(j for j in q.JOBS if j.name == name)


def parse(name):
    """The job's command line through the parser that will receive it, which is what no earlier queue test did."""
    cli = list(job(name).cli)
    parser = {"tracking_baselines.py": tb.build_parser, "average_checkpoints.py": avg.build_parser,
              "train_takheliped.py": trainer_parser}[cli[0]]
    return parser().parse_args(cli[1:])


class TestEveryJobIsACommandItsOwnParserAccepts:
    @pytest.mark.parametrize("name", [j.name for j in q.JOBS])
    def test_it_parses(self, name):
        assert parse(name) is not None

    def test_an_unknown_flag_is_what_this_catches(self):
        """The exact failure that stopped the previous chain, so the guard itself is shown to work."""
        with pytest.raises(SystemExit):
            trainer_parser().parse_args(["step3a", "--stage2", "x.pth", "--max-range-m", "29.99"])

    def test_the_chunk_arm_does_not_pass_the_range_limit_because_it_comes_with_the_weights(self):
        assert "--max-range-m" not in job("drow_chunk35").cli
        assert network_shape({"max_range_m": "29.99"})["max_range_m"] == 29.99


class TestTheTimingIsTheCpuProtocolAndTheIdleHostComesFirst:
    @pytest.mark.parametrize("name", LATENCY_JOBS)
    def test_it_times_the_cpu_because_a_gpu_figure_does_not_answer_the_question(self, name):
        args = parse(name)
        assert args.device == "cpu" and args.threads == 4 and args.frames == 2000

    def test_the_two_arms_differ_only_in_the_sensor_and_the_checkpoints(self):
        """Anything else differing would price that instead of the resolution."""
        full, half = (set(job(n).cli) for n in LATENCY_JOBS)
        assert (half - full) >= {"--beam-stride", "2"}
        for flag in ("--device", "--threads", "--frames"):
            assert flag in full and flag in half

    def test_only_the_decimated_arm_is_decimated(self):
        assert parse("latency_half").beam_stride == 2
        assert parse("latency_full").beam_stride == 1

    def test_each_arm_writes_its_own_directory_because_the_filename_is_the_same(self):
        """`latency` names its file after device and thread count, so a shared --out silently keeps one arm."""
        assert parse("latency_full").out != parse("latency_half").out
        assert {p.name for n in LATENCY_JOBS for p in job(n).outputs} == {q.LATENCY_FILE}

    def test_nothing_touching_the_gpu_runs_before_the_timing(self):
        """The protocol is an idle host, so every other job waits on the second timing."""
        order = [j.name for j in q.JOBS]
        assert order[:2] == LATENCY_JOBS
        assert job("latency_half").needs == ("latency_full",)
        assert job("rerank_half").needs == ("latency_half",)


class TestTheRerunksReadTheirOwnValSplitAndKeepTheirWeights:
    @pytest.mark.parametrize("name", RERANK_JOBS)
    def test_it_ranks_only_so_a_trained_stage_threes_weights_stay_put(self, name):
        assert parse(name).rank_only is True

    def test_each_run_is_ranked_on_the_data_it_trained_on(self):
        assert parse("rerank_half").beam_stride == 2 and parse("rerank_half").dataset == "frog"
        assert parse("rerank_drow").dataset == "drow"
        assert parse("rerank_balanced").frog_mode == "balanced"

    def test_the_three_runs_are_exactly_the_three_that_were_ranked_wrongly(self):
        assert {parse(n).runs[0].name for n in RERANK_JOBS} == \
            {"frog_half_stage2_s0", "drow_stage2_s0", "balanced_stage2_s0"}

    def test_none_of_them_is_the_frog_official_default_which_was_always_correct(self):
        """Eight queues train FROG `official` at full resolution; their windows were never wrong."""
        for name in RERANK_JOBS:
            args = parse(name)
            assert (args.dataset, args.frog_mode, args.beam_stride) != ("frog", "official", 1)


class TestTheLongArmIsTheFirstLengthThatReachesASecondBurst:
    def test_it_is_thirty_five_seconds(self):
        """31.5 s is DROW's burst period; 10 s is 0.32 of one and 35 s is the first length spanning two."""
        assert parse("drow_chunk35").chunk_s == [35.0]

    def test_it_reads_the_headlines_averaged_stage_two_rather_than_training_a_new_one(self):
        args = parse("drow_chunk35")
        assert args.stage2.parent == q.DROW_STAGE2 and args.stage2.name == q.AVERAGED

    def test_it_writes_somewhere_else_so_the_ten_second_headline_stays_readable_beside_it(self):
        assert parse("drow_chunk35").out_dir == q.DROW_CHUNK35
        assert parse("drow_chunk35").out_dir != q.DROW_STAGE2

    def test_it_keeps_the_frozen_memory_the_headline_was_measured_with(self):
        """`--no-*` are `BooleanOptionalAction` negations, so the arm is frozen when all three read False."""
        args = parse("drow_chunk35")
        assert not args.exchange and not args.learned_match and not args.coast

    def test_it_runs_last_because_it_is_the_only_job_that_takes_hours(self):
        assert [j.name for j in q.JOBS][-1] == "drow_chunk35"


class TestEveryJobDeclaresWhatItWritesLast:
    """The runner waits 600 s after a declared output appears and then kills the child, so a job that names an
    **early** file is a job cut off mid-work -- that silently dropped a whole baseline method once."""

    @pytest.mark.parametrize("name", [j.name for j in q.JOBS])
    def test_it_names_at_least_one(self, name):
        assert job(name).outputs

    def test_the_chunk_arm_is_proven_done_by_its_results_and_not_by_a_checkpoint(self):
        """Checkpoints are written every epoch, so one of those would end a 7 h job in its first minutes."""
        assert [p.name for p in job("drow_chunk35").outputs] == ["step3a_results.json"]

    @pytest.mark.parametrize("name", RERANK_JOBS)
    def test_a_rerank_is_proven_done_by_the_ranking_it_writes(self, name):
        """`--rank-only` writes no checkpoint, so the ranking file is the only thing it could be waited on --
        and it must be the rank-only name, or the runner skips the job because the old wrong ranking is there."""
        assert [p.name for p in job(name).outputs] == [avg.RANK_ONLY_NAME]

    @pytest.mark.parametrize("name", RERANK_JOBS)
    def test_a_rerank_does_not_declare_the_file_the_wrong_ranking_already_wrote(self, name):
        """Found by `--list` on 2026-11-06: all three jobs read as already done."""
        assert not [p for p in job(name).outputs if p.name == avg.RANKING_NAME and p.exists()]


class TestTheGateRunsTheSuiteFirst:
    def test_the_queue_will_not_start_on_a_red_suite(self):
        assert q.GATE.cli[:3] == ("-m", "pytest", "tests")

    def test_the_inputs_this_queue_reads_all_exist_already(self):
        """Every job here reads a finished run; none of them trains a stage 2.

        The 0.5 deg averaged checkpoint is the one exception, and deliberately so: the re-ranking this queue ran
        showed its window moves, so `queue_half_corrected.py` regenerates that file and the superseded weights
        live beside it under their window's name. Either satisfies this precondition -- what must not happen is a
        job queued against weights that exist nowhere.
        """
        for path in (q.FULL_STAGE2 / q.AVERAGED, q.FULL_STAGE3 / q.MEMORY,
                     q.HALF_STAGE3 / q.MEMORY, q.DROW_STAGE2 / q.AVERAGED, q.BALANCED_STAGE2 / q.AVERAGED):
            assert path.exists(), path
        assert any((q.HALF_STAGE2 / name).exists() for name in
                   (q.AVERAGED, "step2_calibration.averaged.window0.50-1.00.pth")), q.HALF_STAGE2
