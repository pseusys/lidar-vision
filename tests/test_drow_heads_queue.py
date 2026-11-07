"""What the DROW head queue runs: every command parsed by the parser that will receive it, and one variable per arm.

Two queue failures are the reason this file exists. A chain stopped nine seconds in because `step3a` was handed
`--max-range-m`, which is `step2`'s flag alone, while the test guarding it asserted the flag's presence as a
*string*; and three jobs read as already done because the runner skips a job whose declared output exists. So
every job here goes through the real parser, and every declared output is the file its subcommand writes last.

The arms are pinned one flag at a time, because an arm that differs from the headline in two things prices
neither: `--beam-stride` must be absent (DROW is read at its own 450 beams), the recipe flags must match
`queue_drow.py`'s exactly, and the seed must be 0 everywhere.
"""
import sys
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import average_checkpoints as avg  # noqa: E402
import queue_drow as headline  # noqa: E402
import queue_drow_heads as q  # noqa: E402
from train_takheliped import build_parser as trainer_parser  # noqa: E402

STAGE2_JOBS = ["heads_softmax", "heads_sigmoid", "range_full", "range10", "gather", "kernel5", "kernel3"]


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

    @pytest.mark.parametrize("name", STAGE2_JOBS)
    def test_only_stage_two_carries_the_range_limit_because_it_shapes_the_network(self, name):
        assert "--max-range-m" in job(name).cli

    def test_stage_three_does_not_carry_it_which_is_what_stopped_a_chain_once(self):
        assert "--max-range-m" not in job("heads_stage3").cli
        with pytest.raises(SystemExit):
            trainer_parser().parse_args(["step3a", "--stage2", "x.pth", "--max-range-m", "29.99"])


class TestTheRegressionArmRunsFirstAndReadsTheFrozenChain:
    def test_it_is_the_first_job_and_every_arm_waits_on_it(self):
        """New code on the FROG path: if it moved, no arm trained after it means anything."""
        assert [j.name for j in q.JOBS][0] == "frog_regression"
        assert job("heads_softmax").needs == ("frog_regression",)

    def test_it_scores_the_checkpoints_whose_numbers_it_must_reproduce(self):
        args = parse("frog_regression")
        assert args.stage2 == q.FROG_STAGE2 and args.memory == q.FROG_MEMORY

    def test_it_scores_at_stride_one_because_that_is_the_recorded_figure(self):
        assert parse("frog_regression").test_strides == [1]

    def test_it_is_frog_rather_than_the_dataset_the_rest_of_this_queue_runs_on(self):
        assert parse("frog_regression").dataset == "frog"

    def test_it_writes_its_own_file_rather_than_the_shared_score_default(self):
        assert job("frog_regression").outputs == (q.REGRESSION,)
        assert parse("frog_regression").out == q.REGRESSION


class TestTheHeadArmsAreTheOwnersTwoAndNothingElseMoves:
    @pytest.mark.parametrize("name,norm", [("heads_softmax", "softmax"), ("heads_sigmoid", "sigmoid")])
    def test_each_arm_is_the_multiclass_head_at_its_own_normalisation(self, name, norm):
        args = parse(name)
        assert args.head == "multiclass" and args.class_norm == norm

    def test_the_two_arms_differ_in_the_normalisation_alone(self):
        soft = [f for f in job("heads_softmax").cli if "softmax" not in str(f) and "drow_heads" not in str(f)]
        sig = [f for f in job("heads_sigmoid").cli if "sigmoid" not in str(f) and "drow_heads" not in str(f)]
        assert soft == sig

    def test_the_head_arms_do_not_also_ignore_the_classes_they_are_learning(self):
        """The mask and the head are alternative treatments of one question; together they price neither."""
        for name in ("heads_softmax", "heads_sigmoid"):
            assert not parse(name).ignore_other_classes

    def test_the_class_count_is_not_a_flag(self):
        """It is the dataset's, read off the spec and stored in the checkpoint, so no arm can disagree with it."""
        for name in ("heads_softmax", "heads_sigmoid"):
            assert "--classes" not in job(name).cli


class TestTheRangeArmsAreReadableAgainstEachOther:
    def test_one_arm_trains_at_drows_limit_and_one_at_frogs(self):
        assert parse("range_full").max_range_m == 29.99
        assert parse("range10").max_range_m == 10.0

    def test_both_report_the_restricted_column_or_neither_comparison_exists(self):
        for name in ("range_full", "range10"):
            assert parse(name).eval_max_range_m == 10.0

    def test_the_full_range_arm_is_the_headline_recipe_again_and_not_a_new_one(self):
        """Its purpose is a matched restricted column, so anything else differing would spoil it."""
        mine = [f for f in job("range_full").cli if "drow_range_full" not in str(f) and f != "--eval-max-range-m"
                and f != "10"]
        theirs = [f for f in job("stage2", headline.JOBS).cli if "drow_stage2_s0" not in str(f)]
        assert mine == theirs

    def test_the_ten_metre_arm_differs_from_it_in_the_limit_alone(self):
        mine = [f for f in job("range10").cli if "drow_range10" not in str(f)]
        theirs = [f for f in job("range_full").cli if "drow_range_full" not in str(f)]
        assert [f for f in mine if f not in ("10", "29.99")] == [f for f in theirs if f not in ("10", "29.99")]


class TestTheTwoGeometryArmsAreTheFlagsThatAlreadyExist:
    def test_the_gather_is_the_width_frog_was_measured_at(self):
        """1.66 m cost 1.42 pp at stage 2 on FROG; the same window on a coarser sensor is the whole question."""
        assert parse("gather").gather_width_m == 1.66

    @pytest.mark.parametrize("name,size", [("kernel5", 5), ("kernel3", 3)])
    def test_each_kernel_arm_is_one_size_against_the_adopted_seven(self, name, size):
        assert parse(name).kernel_size == size

    def test_no_arm_decimates_the_beams_because_drow_is_read_at_its_own(self):
        for name in STAGE2_JOBS:
            assert "--beam-stride" not in job(name).cli


class TestTheExpensiveJobIsLastAndPreCommitted:
    def test_stage_three_runs_off_the_softmax_arms_averaged_weights(self):
        args = parse("heads_stage3")
        assert args.stage2.parent == q.HEADS_SOFTMAX and args.stage2.name == q.AVERAGED

    def test_the_averaging_is_told_the_dataset_its_run_trained_on(self):
        assert parse("heads_avg").dataset == "drow"
        assert parse("heads_avg").runs == [q.HEADS_SOFTMAX]

    def test_it_keeps_the_frozen_memory_and_the_adopted_chunk(self):
        args = parse("heads_stage3")
        assert args.chunk_s == [10.0] and not args.exchange and not args.learned_match and not args.coast

    def test_it_writes_beside_the_headline_run_rather_than_over_it(self):
        assert parse("heads_stage3").out_dir == q.HEADS_STAGE3
        assert parse("heads_stage3").out_dir != q.CHECKPOINTS / "drow_stage3_s0"

    def test_it_runs_last_because_every_cheap_screen_should_be_readable_before_it_starts(self):
        """The screens finishing first is what makes re-pointing this job a decision someone can still take."""
        assert [j.name for j in q.JOBS][-1] == "heads_stage3"
        assert job("heads_stage3").needs == ("heads_avg",)

    def test_every_job_is_a_strict_chain_so_two_arms_never_share_the_gpu(self):
        names = [j.name for j in q.JOBS]
        for earlier, later in zip(names[:-1], names[1:], strict=True):
            assert job(later).needs == (earlier,), f"{later} should wait on {earlier}"


class TestEveryJobDeclaresWhatItWritesLast:
    @pytest.mark.parametrize("name", [j.name for j in q.JOBS])
    def test_it_names_at_least_one_output(self, name):
        assert job(name).outputs

    @pytest.mark.parametrize("name", STAGE2_JOBS)
    def test_a_stage_two_arm_is_proven_done_by_its_results_file(self, name):
        """A checkpoint would be written in the first minutes and cut the job off 600 s later."""
        assert [p.name for p in job(name).outputs] == ["step2_results.json"]

    def test_no_two_arms_write_the_same_directory(self):
        """`heads_avg` is the exception and deliberately so: an averaged checkpoint belongs beside the run whose
        epochs it averages, which is why it is excluded here rather than given a directory of its own."""
        directories = [p.parent for name in STAGE2_JOBS + ["heads_stage3"] for p in job(name).outputs]
        assert len(set(directories)) == len(directories)
        assert job("heads_avg").outputs[0].parent == q.HEADS_SOFTMAX


class TestTheGateAndThePreconditions:
    def test_the_queue_will_not_start_on_a_red_suite(self):
        assert q.GATE.cli[:3] == ("-m", "pytest", "tests")

    def test_the_frozen_chain_the_regression_reads_is_on_disk(self):
        assert q.FROG_STAGE2.exists(), q.FROG_STAGE2
        assert q.FROG_MEMORY.exists(), q.FROG_MEMORY

    def test_the_headline_run_every_arm_is_read_against_is_on_disk(self):
        assert (q.CHECKPOINTS / "drow_stage2_s0" / "step2_results.json").exists()
        assert (q.CHECKPOINTS / "drow_stage3_s0" / "step3a_results.json").exists()

    def test_every_declared_output_is_a_path_its_own_job_writes(self):
        """The runner skips a job whose declared output exists, which is only safe while the declaration and
        the command agree. It has been wrong three times. This is the invariant, not a pre-launch guard: the
        old test asserted that no arm had run yet, which stopped being true the moment the chain succeeded."""
        for spec in q.JOBS:
            cli = spec.cli
            if "--out-dir" in cli:
                out_dir = Path(cli[cli.index("--out-dir") + 1])
                for path in spec.outputs:
                    assert path.parent == out_dir, f"{spec.name}: declares {path}, writes into {out_dir}"
            elif "--out" in cli:
                assert spec.outputs == (Path(cli[cli.index("--out") + 1]),), spec.name
            else:
                # `heads_avg` writes beside the run it averages rather than taking a path of its own.
                assert spec.name == "heads_avg", f"{spec.name} declares an output no flag accounts for"
                assert spec.outputs[0].parent == q.HEADS_SOFTMAX
