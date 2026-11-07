"""What the DROW seed-and-sigmoid queue runs: the item-88 re-run, two replicates, and one capped stage 3.

Three findings from 2026-11-07 set this queue's shape, and each one is asserted here rather than trusted:

* **`heads_sigmoid`'s +2.65 pp was two variables.** The sigmoid branch counted `pos_weight` from the soft
  target, so the arm changed the head *and* the class balance. The fix landed with the test that caught it;
  this queue re-runs the arm, and that re-run goes first because every later job reads its weights.
* **DROW single-seed spread is unmeasured and may be several points.** `kernel5` (24.20%) and `kernel3`
  (28.57%) are not monotone in `k`, which no mechanism explains, so the headline recipe and the sigmoid arm
  each get a second seed. Without them neither 29.45% nor 32.10% is a figure.
* **A val memory lift does not predict a test one** -- the two moved in opposite directions on the softmax
  chain -- so the one expensive job here exists to produce a *test* whole-detector number and nothing else.

The time box is the owner's: the chain must land in about five to six hours. Everything cheap is therefore
pinned to one arm of one flag, and stage 3 carries `--max-epochs`, which the two runs it is compared against
did not. The cap has to sit above both of their best epochs or the comparison is about the cap instead.
"""
import sys
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import average_checkpoints as avg  # noqa: E402
import queue_drow_heads as heads  # noqa: E402
import queue_drow_seeds as q  # noqa: E402
from train_takheliped import build_parser as trainer_parser  # noqa: E402

STAGE2_JOBS = ["sigmoid_fixed", "headline_s1", "sigmoid_s1"]

#: The best epoch of each run the capped stage 3 is read against. The cap must clear both.
REFERENCE_BEST_EPOCHS = {"drow_stage3_s0": 36, "drow_heads_stage3_s0": 30}


def job(name):
    return next(j for j in q.JOBS if j.name == name)


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
        assert "--max-range-m" not in job("sigmoid_stage3").cli


class TestTheFixedArmRunsFirstBecauseEverythingElseReadsItsWeights:
    def test_it_is_the_first_job_and_the_chain_waits_on_it(self):
        assert [j.name for j in q.JOBS][0] == "sigmoid_fixed"

    def test_it_is_the_sigmoid_arm_at_the_seed_the_headline_was_measured_at(self):
        args = parse("sigmoid_fixed")
        assert args.head == "multiclass" and args.class_norm == "sigmoid" and args.seed == 0

    def test_it_writes_a_new_directory_rather_than_over_the_confounded_run(self):
        """The 32.10% arm stays on disk: it is the measurement the fix is read against."""
        assert parse("sigmoid_fixed").out_dir == q.SIGMOID_S0
        assert parse("sigmoid_fixed").out_dir != heads.HEADS_SIGMOID


class TestTheTwoReplicatesDifferFromTheirFirstSeedInNothingButTheSeed:
    def test_the_headline_replicate_carries_no_head_flag_at_all(self):
        """A replicate of the headline is the binary head, which is the default and must stay unspoken."""
        assert "--head" not in job("headline_s1").cli
        assert parse("headline_s1").head == "binary"

    def test_the_sigmoid_replicate_is_the_fixed_arm_at_another_seed(self):
        first, second = parse("sigmoid_fixed"), parse("sigmoid_s1")
        assert first.seed == 0 and second.seed == 1
        assert (first.head, first.class_norm) == (second.head, second.class_norm)
        assert first.kernel_size == second.kernel_size
        assert first.max_range_m == second.max_range_m

    @pytest.mark.parametrize("name", STAGE2_JOBS)
    def test_every_arm_carries_the_adopted_recipe_so_it_is_comparable_to_the_headline(self, name):
        args = parse(name)
        assert args.checkpoint_every == 0.25 and not args.aggregator and args.label_sigma_m == 0.3

    @pytest.mark.parametrize("name", STAGE2_JOBS)
    def test_every_arm_scores_the_restricted_column_because_it_is_free(self, name):
        """Item 86's +5.20 pp was read on this column, and a replicate of it costs nothing but the flag."""
        assert parse(name).eval_max_range_m == 10.0

    @pytest.mark.parametrize("name", STAGE2_JOBS)
    def test_no_arm_decimates_the_beams_because_drow_is_read_at_its_own(self, name):
        assert "--beam-stride" not in job(name).cli


class TestTheOneExpensiveJobIsBoundedAndStillComparable:
    def test_it_runs_off_the_fixed_sigmoid_arms_averaged_weights(self):
        args = parse("sigmoid_stage3")
        assert args.stage2.parent == q.SIGMOID_S0 and args.stage2.name == q.AVERAGED

    def test_the_averaging_is_told_the_dataset_its_run_trained_on(self):
        """Without it the windows are ranked on FROG val, which is the defect of 2026-11-06."""
        assert parse("sigmoid_avg").dataset == "drow"
        assert parse("sigmoid_avg").runs == [q.SIGMOID_S0]

    def test_it_keeps_the_frozen_memory_and_the_adopted_chunk(self):
        args = parse("sigmoid_stage3")
        assert args.chunk_s == [10.0] and not args.exchange and not args.learned_match and not args.coast

    def test_the_epoch_cap_clears_every_best_epoch_it_is_compared_against(self):
        """A cap below either reference run's best epoch would measure the time box, not the head."""
        cap = parse("sigmoid_stage3").max_epochs
        assert cap > max(REFERENCE_BEST_EPOCHS.values())
        assert cap < 100, "an uncapped run is what the owner's five-to-six hours rules out"

    def test_it_is_last_and_writes_its_own_directory(self):
        assert [j.name for j in q.JOBS][-1] == "sigmoid_stage3"
        assert parse("sigmoid_stage3").out_dir == q.SIGMOID_STAGE3
        for taken in ("drow_stage3_s0", "drow_heads_stage3_s0"):
            assert parse("sigmoid_stage3").out_dir != q.CHECKPOINTS / taken


class TestTheChainIsStrictAndDeclaresWhatItWrites:
    def test_every_job_waits_on_the_one_before_it_so_two_never_share_the_gpu(self):
        names = [j.name for j in q.JOBS]
        for earlier, later in zip(names[:-1], names[1:], strict=True):
            assert job(later).needs == (earlier,), f"{later} should wait on {earlier}"

    def test_the_gate_will_not_let_the_queue_start_on_a_red_suite(self):
        assert q.GATE.cli[:3] == ("-m", "pytest", "tests")

    @pytest.mark.parametrize("name", STAGE2_JOBS)
    def test_a_stage_two_arm_is_proven_done_by_its_results_file(self, name):
        """A checkpoint appears in the first minutes and would cut the job off."""
        assert [p.name for p in job(name).outputs] == ["step2_results.json"]

    def test_every_declared_output_is_a_path_its_own_job_writes(self):
        for spec in q.JOBS:
            cli = spec.cli
            if "--out-dir" in cli:
                out_dir = Path(cli[cli.index("--out-dir") + 1])
                for path in spec.outputs:
                    assert path.parent == out_dir, f"{spec.name}: declares {path}, writes into {out_dir}"
            else:
                assert spec.name == "sigmoid_avg", f"{spec.name} declares an output no flag accounts for"
                assert spec.outputs[0].parent == q.SIGMOID_S0

    def test_no_two_jobs_write_the_same_directory(self):
        directories = [p.parent for name in STAGE2_JOBS + ["sigmoid_stage3"] for p in job(name).outputs]
        assert len(set(directories)) == len(directories)
