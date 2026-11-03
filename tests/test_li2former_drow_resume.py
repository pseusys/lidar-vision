"""The resume leg: continue the paused DROW run without destroying what it already earned.

The run was paused by the owner on 2026-11-01 one epoch in, at 31.3% val wp-AUC, and
`checkpoints_drow/li2former_official_parallel.best.pth` is a full resume point -- weights, Adam's two moments
(`step 17665`), `epoch 1` and the `val_wp_auc` that `train.py` seeds `best_val_auc` from.

Two ways to lose it, both pinned here:

* **Reusing the paused leg's `--out`.** `run_queue.finished()` judges a job by its declared outputs, and the paused
  leg's `--out` was never written, so re-running `queue_li2former_drow.py` would restart from epoch 1 and overwrite
  `.best.pth` as soon as its own first epoch beat `-inf`. The resume leg therefore writes a *different* `--out`, and
  its `.best.pth` is a different file from the one it resumes from.
* **Declaring `.best.pth`.** `spawn_process` arms a 600 s kill timer the moment every declared output exists, and
  `.best.pth` appears at the first improving epoch -- hours before the run ends. That is the runner's kill-timer defect, and
  `test_queue_declarations.py` pins it for that chain; this file pins it for this one.

`--lr 1e-4` is pinned because the `1e-3` default killed their regression head in two epochs on attempt 1, and a
resume that silently reverted to the default would repeat it with the optimiser state of a healthy run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

import queue_li2former_drow_resume as leg


def flag(job, name: str) -> str:
    """The value following `name` in a job's CLI, so an assertion reads the command the runner will run."""
    cli = list(job.cli)
    assert name in cli, f"{job.name} does not pass {name}: {cli}"
    return cli[cli.index(name) + 1]


class TestItResumesTheOwnersPausedRun:
    def test_it_resumes_from_the_epoch_1_checkpoint(self):
        assert flag(leg.TRAIN, "--resume") == str(leg.RESUME_FROM)
        assert leg.RESUME_FROM.name == "li2former_official_parallel.best.pth"

    def test_it_runs_the_eleven_epochs_the_pause_left(self):
        """Attempt 3 was a 12-epoch plan and epoch 1 is on disk; `--resume` runs N *more* epochs, not N total."""
        assert flag(leg.TRAIN, "--epochs") == "11"

    def test_it_keeps_the_learning_rate_that_made_epoch_1_work(self):
        assert flag(leg.TRAIN, "--lr") == "1e-4"

    def test_it_keeps_batch_1_for_comparability_with_attempt_3(self):
        assert flag(leg.TRAIN, "--batch-size") == "1"


class TestItCannotOverwriteWhatItResumesFrom:
    def test_the_output_is_not_the_paused_legs_output(self):
        assert leg.CKPT != leg.RESUME_FROM.with_suffix("").with_suffix(".pth")
        assert leg.CKPT.name != "li2former_official_parallel.pth"

    def test_its_best_checkpoint_is_a_different_file_from_the_resume_source(self):
        assert leg.BEST != leg.RESUME_FROM

    def test_the_declared_output_is_the_final_checkpoint_alone(self):
        """Written once, at the end of `train_model` -- so the kill timer cannot start while the run still works."""
        assert leg.TRAIN.outputs == (leg.CKPT,)

    def test_no_job_declares_a_best_checkpoint(self):
        for job in leg.JOBS:
            assert all(not path.name.endswith(".best.pth") for path in job.outputs), job.name


class TestBothRadiiAreScoredFromTheNewWeights:
    def test_each_eval_waits_for_the_training_job(self):
        assert leg.EVAL.needs == ("drow_train_resume",)
        assert leg.EVAL_03.needs == ("drow_train_resume",)

    def test_each_eval_scores_the_resumed_runs_own_best_checkpoint(self):
        assert flag(leg.EVAL, "--li2former-official") == str(leg.BEST)
        assert flag(leg.EVAL_03, "--li2former-official") == str(leg.BEST)

    def test_only_one_of_them_moves_the_matching_radius(self):
        """Their 76.4% is AP@0.5 and their 72.5% is AP@0.3; `evaluate.py` takes one radius per pass."""
        assert "--eval-r" not in leg.EVAL.cli
        assert flag(leg.EVAL_03, "--eval-r") == "0.3"

    def test_the_evals_declare_nothing_because_evaluate_py_prints(self):
        assert leg.EVAL.outputs == ()
        assert leg.EVAL_03.outputs == ()
