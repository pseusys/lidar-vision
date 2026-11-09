#!/usr/bin/env python3
"""Is the DROW path sound? The known-answer checks, run before any DROW lever is read.

Nothing trained in this repository on DROW had been checked against a published number, and the two networks trained
here both land ~30 pp short of their papers while every published weight file reproduces within 1-5 pp. So, in order,
each answering one question and none gating another:

1. **The scoring path** (`check_drow_scoring.py`, CPU): does the path every TAKHeLiPeD DROW number came through --
   `FrameSplit`'s labels and `score_detections` -- give DROW3's published weights the 66.5% `evaluate.py` gives them?
2. **The labels on the scans** (`drow_label_geometry.py`, CPU): are they aligned with the returns, and how much of a
   person's disk does another class's annotation take?
3. **Fit or generalise** (`train_takheliped.py score`, GPU): the same stage-2 weights' AP on train, val and test,
   DROW's headline two seeds and FROG's frozen seed 0 beside them. Minutes, so it goes before the long job. Beside it,
   FROG's stage 2 trained on DROW's annotation pattern, which prices the supervision count at stage 2.
4. **Training against a known answer** (`train.py`, GPU): DROW3 at `T=1` retrained on DR-SPAAM's own
   `base_drow_drow_cfg.yaml`, then scored by `evaluate.py`. Within ~5 pp of their 66.6% clears the shared training path.

    python queue_drow_validation.py --list
    python queue_drow_validation.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from run_queue import CHECKPOINTS, Job, REPO, main

RESULTS = REPO / "results"
#: Where `train.py`'s DROW checkpoints have always gone.
DROW_RUNS = REPO / "checkpoints_drow"
RECIPE_WEIGHTS = DROW_RUNS / "drow_t1_drspaam_recipe.pth"
BEST = "step2_calibration.best.pth"

#: `cfgs/base_drow_drow_cfg.yaml` in DR-SPAAM's repository, read 2026-11-08, flag for flag: the DROW network with one
#: person logit and their window, person-only targets at their 0.35 m person radius, plain BCE, an unweighted
#: Euclidean vote loss, Adam with AMSGrad and no weight decay, 1e-3 to 1e-6 over 40 epochs, batch 8, dropout 0.5, and
#: the last weights (no early stopping).
DRSPAAM_RECIPE = ("--detector", "drow", "--dataset", "drow", "--time-frame", "1", "--drow-cutout", "drspaam",
                  "--person-only", "--vote-radius", "0.35", "--no-class-balance", "--vote-loss", "euclidean",
                  "--vote-weight", "1", "--amsgrad", "--weight-decay", "0", "--lr", "1e-3", "--lr-schedule",
                  "exponential", "--min-lr", "1e-6", "--epochs", "40", "--batch-size", "8", "--dropout", "0.5",
                  "--patience", "0")

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def fit(name: str, dataset: str, stage2, split: str, stride: int) -> Job:
    """One stage-2 checkpoint scored alone on one split, by the pass its own run scored test with."""
    out = RESULTS / f"fit_{name}_{split}.json"
    return Job(name=f"fit_{name}_{split}", outputs=(out,),
               cli=("train_takheliped.py", "score", "--dataset", dataset, "--stage2", str(stage2), "--split", split,
                    "--test-strides", str(stride), "--out", str(out)))


JOBS = [
    Job(name="scoring_path", outputs=(RESULTS / "check_drow_scoring.json",),
        cli=("check_drow_scoring.py", "--out", str(RESULTS / "check_drow_scoring.json"))),
    Job(name="label_geometry", outputs=(RESULTS / "drow_label_geometry.json",),
        cli=("drow_label_geometry.py", "--out", str(RESULTS / "drow_label_geometry.json"))),
    # The same weights on every split: test must reproduce the run's own 29.45% and 26.26% before train means anything.
    *[fit(run, "drow", CHECKPOINTS / run / BEST, split, 1)
      for run in ("drow_stage2_s0", "drow_headline_s1") for split in ("test", "train", "val")],
    # The published schedule's last weights, 40 epochs on a fixed decay: the same question at the far end of training.
    *[fit("drow_schedule_s0_last", "drow", CHECKPOINTS / "drow_schedule_s0" / "step2_calibration.last.pth", split, 1)
      for split in ("test", "train")],
    # FROG at its own strides: test every 5th frame as the frozen figures are, train every 25th, since it is 40 Hz.
    fit("a69_no_aggregator_s0", "frog", CHECKPOINTS / "a69_no_aggregator_s0" / BEST, "test", 5),
    fit("a69_no_aggregator_s0", "frog", CHECKPOINTS / "a69_no_aggregator_s0" / BEST, "train", 25),
    # FROG's stage 2 on DROW's annotation pattern (5% of FROG's frames, in DROW's bursts), against the adopted recipe's
    # own seed 0: what "17.8x less supervision" costs at stage 2, where it was never priced.
    Job(name="frog_drow_pattern_s0", outputs=(CHECKPOINTS / "frog_drow_pattern_s0" / "step2_results.json",),
        cli=("train_takheliped.py", "step2", "--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3",
             "--annotation-pattern", "drow", "--seed", "0", "--out-dir", str(CHECKPOINTS / "frog_drow_pattern_s0"), "--resume")),
    Job(name="drspaam_recipe", outputs=(RECIPE_WEIGHTS,), cli=("train.py", *DRSPAAM_RECIPE, "--out", str(RECIPE_WEIGHTS))),
    Job(name="drspaam_recipe_test", outputs=(RESULTS / "drow_t1_drspaam_recipe_test.json",), needs=("drspaam_recipe",),
        cli=("evaluate.py", "--dataset", "drow", "--split", "test", "--drow", str(RECIPE_WEIGHTS), "--no-bench",
             "--out", str(RESULTS / "drow_t1_drspaam_recipe_test.json"))),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
