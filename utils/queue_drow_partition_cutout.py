#!/usr/bin/env python3
"""Phase 2 of `TODO.md` row 106: DROW3's cutout trained on `geometry-balanced`, then scored on its test and train.

On the geometry-balanced partition our stage 2's train / test AP ratio is 1.76, short of the 1.4 the rule asked for,
but that bar came from FROG (1.16) and from the cutout on official DROW (1.05), neither measured on this partition. The
cutout on the same partition says how far 1.76 is from what a model that transfers scores here (owner, 2026-11-09:
"schedule phase 2 overnight"). One seed, DR-SPAAM's DROW recipe exactly as `queue_drow_validation.py` validated it
(63.20% on official test, 66.50% on official train), only the partition changed.

    python queue_drow_partition_cutout.py --list
    python queue_drow_partition_cutout.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from queue_drow_validation import DRSPAAM_RECIPE
from run_queue import Job, REPO, main

PARTITION = ("--frog-mode", "geometry-balanced")
WEIGHTS = REPO / "checkpoints_drow" / "drow_t1_drspaam_recipe_geometry_balanced.pth"
RESULTS = REPO / "results"

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def score(split: str) -> Job:
    out = RESULTS / f"drow_t1_drspaam_recipe_geometry_balanced_{split}.json"
    return Job(name=f"cutout_balanced_{split}", outputs=(out,), needs=("cutout_balanced",),
               cli=("evaluate.py", "--dataset", "drow", *PARTITION, "--split", split, "--drow", str(WEIGHTS), "--no-bench",
                    "--out", str(out)))


JOBS = [Job(name="cutout_balanced", outputs=(WEIGHTS,), cli=("train.py", *DRSPAAM_RECIPE, *PARTITION, "--out", str(WEIGHTS))),
        score("test"), score("train")]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
