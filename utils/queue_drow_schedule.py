#!/usr/bin/env python3
"""DROW stage 2 on the published schedule: a fixed exponential decay for 40 epochs, tested on its last weights.

**Why this arm runs first among the levers.** Every DROW stage 2 on file stopped between epochs 6.7 and 10.7, with the
training loss still falling (0.99 -> 0.22 on `drow_sigmoid_fixed_s1`) and the plateau rule having already cut the rate
10-100x, while every published DROW recipe -- Beyer et al. 2018, DR-SPAAM (its `base_drow_drow_cfg.yaml`: 1e-3 to 1e-6
over epochs 0-40, no early stopping) and Jia et al. 2021 -- trains 40-50 epochs on a fixed decay. DROW val is
effectively one recording (71% of its pedestrians), so a stopping rule that reads it is reading one recording.

**One thing changes.** The adopted recipe, DROW's own range limit and the restricted column stay exactly as the
headline ran them (`queue_drow.py`, `queue_drow_seeds.py`); only the schedule, the stopping rule and the selection move.
Batch 32 and weight decay 1e-2 stay ours, so this is the schedule's price and not DR-SPAAM's whole protocol.
The val-best weights are still written beside the last ones, so both readings can be scored afterwards.

**Read against** the headline's two seeds, 29.45% and 26.26% (mean 27.86%, spread 3.19 pp): a lever counts if its
two-seed mean clears that spread.

    python queue_drow_schedule.py --list
    python queue_drow_schedule.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from run_queue import CHECKPOINTS, Job, REPO, main

SCHEDULE_S0 = CHECKPOINTS / "drow_schedule_s0"
SCHEDULE_S1 = CHECKPOINTS / "drow_schedule_s1"

#: The dataset, which every subcommand takes.
DROW = ("--dataset", "drow")
#: DROW's own sensor limit (owner, 2026-11-04); `step2`'s flag alone.
LIMIT = ("--max-range-m", "29.99")
#: Item 82's column at stage 2, kept because the headline carries it.
RESTRICTED = ("--eval-max-range-m", "10")
#: The adopted recipe, spelled out as the headline's queue spells it.
#: Pinned at 0.6 m: every run on file from this queue trained there, before DROW's default became its own per-class
#: radii (2026-11-09). An arm's own --label-radius comes after it on the command line and wins.
RECIPE = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3", "--label-radius", "0.6")
#: The published schedule: a fixed decay to 1e-6 that cannot stop early, and the weights it ends on.
PUBLISHED_SCHEDULE = ("--schedule", "exponential", "--min-lr", "1e-6", "--max-epochs", "40", "--min-epochs", "40",
                      "--select", "last")

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2(name: str, out, seed: int, needs=()) -> Job:
    """One DROW stage-2 arm on the published schedule."""
    return Job(name=name, outputs=(out / "step2_results.json",), needs=needs,
               cli=("train_takheliped.py", "step2", *DROW, *LIMIT, *RECIPE, *RESTRICTED, *PUBLISHED_SCHEDULE,
                    "--seed", str(seed), "--out-dir", str(out), "--resume"))


JOBS = [
    stage2("schedule_s0", SCHEDULE_S0, seed=0),
    stage2("schedule_s1", SCHEDULE_S1, seed=1, needs=("schedule_s0",)),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
