#!/usr/bin/env python3
"""Our stage 2 on DROW repartitioned by scan geometry, and on the control built the same way: phase 1 of `TODO.md` row 106.

DROW's official test is enclosed and its train open, and on held-out val our stage 2's scores stop comparing between
scans exactly on the enclosed ones. `geometry-balanced` gives every split DROW's overall share of enclosed scans;
`geometry-shifted` keeps the official test and is built by the same rule, so the two differ in geometry alone
(`utils/drow_geometry_partition.py`). Two seeds of the unmodified stage 2 on each, at DROW's default label radii, then
reading 10 of `drow_locality_property.py`, whose verdict is the rule written into row 106 before this ran.

    python queue_drow_partitions.py --list
    python queue_drow_partitions.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from drow_locality_property import PARTITION_RUNS, PARTITION_SEEDS
from run_queue import Job, REPO, main

DROW = ("--dataset", "drow")
LIMIT = ("--max-range-m", "29.99")
RESTRICTED = ("--eval-max-range-m", "10")
#: The adopted DROW recipe at DROW's own default radii -- deliberately no --label-radius, unlike the older DROW queues.
RECIPE = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3")
READING = REPO / "results" / "drow_partitions.json"

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2(mode: str, seed: int) -> Job:
    out = PARTITION_RUNS[mode][PARTITION_SEEDS.index(seed)].parent
    return Job(name=out.name, outputs=(out / "step2_results.json",),
               cli=("train_takheliped.py", "step2", *DROW, "--frog-mode", mode, *LIMIT, *RECIPE, *RESTRICTED,
                    "--seed", str(seed), "--out-dir", str(out), "--resume"))


TRAIN = [stage2(mode, seed) for seed in PARTITION_SEEDS for mode in PARTITION_RUNS]
JOBS = TRAIN + [Job(name="partition_reading", outputs=(READING,), needs=tuple(j.name for j in TRAIN),
                    cli=("drow_locality_property.py", "--readings", "partitions", "--out", str(READING)))]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
