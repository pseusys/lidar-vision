#!/usr/bin/env python3
"""Item 107 (h) and (i): DROW test part by part, then FROG's cost of a smaller receptive field under the per-beam norm.

(h) is reading 17 of `drow_locality_property.py`, the owner's method: DROW test split by people per scan, scan geometry,
look-alikes and range, every run's AP per part over the cutout's. (i): on DROW a smaller field under `--norm position`
added 3.5 pp, but the kernels count beams, so on FROG's 0.25-deg grid the same kernels span half the angle. Kernel 5
and kernel 3 under the per-beam norm, two FROG seeds each, read against the per-beam norm's 79.90% and the scan norm's
81.30% (`frog_position_norm`, `frog_scan_norm`, trained the same way). Predictions in `TODO.md` row 107 before this ran.

    python queue_frog_field.py --list
    python queue_frog_field.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from drow_locality_property import FROG_FIELD_RUNS, NORM_SEEDS
from run_queue import Job, REPO, main

RECIPE = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3", "--norm", "position")
ARMS = {"frog_field_k5": ("--kernel-size", "5"), "frog_field_k3": ("--kernel-size", "3")}
PARTS = REPO / "results" / "drow_parts.json"
READING = REPO / "results" / "frog_field.json"

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2(name: str, seed: int) -> Job:
    out = FROG_FIELD_RUNS[name][NORM_SEEDS.index(seed)].parent
    return Job(name=out.name, outputs=(out / "step2_results.json",), needs=("parts_reading",),
               cli=("train_takheliped.py", "step2", *RECIPE, *ARMS[name], "--seed", str(seed), "--out-dir", str(out), "--resume"))


TRAIN = [stage2(name, seed) for seed in NORM_SEEDS for name in ARMS]
JOBS = [Job(name="parts_reading", outputs=(PARTS,), cli=("drow_locality_property.py", "--readings", "parts", "--out", str(PARTS))),
        *TRAIN,
        Job(name="frog_field_reading", outputs=(READING,), needs=tuple(j.name for j in TRAIN),
            cli=("drow_locality_property.py", "--readings", "frog-field", "--out", str(READING)))]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
