#!/usr/bin/env python3
"""Item 107 (f): stage 2 normalised per beam, as ConvNeXt is, against the scan-wide GroupNorm every checkpoint used.

Every ConvNeXt block in stage 2 normalises with `GroupNorm(1, C)`, which on a `[B, C, N]` tensor pools over the
channels **and every beam of the scan**: each layer rescales each beam by the whole scene's statistics, and the
receptive field is the whole scan whatever the kernels. ConvNeXt's own LayerNorm is per position over the channels
(`--norm position`). The prediction, in `TODO.md` row 107 before this ran: if the scan-wide normalisation is what couples
each beam to its scene, DROW test AP rises well above 29% and its train / test by range flattens, while on FROG -- whose
train and test share their scenes' statistics -- the change is small.

- `drow_position_norm`: the DROW headline recipe at the default radii, `--norm position`, two seeds, read beside
  `drow_default_radii_s0` / `_s1`;
- `frog_scan_norm`, `frog_position_norm`: FROG at today's step-2 recipe, the two norms, two seeds each, so FROG's cost
  is read against a control trained the same day and the same way.

    python queue_drow_norm.py --list
    python queue_drow_norm.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from drow_locality_property import NORM_RUNS, NORM_SEEDS
from run_queue import Job, REPO, main

RECIPE = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3")
DROW = ("--dataset", "drow", "--max-range-m", "29.99", "--eval-max-range-m", "10", "--label-radius", "0.35", "0.6", "0.4")
ARMS = {"drow_position_norm": (*DROW, "--norm", "position"), "frog_scan_norm": (), "frog_position_norm": ("--norm", "position")}
READING = REPO / "results" / "drow_norm.json"

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2(name: str, seed: int) -> Job:
    out = NORM_RUNS[name][NORM_SEEDS.index(seed)].parent
    return Job(name=out.name, outputs=(out / "step2_results.json",),
               cli=("train_takheliped.py", "step2", *RECIPE, *ARMS[name], "--seed", str(seed), "--out-dir", str(out), "--resume"))


TRAIN = [stage2(name, seed) for seed in NORM_SEEDS for name in ARMS]
JOBS = TRAIN + [Job(name="norm_reading", outputs=(READING,), needs=tuple(j.name for j in TRAIN),
                    cli=("drow_locality_property.py", "--readings", "norm", "--out", str(READING)))]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
