#!/usr/bin/env python3
"""Item 104's two arms on official DROW, two seeds each, then reading 11 of `drow_locality_property.py`.

Our stage 2 ranks people inside a DROW test scan nearly as well as on familiar data, but its score level moves between
scans of recordings unlike the training ones (`CHANGELOG.md`, 2026-11-09). Two architectural answers, each behind a
flag that leaves the network bit-identical when off (`TODO.md` row 104, confirmed by the owner the same day):

- **`drow_scan_norm`**, `--input-features scan-norm`: the range channel read as a ratio to the scan's median return
  range, the other nine metric -- the owner's beam normalisation, in its cheapest form;
- **`drow_local_level`**, `--local-level`: a cutout-like branch over a 1 m metric window sets each beam's logit, and the
  U-Net's is added with its mean over the scan removed, so whole-scan context cannot move a scan's level.

The headline at the same seeds is `drow_default_radii_s0` / `_s1`, already trained; reading 11 scores all three the same
way. No bar: the owner reads the results, and nothing is adopted automatically.

    python queue_drow_locality_arms.py --list
    python queue_drow_locality_arms.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from drow_locality_property import ARM_RUNS, ARM_SEEDS
from run_queue import Job, REPO, main

DROW = ("--dataset", "drow")
LIMIT = ("--max-range-m", "29.99")
RESTRICTED = ("--eval-max-range-m", "10")
#: The headline's recipe at DROW's default radii -- as `drow_default_radii` trained, so the arm is the only difference.
RECIPE = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3", "--label-radius", "0.35", "0.6", "0.4")
ARMS = {"drow_scan_norm": ("--input-features", "scan-norm"), "drow_local_level": ("--local-level",)}
READING = REPO / "results" / "drow_locality_arms.json"

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2(name: str, seed: int) -> Job:
    out = ARM_RUNS[name][ARM_SEEDS.index(seed)].parent
    return Job(name=out.name, outputs=(out / "step2_results.json",),
               cli=("train_takheliped.py", "step2", *DROW, *LIMIT, *RECIPE, *RESTRICTED, *ARMS[name],
                    "--seed", str(seed), "--out-dir", str(out), "--resume"))


TRAIN = [stage2(name, seed) for seed in ARM_SEEDS for name in ARMS]
JOBS = TRAIN + [Job(name="arms_reading", outputs=(READING,), needs=tuple(j.name for j in TRAIN),
                    cli=("drow_locality_property.py", "--readings", "arms", "--out", str(READING)))]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
