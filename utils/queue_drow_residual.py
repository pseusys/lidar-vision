#!/usr/bin/env python3
"""Item 107's training tests of the residual DROW gap, then reading 13 of `drow_locality_property.py`.

Our stage 2 transfers almost like the cutout where a person is close, many beams wide, and worst where they are far,
a few beams wide (reading 12). The hypothesis this queue tests, its predictions in `TODO.md` row 107 before it ran:
where local evidence is weak the whole-scan network leans on context priors learnt from the training recordings, and
those misfire on new ones.

- `drow_context_dropout`: `--context-dropout 0.5`, half the training scans kept only inside a random 30-70% window,
  so no fixed layout of a training recording is reliable;
- `drow_places_50`, `drow_places_25`: the headline recipe on official DROW with its training places cut to half and
  a quarter (`official-places-50` / `-25`), val and test untouched -- the learning curve over places.

Two seeds each, the headline's exact recipe at DROW's default radii; `drow_default_radii_s0` / `_s1` is the reference.

    python queue_drow_residual.py --list
    python queue_drow_residual.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from drow_locality_property import RESIDUAL_RUNS, RESIDUAL_SEEDS
from run_queue import Job, REPO, main

DROW = ("--dataset", "drow")
LIMIT = ("--max-range-m", "29.99")
RESTRICTED = ("--eval-max-range-m", "10")
RECIPE = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3", "--label-radius", "0.35", "0.6", "0.4")
ARMS = {"drow_context_dropout": ("--context-dropout", "0.5"), "drow_places_50": ("--frog-mode", "official-places-50"),
        "drow_places_25": ("--frog-mode", "official-places-25")}
READING = REPO / "results" / "drow_residual.json"

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2(name: str, seed: int) -> Job:
    out = RESIDUAL_RUNS[name][1][RESIDUAL_SEEDS.index(seed)].parent
    return Job(name=out.name, outputs=(out / "step2_results.json",),
               cli=("train_takheliped.py", "step2", *DROW, *LIMIT, *RECIPE, *RESTRICTED, *ARMS[name],
                    "--seed", str(seed), "--out-dir", str(out), "--resume"))


TRAIN = [stage2(name, seed) for seed in RESIDUAL_SEEDS for name in ARMS]
JOBS = TRAIN + [Job(name="residual_reading", outputs=(READING,), needs=tuple(j.name for j in TRAIN),
                    cli=("drow_locality_property.py", "--readings", "residual", "--out", str(READING)))]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
