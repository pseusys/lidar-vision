#!/usr/bin/env python3
"""DROW's person radius, 0.35 m against the adopted 0.6 m, at five seeds each: the measurement row 100 waits on.

Two seeds each put 0.35 m at +2.38 pp (30.24% against 27.86%), but the two seeds disagree by 7 pp between themselves
(-1.19 and +5.95 against their headline seeds), so the difference is inside its own noise and the owner asked for a
measurement before DROW's default moves (2026-11-09). Seeds 2-4 of both arms, interleaved so every finished pair is
paired, on the headline's exact recipe; with the existing seeds 0-1 each radius then has five.

**Read, as written in `TODO.md` before the run:** 0.35 m becomes DROW's default if its five-seed mean clears 0.6 m's
by more than twice the standard error of the difference; otherwise the tie stands and the owner decides with the
numbers in hand. Seeds 0-1 are `drow_stage2_s0` / `drow_headline_s1` and `drow_person_radius_s0` / `_s1`.

**Then the new default itself** (owner, 2026-11-09: the tie went to 0.35 m, with DROW's own radius for each class).
`drow_default_radii`, five seeds of DROW's 0.35 / 0.6 / 0.4 m, read against `drow_person_radius` seed for seed. On the
binary head the other classes' radii turned out to be unread -- they draw an ignore region only under
--ignore-other-classes, which is off -- so the pair is one objective trained twice, and measured this machine's
run-to-run spread (`CHANGELOG.md`, 2026-11-09). Finished jobs are skipped.

    python queue_drow_radius_seeds.py --list
    python queue_drow_radius_seeds.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from run_queue import CHECKPOINTS, Job, REPO, main

DROW = ("--dataset", "drow")
LIMIT = ("--max-range-m", "29.99")
RESTRICTED = ("--eval-max-range-m", "10")
#: The adopted recipe, spelled out as the headline's queue spells it.
#: Pinned at 0.6 m: every run on file from this queue trained there, before DROW's default became its own per-class
#: radii (2026-11-09). An arm's own --label-radius comes after it on the command line and wins.
RECIPE = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3", "--label-radius", "0.6")
ARMS = {"drow_headline": (), "drow_person_radius": ("--label-radius", "0.35")}
SEEDS = (2, 3, 4)

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2(name: str, flags, seed: int) -> Job:
    out = CHECKPOINTS / f"{name}_s{seed}"
    return Job(name=f"{name}_s{seed}", outputs=(out / "step2_results.json",),
               cli=("train_takheliped.py", "step2", *DROW, *LIMIT, *RECIPE, *RESTRICTED, *flags,
                    "--seed", str(seed), "--out-dir", str(out), "--resume"))


JOBS = [stage2(name, flags, seed) for seed in SEEDS for name, flags in ARMS.items()]
JOBS += [stage2("drow_default_radii", ("--label-radius", "0.35", "0.6", "0.4"), seed) for seed in range(5)]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
