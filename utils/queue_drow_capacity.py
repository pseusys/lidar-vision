#!/usr/bin/env python3
"""Does a drastically smaller stage 2 stop memorising DROW? Two seeds per width, and the train/test ratio of each.

**The question (owner, 2026-11-08).** Our stage 2 fits DROW's training recordings (63.6% train AP) and reaches its test
ones at 29.5%, where DROW3's cutout -- retrained here on the same frames -- scores 66.5% / 63.2%. The obvious remedy for
fitting what does not transfer is fewer parameters. One reduction is on file, 4x to (16, 32, 64) at one seed: 24.55%,
worse on test and on train alike. This goes 15x and 58x down, two seeds each, and scores every seed-0 arm on DROW's
training split too, because the ratio -- not test AP alone -- is what says whether memorising stopped.

**What would separate the readings, written before the run.** A ratio falling towards the cutout's 1.05 *with* test
AP holding or rising means capacity was the lever after all. A ratio falling while test AP falls with it means the
network is simply learning less of everything -- underfitting, not transfer. A ratio that stays near 2 at 16,667
parameters means the memorising is not done by parameters but by what each beam can see, which is the receptive
field and the next experiment. Note the cutout this is read against has 1,518,531 parameters, 1.6x ours.

**The LFE-shaped arm** carries LFE's input and LFE's widths on our topology: the nearest thing to asking whether a
full-scan network like LFE suffers the same way, since LFE itself is ONNX-only here and no one has trained it on DROW.
It is a proxy for LFE, not a measurement of it.

    python queue_drow_capacity.py --list
    python queue_drow_capacity.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from run_queue import CHECKPOINTS, Job, REPO, main

RESULTS = REPO / "results"
BEST = "step2_calibration.best.pth"
DROW = ("--dataset", "drow")
LIMIT = ("--max-range-m", "29.99")
RESTRICTED = ("--eval-max-range-m", "10")
#: The adopted recipe, spelled out as the headline's queue spells it.
#: Pinned at 0.6 m: every run on file from this queue trained there, before DROW's default became its own per-class
#: radii (2026-11-09). An arm's own --label-radius comes after it on the command line and wins.
RECIPE = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3", "--label-radius", "0.6")

#: 62,867 and 16,667 parameters against the headline's 960,323.
ARMS = {
    "drow_widths8": ("--channels", "8", "16", "32"),
    "drow_widths4": ("--channels", "4", "8", "16"),
    "drow_lfe_shaped": ("--channels", "8", "16", "32", "--input-features", "lfe"),
}

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2(name: str, flags, seed: int) -> Job:
    out = CHECKPOINTS / f"{name}_s{seed}"
    return Job(name=f"{name}_s{seed}", outputs=(out / "step2_results.json",),
               cli=("train_takheliped.py", "step2", *DROW, *LIMIT, *RECIPE, *RESTRICTED, *flags,
                    "--seed", str(seed), "--out-dir", str(out), "--resume"))


def train_ap(name: str) -> Job:
    """Seed 0's own weights on DROW's training split: the numerator of the ratio the arm is read by."""
    out = RESULTS / f"fit_{name}_s0_train.json"
    return Job(name=f"fit_{name}_s0_train", outputs=(out,), needs=(f"{name}_s0",),
               cli=("train_takheliped.py", "score", *DROW, "--stage2", str(CHECKPOINTS / f"{name}_s0" / BEST),
                    "--split", "train", "--test-strides", "1", "--out", str(out)))


JOBS = [*[stage2(name, flags, seed) for seed in (0, 1) for name, flags in ARMS.items()],
        *[train_ap(name) for name in ARMS]]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
