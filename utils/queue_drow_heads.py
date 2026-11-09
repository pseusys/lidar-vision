#!/usr/bin/env python3
"""DROW with the classes it annotates, and the three cheap levers the evidence has not yet ruled out.

**The gap this queue attacks.** The architecture scores **46.01%** end to end on DROW against **66.5%** for
DROW3 *through this repository's own pipeline* and 71.7% as published -- a deficit of 20.5 pp, where on FROG the
same parameters beat every published baseline. Four readings of that gap have already been measured and three
are closed:

* **capacity** -- ruled out in both directions: (16, 32, 64) scores 24.55% and dropout 0.3 scores 26.32%, both
  *below* the headline's 29.45%, and both fit the training set less well too. Less model is not the answer, and
  neither is more regularisation of the kind that pays on FROG;
* **the angular grid** -- priced on FROG at 0.5 deg and worth **0.61 pp** of stage 2, so DROW's coarser beams
  are not where 20 pp hides;
* **the ignore region** -- flat AP (29.45% -> 29.59%) for a third fewer false positives. Discarding the
  wheelchairs moves the operating point, not the ranking;
* **what is left** is the ranking itself: 8.2% precision at 90.6% recall. The network finds people and cannot
  order them.

**So this queue runs the four untested cheap levers, and none of them is a tweak of a lever already measured.**

1. **The multi-class head** (the owner's item, two arms). The ignore region throws away the information that an
   object is there; a head with a column per class *uses* it, and the vote regression stays shared, which is
   DROW3's own split. `softmax` is DROW's normalisation; `sigmoid` makes the binary head a strict special case.
2. **The range span**, which nothing has isolated. DROW trains at 29.99 m against FROG's 10 m -- 3x the depth,
   with no distance normalisation anywhere in this architecture, where every published DROW baseline resamples
   a fixed metric window per beam. A 10 m-trained arm and the headline recipe re-scored on the same restricted
   label set are what separate "the model cannot rank at depth" from "the model cannot rank".
3. **The metric gather at 1.66 m**, which is the same lever, built. It cost **1.42 pp** at stage 2 on FROG and
   was rejected there; on DROW a beam covers twice the arc and three times the depth, so it is a different
   physical quantity on the same flag -- the identical argument that makes the kernel arc worth one run.
4. **The kernel's arc**: `k = 7` spans 1.75 deg on FROG and 3.5 deg on DROW, so each layer covers two people
   where on FROG it covers most of one. The FROG sweep found `k` flat, which is why this is last.

**The regression arm runs first, and everything after it depends on it.** The head is new code on the path
every FROG number was measured through, so the frozen seed-0 chain is re-scored before any arm is trained:
candidates **82.76%** and whole detector **86.13%** at stride 1, the figures `a71_freeze_s0` recorded. Not
"about" -- the binary path is meant to be untouched, so anything but those two numbers means the head change
moved FROG and every arm behind it is void. `tests/test_decision_head.py` proves the same thing structurally;
this proves it on the real split.

**Stage 3 is pre-committed to the `softmax` arm**, because a runner cannot branch on a result and because the
standing rule picks the published mechanism on a tie. It is the only job here that costs hours, and it is the
one that answers the owner's question -- the memory is worth **+22.56 pp** on DROW against +3.49 on FROG, so
the whole-detector figure is where a better stage 2 pays most. If the screens land before it starts and
`sigmoid` beats `softmax` by more than 1 pp, stop the chain and re-point this job by hand; that is a decision,
not an automation.

**What this queue is not.** Not another capacity or dropout arm -- those are measured and both lose
(`memory/rejected-ideas.md`, and item 75's table). Not a second seed of anything. Not an attempt to reach
66.5% in one night: no single flag here is worth 20 pp, and the honest outcome is a better-ranked stage 2 plus
a measurement of which of the four levers carries any of the deficit.

    python queue_drow_heads.py --list
    python queue_drow_heads.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from run_queue import CHECKPOINTS, REPO, Job, main

#: The frozen FROG seed-0 chain, re-scored as the regression arm. Its own `step3a_results.json` holds the two
#: numbers this must reproduce, which is why the arm reads these exact files rather than a newer seed.
FROG_STAGE2 = CHECKPOINTS / "a71_soft_label_s0" / "step2_calibration.averaged.pth"
FROG_MEMORY = CHECKPOINTS / "a71_freeze_s0" / "step3a_object_memory.best.pth"
REGRESSION = REPO / "results" / "frog_binary_regression.json"

#: One directory per arm, each naming the one thing it changes.
HEADS_SOFTMAX = CHECKPOINTS / "drow_heads_softmax_s0"
HEADS_SIGMOID = CHECKPOINTS / "drow_heads_sigmoid_s0"
RANGE_FULL = CHECKPOINTS / "drow_range_full_s0"
RANGE_10M = CHECKPOINTS / "drow_range10_s0"
GATHER = CHECKPOINTS / "drow_gather_s0"
KERNEL5 = CHECKPOINTS / "drow_kernel5_s0"
KERNEL3 = CHECKPOINTS / "drow_kernel3_s0"
HEADS_STAGE3 = CHECKPOINTS / "drow_heads_stage3_s0"
AVERAGED = "step2_calibration.averaged.pth"

#: The dataset, which every subcommand takes.
DROW = ("--dataset", "drow")
#: DROW's own sensor limit (owner, 2026-11-04). `step2`'s flag alone -- it shapes the network, so stage 3 reads
#: it back out of the checkpoint and refuses the flag, which stopped a chain nine seconds in on 2026-11-05.
LIMIT = ("--max-range-m", "29.99")
#: The 10 m arm's limit, FROG's, with the restricted column that makes the two arms readable against each other.
LIMIT_10M = ("--max-range-m", "10")
#: Item 82's column, now at stage 2: the same test pass scored again inside 10 m, both readings.
RESTRICTED = ("--eval-max-range-m", "10")
#: The adopted recipe, spelled out as `queue_drow.py` spells it, so every arm here is comparable with the
#: headline rather than with a default that has moved since.
#: Pinned at 0.6 m: every run on file from this queue trained there, before DROW's default became its own per-class
#: radii (2026-11-09). An arm's own --label-radius comes after it on the command line and wins.
RECIPE = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3", "--label-radius", "0.6")
#: The multi-class head. The class count is the dataset's and is not a flag: DROW annotates three.
MULTICLASS = ("--head", "multiclass")
#: The frozen memory: no slot exchange, no learned matching, no coasting head.
FROZEN = ("--no-exchange", "--no-learned-match", "--no-coast")

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2(name: str, out, *flags, limit=LIMIT, needs=()) -> Job:
    """One DROW stage-2 arm: the adopted recipe, DROW's limit, and the one flag the arm is named for."""
    return Job(name=name, outputs=(out / "step2_results.json",), needs=needs,
               cli=("train_takheliped.py", "step2", *DROW, *limit, *RECIPE, *flags,
                    "--seed", "0", "--out-dir", str(out), "--resume"))


JOBS = [
    # First, and everything else waits on it: the binary path must still produce the frozen chain's own numbers.
    Job(name="frog_regression", outputs=(REGRESSION,),
        cli=("train_takheliped.py", "score", "--stage2", str(FROG_STAGE2), "--memory", str(FROG_MEMORY),
             "--test-strides", "1", "--out", str(REGRESSION))),
    # The owner's item: DROW's wheelchairs and walkers as classes rather than as background or as holes.
    stage2("heads_softmax", HEADS_SOFTMAX, *MULTICLASS, "--class-norm", "softmax", needs=("frog_regression",)),
    stage2("heads_sigmoid", HEADS_SIGMOID, *MULTICLASS, "--class-norm", "sigmoid", needs=("heads_softmax",)),
    # The range span, both halves. `range_full` is the headline recipe again with item 82's column added, which
    # is the only footing the 10 m arm can be read on -- and it re-runs the headline under today's code.
    stage2("range_full", RANGE_FULL, *RESTRICTED, needs=("heads_sigmoid",)),
    stage2("range10", RANGE_10M, *RESTRICTED, limit=LIMIT_10M, needs=("range_full",)),
    # The same metric window that cost 1.42 pp on FROG, on a sensor whose beams cover twice the arc.
    stage2("gather", GATHER, "--gather-width-m", "1.66", needs=("range10",)),
    stage2("kernel5", KERNEL5, "--kernel-size", "5", needs=("gather",)),
    stage2("kernel3", KERNEL3, "--kernel-size", "3", needs=("kernel5",)),
    # `*DROW` is what makes the ranking correct: without it the windows are scored on FROG val, which is the
    # defect of 2026-11-06 (`memory/gotchas.md`).
    Job(name="heads_avg", outputs=(HEADS_SOFTMAX / AVERAGED,), needs=("kernel3",),
        cli=("average_checkpoints.py", str(HEADS_SOFTMAX), *DROW)),
    # The 7 h job and the one that answers the question: the whole detector on DROW with a head that can say
    # what it found. Against `drow_stage3_s0`'s 46.01% whole detector and 23.45% candidates, one variable apart.
    Job(name="heads_stage3", outputs=(HEADS_STAGE3 / "step3a_results.json",), needs=("heads_avg",),
        cli=("train_takheliped.py", "step3a", *DROW, "--stage2", str(HEADS_SOFTMAX / AVERAGED), *FROZEN,
             "--chunk-s", "10", "--seed", "0", "--out-dir", str(HEADS_STAGE3), "--resume")),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
