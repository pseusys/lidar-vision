#!/usr/bin/env python3
"""How much of DROW's 20 pp deficit is our own handling, and the one FROG arm that cannot be invalidated by it.

**What this queue answers.** The DROW cell landed at 46.01% wp-AUC against DROW3's 66.5% through this same
harness, and two measurements made on 2026-11-05 say part of that gap is ours rather than the architecture's:
**56.4% of the objects DROW annotates in its test split are not pedestrians**, and every one of them is trained
as background and scored as a false positive; and the headline figure was produced at DROW's 29.99 m sensor
limit, with no 10 m-restricted column beside it, which the owner's first decision of 2026-11-04 asked for and
nobody produced. The first two jobs here are those two things, in that order, and both are cheap.

**Why the cheap arms come before anything long.** The ignore region costs about a quarter of an hour, because
stage 2 trains on annotated frames alone -- 17,665 on DROW -- where stage 3 streams every frame and is the 6-7 h
half. It also decides something: a swappable multi-class head is the other treatment of the same defect, and
running one 7 h stage 3 on whichever of the two wins is a third of the cost of running one on each. So no DROW
stage 3 is queued here at all; it waits for both arms to report.

**Why a FROG chain is in a DROW queue.** `--beam-stride 2` reads FROG on DROW's 0.5 deg grid, and it is the only
thing that prices the resampler `docs/PROPOSAL.md` section 5.3 assumed was needed -- a question that matters before
JRDB's 1091 beams arrive, and one that nothing in the DROW chain can invalidate, since it trains on FROG and is
scored on FROG. It was also the source model for the two transfer regimes until the owner parked both on
2026-11-06, so it now stands on the resolution question alone. It fills a window the DROW work deliberately
leaves empty rather than being hurried in front of it.

**What is not here.** The multi-class head, which is a day of code before it is a run; the chunk-length arms,
which are stage 3 off a stage 2 the first job may replace; and the two transfer regimes, parked.

    python queue_drow_handling.py --list
    python queue_drow_handling.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from run_queue import CHECKPOINTS, REPO, Job, main

#: The finished DROW cell, which job one scores a second time and job two is an arm off.
DROW_STAGE2 = CHECKPOINTS / "drow_stage2_s0"
DROW_STAGE3 = CHECKPOINTS / "drow_stage3_s0"
#: The ignore-region arm's own directory, kept apart so the headline stage 2 stays readable beside it.
DROW_IGNORE = CHECKPOINTS / "drow_stage2_ignore_s0"
#: FROG at DROW's angular spacing: 360 beams over 180 deg, 45 sectors, no padding.
FROG_HALF_STAGE2 = CHECKPOINTS / "frog_half_stage2_s0"
FROG_HALF_STAGE3 = CHECKPOINTS / "frog_half_stage3_s0"
AVERAGED = "step2_calibration.averaged.pth"
MEMORY = "step3a_object_memory.best.pth"

#: DROW, which every DROW job names because the flag defaults to `frog`.
DROW = ("--dataset", "drow")
#: The range limit, `step2`'s alone -- `network_shape` reads `max_range_m` back out of the checkpoint, so a
#: later stage inherits it with the weights and rejects the flag. Passing it to `step3a` anyway stopped the
#: previous chain three times in nine seconds on 2026-11-05.
LIMIT = ("--max-range-m", "29.99")
#: DROW's 0.5 deg grid met by decimating FROG rather than interpolating DROW. Unlike the range limit this **is**
#: every stage's flag, because `dataset_spec` reads it from the command line rather than from the checkpoint.
HALF = ("--beam-stride", "2")
#: The adopted recipe, spelled out rather than defaulted, so these runs read against the ones they are compared with.
#: Pinned at 0.6 m: every run on file from this queue trained there, before DROW's default became its own per-class
#: radii (2026-11-09). An arm's own --label-radius comes after it on the command line and wins.
RECIPE = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3", "--label-radius", "0.6")
#: The frozen memory: no slot exchange, no learned matching, no coasting head.
FROZEN = ("--no-exchange", "--no-learned-match", "--no-coast")

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

JOBS = [
    # Free, and owed since 2026-11-04: the same replay scored again inside 10 m, under both readings of a
    # restricted label set. It reuses the built candidate cache, and it reruns the unrestricted column too,
    # which is the check that this pass reproduces the 46.01% it is adding a column to.
    Job(name="score_10m", outputs=(REPO / "results" / "drow_within_10m.json",),
        cli=("train_takheliped.py", "score", *DROW,
             "--stage2", str(DROW_STAGE2 / AVERAGED), "--memory", str(DROW_STAGE3 / MEMORY),
             "--eval-max-range-m", "10", "--test-strides", "1", "5",
             "--out", str(REPO / "results" / "drow_within_10m.json"))),
    # The ignore region. Read the **precision** and the false positives per frame, not only val AP: the arm
    # removes contradictory supervision rather than adding signal, so recall should hold and precision move.
    Job(name="stage2_ignore", outputs=(DROW_IGNORE / "step2_results.json",),
        cli=("train_takheliped.py", "step2", *DROW, *LIMIT, *RECIPE, "--ignore-other-classes",
             "--seed", "0", "--out-dir", str(DROW_IGNORE), "--resume")),
    # FROG at 0.5 deg, the independent half of the queue.
    Job(name="frog_half_stage2", outputs=(FROG_HALF_STAGE2 / "step2_results.json",),
        cli=("train_takheliped.py", "step2", *HALF, *RECIPE,
             "--seed", "0", "--out-dir", str(FROG_HALF_STAGE2), "--resume")),
    # `*HALF` is not decoration: without it the windows are ranked at 0.25 deg on a 0.5 deg network, which is
    # what happened here and made this arm's result an upper bound (`memory/gotchas.md`).
    Job(name="frog_half_avg", outputs=(FROG_HALF_STAGE2 / AVERAGED,), needs=("frog_half_stage2",),
        cli=("average_checkpoints.py", str(FROG_HALF_STAGE2), *HALF)),
    Job(name="frog_half_stage3", outputs=(FROG_HALF_STAGE3 / "step3a_results.json",), needs=("frog_half_avg",),
        cli=("train_takheliped.py", "step3a", *HALF, "--stage2", str(FROG_HALF_STAGE2 / AVERAGED), *FROZEN,
             "--chunk-s", "10", "--seed", "0", "--out-dir", str(FROG_HALF_STAGE3), "--resume")),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
