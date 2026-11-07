#!/usr/bin/env python3
"""The FROG 0.5 deg arm again, on the averaging window a correct ranking picks.

**Why this exists.** The 0.5 deg result of 2026-11-06 -- whole detector 85.42% against the 0.25 deg headline's
86.30%, a deficit of **0.88 pp** -- was reported as an *upper bound*, because that arm's averaging window had been
ranked at 0.25 deg on a 0.5 deg network while the headline's was ranked correctly. The re-ranking run that same
day confirmed the window moves: **0.50-1.00 -> 0.50-2.00**. This chain re-averages on the correct window and
re-runs stage 3, so the deficit becomes a figure rather than a bound.

**How much is expected to move, stated before it runs.** Very little. The correct ranking is nearly flat -- the
six windows span **0.45 pp** of stage-2 val (76.53 to 76.98) where the mis-ranked one fell monotonically from
64.50 to 61.73 -- and the window actually used scores **76.66** against the chosen **76.98**, so the arm was
handed weights **0.32 pp** of stage-2 val short of the best available. Expect a few tenths of a point at most at
stage 2 and less after the memory. **The finding is not at stake**: 0.5 deg costs under a point and the memory's
contribution is undiminished either way. What is at stake is whether the published number is the measurement or a
bound on it, and this project does not leave that open when closing it costs one idle night.

**What this chain is not.** Not a second seed. `--seed 0` is kept, so exactly one thing differs from the run it
replaces and the comparison is clean. Seed noise is the *other* half of that number's uncertainty -- the 0.25 deg
headline's three seeds span 0.31 pp -- and a second seed is a separate night, worth having only once the window
question is closed.

**The old weights are kept.** `step2_calibration.averaged.window0.50-1.00.pth` beside the new average is the file
the published 85.42% was trained on, so re-averaging in place does not strand that result's provenance. Stage 3
writes to its own directory for the same reason.

    python queue_half_corrected.py --list
    python queue_half_corrected.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from run_queue import CHECKPOINTS, REPO, Job, main

#: The 0.5 deg stage 2, whose periodic checkpoints are re-averaged on the correct window.
HALF_STAGE2 = CHECKPOINTS / "frog_half_stage2_s0"
#: Stage 3's own directory, so the 85.42% run stays readable beside its replacement.
HALF_STAGE3_FIXED = CHECKPOINTS / "frog_half_stage3_fixed_s0"

AVERAGED = "step2_calibration.averaged.pth"
#: What the published 85.42% was trained on, preserved before this chain overwrites `AVERAGED`.
SUPERSEDED = "step2_calibration.averaged.window0.50-1.00.pth"

#: DROW's 0.5 deg grid met by decimating FROG. Every stage's flag, the helpers that re-score included.
HALF = ("--beam-stride", "2")
#: The frozen memory: no slot exchange, no learned matching, no coasting head.
FROZEN = ("--no-exchange", "--no-learned-match", "--no-coast")

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

JOBS = [
    # No `--rank-only` here: this pass is meant to replace the averaged checkpoint, which is why the superseded
    # one was copied aside first. `*HALF` is what makes the ranking correct at all.
    Job(name="half_avg", outputs=(HALF_STAGE2 / AVERAGED,),
        cli=("average_checkpoints.py", str(HALF_STAGE2), *HALF)),
    Job(name="half_stage3_fixed", outputs=(HALF_STAGE3_FIXED / "step3a_results.json",), needs=("half_avg",),
        cli=("train_takheliped.py", "step3a", *HALF, "--stage2", str(HALF_STAGE2 / AVERAGED), *FROZEN,
             "--chunk-s", "10", "--seed", "0", "--out-dir", str(HALF_STAGE3_FIXED), "--resume")),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
