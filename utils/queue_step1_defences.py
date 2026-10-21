#!/usr/bin/env python3
"""Step 1's three remaining defence items as one chain: 16, 18 and 19 (`TODO.md` §A).

All three answer "why this and not the obvious alternative?" rather than hoping for a gain, and all three are already
written -- this module exists only so the machine runs them back to back instead of idling between two launches. The
runner judges a job by its outputs, so whatever is already finished is skipped: on 2026-10-20 that is item 16's
`selective_cpu`, `mingru` and `mingru_cpu`, leaving the `gru` arm and the whole of 18 and 19.

Order is the concatenation, not a priority: item 16's `gru` is the long job (~10 h, a stage-3 arm) and items 18 and 19
are nine stage-2 runs of about twenty minutes each. Nothing here depends on anything there, so a failure in one leaves
the other running -- which is the point of one chain rather than two tasks contending for the GPU.

**The CPU rows in this chain are measured in whichever session their job happens to run.** That is good enough as the
queue's output marker and *not* good enough for the table: on 2026-10-20 the adopted arm's own stage-2 phase, whose
weights are byte-identical across arms, read 8.47 ms in one session and 8.78-8.90 ms in another. Quote a cross-arm CPU
comparison only from readings taken back to back in one session, with the stage-2 phase as the control that proves the
noise floor (`memory/interpreting-evaluation.md`).

    python queue_step1_defences.py --list
"""

from queue_item16 import JOBS as RECURRENCE
from queue_items1819 import JOBS as HEAD_AND_BLOCK
from run_queue import CHECKPOINTS, REPO, Job, main

RESULTS = REPO / "results"
STAGE2 = CHECKPOINTS / "a59_c1_recipe_dropout01_s0" / "step2_calibration.averaged.pth"
FROZEN_MEMORY = CHECKPOINTS / "a64_no_coast_s0" / "step3a_object_memory.best.pth"
# `a64_no_coast_s0`'s own matched-false-positive threshold, from its `step3a_results.json` (`own_threshold`).
MATCHED_FP_THRESHOLD = 0.0333
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def oracle(name: str, threshold: float) -> Job:
    """How much of the interpolation bound the frozen memory actually takes, at one operating threshold.

    `takheliped_oracle.py` sweeps `BRIDGES_S` in one pass, so a run gives the whole ceiling curve from 0.2 s to 60 s. Two
    thresholds, because **a ceiling has to be read at the operating point its consumer uses** (item 34's lesson, and
    `AGENTS.md` rule 3's family of error):

        t03         0.3, the threshold A50 measured at, so the result is comparable to its 17.5% figure
        matched_fp  the frozen model's own threshold, where its false-positive rate matches the candidates' and the
                    recall columns of the result tables are read

    Neither can contradict an AP integrated over every threshold, and at the operating threshold the memory trades recall
    for precision, so a recovery count quoted alone flatters it -- the script reports both directions for that reason.
    """
    out = RESULTS / f"oracle_frozen_{name}.json"
    return Job(name=f"oracle_{name}", outputs=(out,),
               cli=("takheliped_oracle.py", "--stage2", str(STAGE2), "--memory", str(FROZEN_MEMORY),
                    "--threshold", str(threshold), "--out", str(out)))


# The oracle sits between item 16 and items 18/19 on purpose: `ordered` keeps declaration order among independent jobs, so
# these run once `gru` is off the GPU and before the nine stage-2 arms, with no false dependency on `gru` succeeding.
ORACLE = [oracle("t03", 0.3), oracle("matched_fp", MATCHED_FP_THRESHOLD)]
JOBS = [*RECURRENCE, *ORACLE, *HEAD_AND_BLOCK]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
