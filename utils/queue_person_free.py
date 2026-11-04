#!/usr/bin/env python3
"""Does training on person-free frames stop the detector hallucinating in an empty room?

The hypothesis has been on the queue since 2026-10-15 and has never been testable. Stage 2 emits **2.62 phantoms per
person-free frame** before the memory filters them to 0.74, every one of them provably a stage-2 candidate rather than
a memory invention -- and the likeliest reason is that **it has never seen a room with nobody in it**. `official`'s
train split is 100% populated and `transferred`'s train and val splits are *identical to it*, so no detector this
project has trained was ever shown an empty room. `balanced` is the one partition that shows it one, 19-21% of every
split, and it now has a trained detector.

**The two arms, counted on the same frames.** `balanced`'s test split is 16-41 + 15-53, and 16-41 is 100% populated,
so every person-free frame in it comes from 15-53 -- a recording **neither** arm trained on. The `official` arm did
*select* on 15-53's populated frames, since `official`'s val holds all three raw recordings; that favours the arm this
run is arguing against, which is the safe direction for the claim.

**Both arms are read on one scale, which needs saying.** Every row this script prints is read at a fixed threshold,
and `calibrate_rescore.py` fitted a temperature and bias into the `official` run's stage 3 that the `balanced` run
has none of. Writing that pair in moved the same checkpoints' count by -11.8% at threshold 0.3 and **8x** at 0.9, so
comparing across it would price the calibration and report it as the training data. `--no-calibration` puts the
calibrated arm on the uncalibrated arm's scale without touching either checkpoint; the third job repeats the
`official` arm *with* its calibration so the as-published row exists beside the matched one.

    python queue_person_free.py --list
    python queue_person_free.py

Minutes to the hour, no training. The first job's candidate cache is already warm from the `balanced` run; the second
builds one over 110,846 frames, which is the long pole at roughly 15 minutes, and the third reuses it.
"""

from run_queue import CHECKPOINTS, REPO, Job, main

RESULTS = REPO / "results"

#: The frozen architecture on `official`, seed 0, calibrated: the pair behind the published 86.29% and behind every
#: person-free row measured so far.
OFFICIAL = (CHECKPOINTS / "a71_soft_label_s0" / "step2_calibration.averaged.pth",
            CHECKPOINTS / "a71_freeze_s0" / "step3a_object_memory.best.pth")
#: The same architecture and the same recipe on `balanced`, seed 0. Never score-calibrated.
BALANCED = (CHECKPOINTS / "balanced_stage2_s0" / "step2_calibration.averaged.pth",
            CHECKPOINTS / "balanced_stage3_s0" / "step3a_object_memory.best.pth")

#: Where both arms are counted: `balanced` test's person-free frames, held out from both.
WHERE = ("--frog-mode", "balanced", "--split", "test")


def count(name: str, stage2, memory, *extra: str, needs=()) -> Job:
    out = RESULTS / f"{name}.json"
    return Job(name=name, outputs=(out,), needs=needs,
               cli=("empty_scene.py", "--stage2", str(stage2), "--memory", str(memory), *WHERE, "--out", str(out), *extra))


GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

JOBS = [
    # First because its candidates are already cached: if anything about the partition flag is wrong, it says so in
    # minutes rather than after a cache build.
    count("person_free_balanced_arm", *BALANCED, "--no-calibration"),
    count("person_free_official_arm", *OFFICIAL, "--no-calibration"),
    # The as-published scale, for continuity with the rows measured on `transferred`. Last, and only after the arm
    # above has built the cache it shares.
    count("person_free_official_arm_calibrated", *OFFICIAL, needs=("person_free_official_arm",)),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
