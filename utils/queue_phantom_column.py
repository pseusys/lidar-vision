#!/usr/bin/env python3
"""The phantom column re-measured on the **calibrated** checkpoints.

The score calibration wrote a `(temperature, bias)` into each frozen stage-3 checkpoint. The transform is monotone, so every rank-based
number it left untouched -- but every number read at the **fixed 0.3 threshold** was measured before the key existed, and the
phantom column (`docs/ARCHITECTURE.md` §5.2: 0.675 FP/frame on person-free frames, 0.0015 at 0.9) is the headline one. This
re-runs it, exactly as the freeze ran it: `empty_scene.py` over all 72,533 person-free frames of FROG `transferred`, three seeds,
each stage 3 paired with the stage 2 whose candidates it learnt from.

**The script needed no change and gets no flag.** `object_memory` restores the key from the checkpoint, so pointing at the same
files the freeze pointed at now score the calibrated model -- which is the whole point of shipping the pair in the checkpoint.

**It points at the originals under `checkpoints_three_horizon/`, deliberately, not at
`weights_archive/frozen-frog-2026-10-31-calibrated/`.** The archive's copies are weight-identical and carry the same key, but
`candidate_cache_path` keys the ~1 GB candidate cache on the stage-2 checkpoint's run directory, file name and mtime, so a copy
would rebuild it from scratch (`weights_archive/.../MANIFEST.md`). The originals' stage-2 files were not touched by the calibration, so
**the freeze's cache should still be warm**: expect ~17 min of replay per seed rather than the ~45 that includes building it.
Budget for the cold case anyway -- `memory/long-running-tasks.md` records a 27.5 min silent cache phase, and silence is not a
hang.

**The stage-2 column will not move and is re-measured regardless.** The key is in stage 3; stage 2's candidates are produced by
an untouched network. Its row coming back at 2.134 FP/frame is the control that proves the comparison is the memory's.

    python queue_phantom_column.py --list                # what is left to run
    python queue_phantom_column.py --steps phantom_s0
    python queue_phantom_column.py                       # the gate, then every seed whose results file is missing

Run through `run_queue.py` rather than a shell loop: `empty_scene.py` ends in `os._exit` and still lingers on the ROCm
teardown hang, and the runner judges a job by its outputs and kills a child that lingers after writing them. Launch it
detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from run_queue import CHECKPOINTS, REPO, Job, main

RESULTS = REPO / "results"
SEEDS = (0, 1, 2)
AVERAGED = "step2_calibration.averaged.pth"
STAGE2_DIRS = {0: "a71_soft_label_s0", 1: "a72_soft_s1", 2: "a72_soft_s2"}
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def phantom(seed: int) -> Job:
    """Seed `seed`'s calibrated detector counted on the person-free frames."""
    out = RESULTS / f"empty_scene_cal_s{seed}.json"
    return Job(name=f"phantom_s{seed}", outputs=(out,),
               cli=("empty_scene.py",
                    "--stage2", str(CHECKPOINTS / STAGE2_DIRS[seed] / AVERAGED),
                    "--memory", str(CHECKPOINTS / f"a71_freeze_s{seed}" / "step3a_object_memory.best.pth"),
                    "--out", str(out)))


JOBS = [phantom(seed) for seed in SEEDS]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
