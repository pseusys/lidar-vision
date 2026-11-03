#!/usr/bin/env python3
"""Fit the post-hoc `(temperature, bias)` on val for the three frozen seeds.

One job per seed, each a full val replay of `a71_freeze_s{seed}`'s memory on its own stage-2 seed, which has to be a
replay rather than a stored score column because no stored column exists for val at the frozen architecture.

**Where the hour actually goes, measured 2026-10-30 and not where this file first guessed.** The replay is **~8 min**
and the fit is seconds. The hour is `score_detections`, called **twice** -- once uncalibrated, once calibrated, which
is the whole point of the report -- and `_prec_rec_2d` inside it walks all **2.06 M** detections in descending score
order, re-running a gated assignment on each frame's accepted prefix at every one of them. So a seed is roughly
8 min of GPU and ~55 min of single-core numpy.

**Why this is a queue module and not three commands in a loop.** The first attempt was exactly that loop, and it cost
an hour: `calibrate_rescore.py` writes its JSON, calls `os._exit`, and the process then **lingers alive** -- the ROCm
teardown hang in `memory/gotchas.md`. In a shell loop the next seed cannot start until the hang is noticed and killed
by hand; seed 0's hang sat for 18 minutes before anyone looked. `run_queue.py` already solves this: it judges a job
by **its outputs, not its exit code**, and kills a child that lingers after writing them. The hang costs
`EXIT_GRACE_SECONDS` and nothing else.

**The frames cache is what makes a mistake cheap.** `--frames` stores the replayed `(score, xy, people)` per frame
beside the JSON, and the script reuses it when it is already there. That is deliberate insurance: the first run of
this item was thrown away not because the replay was wrong but because the **label rule** was -- the labels came from
`candidate_targets`, which minimises distance and is score-blind, so it handed a person to a candidate 0.1 m away
scoring 0.05 over one 0.3 m away scoring 0.9 and labelled the metric's own true positive a negative. Re-fitting after
that fix no longer needs the GPU, the checkpoints or the 1 GB candidate cache -- it is a CPU-only job against a 25 MB
file, and a crashed fit retries without replaying. It does **not** make a re-fit cheap in wall clock: the cache skips
the ~8 min replay, not the ~55 min of scoring, so budget most of an hour per seed either way. The npz is **not**
listed as an output: it is written before the fit, so a run that replayed and then failed must not read as finished.

`--write` is deliberately absent here. The fit is reported first and written into the checkpoints as a separate,
explicit step, because writing is the irreversible half and it edits the three artifacts the freeze is quoted from.

Launch per `memory/long-running-tasks.md` -- Task Scheduler, `/it`, a **past** `/st`, never `Start-Process`:

    python queue_score_calibration.py --list
    schtasks /create /tn "lidar_item48" /tr "`"$python`" `"$queue`"" /sc once /st $past /it /f
    schtasks /run   /tn "lidar_item48"
"""

from run_queue import CHECKPOINTS, REPO, Job, main

AVERAGED = "step2_calibration.averaged.pth"
RESULTS = REPO / "results"
SEEDS = (0, 1, 2)
#: Each frozen stage-3 seed reads the stage-2 seed it was trained on; `candidate_cache_path` keys on that checkpoint's
#: directory, name and modification time, so pointing a seed at the wrong one silently rebuilds a 1 GB cache.
STAGE2_DIRS = {0: "a71_soft_label_s0", 1: "a72_soft_s1", 2: "a72_soft_s2"}
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def calibrate(seed: int) -> Job:
    """Fit seed `seed`'s transform on val, caching the replayed frames so a re-fit does not replay again."""
    return Job(name=f"calibrate_s{seed}", outputs=(RESULTS / f"calibration_s{seed}.json",),
               cli=("calibrate_rescore.py",
                    "--memory", str(CHECKPOINTS / f"a71_freeze_s{seed}" / "step3a_object_memory.best.pth"),
                    "--stage2", str(CHECKPOINTS / STAGE2_DIRS[seed] / AVERAGED),
                    "--frames", str(RESULTS / f"calibration_frames_s{seed}.npz"),
                    "--out", str(RESULTS / f"calibration_s{seed}.json")))


JOBS = [calibrate(seed) for seed in SEEDS]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
