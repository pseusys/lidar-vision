#!/usr/bin/env python3
"""The per-beam norm at kernel 5: its two DROW follow-ups first, then FROG end to end (`TODO.md` rows 108-110).

Item 107 found the change -- ConvNeXt's per-beam norm in place of the scan-wide `GroupNorm`, kernel 5 in place of 7:
DROW stage 2 29.03 -> 41.79%, FROG stage 2 81.30 -> 79.80%. The owner agreed three readings of it (2026-11-12):

- **row 110**, `drow_balanced_k5`: two seeds on `geometry-balanced`, read against the cutout trained there, to split
  the remaining official gap into the train / test geometric asymmetry and the architecture's own share;
- **row 109**, `drow_multiclass_k5`: two seeds on official DROW with `--head multiclass --class-norm softmax`, for the
  look-alike part (a wheelchair or walker near nobody, 0.39-0.55 of the cutout's AP);
- **row 108**, FROG end to end: stage-2 seed 2 (seeds 0-1 are `frog_field_k5_s0` / `_s1`, the same recipe), each seed
  averaged over its val-chosen window, then stage 3 at the frozen recipe on each -- `a71_freeze_s0-2`'s pipeline with
  only stage 2's norm and kernel changed, read against their 86.29%.

The DROW pair runs first, so its two readings land within the hour; FROG's three stage-3 arms are about a GPU-day.
Predictions are in `TODO.md` before this ran. The CPU sweep is not here: it is matched and interleaved by hand on an
idle host (`memory/commands.md`).

    python queue_position_norm_k5.py --list
    python queue_position_norm_k5.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from drow_locality_property import FOLLOWUP_RUNS, FREEZE_RUNS, FREEZE_SEEDS, NORM_SEEDS
from run_queue import CHECKPOINTS, Job, REPO, main

NEW = ("--norm", "position", "--kernel-size", "5")
STAGE2 = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3", *NEW)
DROW = ("--dataset", "drow", "--max-range-m", "29.99", "--eval-max-range-m", "10", "--label-radius", "0.35", "0.6", "0.4")
DROW_ARMS = {"drow_balanced_k5": ("--frog-mode", "geometry-balanced"), "drow_multiclass_k5": ("--head", "multiclass", "--class-norm", "softmax")}
#: Row 109's rerun (2026-11-12): both multi-class runs above stopped early (best epochs 1.0 and 4.24), so they are
#: retrained with the patience rule held off until epoch 8, the binary head's best epochs being 7.75-9.0.
DROW_ARMS["drow_multiclass_k5_long"] = (*DROW_ARMS["drow_multiclass_k5"], "--min-epochs", "8")
#: The frozen detector's stage 3, flag for flag as `a71_freeze_s0-2` stored them.
FROZEN = ("--recurrence", "mingru", "--chunk-s", "10", "--no-exchange", "--no-learned-match", "--no-coast")
AVERAGED = "step2_calibration.averaged.pth"
RESULTS = REPO / "results"

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def drow_stage2(name: str, seed: int) -> Job:
    out = FOLLOWUP_RUNS[name][NORM_SEEDS.index(seed)].parent
    return Job(name=out.name, outputs=(out / "step2_results.json",),
               cli=("train_takheliped.py", "step2", *DROW, *STAGE2, *DROW_ARMS[name], "--seed", str(seed), "--out-dir", str(out),
                    "--resume"))


DROW_JOBS = [drow_stage2(name, seed) for seed in NORM_SEEDS for name in DROW_ARMS if name != "drow_multiclass_k5_long"]
LONG_JOBS = [drow_stage2("drow_multiclass_k5_long", seed) for seed in NORM_SEEDS]
FOLLOWUPS = Job(name="followups_reading", outputs=(RESULTS / "drow_followups.json",), needs=tuple(j.name for j in DROW_JOBS),
                cli=("drow_locality_property.py", "--readings", "followups", "--out", str(RESULTS / "drow_followups.json")))

FROG_STAGE2 = [CHECKPOINTS / f"frog_field_k5_s{s}" for s in FREEZE_SEEDS]
#: No `needs` on the DROW reading: a reading that fails must not hold back a GPU-day of FROG training.
FROG_JOBS = [Job(name="frog_field_k5_s2", outputs=(FROG_STAGE2[2] / "step2_results.json",),
                 cli=("train_takheliped.py", "step2", *STAGE2, "--seed", "2", "--out-dir", str(FROG_STAGE2[2]), "--resume"))]
for s, (stage2, stage3) in enumerate(zip(FROG_STAGE2, FREEZE_RUNS["per_beam_norm_k5"], strict=True)):
    FROG_JOBS.append(Job(name=f"frog_field_k5_s{s}_avg", outputs=(stage2 / AVERAGED,), needs=("frog_field_k5_s2",),
                         cli=("average_checkpoints.py", str(stage2))))
    FROG_JOBS.append(Job(name=stage3.name, outputs=(stage3 / "step3a_results.json",), needs=(f"frog_field_k5_s{s}_avg",),
                         cli=("train_takheliped.py", "step3a", "--stage2", str(stage2 / AVERAGED), *FROZEN, "--seed", str(s),
                              "--out-dir", str(stage3), "--resume")))
FREEZE = Job(name="freeze_reading", outputs=(RESULTS / "frog_freeze_k5.json",),
             needs=tuple(r.name for r in FREEZE_RUNS["per_beam_norm_k5"]),
             cli=("drow_locality_property.py", "--readings", "freeze", "--out", str(RESULTS / "frog_freeze_k5.json")))

MULTICLASS = Job(name="multiclass_reading", outputs=(RESULTS / "drow_multiclass_long.json",), needs=tuple(j.name for j in LONG_JOBS),
                 cli=("drow_locality_property.py", "--readings", "multiclass", "--out", str(RESULTS / "drow_multiclass_long.json")))

#: The rerun was added after the chain had started, so it runs on the next launch, once FROG's stage 3 has the GPU back;
#: it does not wait on FROG's reading, so a FROG failure cannot block it.
JOBS = [*DROW_JOBS, FOLLOWUPS, *FROG_JOBS, FREEZE, *LONG_JOBS, MULTICLASS]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
