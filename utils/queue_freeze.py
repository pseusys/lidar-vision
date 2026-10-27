#!/usr/bin/env python3
"""`TODO.md` item 47: the three-seed freeze of the adopted architecture. **Prepared, not launched.**

The architecture settled on 2026-10-25 and every published number predates it. `docs/PAPER.md`'s table is the
**selective** recurrence on the aggregator-ful stage 2 with the 1 s curriculum; the adopted model is **slots + minGRU,
no aggregator, no curriculum**, and has one seed at 86.20%. All three changes tied inside E13's 1.32 pp bar, so the old
figures are not wrong -- they measure a model this project no longer proposes. This chain replaces them.

**Why stage 2 is in here at all.** Item 40 removed the whole-scan aggregator, which lives in stage 2, so the three
control seeds (`a59_c1_recipe_dropout01_s{0,1,2}`) no longer match the adopted architecture and cannot be reused. Only
seed 0 of the aggregator-free stage 2 exists (`a69_no_aggregator_s0`), so seeds 1 and 2 are trained here first. Each is
averaged over the val-chosen window, because **the averaged weights are what stage 3 loads** -- averaging gained 15 of
18 A59 runs, +0.66 pp mean, and one arm +1.40 pp.

**The stage-2 target is the one thing still open**, and it decides which checkpoints this chain reads. `TODO.md`
item 44(a) -- the soft Gaussian beam target -- is being confirmed at three seeds by `queue_item44a_confirm.py`, which
deliberately names its hard-label arms `a71_stage2_s1/s2`, the very names below. So **if hard labels win, this chain's
stage-2 half is already done** and only the three stage-3 arms remain (~22 h); if soft labels win, point `stage2()` at
`a71_soft_label_s0` / `a72_soft_s1` / `a72_soft_s2` and add `--label-sigma-m 0.3` to the stage-2 jobs.

**Cost: about 24 h, not the 13 h first quoted here.** That estimate put a stage-3 arm at ~4 h; the simplified
baseline actually ran **7.34 h** (52 epochs at 404 s train + 103 s eval). The 507 s figure in `CHANGELOG.md` is the
per-epoch *total*, and reading it as train time is what produced the error. So: two new stage-2 arms at ~20 min plus
averaging (~1 h), two candidate-cache rebuilds (~1.3 h, below), then **three stage-3 arms at ~7.3 h** = **~24 h**.
Dropping the curriculum still pays for the extra stage-2 runs -- it is why an arm is 7.3 h and not 10 -- but this is a
full day of GPU, so it is worth being sure the architecture is final before starting.

**One cost that is easy to forget:** `candidate_cache_path` keys on the stage-2 checkpoint's directory, name and
modification time, so seeds 1 and 2 each rebuild train, val and test candidate caches the first time their stage 3
runs -- roughly 40 minutes apiece, already inside the ~4 h figure but invisible until it happens.

**What this chain does NOT do**, deliberately, because neither belongs in a sequential queue:

* **CPU latency.** Item 29's protocol is a *matched, interleaved* sweep on an idle host -- every arm plus the control,
  back to back, two or three rounds, reading the `stage2` phase first as the noise floor. A sequential queue runs each
  job in its own session, which is exactly how item 16's wrong `gru_cpu` row was produced (23.87 ms against a true
  20.57). Run it by hand afterwards: `tracking_baselines.py latency --device cpu --threads 4`, 2,000 consecutive scans,
  batch 1, against FROG's 25 ms period.
* **The person-free phantom column.** `empty_scene.py` at three seeds, after the arms exist.

Launch, when the owner says so, per `memory/long-running-tasks.md` -- Task Scheduler, `/it`, a **past** `/st`, never
`Start-Process`:

    python queue_freeze.py --list
    schtasks /create /tn "lidar_freeze" /tr "`"$python`" `"$queue`"" /sc once /st $past /it /f
    schtasks /run   /tn "lidar_freeze"
"""

from run_queue import CHECKPOINTS, REPO, Job, main

AVERAGED = "step2_calibration.averaged.pth"
SEEDS = (0, 1, 2)
#: The adopted stage-3 recipe: minGRU per slot, no slot exchange, no learned matching, no coasting head, no 1 s
#: curriculum stage, and a **64-wide** slot state. Item 38 pinned the recurrence; items 40 and 41 removed the
#: curriculum and the aggregator; **item 49 halved `dim` on 2026-10-27** -- it ties on accuracy (86.12% against
#: 86.20%, recall at matched false positives 96.5% against 96.43%) and takes **28% off the memory phase** on the CPU
#: (9.30 -> 6.70 ms), which nearly doubles p99 margin against FROG's 25 ms period. That is the axis the owner made
#: decisive on 2026-10-19, so a tie on accuracy plus a win on CPU is an adoption.
ADOPTED = ("--no-exchange", "--no-learned-match", "--no-coast", "--recurrence", "mingru", "--chunk-s", "10",
           "--dim", "64")
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2(seed: int) -> list:
    """One aggregator-free stage-2 seed and its weight average. Seed 0 already exists and is reused by name."""
    out = CHECKPOINTS / ("a69_no_aggregator_s0" if seed == 0 else f"a71_stage2_s{seed}")
    return [
        Job(name=f"stage2_s{seed}", outputs=(out / "step2_results.json",),
            cli=("train_takheliped.py", "step2", "--checkpoint-every", "0.25", "--no-aggregator",
                 "--seed", str(seed), "--out-dir", str(out), "--resume")),
        Job(name=f"stage2_s{seed}_avg", outputs=(out / AVERAGED,), needs=(f"stage2_s{seed}",),
            cli=("average_checkpoints.py", str(out))),
    ]


def stage3(seed: int) -> Job:
    """One stage-3 seed of the frozen architecture, on the matching stage-2 seed's averaged weights."""
    stage2_dir = CHECKPOINTS / ("a69_no_aggregator_s0" if seed == 0 else f"a71_stage2_s{seed}")
    out = CHECKPOINTS / f"a71_freeze_s{seed}"
    return Job(name=f"freeze_s{seed}", outputs=(out / "step3a_results.json",), needs=(f"stage2_s{seed}_avg",),
               cli=("train_takheliped.py", "step3a", "--stage2", str(stage2_dir / AVERAGED), *ADOPTED,
                    "--seed", str(seed), "--out-dir", str(out), "--resume"))


# Every stage 2 and its average first, then the three stage-3 arms: `run_queue`'s `ordered()` is layered and keeps
# declaration order within a dependency layer, so a stage-2 failure stops its own seed without blocking the others.
JOBS = [job for seed in SEEDS for job in stage2(seed)] + [stage3(seed) for seed in SEEDS]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
