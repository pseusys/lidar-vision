#!/usr/bin/env python3
"""The frozen architecture on FROG `balanced`, one seed, and the published pair scored on the same split.

**Why `balanced` first among the other datasets.** It is the cheapest: same sensor, same venue, same loader,
no part of the dataset-agnostic work needed. It is also the only partition with **person-free frames in train
and val** -- `official` and `transferred` have them only in a test split -- so it is the one place the standing
hypothesis can be tested, that stage 2 emits 2.62 phantoms per empty frame because it has never seen a room
with nobody in it.

**One seed, deliberately** (owner, 2026-11-03). If `balanced` lands meaningfully away from `official`'s
86.29%, the other two seeds are worth their GPU; if it lands on top of it, one seed already said what there
was to say. The three-seed spread on `official` is 0.22 pp, so "meaningfully" means more than about half a
point either way.

**What can actually be trained here, which is less than it sounds.** Only this project's own detector. LFE is
an inference graph -- the PyTorch port was attempted and dropped -- and DR-SPAAM and DROW3 are not in
`train.py`'s trainable set either, so all three can only be *scored* with their authors' weights. That is not
a limitation of this queue: `balanced`'s test holds out both recordings the published detectors were trained
on, which is exactly what makes scoring them there honest, and it is the cell that has been open since the
mode comparison began.

**What this queue does not do, and why.**

* **The `official`-test subset of the same pass.** `balanced`'s test contains `frog_16-41` whole -- verified
  frame for frame, 40,022 frames, 100% of `official`'s test (`check_balanced_partition.py`) -- so a
  balanced-trained model can be scored on exactly the published split. There is no entry point for it: this
  project's detector is scored by `train_takheliped.py`'s own test pass, which scores the partition it
  trained on, and nothing scores a trained checkpoint on a *different* partition. Filed rather than invented.
* **The phantom column on `balanced`.** `empty_scene.py` takes no partition, so the side-by-side that makes
  the person-free-negatives case needs a flag first. Also filed.

    python queue_balanced.py --list
    python queue_balanced.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows. Stage 2 is the long pole
at roughly a GPU-day for the pair; the published-baseline job is minutes and runs first so a cheap result
lands before anything expensive starts.
"""

from run_queue import CHECKPOINTS, REPO, Job, main

#: `balanced`'s own directory per stage, kept apart from every `official` run so no averaging or replay can
#: mix partitions. The names carry the partition for the same reason the checkpoints now store `frog_mode`.
STAGE2 = CHECKPOINTS / "balanced_stage2_s0"
STAGE3 = CHECKPOINTS / "balanced_stage3_s0"
AVERAGED = "step2_calibration.averaged.pth"

#: The adopted recipe, spelled out rather than defaulted, so this run is readable against the `official` ones
#: it will be compared with: no whole-scan aggregator, the soft Gaussian beam target at 0.3 m.
RECIPE = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3")
#: The frozen memory: no slot exchange, no learned matching, no coasting head.
FROZEN = ("--no-exchange", "--no-learned-match", "--no-coast")

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

#: Minutes, and the oldest open cell in the mode comparison: LFE-Peaks and LFE-PPN on a test split that holds
#: out the recordings their authors trained on. No `--drow`/`--drspaam` here -- they run one forward pass per
#: beam, which over this split's 110,846 frames is days rather than minutes, and their FROG numbers already
#: exist on `official`.
BASELINES = Job(
    name="published_baselines",
    cli=("evaluate.py", "--dataset", "frog", "--frog-mode", "balanced", "--split", "test",
         "--lfe-peaks", "--lfe-ppn", "--no-bench"))

JOBS = [
    BASELINES,
    Job(name="stage2", outputs=(STAGE2 / "step2_results.json",),
        cli=("train_takheliped.py", "step2", "--frog-mode", "balanced", *RECIPE,
             "--seed", "0", "--out-dir", str(STAGE2), "--resume")),
    Job(name="stage2_avg", outputs=(STAGE2 / AVERAGED,), needs=("stage2",),
        cli=("average_checkpoints.py", str(STAGE2))),
    Job(name="stage3", outputs=(STAGE3 / "step3a_results.json",), needs=("stage2_avg",),
        cli=("train_takheliped.py", "step3a", "--frog-mode", "balanced",
             "--stage2", str(STAGE2 / AVERAGED), *FROZEN,
             "--chunk-s", "1", "10", "--seed", "0", "--out-dir", str(STAGE3), "--resume")),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
