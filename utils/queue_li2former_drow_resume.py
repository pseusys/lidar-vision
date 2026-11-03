#!/usr/bin/env python3
"""`TODO.md` item 31, leg 1 continued: the DROW run the owner paused on 2026-11-01, resumed from its epoch 1.

Attempt 3 launched 13:18:02 and was stopped at the owner's request at 15:58, 87% through epoch 2's train pass.
Epoch 1 is on disk and is a **full** resume point, measured rather than assumed
(`checkpoints_drow/li2former_official_parallel.best.pth`, 203,879,046 bytes): `epoch 1`, `val_wp_auc 0.31308`,
`detector li2former_official`, `dataset drow`, 145 tensors / 17,224,533 elements -- which is the corrected
`MODE: PARALLEL`, `add_bf=False`, `NUM_CTS: 720` class exactly -- and Adam's `exp_avg`/`exp_avg_sq` for all 110
parameter tensors at `step 17665`, one epoch of batch-1 steps, with `lr 1e-4`. So the pause cost epoch 2's 68
minutes of compute and nothing structural: optimiser moments, epoch counter and the selection bar all come back.

**Why a separate module instead of re-running `queue_li2former_drow.py`.** `run_queue.finished()` judges a job by its
declared outputs, and the paused leg's `--out` (`li2former_official_parallel.pth`) was never written -- it is
written once, at the end of `train_model`. A bare re-run would therefore not skip the job, it would start a fresh
12 epochs, and overwrite `li2former_official_parallel.best.pth` the moment its own first epoch beat `-inf`. This
module writes a different `--out`, so its `.best.pth` is a different file and the 31.3% checkpoint survives
whatever happens here. `tests/test_li2former_drow_resume.py` pins both halves of that.

**`--resume` runs N *more* epochs, not N total**, so `--epochs 11` completes the original 12-epoch plan.
At the measured 90.7 min/epoch (train 17,665 frames at 3.83 fr/s, val and the two AUC probes ~10 min together)
that is **~16.6 h**. `--patience 15` cannot fire inside 11 epochs and is inert; it is kept so the command matches
the paused leg's.

**`train.py` seeds `best_val_auc` from the checkpoint's `val_wp_auc`** (0.31308), which is what stops the first
epoch after a resume overwriting a better checkpoint with a worse one. The consequence to know: if **no** epoch
here beats 31.3%, `li2former_official_parallel_resumed.best.pth` is never written and both eval jobs fail on a
missing file. That is the correct failure -- it says the run learnt nothing after epoch 1 -- and the fallback is
to score the paused leg's own `.best.pth` instead.

**The one reading this leg exists for arrives at its third epoch** (epoch 4 of 12). Epoch 1's `vote=` was
bit-identical to attempt 1's frozen pair, 0.1816 train and 0.1410 val, while the classification term moved. Those
are the values their regression loss takes when a ReLU-clamped head emits zeros, which is also where an
almost-untrained head sits, so one epoch cannot tell the two apart. If `vote=` is still bit-identical at epoch 4,
the terminal `nn.ReLU()` on their `reg_head` is fatal on any sane recipe -- a finding about their released code,
not a failure of this port -- and the clamp ablation becomes the next arm rather than an optional one.

**`drow_eval_r03` is a result here, not a diagnostic.** A head emitting zero votes puts every detection at its
own beam's position, which at `d=0.5` is often still inside the matching radius and at `d=0.3` is not. A strong
0.5 column beside a collapsed 0.3 column is that signature, so the two radii are read together: they publish
76.4% AP@0.5 and 72.5% at `d=0.3` (Li2Former-A, `T=5`; 74.8% without the adaptive focus mechanism).

Both eval jobs **declare no outputs, because `evaluate.py` prints**; the runner then waits for the child to exit,
which is correct, since `finished(job)` is false for a job with no declared outputs and no kill timer is ever
armed. They run on **CPU**, as every evaluation here does.

    python queue_li2former_drow_resume.py --list
    python queue_li2former_drow_resume.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows. Expect the **first optimizer
step to take 230-370 s** of ROCm kernel compilation with no output at all -- that is not a hang, and it is per
process and per batch shape (`memory/gotchas.md`). An **exit code of 1 with the declared output present is
success**, not failure: this host's ROCm teardown hangs after the work is done and the runner kills it on its
outputs, as designed.
"""

from run_queue import REPO, Job, main

# What the pause left on disk. Read, never written, by this chain.
RESUME_FROM = REPO / "checkpoints_drow" / "li2former_official_parallel.best.pth"

# A new name, so nothing here can reach `RESUME_FROM`. `.best.pth` is derived rather than spelled out, so the two
# cannot drift apart: the eval jobs score whichever file `train.py` actually writes.
CKPT = REPO / "checkpoints_drow" / "li2former_official_parallel_resumed.pth"
BEST = CKPT.with_suffix(".best.pth")
EPOCHS = "11"

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

# `--out` is written once, at the end of train_model; `.best.pth` is written whenever val improves, so declaring
# that one would arm the runner's kill timer mid-run (item 56's defect).
TRAIN = Job(
    name="drow_train_resume", outputs=(CKPT,),
    cli=("train.py", "--detector", "li2former_official", "--dataset", "drow",
         "--batch-size", "1", "--epochs", EPOCHS, "--patience", "15",
         "--lr", "1e-4", "--no-frame-cache",
         "--resume", str(RESUME_FROM), "--out", str(CKPT)))

# DROW test at d=0.5 m, the radius their 76.4% is quoted at, and evaluate.py's own default.
EVAL = Job(
    name="drow_eval", needs=("drow_train_resume",),
    cli=("evaluate.py", "--dataset", "drow", "--split", "test",
         "--li2former-official", str(BEST)))

# Their d=0.3 column, 72.5% against 76.4%; the column a dead vote head cannot hide in.
EVAL_03 = Job(
    name="drow_eval_r03", needs=("drow_train_resume",),
    cli=("evaluate.py", "--dataset", "drow", "--split", "test", "--eval-r", "0.3",
         "--li2former-official", str(BEST)))

JOBS = [TRAIN, EVAL, EVAL_03]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
