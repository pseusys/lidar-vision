#!/usr/bin/env python3
"""`TODO.md` item 31, leg 1: Li2Former's **own** network trained on DROW, the dataset it published on.

Item 31 wanted a measured FROG row for Li2Former. It cannot come from `Li2FormerDetector`: that class was written from the
paper's prose and is a conv backbone with one temporal encoder layer and no cross-beam path, where the paper cascades
omni-dimension aggregation blocks with a batch-attending auxiliary stream and an adaptive focus mechanism in its regression
loss. Training it would have filled the Li2Former row with a number Li2Former did not produce. The authors publish code
(`ai-winter/Li2Former-v1`, pinned at `a19e134`), so `OfficialLi2FormerDetector` wraps **theirs**, downloaded by
`library/setup.py` into the gitignored `include/` tree rather than vendored, because it is GPL-3.0 and this repository is MIT.

**DROW comes before FROG, and that ordering is the whole point of this leg.** They publish 76.4% AP@0.5 on DROW
(Li2Former-A, `T=5`; 74.8% without the AFM, 72.5% at `d=0.3`). Nothing else can tell a correct port from a broken one:
their repository ships **no weights** -- no releases, nothing in the tree, no link in the README, confirmed from both ends on
2026-11-01 -- so there is no output to match numerically, and no training code either, so the optimiser, schedule and epoch
count below are this project's. A FROG number from an unvalidated port would be exactly the mislabelled row item 31 exists to
avoid, one dataset further from anything that could catch it.

**This is attempt 2, and attempt 1's configuration is the reason.** The first run (2026-11-01 00:51 to 11:30, abandoned at
epoch 10 of 12) used `MODE: TEMPORAL` and `train.py`'s default `--lr 1e-3`, and converged to ~30% val wp-AUC. Both were
wrong. `TEMPORAL` is the setting in their `cfgs/ros_li2former.yaml`, which is their **ROS robot deployment** config: it runs
the 1-layer temporal encoder alone, never calls the 3-layer spatial encoder, and so has no cross-beam path at all --
**22,068,736 of their 29,466,056 parameters received no gradient**, and the active inference path was 4,243,912, the size of
the hand-written detector this port replaced. The paper's ODA module "jointly encode[s] temporal and spatial information",
so `PARALLEL` is the mode (`SERIAL` is the alternative; `PARALLEL` is the only one that uses their `fusion` MLP). And `1e-3`
on plain Adam with no warmup is ~10x the usual for a 29.5 M pre-LN transformer, which is what killed their regression head
in two epochs. All three departures from their yaml are asserted in `tests/test_li2former_port.py`.

**Attempt 2 then crashed on its first batch, three times, and the third departure is why.** `NUM_CTS: 90` is their robot's
cutout count, and `Li2Former.__init__` also uses it as `PositionalEncodingSine(max_len=num_cts)`, whose `forward` returns
`pe[:, :x.size(1), :]` -- short rather than raising. `TEMPORAL` attends over the `T` scans and never reached it; the
omni-dimension modes attend over the beams, and DROW has 450 of them, so every attempt died 01:02 in with
`The size of tensor a (450) must match the size of tensor b (90)` and then sat ~12 minutes in ROCm teardown. It is **720**
here, the widest scan in this repository (DROW 450, JRDB 541, FROG 720); the encoding is a registered buffer and
closed-form in the position, so this changes no parameter and no value, and `ConvBackbone` ignores the argument.
Attempt 3 is attempt 2's configuration with that one value corrected, preceded by a one-epoch `--subsample 0.01` smoke run
that reached the first optimiser step in ~1 minute -- which is the cheap check the three launches paid 40 minutes to skip.

**One deviation is forced and is stated rather than hidden:** their `__init__` hardcodes the BatchFormer auxiliary stream on,
but their `SERIAL`/`PARALLEL` branches never reshape the doubled batch and carry the comment *"Keep add_bf option False"*, so
in the released code the auxiliary stream and the omni-dimension modes are mutually exclusive. It is therefore off here.
The paper's abstract describes ODA as having both a main and an auxiliary stream, so **no mode in their released code is the
full published architecture** -- the single-dimension modes also own their learnable `alpha` pooling, which `PARALLEL` does
not read. The BatchFormer is training-only, so the trained model's inference path is unaffected.

**Twelve epochs now, not forty.** DROW train is ~17.5k annotated frames. The 5.25 fr/s measured for attempt 1 no longer
applies -- `PARALLEL` adds 3 layers of 450-token cross-beam attention and drops the batch-dimension attention with the
auxiliary stream, so the rate has to be read off this run's own log. Twelve epochs buys the val wp-AUC curve, which is what
distinguishes "the port is wrong" from "it is still climbing"; extending is one `--resume`, which runs N *more* epochs, so
leg 2 would be a separate job with its own `--out`, since this one's output file existing is what makes the runner skip it.

**Batch 1 is kept for comparability, no longer forced.** It was forced by the BatchFormer attending over `B x N_beams`
tokens (3,600 at batch 8, ~2 GB of attention per layer against this card's ~10.7 GB). With the auxiliary stream off that
constraint is gone and a larger batch may well fit, but changing it would move two variables at once against attempt 1.

**The train probe stays on** (~6 min/epoch). If the number undershoots, the question is immediately whether the port is wrong
or DROW is too small for the model -- and DROW-native training here has textbook-overfitted before, at 6.1x fewer annotated
frames than FROG (`CHANGELOG.md`). The probe is what separates those two without another run.

**`drow_eval` declares no outputs, because `evaluate.py` writes none** -- it prints. The runner then simply waits for the
child to exit, which is correct: `finished(job)` is false for a job with no declared outputs, so no kill timer is ever armed.
Its product is its own log under `checkpoints_three_horizon/queue_logs/`. It runs on **CPU**, as every evaluation here does,
so it does not contend with anything on the GPU.

    python queue_li2former_drow.py --list
    python queue_li2former_drow.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows. Expect the **first optimizer step of any
run to take 230-370 s** of ROCm kernel compilation with no output at all -- that is not a hang, and it is per process and per
batch shape (`memory/gotchas.md`).
"""

from run_queue import REPO, Job, main

# A new name, so attempt 1's abandoned `.best.pth` is not silently overwritten by a run it cannot be
# compared with. It is kept on disk, unreferenced, until the owner says to delete it.
CKPT = REPO / "checkpoints_drow" / "li2former_official_parallel.pth"
EPOCHS = "12"
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

# `--out` is written once, at the end of train_model; `.best.pth` is written whenever val improves,
# so declaring that one would arm the runner's kill timer mid-run (item 56's defect).
TRAIN = Job(
    name="drow_train", outputs=(CKPT,),
    cli=("train.py", "--detector", "li2former_official", "--dataset", "drow",
         "--batch-size", "1", "--epochs", EPOCHS, "--patience", "15",
         "--lr", "1e-4", "--no-frame-cache", "--out", str(CKPT)))

# DROW test at d=0.5 m, the radius their 76.4% is quoted at, and evaluate.py's own default.
EVAL = Job(
    name="drow_eval", needs=("drow_train",),
    cli=("evaluate.py", "--dataset", "drow", "--split", "test",
         "--li2former-official", str(CKPT.with_suffix(".best.pth"))))

# Their d=0.3 column, 72.5% against 76.4%. `evaluate.py` takes one radius per pass, so it is a second
# job rather than a flag -- and it is the column a broken spatial head cannot hide in, which is why it is
# in the chain this time instead of a follow-on module (`memory/interpreting-evaluation.md`).
EVAL_03 = Job(
    name="drow_eval_r03", needs=("drow_train",),
    cli=("evaluate.py", "--dataset", "drow", "--split", "test", "--eval-r", "0.3",
         "--li2former-official", str(CKPT.with_suffix(".best.pth"))))

JOBS = [TRAIN, EVAL, EVAL_03]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
