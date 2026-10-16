#!/usr/bin/env python3
"""Step 1's remaining experiments, chained so the machine runs unattended (`TODO.md` §A, the owner's restructure).

**Why a chain.** Each of these is hours, and watching them one at a time costs more attention than the results are worth
until they land. The runner takes them in order, waits out the ROCm teardown hang, judges a job by its **outputs** rather
than its exit code, and skips what is already done -- so an interruption costs the job in flight and nothing else.

**The order is by what a negative result would save.** Item 15 first: it is the only untested stage-2 idea with a prior
behind it, and it is the one that would change the backbone every later arm is screened on, so learning it early is worth
more than learning it late. The five stage-3 design arms follow, cheapest risk first.

    item15_stage2    a first spatial layer reading 1.66 m of arc, resampled per beam from its own range, instead of a
                     fixed seven beams. Tier 0's 1-NN probe priced the adaptive window at +2.5 pp with the advantage
                     **purely long-range** -- +4.9 pp at 8-10 m, zero at 4 m, where the fixed window is calibrated. A59's
                     F1 arm moved the prior against input-*feature* changes, but this changes which beams the layer reads.
    item15_average   weight averaging, which is part of the adopted recipe and a separate script over the periodic
                     checkpoints -- it gained on every seed of C1, +0.11 to +0.36 pp.
    item15_gather    the memory on those candidates, so the arm is judged end to end rather than on stage 2 alone. Its
                     control is the adopted `a60_e13_static_s0`, already trained.

    d3_fixed_decay   one decay time for every slot instead of a learned one (A51 D3; B0 put it at -0.90)
    d2_latest_value  hold a slot by the latest score, not accumulated value (A51 D2; unsupported, not refuted)
    t1_no_rescore_gate  zero-initialise the head instead of scaling it by a scalar that starts at zero (A51 T1)
    o2_coast_weight  weight the coasting loss 3x (A51 O2) -- **asks whether the head was untrained rather than
                     unnecessary**, which is the converse of item 24, so it opts the head back in with `--coast`
    t2_plateau       step 2's plateau schedule against step 3's cosine (A51 T2)
    o1_rescore_limit bound the rescoring to +-4 logits (A51 O1 / item 25). **Diverged with NaN twice and was never
                     explained**, so it is last: a third divergence then costs the queue nothing but its own slot.

Every stage-3 arm is E13's configuration -- the adopted memory, no slot-to-slot exchange, no learned matching -- plus the
one flag it is named for, so the only difference from the model being proposed is the thing under test.

    python queue_step1.py --list
    python queue_step1.py --steps item15_stage2 item15_stage3
"""

from run_queue import CHECKPOINTS, REPO, Job, main

ADOPTED = CHECKPOINTS / "a59_c1_recipe_dropout01_s0" / "step2_calibration.averaged.pth"
GATHER = CHECKPOINTS / "a64_item15_gather_s0"
E13 = ("--no-exchange", "--no-learned-match")
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def arm(name: str, *flags: str, stage2=ADOPTED, needs=()) -> Job:
    """One step-3a run: E13's settings, seed 0, and the single flag under test."""
    out = CHECKPOINTS / f"a64_{name}_s0"
    return Job(name=name, outputs=(out / "step3a_results.json",), needs=needs,
               cli=("train_takheliped.py", "step3a", "--stage2", str(stage2), *E13, *flags,
                    "--chunk-s", "1", "10", "--seed", "0", "--out-dir", str(out), "--resume"))


# A59 C1's own recorded arguments, so the gather is the only difference between this arm and the adopted stage 2. Taken
# from the checkpoint rather than from prose: the recipe is described as "nms-mean decoding, mirror augmentation,
# nearest-person labels, dropout 0.1, weight averaging", and three of those are not flags at all -- nms-mean is the
# decoder everywhere since 2026-10-06, and averaging is a separate script over the periodic checkpoints.
C1 = ("--dropout", "0.1", "--checkpoint-every", "0.25", "--label-assign", "nearest", "--label-radius", "0.6", "--mirror",
      "--batch", "32", "--eval-batch", "64", "--max-epochs", "20", "--eval-every", "0.25", "--patience", "12",
      "--min-epochs", "2", "--val-frames", "2000", "--test-stride", "1", "--lr", "0.001", "--weight-decay", "0.01",
      "--schedule", "plateau", "--lr-factor", "0.3", "--lr-patience", "4", "--min-lr", "1e-05")

JOBS = [
    # `step2_results.json` and not `step2_calibration.best.pth`: the runner stops a job once its outputs exist, and the
    # best checkpoint appears at the first improving epoch. Declaring it cut this run off at epoch 3.25 of 20 (2026-10-16),
    # and the runner was right to call that done -- the output it was told to wait for was there.
    Job(name="item15_stage2", outputs=(GATHER / "step2_results.json",),
        cli=("train_takheliped.py", "step2", "--gather-width-m", "1.66", *C1,
             "--seed", "0", "--out-dir", str(GATHER), "--resume")),
    # Weight averaging is part of the recipe, and a separate script over the periodic checkpoints the run just wrote.
    Job(name="item15_average", outputs=(GATHER / "step2_calibration.averaged.pth",), needs=("item15_stage2",),
        cli=("average_checkpoints.py", str(GATHER))),
    arm("item15_gather", stage2=GATHER / "step2_calibration.averaged.pth", needs=("item15_average",)),

    arm("d3_fixed_decay", "--fixed-decay-s", "8"),
    arm("d2_latest_value", "--latest-value"),
    arm("t1_no_rescore_gate", "--no-rescore-gate"),
    arm("o2_coast_weight", "--coast", "--coast-weight", "3"),
    arm("t2_plateau", "--schedule", "plateau"),
    arm("o1_rescore_limit", "--rescore-limit", "4"),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
