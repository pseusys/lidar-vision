#!/usr/bin/env python3
"""The sigmoid arm without its confound, the DROW seed spread nobody has measured, and one capped stage 3.

**Three things 2026-11-07 left behind**, and this queue is sized to the owner's five-to-six hour box rather
than to how much there is to ask.

1. **The sigmoid arm has to be run again, because it measured two things.** `compute_loss` promises that
   `pos_weight` under a soft target is still a count of positive *beams*; the sigmoid branch summed its
   positives after the person column had been overwritten with the Gaussian, so that column's weight was the
   Gaussian's mass. The hard label marks a person's whole extent at 1 while the Gaussian decays, so the weight
   came out larger and the arm fired harder -- 9.349 FP/frame at 93.8% recall against the headline's 8.279 and
   90.6%. Its **+2.65 pp** was therefore a head change *and* a class-balance change. The fix is one expression
   and the test that caught it is in `tests/test_decision_head.py`; the arm runs first here because every later
   job reads its weights.

2. **No DROW number here has an error bar, and the arc arms suggest they need one.** `kernel5` scored 24.20%
   and `kernel3` 28.57%, both below `k = 7`, and **not monotone in `k`** -- no mechanism makes a 5-beam arc
   worse than a 3-beam one. Either the response is genuinely non-monotone or single-seed spread on DROW is
   several points, where FROG's is 0.16 pp over three seeds. Until that is known, 29.45% and 32.10% are not
   figures and the 2.65 pp between them is not a result. So the headline recipe and the fixed sigmoid arm each
   get a **second seed**, which is the cheapest thing in this file and the one that decides what the rest of
   the day's table means.

3. **The whole detector is the only place the multi-class head has left to pay.** `softmax` lost on both halves
   -- 19.50% at stage 2 and 38.70% end to end against 46.01% -- and the val reading that said otherwise
   inverted on test (+23.20 pp against +22.56 the other way), which is why that arm is not repeated and why
   this one produces a test number or nothing.

**The epoch cap is the one place the time box shows.** `--max-epochs 42` is not what the two runs this is read
against had: `drow_stage3_s0` took its best at epoch 36 of 44, `drow_heads_stage3_s0` at 30 of 38. The cap
clears both, so a best inside 42 is comparable and a run still improving at 42 is a result about the cap -- the
log says which, and that is the honest cost of landing inside six hours. The 70 minutes before epoch 1 are the
candidate cache, which every new stage-2 checkpoint invalidates.

**What this queue is not.** Not another capacity, dropout, kernel or gather arm: all four are measured on DROW
and all four lose. Not a `softmax` anything. Not the distance work, which is the one live lead left after
today (+5.20 pp at stage 2 from training at 10 m, with the input-crop explanation dead in the same chain) and
which wants a design rather than a flag.

    python queue_drow_seeds.py --list
    python queue_drow_seeds.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from run_queue import CHECKPOINTS, Job, REPO, main

#: One directory per arm, each naming the one thing it changes. The confounded 32.10% run stays where it is:
#: it is the measurement the fix is read against.
SIGMOID_S0 = CHECKPOINTS / "drow_sigmoid_fixed_s0"
HEADLINE_S1 = CHECKPOINTS / "drow_headline_s1"
SIGMOID_S1 = CHECKPOINTS / "drow_sigmoid_fixed_s1"
SIGMOID_STAGE3 = CHECKPOINTS / "drow_sigmoid_stage3_s0"
AVERAGED = "step2_calibration.averaged.pth"

#: The dataset, which every subcommand takes.
DROW = ("--dataset", "drow")
#: DROW's own sensor limit (owner, 2026-11-04). `step2`'s flag alone -- it shapes the network, so stage 3 reads
#: it back out of the checkpoint and refuses the flag, which stopped a chain nine seconds in on 2026-11-05.
LIMIT = ("--max-range-m", "29.99")
#: Item 82's column at stage 2, both readings. Free on a pass that is already being scored, and it gives item
#: 86's +5.20 pp a second seed on the 29.99 m side without a job of its own.
RESTRICTED = ("--eval-max-range-m", "10")
#: The adopted recipe, spelled out as `queue_drow.py` spells it, so every arm is comparable with the headline
#: rather than with a default that has moved since.
RECIPE = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3")
#: The head whose confound this queue removes. The class count is the dataset's and is not a flag.
SIGMOID = ("--head", "multiclass", "--class-norm", "sigmoid")
#: The frozen memory: no slot exchange, no learned matching, no coasting head.
FROZEN = ("--no-exchange", "--no-learned-match", "--no-coast")
#: Above `drow_stage3_s0`'s best epoch (36) and `drow_heads_stage3_s0`'s (30), below the 100 an uncapped run
#: would allow -- which at five minutes an epoch plus the cache is most of a day.
EPOCH_CAP = ("--max-epochs", "42")

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2(name: str, out, *flags, seed: int = 0, needs=()) -> Job:
    """One DROW stage-2 arm: the adopted recipe, DROW's limit, the restricted column, and one named flag."""
    return Job(name=name, outputs=(out / "step2_results.json",), needs=needs,
               cli=("train_takheliped.py", "step2", *DROW, *LIMIT, *RECIPE, *RESTRICTED, *flags,
                    "--seed", str(seed), "--out-dir", str(out), "--resume"))


JOBS = [
    # First, because everything after it reads these weights: the sigmoid arm with `pos_weight` counted from
    # the hard labels, against the confounded run's 32.10% and the headline's 29.45%.
    stage2("sigmoid_fixed", SIGMOID_S0, *SIGMOID),
    # The two replicates. The headline's is the reference: without it a 2.65 pp difference has no scale.
    stage2("headline_s1", HEADLINE_S1, seed=1, needs=("sigmoid_fixed",)),
    stage2("sigmoid_s1", SIGMOID_S1, *SIGMOID, seed=1, needs=("headline_s1",)),
    # `*DROW` is what makes the ranking correct: without it the windows are scored on FROG val, which is the
    # defect of 2026-11-06 (`memory/gotchas.md`).
    Job(name="sigmoid_avg", outputs=(SIGMOID_S0 / AVERAGED,), needs=("sigmoid_s1",),
        cli=("average_checkpoints.py", str(SIGMOID_S0), *DROW)),
    # The whole detector, capped. Against `drow_stage3_s0`'s 46.01% / 23.45% and the memory's +22.56 pp, and
    # against `drow_heads_stage3_s0`'s 38.70% / 20.73% -- the same head, the other normalisation.
    Job(name="sigmoid_stage3", outputs=(SIGMOID_STAGE3 / "step3a_results.json",), needs=("sigmoid_avg",),
        cli=("train_takheliped.py", "step3a", *DROW, "--stage2", str(SIGMOID_S0 / AVERAGED), *FROZEN,
             *EPOCH_CAP, "--chunk-s", "10", "--seed", "0", "--out-dir", str(SIGMOID_STAGE3), "--resume")),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
