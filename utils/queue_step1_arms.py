#!/usr/bin/env python3
"""Step 1's five stage-3 design arms, on the adopted recipe (`TODO.md` §A, the owner's restructure of 2026-10-15).

**Why these run before the big ablation rather than inside it.** The owner's call: the two GPU-days go to the *final*
architecture only, so every choice that decides what the model **is** gets settled first. Five of A51's six arms are exactly
that -- what holds a slot, how it decays, how the rescoring is gated and bounded, what coasting is worth -- and only T2 is a
training schedule rather than a design choice. They were screened once on the superseded no-temporal backbone, which is the
staleness that made the owner's (a)/(b) question necessary; re-screening them here removes the question instead of answering
it.

Every arm is E13's own configuration -- the adopted memory, without the slot-to-slot exchange and without learned matching --
plus the one flag it is named for, so the only difference from the model being proposed is the thing under test.

    d2_latest_value     hold a slot by the latest candidate score instead of its accumulated value (A51 D2). The design
                        argues accumulation keeps a briefly hidden person; one seed on B0 scored 83.94 against 83.34 and so
                        did not lose, which is "unsupported", not "wrong".
    d3_fixed_decay      one decay time for every slot instead of a learned, input-dependent one (A51 D3). B0 put this at
                        -0.90, and E10 says the learned *state* is the mechanism while the learned *decay* is worth little --
                        which, if it holds here, is a simplification to take.
    t1_no_rescore_gate  zero-initialise the head's last layer instead of scaling it by a scalar that starts at zero (A51 T1).
                        Both start silent; this leaves the head's own gradient unscaled.
    t2_plateau          the plateau schedule step 2 uses, against step 3's cosine (A51 T2). The one arm here that is not an
                        architectural choice, included because it is cheap beside the others and the table wants it.
    o1_rescore_limit    bound the rescoring correction to +-4 logits through a tanh (A51 O1, `TODO.md` item 25). **This arm
                        diverged with NaN twice and was never explained**, so it is last: the runner judges a job by its
                        outputs, so a third divergence costs the queue nothing but its own slot. The bound is 4 because
                        item 23 measured the unbounded head shifting kept people down by a median of 3.292 logits, so +-4
                        leaves the measured correction room and refuses the runaway.
    o2_coast_weight     weight the coasting loss 3x against the candidate loss (A51 O2). Item 27 measured the coasting head
                        firing 2 / 0 / 0 times across 72,533 person-free frames and ~0.001 per populated frame, so this asks
                        whether it is untrained rather than unnecessary -- and item 24 asks the converse.

    python queue_step1_arms.py --list
    python queue_step1_arms.py --steps d3_fixed_decay
"""

from run_queue import CHECKPOINTS, REPO, Job, main

STAGE2 = CHECKPOINTS / "a59_c1_recipe_dropout01_s0" / "step2_calibration.averaged.pth"
E13 = ("--no-exchange", "--no-learned-match")
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def arm(name: str, *flags: str) -> Job:
    """One step-3a run: E13's settings, the adopted stage 2, seed 0, and the single flag under test."""
    out = CHECKPOINTS / f"a64_{name}_s0"
    return Job(name=name, outputs=(out / "step3a_results.json",),
               cli=("train_takheliped.py", "step3a", "--stage2", str(STAGE2), *E13, *flags,
                    "--chunk-s", "1", "10", "--seed", "0", "--out-dir", str(out), "--resume"))


JOBS = [
    arm("d3_fixed_decay", "--fixed-decay-s", "8"),
    arm("d2_latest_value", "--latest-value"),
    arm("t1_no_rescore_gate", "--no-rescore-gate"),
    arm("o2_coast_weight", "--coast-weight", "3"),
    arm("t2_plateau", "--schedule", "plateau"),
    arm("o1_rescore_limit", "--rescore-limit", "4"),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
