#!/usr/bin/env python3
"""The design choices the write-up has to defend, on the adopted recipe (`TODO.md` §A items 8 and 28).

Both jobs ablate or re-measure the *adopted* configuration -- E13, which is the memory without the slot-to-slot exchange and
without learned matching -- rather than the superseded B0 backbone the original screens used.

    d4_learn_value   `step3a --learn-value` makes the value rule's two hand-set constants trainable (`gain`, `fade_time_s`),
                     leaving every discrete decision under `no_grad`. It is the one choice in A51 phase 3b with no measurement,
                     and it is two parameters: either the learned values land on the hand-set ones, which defends them, or they
                     move and the model improves.
    e1_horizon       E1's reset sweep, which `TODO.md` item 5 leans on: A35's "strictly benefits from ordered time" criterion
                     rests on 4.47 pp lost when the memory is emptied every 0.5 s, measured on B0 where the memory was worth
                     +6.22 pp. On the adopted recipe it is worth +3.68, and the sweep has never been re-run there.

    python queue_design_defence.py --list
    python queue_design_defence.py --steps e1_horizon
"""

from run_queue import CHECKPOINTS, REPO, Job, main

RESULTS = REPO / "results"
STAGE2 = CHECKPOINTS / "a59_c1_recipe_dropout01_s0" / "step2_calibration.averaged.pth"
MEMORY = CHECKPOINTS / "a60_e13_static_s0" / "step3a_object_memory.best.pth"
D4 = CHECKPOINTS / "a51_d4_learn_value_s0"
HORIZON = RESULTS / "a51_horizon_adopted.json"
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

JOBS = [
    # The cheap one first: it needs no training and it closes item 5's open caveat.
    Job(name="e1_horizon", outputs=(HORIZON,),
        cli=("slot_experiments.py", "horizon", "--stage2", str(STAGE2), "--memory", str(MEMORY), "--out", str(HORIZON))),
    # E13's own settings plus the one flag, so the only difference from the adopted model is what this measures.
    Job(name="d4_learn_value", outputs=(D4 / "step3a_results.json",),
        cli=("train_takheliped.py", "step3a", "--stage2", str(STAGE2), "--no-exchange", "--no-learned-match",
             "--learn-value", "--chunk-s", "1", "10", "--seed", "0", "--out-dir", str(D4), "--resume")),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
