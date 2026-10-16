#!/usr/bin/env python3
"""Item 24: does the coasting head earn its parameters? Ablate it outright on the adopted recipe.

The head lets a slot report a person no candidate covers, and two measurements say it almost never does: **0.0012 / 0.0000 /
0.0020** reports per scored frame on the populated split -- one seed emits nothing at all -- and item 27 found **2 / 0 / 0**
across all 72,533 person-free frames at the operating threshold, with exactly zero at 0.5 and 0.9. A paper should not
describe a mechanism that never acts, so this run settles it either way: removing it costs nothing, which is a result, or it
costs something, which is the defence.

`--no-coast` removes the head's parameters *and* its reports. Zeroing `--coast-weight` would only stop it training while
leaving it free to fire at inference, which is a different experiment (that one is A51's O2, in the arm queue).

The control is the adopted model itself -- `a60_e13_static_s0`, already trained and scored -- so this is one run, not two.

    python queue_item24.py --list
"""

from run_queue import CHECKPOINTS, REPO, Job, main

STAGE2 = CHECKPOINTS / "a59_c1_recipe_dropout01_s0" / "step2_calibration.averaged.pth"
OUT = CHECKPOINTS / "a64_no_coast_s0"
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

JOBS = [
    Job(name="no_coast", outputs=(OUT / "step3a_results.json",),
        cli=("train_takheliped.py", "step3a", "--stage2", str(STAGE2), "--no-exchange", "--no-learned-match",
             "--no-coast", "--chunk-s", "1", "10", "--seed", "0", "--out-dir", str(OUT), "--resume")),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
