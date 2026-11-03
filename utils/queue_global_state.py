#!/usr/bin/env python3
"""one global scene state in place of 256 addressable slots (`TODO.md` §A).

**The untested cell of the 2x2.** Slots *with* recurrence is the adopted model; slots *without* is `--no-recurrent`
 and costs a lot; **recurrence without slots has never been run**. It is the strongest available attack on this
project's own contribution: if one scene-level state matches 256 addressable per-object slots, the slots are decoration.

**One arm, attention pooling, minGRU** (the owner's call, 2026-10-22): run the *strongest* global variant first to
estimate the ceiling of a swap. If it cannot reach the frozen model's 86% then a masked mean cannot either, and
`global + selective` is moot -- the slots have earned their place and the recurrence is decided by the three-way tie alone. If it
*can*, the swap is not merely "minGRU instead of our rule" but a far simpler architecture with a citable mechanism and
no bespoke slot machinery to defend, and the next question is three seeds (~30 h) to carry the headline.

**Comparator**: seed 0's **86.08%**, frozen three-seed range **85.88-86.08** (mean 86.01), test stride 5, memory column.
Against E13's 1.32 pp three-sigma bar, anything inside that bar of the range is **ambiguous** and must be reported as
ambiguous rather than forced into a verdict.

**Cost.** minGRU with slots took 10.06 h, but this arm is 1.88 ms per frame against its 4.84 -- no 256-slot bookkeeping
and no 256-way attention -- so expect roughly 4-5 h. `--resume` is on: evaluation runs once per epoch and
`step3a_resume.pth` is written after each one, so a pause costs at most one epoch. **A resumed run is a different draw,
not the same run continued** (`memory/interpreting-evaluation.md`); that is fine for a screening arm like this one, and
the report has to say which it was.

    python queue_global_state.py --list
"""

from run_queue import CHECKPOINTS, REPO, Job, main

STAGE2 = CHECKPOINTS / "a59_c1_recipe_dropout01_s0" / "step2_calibration.averaged.pth"
OUT = CHECKPOINTS / "a68_global_mingru_s0"
FROZEN = ("--no-exchange", "--no-learned-match", "--no-coast")     # the frozen memory, so only addressability differs
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

JOBS = [
    Job(name="global_mingru", outputs=(OUT / "step3a_results.json",),
        cli=("train_takheliped.py", "step3a", "--stage2", str(STAGE2), *FROZEN,
             "--global-state", "--recurrence", "mingru",
             "--chunk-s", "1", "10", "--seed", "0", "--out-dir", str(OUT), "--resume")),
    # Its CPU row under the shared protocol. Cheap, and the arm's whole selling point beside simplicity is that it drops
    # the 256-slot bookkeeping -- but quote it only against rows measured in the same session (`interpreting-evaluation.md`).
    Job(name="global_mingru_cpu", outputs=(OUT / "latency_cpu_t4.json",), needs=("global_mingru",),
        cli=("tracking_baselines.py", "latency", "--device", "cpu", "--threads", "4", "--frames", "2000",
             "--stage2", str(STAGE2), "--memory", str(OUT / "step3a_object_memory.best.pth"), "--out", str(OUT))),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
