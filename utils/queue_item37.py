#!/usr/bin/env python3
"""Item 37's flaky-stream sweep over four arms, plus item 35's oracle pair (`TODO.md` §A).

Everything here is **evaluation only** -- no training, no checkpoints written -- so the whole chain is about two hours and
every job writes its own results file, which makes a pause between jobs free.

**Why four arms and not three.** Item 16 found the three recurrences tie, and the prerequisite check found why that was
predictable: what reaches `exp(-dt * rate)` is `frame_periods`' 32-interval average, measured at **25.04 ms with a 5-95%
spread of 24.51-25.59** on FROG. A 2% spread makes the adopted rule arithmetically a fixed decay factor, so the arms were
never distinguishable. Dropping frames creates the gaps; `dt_mode` decides what the model is told about them.

    selective   the adopted rule, the control
    mingru      no notion of elapsed time; takes a step
    gru         the same, and the arm a reviewer names
    d3          **the sharpest arm**: the adopted rule with ONE fixed decay time for every slot instead of a learned,
                input-dependent one. D3 cost 1.04 pp on the uniform stream and is the only A51 arm below its floor
                besides O1, so the learned decay is load-bearing *somehow* -- and if what it is for is handling
                non-uniform timing, D3 should lose more here than it did there. If it does not, the learned decay is
                doing something other than tracking real time, which is worth knowing before the mechanism is written up.

`d3` runs on the frozen recipe (`a64_*`), so it is matched to the other three. E10's no-recurrence floor is **not**
included: the only checkpoints for it are `a51_e10_norecurrent_s*` on the older B0 backbone, whose stage 2 differs, and a
cross-backbone row in this table would not be comparable.

    python queue_item37.py --list
"""

from queue_step1_defences import ORACLE
from run_queue import CHECKPOINTS, REPO, Job, main

RESULTS = REPO / "results"
STAGE2 = CHECKPOINTS / "a59_c1_recipe_dropout01_s0" / "step2_calibration.averaged.pth"
ARMS = {"selective": "a64_no_coast_s0", "mingru": "a66_mingru_s0", "gru": "a66_gru_s0", "d3": "a64_d3_fixed_decay_s0"}
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def flaky(arm: str) -> Job:
    """One arm over every drop pattern in both `dt` modes; `slot_experiments.py` sweeps the patterns itself."""
    out = RESULTS / f"flaky_{arm}.json"
    return Job(name=f"flaky_{arm}", outputs=(out,),
               cli=("slot_experiments.py", "flaky", "--stage2", str(STAGE2),
                    "--memory", str(CHECKPOINTS / ARMS[arm] / "step3a_object_memory.best.pth"), "--out", str(out)))


JOBS = [*(flaky(arm) for arm in ARMS), *ORACLE]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
