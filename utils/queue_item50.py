#!/usr/bin/env python3
"""`TODO.md` item 50: confirm 128 slots with a trained arm. The last unpriced number in the design.

**What the replay said** (2026-10-28, `results/item50_slots.json`, `a71_dim64_s0` played at other capacities with its
weights untouched):

    slots    32   64   128   256   512
    gain   2.03 1.91  2.25  2.26  2.26

**128 ties 256 and 512 adds nothing, but there is a real cliff below it** -- 64 loses 0.35 pp. So 128 is the knee and
256 is twice the capacity the model uses.

**Why a trained arm anyway.** The replay's weights adapted to 256, which makes a flat curve weak evidence in general
-- though less so in the *reducing* direction, since a model that needed those slots would lose immediately when they
were removed, and at 64 it visibly does. The owner's call (2026-10-28) is to confirm rather than adopt on a replay,
because this would otherwise stack an unexercised capacity on top of an already-untested soft-label + `dim 64`
combination.

**The control is `a71_dim64_s0` (86.12%), which already exists -- so this arm trains on the HARD-label stage 2.**
That is deliberate and worth stating, because the adopted recipe uses soft labels. Running 128 against the adopted
stage 2 would mean comparing against a 256-slot arm that does not exist yet (it *is* the freeze's seed 0), i.e. two
variables at once. Matching `a71_dim64_s0`'s stage 2 instead gives **one** variable -- capacity -- which is how every
other knob in this sweep was tested. If capacity and the label target interact, that is a separate question and a
weaker prior than the one this arm answers.

**Reading it.** Against `a71_dim64_s0`'s **86.12%** and E13's 1.32 pp bar. A tie adopts 128 and takes a second bite
out of the CPU budget after `--dim 64`'s 28%, since both of stage 3's costly terms -- the recall attention and
`manage_slots`' distance matrix -- are O(K). A loss keeps 256 and closes the question for good.

    python queue_item50.py --list
"""

from run_queue import CHECKPOINTS, REPO, Job, main

AVERAGED = "step2_calibration.averaged.pth"
#: `a71_dim64_s0`'s own stage 2 -- hard labels -- so capacity is the only thing that differs from the control.
STAGE2 = CHECKPOINTS / "a69_no_aggregator_s0" / AVERAGED
#: The adopted stage-3 recipe at the time the control was trained. `--dim 64` is now a CLI default (item 52) but is
#: passed explicitly here, because a queue module should say what it ran rather than depend on a default that moves.
ADOPTED = ("--no-exchange", "--no-learned-match", "--no-coast", "--recurrence", "mingru", "--chunk-s", "10",
           "--dim", "64")
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

OUT = CHECKPOINTS / "a74_slots128_s0"

JOBS = [
    Job(name="slots128", outputs=(OUT / "step3a_results.json",),
        cli=("train_takheliped.py", "step3a", "--stage2", str(STAGE2), *ADOPTED, "--slots", "128",
             "--seed", "0", "--out-dir", str(OUT), "--resume")),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
