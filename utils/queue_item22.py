#!/usr/bin/env python3
"""`TODO.md` item 22: is spatial binding enough, or does the memory need object binding? The last pre-freeze arm.

**The gap this closes.** Item 36 proved *addressability* load-bearing -- one global state reaches 82.44% against the
slots' 85.88-86.08% -- but it never tested **which kind** of addressability. "Why slots rather than an occupancy grid?"
is the obvious question about this architecture and a learned room grid is the classical robotics answer, so the
contribution claim has been assumed rather than supported. This arm supports it or breaks it.

**Why a dense grid is not the experiment, measured rather than assumed.** FROG's venue is **120-250 m across** (the
robot covers 670-1,260 m of path per recording). A dense person-scale grid over that needs **~46,000 cells**, which is
180x the recall attention's O(P*K) cost against a memory phase that is 6.70 ms of a 25 ms budget -- a cost experiment
with a foregone verdict. Tiling the same extent with 256 cells instead gives **~11 x 17 m** cells against a 0.5 m match
radius, which is a strawman. Both would answer the wrong question.

**Grid slots (the owner's design, and the headline arm).** Keep the 256 slots; give each one a **quantised room cell**;
never instantiate an empty cell. Capacity, room frame, recurrence, recall, rescoring head and eviction are all
unchanged, so **exactly one thing varies**: an adopted slot's position is continuous and follows its object, while a
grid slot's position is its cell centre and never moves. If object binding does real work, this must cost. Sparsity is
what makes it affordable -- only occupied cells take a slot, so a 150 m venue costs the same 256 states as a small one.

**Cell size is the one free parameter**, and 1.0 m is chosen against the geometry already measured rather than by
taste: the match radius is 0.5 m, the association gate 0.6 m, and only 1.2% of test people have a neighbour within
0.3 m. A 1 m cell therefore holds one person in the ordinary case and is the coarsest size that still separates the
9.3% of test people whose nearest neighbour is under 0.6 m. 2.0 m is queued second as the sensitivity check, because a
single cell size would leave "the grid lost because the cells were wrong" open.

**Reading the result.** Against `a71_dim64_s0`'s **86.12%** and E13's 1.32 pp bar. A tie would be the surprising and
important outcome: it would say the object binding this paper is built on is doing nothing a fixed lattice cannot, and
that has to be known **before** the freeze rather than after publication. A clear loss is the expected one and is worth
as much, because it converts the contribution from an assumption into a measurement.

    python queue_item22.py --list
"""

from run_queue import CHECKPOINTS, REPO, Job, main

AVERAGED = "step2_calibration.averaged.pth"
#: The adopted stage 2 after items 40 and 44(a): aggregator-free, soft Gaussian beam target.
STAGE2 = CHECKPOINTS / "a71_soft_label_s0" / AVERAGED
#: The adopted stage-3 recipe after items 38, 40, 41 and 49.
ADOPTED = ("--no-exchange", "--no-learned-match", "--no-coast", "--recurrence", "mingru", "--chunk-s", "10",
           "--dim", "64")
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def arm(name: str, *flags: str) -> Job:
    out = CHECKPOINTS / f"a73_{name}_s0"
    return Job(name=name, outputs=(out / "step3a_results.json",),
               cli=("train_takheliped.py", "step3a", "--stage2", str(STAGE2), *ADOPTED, *flags,
                    "--seed", "0", "--out-dir", str(out), "--resume"))


JOBS = [
    # The headline: 1 m cells, one person per cell in the ordinary case.
    arm("grid1m", "--cell-m", "1.0"),
    # The sensitivity check, so a loss cannot be blamed on the cell size alone.
    arm("grid2m", "--cell-m", "2.0"),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
