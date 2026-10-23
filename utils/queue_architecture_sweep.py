#!/usr/bin/env python3
"""The architecture re-think's sweep: items 39-46, plus 21 and 22, one arm each (`TODO.md` §A).

**The owner's rule, 2026-10-23:** run every idea **at least once**; if the result sits inside variance, drop the idea and
move on -- but do not rule out a *hidden gem*, a sudden substantial gain. The search is explicitly **not** exhaustive and
the write-up says so (`memory/interpreting-evaluation.md`).

**The control is `a66_mingru_s0`** -- the newly adopted architecture at seed 0, test stride 5 memory AP **86.40%** -- and
the bar is E13's **1.32 pp** three-sigma AP bar. An arm inside 85.08-87.72 is "no evidence of a difference", which is a
weaker statement than equivalence and the one to report.

Arms are added here as their code lands, so `--list` is the current state of the sweep rather than the plan. The plan,
with what each one still needs:

    41  the curriculum, `--chunk-s 10` alone            NO CODE -- queued
    39  `sigma_m`, the recall attention's width         a replay mode in slot_experiments (minutes, not an arm)
    40  stage 2's whole-scan max aggregator             a `--no-aggregator` flag and a network_shape key
    45  kernel_size / bottleneck_blocks / neighbours    stage-2 flags; `dim`/`heads` are stage-3 and wait on the cheap half
    43  Hungarian association                           moderate: replace the greedy match in `manage_slots`
    46  O3 position refinement                          moderate: a correction head and an L1 loss
    42  Slot Attention's own update                     substantial: iterated competitive attention
    44  an LFE-style Gaussian heatmap head              substantial: changes the loss and the whole decode path
    21  training stage 2 and stage 3 together           a joint arm; the feedback path is closed (E12), this is not
    22  a learned grid over the room                    substantial, and a sanity check rather than a candidate

Item **37 is closed** (2026-10-22) and item **20 is covered by item 39**, which asks the sharper question: S4 already
showed the *association* gate flat, while `sigma_m` -- the width of the **recall attention's** distance bias, and the
parameter item 36 made load-bearing -- has never been varied at all.

    python queue_architecture_sweep.py --list
"""

from run_queue import CHECKPOINTS, REPO, Job, main

STAGE2 = CHECKPOINTS / "a59_c1_recipe_dropout01_s0" / "step2_calibration.averaged.pth"
ADOPTED = ("--no-exchange", "--no-learned-match", "--no-coast", "--recurrence", "mingru")
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage3(name: str, *flags: str) -> Job:
    """One stage-3 arm of the adopted architecture plus `flags`, seed 0, against `a66_mingru_s0`."""
    out = CHECKPOINTS / f"a69_{name}_s0"
    return Job(name=name, outputs=(out / "step3a_results.json",),
               cli=("train_takheliped.py", "step3a", "--stage2", str(STAGE2), *ADOPTED, *flags,
                    "--seed", "0", "--out-dir", str(out), "--resume"))


JOBS = [
    # Item 41. The only arm on the list that needs no new code: drop the 1 s warm-up and train on 10 s chunks alone.
    # minGRU spent 1.16 h of its 10.06 h in the 1 s stage, so if this ties, every later arm gets ~12% cheaper and the
    # recipe loses a described component that does nothing.
    stage3("curriculum_10s", "--chunk-s", "10"),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
