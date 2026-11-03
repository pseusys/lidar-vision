#!/usr/bin/env python3
"""Step 2: the frozen architecture at three seeds, which is the pre-print's headline (`TODO.md` §A).

**The architecture was frozen on 2026-10-18**, after step 1 ran eight arms and found nothing better than the adopted
choice. What it is, exactly:

    stage 2   the adopted recipe -- dropout 0.1, nearest-person labels at 0.6 m, mirror augmentation, plateau schedule,
              weight averaging over the val-chosen window -- over a **fixed-beam** ConvNeXt U-Net. No metric gather:
              the gather arm cost 1.42 pp at stage 2 and 0.21 pp below the control's floor end to end.
    stage 3   E13's memory: 256 slots in room coordinates, **accumulated** value (not the latest score, D2),
              **learned input-dependent** decay (not one fixed time constant, D3 costs 1.04 pp), no slot-to-slot
              exchange, no learned matching, a scalar rescore gate (T1 is interchangeable), **unbounded** rescoring
              (O1's +-4 logit bound silences the head), a cosine schedule (T2 is interchangeable), and **no coasting
              head** (removing it costs nothing measurable and it fired essentially never).

**Why only two jobs.** Stage 2 is unchanged by the freeze, so the three adopted stage-2 checkpoints stand. Seed 0 of the
frozen memory is already trained as `a64_no_coast_s0` (2026-10-16) with byte-identical flags -- `coast=False`,
`exchange=False`, `learned_match=False`, `schedule=cosine`, `rescore_gate=True`, `rescore_limit=0.0`,
`fixed_decay_s=0.0`, `latest_value=False` -- so re-running it would buy a tidier directory name for five hours of GPU.
It is read into the table from where it is.

**What this table replaces.** The published **85.87%** three-seed mean describes the model *with* a coasting head, which
the architecture no longer has. Until these two land, the headline number and the architecture description are one
component apart, and anything quoting them together has to say so.

    python queue_step2.py --list
"""

from run_queue import CHECKPOINTS, REPO, Job, main

ADOPTED = "a59_c1_recipe_dropout01_s{seed}"
FROZEN = ("--no-exchange", "--no-learned-match", "--no-coast")     # the frozen memory, spelled out
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def seed(n: int) -> Job:
    """The frozen memory on seed `n`'s own stage 2, so the pair is matched rather than mixed."""
    out = CHECKPOINTS / f"a65_frozen_s{n}"
    stage2 = CHECKPOINTS / ADOPTED.format(seed=n) / "step2_calibration.averaged.pth"
    return Job(name=f"frozen_s{n}", outputs=(out / "step3a_results.json",),
               cli=("train_takheliped.py", "step3a", "--stage2", str(stage2), *FROZEN,
                    "--chunk-s", "1", "10", "--seed", str(n), "--out-dir", str(out), "--resume"))


JOBS = [seed(1), seed(2)]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
