#!/usr/bin/env python3
"""A59's attribution ladder has one unmeasured cell: F1 and W together (`TODO.md` §A item 12).

The ladder asked what our stage 2's lead over LFE is *not* made of, one ingredient at a time, against the adopted recipe C1:

    F1  LFE's input alone -- its `_normalize_scan`, `1 - r/10` clipped to 0.2-10 m, as the only channel: **worth nothing
        measurable** once the weights are averaged (82.34% against C1's 82.20%)
    W   LFE's widths, 8/16/32, everything else ours: **-1.4 pp**, and all of it on far people, at 64k parameters

Each was measured alone, so the ladder cannot say whether the two costs add. **That is the difference between "each ingredient
is worth little" and "the combination is what matters"** -- and the second reading is the one a reviewer reaches for, because
LFE has both at once. Three seeds, averaged and probed exactly as the other arms were, so the number lands in the same table.

    python queue_ladder_f1w.py --list
"""

from run_queue import CHECKPOINTS, REPO, Job, main

RESULTS = REPO / "results"
SEEDS = (0, 1, 2)
PROBE = RESULTS / "stage2_probe_a59_f1w.json"
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def arm(seed: int) -> Job:
    """LFE's input and LFE's widths at once, every other setting the adopted recipe's."""
    out = CHECKPOINTS / f"a59_f1w_lfe_input_width_s{seed}"
    return Job(name=f"f1w_s{seed}", outputs=(out / "step2_results.json",),
               cli=("train_takheliped.py", "step2", "--input-features", "lfe", "--channels", "8", "16", "32",
                    "--checkpoint-every", "0.25", "--seed", str(seed), "--out-dir", str(out), "--resume"))


AVERAGED = [CHECKPOINTS / f"a59_f1w_lfe_input_width_s{seed}" for seed in SEEDS]
JOBS = [
    *[arm(seed) for seed in SEEDS],
    Job(name="f1w_average", needs=tuple(f"f1w_s{seed}" for seed in SEEDS),
        outputs=tuple(run / "step2_calibration.averaged.pth" for run in AVERAGED),
        cli=("average_checkpoints.py", *[str(run) for run in AVERAGED])),
    Job(name="f1w_probe", needs=("f1w_average",), outputs=(PROBE,),
        cli=("stage2_probe.py", "--stage2", *[str(run / "step2_calibration.averaged.pth") for run in AVERAGED],
             "--stride", "5", "--out", str(PROBE))),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
