#!/usr/bin/env python3
"""Item 107 (g): the receptive field, shrunk under the per-beam norm that made it finite.

With `--norm position` stage 2 reads 256 beams (128 deg) instead of the whole scan, and still reaches far DROW
training people -- 64-65% train AP past 5 m against the cutout's 35-52% -- that it cannot reach on the test recordings.
Three smaller fields at almost unchanged capacity, measured by gradient: kernel 5 (180 beams, 958k parameters),
kernel 3 (104 beams, 957k), and kernel 3 without bottleneck blocks and one block per stage (58 beams, 517k). The
prediction, in `TODO.md` row 107 before this ran: if distant context is what reaches far training people, far-range
train AP falls toward the cutout's and far-range test AP rises; nothing moving at range says the field is not it.

    python queue_drow_field.py --list
    python queue_drow_field.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from drow_locality_property import FIELD_RUNS, NORM_SEEDS
from run_queue import Job, REPO, main

RECIPE = ("--dataset", "drow", "--max-range-m", "29.99", "--eval-max-range-m", "10", "--label-radius", "0.35", "0.6", "0.4",
          "--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3", "--norm", "position")
ARMS = {"drow_field_k5": ("--kernel-size", "5"), "drow_field_k3": ("--kernel-size", "3"),
        "drow_field_k3_small": ("--kernel-size", "3", "--bottleneck-blocks", "0", "--blocks", "1", "1", "1")}
READING = REPO / "results" / "drow_field.json"

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2(name: str, seed: int) -> Job:
    out = FIELD_RUNS[name][NORM_SEEDS.index(seed)].parent
    return Job(name=out.name, outputs=(out / "step2_results.json",),
               cli=("train_takheliped.py", "step2", *RECIPE, *ARMS[name], "--seed", str(seed), "--out-dir", str(out), "--resume"))


TRAIN = [stage2(name, seed) for seed in NORM_SEEDS for name in ARMS]
JOBS = TRAIN + [Job(name="field_reading", outputs=(READING,), needs=tuple(j.name for j in TRAIN),
                    cli=("drow_locality_property.py", "--readings", "field", "--out", str(READING)))]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
