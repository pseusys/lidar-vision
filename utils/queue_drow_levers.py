#!/usr/bin/env python3
"""The stage-2 levers DROW's gap has left, two seeds each, on the headline's own recipe -- after one score.

**First, the cutout's own train-test gap on DROW.** Our stage 2 fits DROW's train split at 63.57% and reaches test at
29.45%, where FROG keeps them within 13 pp; item 92's retrained DROW3 scored on its training split says whether a local,
depth-normalised window generalises on the same data path. It is the first cell of the four-cell design.

Read against the headline's two seeds -- 29.45% and 26.26%, a 3.19 pp spread -- or, for the multi-class head, against
its own two-seed footing at 0.6 m (`drow_sigmoid_fixed_s0` / `_s1`): a lever counts only if its two-seed mean clears that
spread. Gated, by hand, on `check_drow_scoring.py` passing, since a scoring defect would make every number here unreadable.

1. **DROW's own person radius**, 0.35 m, on the binary head: every stage 2 on file trained a 0.6 m disk, which is
   DR-SPAAM's radius for wheelchairs.
2. **A scale-free input**, `--input-features lfe`, at DROW's 29.99 m: the cutout's depth axis is clipped and
   normalised where ours is raw metres to 30 m.
3. **The cutout's depth tunnel on our input**, `--input-features tunnel`: the ten features with every depth delta
   clipped to DROW3's 1 m, so a doorway or a far wall reads like a cutout's edge rather than as metres of room geometry.
4. **DROW's three radii on the multi-class head** -- person 0.35, wheelchair 0.6, walker 0.4 m -- with the published
   deepest-inside assignment, under both normalisations; the owner's per-class-radius request (2026-11-08).

    python queue_drow_levers.py --list
    python queue_drow_levers.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from run_queue import CHECKPOINTS, Job, REPO, main

DROW = ("--dataset", "drow")
LIMIT = ("--max-range-m", "29.99")
RESTRICTED = ("--eval-max-range-m", "10")
#: The adopted recipe, spelled out as the headline's queue spells it.
RECIPE = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3")
#: DROW's and DR-SPAAM's own radii in `DatasetSpec.classes` order: pedestrian, wheelchair, walker.
DROW_RADII = ("--label-radius", "0.35", "0.6", "0.4")

ARMS = {
    "drow_person_radius": ("--label-radius", "0.35"),
    "drow_lfe_input": ("--input-features", "lfe"),
    "drow_tunnel_input": ("--input-features", "tunnel"),
    "drow_radii_sigmoid": ("--head", "multiclass", "--class-norm", "sigmoid", *DROW_RADII),
    "drow_radii_softmax": ("--head", "multiclass", "--class-norm", "softmax", *DROW_RADII),
}

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2(name: str, flags, seed: int, needs=()) -> Job:
    out = CHECKPOINTS / f"{name}_s{seed}"
    return Job(name=f"{name}_s{seed}", outputs=(out / "step2_results.json",), needs=needs,
               cli=("train_takheliped.py", "step2", *DROW, *LIMIT, *RECIPE, *RESTRICTED, *flags,
                    "--seed", str(seed), "--out-dir", str(out), "--resume"))


#: Item 92's retrained DROW3, scored on its own training split: the cutout's train-test gap on DROW, the first cell of
#: the four-cell design, beside our stage 2's 63.57% train against 29.45% test.
RECIPE_WEIGHTS = REPO / "checkpoints_drow" / "drow_t1_drspaam_recipe.pth"
RECIPE_TRAIN = REPO / "results" / "drow_t1_drspaam_recipe_train.json"

JOBS = [
    Job(name="drspaam_recipe_train", outputs=(RECIPE_TRAIN,),
        cli=("evaluate.py", "--dataset", "drow", "--split", "train", "--drow", str(RECIPE_WEIGHTS), "--no-bench",
             "--out", str(RECIPE_TRAIN))),
    *[stage2(name, flags, seed) for seed in (0, 1) for name, flags in ARMS.items()],
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
