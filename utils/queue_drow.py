#!/usr/bin/env python3
"""The proposed architecture trained natively on DROW, one seed, with the stage-2 arms that make it readable.

**What this queue answers.** Whether the architecture carries to a second dataset by changing parameters only,
never the architecture -- the property that makes "adaptive" a fact about the code rather than a claim in the
paper. It is the headline DROW cell, and it is the owner's standing question since 2026-09-15: one model per
dataset, one transferred, or both.

**The four decisions it runs under** (owner, 2026-11-04), each of which changes what the number means:

1. **DROW's own sensor limit, 29.99 m**, not FROG's 10 m. `sanitize_ranges` maps every reading at or past the
   limit onto it, so a 10 m model cannot represent the 8% of DROW's test labels that lie past 10 m, and the
   published DROW baselines score every label. The 10 m-restricted column is reported beside it, because that
   is the only footing a FROG-trained arm can be read on. The flag is `step2`'s alone: the limit shapes the
   network, so stage 3 reads it back out of the stage-2 checkpoint it is given.
2. **One angular grid, DROW's 0.5 deg**, met by decimating FROG rather than interpolating DROW (`--beam-stride
   2`, a separate queue). Nothing is invented here: this queue reads DROW's own 450 beams.
3. The published-figure audit ran first and inverted its own premise -- our DROW3 reproduction *undershoots*
   by 5.2 pp rather than overshooting, so this column is read as a floor and the comparison inside it is fair.
4. **One seed.** Whether a second and third are worth their GPU is decided from this one's distance to DROW's
   baselines, not committed to now.

**Why the two cheap arms run before the expensive stage 3, and not after it.** Stage 2 trains on annotated
frames only -- 17,665 on DROW against FROG's 108,356 -- so each arm is about a quarter of an hour, where stage
3 streams every frame and is the 6-7 h half. DROW's effective training set is **17.8x** smaller than FROG's,
and the one architecture ever trained natively on DROW here overfit textbook-style on regularisation that
works on FROG. So a DROW figure well below FROG's 86.29% has two readings -- the architecture does not carry,
or the data ran out -- and these arms are what tell them apart. Read the **train/val gap**, not only val AP: a
still-falling train loss with train AP climbing away from val is the data-scarcity signature.

**The headline arm keeps the adopted recipe whatever the arms say.** `docs/ARCHITECTURE.md` section 5.1's
table unchanged except the two values the *dataset* sets, because carrying the published parameter table over
untouched is the claim being tested. An arm earns a stage 3 of its own only if the gap it shows is worth 7 h.

**What is not here.** The beam-roll augmentation arm, which needs the augmentation implementing first and is
queued separately; and the FROG->DROW transfer, which needs the decimated FROG chain to exist.

    python queue_drow.py --list
    python queue_drow.py

Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from run_queue import CHECKPOINTS, REPO, Job, main

#: DROW's own directories, kept apart from every FROG run so no averaging or replay can mix datasets. The
#: names carry the dataset for the same reason the checkpoints now store it.
STAGE2 = CHECKPOINTS / "drow_stage2_s0"
CAPACITY_SMALL = CHECKPOINTS / "drow_stage2_small_s0"
CAPACITY_DROPOUT = CHECKPOINTS / "drow_stage2_dropout_s0"
STAGE3 = CHECKPOINTS / "drow_stage3_s0"
AVERAGED = "step2_calibration.averaged.pth"

#: The dataset, which every subcommand takes.
DROW = ("--dataset", "drow")
#: The range limit, which only `step2` takes -- and deliberately, because the limit shapes the network rather
#: than the run: `network_shape` reads `max_range_m` back out of the checkpoint, so stage 3 inherits DROW's
#: 29.99 m from the `--stage2` it is given and would reject the flag. Passing it there anyway stopped this
#: chain on 2026-11-05 after its stage 2 and both arms had already finished.
LIMIT = ("--max-range-m", "29.99")
#: The adopted recipe, spelled out rather than defaulted, so this run reads against the FROG runs it will be
#: compared with: no whole-scan aggregator, the soft Gaussian beam target at 0.3 m.
#: Pinned at 0.6 m: every run on file from this queue trained there, before DROW's default became its own per-class
#: radii (2026-11-09). An arm's own --label-radius comes after it on the command line and wins.
RECIPE = ("--checkpoint-every", "0.25", "--no-aggregator", "--label-sigma-m", "0.3", "--label-radius", "0.6")
#: The frozen memory: no slot exchange, no learned matching, no coasting head.
FROZEN = ("--no-exchange", "--no-learned-match", "--no-coast")

GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

JOBS = [
    # The headline arm's stage 2, at the recipe's own (32, 64, 128) and dropout 0.1.
    Job(name="stage2", outputs=(STAGE2 / "step2_results.json",),
        cli=("train_takheliped.py", "step2", *DROW, *LIMIT, *RECIPE,
             "--seed", "0", "--out-dir", str(STAGE2), "--resume")),
    # Stage 2's 976,707 parameters were sized for FROG's data volume. This one is sized for DROW's, which is
    # 17.8x smaller; it is the cheapest arm on the DROW list and the one the evidence most directly predicts.
    Job(name="capacity_small", outputs=(CAPACITY_SMALL / "step2_results.json",),
        cli=("train_takheliped.py", "step2", *DROW, *LIMIT, *RECIPE, "--channels", "16", "32", "64",
             "--seed", "0", "--out-dir", str(CAPACITY_SMALL), "--resume")),
    # The same question asked of regularisation rather than width, so the two can disagree and mean something.
    Job(name="capacity_dropout", outputs=(CAPACITY_DROPOUT / "step2_results.json",),
        cli=("train_takheliped.py", "step2", *DROW, *LIMIT, *RECIPE, "--dropout", "0.3",
             "--seed", "0", "--out-dir", str(CAPACITY_DROPOUT), "--resume")),
    # `*DROW` is not decoration: without it the windows are ranked on FROG val, which is what happened to this
    # run's first averaging and is why `--rank-only` exists (`memory/gotchas.md`).
    Job(name="stage2_avg", outputs=(STAGE2 / AVERAGED,), needs=("stage2",),
        cli=("average_checkpoints.py", str(STAGE2), *DROW)),
    # A flat 10 s with no curriculum, which is the adopted recipe and **not** what the `balanced` run used.
    # The chunk is in seconds and so transfers by construction, but what it *delivers* does not: 10 s is ~400
    # supervised frames on FROG and about 25 on DROW, which annotates every 5th scan. Pricing a longer chunk
    # is its own arm, after this reports.
    Job(name="stage3", outputs=(STAGE3 / "step3a_results.json",), needs=("stage2_avg",),
        cli=("train_takheliped.py", "step3a", *DROW, "--stage2", str(STAGE2 / AVERAGED), *FROZEN,
             "--chunk-s", "10", "--seed", "0", "--out-dir", str(STAGE3), "--resume")),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
