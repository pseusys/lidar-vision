#!/usr/bin/env python3
"""`TODO.md` item 44(a): confirm the soft Gaussian beam target at three seeds, the last thing before the freeze.

**Why confirm rather than adopt.** The single-seed probe gave **+0.54 pp** at the adopted `nms-mean` decoder
(82.24% -> 82.78%), up in 13 of 14 decoders, with false positives down 21% at the fixed 0.3 threshold. But those
fourteen decoders all read the **same probability map from the same seed**, so they are not fourteen independent
measurements -- a lucky seed moves them together. The honest reading is one measurement of +0.54 pp against a stage-2
run-to-run spread of **~0.30 pp**: about 1.8x the noise. Suggestive, not established, and it would be baked into the
frozen recipe.

**Why it is nearly free.** The freeze needs three stage-2 seeds whichever target wins, so four of the six arms here are
the freeze's own inputs rather than extra work. Net cost over the freeze's baseline plan: **two** stage-2 arms, about
40 minutes.

**The naming is deliberate so nothing is retrained twice.** The hard-label seeds are `a71_stage2_s1/s2`, exactly the
names `queue_freeze.py` already expects, so if hard labels win the freeze's stage-2 half is **already done**. The
soft-label seeds are `a72_soft_s1/s2` beside the existing `a71_soft_label_s0`, so if soft labels win that trio is
complete too. Seed 0 of each arm exists already and is reused in place.

**The decision rule, stated before the numbers arrive so it cannot be fitted to them.** Compare the two trios' mean
`nms-mean` AP from the single probe pass, and read it against the spread *within* each trio, which is this measurement's
own noise estimate:

* mean gain **larger** than the within-trio spread -> adopt the soft target for the freeze;
* mean gain **inside** the spread -> keep hard labels, record 44(a) as "no evidence of a difference at three seeds",
  and note that the single-seed +0.54 pp did not survive replication.

Either way the 21% false-positive reduction at a fixed threshold is worth re-checking in the same pass, because it is
the deployment-relevant column and `deploy/conf.env` runs a fixed threshold.

**No stage-3 arm here.** Arm (a) keeps the vote head and `nms-mean`, so stage 2 alone is directly comparable -- which
is the whole reason this half of item 44 is cheap to settle. A stage-3 arm follows only if the target is adopted, and
then it is the freeze itself.

    python queue_item44a_confirm.py --list
"""

from run_queue import CHECKPOINTS, REPO, Job, main

RESULTS = REPO / "results"
AVERAGED = "step2_calibration.averaged.pth"
#: Seed 0 of each trio, already trained and reused in place rather than repeated.
HARD_S0 = CHECKPOINTS / "a69_no_aggregator_s0"
SOFT_S0 = CHECKPOINTS / "a71_soft_label_s0"
SIGMA_M = "0.3"
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2(name: str, out, seed: int, *flags: str) -> list:
    """One stage-2 arm and its weight average -- the averaged weights are what stage 3 loads and what the probe reads."""
    return [
        Job(name=name, outputs=(out / "step2_results.json",),
            cli=("train_takheliped.py", "step2", "--checkpoint-every", "0.25", "--no-aggregator", *flags,
                 "--seed", str(seed), "--out-dir", str(out), "--resume")),
        Job(name=f"{name}_avg", outputs=(out / AVERAGED,), needs=(name,),
            cli=("average_checkpoints.py", str(out))),
    ]


HARD = [(f"hard_s{s}", CHECKPOINTS / f"a71_stage2_s{s}", s) for s in (1, 2)]
SOFT = [(f"soft_s{s}", CHECKPOINTS / f"a72_soft_s{s}", s) for s in (1, 2)]

JOBS = [
    *(job for name, out, seed in HARD for job in stage2(name, out, seed)),
    *(job for name, out, seed in SOFT for job in stage2(name, out, seed, "--label-sigma-m", SIGMA_M)),
    # One pass over all six, so every checkpoint is scored by the same decoder in the same process. Scoring the trios
    # in separate runs would reintroduce exactly the cross-session drift that invalidated item 16's CPU row.
    Job(name="probe", outputs=(RESULTS / "stage2_probe_item44a_3seed.json",),
        needs=tuple(f"{name}_avg" for name, _, _ in HARD + SOFT),
        cli=("stage2_probe.py", "--out", str(RESULTS / "stage2_probe_item44a_3seed.json"), "--stage2",
             str(HARD_S0 / AVERAGED), *(str(out / AVERAGED) for _, out, _ in HARD),
             str(SOFT_S0 / AVERAGED), *(str(out / AVERAGED) for _, out, _ in SOFT))),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
