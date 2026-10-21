#!/usr/bin/env python3
"""Items 18 and 19: the head's target frame, and the block that carries the depth (`TODO.md` §A).

Both are **defence items** rather than hoped-for improvements, and both are cheap: a stage-2 run is about twenty
minutes, so three arms at three seeds cost roughly the same as half of one stage-3 arm.

    sensor      item 18. The head regresses the offset to a person's centre in the **sensor's** one Cartesian frame
                instead of DROW's beam-relative `(across, along)`. A59 put about half this project's lead over
                LFE-Peaks in the output head, so the head is already the good part -- the question is whether a frame
                that makes one person one target, rather than a different target for every bearing that sees it, is
                easier to learn. The two name the same point, so only the decoders' input is rotated.
    residual    item 19a. A plain residual block (He et al., CVPR 2016) in place of ConvNeXt's: two full convolutions,
                no depthwise separation, no inverted bottleneck. The block has never been ablated.
    wide        item 19b. The widths doubled to 64/128/256. A59's W arm priced *shrinking* to LFE's widths at 1.4 pp
                and bought no speed at batch 1, so capacity is not where the score is -- but widening was never tried,
                and this is the arm that says whether the ladder is flat in both directions.

**The control is already trained**: `a59_c1_recipe_dropout01_s{0,1,2}` is the adopted stage 2 at the same three seeds,
on the same recipe, so the arms differ from it in exactly one flag each. Nothing here retrains it.

Every arm goes through the whole stage-2 recipe, averaging included (`average_checkpoints.py`, +0.66 pp on average),
because the averaged weights are what stage 3 actually loads; scoring the best checkpoint instead would compare
something the architecture does not use. The probe then scores all twelve in one pass against the LFE baselines, with
the per-range and per-crowding breakdowns that priced A59's arms.

**This is a stage-2 answer, not an end-to-end one.** A winner here earns a stage-3 run on top; a loser does not.

    python queue_items1819.py --list
"""

from run_queue import CHECKPOINTS, REPO, Job, main

RESULTS = REPO / "results"
SEEDS = (0, 1, 2)
CONTROL = "a59_c1_recipe_dropout01_s{seed}"
AVERAGED = "step2_calibration.averaged.pth"
ARMS = {"sensor": ("--vote-frame", "sensor"),
        "residual": ("--block", "residual"),
        "wide": ("--channels", "64", "128", "256")}
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def run(arm: str, seed: int) -> list:
    """One arm on one seed: the adopted recipe plus that arm's single flag, then its weight average."""
    out = CHECKPOINTS / f"a67_{arm}_s{seed}"
    name = f"{arm}_s{seed}"
    return [
        Job(name=name, outputs=(out / "step2_results.json",),
            cli=("train_takheliped.py", "step2", "--checkpoint-every", "0.25", *ARMS[arm],
                 "--seed", str(seed), "--out-dir", str(out), "--resume")),
        Job(name=f"{name}_avg", outputs=(out / AVERAGED,), needs=(name,),
            cli=("average_checkpoints.py", str(out))),
    ]


JOBS = [job for arm in ARMS for seed in SEEDS for job in run(arm, seed)]
JOBS.append(Job(name="probe", outputs=(RESULTS / "stage2_probe_items1819.json",),
                needs=tuple(f"{arm}_s{seed}_avg" for arm in ARMS for seed in SEEDS),
                cli=("stage2_probe.py", "--out", str(RESULTS / "stage2_probe_items1819.json"), "--stage2",
                     *(str(CHECKPOINTS / CONTROL.format(seed=s) / AVERAGED) for s in SEEDS),
                     *(str(CHECKPOINTS / f"a67_{arm}_s{s}" / AVERAGED) for arm in ARMS for s in SEEDS))))


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
