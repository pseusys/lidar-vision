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

    41  the curriculum, `--chunk-s 10` alone            ANSWERED -- it ties, and the simplified recipe adopts it
    39  `sigma_m`, the recall attention's width         a replay mode in slot_experiments (minutes, not an arm)
    40  stage 2's whole-scan max aggregator             a `--no-aggregator` flag and a network_shape key
    45  kernel_size / bottleneck_blocks / neighbours    stage-2 flags; `dim`/`heads` are stage-3 and wait on the cheap half
    43  Hungarian association                           moderate: replace the greedy match in `manage_slots`
    46  O3 position refinement                          moderate: a correction head and an L1 loss
    42  Slot Attention's own update                     LANDED 2026-10-25 as `--iterations N`; arm queued below
    44  an LFE-style Gaussian heatmap head              substantial: changes the loss and the whole decode path
    21  training stage 2 and stage 3 together           a joint arm; the feedback path is closed (E12), this is not
    22  a learned grid over the room                    substantial, and a sanity check rather than a candidate

Item **37 is closed** (2026-10-22) and item **20 is covered by item 39**, which asks the sharper question: S4 already
showed the *association* gate flat, while `sigma_m` -- the width of the **recall attention's** distance bias, and the
parameter item 36 made load-bearing -- has never been varied at all.

    python queue_architecture_sweep.py --list
"""

from run_queue import CHECKPOINTS, REPO, Job, main

RESULTS = REPO / "results"
STAGE2 = CHECKPOINTS / "a59_c1_recipe_dropout01_s0" / "step2_calibration.averaged.pth"
ADOPTED_MEMORY = CHECKPOINTS / "a66_mingru_s0" / "step3a_object_memory.best.pth"
CONTROL_STAGE2 = "a59_c1_recipe_dropout01_s{seed}"
AVERAGED = "step2_calibration.averaged.pth"
ADOPTED = ("--no-exchange", "--no-learned-match", "--no-coast", "--recurrence", "mingru")
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def stage2_arm(name: str, *flags: str) -> list:
    """One stage-2 arm plus its weight average, because the averaged weights are what stage 3 loads."""
    out = CHECKPOINTS / f"a69_{name}_s0"
    return [
        Job(name=name, outputs=(out / "step2_results.json",),
            cli=("train_takheliped.py", "step2", "--checkpoint-every", "0.25", *flags,
                 "--seed", "0", "--out-dir", str(out), "--resume")),
        Job(name=f"{name}_avg", outputs=(out / AVERAGED,), needs=(name,),
            cli=("average_checkpoints.py", str(out))),
    ]


def stage3(name: str, *flags: str) -> Job:
    """One stage-3 arm of the adopted architecture plus `flags`, seed 0, against `a66_mingru_s0`.

    **No CPU latency job, deliberately** (`memory/interpreting-evaluation.md`, 2026-10-24). `queue_item16.py` and
    `queue_item36.py` each paired an arm with one, which looked like hygiene and produced item 16's wrong `gru_cpu` row
    -- 23.87 ms against a true 20.57, because that job's `stage2` control phase read 10.49 ms instead of ~8.6 under
    session load. A sequential queue runs every job in its own session by construction, so it **cannot** produce
    comparable rows. Measure latency as a matched sweep by hand after the chain: every arm plus the control, back to
    back, interleaved, two or three rounds, on an idle host, reading the `stage2` phase first as the noise floor.
    """
    out = CHECKPOINTS / f"a69_{name}_s0"
    return Job(name=name, outputs=(out / "step3a_results.json",),
               cli=("train_takheliped.py", "step3a", "--stage2", str(STAGE2), *ADOPTED, *flags,
                    "--seed", "0", "--out-dir", str(out), "--resume"))


AGGREGATOR = CHECKPOINTS / "a69_no_aggregator_s0"
#: The architecture after the owner's two simplifications of 2026-10-25, both adopted because they tie: stage 2 without
#: the whole-scan aggregator (item 40, 16,384 parameters) and stage 3 without the 1 s curriculum stage (item 41, 24% of
#: training time). Dropping the curriculum also removes the 26-stream configuration behind two of this sweep's five
#: out-of-memory failures, so it buys reliability as well as time.
SIMPLE_STAGE2 = AGGREGATOR / AVERAGED
SIMPLE = (*ADOPTED, "--chunk-s", "10")


def simplified(name: str, *flags: str) -> Job:
    """One arm of the **simplified** architecture, seed 0, against `a70_simple_baseline_s0`.

    Separate from `stage3` on purpose rather than changing it: the arms already finished were run on the aggregator-ful
    stage 2 with the 1 s curriculum, and a job is judged by its output file, so editing `stage3`'s flags would leave
    this module claiming those arms ran with settings they never saw.
    """
    out = CHECKPOINTS / f"a70_{name}_s0"
    return Job(name=f"simple_{name}", outputs=(out / "step3a_results.json",),
               cli=("train_takheliped.py", "step3a", "--stage2", str(SIMPLE_STAGE2), *SIMPLE, *flags,
                    "--seed", "0", "--out-dir", str(out), "--resume"))


JOBS = [
    # The simplified architecture's own baseline, and everything after the 2026-10-25 simplifications is measured
    # against it rather than against `a66_mingru_s0`, whose stage 2 still had the aggregator.
    simplified("baseline"),

    # Item 42. The conclusion of item 38's own logic: item 36 proved the slot abstraction load-bearing, item 38 preferred
    # a citable cell to a re-implemented one, so the remaining question is whether Slot Attention's *own* update also
    # works -- in which case the architecture becomes "Slot Attention over LiDAR candidates". Three iterations, the
    # paper's own setting. `manage_slots` still runs, because it is what gives a slot its room-frame position, and the
    # proximity bias stays because item 39 priced it: this arm varies the update rule and nothing else.
    simplified("slot_attention", "--iterations", "3"),

    # Item 41. The only arm on the list that needed no new code: drop the 1 s warm-up and train on 10 s chunks alone.
    # minGRU spent 1.16 h of its 10.06 h in the 1 s stage, so if this ties, every later arm gets ~12% cheaper and the
    # recipe loses a described component that does nothing.
    stage3("curriculum_10s", "--chunk-s", "10"),

    # Item 39. Minutes, not an arm: `sigma_m` is replayed on the adopted memory with its weights untouched, because
    # `step` reads it on every call. Read a flat curve weakly -- the weights adapted to 0.6 -- and train only if live.
    Job(name="sigma", outputs=(RESULTS / "sigma_mingru.json",),
        cli=("slot_experiments.py", "sigma", "--stage2", str(STAGE2), "--memory", str(ADOPTED_MEMORY),
             "--out", str(RESULTS / "sigma_mingru.json"))),

    # Item 40. A stage-2 arm, so it follows A59's shape: train, average over the val-chosen window because the averaged
    # weights are what stage 3 loads, then one probe pass against the three control seeds (82.19% mean).
    Job(name="no_aggregator", outputs=(AGGREGATOR / "step2_results.json",),
        cli=("train_takheliped.py", "step2", "--checkpoint-every", "0.25", "--no-aggregator",
             "--seed", "0", "--out-dir", str(AGGREGATOR), "--resume")),
    Job(name="no_aggregator_avg", outputs=(AGGREGATOR / AVERAGED,), needs=("no_aggregator",),
        cli=("average_checkpoints.py", str(AGGREGATOR))),
    Job(name="no_aggregator_probe", outputs=(RESULTS / "stage2_probe_item40.json",), needs=("no_aggregator_avg",),
        cli=("stage2_probe.py", "--out", str(RESULTS / "stage2_probe_item40.json"), "--stage2",
             *(str(CHECKPOINTS / CONTROL_STAGE2.format(seed=s) / AVERAGED) for s in (0, 1, 2)),
             str(AGGREGATOR / AVERAGED))),

    # Item 43. The standard primitive in place of mutual nearest-neighbour. Expect it to tie on AP -- S4 put the
    # association's remaining headroom at about 0.8% of people -- and to cost more on the CPU, since it adds a scipy
    # call per frame to the phase item 36 showed is already the larger half of the budget. Worth one arm anyway:
    # "why not Hungarian?" is a reviewer's question about the one component item 36 proved load-bearing.
    stage3("hungarian", "--association", "hungarian", "--chunk-s", "1", "10"),

    # Item 45's cheap half: four stage-2 knobs chosen once and never varied, ~20 min each plus averaging. Scored by one
    # probe pass over all of them against the three control seeds, the same shape items 18/19 and 40 used. `--neighbours`
    # and `--top-p` from the same item are deliberately absent -- the first changes the stem's input width, the second is
    # a candidate-cache key and rebuilding it costs more than the arm.
    *(job for name, flags in (("kernel5", ("--kernel-size", "5")),
                              ("kernel11", ("--kernel-size", "11")),
                              ("deeper", ("--blocks", "3", "3", "3")),
                              ("expansion2", ("--expansion", "2")))
      for job in stage2_arm(name, *flags)),
    # Item 46 (A51's O3). The one arm on this list whose metric is **not** AP: item 35 measured the memory as net
    # negative on coverage -- 203 person-observations recovered against 2,990 lost at its own threshold -- so all of its
    # gain is ranking, and a position correction is the only mechanism here that could turn carried state into recall.
    # Score it on item 35's coverage columns beside AP; an AP-neutral result with better coverage is still a finding.
    stage3("refine", "--refine", "--chunk-s", "1", "10"),

    Job(name="item45_probe", outputs=(RESULTS / "stage2_probe_item45.json",),
        needs=tuple(f"{name}_avg" for name in ("kernel5", "kernel11", "deeper", "expansion2")),
        cli=("stage2_probe.py", "--out", str(RESULTS / "stage2_probe_item45.json"), "--stage2",
             *(str(CHECKPOINTS / CONTROL_STAGE2.format(seed=s) / AVERAGED) for s in (0, 1, 2)),
             *(str(CHECKPOINTS / f"a69_{name}_s0" / AVERAGED) for name in ("kernel5", "kernel11", "deeper", "expansion2")))),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
