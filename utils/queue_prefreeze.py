#!/usr/bin/env python3
"""Everything that must settle **before** the freeze, because adopting any of it afterwards would waste a day of GPU.

The test for belonging here is one question: *could this change the architecture the freeze measures?* The freeze is
~24 h (`queue_freeze.py`), and this exact mistake has already been paid once -- an ablation removed the whole-scan
aggregator **after** `a66_mingru_s0` existed, which invalidated it as a control and forced an entire new baseline run
(`CHANGELOG.md`, 2026-10-25). Three arms answer yes to that question and so they run first.

    44(a)  stage 2's soft Gaussian beam target     changes stage 2's weights -> invalidates all three stage-2 seeds
    49a    stage 3 at `--dim 64`                   halves the slot state -> a different model, and the CPU upside
    49b    stage 3 at `--heads 8`                  no expected upside, but an unrun knob is not a measured one

**The soft Gaussian target is the cheap one and it goes first.** `memory/static-detector-diagnosis.md` already measured the *decoder*
half of that pair on fixed weights -- our network through LFE-Peaks' own decoder scores 71.26% against `nms-mean`'s
79.88%, and 2.84 pp of that gap is centre regression a voteless head cannot recover. What no fixed-weight decomposition
can bound is its own caveat: "the network's +6.15 is a lower bound, since our probability map was trained for votes,
not for peaks". So the objective is the open question, and because arm (a) keeps the vote head and `nms-mean`, stage 2
alone is directly comparable through `stage2_probe.py` -- about **1 h**, no stage-3 arm needed unless it wins.

**Its control is `a69_no_aggregator_s0`, not the stage-2 study's three seeds.** The adopted stage 2 has no aggregator, so probing a
soft-label arm against the aggregator-ful control seeds would conflate two changes. One against one at seed 0, which is
how every arm in the sweep was scored.

**One risk taken knowingly, stated rather than discovered.** The two stage-3 arms below run on the *current* stage 2,
so if the soft target wins they would need re-running on the new one -- the aggregator mistake in miniature. It is accepted because a
soft target has a reason to lose that was found while writing it: it **fights the ~50:1 `pos_weight`**, since weighted
BCE is minimised at `w*t / (w*t + 1 - t)` rather than at `t`, so a hard 1 inside the footprint sits closer to the
weighted optimum than the Gaussian does. If it wins anyway, that is a bigger rethink than a re-run.

**Expect ties, and that is the point.** Ten architectural questions have been measured and one was load-bearing. These
three exist so the freeze is not launched on an unpriced knob, and a tie at `--dim 64` is itself a **win** -- it halves
the slot state on the axis that decides deployment, where an architecture that wins on accuracy and loses on CPU has
not won.

**No latency job here, deliberately** (`memory/interpreting-evaluation.md`): the CPU protocol is a matched,
interleaved sweep on an idle host, and a sequential queue runs every job in its own session -- which is how the
wrong `gru_cpu` row was produced, 23.87 ms against a true 20.57. Run it by hand once the arms exist, `dim 64` beside
`dim 128`, reading the `stage2` phase first as the noise floor.

    python queue_prefreeze.py --list
"""

from run_queue import CHECKPOINTS, REPO, Job, main

RESULTS = REPO / "results"
AVERAGED = "step2_calibration.averaged.pth"
#: The adopted stage 2 after the aggregator went, and the control for the soft-target arm.
BASE_STAGE2 = CHECKPOINTS / "a69_no_aggregator_s0"
#: The adopted stage-3 recipe, minGRU and no aggregator or curriculum. `a70_baseline_s0` is its seed-0 score, 86.20%.
ADOPTED = ("--no-exchange", "--no-learned-match", "--no-coast", "--recurrence", "mingru", "--chunk-s", "10")
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

SOFT = CHECKPOINTS / "a71_soft_label_s0"


def stage3(name: str, *flags: str) -> Job:
    """One stage-3 arm of the adopted recipe plus `flags`, seed 0, against `a70_baseline_s0`'s 86.20%.

    **`needs` the soft-target probe, and not because the data flows that way.** `run_queue.ordered()` is layered and
    keeps declaration order *within* a dependency layer, so two jobs with no `needs` outrank anything that waits on a
    predecessor -- which put these 14 h of training ahead of a 10-minute probe. The dependency is the only lever that
    fixes the order, and it buys the right thing: the cheap verdict that could invalidate stage 2 arrives first. The
    cost is that a failed probe leaves these BLOCKED rather than running, which is the correct trade -- a probe that
    cannot run is something to look at before spending a night of GPU, not after.
    """
    out = CHECKPOINTS / f"a71_{name}_s0"
    return Job(name=name, outputs=(out / "step3a_results.json",), needs=("soft_label_probe",),
               cli=("train_takheliped.py", "step3a", "--stage2", str(BASE_STAGE2 / AVERAGED), *ADOPTED, *flags,
                    "--seed", "0", "--out-dir", str(out), "--resume"))


JOBS = [
    # The soft Gaussian target, ~1 h and first because it is the one that could invalidate stage 2 itself. Sigma 0.3 m is half the
    # 0.6 m collect radius, so the target falls to exp(-2) = 0.135 at the edge of its own support -- soft throughout
    # rather than a hard label with rounded corners.
    Job(name="soft_label", outputs=(SOFT / "step2_results.json",),
        cli=("train_takheliped.py", "step2", "--checkpoint-every", "0.25", "--no-aggregator",
             "--label-sigma-m", "0.3", "--seed", "0", "--out-dir", str(SOFT), "--resume")),
    Job(name="soft_label_avg", outputs=(SOFT / AVERAGED,), needs=("soft_label",),
        cli=("average_checkpoints.py", str(SOFT))),
    Job(name="soft_label_probe", outputs=(RESULTS / "stage2_probe_item44a.json",), needs=("soft_label_avg",),
        cli=("stage2_probe.py", "--out", str(RESULTS / "stage2_probe_item44a.json"), "--stage2",
             str(BASE_STAGE2 / AVERAGED), str(SOFT / AVERAGED))),

    # `dim` first: it is the only one of the two with an upside, and it is the knob that could cut the memory
    # phase, which is the larger half of the CPU budget.
    stage3("dim64", "--dim", "64"),
    # `heads` second and last. Same parameter count at any head split, so nothing is expected -- but it could still be
    # adopted on a win, and anything adoptable has to precede the freeze or be formally dropped. Running it once is
    # cheaper than arguing about it, and it is the owner's standing rule (2026-10-23): every idea at least once.
    stage3("heads8", "--heads", "8"),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
