#!/usr/bin/env python3
"""Item 16: why a bespoke selective recurrence and not a GRU? -- answered on **both** axes (`TODO.md` §A).

This is the question a reviewer asks about the paper's central mechanism, and A51's D3 made it sharper rather than softer:
fixing the decay rate costs **1.04 pp**, so the learned, input-dependent rate is load-bearing. The defence has to be
against the obvious alternatives.

    selective   the adopted rule: a diagonal state whose decay rate is read off the input and applied over the **real
                elapsed time**, `exp(-dt * rate)` in seconds. Already measured -- it is the frozen model, so this queue
                does not retrain it.
    gru         the textbook gated recurrent unit (Cho et al., 2014): update gate, reset gate, state-dependent candidate.
    mingru      Feng et al., *Were RNNs All We Needed?* (2024): no reset gate, no hidden-state dependence in the gate.

**Neither `gru` nor `mingru` sees `dt`.** That is the comparison, not an oversight in the arm: they take a *step*, and a
robot's frames are not uniformly spaced. Bolting `dt` onto a GRU would measure something else.

**Every arm is timed on the CPU, and that is not a secondary axis for this project.** DROW and LFE were built for cheap,
CPU-only robots and this project has no reason to assume otherwise, so an architecture that wins on AP and loses on CPU
has not won. The per-arm `memory_ms_per_frame` already in `step3a_results.json` is measured on **whatever device trained
the arm** -- the GPU here -- so it does not answer the question; each arm therefore gets a `tracking_baselines.py latency
--device cpu --threads 4` job of its own, which is the same protocol as item 29's table (`memory/performance-log.md`).

That matters most for exactly this item: **minGRU exists because it is cheaper**. If it matches on AP and costs less on
the CPU, that is the result; if the GRU matches on AP and costs more, that is also the result.

    python queue_item16.py --list
"""

from run_queue import CHECKPOINTS, REPO, Job, main

RESULTS = REPO / "results"
STAGE2 = CHECKPOINTS / "a59_c1_recipe_dropout01_s0" / "step2_calibration.averaged.pth"
FROZEN = ("--no-exchange", "--no-learned-match", "--no-coast")      # the frozen memory, so only the recurrence differs
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def arm(name: str) -> list:
    """One recurrence: train it, then measure what it costs on a CPU."""
    out = CHECKPOINTS / f"a66_{name}_s0"
    return [
        Job(name=name, outputs=(out / "step3a_results.json",),
            cli=("train_takheliped.py", "step3a", "--stage2", str(STAGE2), *FROZEN, "--recurrence", name,
                 "--chunk-s", "1", "10", "--seed", "0", "--out-dir", str(out), "--resume")),
        Job(name=f"{name}_cpu", outputs=(out / "latency_cpu_t4.json",), needs=(name,),
            cli=("tracking_baselines.py", "latency", "--device", "cpu", "--threads", "4", "--frames", "2000",
                 "--stage2", str(STAGE2), "--memory", str(out / "step3a_object_memory.best.pth"), "--out", str(out))),
    ]


# The adopted arm's own CPU row, measured the same way, so the three are comparable rather than two measured and one quoted.
SELECTIVE = CHECKPOINTS / "a64_no_coast_s0"
JOBS = [
    Job(name="selective_cpu", outputs=(SELECTIVE / "latency_cpu_t4.json",),
        cli=("tracking_baselines.py", "latency", "--device", "cpu", "--threads", "4", "--frames", "2000",
             "--stage2", str(STAGE2), "--memory", str(SELECTIVE / "step3a_object_memory.best.pth"), "--out", str(SELECTIVE))),
    *arm("mingru"),
    *arm("gru"),
]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
