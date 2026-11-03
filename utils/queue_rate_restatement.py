#!/usr/bin/env python3
"""The FROG analyses whose seconds-axes moved when the frame rate was settled, plus the end-to-end cost table.

Every job here is evaluation only. The rate went from the stamped median's 26.2 Hz to the sensor's
real 40.06 Hz, so a horizon of "10 s" was really 6.5 s in every curve below, and each script's own detection cache stays valid
because only the conversion from frames to seconds changed.

Most jobs write no results file and are judged by their exit code; their numbers are in the per-attempt logs under
`checkpoints_three_horizon/queue_logs/`.

    python queue_rate_restatement.py --list
    python queue_rate_restatement.py --steps cost_cpu cost_gpu
"""

from run_queue import CHECKPOINTS, REPO, Job, main

RESULTS = REPO / "results"
STAGE2 = CHECKPOINTS / "a59_c1_recipe_dropout01_s0" / "step2_calibration.averaged.pth"
MEMORY = CHECKPOINTS / "a60_e13_static_s0" / "step3a_object_memory.best.pth"
LATENCY = CHECKPOINTS / "a61"
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)

COST = [
    # stage 2, candidate decoding and the memory, one scan at a time, timed stage by stage (E22).
    Job(name="cost_cpu", outputs=(LATENCY / "latency_cpu_t4.json",),
        cli=("tracking_baselines.py", "latency", "--device", "cpu", "--threads", "4",
             "--stage2", str(STAGE2), "--memory", str(MEMORY), "--out", str)),
    Job(name="cost_gpu", outputs=(LATENCY / "latency_gpu.json",), needs=("cost_cpu",),
        cli=("tracking_baselines.py", "latency", "--device", "auto",
             "--stage2", str(STAGE2), "--memory", str(MEMORY), "--out", str)),
]

RESTATED = [
    # The frame-rate restatement's re-runs: every FROG analysis with a seconds axis.
    Job(name="rate_motion_people", cli=("motion_analysis.py", "--dataset", "frog", "--people")),
    Job(name="rate_motion_phantoms", cli=("motion_analysis.py", "--phantoms")),
    Job(name="rate_motion_headroom", cli=("motion_analysis.py", "--headroom")),
    Job(name="rate_temporal_oracle", cli=("temporal_oracle.py",)),
    Job(name="rate_horizon_trails", cli=("horizon_sweep.py", "--detector", "lfe-peaks")),
    Job(name="rate_horizon_discrimination", cli=("horizon_sweep.py", "--discrimination")),
    Job(name="rate_persistence", cli=("persistence_map.py",)),
    Job(name="rate_persistence_background", cli=("persistence_map.py", "--background")),
    Job(name="rate_slot_budget", cli=("slot_budget.py",)),
    Job(name="rate_slot_budget_memory", cli=("slot_budget.py", "--memory")),
    Job(name="rate_capacity", cli=("informational_capacity.py",)),
    Job(name="rate_oracle", outputs=(CHECKPOINTS / "oracle_ours.json",),
        cli=("takheliped_oracle.py", "--stage2", str(STAGE2), "--memory", str(MEMORY))),
]

JOBS = [*COST, *RESTATED]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
