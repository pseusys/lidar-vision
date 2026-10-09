#!/usr/bin/env python3
"""A50 phase 1 item 7's last measurement: the whole detector on FROG `transferred`'s person-free frames, at three seeds.

One job per seed of the adopted recipe -- C1's averaged stage 2 and the E13 memory trained on its candidates (A59, A60, A61) --
each writing `results/empty_scene_s<seed>.json`. The first job of a seed pays about 28 minutes to build that stage 2's candidate
cache over all 195,058 frames of the three raw recordings (~1 GB each), then about 17 minutes to replay the memory over them.

    python queue_a50_person_free.py --list              # what is left to run
    python queue_a50_person_free.py --steps person_free_s0
    python queue_a50_person_free.py                     # the gate, then every seed whose results file is missing

Launch it detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from run_queue import CHECKPOINTS, REPO, Job, main

RESULTS = REPO / "results"
SEEDS = (0, 1, 2)
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def person_free(seed: int) -> Job:
    """Seed `seed` of the adopted recipe counted on the person-free frames."""
    out = RESULTS / f"empty_scene_s{seed}.json"
    return Job(name=f"person_free_s{seed}", outputs=(out,),
               cli=("empty_scene.py",
                    "--stage2", str(CHECKPOINTS / f"a59_c1_recipe_dropout01_s{seed}" / "step2_calibration.averaged.pth"),
                    "--memory", str(CHECKPOINTS / f"a60_e13_static_s{seed}" / "step3a_object_memory.best.pth"),
                    "--out", str(out)))


JOBS = [person_free(seed) for seed in SEEDS]


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
