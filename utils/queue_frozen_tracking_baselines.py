#!/usr/bin/env python3
"""The tracking baselines re-run on the **frozen** candidates.

`docs/PAPER.md`'s argument 2 -- "the gain is the learned memory, not association" -- is the one a tracking reviewer will
attack, and its table is measured on the **superseded** stage 2: nothing 82.19%, SORT 82.31%, ByteTrack 81.17%,
ByteTrack+CBMOT tuned 83.51%, the memory 86.01%, hence **+2.36 pp over the tuned tracker**. The frozen candidates are a
different network (82.80%, soft Gaussian target, no aggregator) and the memory's gain over them is +3.49 pp rather than +3.82,
so **the margin over the tuned tracker is unmeasured on the model the paper proposes**. A better stage 2 helps a tracker too,
and could help it more or less than it helps the memory -- which is the question, and no arithmetic on the old row answers it.

**Four steps per seed, in the original order, ~4.3 h each.** `rule` (~2.5 h) tunes the memory's own bookkeeping with a rule score over
a 54-config grid on val, then replays test; `track` (~35 min) tunes SORT, ByteTrack and ByteTrack+CBMOT by coordinate descent;
`arms` (~1 h) replays the memory itself and the candidates; `report` (~15 min) does AP with 95% block-bootstrap intervals,
GOSPA, the strata and first-detection latency. Then one `summary` lines the three reports up by arm.

**`rule` cannot be dropped to save 7.5 h, tempting as that is.** `report` picks `best_classical` as the strongest of whichever
of `bytetrack_cbmot`, `rule_cbmot`, `bytetrack`, `sort_m1` happen to **exist on disk** -- and the baseline study found the rule scorer
lands within 1 pp of E14's best, so on the frozen candidates it might *be* the strongest. Omitting it would silently inflate
the memory's margin, which is the failure `AGENTS.md` rule 3 exists to prevent. That is also why every `report` declares
`needs`: a failed or blocked `rule` must **block** its report rather than let it quietly score three methods instead of four.

**Seed 0 first, as the freeze itself was run** -- and that takes an explicit chain, not just declaration order.
`run_queue.ordered()` is Kahn's algorithm **by levels**: it emits every job whose dependencies are satisfied before any job
that depends on one, so a single `needs` on `report_s0` was enough to push all three reports behind all nine replays. Each
job therefore names its predecessor as well as its real data dependencies, which makes the queue a strict sequence: seed 0's
four finish (~4.3 h) and its `report.json` can be read against the unfrozen row while seeds 1-2 run (~8.6 h more). Verify with
`--list`, which prints the real execution order.

**A job declares the files it writes last, and `track` learnt that the hard way.** `run_queue.spawn_process` starts a
600 s kill timer the moment every declared output exists, so declaring `track_rows.json` -- which the coordinate descent
writes ten minutes before the replays finish -- made the runner cut the job off mid-work. Seed 0 survived by six seconds;
seeds 1 and 2 lost `bytetrack_cbmot`'s test pickle, `run_report` dropped the method silently, and `best_classical` fell to
`sort_m1`, reading the memory's margin as +3.56 pp instead of +1.98 pp. Every replay is therefore declared.

**A strict chain means one failure blocks the rest, and that is the intended trade.** A transient failure in seed 1 costs
seed 2's 4.3 h, recovered by relaunching (the runner skips what is already done). A **partial** result set that looks
complete costs more: `summary` is only meaningful over three seeds, and a report missing a method silently flatters the
memory.

**It points at the originals under `checkpoints_three_horizon/`, not at `weights_archive/frozen-frog-2026-10-30/`.** The
archive is the citable copy and `candidate_cache_path` keys the ~1 GB candidate cache on the stage-2 checkpoint's run
directory, file name and mtime, so a copy rebuilds it from scratch (`weights_archive/.../MANIFEST.md`).

**The arm is labelled `FRZ`, deliberately not `E13`.** `summary` lines arms up by label across seeds, and these are different
weights from the unfrozen static arm; sharing its name is how a cross-regime comparison gets made by accident.

    python queue_frozen_tracking_baselines.py --list                    # what is left to run
    python queue_frozen_tracking_baselines.py --steps rule_s0 track_s0
    python queue_frozen_tracking_baselines.py                           # the gate, then every step whose outputs are missing

Expect `arms` to hang in ROCm teardown after writing everything, as it did before -- the runner judges a job by its outputs and
kills a child that lingers past them. Launch detached through the Task Scheduler, as `memory/long-running-tasks.md` shows.
"""

from run_queue import CHECKPOINTS, REPO, Job, main

SEEDS = (0, 1, 2)
AVERAGED = "step2_calibration.averaged.pth"
STAGE2_DIRS = {0: "a71_soft_label_s0", 1: "a72_soft_s1", 2: "a72_soft_s2"}
MEMORY_RUNS = {seed: f"a71_freeze_s{seed}" for seed in SEEDS}
ARM = "FRZ"
TRACK_METHODS = ("sort_m1", "sort_m3", "bytetrack", "bytetrack_cbmot")
BASE = CHECKPOINTS / "a74"
GATE = Job(name="gate", cli=("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"), cwd=REPO)


def out_dir(seed: int):
    return BASE / f"s{seed}"


def stage2(seed: int) -> str:
    return str(CHECKPOINTS / STAGE2_DIRS[seed] / AVERAGED)


def arm_of(seed: int) -> str:
    return f"{ARM}:{MEMORY_RUNS[seed]}"


def rule(seed: int) -> Job:
    """The memory's bookkeeping with a rule score: a 54-config grid on val, then test."""
    out = out_dir(seed)
    return Job(name=f"rule_s{seed}", outputs=(out / "rule_val.pkl", out / "rule_test.pkl"),
               cli=("tracking_baselines.py", "rule", "--stage2", stage2(seed), "--out", str(out)))


def track(seed: int) -> Job:
    """SORT, ByteTrack and ByteTrack+CBMOT, tuned by coordinate descent on val, then test.

    The eight replays are the product; `track_rows.json` is written before them and cannot stand for the job being done.
    """
    out = out_dir(seed)
    replays = tuple(out / f"track_{method}_{split}.pkl" for method in TRACK_METHODS for split in ("val", "test"))
    return Job(name=f"track_s{seed}", outputs=(out / "track_rows.json", *replays),
               cli=("tracking_baselines.py", "track", "--workers", "6", "--stage2", stage2(seed), "--out", str(out)))


def arms(seed: int) -> Job:
    """The memory itself and the candidates it was given, replayed on test and val."""
    out = out_dir(seed)
    stem = f"arm_{ARM}_{MEMORY_RUNS[seed]}"
    return Job(name=f"arms_s{seed}",
               outputs=(out / f"{stem}_test.pkl", out / f"{stem}_val.pkl",
                        out / "candidates_test.pkl", out / "candidates_val.pkl"),
               cli=("tracking_baselines.py", "arms", "--stage2", stage2(seed), "--out", str(out),
                    "--arms", arm_of(seed)))


def report(seed: int) -> Job:
    """AP with 95% block-bootstrap intervals, GOSPA, strata, first-detection latency -- over all three of the seed's steps."""
    out = out_dir(seed)
    return Job(name=f"report_s{seed}", outputs=(out / "report.json",),
               needs=(f"rule_s{seed}", f"track_s{seed}", f"arms_s{seed}"),
               cli=("tracking_baselines.py", "report", "--stage2", stage2(seed), "--out", str(out),
                    "--arms", arm_of(seed)))


SUMMARY = Job(name="summary", outputs=(BASE / "summary.json",),
              needs=tuple(f"report_s{seed}" for seed in SEEDS),
              cli=("tracking_baselines.py", "summary", "--out", str(BASE)))


def chained(jobs: list[Job]) -> list[Job]:
    """Each job also needs the one before it, so `ordered()`'s levels collapse to the sequence written here."""
    out = []
    for previous, job in zip([None, *jobs], jobs):
        needs = job.needs if previous is None else tuple(dict.fromkeys((*job.needs, previous.name)))
        out.append(Job(name=job.name, cli=job.cli, outputs=job.outputs, needs=needs, cwd=job.cwd))
    return out


JOBS = chained([step(seed) for seed in SEEDS for step in (rule, track, arms, report)] + [SUMMARY])


if __name__ == "__main__":
    raise SystemExit(main(JOBS, gate=GATE))
