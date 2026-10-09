#!/usr/bin/env python3
"""One queue runner for the project's long jobs, in place of a launcher script hand-copied per session (`TODO.md` A57).

A queue is a Python module that builds a list of `Job`s and calls `main`; it is committed, so the exact queue behind a result
stays readable after the session that ran it (two launchers were already lost that way). The runner owns the machine for the
length of the queue, which is what `memory/gotchas.md` requires -- nothing that imports torch may run beside a training job --
and it is the reason a job here states its dependencies instead of waiting for an idle machine: a chain that stops on failure
makes the machine idle, and the old launchers read that as permission to start the next step.

Launch it detached, as `memory/long-running-tasks.md` shows, so it outlives the session that started it:

    schtasks /create /tn "lidar_queue" /tr "<python> <queue module>" /sc once /st 23:59 /it /f
    schtasks /run /tn "lidar_queue"

Progress goes to `checkpoints_three_horizon/progress.txt` in the format the watchers already poll (`QUEUE <job> ... <stamp>`).
"""

from argparse import ArgumentParser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from re import MULTILINE, search
from subprocess import STDOUT, Popen, TimeoutExpired
from sys import executable

MAX_ATTEMPTS = 3
POLL_SECONDS = 30
EXIT_GRACE_SECONDS = 600
NUMBER_WORDS = {1: "one attempt", 2: "two attempts", 3: "three attempts"}
TRAINING_GUARDS = ("not finite at chunk", "candidates' val AP moved", "the training loss is")
FINISH_MARKERS = (r"^wrote", "Traceback")
LOG_STAMP = "%Y%m%d-%H%M%S"
REPO = Path(__file__).resolve().parent.parent
UTILS = REPO / "utils"
CHECKPOINTS = REPO / "checkpoints_three_horizon"
PROGRESS = CHECKPOINTS / "progress.txt"
QUEUE_LOGS = CHECKPOINTS / "queue_logs"


@dataclass(frozen=True)
class Job:
    """One command the queue runs: what to run, which files prove it finished, and which jobs must finish before it starts."""

    name: str
    cli: tuple[str, ...]
    outputs: tuple[Path, ...] = ()
    needs: tuple[str, ...] = ()
    cwd: Path = UTILS


class Outcome(StrEnum):
    """How a job ended. `BLOCKED` is the state the duplicated launchers had no word for, and the one that cost a wasted run."""

    DONE = "DONE"
    SKIPPED = "SKIP"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class Result:
    """What became of one job: its outcome, how many attempts it took, and why it stopped if it did."""

    outcome: Outcome
    attempts: int = 0
    reason: str = ""


Spawn = Callable[[Job, Path], int]
Note = Callable[[str], None]


def validate(jobs: Sequence[Job]) -> None:
    """Reject a queue that cannot run: a repeated name, a dependency naming no job, or a cycle."""
    names = [job.name for job in jobs]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"two jobs share a name: {', '.join(repeated)}")
    for job in jobs:
        for need in job.needs:
            if need not in names:
                raise ValueError(f"job {job.name} needs {need}, which names no job in the queue")
    ordered(jobs)


def ordered(jobs: Sequence[Job]) -> list[Job]:
    """The jobs with every dependency ahead of what depends on it; declaration order decides between independent jobs."""
    remaining, satisfied, order = list(jobs), set(), []
    while remaining:
        ready = [job for job in remaining if satisfied.issuperset(job.needs)]
        if not ready:
            raise ValueError(f"the dependencies form a cycle among: {', '.join(job.name for job in remaining)}")
        order.extend(ready)
        satisfied.update(job.name for job in ready)
        remaining = [job for job in remaining if job.name not in satisfied]
    return order


def guard_fired(log: str) -> str | None:
    """The training guard this log reports, if any. A guard means the run diverged, so it earns no further attempt."""
    return next((guard for guard in TRAINING_GUARDS if guard in log), None)


def finished(job: Job) -> bool:
    """Has this job already written every file it is supposed to? A job with no outputs is judged by its exit code instead."""
    return bool(job.outputs) and all(output.exists() for output in job.outputs)


def report(results: Mapping[str, Result]) -> str:
    """Every job's outcome as a table, for the end of the run and for the report to the owner."""
    width = max((len(name) for name in results), default=3)
    rows = [f"{'job':<{width}}  {'outcome':<7}  attempts  reason", f"{'-' * width}  {'-' * 7}  --------  ------"]
    rows += [f"{name:<{width}}  {result.outcome:<7}  {result.attempts:>8}  {result.reason}".rstrip() for name, result in results.items()]
    return "\n".join(rows)


def spawn_process(job: Job, log: Path) -> int:
    """Run one job as a child process writing to `log`, polling until it exits, writes its outputs, or prints a traceback."""
    with log.open("w", encoding="utf-8") as stream:
        process = Popen([executable, "-u", *job.cli], cwd=str(job.cwd), stdout=stream, stderr=STDOUT)
    while not _exited_within(process, POLL_SECONDS):
        if finished(job) or _printed_its_last(log):
            if not _exited_within(process, EXIT_GRACE_SECONDS):
                process.kill()
            break
    return process.wait()


def run(jobs: Sequence[Job], *, spawn: Spawn = spawn_process, progress: Path = PROGRESS, log_root: Path = QUEUE_LOGS,
        gate: Job | None = None, steps: Sequence[str] = ()) -> dict[str, Result]:
    """Run every job whose dependencies are met and whose outputs are missing, in dependency order, and record each outcome."""
    validate(jobs)
    note = _notes(progress)
    note("chain START" + (f" steps {','.join(steps)}" if steps else ""))
    results: dict[str, Result] = {}
    if gate is not None:
        results[gate.name] = _attempt(gate, spawn, log_root, note, attempts=1)
    for job in ordered(jobs):
        blocker = _blocker(job, results, gate)
        if blocker is not None:
            results[job.name] = Result(Outcome.BLOCKED, reason=f"{blocker} did not finish")
            note(f"{job.name} BLOCKED, {blocker} did not finish")
        elif steps and job.name not in steps:
            results[job.name] = Result(Outcome.SKIPPED, reason="not among the steps asked for")
        elif finished(job):
            results[job.name] = Result(Outcome.SKIPPED, reason="already done")
            note(f"{job.name} SKIP, already done")
        else:
            results[job.name] = _attempt(job, spawn, log_root, note, attempts=MAX_ATTEMPTS)
    note(_closing_line(results))
    return results


def main(jobs: Sequence[Job], *, gate: Job | None = None, argv: Sequence[str] | None = None, spawn: Spawn = spawn_process) -> int:
    """The entry point a queue module calls: parse the options the PowerShell launchers took, run the queue, print the table."""
    parser = ArgumentParser(description="Run this queue's jobs, skipping the ones whose outputs already exist.")
    parser.add_argument("--steps", nargs="+", default=[], help="run only these jobs; the others count as satisfied")
    parser.add_argument("--progress", type=Path, default=PROGRESS, help="the file the watchers poll")
    parser.add_argument("--log-root", type=Path, default=QUEUE_LOGS, help="where each attempt's log goes")
    parser.add_argument("--list", action="store_true", help="print the queue in dependency order and run nothing")
    args = parser.parse_args(argv)
    if args.list:
        validate(jobs)
        for job in ordered([*([gate] if gate is not None else []), *jobs]):
            print(f"{job.name:<28} needs {','.join(job.needs) or '-':<28} {'done' if finished(job) else 'to run'}")
        return 0
    results = run(jobs, spawn=spawn, progress=args.progress, log_root=args.log_root, gate=gate, steps=args.steps)
    print(report(results))
    return 0 if all(result.outcome in (Outcome.DONE, Outcome.SKIPPED) for result in results.values()) else 1


def _notes(progress: Path) -> Note:
    """Appends one timestamped `QUEUE` line per event, in the format `memory/long-running-tasks.md`'s watchers poll for."""
    progress.parent.mkdir(parents=True, exist_ok=True)

    def note(text: str) -> None:
        with progress.open("a", encoding="utf-8") as stream:
            stream.write(f"QUEUE {text} {datetime.now().isoformat(timespec='seconds')}\n")

    return note


def _attempt(job: Job, spawn: Spawn, log_root: Path, note: Note, *, attempts: int) -> Result:
    """Run one job up to `attempts` times, stopping early when it succeeds or when a training guard fires."""
    log_root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        log = log_root / f"{job.name}.{datetime.now().strftime(LOG_STAMP)}.attempt{attempt}.log"
        note(f"{job.name} attempt {attempt} START")
        code = spawn(job, log)
        note(f"{job.name} attempt {attempt} END exit {code}")
        if finished(job) if job.outputs else code == 0:
            return Result(Outcome.DONE, attempts=attempt)
        guard = guard_fired(log.read_text(encoding="utf-8", errors="replace") if log.exists() else "")
        if guard is not None:
            note(f"{job.name} STOPPED, a training guard fired")
            return Result(Outcome.FAILED, attempts=attempt, reason=f"a training guard fired: {guard}")
        if attempt < attempts:
            note(f"{job.name} attempt {attempt} RETRY")
    note(f"{job.name} STOPPED after {NUMBER_WORDS[attempts]}")
    return Result(Outcome.FAILED, attempts=attempts, reason=f"produced none of its outputs in {NUMBER_WORDS[attempts]}")


def _blocker(job: Job, results: Mapping[str, Result], gate: Job | None) -> str | None:
    """The name of the job that stops this one from starting: the gate, or whichever dependency did not finish."""
    if gate is not None and results[gate.name].outcome is Outcome.FAILED:
        return gate.name
    unfinished = (need for need in job.needs if results[need].outcome in (Outcome.FAILED, Outcome.BLOCKED))
    return next(unfinished, None)


def _closing_line(results: Mapping[str, Result]) -> str:
    """The chain's last progress line: `DONE`, or `STOPPED` naming what failed and what that blocked."""
    stopped = [f"{name} {result.outcome.lower()}" for name, result in results.items() if result.outcome in (Outcome.FAILED, Outcome.BLOCKED)]
    return f"chain STOPPED: {', '.join(stopped)}" if stopped else "chain DONE"


def _exited_within(process: Popen, timeout: float) -> bool:
    """Did this process exit within `timeout` seconds? A job that has written its outputs and lingers is killed, not waited on."""
    try:
        process.wait(timeout=timeout)
    except TimeoutExpired:
        return False
    return True


def _printed_its_last(log: Path) -> bool:
    """Has the job printed the line it ends on -- its results file, or a traceback -- while its process is still alive?"""
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    return any(search(marker, text, MULTILINE) for marker in FINISH_MARKERS)
