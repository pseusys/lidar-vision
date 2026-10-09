#!/usr/bin/env python3
"""Is this machine free for a second job? Exit 0 if yes, 1 if not.

Run it before `pytest` or any evaluation while something long may be training. Twice now an agent has started the test suite
beside a live training run and killed it: `pytest` imports torch, the run's caches already hold several GB, and the loser is the
run, which dies in `gather` with a tiny `_ArrayMemoryError` or a paging-file error hours in (`memory/gotchas.md`,
`memory/long-running-tasks.md`).

    python memory/scripts/machine_is_free.py && <the command you were about to run>

Reports what it found either way, because a silent pass is indistinguishable from a script that did nothing.
"""

import subprocess
import sys
from pathlib import Path

FREE_MB_REQUIRED = 2048
UTILS = Path(__file__).resolve().parent.parent.parent / "utils"
TEST_SUITE = "-m pytest"       # the suite imports torch too, so it is a job for this purpose
QUOTES = "\"'"                 # Windows command lines quote a path with spaces


def is_project_job(command_line: str) -> bool:
    """Would this python process lose, or cause the loss of, a run? Any script of `utils/`, or the test suite.

    Derived from the directory rather than from a list: the list went stale and reported a free machine during a live job."""
    if TEST_SUITE in command_line:
        return True
    tokens = [token.strip(QUOTES) for token in command_line.split()]
    return any((UTILS / Path(token).name).exists() for token in tokens if token.endswith(".py"))


def running_jobs() -> list:
    """Command lines of the python processes running one of the project's long scripts, or its test suite."""
    query = "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ForEach-Object { $_.CommandLine }"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines() if is_project_job(line)]


def free_megabytes() -> int:
    query = "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True).stdout.strip()
    return int(out) // 1024 if out.isdigit() else -1


def main() -> int:
    jobs, free = running_jobs(), free_megabytes()
    for job in jobs:
        print(f"busy: {job[:120]}")
    if jobs:
        print(f"{len(jobs)} project job(s) running -- do not start another; watch instead (memory/long-running-tasks.md)")
        return 1
    if 0 <= free < FREE_MB_REQUIRED:
        print(f"only {free} MB of memory free, under the {FREE_MB_REQUIRED} MB a second job needs")
        return 1
    print(f"free: no project job running, {free} MB of memory available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
