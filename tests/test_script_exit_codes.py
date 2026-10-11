"""`--help` must exit 0, in every script that carries the ROCm teardown workaround (`TODO.md` §A item 11).

Those scripts end with `os._exit(code)` rather than falling off the end of `main()`, because a finished ROCm process can hang
past a normal exit (`memory/gotchas.md`) and only `os._exit` is certain to end it. The wrapper around it catches
`BaseException`, which also catches the `SystemExit` that `argparse` raises after printing `--help` -- so every one of these
scripts printed its help correctly and then reported failure. A tool whose `--help` exits non-zero breaks any caller that checks
the status, and it is how this was found: a loop verifying that renamed scripts still ran said four of six had failed.

This is checked by running them, because the bug is in the exit path and nothing short of exercising it would catch it.
"""
import subprocess
import sys
from pathlib import Path

import pytest

UTILS = Path(__file__).parent.parent / "utils"
WRAPPED = sorted(path.name for path in UTILS.glob("*.py")
                 if "os._exit(code)" in path.read_text(encoding="utf-8", errors="replace"))


def test_the_scripts_under_test_are_the_ones_carrying_the_workaround():
    """If this list empties, the workaround is gone and so is the reason for this file."""
    assert WRAPPED, "no script carries the os._exit workaround any more"


@pytest.mark.parametrize("script", WRAPPED)
def test_help_exits_zero(script):
    done = subprocess.run([sys.executable, script, "--help"], cwd=UTILS, capture_output=True, text=True, timeout=300)

    assert done.returncode == 0, f"{script} --help exited {done.returncode}\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}"
    assert done.stdout.strip(), f"{script} --help printed nothing"
