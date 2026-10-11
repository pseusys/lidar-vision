"""The guard in `memory/scripts/machine_is_free.py`, which decides whether a second job may start.

It exists because an agent killed a live training run with `pytest` twice (`memory/gotchas.md`: the commit limit, not RAM, and
the victim is whatever allocates next). It used to carry a hand-written list of six script names, and on 2026-10-09 it reported
a free machine while `empty_scene.py` was replaying the memory on the GPU -- a stale list is indistinguishable from a free
machine, which is the one failure this script must not have. So the rule is derived from `utils/` instead of listed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "memory" / "scripts"))

from machine_is_free import UTILS, is_project_job

REPO = Path(__file__).parent.parent


class TestIsProjectJob:
    def test_a_training_run_is_a_project_job(self):
        assert is_project_job(r"C:\repo\.venv\Scripts\python.exe -u train_takheliped.py step3a --stage2 x.pth")

    def test_a_script_added_to_utils_after_this_test_was_written_is_recognised(self):
        """The list went stale once; every `.py` in `utils/` counts, so adding a script cannot silently defeat the guard."""
        for script in ("empty_scene.py", "tracking_baselines.py", "stage2_probe.py", "average_checkpoints.py", "run_queue.py"):
            assert (UTILS / script).exists(), f"{script} is gone -- update this test, not the guard"
            assert is_project_job(f"python -u {script} --out x.json"), script

    def test_a_relative_path_to_the_script_is_recognised(self):
        assert is_project_job(r"python ..\utils\slot_experiments.py cost --memory x.pth")

    def test_the_test_suite_counts_because_it_imports_torch_too(self):
        assert is_project_job(r"C:\repo\.venv\Scripts\python.exe -m pytest tests -q")

    def test_an_unrelated_python_process_is_not_a_project_job(self):
        assert not is_project_job(r"C:\Python312\python.exe -c print(1)")
        assert not is_project_job(r"python -m http.server")

    def test_a_script_that_is_not_this_project_is_not_a_project_job(self):
        assert not is_project_job(r"python C:\somewhere\else\train_other_thing.py")

    def test_an_empty_command_line_is_not_a_project_job(self):
        assert not is_project_job("")
