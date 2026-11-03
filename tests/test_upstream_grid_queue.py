"""The upstream-grid chain: the 2x2 at both of their published radii, over epoch 8's weights.

Two jobs, not eight. Each pass runs inference once and scores it four ways, so a radius costs one pass rather
than four, and the baseline cell of each is the number `drow_eval` already reported at that radius -- 48.1% at
d=0.5 and 40.5% at d=0.3. That equality is what `tests/test_upstream_grid_wiring.py` pins; this file pins that
the chain asks for it at all, and that it scores the checkpoint the collapse spared rather than the final one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

import queue_upstream_grid as item  # noqa: E402


def flag(job, name: str) -> str:
    """The value following `name` in a job's CLI, so an assertion reads the command the runner will run."""
    cli = list(job.cli)
    assert name in cli, f"{job.name} does not pass {name}: {cli}"
    return cli[cli.index(name) + 1]


class TestItScoresTheCheckpointTheCollapseSpared:
    def test_both_passes_score_the_resumed_legs_best_checkpoint(self):
        """Epoch 8's weights. The run's own `--out` holds epoch 12, whose vote head had left the basin."""
        for job in item.JOBS:
            assert flag(job, "--li2former-official") == str(item.BEST)
        assert item.BEST.name == "li2former_official_parallel_resumed.best.pth"

    def test_it_does_not_score_the_final_checkpoint(self):
        for job in item.JOBS:
            assert "li2former_official_parallel_resumed.pth" not in " ".join(str(c) for c in job.cli)


class TestBothPublishedRadiiAreCovered:
    def test_one_pass_per_radius(self):
        radii = {flag(job, "--eval-r") for job in item.JOBS}

        assert radii == {"0.5", "0.3"}

    def test_every_pass_asks_for_the_grid(self):
        for job in item.JOBS:
            assert "--upstream-grid" in job.cli, job.name

    def test_each_pass_runs_on_the_drow_test_split(self):
        for job in item.JOBS:
            assert flag(job, "--dataset") == "drow"
            assert flag(job, "--split") == "test"


class TestTheRunnerCannotKillThesePassesEarly:
    def test_no_job_declares_an_output_because_evaluate_py_prints(self):
        """A declared output arms a 600 s kill timer the moment it exists -- the runner's own defect."""
        for job in item.JOBS:
            assert job.outputs == (), job.name

    def test_the_gate_runs_the_test_suite_first(self):
        assert "pytest" in " ".join(str(c) for c in item.GATE.cli)
