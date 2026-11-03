"""The `balanced` chain: every job points at `balanced`, and nothing it writes can be mistaken for `official`.

The failure this file exists to prevent is not a crash. A job that forgets `--frog-mode balanced` trains on
`official` and succeeds, writing a checkpoint into a directory named `balanced_*` -- and the comparison the run
exists to make then has the same partition on both sides. `balanced`'s val also contains recordings `official`
trains on, so the mistake in the other direction scores a model on its own training data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

import queue_balanced as chain  # noqa: E402


def flag(job, name: str) -> str:
    cli = list(job.cli)
    assert name in cli, f"{job.name} does not pass {name}: {cli}"
    return cli[cli.index(name) + 1]


class TestEveryJobNamesThePartition:
    def test_no_job_leaves_the_partition_to_the_default(self):
        """The default is `official`, so an omission here is a silent `official` run in a `balanced` directory."""
        for job in chain.JOBS:
            if job.name == "stage2_avg":
                continue        # averaging reads a directory, it never loads frames
            assert flag(job, "--frog-mode") == "balanced", job.name

    def test_the_training_jobs_write_outside_every_official_directory(self):
        for job in chain.JOBS:
            for output in job.outputs:
                assert "balanced" in str(output), f"{job.name} writes to {output}"


class TestTheChainIsOrderedByWhatItCosts:
    def test_the_minutes_long_baseline_job_comes_first(self):
        """A cheap result before a GPU-day, so a broken partition shows up in minutes rather than hours."""
        assert chain.JOBS[0].name == "published_baselines"

    def test_stage_3_waits_for_the_averaged_stage_2(self):
        stage3 = next(j for j in chain.JOBS if j.name == "stage3")

        assert stage3.needs == ("stage2_avg",)
        assert chain.AVERAGED in " ".join(str(c) for c in stage3.cli)

    def test_the_gate_runs_the_suite_first(self):
        assert "pytest" in " ".join(str(c) for c in chain.GATE.cli)


class TestItTrainsTheArchitectureTheProjectProposes:
    def test_stage_2_uses_the_adopted_recipe_rather_than_defaults(self):
        stage2 = next(j for j in chain.JOBS if j.name == "stage2")
        cli = " ".join(str(c) for c in stage2.cli)

        assert "--no-aggregator" in cli, "the aggregator was ablated away; a balanced run must not reinstate it"
        assert flag(stage2, "--label-sigma-m") == "0.3", "the soft Gaussian beam target is adopted"

    def test_stage_3_is_the_frozen_memory(self):
        stage3 = next(j for j in chain.JOBS if j.name == "stage3")
        cli = " ".join(str(c) for c in stage3.cli)

        for frozen in ("--no-exchange", "--no-learned-match", "--no-coast"):
            assert frozen in cli, f"{frozen} is part of the frozen architecture"

    def test_one_seed_only(self):
        """The owner's call: three seeds only if `balanced` lands meaningfully away from `official`."""
        seeds = {flag(job, "--seed") for job in chain.JOBS if "--seed" in job.cli}

        assert seeds == {"0"}, seeds


class TestOnlyTheAffordableBaselinesAreScored:
    def test_it_scores_the_two_onnx_detectors(self):
        cli = " ".join(str(c) for c in chain.BASELINES.cli)

        assert "--lfe-peaks" in cli and "--lfe-ppn" in cli

    def test_it_does_not_score_the_per_beam_detectors(self):
        """DROW3 and DR-SPAAM run one forward pass per beam, which is days over 110,846 frames."""
        cli = " ".join(str(c) for c in chain.BASELINES.cli)

        assert "--drow" not in cli and "--drspaam" not in cli
