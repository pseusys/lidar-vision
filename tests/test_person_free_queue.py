"""The person-free comparison: both arms counted on the same frames, read on the same scale.

Three ways this run can succeed and mean nothing, and one test each.

**Different frames.** If an arm is pointed at a different partition or split, the two counts are of different rooms
and the difference between them is the rooms, not the training data.

**Different scales.** Every row is read at a fixed threshold and only the `official` arm carries a fitted score
calibration, worth -11.8% at threshold 0.3 and 8x at 0.9 on the same checkpoints. An arm that keeps it while the
other has none reports the calibration as the finding.

**The same model twice.** The arms differ in exactly one thing -- which partition trained them -- so if a path slips
and both jobs load one pair, the comparison is a model against itself and the answer is zero by construction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

import queue_person_free as chain  # noqa: E402


def flag(job, name: str) -> str:
    cli = list(job.cli)
    assert name in cli, f"{job.name} does not pass {name}: {cli}"
    return cli[cli.index(name) + 1]


def arms():
    return [job for job in chain.JOBS if job.name != "gate"]


class TestBothArmsAreCountedOnTheSameFrames:
    def test_every_job_counts_on_balanced_test(self):
        """`balanced` test is the only split both arms are held out from: its person-free frames are all 15-53's."""
        for job in arms():
            assert flag(job, "--frog-mode") == "balanced", job.name
            assert flag(job, "--split") == "test", job.name

    def test_no_job_leaves_the_partition_to_the_default(self):
        """The default is `transferred`, whose person-free frames include two recordings `balanced` trains on."""
        for job in arms():
            assert "--frog-mode" in job.cli, job.name


class TestBothArmsAreReadOnTheSameScale:
    def test_the_two_compared_arms_ignore_the_score_calibration(self):
        compared = [job for job in arms() if not job.name.endswith("calibrated")]

        assert len(compared) == 2
        for job in compared:
            assert "--no-calibration" in job.cli, job.name

    def test_the_as_published_row_keeps_its_calibration(self):
        """Dropping it everywhere would lose continuity with every person-free row measured before this run."""
        published = next(job for job in arms() if job.name.endswith("calibrated"))

        assert "--no-calibration" not in published.cli


class TestTheArmsAreActuallyDifferentModels:
    def test_the_two_arms_load_different_checkpoints(self):
        official, balanced = chain.OFFICIAL, chain.BALANCED

        assert set(official).isdisjoint(balanced)
        assert all("balanced" in str(path) for path in balanced)
        assert not any("balanced" in str(path) for path in official)

    def test_each_job_names_its_arm_in_the_file_it_writes(self):
        """Two JSON files that cannot be told apart later is the same failure as not running the comparison."""
        written = [str(output) for job in arms() for output in job.outputs]

        assert len(set(written)) == len(written)
        for job in arms():
            assert all(job.name in str(output) for output in job.outputs), job.name


class TestTheChainIsOrderedByWhatItCosts:
    def test_the_arm_with_a_warm_candidate_cache_runs_first(self):
        """A broken flag then shows up in minutes, before anything spends a quarter of an hour on a cache."""
        assert chain.JOBS[0].name == "person_free_balanced_arm"

    def test_the_calibrated_repeat_waits_for_the_arm_whose_cache_it_shares(self):
        published = next(job for job in arms() if job.name.endswith("calibrated"))

        assert published.needs == ("person_free_official_arm",)

    def test_the_gate_runs_the_suite_first(self):
        assert "pytest" in " ".join(str(c) for c in chain.GATE.cli)
