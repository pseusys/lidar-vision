"""The averaging script scores every candidate window on val, and it has to score them on the run's own data.

`average_checkpoints.py` took the run *directory* and nothing else, so it inherited none of the flags that
produced the run: it hardcoded FROG `official`'s 720-beam grid and loaded FROG `official` val whatever the
checkpoints were trained on. It cannot fail loudly, because an aggregator-free calibration network is fully
convolutional over beams -- a 360-beam state dict loads into a 720-angle network without a shape error, and a
DROW network loads into a FROG one. Three runs were ranked on the wrong data before anyone noticed, and the only
tell was the printed val figure sitting 12 pp below its peers.

So these tests pin the flags, not the arithmetic: that the script can be told which dataset, partition and beam
stride a run used, that **one** spec reaches both the network and the frames, that FROG `official` stays the
default so the correctly-ranked runs reproduce, and that every queue hands the script the same flags its own
stage 2 ran with -- which is the one place the mismatch can come back.
"""
import sys
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import average_checkpoints as avg  # noqa: E402
from train_takheliped import DROW, FROG, dataset_spec  # noqa: E402

RUN = "../checkpoints_three_horizon/some_run"

#: Every queue that averages. Eight of them train FROG `official` at full resolution and so were ranked
#: correctly by the old defaults; three were not, and this list is what stops a ninth joining them quietly.
AVERAGING_QUEUES = ["queue_architecture_sweep", "queue_balanced", "queue_drow", "queue_drow_handling",
                    "queue_freeze", "queue_ladder_f1w", "queue_prefreeze", "queue_soft_label_confirm",
                    "queue_step1", "queue_step1_defences", "queue_vote_frame_and_block"]


def parse(*argv):
    return avg.build_parser().parse_args([RUN, *argv])


class TestTheRunsOwnFlagsCanBeGiven:
    """The three flags that change what val means, named exactly as the trainer names them."""

    @pytest.mark.parametrize("flag", ["--dataset", "--frog-mode", "--beam-stride"])
    def test_the_flag_exists(self, flag):
        assert flag in {s for a in avg.build_parser()._actions for s in a.option_strings}

    def test_the_dataset_is_resolved_by_the_trainers_own_resolver(self):
        """Identity, not equality: a second copy of this resolution is how the two drift apart again."""
        assert avg.dataset_spec is dataset_spec

    def test_a_drow_run_is_ranked_on_drow(self):
        spec = avg.dataset_spec(parse("--dataset", "drow"))
        assert spec.name == "drow" and spec.beams == DROW.beams and spec.max_range_m == DROW.max_range_m

    def test_a_decimated_run_is_ranked_at_its_own_spacing(self):
        spec = avg.dataset_spec(parse("--beam-stride", "2"))
        assert spec.beams == FROG.source_beams // 2 and spec.spacing_deg == 0.5

    def test_the_partition_reaches_the_loader_rather_than_being_assumed(self):
        assert parse("--frog-mode", "balanced").frog_mode == "balanced"


class TestFrogOfficialStaysTheDefault:
    """Eleven runs were ranked correctly under the old script. Their windows must not move."""

    def test_the_default_spec_is_frog_official_at_full_resolution(self):
        spec = avg.dataset_spec(parse())
        assert spec.name == "frog" and spec.decimate == 1 and spec.beams == 720

    def test_the_default_partition_is_official(self):
        assert parse().frog_mode == "official"


class TestOneSpecReachesBothTheNetworkAndTheFrames:
    """The defect was two sources of truth: a hardcoded 720-beam grid beside a separately loaded val split.

    A network built on one grid and scored on frames from another is the failure, so the angles the script
    hands the network must be the same spec's angles as the frames it loads.
    """

    @pytest.mark.parametrize("argv", [(), ("--dataset", "drow"), ("--beam-stride", "2"), ("--dataset", "drow", "--beam-stride", "2")])
    def test_the_angles_come_from_the_spec_and_match_its_beam_count(self, argv):
        spec = avg.dataset_spec(parse(*argv))
        assert len(spec.angles()) == spec.beams

    def test_no_beam_count_is_hardcoded_in_the_module(self):
        """`frog_laser_angles(720)` was the whole bug; a literal grid here cannot follow a flag."""
        source = Path(avg.__file__).read_text(encoding="utf-8")
        assert "frog_laser_angles(720)" not in source


class TestRankingAlonePreservesWeightsATrainedStageThreeDependsOn:
    """Re-ranking an old run must not silently replace the checkpoint its stage 3 was trained on.

    `drow_stage2_s0`'s averaged file is named inside the DROW headline's stage-3 checkpoint, so overwriting it
    would leave a trained model and the weights it claims to come from disagreeing.
    """

    def test_rank_only_exists_so_an_old_run_can_be_re_ranked_without_being_rewritten(self):
        assert parse("--rank-only").rank_only is True

    def test_writing_is_the_default_because_that_is_what_the_recipe_does(self):
        assert parse().rank_only is False

    def test_a_ranking_only_pass_writes_its_own_file(self):
        """Two reasons, both found by `--list` on 2026-11-06. The runner skips a job whose declared output already
        exists, and every wrongly-ranked run already has an `averaging.json`; and that file *is* the evidence that
        diagnosed the defect -- the 64.50% and the monotonic fall across windows -- so overwriting it would
        destroy what the fix is being checked against."""
        run = Path("run")
        assert avg.ranking_path(run, rank_only=True) != avg.ranking_path(run, rank_only=False)
        assert avg.ranking_path(run, rank_only=False).name == "averaging.json"

    def test_the_recipes_own_pass_still_writes_the_file_every_earlier_run_has(self):
        assert avg.ranking_path(Path("run"), rank_only=False) == Path("run") / "averaging.json"


class TestEveryQueueHandsItTheFlagsItsStageTwoRan:
    """The one place the mismatch can come back: a queue that trains with a flag and averages without it."""

    @pytest.mark.parametrize("module", AVERAGING_QUEUES)
    def test_the_averaging_job_carries_the_same_dataset_flags_as_the_stage_two_it_averages(self, module):
        import importlib

        q = importlib.import_module(module)
        averaging = [j for j in q.JOBS if j.cli and j.cli[0] == "average_checkpoints.py"]
        assert averaging, f"{module} runs no averaging job"
        for job in averaging:
            trainers = [j for j in q.JOBS if "step2" in j.cli and j.name in job.needs]
            assert trainers, f"{job.name} averages a stage 2 it does not declare in needs"
            for stage2 in trainers:
                for flag in ("--dataset", "--beam-stride", "--frog-mode"):
                    if flag in stage2.cli:
                        value = stage2.cli[stage2.cli.index(flag) + 1]
                        assert flag in job.cli and job.cli[job.cli.index(flag) + 1] == value, \
                            f"{job.name} does not pass {flag} {value} from {stage2.name}"
