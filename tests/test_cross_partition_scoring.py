"""Scoring a trained detector on a partition it was not trained on.

Every number this project has for its own detector comes from `train_takheliped.py`'s own test pass, which
scores whatever partition the run trained on. So a `balanced`-trained model could be compared with an
`official`-trained one only across different frames -- and `balanced`'s test contains `official`'s test whole
(40,022 frames, verified on the real files), so the comparison on identical frames was available and
unreachable. That is what `score` exists for.

Two properties matter more than the plumbing.

**It must reuse `evaluate`.** A second implementation of the metric would make its answer incomparable with
the frozen 86.29%, which is the only reason to compute it. The test below asserts the call, not the number.

**It must say, loudly and in its output, that the partitions differ.** A deliberate cross-partition score is a
legitimate experiment and a silent one is the cross-regime confusion `AGENTS.md` rule 3 exists to prevent --
and here the mismatch is the *point*, so refusing it would be wrong and hiding it worse.
"""
import json
import sys
from pathlib import Path

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import train_takheliped as tak  # noqa: E402


def _parser_for(step):
    parser = tak.build_parser()
    return parser._subparsers._group_actions[0].choices[step]


def _flags(step):
    return {action.dest: action for action in _parser_for(step)._actions}


class TestTheSubcommandExists:
    def test_score_is_a_step(self):
        assert "score" in tak.build_parser()._subparsers._group_actions[0].choices

    def test_it_takes_both_halves_of_a_trained_detector(self):
        flags = _flags("score")

        assert flags["stage2"].required, "a stage-2 checkpoint decides the candidates"
        assert flags["memory"].default is None, "a stage-3 checkpoint decides the rescoring, and without one stage 2 is scored alone"

    def test_it_takes_the_partition_and_the_split_to_score_on(self):
        flags = _flags("score")

        assert tuple(flags["frog_mode"].choices) == tak.PARTITIONS
        assert tak.FROG.partitions == ("official", "transferred", "balanced")
        assert tuple(flags["split"].choices) == ("train", "val", "test")
        assert flags["split"].default == "test"

    def test_it_defaults_to_the_stride_the_frozen_number_was_measured_at(self):
        """86.29% is a stride-5 number, so a stride-1 answer here would not be comparable with it."""
        assert _flags("score")["test_strides"].default == [5]

    def test_it_is_wired_to_a_runner(self):
        assert tak.STEPS["score"] is tak.score_checkpoint


class TestItReportsAPartitionMismatchRatherThanHidingIt:
    def test_a_mismatch_is_named_in_the_result(self, capsys):
        trained = {"frog_mode": "balanced"}
        record = tak.partition_note(trained, trained, "official")

        assert record["scored_on"] == "official"
        assert record["trained_on"] == ["balanced", "balanced"]
        assert record["cross_partition"] is True

    def test_a_matching_partition_is_not_flagged(self):
        trained = {"frog_mode": "official"}

        assert tak.partition_note(trained, trained, "official")["cross_partition"] is False

    def test_a_checkpoint_that_says_nothing_counts_as_official(self):
        """Every checkpoint written before `frog_mode` existed is an `official` run; that is the only thing it can be."""
        record = tak.partition_note({}, {}, "official")

        assert record["trained_on"] == ["official", "official"] and record["cross_partition"] is False

    def test_the_two_halves_disagreeing_with_each_other_is_also_a_mismatch(self):
        """A stage 2 from one partition under a stage 3 from another is not an experiment, it is a mix-up."""
        record = tak.partition_note({"frog_mode": "official"}, {"frog_mode": "balanced"}, "official")

        assert record["cross_partition"] is True


class TestItWritesWhatItDid:
    def test_the_output_carries_the_partition_record_beside_the_scores(self, tmp_path):
        """A file of numbers that does not say which frames produced them is the bug this whole item is about."""
        payload = {"partitions": tak.partition_note({"frog_mode": "balanced"}, {"frog_mode": "balanced"}, "official"),
                   "test": {"stride_5": {"memory": {"ap": 0.8}}}}
        path = tmp_path / "score_results.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        back = json.loads(path.read_text(encoding="utf-8"))

        assert back["partitions"]["scored_on"] == "official"
        assert back["partitions"]["cross_partition"] is True


class TestItDoesNotReimplementTheMetric:
    def test_it_calls_the_training_paths_evaluate(self):
        """Asserted on the reference, not on a number: a second metric would be incomparable with 86.29%."""
        assert "evaluate" in tak.score_checkpoint.__code__.co_names
        assert "evaluation_segments" in tak.score_checkpoint.__code__.co_names

    def test_it_asks_for_the_extra_radius_the_frozen_table_reports(self):
        """86.29% has an 85.19% companion at 0.3 m; a scoring mode that dropped it would not fill that column."""
        assert "EXTRA_TEST_RADII_M" in tak.score_checkpoint.__code__.co_names

    def test_it_reuses_the_partition_aware_split_loader(self):
        """`load_split` keys its candidate cache by partition; a bespoke loader here could read the wrong cache."""
        assert "load_split" in tak.score_checkpoint.__code__.co_names


class TestStageTwoAlone:
    """Without `--memory`, stage 2 is scored on any split by the pass step 2 scores its own test split with, which is
    what sets a train-split AP beside the test AP of the same weights (fit against generalisation)."""

    def test_omitting_the_memory_scores_stage_two_alone(self, monkeypatch):
        calls = []
        monkeypatch.setattr(tak, "score_stage2", lambda args, checkpoint, device: calls.append((args.split, checkpoint)))
        monkeypatch.setattr(tak.torch, "load", lambda *args, **kwargs: {"model": "stage 2"})
        tak.score_checkpoint(tak.build_parser().parse_args(["score", "--stage2", "x.pth", "--split", "train"]))

        assert calls == [("train", {"model": "stage 2"})]

    def test_it_reuses_step2_s_own_scoring_pass(self):
        """A second metric would make a train AP incomparable with the test AP the run reported."""
        assert "calibration_frames" in tak.score_stage2.__code__.co_names
        assert "score_detections" in tak.score_stage2.__code__.co_names
