"""Training this project's own detector on a FROG partition other than `official`.

`train.py` has taken `--frog-mode` since the three partitions were defined, but `train_takheliped.py` never did:
`load_frames` asked for `mode="official"` in its body, so every stage-2 and stage-3 run this project has ever
done is an `official` run whatever anyone intended. That is harmless while `official` is the only partition
being used and a silent wrong answer the moment it is not -- and `balanced`'s val contains recordings
`official` trains on, so the failure mode is scoring a model on its own training data.

Two things are pinned here. The flag has to exist and reach the loader, and **the checkpoint has to record
which partition it was trained on**, because `evaluate.py` resolves the evaluation partition from the
checkpoints themselves (`_resolve_frog_mode`) and refuses to put two partitions in one pass. A checkpoint that
does not say is treated as `official`, so a `balanced` checkpoint that forgot to say would be scored on
`official` without a word.
"""
import sys
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import train_takheliped as tak  # noqa: E402

MODES = ("official", "transferred", "balanced")


def _parser_for(step):
    """The subparser for one step, reached the way `main` reaches it."""
    parser = tak.build_parser()
    return parser._subparsers._group_actions[0].choices[step]


def _flags(step):
    return {action.dest: action for action in _parser_for(step)._actions}


class TestTheFlagExists:
    @pytest.mark.parametrize("step", ("step2", "step3a", "step3b", "step4"))
    def test_every_step_that_touches_frames_takes_the_partition(self, step):
        assert "frog_mode" in _flags(step), f"{step} cannot be pointed at another partition"

    @pytest.mark.parametrize("step", ("step2", "step3a", "step3b", "step4"))
    def test_it_offers_the_three_partitions_and_defaults_to_official(self, step):
        flag = _flags(step)["frog_mode"]

        assert tuple(flag.choices) == MODES
        assert flag.default == "official", "the default must reproduce every number measured so far"


class TestTheFlagReachesTheLoader:
    def test_load_frames_asks_the_dataset_for_the_mode_it_was_given(self, monkeypatch):
        asked = []

        def fake_dataset(**kwargs):
            asked.append(kwargs["mode"])
            raise _Stop()

        monkeypatch.setattr(tak, "FROG_Dataset", fake_dataset)
        for mode in MODES:
            with pytest.raises(_Stop):
                tak.load_frames("train", mode)

        assert asked == list(MODES)

    def test_load_frames_still_defaults_to_official(self, monkeypatch):
        asked = []
        monkeypatch.setattr(tak, "FROG_Dataset", lambda **kw: asked.append(kw["mode"]) or _raise(_Stop()))
        with pytest.raises(_Stop):
            tak.load_frames("train")

        assert asked == ["official"]

    def test_the_candidate_cache_is_keyed_by_partition(self, tmp_path):
        """Otherwise a `balanced` run reads `official`'s cached candidates and trains on the wrong frames."""
        # `stage2=None` is the LFE-Peaks source, which needs no checkpoint on disk to key against.
        paths = {mode: tak.candidate_cache_path("train", 0, None, mode) for mode in MODES}

        assert len({str(p) for p in paths.values()}) == len(MODES), paths


class TestTheCheckpointSaysWhichPartition:
    @pytest.mark.parametrize("step", ("step2", "step3a"))
    def test_the_payload_carries_frog_mode_at_the_top_level(self, step):
        """Where `evaluate.py._resolve_frog_mode` looks for it -- not inside `args`, which it does not read."""
        source = (UTILS / "train_takheliped.py").read_text(encoding="utf-8")

        assert "frog_mode=args.frog_mode" in source or 'frog_mode": args.frog_mode' in source, (
            "no checkpoint write passes the partition, so evaluate.py will read every checkpoint as official")


class TestResumeRefusesToChangePartition:
    """Every queued job passes `--resume`, so this is the realistic way two partitions get mixed."""

    def _resume_file(self, tmp_path, mode):
        import torch
        path = tmp_path / "step2_resume.pth"
        torch.save({"frog_mode": mode, "epoch": 1}, path)
        return path

    def test_it_refuses_a_file_written_under_another_partition(self, tmp_path):
        path = self._resume_file(tmp_path, "official")

        with pytest.raises(SystemExit) as refused:
            tak.load_resume(path, True, "balanced")

        assert "official" in str(refused.value) and "balanced" in str(refused.value)

    def test_it_resumes_its_own_partition(self, tmp_path):
        path = self._resume_file(tmp_path, "balanced")

        assert tak.load_resume(path, True, "balanced")["epoch"] == 1

    def test_a_file_from_before_the_field_existed_is_allowed(self, tmp_path):
        """It can only be an `official` run, and refusing it would strand every resume file on disk."""
        import torch
        path = tmp_path / "step2_resume.pth"
        torch.save({"epoch": 2}, path)

        assert tak.load_resume(path, True, "official")["epoch"] == 2


class _Stop(Exception):
    """Raised by the fakes above once they have recorded the call, so no dataset is ever loaded."""


def _raise(error):
    raise error
