"""Step 3b must fine-tune stage 2 against the targets stage 2 was trained with.

`joint_frame_loss` built its per-beam targets from `make_targets`' bare defaults: the last-listed assignment for
a beam inside two people, a 0.6 m collect radius, and the **hard** label. The weights it loads were trained with
the nearest-person assignment and a soft Gaussian of 0.3 m, so the fine-tuning objective disagreed with the
pre-training objective in two ways at once -- and the "feedback contribution" the step exists to measure would
have carried that target change inside it. Dropout was a third, smaller divergence: the net came back at 0.0
where stage 2 trained at 0.1.

The convention is read from the **stage-2 checkpoint**, not from a flag: `--label-assign`, `--label-radius` and
`--label-sigma-m` are step-2 arguments, and the only correct value is the one those weights were trained under.
The fallbacks matter as much as the values, because a checkpoint written before a flag existed must come back as
the behaviour of its own time -- which for the soft target means **off**, not the parser's current 0.3.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from takheliped.detectors.takheliped import MemoryOutput  # noqa: E402

from train import beam_heatmap, compute_loss, make_targets  # noqa: E402
from train_takheliped import JointOutput, joint_frame_loss, stage2_training  # noqa: E402

N_BEAMS = 64
ANGLES = np.linspace(-np.pi / 2, np.pi / 2, N_BEAMS, dtype=np.float32)


def _scan():
    return np.full(N_BEAMS, 4.0, dtype=np.float32)


def _two_people_sharing_beams():
    """Two people close enough that beams fall inside both radii, which is the only case `nearest` changes."""
    return np.array([[4.0, 0.00], [4.0, 0.12]], dtype=np.float32)


def _joint(n_candidates=2, n_slots=3):
    """The smallest `JointOutput` `joint_frame_loss` reads: one stream, per-beam logits and votes, and a memory step."""
    torch.manual_seed(0)
    logit = torch.zeros(1, N_BEAMS, requires_grad=True)
    votes = torch.zeros(1, N_BEAMS, 2, requires_grad=True)
    cand_xy = torch.tensor([[[4.0, 0.0], [1.0, 1.0]]][:1], dtype=torch.float32)[:, :n_candidates]
    memory = MemoryOutput(
        candidate_logit=torch.zeros(1, n_candidates, requires_grad=True),
        candidate_xy=cand_xy,
        slot_logit=torch.zeros(1, n_slots, requires_grad=True),
        slot_xy=torch.zeros(1, n_slots, 2),
        coasting=torch.zeros(1, n_slots, dtype=torch.bool),
        slots=None,
        events=None,
    )
    return JointOutput(logit=logit, votes=votes, cand_xy=cand_xy,
                       cand_score=torch.zeros(1, n_candidates), cand_valid=torch.ones(1, n_candidates, dtype=torch.bool),
                       cand_features=torch.zeros(1, n_candidates, 1), prior=torch.zeros(1, 2, N_BEAMS // 8),
                       memory=memory)


def _loss(**convention):
    people = _two_people_sharing_beams()
    return joint_frame_loss(_joint(), _scan()[None], ANGLES, [people],
                            [np.array([[4.0, 0.0], [3.97, 0.48]], dtype=np.float32)], **convention)


BASE = dict(vote_collect_radius=0.6, nearest=True, sigma_m=0.3)


class TestTheConventionComesFromTheCheckpoint:
    def test_it_reads_what_the_stage_2_run_recorded(self):
        stored = {"label_radius": "0.45", "label_assign": "nearest", "label_sigma_m": "0.3", "dropout": "0.1"}

        assert stage2_training(stored) == dict(vote_collect_radius=0.45, nearest=True, sigma_m=0.3, dropout=0.1)

    def test_last_listed_is_read_as_not_nearest(self):
        assert stage2_training({"label_assign": "last"})["nearest"] is False

    def test_a_checkpoint_predating_the_flags_comes_back_as_its_own_behaviour(self):
        """The fallbacks are history, not today's defaults: before `--label-sigma-m` the target was hard."""
        old = stage2_training({})

        assert old["sigma_m"] == 0.0, "a pre-soft-target checkpoint must not acquire a Gaussian"
        assert old["nearest"] is False, "the assignment before the flag was last-listed"
        assert old["vote_collect_radius"] == 0.6
        assert old["dropout"] == 0.0

    def test_the_string_none_is_treated_as_unset(self):
        """Every arg is stored with `str(v)`, so an unset flag arrives as the text "None"."""
        assert stage2_training({"label_sigma_m": "None", "dropout": "None"})["sigma_m"] == 0.0


class TestTheLossCannotFallBackToTheOldDefaults:
    def test_the_convention_is_required(self):
        """A default here would let the divergence back in silently, which is how it survived to begin with."""
        with pytest.raises(TypeError):
            joint_frame_loss(_joint(), _scan()[None], ANGLES, [_two_people_sharing_beams()], [np.zeros((1, 2), np.float32)])


class TestEveryHalfOfTheConventionReachesTheTargets:
    def test_the_assignment_reaches_make_targets(self):
        nearest = _loss(**{**BASE, "nearest": True})
        last = _loss(**{**BASE, "nearest": False})

        assert not torch.isclose(nearest, last), "nearest= does not reach make_targets"

    def test_the_collect_radius_reaches_make_targets(self):
        narrow = _loss(**{**BASE, "vote_collect_radius": 0.3})
        wide = _loss(**{**BASE, "vote_collect_radius": 0.9})

        assert not torch.isclose(narrow, wide), "vote_collect_radius= does not reach make_targets"

    def test_the_soft_target_reaches_compute_loss(self):
        hard = _loss(**{**BASE, "sigma_m": 0.0})
        soft = _loss(**{**BASE, "sigma_m": 0.3})

        assert not torch.isclose(hard, soft), "sigma_m= does not reach compute_loss"

    def test_the_soft_target_is_built_the_way_stage_2_builds_it(self):
        """Same geometry, same radius, same `beam_heatmap` -- or the two objectives are still different.

        Compared as a *difference* between the soft and hard targets, because the stage-3 focal terms are not zero
        for an empty annotation either and would otherwise have to be re-implemented here to be subtracted. They are
        identical across the two calls, so they cancel and what is left is stage 2's half responding to `sigma_m`.
        """
        people = _two_people_sharing_beams()
        scan, gt = _scan(), {3: [tuple(p) for p in people]}
        labels, vote_targets = make_targets(scan, ANGLES, gt, vote_collect_radius=0.45, nearest=True)
        soft = beam_heatmap(scan, ANGLES, gt, sigma_m=0.25, vote_collect_radius=0.45)
        joint, xy = _joint(), np.zeros((0, 2), np.float32)
        hard_half, _, _ = compute_loss(joint.logit[0].reshape(-1, 1), joint.votes[0],
                                       torch.from_numpy(labels), torch.from_numpy(vote_targets))
        soft_half, _, _ = compute_loss(joint.logit[0].reshape(-1, 1), joint.votes[0],
                                       torch.from_numpy(labels), torch.from_numpy(vote_targets),
                                       soft_labels=torch.from_numpy(soft).reshape(-1))

        convention = dict(vote_collect_radius=0.45, nearest=True)
        measured = (joint_frame_loss(joint, scan[None], ANGLES, [people], [xy], sigma_m=0.25, **convention)
                    - joint_frame_loss(joint, scan[None], ANGLES, [people], [xy], sigma_m=0.0, **convention))

        assert torch.isclose(measured, soft_half - hard_half, atol=1e-6), (
            "the joint loss does not respond to sigma_m the way compute_loss does on beam_heatmap's target")
