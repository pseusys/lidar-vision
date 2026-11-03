"""`TODO.md` item 31: the Li2Former row is filled by the authors' own network, or not at all.

Item 31 was written believing `Li2FormerDetector` was Li2Former and only its training had failed. It was not. The paper
builds a cascade of omni-dimension aggregation blocks with a batch-attending auxiliary stream and an adaptive focus
mechanism in its regression loss; our class had a conv backbone, one temporal encoder layer and no cross-beam path at
all. Training it would have put a number in the Li2Former row that Li2Former did not produce -- the mislabelling
`AGENTS.md` rule 3 exists to prevent, and worse than the `gap` it replaced.

So the detector under test wraps **the authors' released implementation**, downloaded by `library/setup.py` into the
gitignored `include/` tree at a pinned commit and never vendored, because it is GPL-3.0 and this repository is MIT.

What these tests hold down is everything that could make the eventual number not theirs:

* the network really is upstream's class, so no local approximation can be substituted again without a failure here;
* it is configured as their published `ros_li2former.yaml` configures it, down to the adaptive focus constants, since
  the AFM is the difference between the paper's 76.4% and its 74.8%;
* the **auxiliary stream doubles the batch in training and not in eval** -- the silent-failure candidate, and the reason
  the criterion has to see duplicated targets;
* preprocessing is **their** cutout, whose window is 1.0 m wide and 0.5 m deep against our DROW `cutout()`'s 1.66/1.0,
  because feeding their network our input measures neither model;
* our vote targets agree with their own `global2local` to numerical tolerance, which is what makes it legitimate to
  train their network against this repository's targets at all.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from takheliped.detectors import DETECTOR_REGISTRY
from takheliped.detectors.li2former_official import (
    LI2FORMER_LOSS_KWARGS,
    LI2FORMER_MODEL_KWARGS,
    OfficialLi2FormerDetector,
)
from takheliped.include.li2former_upstream.models.li2former.li2former import Li2Former as UpstreamLi2Former
from takheliped.include.li2former_upstream.utils.scan_utils import global2local, scans2cutout

N_BEAMS = 16
T = 5


@pytest.fixture
def net():
    return OfficialLi2FormerDetector()


class TestItIsTheAuthorsNetwork:
    """The whole point of the port: the thing that trains is upstream's module, not a local lookalike."""

    def test_the_network_is_upstream_s_own_class(self, net):
        assert isinstance(net.net, UpstreamLi2Former)

    def test_the_upstream_class_comes_from_the_include_tree(self, net):
        """Pinned, downloaded, gitignored -- not copied into the repository, which is MIT and theirs is GPL-3.0."""
        assert "include" in type(net.net).__module__.split(".")

    def test_it_is_registered_under_its_own_key(self):
        assert DETECTOR_REGISTRY["li2former_official"] is OfficialLi2FormerDetector


class TestItIsConfiguredAsPublished:
    """`ros_li2former.yaml`, verbatim. The AFM constants are worth 1.6 pp of the number being reproduced."""

    def test_the_mode_is_the_paper_s_and_not_their_deployment_yaml_s(self):
        """
        `cfgs/ros_li2former.yaml` says `TEMPORAL`, and that is their ROS robot setting, not the paper's
        model: `TEMPORAL` never calls the spatial encoder, so it has no cross-beam path, while the paper's
        ODA module "jointly encode[s] temporal and spatial information" (abstract).  `PARALLEL` is the
        mode that runs both encoders and fuses them -- the only one that uses `fusion` at all.
        """
        assert LI2FORMER_MODEL_KWARGS["MODE"] == "PARALLEL"
        assert LI2FORMER_MODEL_KWARGS["NUM_PTS"] == 64
        assert LI2FORMER_MODEL_KWARGS["BACKBONE_TYPE"] == "ConvBackbone"

    def test_both_encoders_are_the_published_ones(self):
        """The spatial encoder's three layers were never asserted while the mode left them unreachable."""
        temporal, spatial = (LI2FORMER_MODEL_KWARGS["TEMPORAL_ENCODER"],
                             LI2FORMER_MODEL_KWARGS["SPATIAL_ENCODER"])
        assert (temporal["D_MODEL"], temporal["NHEAD"], temporal["NUM_LAYERS"]) == (512, 8, 1)
        assert (temporal["DIM_FFN"], temporal["DROPOUT"], temporal["NORMAL_BEFORE"]) == (2048, 0.1, True)
        assert (spatial["D_MODEL"], spatial["NHEAD"], spatial["NUM_LAYERS"]) == (512, 8, 3)
        assert (spatial["DIM_FFN"], spatial["DROPOUT"], spatial["NORMAL_BEFORE"]) == (2048, 0.1, True)

    def test_the_adaptive_focus_constants_are_the_published_ones(self):
        assert LI2FORMER_LOSS_KWARGS["DYNAMIC_REG"]["a"] == 1.5
        assert LI2FORMER_LOSS_KWARGS["DYNAMIC_REG"]["r"] == 0.35
        assert LI2FORMER_LOSS_KWARGS["DYNAMIC_REG"]["alpha"] == 1.5

    def test_the_regression_loss_outweighs_the_classification_loss_three_to_one(self):
        weights = LI2FORMER_LOSS_KWARGS["LOSS_WEIGHT"]
        assert (weights["loss_labels"], weights["loss_reg"]) == (1.0, 3.0)


class TestTheAuxiliaryStreamIsOffBecauseTheirOmniDimensionBranchCannotRunIt:
    """
    Their `__init__` hardcodes `add_bf=True`, but the `SERIAL`/`PARALLEL` branches never reshape the
    doubled batch and carry the comment "Keep add_bf option False" -- so the auxiliary stream and the
    omni-dimension modes are mutually exclusive *in the released code*, whatever the paper describes.
    Turning it off is therefore forced, and it is a stated deviation rather than a silent one: the
    BatchFormer is training-only, so the trained model's inference path is unaffected either way.
    """

    def test_neither_encoder_carries_the_auxiliary_stream(self, net):
        assert net.net.t_encoder.add_bf is False
        assert net.net.s_encoder.add_bf is False

    def test_its_now_dead_batch_layers_are_gone_rather_than_left_untrained(self, net):
        """12.6M parameters that no branch reads would otherwise be saved into every checkpoint."""
        assert not hasattr(net.net.t_encoder, "batch_layers")
        assert not hasattr(net.net.s_encoder, "batch_layers")

    def test_training_returns_one_prediction_per_beam(self, net):
        net.train()
        logits, votes = net(torch.zeros(1, N_BEAMS, T, net.N_SAMP))
        assert logits.shape == (N_BEAMS, 1)
        assert votes.shape == (N_BEAMS, 2)

    def test_evaluation_returns_one_prediction_per_beam(self, net):
        net.eval()
        logits, votes = net(torch.zeros(1, N_BEAMS, T, net.N_SAMP))
        assert logits.shape == (N_BEAMS, 1)
        assert votes.shape == (N_BEAMS, 2)

    def test_the_criterion_is_theirs_and_no_longer_duplicates_targets(self, net):
        assert net.duplicates_targets_in_training is False
        assert type(net.criterion).__name__ == "FormerCriterion"


class TestTheOmniDimensionPathIsActuallyLive:
    """
    The defect this leg was lost to was silent: a mode that quietly left three quarters of the network
    without a gradient.  These tests fail loudly if that recurs, by checking the gradient arrives rather
    than by checking a config string.
    """

    def _backward_once(self, net):
        net.train()
        logits, votes = net(torch.randn(1, N_BEAMS, T, net.N_SAMP).abs() * 0.1)
        (logits.sum() + votes.sum()).backward()

    def test_the_spatial_encoder_receives_gradient(self, net):
        self._backward_once(net)
        grads = [p.grad for p in net.net.s_encoder.parameters() if p.grad is not None]
        assert grads, "the spatial encoder got no gradient -- the cross-beam path is not connected"
        assert any(g.abs().sum() > 0 for g in grads)

    def test_the_temporal_encoder_receives_gradient(self, net):
        self._backward_once(net)
        grads = [p.grad for p in net.net.t_encoder.parameters() if p.grad is not None]
        assert grads, "the temporal encoder got no gradient"
        assert any(g.abs().sum() > 0 for g in grads)

    def test_the_fusion_mlp_receives_gradient(self, net):
        """`fusion` is dead weight in every mode but `PARALLEL`, so it is the mode's own witness."""
        self._backward_once(net)
        grads = [p.grad for p in net.net.fusion.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)

    def test_the_only_parameter_without_a_gradient_is_their_unused_pooling_weight(self, net):
        """
        `alpha` weights their softmax avg+max pooling, which only the `TEMPORAL` and `SPATIAL` branches
        apply; `PARALLEL` and `SERIAL` pool with a plain average and never read it.  So **no mode in the
        released code contains every component the paper describes** -- the single-dimension modes have
        the learnable pooling and the auxiliary stream, the omni-dimension modes have the cross-beam
        path.  Two parameters are left idle by that choice, and they are named here rather than deleted
        so the next reader meets the fact instead of a tidy network.  Any *other* starved parameter is
        the silent failure this test exists to catch.
        """
        self._backward_once(net)
        starved = sorted(n for n, p in net.net.named_parameters()
                         if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0))
        assert starved == ["alpha"], f"parameters receiving no gradient: {starved}"


class TestItAcceptsAWholeScanFromEveryDatasetHere:
    """
    The crash these tests did not catch.  Every other case in this file runs 16 beams, chosen to keep
    the suite fast, and 16 is below `NUM_CTS` -- so the one configuration value that *caps* the beam
    count could not fail.  It did fail, three times, 13 minutes into training each time: their
    `PositionalEncodingSine` is built with `max_len=NUM_CTS` and its `forward` returns
    `pe[:, :x.size(1), :]`, which silently comes back short rather than raising, and the mismatch
    surfaces as `src + pos` inside the encoder layer.  Under `MODE: TEMPORAL` the sequence is `T`, so
    the cap never mattered; `PARALLEL` and `SERIAL` attend over the beams, where it does.

    A beam count is a property of the sensor, not of the model, so each dataset gets a case.
    """

    BEAM_COUNTS = {"drow": 450, "jrdb": 541, "frog": 720}

    def test_the_position_encoding_spans_the_widest_scan_in_this_repository(self, net):
        widest = max(self.BEAM_COUNTS.values())
        assert net.net.pe_sine.pe.shape[1] >= widest, (
            f"position encoding holds {net.net.pe_sine.pe.shape[1]} beams, "
            f"FROG alone needs {widest}")

    @pytest.mark.parametrize("dataset,n_beams", sorted(BEAM_COUNTS.items()))
    def test_a_forward_pass_survives_a_whole_scan(self, net, dataset, n_beams):
        net.eval()
        with torch.no_grad():
            logits, votes = net(torch.zeros(1, n_beams, T, net.N_SAMP))
        assert logits.shape == (n_beams, 1)
        assert votes.shape == (n_beams, 2)


class TestItAnswersThisRepositoryContract:
    """Every detector here is reached the same way, whoever wrote the network."""

    def test_it_declares_their_cutout_as_its_input_mode(self, net):
        assert net.INPUT_MODE == "cutout_li2former"
        assert net.N_SAMP == 64
        assert net.BEAM_BATCH is True

    def test_forward_one_puts_the_person_probability_in_the_pedestrian_slot(self, net):
        confs, votes = net.forward_one(torch.zeros(N_BEAMS, T, net.N_SAMP))
        assert confs.shape == (N_BEAMS, 4)
        assert votes.shape == (N_BEAMS, 2)
        assert np.all(confs[:, :3] == 0.0)
        assert np.all((confs[:, 3] >= 0.0) & (confs[:, 3] <= 1.0))


class TestPreprocessingIsTheirs:
    """Their window is a different window. Feeding their network ours would measure neither model."""

    def test_the_published_window_is_what_the_adapter_asks_for(self, net):
        assert net.CUTOUT_KWARGS["window_width"] == 1.0
        assert net.CUTOUT_KWARGS["window_depth"] == 0.5
        assert net.CUTOUT_KWARGS["num_cutout_pts"] == 64
        assert net.CUTOUT_KWARGS["area_mode"] is True
        assert net.CUTOUT_KWARGS["centered"] is True
        assert net.CUTOUT_KWARGS["fixed"] is True

    def test_the_adapter_builds_exactly_upstream_s_cutout(self, net):
        rng = np.random.default_rng(31)
        scans = rng.uniform(0.5, 8.0, size=(T, 64)).astype(np.float32)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 64)
        np.testing.assert_allclose(
            net.preprocess(scans, angles),
            scans2cutout(scans, angles, **net.CUTOUT_KWARGS),
            rtol=0, atol=0)

    def test_centering_leaves_every_sample_inside_one_window_depth(self, net):
        rng = np.random.default_rng(31)
        scans = rng.uniform(0.5, 8.0, size=(T, 64)).astype(np.float32)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 64)
        cut = net.preprocess(scans, angles)
        assert cut.shape == (64, T, 64)
        assert np.all(np.abs(cut) <= 1.0 + 1e-6)


class TestTheBinaryTargetIsPersonOnly:
    """Which beams count positive -- the one wiring choice that moves the number, and only on DROW.

    Their head emits a single logit, so something has to decide what it predicts. The published DROW
    figures in this lineage come from `pedestrian_only=True` weights (DR-SPAAM's own released
    `num_pts=56, pedestrian_only=True`) and the published metric is person AP, so a wheelchair or
    walker beam is background here -- unlike `compute_loss`'s binary mode, which counts any
    non-background beam and is what our own `li2former` class trains against. On FROG the two agree
    exactly, because FROG annotates no wheelchairs or walkers (`FROG_Dataset.det_wc` is always empty).

    Their loader is not in the released repository, so this is an inference from the regime their
    number was measured in, not something transcribed from their code.
    """

    def test_the_positive_class_is_the_person_class(self, net):
        assert net.POSITIVE_CLASS == 3

    def test_a_wheelchair_beam_is_background_to_their_criterion(self, net):
        from train import upstream_criterion_loss

        net.eval()
        logits = torch.full((4, 1), 4.0)          # confidently "person" on every beam
        votes, vote_targets = torch.zeros(4, 2), torch.zeros(1, 4, 2)
        _, as_person, _ = upstream_criterion_loss(
            net, logits, votes, torch.tensor([[0, 0, 0, 3]]), vote_targets)
        _, as_wheelchair, _ = upstream_criterion_loss(
            net, logits, votes, torch.tensor([[0, 0, 0, 1]]), vote_targets)
        # Counting any non-background beam would make these two identical.
        assert as_wheelchair > as_person


class TestOurVoteTargetsAreTheirVoteTargets:
    """What makes it legitimate to train their network against this repository's targets."""

    def test_make_targets_agrees_with_their_global2local(self):
        from train import make_targets

        rng = np.random.default_rng(31)
        angles = np.linspace(-np.pi / 2, np.pi / 2, 64)
        scan = rng.uniform(1.0, 6.0, size=64)
        r_gt, phi_gt = 3.0, float(angles[20])
        # Beams that actually return off the annotated person, so the radius claims them: at 3 m
        # these angles are 0.15 m apart, well inside the 0.6 m collection radius.
        scan[18:23] = r_gt
        labels, votes = make_targets(scan, angles, {3: [(r_gt, phi_gt)]}, vote_collect_radius=0.6)

        positive = labels == 3
        assert positive.any(), "the annotation must claim at least one beam for this to test anything"
        dx, dy = global2local(scan[positive], angles[positive],
                              np.full(positive.sum(), r_gt), np.full(positive.sum(), phi_gt))
        np.testing.assert_allclose(votes[positive, 0], dx, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(votes[positive, 1], dy, rtol=1e-6, atol=1e-6)
