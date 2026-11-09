"""DROW3 trainable through `train.py` on DR-SPAAM's own DROW recipe -- the known answer DROW training is checked against.

Nothing trained in this repository on DROW had been checked against a published number: every DROW baseline reproduced
here is a published weight file. DR-SPAAM's `cfgs/base_drow_drow_cfg.yaml` is the recipe behind its retrained `T=1` DROW
row, and each part of it is reachable here: the DROW network with one person logit, person-only targets, their
1.0 m / 0.5 m / 56-point cutout, BCE with no class weighting, an unweighted Euclidean vote loss on the positive beams,
Adam with AMSGrad, a per-iteration exponential decay, and the last weights. The published network and every default stay
exactly what they were, which is what the first class of tests pins.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import train as train_mod  # noqa: E402
from train import (LidarFrameDataset, _build_model, _default_args, _extract_input, compute_loss, lr_scheduler,  # noqa: E402
                   train_model)

from takheliped.detectors import DrowDetector  # noqa: E402
from takheliped.utils.drow_utils import laser_angles, laser_increment  # noqa: E402

N_BEAMS = 32
ODOM_DTYPE = np.dtype([("eq", np.uint32), ("t", np.float64), ("xya", np.float32, 3)])


class _TinyDrow:
    """The smallest object with the `DROW_Dataset` interface `LidarFrameDataset` and `evaluate_auc` read: one recording
    of a wall at 4 m, one person straight ahead and one wheelchair to the side in every frame."""

    def __init__(self, n=12, time_frame=1):
        angles = laser_angles(N_BEAMS)
        self.time_frame = time_frame
        self.scan_id = [np.arange(n, dtype=np.uint32)]
        self.scan_time = [np.arange(n, dtype=np.float64) * 0.0787]
        self.scans = [np.full((n, N_BEAMS), 4.0, np.float32)]
        odoms = np.zeros(n, dtype=ODOM_DTYPE)
        odoms["t"] = self.scan_time[0]
        self.odoms = [odoms]
        self.det_id = [np.arange(n, dtype=np.uint32)]
        self.det_wp = [[[(4.0, 0.0)] for _ in range(n)]]
        self.det_wc = [[[(4.0, float(angles[4]))] for _ in range(n)]]
        self.det_wa = [[[] for _ in range(n)]]
        self.idet2iscan = [{i: i for i in range(n)}]

    def get_scan(self, sequence_id, scan_id, time_window, dtime=1):
        idx = np.clip(scan_id - dtime * np.arange(time_window - 1, -1, -1), 0, scan_id)
        return self.scans[sequence_id][idx], self.odoms[sequence_id][idx]


def _cfg():
    return SimpleNamespace(name="drow", angles_fn=laser_angles, fov_min=0.0, fov_max=0.0, laser_inc=laser_increment)


def _published_recipe(tmp_path, **kw):
    opts = dict(detector="drow", dataset="drow", force_cpu=True, time_frame=1, person_only=True, cutout_name="drspaam",
                vote_radius=0.35, class_balance=False, vote_loss="euclidean", vote_weight=1.0, amsgrad=True,
                weight_decay=0.0, lr=1e-3, lr_schedule="exponential", min_lr=1e-6, epochs=2, batch_size=4, dropout=0.5,
                patience=0, auc_every=0, train_probe=False, out=tmp_path / "drow_t1.pth")
    opts.update(kw)
    return _default_args(**opts)


class TestThePublishedNetworkIsUntouched:
    def test_the_default_drow_network_has_four_class_logits_and_drow_s_window(self):
        net = DrowDetector(dropout=0.5, time_frame_size=5, verbose=False)
        assert net.conv4p.out_channels == 4
        assert (net.N_SAMP, net.WIN_SZ, net.THRESH_DIST) == (48, 1.66, 1.0)
        assert net.num_scans == 5

    def test_the_published_weights_still_fit_the_default_network(self):
        weights = torch.load(DrowDetector.DEFAULT_WEIGHTS, map_location="cpu")["model"]
        DrowDetector(dropout=0.5, time_frame_size=5, verbose=False).load_state_dict(weights)

    def test_the_loss_defaults_are_the_ones_every_number_on_file_was_trained_with(self):
        torch.manual_seed(0)
        logits, votes = torch.randn(10, 1), torch.randn(10, 2)
        labels, targets = torch.tensor([3, 0, 0, 3, 0, 0, 0, 0, 0, 0]), torch.randn(10, 2)
        default = compute_loss(logits, votes, labels, targets)
        explicit = compute_loss(logits, votes, labels, targets, class_balance=True, vote_loss="mse")
        assert default[0].item() == explicit[0].item()


class TestTheDrSpaamNetwork:
    def test_a_person_only_network_emits_one_logit(self):
        net = DrowDetector(dropout=0.5, time_frame_size=1, verbose=False, person_only=True, cutout="drspaam").eval()
        logits, votes = net(torch.randn(3, 1, 56))
        assert logits.shape == (3, 1) and votes.shape == (3, 2)

    def test_it_reads_dr_spaam_s_window(self):
        net = DrowDetector(dropout=0.5, time_frame_size=1, verbose=False, cutout="drspaam")
        assert (net.N_SAMP, net.WIN_SZ, net.THRESH_DIST) == (56, 1.0, 0.5)

    def test_the_last_pool_averages_every_remaining_position(self):
        """56 samples leave 7 positions after three halvings; DR-SPAAM pools all of them, a fixed kernel of 6 drops one."""
        torch.manual_seed(0)
        net = DrowDetector(dropout=0.5, time_frame_size=1, verbose=False, person_only=True, cutout="drspaam").eval()
        # DROW3 starts its logit head at zero, which would make every pooling read the same; wake it first.
        torch.nn.init.normal_(net.conv4p.weight)
        seen = {}
        net.bn4b.register_forward_hook(lambda module, inputs, output: seen.update(features=output))
        logits, _ = net(torch.randn(2, 1, 56))
        pooled = F.leaky_relu(seen["features"], 0.1).mean(-1, keepdim=True)
        assert torch.allclose(logits, net.conv4p(pooled)[:, :, 0], atol=1e-6)


class TestTheTargetsAndTheInput:
    def test_person_only_targets_leave_a_wheelchair_background(self):
        frames = LidarFrameDataset(_TinyDrow(), _cfg(), "cutout", 0.35, 56, person_only=True, win_sz=1.0, thresh_dist=0.5)
        _, labels, _ = frames[0]
        assert set(labels.tolist()) == {0, 3}

    def test_every_class_is_a_target_by_default(self):
        _, labels, _ = LidarFrameDataset(_TinyDrow(), _cfg(), "cutout", 0.35, 56)[0]
        assert 1 in labels.tolist()

    def test_training_reads_the_network_s_own_window_as_evaluation_does(self):
        """`_extract_input` always read the network's window and the training dataset never did, so a network with any
        window but DROW's own trained on one cutout and was scored on another."""
        ds = _TinyDrow()
        ds.scans[0][0, 10:14] = 2.5
        net = DrowDetector(dropout=0.5, time_frame_size=1, verbose=False, person_only=True, cutout="drspaam")
        x, _, _ = LidarFrameDataset(ds, _cfg(), "cutout", 0.35, net.N_SAMP, win_sz=net.WIN_SZ, thresh_dist=net.THRESH_DIST)[0]
        scans, odoms = ds.get_scan(0, 0, 1)
        expected = _extract_input(net, ds.scans[0][0], scans, odoms, laser_angles(N_BEAMS), _cfg(), "cpu")
        assert torch.equal(x, expected.cpu())


class TestTheLoss:
    def setup_method(self):
        torch.manual_seed(0)
        self.logits = torch.randn(10, 1, requires_grad=True)
        self.votes = torch.randn(10, 2, requires_grad=True)
        self.labels = torch.tensor([3, 3, 0, 0, 0, 0, 0, 0, 0, 0])
        self.targets = torch.randn(10, 2)

    def test_without_class_balance_the_classification_term_is_plain_bce(self):
        _, l_class, _ = compute_loss(self.logits, self.votes, self.labels, self.targets, class_balance=False)
        plain = F.binary_cross_entropy_with_logits(self.logits[:, 0], (self.labels > 0).float())
        assert l_class == pytest.approx(plain.item())

    def test_the_euclidean_vote_loss_is_the_mean_distance_over_the_positives(self):
        _, _, l_vote = compute_loss(self.logits, self.votes, self.labels, self.targets, vote_loss="euclidean")
        expected = (self.votes[:2] - self.targets[:2]).norm(dim=-1).mean()
        assert l_vote == pytest.approx(expected.item())

    def test_an_exact_vote_still_has_a_finite_gradient(self):
        """A norm's derivative is undefined at zero, and a perfect vote is exactly that point."""
        votes = self.targets.clone().requires_grad_(True)
        loss, _, _ = compute_loss(self.logits, votes, self.labels, self.targets, vote_loss="euclidean", vote_weight=1.0)
        loss.backward()
        assert torch.isfinite(votes.grad).all()


class TestTheSchedule:
    def _optimizer(self):
        return torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)

    def test_the_exponential_decay_reaches_the_floor_at_the_last_iteration(self):
        optimizer = self._optimizer()
        schedule, per_iteration = lr_scheduler(optimizer, "exponential", epochs=40, iterations=10, lr=1e-3, min_lr=1e-6)
        assert per_iteration
        rates = []
        for _ in range(400):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            schedule.step()
        assert rates[0] == pytest.approx(1e-3)
        assert rates[200] == pytest.approx(1e-3 * (1e-3) ** 0.5, rel=1e-6)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6, rel=1e-6)

    def test_the_older_schedules_still_step_once_an_epoch(self):
        schedule, per_iteration = lr_scheduler(self._optimizer(), "cosine", epochs=40, iterations=10, lr=1e-3, min_lr=1e-6)
        assert not per_iteration and isinstance(schedule, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_amsgrad_reaches_the_optimizer(self):
        assert train_mod._make_optimizer([torch.nn.Parameter(torch.zeros(1))], 1e-3, 0.0, False, amsgrad=True).defaults["amsgrad"]


class TestTheQueueAsksForThePublishedRecipe:
    def test_its_flags_parse_into_dr_spaam_s_configuration(self, monkeypatch):
        from queue_drow_validation import DRSPAAM_RECIPE
        seen = {}
        monkeypatch.setattr(train_mod, "train_model", lambda args: seen.update(vars(args)))
        monkeypatch.setattr(sys, "argv", ["train.py", *DRSPAAM_RECIPE, "--out", "x.pth"])
        train_mod.main()
        assert seen["detector"] == "drow" and seen["dataset"] == "drow" and seen["time_frame"] == 1
        assert seen["person_only"] and seen["cutout_name"] == "drspaam" and seen["vote_radius"] == 0.35
        assert not seen["class_balance"] and seen["vote_loss"] == "euclidean" and seen["vote_weight"] == 1.0
        assert seen["amsgrad"] and seen["weight_decay"] == 0.0 and seen["lr"] == 1e-3
        assert seen["lr_schedule"] == "exponential" and seen["min_lr"] == 1e-6 and seen["epochs"] == 40
        assert seen["batch_size"] == 8 and seen["dropout"] == 0.5 and seen["patience"] == 0


class TestTheWholeRecipeTrainsAndRebuilds:
    def test_two_epochs_of_the_recipe_save_a_checkpoint_that_rebuilds_into_the_same_network(self, monkeypatch, tmp_path):
        monkeypatch.setattr(train_mod, "_setup_datasets", lambda args: (_TinyDrow(), _TinyDrow(), _cfg()))
        args = _published_recipe(tmp_path)
        train_model(args)
        ckpt = torch.load(args.out, map_location="cpu")
        assert (ckpt["person_only"], ckpt["cutout_name"], ckpt["num_scans"]) == (True, "drspaam", 1)
        rebuilt = _build_model(_default_args(detector="drow", dataset="drow", time_frame=ckpt["num_scans"],
                                             person_only=ckpt["person_only"], cutout_name=ckpt["cutout_name"]))
        rebuilt.load_state_dict(ckpt["model"])
        assert ckpt["epoch"] == 2, "the last weights, not a val-selected best"
