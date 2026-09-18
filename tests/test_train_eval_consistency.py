"""
The reported number must describe the saved weights, evaluated the way they
were trained -- TODO.md A15, A16, A17.

Three bugs these lock down, all measured on the real dtime=10 run:

A15  `evaluate_auc()`'s `dtime` defaults to 1 and `train_model()` never passed
     it, so a model trained on 0.38 s windows was scored on 0.038 s windows.
     Cost: 76.3% vs 78.8% wp-AUC on held-out val.
A16  the final AUC was computed *before* `load_checkpoint(best_path)`, so the
     printed number came from the discarded last epoch while the file on disk
     held the best one. The two were never the same network.
A17  early stopping ran on `val_loss`, whose scale swings 3x with batch
     composition; a noise dip at epoch 3 killed a 30-epoch run at epoch 8.
     Cost: 78.8% (epoch 3) vs 79.7% (epoch 14).
"""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import train as train_mod  # noqa: E402
from train import _default_args, save_checkpoint, train_model  # noqa: E402

from follow_the_drow.datasets.frog_dataset import frog_laser_angles  # noqa: E402


N_BEAMS = 64
N_FRAMES = 24
ODOM_DTYPE = np.dtype([("eq", np.uint32), ("t", np.float64), ("xya", np.float32, 3)])


class _TinyDataset:
    """The smallest object satisfying the DROW_Dataset/FROG_Dataset interface
    that LidarFrameDataset and evaluate_auc actually use."""

    def __init__(self, n_seq=2, n=N_FRAMES, seed=0):
        rng = np.random.RandomState(seed)
        self.time_frame = 5
        mk = lambda vals: np.array([None] * n_seq, dtype=object)  # noqa: E731
        self.scan_id, self.scan_time = mk(None), mk(None)
        self.scans, self.odoms = mk(None), mk(None)
        self.det_id, self.det_wc, self.det_wa, self.det_wp = mk(None), mk(None), mk(None), mk(None)
        for s in range(n_seq):
            self.scan_id[s] = np.arange(n, dtype=np.uint32)
            self.scan_time[s] = np.arange(n, dtype=np.float64) * 0.038
            self.scans[s] = (rng.rand(n, N_BEAMS).astype(np.float32) * 2.0 + 2.0)
            od = np.zeros(n, dtype=ODOM_DTYPE)
            od["t"] = self.scan_time[s]
            self.odoms[s] = od
            self.det_id[s] = np.arange(n, dtype=np.uint32)
            self.det_wc[s] = np.array([[] for _ in range(n)], dtype=object)
            self.det_wa[s] = np.array([[] for _ in range(n)], dtype=object)
            # one person straight ahead, at the range the middle beam sees
            self.det_wp[s] = np.array(
                [[(float(self.scans[s][i, N_BEAMS // 2]), 0.0)] for i in range(n)],
                dtype=object)
        self.idet2iscan = [{i: int(self.det_id[s][i]) for i in range(n)}
                           for s in range(n_seq)]

    def get_scan(self, sequence_id, scan_id, time_window, dtime=1):
        idx = np.clip(scan_id - dtime * np.arange(time_window - 1, -1, -1), 0, scan_id)
        return self.scans[sequence_id][idx], self.odoms[sequence_id][idx]


def _tiny_cfg():
    return SimpleNamespace(
        name="frog", angles_fn=frog_laser_angles,
        fov_min=-np.pi / 2, fov_max=np.pi / 2, laser_inc=np.pi / 720,
    )


@pytest.fixture
def tiny_setup(monkeypatch):
    """Point train_model() at an in-memory dataset instead of the real files."""
    train_ds, val_ds = _TinyDataset(seed=0), _TinyDataset(seed=1)
    monkeypatch.setattr(train_mod, "_setup_datasets",
                        lambda args: (train_ds, val_ds, _tiny_cfg()))
    return train_ds, val_ds


def _fingerprint(net) -> float:
    return float(sum(p.detach().float().sum().item() for p in net.parameters()))


def _args(tmp_path, **kw):
    opts = {
        "detector": "spacetime_cnn", "dataset": "frog", "force_cpu": True,
        "channels": 24, "n_spatial_stages": 1, "batch_size": 4, "epochs": 2,
        "patience": 0, "lr_schedule": "none", "auc_every": 1, "subsample": 1.0,
        "out": tmp_path / "w.pth"}
    opts.update(kw)
    return _default_args(**opts)


# ---------------------------------------------------------------------------
# A15 -- dtime must reach the evaluator
# ---------------------------------------------------------------------------

class TestDtimeReachesTheEvaluator:
    @pytest.mark.parametrize("dtime", [1, 7, 10])
    def test_train_model_evaluates_at_the_training_dtime(self, tiny_setup, tmp_path,
                                                         monkeypatch, dtime):
        seen = []
        real = train_mod.evaluate_auc

        def spy(net, dataset, cfg, **kw):
            seen.append(kw.get("dtime"))
            return real(net, dataset, cfg, **kw)

        monkeypatch.setattr(train_mod, "evaluate_auc", spy)
        train_model(_args(tmp_path, dtime=dtime, auc_every=1))
        assert seen, "evaluate_auc was never called"
        assert set(seen) == {dtime}, f"evaluate_auc saw dtime={seen}, expected {dtime}"

    def test_every_evaluate_auc_call_site_passes_dtime(self):
        """A static guard on the exact bug: three call sites, none of which
        passed `dtime`, silently taking the parameter's default of 1."""
        offenders = []
        for path in (UTILS / "train.py", UTILS / "train_all.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "evaluate_auc"
                        and not any(k.arg == "dtime" for k in node.keywords)):
                    offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, f"evaluate_auc called without dtime at {offenders}"

    def test_checkpoint_roundtrips_dtime_and_time_frame(self, tmp_path):
        from follow_the_drow.detectors.full_scan import SpaceTimeCNNDetector
        net = SpaceTimeCNNDetector(n_time=5, channels=24, n_spatial_stages=1)
        opt = torch.optim.Adam(net.parameters())
        p = tmp_path / "c.pth"
        save_checkpoint(p, net, opt, epoch=3, detector_name="spacetime_cnn",
                        dataset_name="frog", dtime=10)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        assert ckpt["dtime"] == 10
        assert ckpt["n_time"] == 5


# ---------------------------------------------------------------------------
# A16 -- the reported number must describe the saved weights
# ---------------------------------------------------------------------------

class TestReportedAucMatchesSavedWeights:
    def test_final_auc_is_computed_on_the_checkpoint_that_gets_saved(
            self, tiny_setup, tmp_path, monkeypatch):
        """Scripted so the peak is in the middle: if the reported number came
        from the last epoch's (discarded) weights it cannot match the file."""
        scripted = iter([0.50, 0.60, 0.72, 0.65, 0.61, 0.60, 0.59])
        calls = []
        by_weights = {}          # AUC is a function of the weights, not of call order

        def spy(net, dataset, cfg, **kw):
            fp = round(_fingerprint(net), 3)
            if fp not in by_weights:
                by_weights[fp] = next(scripted, 0.59)
            v = by_weights[fp]
            calls.append((fp, v))
            return {"agnostic": v, "wc": float("nan"), "wa": float("nan"), "wp": v}

        monkeypatch.setattr(train_mod, "evaluate_auc", spy)
        args = _args(tmp_path, dtime=1, epochs=10, patience=3, auc_every=1)
        history = train_model(args)

        saved = torch.load(args.out, map_location="cpu", weights_only=False)
        from follow_the_drow.detectors.full_scan import SpaceTimeCNNDetector
        net = SpaceTimeCNNDetector(n_time=5, channels=24, n_spatial_stages=1)
        net.load_state_dict(saved["model"])

        reported = history["val_auc_wp"][-1]
        assert reported == pytest.approx(0.72), (
            f"reported {reported}, expected the saved checkpoint's own 0.72")
        matching = [auc for fp, auc in calls
                    if abs(fp - round(_fingerprint(net), 3)) < 1e-3]
        assert matching, (
            "the reported AUC was computed on weights that were then discarded")
        assert reported == pytest.approx(matching[-1])


# ---------------------------------------------------------------------------
# A17 -- select and stop on val wp-AUC
# ---------------------------------------------------------------------------

class TestEarlyStoppingOnWpAuc:
    def _scripted(self, monkeypatch, aucs):
        """Gate-only: the memorization probe is off, so one scripted value is
        consumed per epoch rather than two. TestMemorizationProbe covers the
        probe itself."""
        it = iter(aucs)

        def fake(net, dataset, cfg, **kw):
            v = next(it, aucs[-1])
            return {"agnostic": v, "wc": float("nan"), "wa": float("nan"), "wp": v}

        monkeypatch.setattr(train_mod, "evaluate_auc", fake)

    def test_best_checkpoint_is_the_highest_auc_epoch_not_the_lowest_loss(
            self, tiny_setup, tmp_path, monkeypatch):
        # AUC peaks at epoch 3, then declines for `patience` epochs.
        self._scripted(monkeypatch, [0.50, 0.60, 0.70, 0.65, 0.61, 0.60, 0.59])
        args = _args(tmp_path, dtime=1, epochs=10, patience=3, auc_every=1,
                     train_probe=False)
        history = train_model(args)
        assert max(history["val_auc_wp"]) == pytest.approx(0.70)
        saved = torch.load(args.out, map_location="cpu", weights_only=False)
        assert saved["epoch"] == 3, (
            f"saved epoch {saved['epoch']}, expected the peak-AUC epoch 3")

    def test_patience_counts_epochs_without_auc_improvement(
            self, tiny_setup, tmp_path, monkeypatch):
        self._scripted(monkeypatch, [0.50, 0.60, 0.70, 0.65, 0.61, 0.60, 0.59, 0.58])
        args = _args(tmp_path, dtime=1, epochs=10, patience=3, auc_every=1,
                     train_probe=False)
        history = train_model(args)
        assert history["stopped_epoch"] == 6, (
            f"stopped at {history['stopped_epoch']}, expected 3 + patience 3")

    def test_a_run_that_keeps_improving_is_not_stopped(
            self, tiny_setup, tmp_path, monkeypatch):
        self._scripted(monkeypatch, [0.50, 0.55, 0.60, 0.65, 0.70])
        args = _args(tmp_path, dtime=1, epochs=5, patience=3, auc_every=1,
                     train_probe=False)
        history = train_model(args)
        assert history["stopped_epoch"] == 5

    def test_val_loss_is_still_recorded(self, tiny_setup, tmp_path, monkeypatch):
        """Dropping val_loss as the *gate* must not drop it as a diagnostic."""
        self._scripted(monkeypatch, [0.50, 0.60])
        history = train_model(_args(tmp_path, dtime=1, epochs=2, patience=0,
                                    auc_every=1, train_probe=False))
        assert len(history["val_loss"]) == 2
        assert all(np.isfinite(history["val_loss"]))


# ---------------------------------------------------------------------------
# Crash-recovery: the best checkpoint must survive a crash and a resume
# ---------------------------------------------------------------------------

class TestCheckpointDurability:
    def test_checkpoint_write_is_atomic(self, tmp_path, monkeypatch):
        """A crash mid-write must leave the previous best intact, not a
        truncated file. torch.save() straight to the path cannot promise that."""
        from follow_the_drow.detectors.full_scan import SpaceTimeCNNDetector
        net = SpaceTimeCNNDetector(n_time=5, channels=24, n_spatial_stages=1)
        opt = torch.optim.Adam(net.parameters())
        p = tmp_path / "c.pth"
        save_checkpoint(p, net, opt, epoch=1, detector_name="spacetime_cnn",
                        dataset_name="frog", dtime=10, val_wp_auc=0.70)
        good = p.read_bytes()

        real_save = torch.save

        def exploding_save(obj, f, *a, **kw):
            real_save(obj, f, *a, **kw)
            raise RuntimeError("power cut mid-write")

        monkeypatch.setattr(torch, "save", exploding_save)
        with pytest.raises(RuntimeError):
            save_checkpoint(p, net, opt, epoch=2, detector_name="spacetime_cnn",
                            dataset_name="frog", dtime=10, val_wp_auc=0.80)
        assert p.read_bytes() == good, "the previous best was clobbered by a failed write"

    def test_checkpoint_stores_the_auc_it_was_selected_on(self, tmp_path):
        from follow_the_drow.detectors.full_scan import SpaceTimeCNNDetector
        net = SpaceTimeCNNDetector(n_time=5, channels=24, n_spatial_stages=1)
        opt = torch.optim.Adam(net.parameters())
        p = tmp_path / "c.pth"
        save_checkpoint(p, net, opt, epoch=7, detector_name="spacetime_cnn",
                        dataset_name="frog", dtime=10, val_wp_auc=0.812)
        assert torch.load(p, map_location="cpu", weights_only=False)["val_wp_auc"] == \
            pytest.approx(0.812)

    def test_resume_does_not_overwrite_a_better_checkpoint(
            self, tiny_setup, tmp_path, monkeypatch):
        """Resuming used to reset best_val_auc to -inf, so the first epoch after
        a resume clobbered .best.pth even when it scored worse."""
        scripted = iter([0.90])           # the run we are "resuming from" hit 0.90
        by_weights = {}

        def spy(net, dataset, cfg, **kw):
            fp = round(_fingerprint(net), 3)
            if fp not in by_weights:
                by_weights[fp] = next(scripted, 0.40)   # every later epoch is worse
            v = by_weights[fp]
            return {"agnostic": v, "wc": float("nan"), "wa": float("nan"), "wp": v}

        monkeypatch.setattr(train_mod, "evaluate_auc", spy)
        args = _args(tmp_path, dtime=1, epochs=1, patience=2, auc_every=1)
        train_model(args)
        best = args.out.with_suffix(".best.pth")
        assert torch.load(best, map_location="cpu", weights_only=False)["val_wp_auc"] \
            == pytest.approx(0.90)

        args2 = _args(tmp_path, dtime=1, epochs=2, patience=2, auc_every=1,
                      resume=best)
        train_model(args2)
        after = torch.load(best, map_location="cpu", weights_only=False)
        assert after["val_wp_auc"] == pytest.approx(0.90), (
            f"resume overwrote the 0.90 best with {after['val_wp_auc']}")


# ---------------------------------------------------------------------------
# The memorization probe: train-subsample wp-AUC alongside the val gate
# ---------------------------------------------------------------------------

class TestMemorizationProbe:
    """The gap between train-subsample AUC and block-val AUC is the direct
    measure of how much the model memorises rather than generalises.

    It exists because the absence of one is what let four measurement bugs
    hide for weeks: FROG's interleaved holdout scored *identically* to a
    stride-9 train subsample (78.8% both at matched dtime), so "val" and
    "train" were the same number and nothing could have shown the difference.
    """

    def _spy(self, monkeypatch, train_auc, val_auc):
        seen = []

        def fake(net, dataset, cfg, **kw):
            # the tiny fixture builds train with seed=0 and val with seed=1
            is_val = dataset is not None and float(dataset.scans[0][0][0]) != \
                float(_TinyDataset(seed=0).scans[0][0][0])
            seen.append("val" if is_val else "train")
            v = val_auc if is_val else train_auc
            return {"agnostic": v, "wc": float("nan"), "wa": float("nan"), "wp": v}

        monkeypatch.setattr(train_mod, "evaluate_auc", fake)
        return seen

    def test_both_splits_are_scored_each_gated_epoch(
            self, tiny_setup, tmp_path, monkeypatch):
        seen = self._spy(monkeypatch, train_auc=0.90, val_auc=0.70)
        train_model(_args(tmp_path, dtime=1, epochs=2, patience=0, auc_every=1))
        assert seen.count("train") >= 2, f"train split never probed: {seen}"
        assert seen.count("val") >= 2, f"val split not scored every epoch: {seen}"

    def test_history_records_the_train_probe(self, tiny_setup, tmp_path, monkeypatch):
        self._spy(monkeypatch, train_auc=0.90, val_auc=0.70)
        history = train_model(_args(tmp_path, dtime=1, epochs=2, patience=0,
                                    auc_every=1))
        assert history["train_auc_wp"] == pytest.approx([0.90, 0.90])
        assert history["val_auc_wp"][:2] == pytest.approx([0.70, 0.70])

    def test_gate_still_uses_val_not_the_train_probe(
            self, tiny_setup, tmp_path, monkeypatch):
        """The probe is a diagnostic. Selecting on it would reintroduce exactly
        the bug it exists to expose."""
        self._spy(monkeypatch, train_auc=0.99, val_auc=0.55)
        args = _args(tmp_path, dtime=1, epochs=2, patience=1, auc_every=1)
        train_model(args)
        saved = torch.load(args.out, map_location="cpu", weights_only=False)
        assert saved["val_wp_auc"] == pytest.approx(0.55), (
            "the checkpoint was selected on the train probe, not on val")

    def test_probe_can_be_turned_off(self, tiny_setup, tmp_path, monkeypatch):
        self._spy(monkeypatch, train_auc=0.90, val_auc=0.70)
        history = train_model(_args(tmp_path, dtime=1, epochs=2, patience=0,
                                    auc_every=1, train_probe=False))
        assert history["train_auc_wp"] == []


# ---------------------------------------------------------------------------
# The val pass must stay bounded now that val is three whole recordings
# ---------------------------------------------------------------------------

class TestValPassIsBounded:
    """Val is 195,058 frames — nearly 2x the training set.

    Walking all of it every epoch for a `val_loss` that is only a logged
    diagnostic cost an OOM on a 17 GB machine (the cached val frames alone were
    ~4.9 GB on top of train's ~2.7 GB) and roughly doubled epoch time for a
    number nothing gates on.
    """

    def test_val_loss_pass_is_capped(self, tiny_setup, tmp_path, monkeypatch):
        seen = []
        real = train_mod.evaluate_loss

        def spy(net, frame_ds, *a, **kw):
            seen.append(len(frame_ds))
            return real(net, frame_ds, *a, **kw)

        monkeypatch.setattr(train_mod, "evaluate_loss", spy)
        train_model(_args(tmp_path, dtime=1, epochs=2, patience=0, auc_every=1,
                          auc_max_frames=8, train_probe=False))
        assert seen, "evaluate_loss was never called"
        assert max(seen) <= 8, f"val loss walked {max(seen)} frames, cap was 8"

    def test_val_loss_sample_is_the_same_every_epoch(
            self, tiny_setup, tmp_path, monkeypatch):
        """A diagnostic you eyeball across epochs has to be the same sample each
        time; the old code drew a fresh random subset per epoch."""
        indices = []
        real = train_mod.evaluate_loss

        def spy(net, frame_ds, *a, **kw):
            indices.append(tuple(getattr(frame_ds, "indices", ())))
            return real(net, frame_ds, *a, **kw)

        monkeypatch.setattr(train_mod, "evaluate_loss", spy)
        train_model(_args(tmp_path, dtime=1, epochs=3, patience=0, auc_every=1,
                          auc_max_frames=8, train_probe=False))
        assert len(set(indices)) == 1, "the val-loss sample changed between epochs"

    def test_val_frames_are_not_cached(self, tiny_setup, tmp_path, monkeypatch):
        """Caching 195k preprocessed frames to read 2,000 of them is what blew
        the memory budget."""
        built = []
        real = train_mod.LidarFrameDataset

        class Spy(real):
            def __init__(self, dataset, *a, **kw):
                built.append(kw.get("cache", True))
                super().__init__(dataset, *a, **kw)

        monkeypatch.setattr(train_mod, "LidarFrameDataset", Spy)
        train_model(_args(tmp_path, dtime=1, epochs=1, patience=0, auc_every=1,
                          train_probe=False))
        assert built[0] is True, "the training set should still be cached"
        assert built[1] is False, "the val set must not be cached"


# ---------------------------------------------------------------------------
# The temporal-contribution diagnostic must be runnable from the CLI
# ---------------------------------------------------------------------------

class TestHistoryAblationIsExposed:
    """A21's cheapest measurements -- ablate the temporal axis on a trained
    checkpoint and see whether wp-AUC moves -- cost one eval pass each and
    decide whether the temporal architectures have a rationale. They were
    unreachable from `evaluate.py`, which never exposed the knob
    `evaluate_auc()` already took.
    """

    def test_evaluate_py_exposes_every_mode(self):
        src = (UTILS / "evaluate.py").read_text(encoding="utf-8")
        assert "--eval-history-mode" in src
        for mode in ("shuffle", "frozen", "zero"):
            assert f'"{mode}"' in src, f"{mode} not offered by evaluate.py"

    def test_the_flag_reaches_evaluate_auc(self):
        """A flag that parses but never arrives is the A15 bug again."""
        tree = ast.parse((UTILS / "evaluate.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "evaluate_auc"):
                assert any(k.arg == "history_mode" for k in node.keywords), (
                    f"evaluate_auc at evaluate.py:{node.lineno} drops history_mode")

    @pytest.mark.parametrize("mode", ["zero", "shuffle", "frozen"])
    def test_each_mode_changes_the_input_but_keeps_the_current_frame(self, mode):
        """Guard the semantics, not just the plumbing."""
        from train import _extract_input
        from follow_the_drow.detectors.full_scan import SpaceTimeCNNDetector
        net = SpaceTimeCNNDetector(n_time=5, channels=24, n_spatial_stages=1)
        rng = np.random.RandomState(0)
        scan = rng.rand(N_BEAMS).astype(np.float32) * 4 + 1
        hist = rng.rand(5, N_BEAMS).astype(np.float32) * 4 + 1
        hist[-1] = scan
        od = np.zeros(5, dtype=ODOM_DTYPE)
        cfg, angles = _tiny_cfg(), frog_laser_angles(N_BEAMS)
        full = _extract_input(net, scan, hist, od, angles, cfg, "cpu",
                              align_scans=True, history_mode="full")
        ablated = _extract_input(net, scan, hist, od, angles, cfg, "cpu",
                                 align_scans=True, history_mode=mode,
                                 history_seed=0)
        assert not torch.equal(full, ablated), f"{mode} changed nothing"
        assert torch.allclose(full[:, -1], ablated[:, -1]), (
            f"{mode} disturbed the current frame, which the labels describe")


def test_both_clis_can_print_help():
    """argparse %-formats every help string, so a literal '%' in one crashes
    `--help` at runtime while every other test still passes. Adding measured
    percentages to a flag's help did exactly that to `evaluate.py`."""
    import subprocess
    py = Path(sys.executable)
    for script in ("train.py", "evaluate.py"):
        r = subprocess.run([str(py), str(UTILS / script), "--help"],
                           capture_output=True, text=True, cwd=str(UTILS))
        assert r.returncode == 0, f"{script} --help failed:\n{r.stderr[-800:]}"


# ---------------------------------------------------------------------------
# Temporal-window integrity: the property the whole research question rests on
# ---------------------------------------------------------------------------

class TestTemporalWindowIntegrity:
    """Shuffling reorders which *samples* are drawn; it never touches the
    history inside one.

    Each sample indexes back into the full ordered scan array for its own
    T-frame window, so draw order is irrelevant to its content. Worth locking
    down rather than asserting, because this project's entire claim is about
    temporal information -- and because there *was* a shuffle-dependent
    cross-frame contamination here: before A18 a minibatch was concatenated
    along the beam axis, so whichever frames happened to be shuffled together
    leaked into each other through GroupNorm, the SE gate and the beam
    convolutions.
    """

    def _fds(self, dtime=10):
        ds = _TinyDataset(seed=0, n=120)
        return ds, train_mod.LidarFrameDataset(
            ds, _tiny_cfg(), "raw_scan", 0.6, 48, cache=False, head="drow",
            align_scans=True, dtime=dtime)

    def test_window_holds_the_right_scans_in_chronological_order(self):
        ds, _ = self._fds()
        T, dtime, sid = 5, 10, 100
        hist, _ = ds.get_scan(0, sid, T, dtime=dtime)
        for slot, want in enumerate(range(sid - dtime * (T - 1), sid + 1, dtime)):
            assert np.array_equal(hist[slot], ds.scans[0][want]), f"slot {slot}"

    def test_current_frame_is_last(self):
        ds, _ = self._fds()
        hist, _ = ds.get_scan(0, 100, 5, dtime=10)
        assert np.array_equal(hist[-1], ds.scans[0][100])

    def test_sample_content_is_independent_of_draw_order(self):
        from torch.utils.data import DataLoader
        _, fds = self._fds()
        before = fds[50][0].clone()
        for seed in (0, 1234):
            torch.manual_seed(seed)
            next(iter(DataLoader(fds, batch_size=4, shuffle=True, num_workers=0)))
        assert torch.equal(before, fds[50][0])

    def test_window_never_crosses_a_sequence_boundary(self):
        """get_scan clamps at index 0 of its own sequence, so an early frame
        repeats the earliest available scan rather than reaching into the
        previous recording."""
        ds, _ = self._fds()
        hist, _ = ds.get_scan(0, 3, 5, dtime=10)
        assert np.array_equal(hist[0], ds.scans[0][0])
        assert np.array_equal(hist[-1], ds.scans[0][3])

    def test_dtime_actually_spaces_the_window(self):
        ds, _ = self._fds()
        near, _ = ds.get_scan(0, 100, 5, dtime=1)
        far, _ = ds.get_scan(0, 100, 5, dtime=10)
        assert np.array_equal(near[-1], far[-1])          # same 'now'
        assert not np.array_equal(near[0], far[0])        # different reach


# ---------------------------------------------------------------------------
# History ablations: in-distribution ways to ask what the temporal axis does
# ---------------------------------------------------------------------------

class TestHistoryModes:
    """`zero` substitutes an input the model has never seen -- a 0 m range is
    physically impossible -- so its 25.1pp cost on the real test set is mostly
    out-of-distribution damage, not information loss. The tell: it scored 42.0%,
    23pp *below* LFE-Peaks' genuinely single-frame 64.9%.

    `shuffle` and `frozen` stay in-distribution and separate the two ways a
    temporal window can help: motion (ordering, velocity) from aggregation
    (several noisy looks at one scene). Shuffle destroys the first and keeps the
    second; frozen removes both.
    """

    def _win(self, T=5, n_beams=32, seed=0):
        rng = np.random.RandomState(seed)
        scans = (rng.rand(T, n_beams).astype(np.float32) * 4 + 1)
        od = np.zeros(T, dtype=ODOM_DTYPE)
        od["xya"][:, 0] = np.arange(T)          # distinct per-slot odometry
        return scans, od

    def test_frozen_replaces_history_with_the_current_frame(self):
        scans, od = self._win()
        s2, o2 = train_mod._apply_history_mode(scans, od, "frozen")
        for slot in range(len(s2)):
            assert np.array_equal(s2[slot], scans[-1]), f"slot {slot}"
        # odometry is frozen too: a static world includes a static robot
        assert np.allclose(o2["xya"], od["xya"][-1])

    def test_shuffle_keeps_the_current_frame_last(self):
        """The label describes the current scan, so it must stay in slot -1."""
        scans, od = self._win()
        s2, _ = train_mod._apply_history_mode(scans, od, "shuffle", seed=3)
        assert np.array_equal(s2[-1], scans[-1])

    def test_shuffle_permutes_history_without_inventing_frames(self):
        scans, od = self._win(T=6)
        s2, _ = train_mod._apply_history_mode(scans, od, "shuffle", seed=1)
        before = sorted(tuple(f) for f in scans[:-1])
        after = sorted(tuple(f) for f in s2[:-1])
        assert before == after, "shuffle changed the multiset of history frames"

    def test_shuffle_moves_scans_and_their_odometry_together(self):
        """Scrambling scans while leaving odometry put would corrupt the
        ego-motion alignment rather than the motion cue."""
        scans, od = self._win(T=6, seed=5)
        s2, o2 = train_mod._apply_history_mode(scans, od, "shuffle", seed=7)
        for slot in range(len(s2)):
            src = next(k for k in range(len(scans)) if np.array_equal(scans[k], s2[slot]))
            assert o2["xya"][slot][0] == od["xya"][src][0], f"slot {slot} odom detached"

    def test_full_is_a_no_op(self):
        scans, od = self._win()
        s2, o2 = train_mod._apply_history_mode(scans, od, "full")
        assert np.array_equal(s2, scans) and np.array_equal(o2["xya"], od["xya"])

    def test_zero_still_available_and_blanks_only_history(self):
        scans, od = self._win()
        s2, _ = train_mod._apply_history_mode(scans, od, "zero")
        assert (s2[:-1] == 0).all()
        assert np.array_equal(s2[-1], scans[-1])

    def test_unknown_mode_is_rejected(self):
        scans, od = self._win()
        with pytest.raises(ValueError):
            train_mod._apply_history_mode(scans, od, "blank")

    def test_shuffle_is_deterministic_for_a_given_seed(self):
        scans, od = self._win(T=6)
        a, _ = train_mod._apply_history_mode(scans, od, "shuffle", seed=11)
        b, _ = train_mod._apply_history_mode(scans, od, "shuffle", seed=11)
        assert np.array_equal(a, b)


class TestRunupScans:
    """Frames closer than RUNUP_SCANS to a sequence start get a padded,
    motion-free window -- a wrong input paired with a correct label, aimed
    squarely at what this project studies. Measured cost of dropping them at
    run-up 40: train -1.81%, val -0.00% (three unbroken recordings), test
    -3.75% (47 fragments, median 688 scans).

    Test keeps them by default: the FROG paper scores every populated scan, and
    a 3.75% frame-set difference would break comparability with the published
    numbers to fix a smaller contaminant.
    """

    def test_runup_is_a_fixed_width_not_a_function_of_dtime(self):
        from follow_the_drow.datasets import frog_dataset as fd
        src = Path(fd.__file__).read_text(encoding="utf-8")
        assert "RUNUP_SCANS" in src
        assert "dtime" not in src.split("RUNUP_SCANS")[1].split("\n")[0]

    def test_train_and_val_drop_the_run_up_but_test_does_not(self):
        from follow_the_drow.datasets.frog_dataset import _SPLIT_RUNUP, RUNUP_SCANS
        assert _SPLIT_RUNUP["train"] == RUNUP_SCANS
        assert _SPLIT_RUNUP["val"] == RUNUP_SCANS
        assert _SPLIT_RUNUP["test"] == 0, (
            "test must keep every populated frame to match the published protocol")


class TestCheckpointCarriesItsFrogPartition:
    """A checkpoint scored under the wrong FROG partition is not a worse number,
    it is a different experiment -- and `balanced`'s val contains recordings
    `official` trains on, so getting it wrong scores a model on its own
    training data. That is exactly the class of bug A11 records, which is why
    the partition travels with the weights rather than with the invocation
    (TODO.md A30).
    """

    def _ckpt(self, tmp_path, name, frog_mode):
        net = torch.nn.Linear(2, 2)
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        path = tmp_path / name
        save_checkpoint(path, net, opt, epoch=1, detector_name="spacetime_cnn",
                        dataset_name="frog", dtime=10, frog_mode=frog_mode)
        return path

    def test_the_mode_round_trips_through_the_checkpoint(self, tmp_path):
        from evaluate import _resolve_frog_mode
        path = self._ckpt(tmp_path, "balanced.pth", "balanced")
        assert torch.load(path, map_location="cpu")["frog_mode"] == "balanced"
        assert _resolve_frog_mode([path]) == "balanced"

    def test_an_older_checkpoint_falls_back_to_official(self, tmp_path):
        """Checkpoints saved before the field existed were trained on the
        official partition, so that is the honest default -- not an error."""
        from evaluate import _resolve_frog_mode
        path = self._ckpt(tmp_path, "old.pth", None)
        assert "frog_mode" not in torch.load(path, map_location="cpu")
        assert _resolve_frog_mode([path]) == "official"

    def test_two_partitions_in_one_pass_is_an_error_not_a_guess(self, tmp_path):
        from evaluate import _resolve_frog_mode
        paths = [self._ckpt(tmp_path, "a.pth", "official"),
                 self._ckpt(tmp_path, "b.pth", "balanced")]
        with pytest.raises(SystemExit, match="disagree on their FROG partition"):
            _resolve_frog_mode(paths)

    def test_transferred_and_official_share_a_checkpoint_by_construction(self):
        """A32b's premise: a `transferred` number must be the *official*
        checkpoint measured elsewhere. That only holds while the two modes'
        train and val entries are the same table rows."""
        from follow_the_drow.datasets.frog_dataset import _MODES
        for split in ("train", "val"):
            assert _MODES["official"][split] == _MODES["transferred"][split], split
        assert _MODES["official"]["test"] != _MODES["transferred"]["test"]
