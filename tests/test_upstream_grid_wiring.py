"""The upstream grid's wiring: one inference pass, four scorings, and cell 1 *identical* to the chain's number.

The 2x2 is only readable if its baseline cell is the same measurement `drow_eval` already ran. The way to
guarantee that is to decode nothing twice: `evaluate_auc` grows one additive `return_raw` hook that hands back
the per-beam confidences, votes and scans it scored, and all four cells are computed from that one bundle. If
our decoder applied to the bundle disagreed with `evaluate_auc`'s own answer by even a rounding step, the grid's
"ours/ours" column would not be the 48.1% it is being compared against -- so that equality is the test this
file exists for, and the rest guard the parts around it.

Dataset-free, per `tests/`'s standing rule. The tiny in-memory dataset mirrors
`tests/test_train_eval_consistency.py`'s `_TinyDataset`, which is the shape `evaluate_auc` actually touches.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from train import _build_model, _default_args, evaluate_auc  # noqa: E402

import upstream_scoring as us  # noqa: E402

N_BEAMS = 64
N_FRAMES = 8
ODOM_DTYPE = np.dtype([("eq", np.uint32), ("t", np.float64), ("xya", np.float32, 3)])


class _TinyDataset:
    """The smallest object `evaluate_auc` accepts: one sequence, one person per frame, straight ahead."""

    def __init__(self, n=N_FRAMES, seed=0):
        rng = np.random.RandomState(seed)
        self.time_frame = 5
        box = lambda: np.array([None], dtype=object)  # noqa: E731
        self.scan_id, self.scan_time = box(), box()
        self.scans, self.odoms = box(), box()
        self.det_id, self.det_wc, self.det_wa, self.det_wp = box(), box(), box(), box()

        self.scan_id[0] = np.arange(n, dtype=np.uint32)
        self.scan_time[0] = np.arange(n, dtype=np.float64) * 0.038
        self.scans[0] = rng.rand(n, N_BEAMS).astype(np.float32) * 2.0 + 2.0
        odom = np.zeros(n, dtype=ODOM_DTYPE)
        odom["t"] = self.scan_time[0]
        self.odoms[0] = odom
        self.det_id[0] = np.arange(n, dtype=np.uint32)
        self.det_wc[0] = np.array([[] for _ in range(n)], dtype=object)
        self.det_wa[0] = np.array([[] for _ in range(n)], dtype=object)
        self.det_wp[0] = np.array(
            [[(float(self.scans[0][i, N_BEAMS // 2]), 0.0)] for i in range(n)], dtype=object)
        self.idet2iscan = [{i: int(self.det_id[0][i]) for i in range(n)}]

    def get_scan(self, sequence_id, scan_id, time_window, dtime=1):
        idx = np.clip(scan_id - dtime * np.arange(time_window - 1, -1, -1), 0, scan_id)
        return self.scans[sequence_id][idx], self.odoms[sequence_id][idx]


def _cfg():
    angles = np.linspace(-np.pi / 2, np.pi / 2, N_BEAMS, dtype=np.float32)
    return SimpleNamespace(name="tiny", angles_fn=lambda n: angles[:n],
                           fov_min=-np.pi / 2, fov_max=np.pi / 2, laser_inc=np.pi / N_BEAMS)


def _net(seed=0):
    """An untrained network, seeded -- and the seed is load-bearing, not tidiness.

    The borrowed metric's `eer()` indexes `np.where(precs != 0)[0][0]`, so it *raises* on a split where no
    detection ever matched an annotation. An unseeded random network produces exactly that split most of the
    time, which made this module pass or fail on the draw. Their code is scored as released, so the fix
    belongs here: fix the weights, and the bundle always carries at least one true positive.
    """
    torch.manual_seed(seed)
    return _build_model(_default_args(
        detector="spacetime_cnn", dataset="frog", force_cpu=True,
        channels=16, n_spatial_stages=1, dropout=0.0))


@pytest.fixture(scope="module")
def scored():
    """One inference pass, kept for the whole module: `evaluate_auc`'s own answer and its raw bundle."""
    dataset, cfg = _TinyDataset(), _cfg()
    return evaluate_auc(_net(), dataset, cfg, eval_r=0.5, device="cpu", return_raw=True)


class TestTheRawHookCarriesWhatBothDecodersNeed:
    def test_it_is_absent_unless_asked_for(self):
        result = evaluate_auc(_net(), _TinyDataset(), _cfg(), eval_r=0.5, device="cpu")

        assert "raw" not in result

    def test_it_has_one_entry_per_scored_frame(self, scored):
        raw = scored["raw"]

        for key in ("scans", "votes", "confs", "wps"):
            assert len(raw[key]) == N_FRAMES, key

    def test_each_frame_carries_a_vote_and_a_confidence_per_beam(self, scored):
        raw = scored["raw"]

        assert raw["scans"][0].shape == (N_BEAMS,)
        assert raw["votes"][0].shape == (N_BEAMS, 2)
        assert len(raw["confs"][0]) == N_BEAMS


class TestCellOneIsTheChainsOwnNumber:
    def test_our_decoder_on_the_bundle_reproduces_evaluate_aucs_answer(self, scored):
        """The comparability guarantee: if this drifts, nothing in the 2x2 may be read against 48.1%."""
        grid = us.score_grid(scored["raw"], eval_r=0.5)

        assert grid["ours/ours"]["wp_auc"] == pytest.approx(scored["wp"], abs=1e-9)


class TestTheGridScoresFourCells:
    def test_it_returns_every_cell_the_owner_approved(self, scored):
        grid = us.score_grid(scored["raw"], eval_r=0.5)

        assert set(grid) == {cell.name for cell in us.GRID}

    def test_our_metric_reports_an_auc_and_theirs_reports_their_three_numbers(self, scored):
        grid = us.score_grid(scored["raw"], eval_r=0.5)

        assert "wp_auc" in grid["ours/ours"]
        assert {"AP", "peak_F1", "EER"} <= set(grid["theirs/theirs"])

    def test_the_two_decoders_do_not_produce_the_same_detections(self, scored):
        """One averages every beam into a grid, the other keeps one beam per 0.5 m. Equal output means a bug."""
        ours = us.decode_ours(scored["raw"])
        theirs = us.decode_theirs(scored["raw"])

        assert len(ours[0]) != len(theirs[0]) or not np.allclose(ours[0], theirs[0])
