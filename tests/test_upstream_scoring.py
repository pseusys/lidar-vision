"""`TODO.md` item 64: score Li2Former with their own decoder and their own metric, as a 2x2 over the two halves.

Item 31's resume leg produced a decoupling worth explaining: their loss improves on held-out DROW in **both** terms
across epochs 5-6 -- `val_loss` 0.7013 -> 0.6511 -> 0.6494, below epoch 1's 0.6784 -- while our `val wp-AUC` sits in
its worst band of the run and does not follow. Either the decode-and-score path disagrees with their loss about what a
good prediction is, or it does not and the gap is in the weights. The 2x2 separates those: ours/ours is the chain's own
`drow_eval`, theirs/theirs is their published protocol against 76.4% AP@0.5, and the two mixed cells say which half
moved the number.

**The upstream tree is GPL-3.0 and this repo is MIT**, so `nms`, `calPrecisionRecall` and `evaluate` are imported from
`takheliped/include/li2former_upstream/` at call time, never copied in. `TestTheUpstreamCodeIsBorrowedNotVendored`
pins that, because the failure mode is a well-meaning copy-paste that relicenses someone else's work.

Two groups of tests here, and they are not the same kind:

* `TestTheTwoProjectionsAlreadyAgree` **characterises code that already exists** in both trees. These pass on the
  first run by design -- they are the written form of the convention audit, kept so a later edit to either projection
  cannot silently invalidate cells 2 and 4.
* Everything else drives new code in `utils/upstream_scoring.py` and fails until it exists.

Dataset-free, per `tests/`'s standing rule. The convention check against **real recorded DROW frames** that rule 4
requires is a separate script, not a unit test: a synthetic scan cannot catch a sign error that only shows up in
recorded geometry, and that is exactly the mistake this repo has made twice.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from takheliped.include.li2former_upstream.utils.eval_utils import nms as upstream_nms
from takheliped.include.li2former_upstream.utils.scan_utils import local2global, rphi2xy
from takheliped.utils.drow_utils import (
    _win2global,
    laser_angles,
    project_cartesian_from_polar,
    standard_cartesian_to_project_cartesian,
)

import upstream_scoring as us


# A plausible knee-height scan: 450 beams over DROW's field of view, people at a few metres.
N_BEAMS = 450
PHI = laser_angles(N_BEAMS)
RNG = np.random.default_rng(31_064)


def a_scan() -> np.ndarray:
    """One scan, shaped `[1, N]` -- their `nms` reads `scans.shape` as 2D and reshapes on `num_pts`."""
    return (3.0 + RNG.normal(0.0, 0.25, size=N_BEAMS)).astype(np.float32)[None, :]


def votes_towards(dx: float, dy: float) -> np.ndarray:
    return np.tile(np.array([dx, dy], dtype=np.float32), (N_BEAMS, 1))


class TestTheTwoProjectionsAlreadyAgree:
    """The convention audit, written down. These pass as soon as they are run -- that is the point of them.

    Their `local2global` and our `_win2global` turn out to be the same formula, and their Cartesian step is ours
    rotated 90 degrees, a rotation the repo already names `standard_cartesian_to_project_cartesian`. So no new
    conversion code is needed and the risk to cells 2 and 4 is a *mixing* risk, not a maths one.
    """

    def test_the_local_offset_step_is_the_same_formula_in_both_trees(self):
        r, phi = a_scan()[0], PHI
        dx, dy = RNG.normal(0.0, 0.4, N_BEAMS), RNG.normal(0.0, 0.4, N_BEAMS)

        ours_r, ours_phi = _win2global(r, phi, dx, dy)
        theirs_r, theirs_phi = local2global(r, phi, dx, dy, cartesian=False)

        np.testing.assert_allclose(ours_r, theirs_r, rtol=1e-6)
        np.testing.assert_allclose(ours_phi, theirs_phi, rtol=1e-6)

    def test_their_cartesian_is_ours_rotated_ninety_degrees(self):
        r, phi = a_scan()[0], PHI

        theirs = standard_cartesian_to_project_cartesian(*rphi2xy(r, phi))
        ours = project_cartesian_from_polar(r, phi)

        np.testing.assert_allclose(theirs, ours, rtol=1e-6)

    def test_the_rotation_cannot_change_any_distance(self):
        """Both radii in this item are distance thresholds, so a frame difference is harmless until frames are mixed."""
        r, phi = a_scan()[0], PHI
        ours = np.stack(project_cartesian_from_polar(r, phi), axis=1)
        theirs = np.stack(rphi2xy(r, phi), axis=1)

        def pairwise(points):
            d = points[:40, None, :] - points[None, :40, :]
            return np.hypot(d[..., 0], d[..., 1])

        np.testing.assert_allclose(pairwise(ours), pairwise(theirs), rtol=1e-5)


class TestTheirDecoderIsReadInOurFrame:
    """Cell 2 feeds their detections to our metric, so this is the one place a frame mix-up would score silently."""

    def test_it_returns_detections_in_the_project_frame(self):
        scan, conf = a_scan(), RNG.uniform(0.0, 1.0, N_BEAMS)
        det_xy, _ = us.nms_detections(scan, PHI, conf, votes_towards(0.0, 0.0))

        # A zero vote leaves every detection on its own beam, so the kept points must lie on the scan itself --
        # in our frame, where y is forward. Their raw output has x forward and would fail this.
        on_beam = np.stack(project_cartesian_from_polar(scan[0], PHI), axis=1)
        for point in det_xy:
            assert np.min(np.hypot(*(on_beam - point).T)) < 1e-4

    def test_it_always_returns_a_two_column_array(self):
        """Their `nms` ends in `np.squeeze`, which collapses a single detection to shape `(2,)`."""
        scan = a_scan()
        conf = np.full(N_BEAMS, 0.9)

        det_xy, det_scores = us.nms_detections(scan, PHI, conf, votes_towards(0.0, 0.0), min_dist=1e3)

        assert det_xy.ndim == 2 and det_xy.shape[1] == 2
        assert len(det_xy) == len(det_scores) == 1

    def test_it_keeps_one_detection_per_suppression_radius(self):
        scan, conf = a_scan(), RNG.uniform(0.0, 1.0, N_BEAMS)

        wide, _ = us.nms_detections(scan, PHI, conf, votes_towards(0.0, 0.0), min_dist=2.0)
        narrow, _ = us.nms_detections(scan, PHI, conf, votes_towards(0.0, 0.0), min_dist=0.1)

        assert len(wide) < len(narrow)

    def test_it_scores_a_detection_with_its_own_beams_confidence(self):
        """Their decoder keeps one beam's vote at one beam's score; ours averages many. That difference is the cell."""
        scan, conf = a_scan(), RNG.uniform(0.1, 0.9, N_BEAMS)
        conf[200] = 1.0

        _, det_scores = us.nms_detections(scan, PHI, conf, votes_towards(0.0, 0.0), min_dist=1e3)

        assert det_scores[0] == pytest.approx(1.0)

    def test_it_rejects_a_scan_window_their_nms_cannot_handle(self):
        """Their suppression reshapes on `num_pts` alone, so a multi-row window cannot work.

        The assertion is on the **message**, not the type: numpy raises `ValueError` here by itself, from a
        reshape deep inside their file, naming neither this project nor the cause. Mutation-tested -- without
        the guard in `nms_detections` this test passed on numpy's incidental error, which made it worthless.
        """
        with pytest.raises(ValueError, match="one scan row"):
            us.nms_detections(np.tile(a_scan(), (5, 1)), PHI, RNG.uniform(0, 1, N_BEAMS), votes_towards(0.0, 0.0))


class TestTheirMetricIsWiredAsAnAP:
    """Their AP is `sklearn.metrics.auc(recalls, precisions)` with **no anchor at recall 0**, which is why none
    of these tests expects 1.0 and why all of them use many frames rather than one.
    """

    def test_it_reports_their_three_numbers(self):
        """`peak_F1` and `EER` come free with their AP and have never been reported for this detector."""
        result = us.upstream_ap(*_frames_on_target(), radius=0.5)

        assert set(result) >= {"AP", "peak_F1", "EER"}

    def test_every_detection_on_target_scores_their_ceiling_not_one(self):
        """With `n` correct detections their integration starts at recall `1/n`, so the ceiling is `1 - 1/n`."""
        result = us.upstream_ap(*_frames_on_target(), radius=0.5)

        assert result["AP"] == pytest.approx(1.0 - 1.0 / N_FRAMES, abs=1e-6)
        assert result["peak_F1"] == pytest.approx(1.0, abs=1e-6)

    def test_the_radius_we_pass_is_the_matching_radius(self):
        """Their 76.4% is AP@0.5 and their 72.5% is AP@0.3; a detection 0.4 m out falls between the two.

        Half the frames are left on target, because a split with *no* true positives raises inside their
        `eer()` -- `np.where(precs != 0)[0][0]` on an all-zero precision array.
        """
        args = _frames_on_target(offset_x=0.4, offset_every=2)

        assert us.upstream_ap(*args, radius=0.5)["AP"] > us.upstream_ap(*args, radius=0.3)["AP"]

    def test_two_detections_cannot_both_claim_one_target(self):
        """They match with `linear_sum_assignment`, so a second detection on one target is a false positive.

        The duplicate is interleaved by score rather than appended: their AP integrates over recall, so false
        positives arriving once recall has saturated add no area and would hide the effect entirely.
        """
        det, scores, frames, gt, gt_frames = _frames_on_target()
        duplicate = det + np.array([0.1, 0.0], dtype=np.float32)

        both = us.upstream_ap(
            np.concatenate([det, duplicate]), np.concatenate([scores, scores - 0.01]),
            np.concatenate([frames, frames]), gt, gt_frames, radius=0.5,
        )
        clean = us.upstream_ap(det, scores, frames, gt, gt_frames, radius=0.5)

        assert both["AP"] < clean["AP"]


class TestTheGridIsFourCellsOnOneCheckpoint:
    def test_it_names_the_four_cells_the_owner_approved(self):
        assert {cell.name for cell in us.GRID} == {
            "ours/ours", "theirs/ours", "ours/theirs", "theirs/theirs",
        }

    def test_each_half_is_varied_independently(self):
        """A 2x2 is only readable if each decoder appears under each metric exactly once."""
        pairs = {(cell.decoder, cell.metric) for cell in us.GRID}

        assert pairs == {("ours", "ours"), ("theirs", "ours"), ("ours", "theirs"), ("theirs", "theirs")}

    def test_the_baseline_cell_is_the_path_the_chain_already_ran(self):
        baseline = next(cell for cell in us.GRID if cell.name == "ours/ours")

        assert (baseline.decoder, baseline.metric) == ("ours", "ours")


class TestTheUpstreamCodeIsBorrowedNotVendored:
    """GPL-3.0 upstream, MIT here. The guard is cheap and the mistake it catches is not."""

    def test_it_does_not_define_their_functions_itself(self):
        source = Path(us.__file__).read_text(encoding="utf-8")

        for borrowed in ("def nms(", "def calPrecisionRecall(", "def local2global("):
            assert borrowed not in source

    def test_it_reaches_them_through_the_gitignored_include_tree(self):
        source = Path(us.__file__).read_text(encoding="utf-8")

        assert "include.li2former_upstream.utils.eval_utils" in source

    def test_it_says_what_is_missing_when_the_tree_is_absent(self):
        """`li2former_official.py` raises a sentence naming the tree; a bare ImportError would not."""
        assert "li2former_upstream" in us.UPSTREAM_MISSING

    def test_the_function_it_borrows_is_the_one_upstream_defines(self):
        assert us.upstream_eval_utils().nms is upstream_nms


N_FRAMES = 20


def _frames_on_target(offset_x: float = 0.0, offset_every: int = 1):
    """`N_FRAMES` frames, one target at 3 m dead ahead in each, one detection per frame.

    `offset_x` displaces every `offset_every`-th detection sideways, which is how a matching radius is made to
    matter. Many frames rather than one, because `sklearn.metrics.auc` refuses a single point outright and
    their AP integrates from the first recall it sees.
    """
    gt = np.tile(np.array([0.0, 3.0], dtype=np.float32), (N_FRAMES, 1))
    det = gt.copy()
    det[::offset_every, 0] += offset_x
    frames = np.arange(N_FRAMES)
    return det, np.linspace(0.9, 0.5, N_FRAMES, dtype=np.float32), frames, gt, frames
