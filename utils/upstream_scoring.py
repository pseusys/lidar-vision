#!/usr/bin/env python3
"""Score Li2Former with **their** decoder and **their** metric, as a 2x2 over the two halves.

Our `drow_eval` number and their published 76.4% AP@0.5 are not the same measurement twice. Ours decodes votes
with `votes_to_detections` -- every beam's vote accumulated into a 0.1 m grid, blurred, collected within 0.3 m --
and scores with `_prec_rec_2d`. Theirs projects each beam's vote to global x-y, greedily suppresses within 0.5 m,
and scores with `calPrecisionRecall`. The 2x2 varies one half at a time, so a difference lands on a named half.

**The upstream tree is GPL-3.0 and this repository is MIT.** Everything here is *borrowed at call time* from
`takheliped/include/li2former_upstream/` (gitignored, fetched by `library/setup.py`); nothing is copied in, and
`tests/test_upstream_scoring.py` pins that, because the failure mode is a well-meaning copy-paste that
relicenses someone else's work.

Three properties of their code that the call sites here have to respect, all established by reading it:

* **Their suppression wants exactly one scan row.** It reads `_, num_pts = scans.shape` and then reshapes on
  `num_pts` alone, so a `[R, N]` window with `R > 1` dies inside numpy with a shape message that names neither
  this project nor the cause. `nms_detections` checks first and says so.
* **Their suppression ends in `np.squeeze`,** which collapses a lone detection from `[1, 2]` to `(2,)`. Every
  caller would otherwise need that special case; this one does the reshape once.
* **Their frame is not ours.** They use `r cos phi, r sin phi`; this project uses `-r sin phi, r cos phi`. The
  offset step is *identical* in both trees (their `local2global` and our `_win2global` are the same formula), so
  the only bridge needed is the 90-degree rotation the repository already names
  `standard_cartesian_to_project_cartesian`. Both radii in this item are **distances**, and a rotation preserves
  distance, so the frame cannot change a score -- as long as detections and ground truth are in the *same* frame.
  That is why everything leaving here is in our frame: cell 2 feeds their detections to our ground truth.

Two properties of their **metric** worth knowing before reading its output against their paper:

* **Their AP integrates over the recall range it observes,** `sklearn.metrics.auc(recalls, precisions)`, with no
  anchor at recall 0. With `n` detections all correct it returns `1 - 1/n`, not 1. At DROW test scale that first
  step is negligible; on a handful of frames it is not, which is why the tests here use many frames.
* **A cell with no true positives raises inside their `eer()`**, which indexes `np.where(precs != 0)[0][0]`.
  That is left to raise. Catching it would hide the only thing it can mean -- the AP of that cell is 0 -- and
  patching their file is out of scope for this item by the owner's instruction: their code is scored as released.
"""

from dataclasses import dataclass
from importlib import import_module

import numpy as np

from takheliped.utils.drow_utils import standard_cartesian_to_project_cartesian

UPSTREAM_MISSING = (
    "Li2Former's own decoder and metric are not in takheliped/include/li2former_upstream -- "
    "reinstall the library (`pip install ./library`) to fetch the upstream tree, which is "
    "GPL-3.0 and therefore downloaded rather than vendored."
)

_EVAL_UTILS = "takheliped.include.li2former_upstream.utils.eval_utils"


def upstream_eval_utils():
    """Their `eval_utils` module itself, so a caller can reach whichever of its functions it needs."""
    try:
        return import_module(_EVAL_UTILS)
    except ImportError as exc:
        raise ImportError(UPSTREAM_MISSING) from exc


def nms_detections(scan: np.ndarray, phi: np.ndarray, pred_cls: np.ndarray, pred_reg: np.ndarray,
                   min_dist: float = 0.5):
    """Their greedy suppression, returning `(det_xy [K, 2], det_scores [K])` in **this project's** frame.

    `scan` is one scan shaped `[1, N]`; `pred_cls` is `[N]` and `pred_reg` is `[N, 2]` in beam-local
    `(dx, dy)`, which is the frame their offset step and ours agree on.
    """
    scan = np.asarray(scan, dtype=np.float32)
    if scan.ndim != 2 or scan.shape[0] != 1:
        raise ValueError(
            f"their suppression reshapes on the beam count alone and needs exactly one scan row, got "
            f"{scan.shape}; pass a single scan as [1, N]"
        )

    det_xy, det_scores, _ = upstream_eval_utils().nms(
        scan,
        np.asarray(phi, dtype=np.float32),
        np.asarray(pred_cls, dtype=np.float32).reshape(-1),
        np.asarray(pred_reg, dtype=np.float32).reshape(-1, 2),
        min_dist=min_dist,
    )

    # Their output is [K, 2, 1], or (2,) when a single detection survives `np.squeeze`.
    det_xy = np.asarray(det_xy, dtype=np.float32).reshape(-1, 2)
    x, y = standard_cartesian_to_project_cartesian(det_xy[:, 0], det_xy[:, 1])
    return np.stack((x, y), axis=1), np.asarray(det_scores, dtype=np.float32).reshape(-1)


def upstream_ap(det_xy: np.ndarray, det_scores: np.ndarray, det_frames: np.ndarray,
                gt_xy: np.ndarray, gt_frames: np.ndarray, radius: float) -> dict:
    """Their metric over a whole split: `AP`, `peak_F1` and `EER` at one matching radius.

    Detections and ground truth are in the same frame -- this project's, since the rotation between the two
    preserves the distances both the matching radius and the suppression threshold are measured in. `peak_F1`
    and `EER` come free with their AP and have never been reported for this detector.
    """
    return upstream_eval_utils().calPrecisionRecall(
        np.asarray(det_xy, dtype=np.float32).reshape(-1, 2),
        np.asarray(det_scores, dtype=np.float32).reshape(-1),
        np.asarray(det_frames, dtype=np.int32).reshape(-1),
        np.asarray(gt_xy, dtype=np.float32).reshape(-1, 2),
        np.asarray(gt_frames, dtype=np.int32).reshape(-1),
        radius,
    )


@dataclass(frozen=True)
class Cell:
    """One cell of the 2x2: which decoder turns network output into detections, and which metric scores them."""

    name: str
    decoder: str
    metric: str
    reads: str


GRID = (
    Cell("ours/ours", "ours", "ours",
         "the chain's own drow_eval -- the baseline this item is read against"),
    Cell("theirs/ours", "theirs", "ours",
         "how much of the gap is our vote-averaging decoder, which an escaped-vote tail corrupts"),
    Cell("ours/theirs", "ours", "theirs",
         "how much of the gap is our AUC against their AP, matching and integration included"),
    Cell("theirs/theirs", "theirs", "theirs",
         "their published protocol end to end -- the only apples-to-apples row against 76.4% AP@0.5"),
)

# `evaluate_auc`'s own decoder settings. Not a new default: the same dict it falls back to, repeated here so
# "ours" in the grid is the path that produced the reported number rather than a second opinion about it.
V2D_CONF = {"blur_sigma": 2.0, "blur_win": 11, "bin_size": 0.1,
            "vote_collect_radius": 0.5, "min_thresh": 1e-3}

WP_COLUMN = 3   # this repository's class coding: 0=bg, 1=wc, 2=wa, 3=wp


def decode_ours(raw: dict, v2d_conf: dict = None):
    """Our decoder over a `return_raw` bundle: `(det_x, det_y, det_p, det_f)`, as `_process_detections` wants.

    Every step is `evaluate_auc`'s, in its order, because `tests/test_upstream_grid_wiring.py` requires the AUC
    computed from this to equal the one that function returned.
    """
    from takheliped.utils.drow_utils import _deep2flat, _win2global, votes_to_detections

    angles = raw["angles"]
    xs, ys = [], []
    for scan, votes in zip(raw["scans"], raw["votes"]):
        r_new, phi_new = _win2global(scan[None], angles[None], votes[None, :, 0], votes[None, :, 1])
        xs.append(r_new * -np.sin(phi_new))
        ys.append(r_new * np.cos(phi_new))

    detections = votes_to_detections(
        np.concatenate(xs, axis=0), np.concatenate(ys, axis=0),
        np.stack(raw["confs"], axis=0), **(v2d_conf or V2D_CONF))
    return _deep2flat(detections)


def decode_theirs(raw: dict, min_dist: float = 0.5):
    """Their decoder over the same bundle, in the same `(det_x, det_y, det_p, det_f)` layout.

    Their suppression takes one confidence per beam, so it is given the **person** column -- the class every
    wp-AUC in this project is about, and the one their `pedestrian_only` weights are trained on. The score goes
    into column 3 for the same reason the heatmap path does it: `_process_detections` reads it there.
    """
    angles = raw["angles"]
    xs, ys, ps, fs = [], [], [], []
    for frame_idx, (scan, votes, conf) in enumerate(zip(raw["scans"], raw["votes"], raw["confs"])):
        conf = np.asarray(conf, dtype=np.float32)
        det_xy, det_scores = nms_detections(
            scan[None], angles, conf[:, WP_COLUMN] if conf.ndim == 2 else conf, votes, min_dist=min_dist)
        xs.append(det_xy[:, 0])
        ys.append(det_xy[:, 1])
        ps.append(det_scores)
        fs.append(np.full(len(det_scores), frame_idx, dtype=np.int64))

    det_p = np.zeros((sum(len(p) for p in ps), 4), dtype=np.float32)
    det_p[:, WP_COLUMN] = np.concatenate(ps) if ps else np.zeros(0, dtype=np.float32)
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32),
            det_p, np.concatenate(fs))


def person_ground_truth(raw: dict):
    """The person annotations as `(gt_xy, gt_frames)` in this project's frame, for their metric."""
    from takheliped.utils.drow_utils import project_cartesian_from_polar

    xy, frames = [], []
    for frame_idx, annotations in enumerate(raw["wps"]):
        for r, phi in annotations:
            x, y = project_cartesian_from_polar(np.float32(r), np.float32(phi))
            xy.append((float(x), float(y)))
            frames.append(frame_idx)
    return (np.array(xy, dtype=np.float32).reshape(-1, 2), np.array(frames, dtype=np.int32))


def score_grid(raw: dict, eval_r: float = 0.5, v2d_conf: dict = None, min_dist: float = 0.5) -> dict:
    """All four cells from one inference pass. Keys are `GRID`'s names.

    Our metric reports `wp_auc`; theirs reports `AP`, `peak_F1` and `EER` at the same matching radius. A cell
    whose detections never hit raises out of their `eer()` -- see this module's docstring; that is left alone.
    """
    from takheliped.utils.drow_utils import _process_detections

    from train import _safe_auc

    decoded = {"ours": decode_ours(raw, v2d_conf), "theirs": decode_theirs(raw, min_dist)}
    gt_xy, gt_frames = person_ground_truth(raw)

    results = {}
    for cell in GRID:
        det_x, det_y, det_p, det_f = decoded[cell.decoder]
        if cell.metric == "ours":
            _, _, _, wp = _process_detections(det_x, det_y, det_p, det_f,
                                              raw["wcs"], raw["was"], raw["wps"], eval_r)
            results[cell.name] = {"wp_auc": _safe_auc(*wp[:2]), "n_det": int(len(det_x))}
        else:
            scored = upstream_ap(np.stack((det_x, det_y), axis=1), det_p[:, WP_COLUMN],
                                 det_f, gt_xy, gt_frames, eval_r)
            results[cell.name] = {key: scored[key] for key in ("AP", "peak_F1", "EER")}
            results[cell.name]["n_det"] = int(len(det_x))
    return results
