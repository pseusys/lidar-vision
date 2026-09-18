#!/usr/bin/env python3
"""
Comprehensive evaluation and throughput benchmark for all person detectors.

All inference runs on CPU for reproducible, device-independent results.

Sections
--------
1. Dataset verification  --verify : FoV coverage + annotation-alignment stats
2. Algorithmic evaluation          : precision / recall on the test set
3. NN model AUC          --drow / --drspaam / --spacetime-cnn / --fullscan-tcn / ...
                                   : load checkpoint, compute AUC on CPU
4. Throughput benchmark  --bench   : forward / backward timing on synthetic data

Usage
-----
  # Evaluate DROW and DR-SPAAM using bundled published weights (no path needed)
  python evaluate.py --dataset drow --drow --drspaam

  # Evaluate trained models on FROG test set, also benchmark
  python evaluate.py --dataset frog --split test
                     --drspaam   checkpoints/drspaam.pth
                     --fullscan-tcn checkpoints/fstcn.pth
                     --bench

  # Benchmark only (no dataset needed)
  python evaluate.py --bench --no-eval

  # Dataset verification only
  python evaluate.py --dataset drow --verify --no-bench
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.detectors import (
    AlgorithmicDetector,
    DrowDetector,
    DrSpaamDetector,
    Li2FormerDetector,
    SpaceTimeCNNDetector,
    FullScanTCNDetector,
    TemporalUNetDetector,
    LFEPeaksDetector,
    LFEPPNDetector,
)
from follow_the_drow.utils.drow_utils import (
    laser_angles, laser_minimum, laser_maximum, laser_increment,
    project_cartesian_from_polar, standard_cartesian_to_project_cartesian, cutout, raw_scan,
    _prec_rec_2d, false_positive_rate, FP_THRESHOLDS,
)
from follow_the_drow.utils.tracking import (
    SimpleTracker, odom_xya_delta_to_tracker_frame,
)
from train import (
    _make_optimizer, _build_model, _default_args,
    evaluate_auc, load_checkpoint, _safe_auc,
)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset setup
# ──────────────────────────────────────────────────────────────────────────────

def _subsample_dataset(dataset, stride: int) -> None:
    """Systematically thin every sequence to every `stride`-th detection
    index, in place (det_id, det_wc/wa/wp, idet2iscan kept aligned).

    Why a stride and not a smaller dataset to begin with: FROG annotates
    every scan at 40 Hz (unlike DROW, where only ~5% of scans are labeled
    to begin with), so consecutive FROG frames are highly temporally
    correlated near-duplicates for AP purposes — evaluating all of them
    has sharply diminishing statistical value per frame, it just multiplies
    wall-clock time. A systematic stride decorrelates the evaluated sample
    (rather than e.g. truncating to a contiguous prefix, which would just
    narrow the time window covered) while cutting eval time by ~stride.
    """
    if stride <= 1:
        return
    for seq in range(len(dataset.det_id)):
        n = len(dataset.det_id[seq])
        keep = list(range(0, n, stride))
        old_idet2iscan = dataset.idet2iscan[seq]
        dataset.det_id[seq] = [dataset.det_id[seq][i] for i in keep]
        dataset.det_wc[seq] = [dataset.det_wc[seq][i] for i in keep]
        dataset.det_wa[seq] = [dataset.det_wa[seq][i] for i in keep]
        dataset.det_wp[seq] = [dataset.det_wp[seq][i] for i in keep]
        dataset.idet2iscan[seq] = {new_i: old_idet2iscan[old_i]
                                    for new_i, old_i in enumerate(keep)}


def _load_dataset(args):
    """Return (dataset, cfg)."""
    # Only relevant when evaluating a checkpoint trained with a non-default
    # --time-frame (e.g. the DROW long-stride experiments, Section 8) — the
    # dataset's own window size, not the model's, is what evaluate_auc()'s
    # get_scan() calls actually use (see train.py's evaluate_auc()). Default
    # 5 matches every existing checkpoint, so omitting this flag is a no-op.
    time_frame_size = getattr(args, "eval_time_frame", 5)
    if args.dataset == "frog":
        from follow_the_drow.datasets import FROG_Dataset, frog_laser_angles
        mode = getattr(args, "frog_mode", None) or "official"
        print(f"Loading FROG dataset (mode='{mode}', split='{args.split}') ...")
        ds = FROG_Dataset(split=args.split, mode=mode,
                          time_frame_size=time_frame_size)
        cfg = SimpleNamespace(
            name="frog",
            angles_fn=frog_laser_angles,
            fov_min=FROG_Dataset.LASER_MIN_ANGLE,
            fov_max=FROG_Dataset.LASER_MAX_ANGLE,
            laser_inc=FROG_Dataset.LASER_INCREMENT,
        )
    elif args.dataset == "jrdb":
        from follow_the_drow.datasets import JRDB_Dataset, jrdb_laser_angles
        print(f"Loading JRDB dataset (split='{args.split}') ...")
        print("  NOTE: JRDB requires manual download — see JRDB_Dataset docstring.")
        ds = JRDB_Dataset(split=args.split)
        cfg = SimpleNamespace(
            name="jrdb",
            angles_fn=jrdb_laser_angles,
            fov_min=JRDB_Dataset.LASER_MIN_ANGLE,
            fov_max=JRDB_Dataset.LASER_MAX_ANGLE,
            laser_inc=JRDB_Dataset.LASER_INCREMENT,
        )
    else:
        from follow_the_drow.datasets import DROW_Dataset
        print("Loading DROW test set ...")
        ds = DROW_Dataset(time_frame_size=time_frame_size)
        cfg = SimpleNamespace(
            name="drow",
            angles_fn=laser_angles,
            fov_min=laser_minimum,
            fov_max=laser_maximum,
            laser_inc=laser_increment,
        )
    n_frames = sum(len(d) for d in ds.det_id)
    print(f"  {len(ds.scan_id)} sequence(s), {n_frames} annotated frames\n")

    if getattr(args, "eval_stride", 1) > 1:
        _subsample_dataset(ds, args.eval_stride)
        n_kept = sum(len(d) for d in ds.det_id)
        print(f"  Subsampled to every {args.eval_stride}-th frame: "
              f"{n_kept} frames evaluated (of {n_frames})\n")

    return ds, cfg


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset verification
# ──────────────────────────────────────────────────────────────────────────────

def verify_dataset(dataset, cfg):
    """Print FoV coverage and annotation-alignment statistics."""
    fov_min, fov_max = cfg.fov_min, cfg.fov_max
    angles_fn = cfg.angles_fn

    # --- FoV coverage ---
    counts = {k: {"total": 0, "in_fov": 0} for k in ("wc", "wa", "wp")}
    for seq in range(len(dataset.det_id)):
        for det in range(len(dataset.det_id[seq])):
            for label, anns in [("wc", dataset.det_wc[seq][det]),
                                 ("wa", dataset.det_wa[seq][det]),
                                 ("wp", dataset.det_wp[seq][det])]:
                for r, phi in anns:
                    counts[label]["total"] += 1
                    if fov_min <= phi <= fov_max:
                        counts[label]["in_fov"] += 1

    print("=== Annotation field-of-view coverage ===")
    print(f"  Laser FoV: {np.degrees(fov_min):.1f} deg ... "
          f"{np.degrees(fov_max):.1f} deg  "
          f"({np.degrees(fov_max - fov_min):.1f} deg total)")
    print(f"  {'Class':<6} {'Total':>6}  {'In-FoV':>8}  {'Out-of-FoV':>12}")
    print("  " + "-" * 38)
    for label, s in counts.items():
        n, k = s["total"], s["in_fov"]
        if n == 0:
            print(f"  {label:<6} {'0':>6}")
        else:
            print(f"  {label:<6} {n:>6}  {k/n:>7.1%}  {(n-k)/n:>11.1%}")
    print()

    # --- Annotation alignment ---
    dists = {"wc": [], "wa": [], "wp": []}
    for seq in range(len(dataset.det_id)):
        angles = angles_fn(dataset.scans[seq].shape[1])
        for det in range(len(dataset.det_id[seq])):
            iscan = dataset.idet2iscan[seq][det]
            scan  = dataset.scans[seq][iscan]
            pts   = np.stack([scan * -np.sin(angles),
                               scan *  np.cos(angles)], axis=1)
            for label, anns in [("wc", dataset.det_wc[seq][det]),
                                 ("wa", dataset.det_wa[seq][det]),
                                 ("wp", dataset.det_wp[seq][det])]:
                for r, phi in anns:
                    gx, gy = project_cartesian_from_polar(r, phi)
                    dists[label].append(
                        float(np.min(np.linalg.norm(pts - [[gx, gy]], axis=1))))

    print("=== Annotation -> nearest LiDAR point distance ===")
    print(f"  {'Class':<6} {'Count':>6}  {'Mean':>7}  {'Median':>7}"
          f"  {'<0.3m':>7}  {'<0.5m':>7}")
    print("  " + "-" * 52)
    for label, d_list in dists.items():
        if not d_list:
            print(f"  {label:<6} {'0':>6}")
            continue
        d = np.array(d_list)
        print(f"  {label:<6} {len(d):>6}  {d.mean():>6.3f}m  "
              f"{np.median(d):>6.3f}m  "
              f"{(d < 0.3).mean():>6.1%}  {(d < 0.5).mean():>6.1%}")
    print()


def _as_rp_array(x) -> np.ndarray:
    """Normalize one class's per-frame annotation list to an (N, 2) array.

    Per-class annotation containers are inconsistently typed across
    datasets (DROW: object-dtype arrays; FROG: plain Python lists, empty
    `[]` for the classes it doesn't have). Concatenating them with `+`
    silently switches from list concatenation to numpy elementwise
    addition whenever a plain list meets a numpy array, and crashes on
    shape mismatch — normalize first instead.
    """
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1, 2) if arr.size else np.empty((0, 2))


def _all_classes_rp(dataset, seq: int, det_idx: int) -> np.ndarray:
    """Concatenate wc+wa+wp (r, phi) annotations for one frame, robustly."""
    return np.concatenate([
        _as_rp_array(dataset.det_wc[seq][det_idx]),
        _as_rp_array(dataset.det_wa[seq][det_idx]),
        _as_rp_array(dataset.det_wp[seq][det_idx]),
    ], axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Algorithmic detector evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_algorithmic(dataset, cfg, eval_r: float = 0.5) -> dict:
    """Run AlgorithmicDetector on the full dataset; return precision/recall."""
    from scipy.spatial.distance import cdist

    print("Running AlgorithmicDetector (CPU) ...")
    total_gt = total_det = total_match = total_fp = 0

    for seq in range(len(dataset.det_id)):
        algo = AlgorithmicDetector(verbose=False)
        for det in range(len(dataset.det_id[seq])):
            iscan = dataset.idet2iscan[seq][det]
            scans, odoms = dataset.get_scan(seq, iscan, algo.time_frame)
            det_xy = np.array(algo.forward_one(scans[-1], odoms[-1]["xya"]))
            if det_xy.ndim == 1:
                det_xy = det_xy.reshape(-1, 2)
            if det_xy.size > 0:
                det_xy = np.column_stack(standard_cartesian_to_project_cartesian(det_xy[:, 0], det_xy[:, 1]))

            all_ann = _all_classes_rp(dataset, seq, det)
            gt_xy = (np.array([project_cartesian_from_polar(r, p) for r, p in all_ann])
                     if len(all_ann) else np.empty((0, 2)))

            n_gt  = len(gt_xy)
            n_det = len(det_xy)
            n_match = n_fp = 0
            if n_det > 0 and n_gt > 0:
                matched = np.min(cdist(det_xy, gt_xy), axis=1) < eval_r
                n_match = int(matched.sum())
                n_fp    = n_det - n_match
            elif n_det > 0:
                n_fp = n_det

            total_gt    += n_gt
            total_det   += n_det
            total_match += n_match
            total_fp    += n_fp

    recall    = total_match / total_gt  if total_gt  > 0 else float("nan")
    precision = total_match / total_det if total_det > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else float("nan"))

    print(f"  GT annotations : {total_gt}")
    print(f"  Detections     : {total_det}")
    print(f"  Near-GT hits   : {total_match}")
    print(f"  False positives: {total_fp}")
    print(f"  Recall         : {recall:.1%}")
    print(f"  Precision      : {precision:.1%}")
    print(f"  F1             : {f1:.1%}")
    print()

    return dict(recall=recall, precision=precision, f1=f1,
                total_gt=total_gt, total_det=total_det)


# ──────────────────────────────────────────────────────────────────────────────
# 3. NN model AUC evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_nn_model(det_name: str, weights_path: Path,
                  dataset, cfg, eval_r: float = 0.5,
                  batch_size: int = 16, dtime: Optional[int] = None,
                  history_mode: str = "full",
                  diff_channels: bool = False, global_normalize: bool = False,
                  local_normalize_window: int = 0,
                  return_curves: bool = False,
                  return_detections: bool = False,
                  tracker_kwargs: Optional[dict] = None,
                  v2d_conf: Optional[dict] = None) -> dict:
    """Load checkpoint and compute AUC on CPU."""
    if det_name == "drspaam" and weights_path == DrSpaamDetector.DEFAULT_WEIGHTS:
        net = DrSpaamDetector.load_published()
        if dtime is None:
            dtime = 1          # published weights carry no dtime of their own
    else:
        # Full-scan architectures save their own capacity hyperparameters
        # (channels, n_spatial_stages, ...) in the checkpoint — read them
        # back rather than always building at _default_args()'s hardcoded
        # sizes, or a checkpoint saved with reduced capacity (e.g. the
        # overfitting-reduction experiments) would shape-mismatch on load.
        raw_ckpt = torch.load(weights_path, map_location="cpu")
        arch_keys = ("channels", "n_spatial_stages", "tcn_channels",
                    "backbone_channels", "unet_channels")
        overrides = {k: raw_ckpt[k] for k in arch_keys if k in raw_ckpt}
        if "n_time" in raw_ckpt:
            overrides["time_frame"] = raw_ckpt["n_time"]  # checkpoint key -> _default_args field name
        args = _default_args(detector=det_name, dataset=cfg.name, **overrides)
        net  = _build_model(args)
        net.load_state_dict(raw_ckpt["model"])
        if dtime is None:
            # The stride the weights were trained at, not evaluate_auc's old
            # hardcoded 1 -- see --eval-dtime's help and TODO.md A15.
            dtime = int(raw_ckpt.get("dtime", 1))
        print(f"  Loaded checkpoint from {weights_path}  (epoch {raw_ckpt.get('epoch', 0)}, "
              f"dtime {raw_ckpt.get('dtime', 'unset -> 1')}, "
              f"arch overrides: {overrides or 'none'})")
    net.eval()
    print(f"Evaluating {det_name} on CPU (eval batch_size={batch_size}, dtime={dtime}, "
          f"diff_channels={diff_channels}, global_normalize={global_normalize}, "
          f"history_mode={history_mode}, "
          f"local_normalize_window={local_normalize_window}, "
          f"tracker={'on' if tracker_kwargs is not None else 'off'}) ...")
    aucs = evaluate_auc(net, dataset, cfg, eval_r=eval_r, device="cpu",
                         batch_size=batch_size, dtime=dtime,
                         history_mode=history_mode,
                         diff_channels=diff_channels, global_normalize=global_normalize,
                         local_normalize_window=local_normalize_window,
                         return_curves=return_curves,
                         return_detections=return_detections,
                         tracker_kwargs=tracker_kwargs,
                         v2d_conf=v2d_conf)
    print(f"  Agnostic (any) : {aucs['agnostic']:.1%}")
    print(f"  Wheelchair(wc) : {aucs['wc']:.1%}")
    print(f"  Walker    (wa) : {aucs['wa']:.1%}")
    print(f"  Person    (wp) : {aucs['wp']:.1%}")
    print()
    return aucs


def _temporal_consensus_filter(detections, seq_id, cache_by_seq,
                               window: int = 5, min_agree: Optional[int] = None,
                               radius: float = 0.3, decay: float = 0.7,
                               weighting: str = "exp"):
    """
    Late-fusion temporal filter over a detector's own raw per-frame outputs —
    no retraining, works with any detector that returns (score, x, y) tuples.

    Keeps a rolling per-sequence cache of the last `window` frames' detections
    (deque, so it correctly resets at sequence boundaries via `cache_by_seq`).
    For each detection in the current frame, greedily nearest-neighbor matches
    it against each cached frame's own detections within `radius` (metres).
    If it was seen in at least `min_agree` of the last `window` frames
    (default: all of them), it survives; detections seen in fewer frames are
    dropped entirely — a single-frame flicker doesn't get reported at all.

    Score = time-weighted average of the matched per-frame scores, most
    recent frame weighted highest. age=0 is the current frame, age=1 is one
    frame back, etc. Two weighting shapes:
      "exp"    : weight(age) = decay**age — front-loaded (most of the
                 discount happens in the first step or two, then flattens);
                 the standard EMA/track-confirmation choice, and the only one
                 that admits a cheap O(1)-memory recursive update if this
                 ever moves from an eval-time pilot into the live ROS node.
      "linear" : weight(age) = window - age — discount spreads evenly across
                 the window; one fewer hyperparameter (no decay constant).
    A frame where the detection wasn't matched contributes no weight (it's
    simply absent from the average, not treated as a zero score) — matching
    this project's convention elsewhere of only ever comparing real
    observations.

    This is a pilot for testing whether a static "object" can be told apart
    from a "person" using persistence across several *independent* per-frame
    detections, as an alternative/complement to early-fusing raw scans before
    the network (what SpaceTimeCNN/FullScanTCN/TemporalUNet do).

    Note: matching is greedy nearest-neighbor per cached frame, not a proper
    multi-object tracker — fine for sparse single/few-person frames, may
    mismatch identities when several people are close together.

    Parameters
    ----------
    detections    : list[(score, x, y)] — current frame's raw detections
    seq_id        : sequence index, used as the cache key (resets across sequences)
    cache_by_seq  : dict[int, deque] — caller-owned, persists across calls
    window        : how many of the most recent frames (current included) to require agreement over
    min_agree     : minimum number of those frames a detection must appear in (default: window, i.e. all)
    radius        : match radius in metres
    decay         : "exp" weighting's per-frame decay in (0, 1] — 1.0 = plain mean, lower = more recency-biased
    weighting     : "exp" or "linear" — see above

    Returns: list[(score, x, y)] — filtered/rescored detections for this frame
    """
    assert weighting in ("exp", "linear"), f"weighting must be 'exp' or 'linear', got {weighting!r}"
    if min_agree is None:
        min_agree = window

    def _weight(age: int) -> float:
        return decay ** age if weighting == "exp" else max(window - age, 0)

    cache = cache_by_seq.setdefault(seq_id, deque(maxlen=window))
    cache.append(detections)

    out = []
    for score, x, y in detections:
        matched = []  # (weight, score) pairs
        for age, frame_dets in enumerate(reversed(cache)):  # age 0 = current frame
            best_score, best_d = None, radius
            for s2, x2, y2 in frame_dets:
                d = ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_score = d, s2
            if best_score is not None:
                matched.append((_weight(age), best_score))
        if len(matched) >= min_agree:
            wsum = sum(w for w, _ in matched)
            out.append((float(sum(w * s for w, s in matched) / wsum), x, y))
    return out


def _lfe_frames(dataset, seq, tracker_kwargs):
    """Which raw scans to run the detector over, and which of them to score.

    Returns ``(walk, scored)``: the scan indices to process in order, and a
    ``{scan index -> detection index}`` map of the ones the protocol scores.

    **A tracker is a deployment-time filter, so it must see every frame the
    sensor produced**, not just the frames the evaluation protocol scores.
    Feeding it only the scored frames breaks it twice over: under
    ``--eval-stride 5`` its time step silently becomes 5x longer than the
    ``dt=1.0`` it assumes, and on the `transferred` split -- where the scored
    frames are the person-free ones, interleaved in the raw recording with
    populated frames the split drops -- it would splice together frames that
    are minutes apart and call the result a track.

    Without a tracker the walk is exactly the scored set, so no number measured
    before this existed moves.
    """
    scored = {int(dataset.idet2iscan[seq][d]): d
              for d in range(len(dataset.det_id[seq]))}
    walk = (range(len(dataset.scans[seq])) if tracker_kwargs is not None
            else sorted(scored))
    return walk, scored


class _EgoMotion:
    """Per-frame ego-motion deltas for `SimpleTracker.step()`.

    FROG odometry is **not** zero, whatever `tracking.py`'s docstring used to
    say: 1.88 cm of translation and up to 0.4 deg of rotation per frame on the
    test recording. That claim dates from when `_load_or_fake_odom()`
    substituted all-zero odometry on any failure (TODO.md A14) -- it described
    the bug, not the robot. Over a track's ~6-frame lifetime 0.4 deg/frame is
    2.4 deg, which at 8 m range is 33 cm: most of the tracker's 0.5 m gate.
    """

    def __init__(self, odoms):
        self._odoms = odoms
        self._prev = None

    def step(self, iscan):
        xya = self._odoms[iscan]["xya"]
        theta, wx, wy = float(xya[2]), float(xya[0]), float(xya[1])
        if self._prev is None:
            self._prev = (theta, wx, wy)
        p_theta, p_wx, p_wy = self._prev
        self._prev = (theta, wx, wy)
        return (theta - p_theta,
                odom_xya_delta_to_tracker_frame(wx - p_wx, wy - p_wy, theta))


def eval_lfe_model(
    det_name:    str,
    detector,
    dataset,
    cfg,
    eval_r:      float = 0.5,
    consensus_window:   int = 1,
    consensus_min_agree: Optional[int] = None,
    consensus_radius:   float = 0.3,
    consensus_decay:    float = 0.7,
    consensus_weighting: str = "exp",
    return_curves:      bool = False,
    tracker_kwargs:     Optional[dict] = None,
) -> dict:
    """
    Evaluate an LFE-style detector (raw scan input, ONNX) and compute AUC.

    LFE detectors operate on full raw scan vectors rather than cutouts, so
    they bypass the standard evaluate_auc pipeline and use their own loop —
    but they still route through the shared `_prec_rec_2d` PR-curve code
    every other model uses, via the same per-frame (score, x, y) + GT
    layout `_process_detections` builds. An earlier version of this function
    computed recall as (found true positives) / (found true positives) —
    i.e. against its own detections rather than against the total ground
    truth — which silently ignored frames where the detector found nothing
    at all near a real person, structurally inflating AUC for any detector
    that fires sparingly. `_prec_rec_2d` (below) counts every GT annotation,
    including ones with zero matching detections, as a false negative.

    Returns the same dict as eval_nn_model:
      {"agnostic": float, "wc": float, "wa": float, "wp": float}
    """
    n_beams = dataset.scans[0].shape[1]
    angles  = cfg.angles_fn(n_beams)

    det_scores_wp,  det_xy_wp,  det_frame_wp  = [], [], []
    det_scores_any, det_xy_any, det_frame_any = [], [], []
    gt_xy_wp,  gt_frame_wp,  gt_r_wp  = [], [], []
    gt_xy_any, gt_frame_any, gt_r_any = [], [], []

    cache_by_seq: dict = {}
    fid = 0
    for seq in range(len(dataset.det_id)):
        walk, scored = _lfe_frames(dataset, seq, tracker_kwargs)
        tracker = SimpleTracker(**tracker_kwargs) if tracker_kwargs is not None else None
        ego = _EgoMotion(dataset.odoms[seq]) if tracker is not None else None

        for iscan in walk:
            scan = dataset.scans[seq][iscan]

            # Process this frame's GT even when detections is empty — a
            # frame with a real person and zero detections is a miss
            # (false negative), not a frame to skip.
            detections = detector.detect(scan, angles)
            if tracker is not None:
                ego_dtheta, ego_dxy = ego.step(iscan)
                detections = tracker.step(detections, ego_dtheta, ego_dxy)
            elif consensus_window > 1:
                detections = _temporal_consensus_filter(
                    detections, seq, cache_by_seq,
                    window=consensus_window, min_agree=consensus_min_agree,
                    radius=consensus_radius, decay=consensus_decay,
                    weighting=consensus_weighting)

            if iscan not in scored:
                continue            # tracked for continuity, not scored
            det_idx = scored[iscan]

            for score, x, y in detections:
                det_scores_wp.append(score);  det_xy_wp.append((x, y));  det_frame_wp.append(fid)
                det_scores_any.append(score); det_xy_any.append((x, y)); det_frame_any.append(fid)

            for r, p in _as_rp_array(dataset.det_wp[seq][det_idx]):
                gx, gy = project_cartesian_from_polar(r, p)
                gt_xy_wp.append((gx, gy)); gt_frame_wp.append(fid); gt_r_wp.append(eval_r)

            for r, p in _all_classes_rp(dataset, seq, det_idx):
                gx, gy = project_cartesian_from_polar(r, p)
                gt_xy_any.append((gx, gy)); gt_frame_any.append(fid); gt_r_any.append(eval_r)

            fid += 1

    def _curve(det_scores, det_xy, det_frame, gt_xy, gt_frame, gt_r):
        if not det_scores or not gt_xy:
            return None
        return _prec_rec_2d(
            np.array(det_scores, dtype=np.float32),
            np.array(det_xy, dtype=np.float32),
            np.array(det_frame),
            np.array(gt_xy, dtype=np.float32),
            np.array(gt_frame),
            np.array(gt_r, dtype=np.float32),
        )

    def _auc(curve):
        return _safe_auc(curve[0], curve[1]) if curve is not None else float("nan")

    wp_curve  = _curve(det_scores_wp,  det_xy_wp,  det_frame_wp,
                       gt_xy_wp, gt_frame_wp, gt_r_wp)
    any_curve = _curve(det_scores_any, det_xy_any, det_frame_any,
                       gt_xy_any, gt_frame_any, gt_r_any)

    aucs = {
        "agnostic": _auc(any_curve),
        "wc":       float("nan"),   # LFE is class-agnostic
        "wa":       float("nan"),
        "wp":       _auc(wp_curve),
    }
    if return_curves:
        aucs["wp_curve"] = wp_curve

    print(f"Evaluating {det_name} (ONNX) on CPU ...")
    print(f"  Agnostic (any) : {aucs['agnostic']:.1%}")
    print(f"  Person    (wp) : {aucs['wp']:.1%}  (wc/wa: n/a — class-agnostic)")
    print()
    return aucs


# ──────────────────────────────────────────────────────────────────────────────
# 3b. False-positive rate on a person-free split  (TODO.md A31)
# ──────────────────────────────────────────────────────────────────────────────

def _frame_period_s(dataset) -> Optional[float]:
    """Median seconds between consecutive raw scans, across every sequence.

    The median, not the mean: FROG recordings carry 60 pauses of up to 74.3 s,
    and a mean would let those set the frame rate. FROG's median comes out at
    ~1/26.2 s despite the sensor's 40 Hz nameplate, which is the rate a
    deployment would actually see.
    """
    diffs = [np.diff(np.asarray(t, dtype=np.float64))
             for t in dataset.scan_time if len(t) > 1]
    if not diffs:
        return None
    all_diffs = np.concatenate(diffs)
    all_diffs = all_diffs[all_diffs > 0]
    return float(np.median(all_diffs)) if len(all_diffs) else None


def _split_has_ground_truth(dataset) -> bool:
    """True if any scored frame carries an annotated person."""
    return any(len(wp) for seq in dataset.det_wp for wp in seq)


def _require_person_free(dataset, what: str) -> None:
    if _split_has_ground_truth(dataset):
        raise SystemExit(
            f"{what} counts every detection as a false positive, which is only "
            f"true on a split with no annotated people -- this one has some. "
            f"Use the wp-AUC path there (it already charges for false positives "
            f"in empty frames), or evaluate --frog-mode transferred."
        )


def eval_detector_false_positives(
    det_name: str, detector, dataset, cfg,
    thresholds=FP_THRESHOLDS, tracker_kwargs: Optional[dict] = None,
) -> dict:
    """Run a `detect(scan, angles)` detector over a person-free split and report
    how often it fires anyway.

    Works for the ONNX LFE models and `AlgorithmicDetector`; PyTorch
    checkpoints go through `eval_nn_false_positives()` instead, because their
    detections come out of the shared cutout/vote pipeline rather than a
    per-scan `detect()`.

    A low internal threshold on the detector is wanted here, not a problem:
    the sweep re-thresholds afterwards, so one pass reports every operating
    point. LFE-PPN's own default of 0.3 would silently truncate the sweep's
    lower half.
    """
    _require_person_free(dataset, f"{det_name}'s false-positive rate")

    n_beams = dataset.scans[0].shape[1]
    angles  = cfg.angles_fn(n_beams)

    det_scores, det_frames = [], []
    fid = 0
    for seq in range(len(dataset.det_id)):
        walk, scored = _lfe_frames(dataset, seq, tracker_kwargs)
        tracker = SimpleTracker(**tracker_kwargs) if tracker_kwargs is not None else None
        ego = _EgoMotion(dataset.odoms[seq]) if tracker is not None else None

        for iscan in walk:
            detections = detector.detect(dataset.scans[seq][iscan], angles)
            if tracker is not None:
                ego_dtheta, ego_dxy = ego.step(iscan)
                detections = tracker.step(detections, ego_dtheta, ego_dxy)
            if iscan not in scored:
                continue            # tracked for continuity, not scored
            for score, _x, _y in detections:
                det_scores.append(score)
                det_frames.append(fid)
            fid += 1

    stats = false_positive_rate(det_scores, det_frames, n_frames=fid,
                                frame_period_s=_frame_period_s(dataset),
                                thresholds=thresholds)
    print_false_positives(det_name, stats)
    return stats


def eval_nn_false_positives(
    det_name: str, weights_path: Path, dataset, cfg,
    batch_size: int = 16, dtime: Optional[int] = None,
    history_mode: str = "full", diff_channels: bool = False,
    global_normalize: bool = False, local_normalize_window: int = 0,
    thresholds=FP_THRESHOLDS, tracker_kwargs: Optional[dict] = None,
) -> dict:
    """The same measurement for a PyTorch checkpoint, reusing `evaluate_auc`'s
    detection pipeline so the detections counted here are exactly the ones its
    wp-AUC would have scored -- rather than a second, subtly different decoder
    written for this metric alone."""
    _require_person_free(dataset, f"{det_name}'s false-positive rate")

    res = eval_nn_model(det_name, weights_path, dataset, cfg,
                        batch_size=batch_size, dtime=dtime,
                        history_mode=history_mode,
                        diff_channels=diff_channels,
                        global_normalize=global_normalize,
                        local_normalize_window=local_normalize_window,
                        tracker_kwargs=tracker_kwargs,
                        return_detections=True)
    dets = res["detections"]
    stats = false_positive_rate(dets["scores"], dets["frames"],
                                n_frames=dets["n_frames"],
                                frame_period_s=_frame_period_s(dataset),
                                thresholds=thresholds)
    print_false_positives(det_name, stats)
    return stats


def print_false_positives(det_name: str, stats: dict) -> None:
    dur = stats["duration_s"]
    print(f"False positives for {det_name} on a person-free split")
    print(f"  {stats['n_frames']} frames"
          + (f", {dur / 60:.1f} min at {1 / stats['frame_period_s']:.1f} Hz"
             if stats["frame_period_s"] else ""))
    print(f"  {'thresh':>7} {'FP/frame':>10} {'FP/s':>8} {'frames w/ FP':>13}")
    for thresh, row in sorted(stats["per_threshold"].items()):
        print(f"  {thresh:>7.2f} {row['fp_per_frame']:>10.3f} "
              f"{row['fp_per_s']:>8.2f} {row['frame_fp_rate']:>12.1%}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Throughput benchmark (CPU only)
# ──────────────────────────────────────────────────────────────────────────────

def _fake_scans(n_beams: int, T: int):
    scans  = np.random.uniform(0.5, 10.0, (T, n_beams)).astype(np.float32)
    odoms  = [{"xya": np.array([0.0, 0.0, i * 0.01], dtype=np.float32)}
               for i in range(T)]
    return scans, odoms


def _fake_input(input_mode: str, n_beams: int, T: int, n_samp: int = 48) -> torch.Tensor:
    if input_mode == "cutout":
        return torch.randn(n_beams, T, n_samp)
    return torch.randn(n_beams, T, 1)


def _fake_targets(n_beams: int):
    return (torch.randint(0, 4, (n_beams,)),
            torch.randn(n_beams, 2))


def _time_fn(fn, warmup: int, iters: int) -> Tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(times[2:])
    return float(arr.mean()), float(arr.std())


def bench_preprocessing(n_beams: int, T: int,
                        warmup: int, iters: int) -> Dict[str, Tuple[float, float]]:
    scans, odoms = _fake_scans(n_beams, T)
    c_m, c_s = _time_fn(
        lambda: cutout(scans, odoms, n_beams, nsamp=48),
        warmup, iters)
    r_m, r_s = _time_fn(
        lambda: raw_scan(scans),
        warmup, iters)
    return {"cutout": (c_m, c_s), "raw_scan": (r_m, r_s)}


def bench_model(model: torch.nn.Module, input_mode: str,
                n_beams: int, T: int,
                warmup: int, iters: int) -> Dict[str, Tuple[float, float]]:
    """Benchmark one model on CPU (eval + train pass)."""
    model = model.to("cpu")
    n_samp = getattr(model, "N_SAMP", 48)
    x = _fake_input(input_mode, n_beams, T, n_samp)
    labels, vote_tgts = _fake_targets(n_beams)
    opt = _make_optimizer(model.parameters(), lr=1e-3,
                          weight_decay=1e-4, using_dml=False)

    # heatmap-head models (e.g. TemporalUNetDetector with head="heatmap")
    # return a 1-tuple (heatmap,), not the (logits, vpred) 2-tuple every
    # other head returns — branch on it instead of assuming 2 outputs.
    _head_type = getattr(model, "head_type", "drow")

    model.eval()
    def run_eval():
        with torch.no_grad():
            out = model(x)
        _ = out[0].reshape(-1)[0].item()

    e_m, e_s = _time_fn(run_eval, warmup, iters)

    model.train()
    def run_train():
        out = model(x)
        if _head_type == "heatmap":
            heatmap = out[0]
            loss = F.binary_cross_entropy_with_logits(
                heatmap.reshape(-1), (labels > 0).float())
        else:
            logits, vpred = out
            pos  = labels > 0
            lv   = (F.mse_loss(vpred[pos], vote_tgts[pos])
                    if pos.any() else vpred.new_tensor(0.0))
            if logits.shape[-1] == 1:
                # Binary logit output (e.g. Li2Former) — use BCE
                lc = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), (labels > 0).float())
            else:
                lc = F.cross_entropy(logits, labels)
            loss = lc + 0.02 * lv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        _ = loss.item()

    t_m, t_s = _time_fn(run_train, warmup, iters)
    return {"eval": (e_m, e_s), "train": (t_m, t_s)}


def _build_bench_models(T: int) -> List[Tuple[str, torch.nn.Module, str]]:
    return [
        ("DrowDetector",
            DrowDetector(dropout=0.5, time_frame_size=T, verbose=False),
            "cutout"),
        ("DrSpaamDetector",
            DrSpaamDetector(dropout=0.5, num_scans=T),
            "cutout"),
        ("Li2FormerDetector",
            Li2FormerDetector(dropout=0.5, num_scans=T),
            "cutout"),
        ("SpaceTimeCNN",
            SpaceTimeCNNDetector(n_time=T),
            "raw_scan"),
        ("FullScanTCN",
            FullScanTCNDetector(n_time=T),
            "raw_scan"),
        ("TemporalUNet",
            TemporalUNetDetector(n_time=T, head="heatmap"),
            "raw_scan"),
    ]


def run_benchmark(args):
    N  = args.n_beams
    T  = args.time_frame
    WU = args.warmup
    IT = args.iters

    print(f"Config: N_beams={N}, T={T}, warmup={WU}, iters={IT}, device=CPU\n")

    # Preprocessing
    print("=" * 60)
    print("  PREPROCESSING  (CPU)")
    print("=" * 60)
    prep = bench_preprocessing(N, T, WU, IT)
    for name, (m, s) in prep.items():
        fps = 1000.0 / m if m > 0 else 0
        print(f"  {name:<22} {m:6.1f} +-{s:4.1f} ms   {fps:6.0f} sc/s")
    print()

    models = _build_bench_models(T)
    col_n  = 24

    for mode_label, mode_key in [("EVAL  (no_grad forward)", "eval"),
                                  ("TRAIN (fwd + bwd + opt.step)", "train")]:
        print("=" * 60)
        print(f"  {mode_label}")
        print("=" * 60)
        print(f"  {'Model':<{col_n}}  {'ms/scan':>12}  {'sc/s':>8}")
        print("  " + "-" * 48)
        for name, model, input_mode in models:
            try:
                result = bench_model(model, input_mode, N, T, WU, IT)
                m, s   = result[mode_key]
                fps    = 1000.0 / m if m > 0 else 0
                print(f"  {name:<{col_n}}  {m:6.1f} +-{s:4.1f} ms  {fps:6.0f} sc/s")
            except Exception as exc:
                print(f"  {name:<{col_n}}  ERROR: {exc}")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def _pct(v):
    return f"{v:.1%}" if not np.isnan(v) else "  n/a  "


def print_summary(algo_stats: Optional[dict],
                  nn_results: Dict[str, dict]):
    if not algo_stats and not nn_results:
        return
    print("=" * 72)
    print("  EVALUATION SUMMARY")
    print("=" * 72)
    col = 26
    print(f"  {'Model':<{col}}  {'AUC(any)':>9}  {'AUC(wc)':>8}"
          f"  {'AUC(wa)':>8}  {'AUC(wp)':>8}")
    print("  " + "-" * 68)

    if algo_stats:
        rec  = _pct(algo_stats["recall"])
        prec = _pct(algo_stats["precision"])
        f1   = _pct(algo_stats["f1"])
        print(f"  {'AlgorithmicDetector':<{col}}  {'n/a':>9}  {'n/a':>8}"
              f"  {'n/a':>8}  {'n/a':>8}")
        print(f"    recall={rec}  prec={prec}  F1={f1}")

    for name, aucs in nn_results.items():
        print(f"  {name:<{col}}  "
              f"{_pct(aucs['agnostic']):>9}  "
              f"{_pct(aucs['wc']):>8}  "
              f"{_pct(aucs['wa']):>8}  "
              f"{_pct(aucs['wp']):>8}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

# PyTorch cutout/full-scan models evaluated via evaluate_auc
_NN_MODELS = {
    "drow":          "drow",
    "drspaam":       "drspaam",
    "li2former":     "li2former",
    "spacetime-cnn": "spacetime_cnn",
    "fullscan-tcn":  "fullscan_tcn",
    "temporal-unet": "temporal_unet",
}

# ONNX-based models evaluated via eval_lfe_model
_LFE_MODELS = {
    "lfe-peaks": ("lfe_peaks", LFEPeaksDetector),
    "lfe-ppn":   ("lfe_ppn",   LFEPPNDetector),
}


def _resolve_frog_mode(weight_paths) -> str:
    """Pick the FROG partition to evaluate on from the checkpoints themselves.

    A checkpoint stores the `frog_mode` it was trained under (train.py's
    `save_checkpoint`), and evaluating it under a different one is not a worse
    number -- it is a different experiment. `balanced`'s val even contains
    recordings `official` trains on, so the wrong default here would silently
    score a model on its own training data, which is the exact class of bug
    TODO.md A11 records.

    Two checkpoints trained under different partitions cannot share one
    evaluation pass, so that is an error rather than a guess. Checkpoints saved
    before the field existed carry nothing and fall back to `official`, which
    is what they were trained on.
    """
    found = {}
    for path in weight_paths:
        if path is None or not Path(path).exists():
            continue
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception:
            continue
        mode = ckpt.get("frog_mode")
        if mode:
            found.setdefault(mode, []).append(Path(path).name)

    if len(found) > 1:
        raise SystemExit(
            "Checkpoints disagree on their FROG partition: "
            + "; ".join(f"{m}: {', '.join(n)}" for m, n in sorted(found.items()))
            + ".\nThey cannot share one evaluation pass -- run them separately, "
              "or pass --frog-mode to override deliberately."
        )
    if found:
        mode = next(iter(found))
        print(f"  FROG partition '{mode}' taken from the checkpoint(s); "
              f"pass --frog-mode to override.")
        return mode
    return "official"


def print_false_positive_summary(fp_results: Dict[str, dict]) -> None:
    """One row per detector, one column per operating point."""
    if not fp_results:
        return
    thresholds = sorted(next(iter(fp_results.values()))["per_threshold"])
    n_frames = next(iter(fp_results.values()))["n_frames"]

    print("=" * 78)
    print(f"FALSE POSITIVES PER FRAME  ({n_frames} person-free frames; "
          f"every detection is a false positive by construction)")
    print("=" * 78)
    header = f"{'detector':<22}" + "".join(f"{('@' + f'{t:g}'):>11}" for t in thresholds)
    print(header)
    print("-" * len(header))
    for det_name, stats in fp_results.items():
        row = f"{det_name:<22}"
        for thresh in thresholds:
            row += f"{stats['per_threshold'][thresh]['fp_per_frame']:>11.3f}"
        print(row)
    print()
    print("Same numbers per second of operation:")
    for det_name, stats in fp_results.items():
        row = f"{det_name:<22}"
        for thresh in thresholds:
            row += f"{stats['per_threshold'][thresh]['fp_per_s']:>11.2f}"
        print(row)
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation + benchmark for all person detectors (CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset
    parser.add_argument("--dataset", choices=["drow", "frog", "jrdb"], default="drow",
                        help="Dataset to evaluate on (default: drow)")
    parser.add_argument("--split",   choices=["test", "train", "val"],
                        default="test",
                        help="Dataset split (default: test; DROW always uses test)")
    parser.add_argument("--frog-mode",
                        choices=["official", "transferred", "balanced"],
                        default=None,
                        help="FROG partition (TODO.md A30). Defaults to the one "
                             "stored in the checkpoint(s) being evaluated, i.e. the "
                             "partition the weights were actually trained under; "
                             "'official' for checkpoints saved before that field "
                             "existed, and for ONNX/algorithmic detectors, which "
                             "carry none. 'transferred' scores person-free frames "
                             "only -- wp-AUC is undefined there, so use --fp-rate. "
                             "Ignored for --dataset drow/jrdb.")
    parser.add_argument("--fp-thresholds", type=float, nargs="+",
                        default=list(FP_THRESHOLDS),
                        help="Confidence operating points for the false-positive "
                             "rate reported on a person-free split (default: "
                             "0.3 0.5 0.7 0.9). 0.3 is LFE-PPN's own published "
                             "threshold; 0.9 is where a phantom is least "
                             "excusable. A sweep rather than one number, because "
                             "a single operating point is a choice presented as a "
                             "measurement.")
    parser.add_argument("--lfe-ppn-score-thresh", type=float, default=0.01,
                        help="Objectness cut LFE-PPN is constructed with for "
                             "evaluation (default: 0.01). Its class default is "
                             "0.3, the authors' deployment operating point, and "
                             "that truncates the precision-recall curve AP is "
                             "integrated over: 64.0%% at 0.3 against 67.3%% at "
                             "0.01, on the same frames, against a published 66.5%%. "
                             "It must also sit at or below the lowest "
                             "--fp-thresholds value, or the detector discards "
                             "proposals the sweep meant to count.")
    parser.add_argument("--eval-r",  type=float, default=0.5,
                        help="Detection radius for AUC / precision-recall (default: 0.5 m)")
    parser.add_argument("--eval-stride", type=int, default=1,
                        help="Evaluate every Nth annotated frame instead of all of "
                             "them (systematic subsample, default: 1 = no "
                             "subsampling). Recommended for FROG (every scan "
                             "annotated at 40 Hz, so adjacent frames are highly "
                             "correlated near-duplicates for AP purposes) — leave "
                             "at 1 for DROW, which is already sparsely labeled "
                             "(~5%% of scans) in the source data.")
    parser.add_argument("--eval-batch-size", type=int, default=16,
                        help="Forward-pass batch size for the AUC evaluation loop "
                             "(default: 16). evaluate_auc()'s own default is 1 "
                             "(single-frame), which is fine for DROW's smaller test "
                             "split but impractically slow on FROG's ~120k-frame "
                             "test split (~3 fr/s at batch_size=1, ~11h/model).")
    parser.add_argument("--eval-dtime", type=int, default=None,
                        help="Stride (in raw scans) between the T frames in each "
                             "detector's temporal window. Defaults to the value "
                             "stored in the checkpoint being evaluated, i.e. the "
                             "stride the weights were actually trained at — pass "
                             "this flag only to deliberately evaluate off-"
                             "distribution. Scoring a dtime=10 model at dtime=1 "
                             "silently cost 2.5pp wp-AUC (76.3%% vs 78.8%%) before "
                             "the default stopped being a hardcoded 1 (TODO.md "
                             "A15). Checkpoints saved before that carry no dtime "
                             "and fall back to 1. Applies to --spacetime-cnn/"
                             "--fullscan-tcn/--temporal-unet/--drow/--drspaam/--li2former.")
    parser.add_argument("--eval-history-mode", default="full",
                        choices=["full", "zero", "shuffle", "frozen"],
                        help="Temporal ablation at evaluation time only, no "
                             "retraining. 'shuffle' permutes the history slots, "
                             "destroying ordering and direction while keeping "
                             "the set of observations and the input "
                             "distribution -- the sharpest test of whether the "
                             "model uses motion or merely aggregates several "
                             "looks at one scene. 'frozen' repeats the current "
                             "scan (static world, static robot). 'zero' blanks "
                             "the history and is **out of distribution** -- a "
                             "0 m range is physically impossible, and it "
                             "measured 42.0%% against 67.1%% full on the test "
                             "recording, 23pp below a genuinely single-frame "
                             "detector, which is how you can tell. See "
                             "TODO.md A21.")
    parser.add_argument("--eval-time-frame", type=int, default=5,
                        help="Number of scans in the temporal window the dataset "
                             "builds via get_scan() (default: 5, matching every "
                             "existing checkpoint). Set to match --time-frame T used "
                             "when *training* the checkpoint being evaluated (e.g. "
                             "the DROW long-stride experiments used --time-frame 10) "
                             "— evaluate_auc() reads dataset.time_frame, not the "
                             "model's own n_time attribute, for window size.")
    parser.add_argument("--eval-diff-channels", action="store_true", default=False,
                        help="Must match --diff-channels used when *training* the "
                             "checkpoint being evaluated (default: False) — a "
                             "representation choice, not an augmentation, so eval "
                             "has to reproduce it exactly. Applies to --spacetime-cnn/"
                             "--fullscan-tcn/--temporal-unet.")
    parser.add_argument("--eval-global-normalize", action="store_true", default=False,
                        help="Must match --global-normalize used when *training* the "
                             "checkpoint being evaluated (default: False) — a "
                             "representation choice, not an augmentation. Applies to "
                             "--spacetime-cnn/--fullscan-tcn/--temporal-unet.")
    parser.add_argument("--eval-local-normalize-window", type=int, default=0,
                        help="Must match --local-normalize-window used when *training* "
                             "the checkpoint being evaluated (default: 0 = off) — a "
                             "representation choice, not an augmentation. Applies to "
                             "--spacetime-cnn/--fullscan-tcn/--temporal-unet.")
    parser.add_argument("--consensus-window", type=int, default=1,
                        help="LFE only: late-fusion temporal consensus over the last "
                             "N frames' own detections (default: 1 = off, i.e. normal "
                             "single-frame behaviour). A detection must be seen within "
                             "--consensus-radius in --consensus-min-agree (default: "
                             "all N) of the last N frames to be reported; its score "
                             "becomes the mean of the matched per-frame scores. No "
                             "retraining needed — post-processes the detector's own "
                             "raw per-frame output. See _temporal_consensus_filter().")
    parser.add_argument("--consensus-min-agree", type=int, default=None,
                        help="Minimum number of the last --consensus-window frames a "
                             "detection must appear in to survive (default: all of them).")
    parser.add_argument("--consensus-radius", type=float, default=0.3,
                        help="Match radius in metres for --consensus-window (default: 0.3).")
    parser.add_argument("--consensus-decay", type=float, default=0.7,
                        help="Per-frame recency weight decay for --consensus-window's "
                             "score when --consensus-weighting=exp (default: 0.7 — "
                             "weight(age)=decay**age, age=0 is the current frame). "
                             "1.0 = plain mean of matched scores; lower = more "
                             "recency-biased. Ignored when --consensus-weighting=linear.")
    parser.add_argument("--consensus-weighting", choices=["exp", "linear"], default="exp",
                        help="Shape of --consensus-window's recency weighting (default: "
                             "exp). 'exp': weight(age)=decay**age — front-loaded discount, "
                             "and the only shape with a cheap O(1)-memory recursive update "
                             "for a live deployment. 'linear': weight(age)=window-age — "
                             "discount spreads evenly, no decay constant to pick.")
    parser.add_argument("--tracker", action="store_true",
                        help="LFE only: use the SimpleTracker (constant-velocity, "
                             "Hungarian-matched, predict/confirm/coast) instead of "
                             "--consensus-window's static-position averaging. Takes "
                             "precedence over --consensus-window if both are given. "
                             "See library/follow_the_drow/utils/tracking.py.")
    parser.add_argument("--tracker-radius", type=float, default=0.5,
                        help="SimpleTracker match gate in metres (default: 0.5).")
    parser.add_argument("--tracker-min-hits", type=int, default=3,
                        help="SimpleTracker: real matches needed before a track is "
                             "reported at all (default: 3) — false-positive suppression.")
    parser.add_argument("--tracker-max-age", type=int, default=3,
                        help="SimpleTracker: consecutive missed frames a confirmed track "
                             "survives, coasting on its predicted position, before being "
                             "dropped (default: 3) — false-negative recovery window.")
    parser.add_argument("--tracker-vel-alpha", type=float, default=0.5,
                        help="SimpleTracker velocity-estimate smoothing in (0,1] "
                             "(default: 0.5).")
    parser.add_argument("--tracker-coast-decay", type=float, default=0.8,
                        help="SimpleTracker: per-miss score decay while coasting "
                             "(default: 0.8; 1.0 = no decay).")

    # What to run
    parser.add_argument("--verify",  action="store_true",
                        help="Print dataset verification stats (FoV + alignment)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip all evaluation (dataset + models)")
    parser.add_argument("--no-bench", action="store_true",
                        help="Skip throughput benchmark")

    # NN model checkpoints
    # --drow and --drspaam accept an optional path; when given without a path
    # they use the bundled published paper weights (downloaded at pip install time).
    parser.add_argument("--drow", nargs="?", type=Path,
                        const=DrowDetector.DEFAULT_WEIGHTS, default=None,
                        metavar="WEIGHTS",
                        help="Checkpoint for drow; omit value to use bundled paper weights")
    parser.add_argument("--drspaam", nargs="?", type=Path,
                        const=DrSpaamDetector.DEFAULT_WEIGHTS, default=None,
                        metavar="WEIGHTS",
                        help="Checkpoint for drspaam; omit value to use bundled paper weights")
    for flag in [f for f in _NN_MODELS if f not in ("drow", "drspaam")]:
        parser.add_argument(f"--{flag}", type=Path, default=None, metavar="WEIGHTS",
                            help=f"Checkpoint for {flag} (enables AUC evaluation)")

    # LFE ONNX models — accept optional path; default to bundled ONNX weights
    parser.add_argument("--lfe-peaks", nargs="?", type=Path,
                        const=LFEPeaksDetector.DEFAULT_WEIGHTS, default=None,
                        metavar="ONNX",
                        help="LFE-Peaks ONNX model; omit value to use bundled weights")
    parser.add_argument("--lfe-ppn", nargs="?", type=Path,
                        const=LFEPPNDetector.DEFAULT_WEIGHTS, default=None,
                        metavar="ONNX",
                        help="LFE-PPN ONNX model; omit value to use bundled weights")

    # Benchmark options
    parser.add_argument("--n-beams",    type=int, default=450,
                        help="Beams per scan for benchmark (450=DROW, 720=FROG; default: 450)")
    parser.add_argument("--time-frame", type=int, default=5,
                        help="Temporal window T for benchmark (default: 5)")
    parser.add_argument("--warmup",     type=int, default=5,
                        help="Warm-up iterations before timing (default: 5)")
    parser.add_argument("--iters",      type=int, default=20,
                        help="Timed iterations per model (default: 20)")

    args = parser.parse_args()

    # Collect which PyTorch NN models to evaluate
    nn_weights = {
        _NN_MODELS[flag]: getattr(args, flag.replace("-", "_"))
        for flag in _NN_MODELS
        if getattr(args, flag.replace("-", "_")) is not None
    }

    # Collect which LFE ONNX models to evaluate
    lfe_weights = {
        det_name: (cls, getattr(args, flag.replace("-", "_")))
        for flag, (det_name, cls) in _LFE_MODELS.items()
        if getattr(args, flag.replace("-", "_")) is not None
    }

    run_eval  = not args.no_eval
    run_bench = not args.no_bench

    if args.dataset == "frog" and args.frog_mode is None:
        args.frog_mode = _resolve_frog_mode(nn_weights.values())

    # ── Load dataset (only if needed) ─────────────────────────────────────────
    dataset = cfg = None
    if run_eval:
        dataset, cfg = _load_dataset(args)

    algo_stats = None
    nn_results = {}
    fp_results = {}

    if run_eval and dataset is not None:
        # A person-free split (--frog-mode transferred) cannot be scored with a
        # precision-recall metric at all: no true positives means precision 0 at
        # every threshold and recall with no denominator. So the metric switches
        # with the data rather than with a flag nobody remembers to pass, and
        # every detection is counted as what it provably is (TODO.md A31).
        person_free = not _split_has_ground_truth(dataset)
        if person_free:
            print("=== Person-free split: reporting false-positive rate, "
                  "not AUC ===\n")

        # 1. Dataset verification
        if args.verify:
            print("=== Dataset Verification ===\n")
            verify_dataset(dataset, cfg)

        # 2. Algorithmic detector
        if person_free:
            print("=== AlgorithmicDetector: skipped ===\n"
                  "  Its precision/recall path needs ground truth, and it has no "
                  "detect(scan, angles) entry point to count firings through.\n")
        else:
            print("=== AlgorithmicDetector ===\n")
            algo_stats = eval_algorithmic(dataset, cfg, eval_r=args.eval_r)

        _tracker_kwargs = dict(
            match_radius=args.tracker_radius,
            min_hits=args.tracker_min_hits,
            max_age=args.tracker_max_age,
            vel_alpha=args.tracker_vel_alpha,
            coast_decay=args.tracker_coast_decay,
        ) if args.tracker else None

        # 3. NN models (cutout / full-scan, PyTorch)
        for det_name, weights_path in nn_weights.items():
            print(f"=== {det_name} ===\n")
            try:
                if person_free:
                    fp_results[det_name] = eval_nn_false_positives(
                        det_name, weights_path, dataset, cfg,
                        batch_size=args.eval_batch_size, dtime=args.eval_dtime,
                        history_mode=args.eval_history_mode,
                        diff_channels=args.eval_diff_channels,
                        global_normalize=args.eval_global_normalize,
                        local_normalize_window=args.eval_local_normalize_window,
                        thresholds=args.fp_thresholds,
                        tracker_kwargs=_tracker_kwargs)
                else:
                    nn_results[det_name] = eval_nn_model(
                        det_name, weights_path, dataset, cfg, eval_r=args.eval_r,
                        batch_size=args.eval_batch_size, dtime=args.eval_dtime,
                        history_mode=args.eval_history_mode,
                        diff_channels=args.eval_diff_channels,
                        global_normalize=args.eval_global_normalize,
                        local_normalize_window=args.eval_local_normalize_window,
                        tracker_kwargs=_tracker_kwargs)
            except Exception as exc:
                print(f"  [ERROR] {exc}\n")

        # 4. LFE ONNX models
        for det_name, (cls, onnx_path) in lfe_weights.items():
            print(f"=== {det_name} ===\n")
            try:
                # A low objectness cut, for both paths: it keeps the whole
                # PR curve available to AP and every operating point available
                # to the false-positive sweep. See --lfe-ppn-score-thresh.
                detector = cls(onnx_path=onnx_path,
                               **({"score_thresh": args.lfe_ppn_score_thresh}
                                  if cls is LFEPPNDetector else {}))
                if person_free:
                    fp_results[det_name] = eval_detector_false_positives(
                        det_name, detector, dataset, cfg,
                        thresholds=args.fp_thresholds,
                        tracker_kwargs=_tracker_kwargs)
                else:
                    nn_results[det_name] = eval_lfe_model(
                        det_name, detector, dataset, cfg, eval_r=args.eval_r,
                        consensus_window=args.consensus_window,
                        consensus_min_agree=args.consensus_min_agree,
                        consensus_radius=args.consensus_radius,
                        consensus_decay=args.consensus_decay,
                        consensus_weighting=args.consensus_weighting,
                        tracker_kwargs=_tracker_kwargs)
            except Exception as exc:
                print(f"  [ERROR] {exc}\n")

        # Summary
        if person_free:
            print_false_positive_summary(fp_results)
        else:
            print_summary(algo_stats, nn_results)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    if run_bench:
        print("=== Throughput Benchmark (CPU) ===\n")
        run_benchmark(args)


if __name__ == "__main__":
    main()
