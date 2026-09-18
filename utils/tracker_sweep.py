#!/usr/bin/env python3
"""
Sweep a SORT-style tracker over a published detector's per-frame output, and
report what temporal association alone buys — on both a populated split and a
person-free one (docs/PAPER.md, "Primitive temporal tracking").

The question is the one docs/PAPER.md is built around: **can temporal
information reduce false positives without giving up detections?** A tracker is
the cheapest possible answer — no retraining, no architecture change, the
published weights untouched.

Which tracker, stated precisely because the paper cites it
(tests/test_tracker.py pins every clause):

  `--tracker sort` (the default) is **SORT** (Bewley et al., ICIP 2016) with its
  track management followed exactly -- confirmation on `min_hits` *consecutive*
  matches, coasted tracks not reported, the early-frame grace period kept --
  and a constant-velocity Kalman filter. It deviates in only two ways, both
  forced by the setting rather than chosen: association on Euclidean distance
  rather than IoU (point detections have no boxes, and circle IoU is a monotone
  function of centre distance anyway), and ego-motion compensation (SORT assumes
  a static camera; this robot moves).

  `--tracker simple` is `SimpleTracker`, a deployment-oriented variant that
  confirms on *total* rather than consecutive hits and *reports* coasted tracks.
  Both differences matter here: the first lets a flickering phantom confirm
  eventually, the second lets a tracker emit detections on frames where the
  detector found nothing. Useful as an ablation, but it is not SORT.

Two implementation points that decide whether the numbers mean anything:

* **The tracker sees every raw frame**, at the sensor's full rate, with real
  odometry-based ego-motion compensation; only the scored subset reaches the
  metric. Feeding it the scored frames instead would make its `dt=1.0`
  assumption wrong under `--eval-stride`, and on `transferred` would splice
  together person-free frames minutes apart (`memory/interpreting-evaluation.md`).
* **Detections are cached to disk once per (model, split)** and replayed
  through each tracker configuration, so a ten-point sweep costs one detection
  pass rather than ten.

Usage
-----
    python tracker_sweep.py                       # the paper's sweep
    python tracker_sweep.py --min-hits 1 5 20     # a custom one
    python tracker_sweep.py --no-cache            # force re-detection
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.datasets.frog_dataset import FROG_Dataset, frog_laser_angles
from follow_the_drow.detectors import LFEPeaksDetector, LFEPPNDetector
from follow_the_drow.utils.drow_utils import (
    _prec_rec_2d, project_cartesian_from_polar, false_positive_rate,
)
from follow_the_drow.utils.tracking import SimpleTracker, SortTracker
from evaluate import _subsample_dataset, _frame_period_s, _EgoMotion
from train import _safe_auc


MODELS = {
    "LFE-Peaks": lambda: LFEPeaksDetector(),
    # A low objectness cut so the swept operating points are all reachable;
    # its class default of 0.3 is a deployment threshold, not an eval one.
    "LFE-PPN":   lambda: LFEPPNDetector(score_thresh=0.01),
}


def cache_detections(ds, detector, angles, cache_path, use_cache=True):
    """One detection pass over every raw frame of every sequence, in order.

    Cached to disk because the sweep replays it many times and the pass itself
    is the only expensive part (~195k frames on `transferred`).
    """
    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    out = []
    for seq in range(len(ds.det_id)):
        scored = {int(ds.idet2iscan[seq][d]): d for d in range(len(ds.det_id[seq]))}
        per_frame = [detector.detect(ds.scans[seq][i], angles)
                     for i in range(len(ds.scans[seq]))]
        out.append((seq, scored, per_frame))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    return out


def replay(ds, cache, tracker_kwargs, tracker_cls=SortTracker):
    """Replay cached detections through one tracker configuration."""
    scored_out = []
    for seq, scored, per_frame in cache:
        tracker = tracker_cls(**tracker_kwargs) if tracker_kwargs else None
        ego = _EgoMotion(ds.odoms[seq]) if tracker else None
        for iscan in range(len(per_frame)):
            dets = per_frame[iscan]
            if tracker is not None:
                dtheta, dxy = ego.step(iscan)
                dets = tracker.step(list(dets), dtheta, dxy)
            if iscan in scored:
                scored_out.append((seq, scored[iscan], dets))
    return scored_out


def score_populated(ds, replayed, n_gt, thresh, eval_r):
    """AP over the full curve, plus TP/FP/FN at one operating threshold.

    FP and FN are derived from the same precision-recall curve the AP comes
    from, rather than from a second matching pass that could disagree with it:
        recall    = TP / (TP + FN)   ->  FN = n_gt - TP
        precision = TP / (TP + FP)   ->  FP = TP * (1 - p) / p
    """
    d_s, d_xy, d_f, g_xy, g_f = [], [], [], [], []
    fid = -1
    for fid, (seq, det_idx, dets) in enumerate(replayed):
        for s, x, y in dets:
            d_s.append(s); d_xy.append((x, y)); d_f.append(fid)
        for r, p in ds.det_wp[seq][det_idx]:
            gx, gy = project_cartesian_from_polar(r, p)
            g_xy.append((gx, gy)); g_f.append(fid)
    n_frames = fid + 1
    recs, precs, threshs = _prec_rec_2d(
        np.array(d_s, np.float32), np.array(d_xy, np.float32), np.array(d_f),
        np.array(g_xy, np.float32), np.array(g_f),
        np.full(len(g_f), eval_r, np.float32))
    idx = np.nonzero(np.asarray(threshs) >= thresh)[0]
    i = int(idx[-1]) if len(idx) else 0
    rec, prec = float(recs[i]), float(precs[i])
    tp = rec * n_gt
    fp = tp * (1 - prec) / prec if prec > 0 else float("nan")
    return dict(ap=_safe_auc(recs, precs), fp_per_frame=fp / n_frames,
                fn_per_frame=(n_gt - tp) / n_frames, miss_rate=1 - rec,
                fp_fn=fp / (n_gt - tp) if n_gt - tp > 0 else float("nan"),
                n_frames=n_frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-hits", type=int, nargs="+",
                    default=[1, 2, 3, 5, 10, 15, 20, 30, 50],
                    help="min_hits values to sweep (default: the paper's sweep). "
                         "A track must accumulate this many matched frames "
                         "before it is reported at all.")
    ap.add_argument("--tracker", choices=["sort", "simple"], default="sort",
                    help="Which tracker (default: sort, the citeable one). "
                         "'simple' confirms on total rather than consecutive "
                         "hits and reports coasted tracks -- see the module "
                         "docstring.")
    ap.add_argument("--max-age", type=int, default=1,
                    help="Frames a track may go unmatched before being "
                         "dropped (default: 1, SORT's own default)")
    ap.add_argument("--match-radius", type=float, default=0.5,
                    help="Association gate in metres (default: 0.5)")
    ap.add_argument("--thresh", type=float, default=0.3,
                    help="Operating threshold the FP/FN columns are read at "
                         "(default: 0.3, LFE-PPN's own published one)")
    ap.add_argument("--eval-r", type=float, default=0.5,
                    help="Association distance for AP (default: 0.5 m). FROG "
                         "reports AP at 0.5 and 0.3; always say which.")
    ap.add_argument("--stride", type=int, default=5,
                    help="Score every Nth frame of the populated split "
                         "(default: 5). The tracker still sees every frame.")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(__file__).parent / ".tracker_cache",
                    help="Where per-(model, split) detection caches live")
    ap.add_argument("--no-cache", action="store_true",
                    help="Re-run detection even if a cache exists")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write results as JSON here as well as printing them")
    args = ap.parse_args()

    angles = frog_laser_angles(720)
    cfg = SimpleNamespace(name="frog", angles_fn=frog_laser_angles,
                          fov_min=FROG_Dataset.LASER_MIN_ANGLE,
                          fov_max=FROG_Dataset.LASER_MAX_ANGLE,
                          laser_inc=FROG_Dataset.LASER_INCREMENT)

    ds_off = FROG_Dataset(split="test", mode="official",
                          auto_download=False, verbose=False)
    period = _frame_period_s(ds_off)
    _subsample_dataset(ds_off, args.stride)
    n_gt = sum(len(wp) for seq in ds_off.det_wp for wp in seq)
    ds_tr = FROG_Dataset(split="test", mode="transferred",
                         auto_download=False, verbose=False)

    print(f"official    : {sum(len(d) for d in ds_off.det_id)} scored frames "
          f"(every {args.stride}th), {n_gt} people, "
          f"{sum(len(s) for s in ds_off.scans)} raw frames tracked")
    print(f"transferred : {sum(len(d) for d in ds_tr.det_id)} scored person-free "
          f"frames, {sum(len(s) for s in ds_tr.scans)} raw frames tracked")
    tracker_cls = SortTracker if args.tracker == "sort" else SimpleTracker
    print(f"tracker     : {args.tracker} ({tracker_cls.__name__}), "
          f"match_radius={args.match_radius} max_age={args.max_age}, "
          f"threshold {args.thresh}, AP at d={args.eval_r} m\n")

    configs = [("none", None)] + [
        (str(m), dict(match_radius=args.match_radius, min_hits=m,
                      max_age=args.max_age))
        for m in args.min_hits]
    results = {}

    for model_name, make in MODELS.items():
        print(f"########## {model_name} ##########", flush=True)
        det = make()
        cache_off = cache_detections(
            ds_off, det, angles,
            args.cache_dir / f"{model_name}_official_s{args.stride}.pkl",
            use_cache=not args.no_cache)
        cache_tr = cache_detections(
            ds_tr, det, angles, args.cache_dir / f"{model_name}_transferred.pkl",
            use_cache=not args.no_cache)

        print(f"  {'min_hits':>9} {'AP':>7} {'FP/frame':>9} {'FN/frame':>9} "
              f"{'FP:FN':>7} {'miss':>7} | {'FP/frame':>9} {'frames w/FP':>12}")
        print(f"  {'':>9} {'':>7} {'(official)':>9} {'(official)':>9} "
              f"{'':>7} {'':>7} | {'(transf.)':>9} {'(transf.)':>12}")
        for label, tk in configs:
            pop = score_populated(ds_off, replay(ds_off, cache_off, tk, tracker_cls),
                                  n_gt, args.thresh, args.eval_r)
            t_s, t_f, fid_t = [], [], -1
            for fid_t, (_seq, _di, dets) in enumerate(
                    replay(ds_tr, cache_tr, tk, tracker_cls)):
                for s, _x, _y in dets:
                    t_s.append(s); t_f.append(fid_t)
            fpr = false_positive_rate(t_s, t_f, n_frames=fid_t + 1,
                                      frame_period_s=period,
                                      thresholds=(args.thresh,))
            row = fpr["per_threshold"][args.thresh]
            results[f"{model_name}|{label}"] = dict(
                **pop, fp_per_frame_transferred=row["fp_per_frame"],
                frame_fp_rate_transferred=row["frame_fp_rate"],
                fp_per_s_transferred=row["fp_per_s"])
            print(f"  {label:>9} {pop['ap']:>6.1%} {pop['fp_per_frame']:>9.3f} "
                  f"{pop['fn_per_frame']:>9.3f} {pop['fp_fn']:>7.2f} "
                  f"{pop['miss_rate']:>6.1%} | {row['fp_per_frame']:>9.3f} "
                  f"{row['frame_fp_rate']:>11.1%}", flush=True)
        print(flush=True)

    if args.out:
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
