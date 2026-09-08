#!/usr/bin/env python3
"""
Source-level NMS/vote-clustering sweep for the two models that don't benefit
from operating-threshold selection alone (utils/error_breakdown.py's
precision-target analysis): FullScanTCN and LFE-PPN.

Motivation: the error breakdown showed false positives dominate for every
model; the threshold sweep showed SpaceTimeCNN/TemporalUNet/LFE-Peaks all
have a favorable FP-for-FN trade at a stricter operating point, but
FullScanTCN and LFE-PPN don't — their precision/recall curves are shaped
differently, so picking a different point on the *same* curve doesn't help.
This sweeps the parameters that actually generate that curve in the first
place:

  FullScanTCN (and any vote-based model): `votes_to_detections()`'s
  `vote_collect_radius` (how far apart votes get merged into one detection)
  and `min_thresh` (minimum vote-grid density to report anything at all).

  LFE-PPN: its own constructor already exposes `score_thresh` (minimum
  objectness) and `nms_radius` (greedy NMS suppression radius) — no need to
  touch internal code, they're already parameters.

Loads the dataset once and loops over the grid in-process (each config is a
fresh model/detector forward pass, not a fresh Python process) — much
cheaper than N separate `evaluate.py` invocations for a small sweep.

Usage
-----
  python nms_sweep.py --dataset frog --eval-stride 40
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.detectors import LFEPPNDetector  # noqa: E402
from evaluate import eval_nn_model, eval_lfe_model, _load_dataset, _subsample_dataset  # noqa: E402
from error_breakdown import _best_f1_point, _breakdown, _total_gt  # noqa: E402

_FULLSCAN_TCN_CKPT = Path("../checkpoints/fullscan_tcn.best.pth")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["drow", "frog"], default="frog")
    p.add_argument("--split", default="test")
    p.add_argument("--eval-r", type=float, default=0.5)
    p.add_argument("--eval-stride", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--skip", nargs="*", default=[], choices=["fullscan_tcn", "lfe_ppn"])
    p.add_argument("--radius-grid", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                   help="vote_collect_radius values to try (FullScanTCN)")
    p.add_argument("--min-thresh-grid", type=float, nargs="+", default=[1e-3, 5e-3, 1e-2],
                   help="min_thresh values to try (FullScanTCN)")
    p.add_argument("--score-thresh-grid", type=float, nargs="+", default=[0.3, 0.4, 0.5],
                   help="score_thresh values to try (LFE-PPN)")
    p.add_argument("--nms-radius-grid", type=float, nargs="+", default=[0.5, 0.8, 1.2],
                   help="nms_radius values to try (LFE-PPN)")
    args = p.parse_args()

    class _Args:
        dataset = args.dataset
        split = args.split
    dataset, cfg = _load_dataset(_Args())
    if args.eval_stride > 1:
        _subsample_dataset(dataset, args.eval_stride)

    total_gt = _total_gt(dataset)
    print(f"\nTotal wp GT annotations in evaluated set: {total_gt}\n")

    # --- FullScanTCN: vote_collect_radius x min_thresh -------------------
    if "fullscan_tcn" not in args.skip:
        print(f"\n{'='*100}\n  FullScanTCN — vote_collect_radius x min_thresh sweep\n{'='*100}\n")
        print(f"  {'radius':>7} {'min_thresh':>11} {'AUC':>7} {'Recall':>8} {'Prec':>8} "
              f"{'F1':>7} {'TP':>8} {'FP':>8} {'FN':>8}")
        base_v2d = {"blur_sigma": 2.0, "blur_win": 11, "bin_size": 0.1}
        best = None
        for radius, min_thresh in product(args.radius_grid, args.min_thresh_grid):
            v2d = dict(base_v2d, vote_collect_radius=radius, min_thresh=min_thresh)
            aucs = eval_nn_model("fullscan_tcn", _FULLSCAN_TCN_CKPT, dataset, cfg,
                                 eval_r=args.eval_r, batch_size=args.eval_batch_size,
                                 return_curves=True, v2d_conf=v2d)
            curve = aucs.get("wp_curve")
            if curve is None:
                continue
            r, prec, f1 = _best_f1_point(curve)
            tp, fp, fn = _breakdown(r, prec, total_gt)
            print(f"  {radius:>7.1f} {min_thresh:>11.4f} {aucs['wp']:>6.1%} {r:>7.1%} "
                  f"{prec:>7.1%} {f1:>6.1%} {tp:>8.0f} {fp:>8.0f} {fn:>8.0f}")
            if best is None or f1 > best[0]:
                best = (f1, radius, min_thresh, aucs["wp"])
        if best:
            print(f"\n  Best F1: radius={best[1]}, min_thresh={best[2]} -> "
                  f"F1={best[0]:.1%}, AUC={best[3]:.1%}\n")

    # --- LFE-PPN: score_thresh x nms_radius -------------------------------
    if "lfe_ppn" not in args.skip:
        print(f"\n{'='*100}\n  LFE-PPN — score_thresh x nms_radius sweep\n{'='*100}\n")
        print(f"  {'score_th':>9} {'nms_rad':>8} {'AUC':>7} {'Recall':>8} {'Prec':>8} "
              f"{'F1':>7} {'TP':>8} {'FP':>8} {'FN':>8}")
        best = None
        for score_thresh, nms_radius in product(args.score_thresh_grid, args.nms_radius_grid):
            detector = LFEPPNDetector(score_thresh=score_thresh, nms_radius=nms_radius)
            aucs = eval_lfe_model("lfe_ppn", detector, dataset, cfg, eval_r=args.eval_r,
                                  return_curves=True)
            curve = aucs.get("wp_curve")
            if curve is None:
                continue
            r, prec, f1 = _best_f1_point(curve)
            tp, fp, fn = _breakdown(r, prec, total_gt)
            print(f"  {score_thresh:>9.2f} {nms_radius:>8.2f} {aucs['wp']:>6.1%} {r:>7.1%} "
                  f"{prec:>7.1%} {f1:>6.1%} {tp:>8.0f} {fp:>8.0f} {fn:>8.0f}")
            if best is None or f1 > best[0]:
                best = (f1, score_thresh, nms_radius, aucs["wp"])
        if best:
            print(f"\n  Best F1: score_thresh={best[1]}, nms_radius={best[2]} -> "
                  f"F1={best[0]:.1%}, AUC={best[3]:.1%}\n")


if __name__ == "__main__":
    main()
