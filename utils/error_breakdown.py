#!/usr/bin/env python3
"""
Error breakdown: is FROG accuracy limited by false positives or false
negatives?

Motivation: before building any temporal-consensus mechanism (naive
averaging, or a proper tracker), we need to know which error type actually
dominates each detector's output — a false-positive filter and a
false-negative recovery mechanism are different pieces of machinery, and
building the wrong one first is wasted effort. See docs/RESEARCH.md Section
8's consensus subsection for the full discussion.

Method: reuses the AUC pipeline's own (recall, precision, threshold) curve
(evaluate_auc(..., return_curves=True) / eval_lfe_model(..., return_curves=True))
instead of a separate detection pass. Picks the best-F1 operating point on
that curve, then backs out TP/FP/FN counts from (recall, precision, total_gt)
— no new inference code, no new matching logic, just a different read of the
data every AUC number in this project's tables is already computed from.

  tp = round(recall * total_gt)
  fn = total_gt - tp
  fp = tp * (1/precision - 1)   [precision = tp / (tp + fp)]

Usage
-----
  python error_breakdown.py --dataset frog --eval-stride 40
  python error_breakdown.py --dataset frog --skip lfe_ppn
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.detectors import LFEPeaksDetector, LFEPPNDetector  # noqa: E402
from evaluate import (  # noqa: E402
    eval_nn_model, eval_lfe_model, _load_dataset, _subsample_dataset,
)

_NN_CKPTS = {
    "spacetime_cnn": Path("../checkpoints/spacetime_cnn.best.pth"),
    "fullscan_tcn":  Path("../checkpoints/fullscan_tcn.best.pth"),
    "temporal_unet": Path("../checkpoints/temporal_unet.best.pth"),
}
_LFE_MODELS = {
    "lfe_peaks": LFEPeaksDetector,
    "lfe_ppn":   LFEPPNDetector,
}


def _best_f1_point(curve):
    """(recs, precs, threshs) -> (recall, precision, f1) at max F1.

    Returns (0.0, 0.0, 0.0) for a curve with no valid points at all — e.g. a
    threshold/min_thresh strict enough that literally zero detections
    survive across the whole evaluated set. That's a real (if useless)
    operating point a sweep can land on, not a crash.
    """
    recs, precs, threshs = curve
    mask = ~np.isnan(recs) & ~np.isnan(precs)
    r, p = recs[mask], precs[mask]
    if len(r) == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * r * p / np.clip(r + p, 1e-12, None)
    i = int(np.argmax(f1))
    return r[i], p[i], f1[i]


def _at_precision_targets(curve, targets, total_gt):
    """
    For each target precision, find the highest-recall point on the curve
    that still meets it (recall is monotonically non-decreasing as the
    threshold sweep accepts more detections, so "highest recall among points
    with precision >= target" is well-defined). Returns a list of
    (target, recall, precision, tp, fp, fn) — target entries with no
    qualifying point (precision never reaches that high) are omitted.
    """
    recs, precs, _ = curve
    mask = ~np.isnan(recs) & ~np.isnan(precs)
    r, p = recs[mask], precs[mask]
    out = []
    for target in targets:
        qualifying = np.where(p >= target)[0]
        if len(qualifying) == 0:
            continue
        i = qualifying[np.argmax(r[qualifying])]
        tp, fp, fn = _breakdown(r[i], p[i], total_gt)
        out.append((target, r[i], p[i], tp, fp, fn))
    return out


def _breakdown(recall: float, precision: float, total_gt: int):
    tp = recall * total_gt
    fn = total_gt - tp
    fp = tp * (1.0 / precision - 1.0) if precision > 0 else float("nan")
    return tp, fp, fn


def _total_gt(dataset) -> int:
    return sum(len(dataset.det_wp[seq][i])
              for seq in range(len(dataset.det_id))
              for i in range(len(dataset.det_id[seq])))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["drow", "frog"], default="frog")
    p.add_argument("--split", default="test")
    p.add_argument("--eval-r", type=float, default=0.5)
    p.add_argument("--eval-stride", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--skip", nargs="*", default=[],
                   choices=list(_NN_CKPTS) + list(_LFE_MODELS))
    args = p.parse_args()

    class _Args:
        dataset = args.dataset
        split = args.split
    dataset, cfg = _load_dataset(_Args())
    if args.eval_stride > 1:
        _subsample_dataset(dataset, args.eval_stride)

    total_gt = _total_gt(dataset)
    print(f"\nTotal wp (person) GT annotations in evaluated set: {total_gt}\n")

    rows = []
    for det, ckpt in _NN_CKPTS.items():
        if det in args.skip or not ckpt.exists():
            continue
        aucs = eval_nn_model(det, ckpt, dataset, cfg, eval_r=args.eval_r,
                             batch_size=args.eval_batch_size, return_curves=True)
        curve = aucs.get("wp_curve")
        if curve is None:
            continue
        r, prec, f1 = _best_f1_point(curve)
        tp, fp, fn = _breakdown(r, prec, total_gt)
        rows.append((det, aucs["wp"], r, prec, f1, tp, fp, fn, curve))

    for det, cls in _LFE_MODELS.items():
        if det in args.skip:
            continue
        detector = cls()
        aucs = eval_lfe_model(det, detector, dataset, cfg, eval_r=args.eval_r,
                              return_curves=True)
        curve = aucs.get("wp_curve")
        if curve is None:
            continue
        r, prec, f1 = _best_f1_point(curve)
        tp, fp, fn = _breakdown(r, prec, total_gt)
        rows.append((det, aucs["wp"], r, prec, f1, tp, fp, fn, curve))

    print(f"\n{'='*100}")
    print(f"  Error breakdown at each model's own best-F1 operating point (wp / person class)")
    print(f"{'='*100}\n")
    print(f"  {'Model':<16} {'AUC':>7} {'Recall':>8} {'Prec':>8} {'F1':>7} "
          f"{'TP':>8} {'FP':>8} {'FN':>8}  {'dominant error'}")
    print(f"  {'-'*16} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")
    for det, auc, r, prec, f1, tp, fp, fn, _curve in rows:
        dominant = "FN (misses)" if fn > fp else "FP (false alarms)"
        ratio = (fn / fp) if fp > 0 else float("inf")
        print(f"  {det:<16} {auc:>6.1%} {r:>7.1%} {prec:>7.1%} {f1:>6.1%} "
              f"{tp:>8.0f} {fp:>8.0f} {fn:>8.0f}  {dominant} ({ratio:.1f}x)")
    print()

    print(f"\n{'='*100}")
    print(f"  Operating-point sweep: what does trading recall for precision cost, per model?")
    print(f"  (moving to a stricter threshold than each model's own best-F1 point)")
    print(f"{'='*100}\n")
    targets = [0.75, 0.80, 0.85, 0.90, 0.95]
    for det, auc, bf1_r, bf1_p, f1, bf1_tp, bf1_fp, bf1_fn, curve in rows:
        print(f"  {det}  (best-F1: recall={bf1_r:.1%} precision={bf1_p:.1%} "
              f"FP={bf1_fp:.0f} FN={bf1_fn:.0f})")
        points = _at_precision_targets(curve, targets, total_gt)
        if not points:
            print(f"    (never reaches {targets[0]:.0%} precision on this curve)\n")
            continue
        print(f"    {'target prec':>12} {'recall':>8} {'actual prec':>12} "
              f"{'TP':>8} {'FP':>8} {'FN':>8}  {'FP saved':>9} {'FN added':>9}")
        for target, r, p, tp, fp, fn in points:
            fp_saved = bf1_fp - fp
            fn_added = fn - bf1_fn
            print(f"    {target:>11.0%} {r:>8.1%} {p:>11.1%} "
                  f"{tp:>8.0f} {fp:>8.0f} {fn:>8.0f}  {fp_saved:>+9.0f} {fn_added:>+9.0f}")
        print()


if __name__ == "__main__":
    main()
