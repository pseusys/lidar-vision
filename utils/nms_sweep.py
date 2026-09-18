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

  Both LFE detectors: the **range-aware** merge radius, `r(d) = a + b*d`
  (`--skip fullscan_tcn lfe_ppn` to run only this). Tier 0 measured the single
  scalar failing in both directions at 0.30 m -- ~5 pp of LFE-Peaks' recall
  spent merging adjacent people into one detection, while LFE-PPN emits 1.609
  duplicates per frame at the same value (`memory/static-detector-diagnosis.md`,
  TODO.md A38). This section sweeps `(a, b)` and scores every cell at **both**
  association distances the FROG benchmark reports, because a 0.5 m gate cannot
  resolve a change of this size -- the mistake the arc-offset calibration
  already made once (`CHANGELOG.md`, 2026-09-10).

  It caches raw centroids rather than re-running the network: the ONNX forward
  does not depend on the radius, so one pass per detector serves the whole grid
  and both association distances. Without that the default grid is ~640k
  detector calls instead of ~20k.

  **The cache is validated, not assumed.** Replaying cached centroids through
  `_merge_nearby` at the detector's own radius must reproduce the detector's own
  AP, and it does -- to 0.00pp for both models at `--raw-score-floor 0.01`.
  At 0.05 it does not: LFE-PPN loses 0.49pp, because AP integrates a PR curve
  whose tail those proposals are. That is what the floor's default is set from.

Usage
-----
  python nms_sweep.py --dataset frog --eval-stride 40
  python nms_sweep.py --skip fullscan_tcn lfe_ppn --eval-stride 5    # just A38
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.detectors import LFEPeaksDetector, LFEPPNDetector  # noqa: E402
from follow_the_drow.detectors.lfe_detector import _merge_nearby  # noqa: E402
from follow_the_drow.utils.drow_utils import project_cartesian_from_polar  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
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
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["fullscan_tcn", "lfe_ppn", "range_aware"])
    p.add_argument("--radius-grid", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                   help="vote_collect_radius values to try (FullScanTCN)")
    p.add_argument("--min-thresh-grid", type=float, nargs="+", default=[1e-3, 5e-3, 1e-2],
                   help="min_thresh values to try (FullScanTCN)")
    p.add_argument("--score-thresh-grid", type=float, nargs="+", default=[0.3, 0.4, 0.5],
                   help="score_thresh values to try (LFE-PPN)")
    p.add_argument("--nms-radius-grid", type=float, nargs="+", default=[0.5, 0.8, 1.2],
                   help="nms_radius values to try (LFE-PPN)")
    p.add_argument("--intercept-grid", type=float, nargs="+",
                   default=[0.10, 0.15, 0.20, 0.25, 0.30],
                   help="a in r(d) = a + b*d (range-aware section)")
    p.add_argument("--slope-grid", type=float, nargs="+",
                   default=[0.0, 0.015, 0.03, 0.045, 0.06],
                   help="b in r(d) = a + b*d (range-aware section)")
    p.add_argument("--assoc-grid", type=float, nargs="+", default=[0.5, 0.3],
                   help="association distances to score every cell at")
    p.add_argument("--range-aware-models", nargs="+", default=["LFE-Peaks", "LFE-PPN"],
                   choices=["LFE-Peaks", "LFE-PPN"],
                   help="which detectors the range-aware section runs on")
    p.add_argument("--score-by", choices=["ap", "operating-point"], default="ap",
                   help="`ap` sweeps against average precision (A38). "
                        "`operating-point` sweeps against what Tier 0 measured "
                        "-- recall, duplicates, phantoms and close-pair misses "
                        "at a fixed threshold (A40). The two disagree about the "
                        "direction of this parameter, which is the point")
    p.add_argument("--op-thresh", type=float, default=0.3,
                   help="the fixed operating point for --score-by operating-point")
    p.add_argument("--raw-score-floor", type=float, default=0.01,
                   help="centroids below this are dropped before caching. 0.01 is "
                        "not a guess: the re-merger was checked against the real "
                        "detectors on identical frames, and 0.01 reproduces both "
                        "to 0.00pp while 0.05 costs LFE-PPN 0.49pp of AP. Raising "
                        "it trades correctness for speed -- do not, without "
                        "re-running that check")
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


    # --- both LFE models: range-aware radius r(d) = a + b*d ---------------
    if "range_aware" not in args.skip:
        _range_aware_sweep(args, dataset, cfg, total_gt)


def _cache_raw(detector, dataset, cfg, floor):
    """One forward pass per detector, keyed by the scan itself.

    `_ReMerger` below replays these through `_merge_nearby` at each (a, b), so
    the network runs once for the whole grid. Keyed on the scan bytes rather
    than a frame counter because `eval_lfe_model` is free to walk the dataset
    more than once and a counter would silently desynchronise.
    """
    angles = cfg.angles_fn(dataset.scans[0].shape[1])
    raw = {}
    for seq in range(len(dataset.det_id)):
        for det_idx in range(len(dataset.det_id[seq])):
            scan = dataset.scans[seq][int(dataset.idet2iscan[seq][det_idx])]
            key = hash(scan.tobytes())
            if key not in raw:
                raw[key] = [d for d in detector.detect(scan, angles) if d[0] >= floor]
    return raw


class _ReMerger:
    """A detector-shaped object that re-merges cached centroids at a new (a, b)."""

    def __init__(self, raw, radius, slope):
        self._raw, self._radius, self._slope = raw, radius, slope

    def detect(self, scan, angles):
        return _merge_nearby(self._raw[hash(scan.tobytes())],
                             self._radius, self._slope)


def _range_aware_sweep(args, dataset, cfg, total_gt):
    models = {
        "LFE-Peaks": lambda: LFEPeaksDetector(merge_radius=0.0),
        "LFE-PPN":   lambda: LFEPPNDetector(score_thresh=0.01, nms_radius=0.0),
    }
    models = {k: v for k, v in models.items() if k in args.range_aware_models}
    if args.score_by == "operating-point":
        return _operating_point_sweep(args, dataset, cfg, models)
    for name, make in models.items():
        print(f"\n{'='*100}\n  {name} — range-aware radius r(d) = a + b*d\n{'='*100}\n")
        raw = _cache_raw(make(), dataset, cfg, args.raw_score_floor)
        n_raw = sum(len(v) for v in raw.values())
        print(f"  cached {n_raw} raw centroids over {len(raw)} frames "
              f"({n_raw / max(len(raw), 1):.1f}/frame, score >= {args.raw_score_floor})\n")

        header = f"  {'a':>6} {'b':>7} {'r(2m)':>7} {'r(8m)':>7}"
        for assoc in args.assoc_grid:
            header += f" | {'AP@' + format(assoc, '.1f'):>8} {'Recall':>7} {'Prec':>7}"
        print(header)

        best = {}
        for a, b in product(args.intercept_grid, args.slope_grid):
            row = f"  {a:>6.2f} {b:>7.3f} {a + 2 * b:>7.2f} {a + 8 * b:>7.2f}"
            for assoc in args.assoc_grid:
                aucs = eval_lfe_model(name, _ReMerger(raw, a, b), dataset, cfg,
                                      eval_r=assoc, return_curves=True)
                curve = aucs.get("wp_curve")
                r, prec, _f1 = _best_f1_point(curve) if curve else (0.0, 0.0, 0.0)
                row += f" | {aucs['wp']:>7.1%} {r:>6.1%} {prec:>6.1%}"
                if assoc not in best or aucs["wp"] > best[assoc][0]:
                    best[assoc] = (aucs["wp"], a, b)
            print(row, flush=True)

        for assoc, (wp, a, b) in sorted(best.items(), reverse=True):
            print(f"\n  Best AP@{assoc:.1f}: a={a:.2f}, b={b:.3f} "
                  f"-> r(2m)={a + 2 * b:.2f} m, r(8m)={a + 8 * b:.2f} m, AP={wp:.1%}",
                  flush=True)
        print(flush=True)


def _frame_metrics(dets, gt, assoc):
    """One frame -> (tp, duplicates, phantoms, close_pair_misses).

    `tp`         ground-truth people with a detection assigned one-to-one.
    `duplicate`  a detection near a person, beyond the one assigned to them.
    `phantom`    a detection near no annotated person at all.
    `close_pair` a MISSED person who nonetheless had a detection within
                 `assoc` -- it went to a neighbour instead. This is the
                 quantity `range_probe.py` called "stolen", and the one the
                 merge radius trades against duplicates.

    `tp + duplicates + phantoms == len(dets)` by construction, so a radius
    change cannot move detections into an uncounted category and look like an
    improvement. `tests/test_operating_point_metrics.py` pins all of it.
    """
    n_d, n_g = len(dets), len(gt)
    if n_d == 0:
        return 0, 0, 0, 0
    if n_g == 0:
        return 0, 0, n_d, 0
    cost = np.linalg.norm(dets[:, None, :] - gt[None, :, :], axis=-1)
    phantom = int((cost.min(axis=1) > assoc).sum())
    gated = np.where(cost > assoc, 1e6, cost)
    rows, cols = linear_sum_assignment(gated)
    ok = gated[rows, cols] < 1e6
    tp = int(ok.sum())
    assigned = set(int(c) for c in cols[ok])
    close = int(sum(1 for j in range(n_g)
                    if j not in assigned and cost[:, j].min() <= assoc))
    return tp, n_d - tp - phantom, phantom, close


def _gt_by_frame(dataset):
    """{scan-bytes hash -> (G, 2) ground-truth people}, keyed like the cache."""
    out = {}
    for seq in range(len(dataset.det_id)):
        for det_idx in range(len(dataset.det_id[seq])):
            iscan = int(dataset.idet2iscan[seq][det_idx])
            g = np.array([project_cartesian_from_polar(r, phi)
                          for r, phi in dataset.det_wp[seq][det_idx]],
                         dtype=float).reshape(-1, 2)
            out[hash(dataset.scans[seq][iscan].tobytes())] = g
    return out


def _operating_point_sweep(args, dataset, cfg, models):
    """A38's grid, scored on Tier 0's quantities instead of AP (A40).

    A38 found AP wanting *more* merging, out to 0.45 m and past the value that
    reproduces the published number, while Tier 0 measured over-merging costing
    ~5 pp of recall. Both are right about their own quantity: a duplicate is a
    full false positive, a close-pair miss is only one missed GT, because the
    surviving merged detection still matches one of the two people. So AP
    under-charges exactly the error Tier 0 found, and A38's verdict is a verdict
    about AP. This asks the other question.
    """
    gt_by_frame = _gt_by_frame(dataset)
    for name, make in models.items():
        print(f"\n{'='*104}\n  {name} — radius swept on Tier 0's quantities, "
              f"score >= {args.op_thresh}\n{'='*104}\n")
        raw = _cache_raw(make(), dataset, cfg, args.raw_score_floor)
        n_frames = len(raw)
        n_gt = sum(len(gt_by_frame[k]) for k in raw)
        print(f"  {n_frames} frames, {n_gt} annotated people, "
              f"{sum(len(v) for v in raw.values()) / n_frames:.1f} raw centroids/frame\n")

        # Merge once per (a, b) and score at every association distance from
        # the same detections -- the merge is the expensive half for LFE-PPN,
        # and it does not depend on the association gate.
        rows = {assoc: [] for assoc in args.assoc_grid}
        for a, b in product(args.intercept_grid, args.slope_grid):
            acc = {assoc: [0, 0, 0, 0] for assoc in args.assoc_grid}
            for key, cents in raw.items():
                d = np.array([[x, y] for sc, x, y in
                              _merge_nearby(cents, a, b) if sc >= args.op_thresh],
                             dtype=float).reshape(-1, 2)
                g = gt_by_frame[key]
                for assoc in args.assoc_grid:
                    t, u, p_, c = _frame_metrics(d, g, assoc)
                    acc[assoc][0] += t; acc[assoc][1] += u
                    acc[assoc][2] += p_; acc[assoc][3] += c
            for assoc, (tp, dup, phantom, close) in acc.items():
                fp, fn = dup + phantom, n_gt - tp
                rows[assoc].append(
                    f"  {a:>6.2f} {b:>7.3f} {a + 2 * b:>7.2f} {a + 8 * b:>7.2f} | "
                    f"{tp / n_gt:>6.1%} {dup / n_frames:>7.3f} "
                    f"{phantom / n_frames:>8.3f} {close / n_frames:>9.3f} "
                    f"{fp / n_frames:>7.3f} {fn / n_frames:>7.3f} "
                    f"{fp / max(fn, 1):>7.2f}")
            print(f"  done a={a:.2f} b={b:.3f}", flush=True)

        for assoc in args.assoc_grid:
            print(f"\n  association {assoc:.1f} m")
            print(f"  {'a':>6} {'b':>7} {'r(2m)':>7} {'r(8m)':>7} | {'recall':>7} "
                  f"{'dup/fr':>7} {'phan/fr':>8} {'close/fr':>9} {'FP/fr':>7} "
                  f"{'FN/fr':>7} {'FP:FN':>7}")
            for line in rows[assoc]:
                print(line)
            print(flush=True)


if __name__ == "__main__":
    main()
