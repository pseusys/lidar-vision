#!/usr/bin/env python3
"""
Characterise a detector's false positives on FROG: are they furniture or noise,
and on a populated split how many are genuine phantoms rather than duplicates?

Two measurements, both feeding docs/PAPER.md's theory section.

**Persistence** (`--persistence`). Two mechanisms predict the same false-positive
*rate* and opposite temporal behaviour:

  furniture   a chair's legs look like a person in a single slice, so the same
              phantom appears in the same place frame after frame -> PERSISTENT
  noise       the detector fires on whatever momentarily resembles a leg pair,
              at an unstable location -> TRANSIENT

Which one it is decides whether temporal reasoning can remove them at all: a
persistent phantom is invisible to a confirm-then-report tracker, a transient
one is exactly what such a tracker removes. Measured on unbroken person-free
runs at the sensor's own rate; ego-motion between consecutive frames is ~1.9 cm,
an order of magnitude under the association radius, so no compensation is
applied and none is needed.

**Phantom vs duplicate** (`--breakdown`). The paper claims the false-positive
rate barely depends on whether anyone is present. Comparing the raw FP columns
understates that, because on a populated split a *second* detection on an
already-matched person counts as a false positive, and on a person-free split no
such duplicate can exist. The like-for-like quantity is the false positives that
are not near any person at all.

Usage
-----
    python phantom_analysis.py                  # both measurements
    python phantom_analysis.py --persistence    # just the first
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.datasets.frog_dataset import FROG_Dataset, frog_laser_angles
from follow_the_drow.detectors import LFEPeaksDetector, LFEPPNDetector
from follow_the_drow.utils.drow_utils import project_cartesian_from_polar
from evaluate import _subsample_dataset


MODELS = {
    "LFE-Peaks": lambda: LFEPeaksDetector(),
    "LFE-PPN":   lambda: LFEPPNDetector(score_thresh=0.01),
}


def _xy(dets, thresh):
    return np.array([[x, y] for s, x, y in dets if s >= thresh],
                    dtype=np.float32).reshape(-1, 2)


def persistence(args, angles):
    """How often does a false positive recur in the very next frame?"""
    ds = FROG_Dataset(split="test", mode="transferred",
                      auto_download=False, verbose=False)
    runs = []
    for seq in range(len(ds.det_id)):
        ids = np.asarray(ds.det_id[seq], dtype=np.int64)
        breaks = np.nonzero(np.diff(ids) != 1)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks + 1, [len(ids)]))
        best = int(np.argmax(ends - starts))
        runs.append((seq, ids[starts[best]:ends[best]][:args.run_frames]))
    print(f"person-free runs: " + ", ".join(f"seq {s} x{len(r)}" for s, r in runs))
    print(f"association radius {args.radius} m, threshold {args.thresh}\n")

    for name, make in MODELS.items():
        det = make()
        matched = total = 0
        per_frame = []
        for seq, run in runs:
            prev = None
            for i in run:
                xy = _xy(det.detect(ds.scans[seq][int(i)], angles), args.thresh)
                per_frame.append(len(xy))
                if prev is not None:
                    if len(xy) and len(prev):
                        d = np.linalg.norm(xy[:, None, :] - prev[None, :, :], axis=-1)
                        matched += int((d.min(axis=1) <= args.radius).sum())
                    total += len(xy)
                prev = xy
        print(f"{name:<12} {np.mean(per_frame):5.2f} phantoms/frame   "
              f"{matched}/{total} = {matched / max(total, 1):5.1%} recur next frame",
              flush=True)

    # The reference point that makes those numbers readable.
    ds_pop = FROG_Dataset(split="test", mode="official",
                          auto_download=False, verbose=False)
    matched = total = 0
    for k, i in enumerate(np.asarray(ds_pop.det_id[0], dtype=np.int64)[:args.run_frames]):
        xy = np.array([project_cartesian_from_polar(r, p)
                       for r, p in ds_pop.det_wp[0][k]], dtype=np.float32).reshape(-1, 2)
        if k and len(xy):
            if len(prev_gt):
                d = np.linalg.norm(xy[:, None, :] - prev_gt[None, :, :], axis=-1)
                matched += int((d.min(axis=1) <= args.radius).sum())
            total += len(xy)
        prev_gt = xy
    print(f"{'real people':<12} {'':>5}                {matched}/{total} = "
          f"{matched / max(total, 1):5.1%} recur next frame   <- the reference")


def breakdown(args, angles):
    """Split a populated split's false positives into duplicates and phantoms."""
    ds = FROG_Dataset(split="test", mode="official",
                      auto_download=False, verbose=False)
    _subsample_dataset(ds, args.stride)
    n_frames = sum(len(d) for d in ds.det_id)
    print(f"official test, {n_frames} scored frames (every {args.stride}th), "
          f"threshold {args.thresh}, association {args.assoc} m\n")

    for name, make in MODELS.items():
        det = make()
        dup = phantom = matched = n_gt = 0
        for seq in range(len(ds.det_id)):
            for det_idx in range(len(ds.det_id[seq])):
                iscan = ds.idet2iscan[seq][det_idx]
                gt = [project_cartesian_from_polar(r, p)
                      for r, p in ds.det_wp[seq][det_idx]]
                n_gt += len(gt)
                d = _xy(det.detect(ds.scans[seq][iscan], angles), args.thresh)
                if not len(d):
                    continue
                if gt:
                    g = np.asarray(gt, dtype=np.float32)
                    near = np.linalg.norm(
                        d[:, None, :] - g[None, :, :], axis=-1).min(1) <= args.assoc
                else:
                    near = np.zeros(len(d), dtype=bool)
                n_near = int(near.sum())
                # One detection per person, by Hungarian assignment gated at
                # `assoc`. The obvious shortcut -- tp = min(n_near, len(gt)) --
                # is wrong and was wrong here until 2026-09-11: it cannot tell
                # "one detection each on five people" from "five detections on
                # one person", so it scores a pile of duplicates as recall. That
                # stayed invisible while LFE-PPN's NMS radius was 0.8 m and it
                # emitted 0.004 duplicates per frame; at the calibrated 0.30 m
                # it emits 1.388, and the shortcut put LFE-PPN's miss rate at
                # 5.0% against the 19.3% an assignment-based count gives.
                # `genuine phantoms` -- detections near no person at all -- is
                # unaffected either way, which is why the paper's headline
                # number survives the correction.
                tp_here = 0
                if gt and n_near:
                    cost = np.linalg.norm(d[:, None, :] - g[None, :, :], axis=-1)
                    gated = np.where(cost > args.assoc, 1e6, cost)
                    rows, cols = linear_sum_assignment(gated)
                    tp_here = int((gated[rows, cols] < 1e6).sum())
                matched += tp_here
                dup += n_near - tp_here
                phantom += int((~near).sum())

        print(f"{name}")
        print(f"  matched (TP)       {matched:8d}  {matched / n_frames:6.3f} /frame")
        print(f"  duplicates         {dup:8d}  {dup / n_frames:6.3f} /frame")
        print(f"  genuine phantoms   {phantom:8d}  {phantom / n_frames:6.3f} /frame"
              f"   <- compare with the `transferred` FP/frame")
        print(f"  total FP           {dup + phantom:8d}  "
              f"{(dup + phantom) / n_frames:6.3f} /frame")
        print(f"  missed people      {n_gt - matched:8d}  "
              f"{(n_gt - matched) / n_frames:6.3f} /frame\n", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persistence", action="store_true",
                    help="Only the furniture-or-noise measurement")
    ap.add_argument("--breakdown", action="store_true",
                    help="Only the phantom-vs-duplicate measurement")
    ap.add_argument("--thresh", type=float, default=0.3,
                    help="Detector operating threshold (default: 0.3)")
    ap.add_argument("--radius", type=float, default=0.3,
                    help="Frame-to-frame association radius, metres (default: 0.3)")
    ap.add_argument("--assoc", type=float, default=0.5,
                    help="Detection-to-person association, metres (default: 0.5)")
    ap.add_argument("--run-frames", type=int, default=2000,
                    help="Frames per person-free run to measure (default: 2000)")
    ap.add_argument("--stride", type=int, default=5,
                    help="Score every Nth frame of the populated split (default: 5)")
    args = ap.parse_args()

    both = not (args.persistence or args.breakdown)
    angles = frog_laser_angles(720)
    if args.persistence or both:
        print("=== Are the phantoms furniture, or noise? ===\n")
        persistence(args, angles)
        print()
    if args.breakdown or both:
        print("=== Phantom or duplicate, on the populated split? ===\n")
        breakdown(args, angles)


if __name__ == "__main__":
    main()
