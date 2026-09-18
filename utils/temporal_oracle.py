#!/usr/bin/env python3
"""
What could a *perfect* temporal model buy over a static detector? An oracle
bound on both error types, computed from trajectories.

The framing (owner's, 2026-09-10) is an upper-bound analysis in the shape of
standard MOT evaluation:

**Recall.** Every annotated person is a trajectory. A perfect temporal model
follows each one, so every point on a trajectory that the static detector
*missed* is potentially recoverable. Inverting it: the only people such a model
could never recover are those detected on too few frames for a trajectory to
exist at all -- the Mostly-Lost class of Li, Huang & Nevatia (CVPR 2009).

**Precision.** Every static prediction is also a trajectory. A perfect temporal
model is not fooled by something that appears from nowhere and vanishes, so
every *short* prediction trajectory is potentially removable. Inverting it: the
only phantoms that survive are the ones producing long consecutive trails --
which, on this data, is most of them.

**How this relates to SORT's `min_hits`, carefully.** SORT's filter is
*causal*: at frame t it may use only frames up to t, so it pays `min_hits`
frames of latency and cannot revisit a decision. This oracle is *acausal* -- it
judges a trajectory having seen all of it.

That does **not** make it an upper bound on SORT, and an earlier draft of this
file wrongly claimed so. SORT additionally suppresses the first `min_hits - 1`
frames of every track that *does* survive, so at the same threshold it removes
*more* phantoms than this filter -- while also delaying every real person by the
same amount. The two are not nested. Read the output as "what fraction of
phantoms live on short trails", which is the quantity that bounds any
lifetime-based rule, causal or not.

Two bounds are reported on the recall side, because the obvious one is very
loose:

    interpolation  a missed point counts as recoverable only if the same
                   trajectory was detected both *before* and *after* it. This is
                   the credible bound: filling a gap between two observations.
    extrapolation  any point on a trajectory detected at least k times counts,
                   including points beyond the last detection. Reported for
                   contrast; a real model cannot do this reliably.

Usage
-----
    python temporal_oracle.py                        # both halves, LFE-Peaks
    python temporal_oracle.py --detector lfe-ppn
    python temporal_oracle.py --recall               # just the false-negative bound
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
from motion_analysis import build_tracks, to_world


def _detector(name):
    return LFEPeaksDetector() if name == "lfe-peaks" else LFEPPNDetector(score_thresh=0.01)


def _detect_xy(det, scan, angles, thresh):
    return np.array([[x, y] for s, x, y in det.detect(scan, angles) if s >= thresh],
                    dtype=np.float64).reshape(-1, 2)


def recall_oracle(args):
    """How many missed people sit on a trajectory that was detected elsewhere?"""
    ds = FROG_Dataset(split="test", mode="balanced", auto_download=False, verbose=False)
    det = _detector(args.detector)
    angles = frog_laser_angles(720)

    tracks_all = []          # each: list of (frame, detected) in frame order
    for seq in range(len(ds.det_id)):
        t = np.asarray(ds.scan_time[seq], dtype=float)
        if len(t) < 2 or (t[-1] - t[0]) < args.min_seq_seconds:
            continue
        hz = 1.0 / float(np.median(np.diff(t)))
        idx, world, found = [], [], {}
        for det_idx in range(len(ds.det_id[seq])):
            iscan = int(ds.idet2iscan[seq][det_idx])
            g = np.array([project_cartesian_from_polar(r, p)
                          for r, p in ds.det_wp[seq][det_idx]],
                         dtype=np.float64).reshape(-1, 2)
            ok = np.zeros(len(g), dtype=bool)
            if len(g):
                d = _detect_xy(det, ds.scans[seq][iscan], angles, args.thresh)
                if len(d):
                    cost = np.linalg.norm(g[:, None, :] - d[None, :, :], axis=-1)
                    gated = np.where(cost > args.assoc, 1e6, cost)
                    for r, c in zip(*linear_sum_assignment(gated)):
                        if gated[r, c] < 1e6:
                            ok[r] = True
            w = to_world(g, ds.odoms[seq][iscan]["xya"])
            idx.append(iscan)
            world.append(w)
            for j in range(len(w)):
                found[(iscan, round(float(w[j][0]), 6), round(float(w[j][1]), 6))] = bool(ok[j])

        for tr in build_tracks(world, scan_index=idx, gate=args.gate):
            seqn = []
            for f in sorted(tr):
                key = (int(f), round(float(tr[f][0]), 6), round(float(tr[f][1]), 6))
                if key in found:
                    seqn.append((int(f), found[key]))
            if seqn:
                tracks_all.append((seqn, hz))

    n_pts = sum(len(s) for s, _ in tracks_all)
    n_det = sum(sum(d for _, d in s) for s, _ in tracks_all)
    print(f"\n=== RECALL ORACLE: {args.detector} at threshold {args.thresh} ===")
    print(f"{len(tracks_all)} ground-truth trajectories, {n_pts} person-observations")
    print(f"currently detected: {n_det}/{n_pts} = {n_det / n_pts:.1%}"
          f"   (miss rate {1 - n_det / n_pts:.1%})\n")

    # --- Mostly Tracked / Partially Tracked / Mostly Lost -------------------
    cov = np.array([sum(d for _, d in s) / len(s) for s, _ in tracks_all])
    wts = np.array([len(s) for s, _ in tracks_all], dtype=float)
    print("  trajectory coverage (MT/PT/ML, Li-Huang-Nevatia):")
    for lab, m in (("Mostly Tracked  (>=80% covered)", cov >= 0.8),
                   ("Partially Tracked (20-80%)", (cov >= 0.2) & (cov < 0.8)),
                   ("Mostly Lost     (<20% covered)", cov < 0.2)):
        print(f"    {lab:<34} {m.sum():>5} trajectories "
              f"({m.sum() / len(cov):>5.1%})  "
              f"{wts[m].sum() / wts.sum():>5.1%} of observations")

    # --- the two bounds -----------------------------------------------------
    interp_rec = 0
    for s, hz in tracks_all:
        bridge = args.max_bridge_s * hz
        det_frames = [f for f, d in s if d]
        if not det_frames:
            continue
        arr = np.array(det_frames)
        for f, d in s:
            if d:
                continue
            before = arr[arr < f]
            after = arr[arr > f]
            if len(before) and len(after) and (after[0] - before[-1]) <= bridge:
                interp_rec += 1
    print(f"\n  INTERPOLATION bound -- a miss counts only if the same trajectory was")
    print(f"  detected both before and after it, within {args.max_bridge_s:.0f} s:")
    print(f"    recoverable misses      {interp_rec:>7} "
          f"({interp_rec / n_pts:>5.1%} of all observations)")
    print(f"    recall: {n_det / n_pts:.1%} -> {(n_det + interp_rec) / n_pts:.1%}"
          f"   (+{interp_rec / n_pts * 100:.1f} pp)")

    print(f"\n  EXTRAPOLATION bound -- every point of any trajectory detected >= k")
    print(f"  times, including beyond its last detection (loose; a real model cannot):")
    for k in (1, 2, 3, 5, 10):
        rec = sum(len(s) for s, _ in tracks_all if sum(d for _, d in s) >= k)
        print(f"    k={k:<3} recall ceiling {rec / n_pts:>6.1%}")

    print(f"\n  Irreducible: people on trajectories never detected at all -- "
          f"{sum(len(s) for s, _ in tracks_all if not any(d for _, d in s)) / n_pts:.1%} "
          f"of observations.")


def precision_oracle(args):
    """How many phantoms sit on short trajectories a perfect model would reject?"""
    ds = FROG_Dataset(split="test", mode="transferred",
                      auto_download=False, verbose=False)
    det = _detector(args.detector)
    angles = frog_laser_angles(720)
    hz = 1.0 / float(np.median(np.diff(np.asarray(ds.scan_time[0], dtype=float))))

    all_tracks, n_frames = [], 0
    for seq in range(len(ds.det_id)):
        ids = np.asarray(ds.det_id[seq], dtype=np.int64)
        brk = np.nonzero(np.diff(ids) != 1)[0]
        starts = np.concatenate(([0], brk + 1))
        ends = np.concatenate((brk + 1, [len(ids)]))
        best = int(np.argmax(ends - starts))
        run = ids[starts[best]:ends[best]][:args.run_frames]
        if len(run) < 500:
            continue
        world = [to_world(_detect_xy(det, ds.scans[seq][int(i)], angles, args.thresh),
                          ds.odoms[seq][int(i)]["xya"]) for i in run]
        all_tracks += build_tracks(world, scan_index=[int(i) for i in run],
                                   gate=args.gate, max_gap=args.max_gap)
        n_frames += len(run)

    lens = np.array([len(t) for t in all_tracks])
    n_ph = int(lens.sum())
    print(f"\n=== PRECISION ORACLE: {args.detector} at threshold {args.thresh} ===")
    print(f"{len(all_tracks)} phantom trajectories over {n_frames} person-free frames, "
          f"{n_ph} phantom detections ({n_ph / n_frames:.2f}/frame)")
    print(f"  trajectory length: median {np.median(lens):.0f} frames "
          f"({np.median(lens) / hz:.2f} s), p90 {np.percentile(lens, 90):.0f}, "
          f"max {lens.max()}\n")
    print(f"  a perfect acausal filter that drops every trajectory shorter than L:")
    print(f"  {'L (frames)':>11} {'L (s)':>7} {'phantoms removed':>18} {'remaining/frame':>16}")
    for L in (2, 3, 5, 10, 26, 52, 131):
        removed = int(lens[lens < L].sum())
        print(f"  {L:>11} {L / hz:>7.2f} {removed / n_ph:>17.1%} "
              f"{(n_ph - removed) / n_frames:>16.2f}")
    print(f"\n  Irreducible: phantoms on trajectories >= 131 frames (5 s) -- "
          f"{lens[lens >= 131].sum() / n_ph:.1%} of all phantom detections.")
    print("  NOT a bound on SORT, despite appearances: SORT also suppresses "
          "the first min_hits-1 frames")
    print("  of every surviving track, so at the same threshold it removes "
          "MORE phantoms -- and delays")
    print("  every real person too. The filters are not nested. Read it as "
          "what fraction of phantoms")
    print("  live on short trails, which bounds any lifetime-based rule, "
          "causal or not.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recall", action="store_true")
    ap.add_argument("--precision", action="store_true")
    ap.add_argument("--detector", choices=["lfe-peaks", "lfe-ppn"], default="lfe-peaks")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--assoc", type=float, default=0.5,
                    help="Detection-to-person association, metres (default: 0.5)")
    ap.add_argument("--gate", type=float, default=0.6,
                    help="Trajectory association gate, metres (default: 0.6)")
    ap.add_argument("--max-gap", type=int, default=0,
                    help="Frames a phantom trajectory may flicker out and still "
                         "count as one trajectory (default: 0). **Leave at 0** "
                         "unless you mean something else: at 0 a trajectory "
                         "breaks on any miss, so its length IS its longest "
                         "consecutive run -- what 'a long consecutive trail of "
                         "static detections' means literally. At 5 the same "
                         "phantoms merge into 3x fewer, 3x longer trajectories "
                         "and the filter looks far weaker than it is.")
    ap.add_argument("--max-bridge-s", type=float, default=2.0,
                    help="Longest gap the interpolation bound will bridge, "
                         "seconds (default: 2)")
    ap.add_argument("--min-seq-seconds", type=float, default=40.0)
    ap.add_argument("--run-frames", type=int, default=8000)
    args = ap.parse_args()

    both = not (args.recall or args.precision)
    if args.recall or both:
        recall_oracle(args)
    if args.precision or both:
        precision_oracle(args)


if __name__ == "__main__":
    main()
