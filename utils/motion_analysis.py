#!/usr/bin/env python3
"""
How much do FROG's annotated people actually move, and over what horizon?

This bounds every motion-based method before one is built. If temporal
information separates a person from a chair because *people move and chairs do
not*, then the fraction of people who are stationary over the horizon a method
can see is a hard ceiling on what that method can ever fix -- and no
architecture crosses it. A museum, which is where FROG was recorded, is exactly
the setting where people stand still to look at things.

Two measurements, both model-free and both from ground truth:

**People** (`--people`). Annotations are associated across frames into
trajectories, converted to a *world* frame using the published odometry (so the
robot's own motion is not counted as theirs), and the displacement of each
person over horizons of 1, 2, 5, 10 and 30 seconds is reported as a
distribution. The number that matters is the fraction below a person's own
width: someone who has moved less than ~0.5 m in 10 seconds is, to any
motion-based discriminator, a chair.

`--dataset drow` runs the same measurement on a different venue, which is the
point of having it: FROG is a museum, where people stand and look at things,
and a ceiling measured there may say more about museums than about 2D LiDAR.
The two datasets are not directly comparable frame-for-frame -- FROG annotates
every scan at 26.2 Hz and holds 3.06 people per frame, DROW annotates every 5th
scan at 10 Hz (so 0.5 s between annotations) and holds 0.81 -- so horizons are
specified in *seconds* and matched to the nearest available annotation within a
tolerance, rather than in frames.

**Phantoms** (`--phantoms`). The same measurement applied to a detector's false
positives on person-free frames. This validates the world transform -- static
furniture must come out near zero -- and quantifies the other side of the
comparison. A method can only exploit the gap between these two distributions,
so the overlap between them is the real ceiling.

Usage
-----
    python motion_analysis.py                 # both
    python motion_analysis.py --people        # just the ceiling measurement
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.datasets.frog_dataset import FROG_Dataset, frog_laser_angles
from follow_the_drow.detectors import LFEPeaksDetector, LFEPPNDetector
from follow_the_drow.utils.drow_utils import project_cartesian_from_polar
from scipy.optimize import linear_sum_assignment

HORIZONS_S = (1.0, 2.0, 5.0, 10.0, 30.0)
FRAME_HZ = 26.2


def to_world(xy, pose):
    """Robot-frame (x=right, y=forward) -> world frame, using odometry.

    The inverse of `tracking.odom_xya_delta_to_tracker_frame`: odometry's
    "xya" is standard robotics (x forward, y left, theta CCW), while this
    codebase's detection frame is x = -r sin(phi), y = r cos(phi) -- so
    detection-y is local-forward and detection-x is *minus* local-left.
    """
    wx, wy, theta = float(pose[0]), float(pose[1]), float(pose[2])
    fwd, left = xy[:, 1], -xy[:, 0]
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([wx + c * fwd - s * left, wy + s * fwd + c * left], axis=1)


def build_tracks(per_frame_world, scan_index=None, gate=0.6, max_gap=0):
    """Greedy nearest-neighbour association into trajectories.

    Sufficient here rather than a full tracker where annotations are dense: at
    FROG's 26.2 Hz a person walking at 1.4 m/s moves 5.3 cm between frames, an
    order of magnitude inside the gate, so association is unambiguous except in
    dense contact -- and a mis-association there swaps two people who are by
    definition close together, which barely perturbs a displacement statistic.

    **The gate must scale with the annotation interval.** DROW labels every 5th
    scan at 10 Hz, so consecutive annotations are 0.5 s apart and the same
    walker covers 0.7 m -- past a gate sized for FROG, which would shatter every
    trajectory into singletons and report a population of stationary people that
    does not exist.

    `scan_index[i]` is the raw scan number of entry `i`, so tracks are keyed by
    scan number and horizons stay in real time. Defaults to consecutive.

    Returns a list of {scan_index: world position} dicts.

    `max_gap` lets a track survive that many consecutive unmatched frames
    before it is dropped, so a flickering detection stays one trajectory rather
    than becoming several. 0 reproduces the strict behaviour. Ground-truth
    annotations need none (FROG labels every scan); detections do.
    """
    if scan_index is None:
        scan_index = list(range(len(per_frame_world)))
    tracks, active = [], []      # active: (track_idx, last_xy, misses)
    for i, pts in enumerate(per_frame_world):
        f = int(scan_index[i])
        used, new_active = set(), []
        for ti, last, misses in active:
            hit = False
            if len(pts):
                d = np.linalg.norm(pts - last[None, :], axis=1)
                for j in np.argsort(d):
                    if d[j] > gate:
                        break        # nothing closer will qualify either
                    if j in used:
                        continue     # taken by another track; try the next
                    tracks[ti][f] = pts[j]
                    used.add(j)
                    new_active.append((ti, pts[j], 0))
                    hit = True
                    break
            if not hit and misses < max_gap:
                new_active.append((ti, last, misses + 1))
        for j in range(len(pts)):
            if j not in used:
                tracks.append({f: pts[j]})
                new_active.append((len(tracks) - 1, pts[j], 0))
        active = new_active
    return tracks


def displacement_stats(tracks, label, hz, tol=0.2):
    """Displacement over each horizon, matched to the nearest annotation.

    A horizon is specified in seconds and converted to scans via `hz`. Because
    a dataset need not annotate every scan, the partner observation is the one
    nearest to `f + horizon`, accepted only within `tol` of the horizon -- so a
    sparsely-labelled dataset is measured over the interval it claims, not over
    whatever gap happened to be available.
    """
    print(f"\n{label}")
    print(f"  {len(tracks)} trajectories, {sum(len(t) for t in tracks)} observations")
    print(f"  {'horizon':>8} {'n pairs':>9} {'median':>9} {'p25':>8} {'p75':>8} "
          f"{'< 0.5 m':>9} {'< 1.0 m':>9}")
    prepared = []
    for t in tracks:
        if len(t) < 2:
            continue
        fr = np.array(sorted(t), dtype=np.int64)
        prepared.append((fr, np.stack([t[int(f)] for f in fr])))
    for h_s in HORIZONS_S:
        h = h_s * hz
        lo, hi = h * (1 - tol), h * (1 + tol)
        disp = []
        for fr, xy in prepared:
            target = fr + h
            j = np.searchsorted(fr, target)
            for i in range(len(fr)):
                for cand in (j[i] - 1, j[i]):
                    if 0 <= cand < len(fr):
                        gap = fr[cand] - fr[i]
                        if lo <= gap <= hi:
                            disp.append(float(np.linalg.norm(xy[cand] - xy[i])))
                            break
        if len(disp) < 50:
            print(f"  {h_s:>7.0f}s {len(disp):>9}   (too few trajectories this long)")
            continue
        d = np.asarray(disp)
        print(f"  {h_s:>7.0f}s {len(d):>9} {np.median(d):>9.2f} "
              f"{np.percentile(d, 25):>8.2f} {np.percentile(d, 75):>8.2f} "
              f"{(d < 0.5).mean():>8.1%} {(d < 1.0).mean():>8.1%}")


def _load(args):
    """Return (dataset, hz, gate, label) for the requested dataset."""
    if args.dataset == "drow":
        from follow_the_drow.datasets import DROW_Dataset
        ds = DROW_Dataset(verbose=False)
        hz = 1.0 / float(np.median(np.diff(np.asarray(ds.scan_time[0], float))))
        # DROW labels every 5th scan, so consecutive annotations are ~0.5 s
        # apart and a walker covers ~0.7 m between them. Gate on a generous
        # 2.5 m/s so trajectories survive; a FROG-sized gate would shatter them.
        gate = max(args.gate_speed * (5.0 / hz), args.gate_floor)
        return ds, hz, gate, "DROW (care facility, 450 beams)"
    ds = FROG_Dataset(split=args.split, mode="balanced",
                      auto_download=False, verbose=False)
    hz = 1.0 / float(np.median(np.diff(np.asarray(ds.scan_time[0], float))))
    gate = max(args.gate_speed / hz, args.gate_floor)
    return ds, hz, gate, "FROG (museum, 720 beams)"


def _person_rp(ds, seq, det_idx, include_aids):
    """(r, phi) of every annotated person in one frame.

    DROW splits its annotations into pedestrian / wheelchair / walker; FROG has
    only the first. `--include-aids` folds the other two in, which is the more
    honest population for a *motion* question -- a wheelchair user is a person
    who moves differently, not a non-person -- but it is off by default so the
    two datasets are compared on the same class.
    """
    out = list(ds.det_wp[seq][det_idx])
    if include_aids:
        out += list(ds.det_wc[seq][det_idx]) + list(ds.det_wa[seq][det_idx])
    return out


def people(args):
    ds, hz, gate, label = _load(args)
    all_tracks, n_obs, n_seq = [], 0, 0
    for seq in range(min(args.max_seqs, len(ds.det_id))):
        # Filter on session *duration*, not annotated-frame count: DROW labels
        # ~486 frames per 18-minute session, FROG 2000+ per 76-second one, so a
        # frame-count threshold means completely different things in the two.
        t = np.asarray(ds.scan_time[seq], dtype=float)
        if len(t) < 2 or (t[-1] - t[0]) < args.min_seq_seconds:
            continue
        idx, per_frame = [], []
        for det_idx in range(len(ds.det_id[seq])):
            iscan = ds.idet2iscan[seq][det_idx]
            rp = _person_rp(ds, seq, det_idx, args.include_aids)
            xy = np.array([project_cartesian_from_polar(r, p) for r, p in rp],
                          dtype=np.float64).reshape(-1, 2)
            idx.append(int(iscan))
            per_frame.append(to_world(xy, ds.odoms[seq][iscan]["xya"]))
            n_obs += len(xy)
        all_tracks += build_tracks(per_frame, scan_index=idx, gate=gate)
        n_seq += 1
    displacement_stats(
        all_tracks,
        f"{label}: annotated people over {n_seq} session(s), {n_obs} annotations, "
        f"{hz:.1f} Hz, association gate {gate:.2f} m",
        hz)


def phantoms(args):
    """The same measurement on a detector's false positives.

    Validates the world transform -- static furniture must come out near zero --
    and gives the other half of the comparison a method would have to exploit.
    FROG only: it needs a split known to contain no people.
    """
    ds = FROG_Dataset(split="test", mode="transferred",
                      auto_download=False, verbose=False)
    hz = 1.0 / float(np.median(np.diff(np.asarray(ds.scan_time[0], float))))
    angles = frog_laser_angles(720)
    for name, make in (("LFE-Peaks", lambda: LFEPeaksDetector()),
                       ("LFE-PPN", lambda: LFEPPNDetector(score_thresh=0.01))):
        det = make()
        seq = 0
        ids = np.asarray(ds.det_id[seq], dtype=np.int64)
        breaks = np.nonzero(np.diff(ids) != 1)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks + 1, [len(ids)]))
        best = int(np.argmax(ends - starts))
        run = ids[starts[best]:ends[best]][:args.run_frames]
        per_frame = []
        for i in run:
            xy = np.array([[x, y] for s, x, y in det.detect(ds.scans[seq][int(i)], angles)
                           if s >= args.thresh], dtype=np.float64).reshape(-1, 2)
            per_frame.append(to_world(xy, ds.odoms[seq][int(i)]["xya"]))
        displacement_stats(
            build_tracks(per_frame, scan_index=[int(i) for i in run],
                         gate=args.gate_speed / hz),
            f"{name} phantoms, person-free run ({len(run)} frames)", hz)


def headroom(args):
    """Of the people a static detector misses, how many move enough to be
    rescued by temporal information?

    This is the question that actually bounds the work, and it is not the
    product of the two rates measured separately -- misses and stationarity may
    be **correlated**. A standing person holds their legs together and presents
    one blob; a walking person shows the separated two-leg signature these
    detectors are built around. If stationary people are disproportionately the
    ones being missed, the addressable set is far smaller than
    `miss_rate x moving_fraction` suggests, and every motion-based method
    inherits that.

    Reports, per horizon:

      P(miss | stationary)  vs  P(miss | moving)   -- the correlation
      P(miss AND moving)                           -- the addressable margin
      1 - P(miss AND stationary)                   -- the recall ceiling for any
                                                      motion-based method, i.e.
                                                      what is left after perfect
                                                      temporal rescue
    """
    ds, hz, gate, label = _load(args)
    if args.dataset != "frog":
        raise SystemExit(
            "Headroom needs a detector native to the dataset. LFE is FROG-trained; "
            "on DROW it is a degraded cross-dataset transfer "
            "(memory/interpreting-evaluation.md) and its miss rate would not be a "
            "static detector's miss rate. Use DrowDetector's published weights "
            "there -- not yet wired into this script.")

    det = (LFEPeaksDetector() if args.detector == "lfe-peaks"
           else LFEPPNDetector(score_thresh=0.01))
    angles = frog_laser_angles(720)

    all_tracks, matched_by = [], {}
    n_seq = 0
    for seq in range(min(args.max_seqs, len(ds.det_id))):
        t = np.asarray(ds.scan_time[seq], dtype=float)
        if len(t) < 2 or (t[-1] - t[0]) < args.min_seq_seconds:
            continue
        idx, per_frame, per_frame_local = [], [], []
        for det_idx in range(len(ds.det_id[seq])):
            iscan = ds.idet2iscan[seq][det_idx]
            rp = _person_rp(ds, seq, det_idx, args.include_aids)
            xy = np.array([project_cartesian_from_polar(r, p) for r, p in rp],
                          dtype=np.float64).reshape(-1, 2)
            idx.append(int(iscan))
            per_frame_local.append(xy)
            per_frame.append(to_world(xy, ds.odoms[seq][iscan]["xya"]))

        # Which ground-truth people did the detector find? Hungarian assignment
        # per frame, gated at `--assoc`, exactly as the AP metric matches -- so
        # "missed" here means what it means in the score table, and a duplicate
        # detection cannot claim two people.
        matched = []
        for k, iscan in enumerate(idx):
            g = per_frame_local[k]
            if not len(g):
                matched.append(np.zeros(0, dtype=bool))
                continue
            d = np.array([[x, y] for sc, x, y in
                          det.detect(ds.scans[seq][iscan], angles)
                          if sc >= args.thresh], dtype=np.float64).reshape(-1, 2)
            ok = np.zeros(len(g), dtype=bool)
            if len(d):
                cost = np.linalg.norm(g[:, None, :] - d[None, :, :], axis=-1)
                gated = np.where(cost > args.assoc, 1e6, cost)
                rows, cols = linear_sum_assignment(gated)
                for r, c in zip(rows, cols):
                    if gated[r, c] < 1e6:
                        ok[r] = True
            matched.append(ok)

        tracks = build_tracks(per_frame, scan_index=idx, gate=gate)
        # Re-attach detection status by (scan, position) so it survives association.
        pos_to_ok = {}
        for k, iscan in enumerate(idx):
            for j in range(len(per_frame[k])):
                pos_to_ok[(int(iscan), round(float(per_frame[k][j][0]), 6),
                           round(float(per_frame[k][j][1]), 6))] = bool(matched[k][j])
        all_tracks += tracks
        matched_by.update(pos_to_ok)
        n_seq += 1

    print(f"\n{label}: {args.detector} at threshold {args.thresh}, "
          f"{n_seq} session(s), {len(all_tracks)} trajectories")
    print(f"  {'horizon':>8} {'n':>8} {'miss|still':>11} {'miss|moving':>12} "
          f"{'addressable':>12} {'ceiling':>9}")
    for h_s in HORIZONS_S:
        h = h_s * hz
        lo, hi = h * (1 - 0.2), h * (1 + 0.2)
        still_miss = still_n = move_miss = move_n = 0
        for t in all_tracks:
            if len(t) < 2:
                continue
            fr = np.array(sorted(t), dtype=np.int64)
            xy = np.stack([t[int(f)] for f in fr])
            j = np.searchsorted(fr, fr + h)
            for i in range(len(fr)):
                for cand in (j[i] - 1, j[i]):
                    if not (0 <= cand < len(fr)):
                        continue
                    gap = fr[cand] - fr[i]
                    if not (lo <= gap <= hi):
                        continue
                    disp = float(np.linalg.norm(xy[cand] - xy[i]))
                    key = (int(fr[i]), round(float(xy[i][0]), 6),
                           round(float(xy[i][1]), 6))
                    found = matched_by.get(key)
                    if found is None:
                        break
                    if disp < args.still:
                        still_n += 1
                        still_miss += not found
                    else:
                        move_n += 1
                        move_miss += not found
                    break
        n = still_n + move_n
        if n < 50:
            print(f"  {h_s:>7.0f}s {n:>8}   (too few)")
            continue
        p_ms = still_miss / max(still_n, 1)
        p_mm = move_miss / max(move_n, 1)
        addressable = move_miss / n
        ceiling = 1 - still_miss / n
        print(f"  {h_s:>7.0f}s {n:>8} {p_ms:>11.1%} {p_mm:>12.1%} "
              f"{addressable:>12.1%} {ceiling:>9.1%}")
    print(f"\n  miss|still  : how often the detector misses a person who moved "
          f"< {args.still} m over the horizon")
    print( "  addressable : share of ALL person-observations that are both "
           "missed AND moving -- the margin temporal information could buy")
    print( "  ceiling     : best recall any motion-based method could reach, "
           "i.e. everything except the missed-and-stationary")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["frog", "drow"], default="frog",
                    help="Which venue to measure (default: frog, a museum). "
                         "'drow' is a care facility -- different venue, "
                         "different movement, and the point of the comparison.")
    ap.add_argument("--include-aids", action="store_true",
                    help="Fold DROW's wheelchair and walker classes in with "
                         "pedestrians (no effect on FROG, which has no such "
                         "classes)")
    ap.add_argument("--gate-speed", type=float, default=2.5,
                    help="Association gate, metres per second of annotation "
                         "interval (default: 2.5, a fast walk plus margin)")
    ap.add_argument("--gate-floor", type=float, default=0.6,
                    help="Minimum association gate in metres (default: 0.6). "
                         "A gate sized from motion alone is far too tight at "
                         "26 Hz -- 2.5 m/s over 38 ms is 9.6 cm, below the "
                         "frame-to-frame jitter of the annotations themselves, "
                         "so trajectories shatter and the statistic is biased "
                         "toward the stationary people whose tracks survive. "
                         "Check any conclusion against --gate-floor before "
                         "believing it.")
    ap.add_argument("--people", action="store_true")
    ap.add_argument("--phantoms", action="store_true")
    ap.add_argument("--headroom", action="store_true",
                    help="Cross-tabulate detector misses against motion -- the "
                         "margin a temporal method could actually buy")
    ap.add_argument("--detector", choices=["lfe-peaks", "lfe-ppn"],
                    default="lfe-peaks")
    ap.add_argument("--assoc", type=float, default=0.5,
                    help="Detection-to-person association, metres (default: 0.5, "
                         "the AP metric's own)")
    ap.add_argument("--still", type=float, default=0.5,
                    help="Displacement below which a person counts as stationary "
                         "(default: 0.5 m, about a body width)")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--run-frames", type=int, default=3000)
    ap.add_argument("--max-seqs", type=int, default=200)
    ap.add_argument("--min-seq-seconds", type=float, default=40.0,
                    help="Skip sessions shorter than this many seconds "
                         "(default: 40) -- a horizon cannot be measured inside "
                         "a session shorter than the horizon")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    if args.headroom:
        print("=== Of the people a static detector misses, how many move? ===")
        headroom(args)
        return
    both = not (args.people or args.phantoms)
    if args.people or both:
        print("=== How far do annotated people move, in the world frame? ===")
        people(args)
    if args.phantoms or both:
        if args.dataset != "frog":
            print("\n(--phantoms needs a person-free split; FROG only)")
        else:
            print("\n=== And the phantoms, for comparison? ===")
            phantoms(args)


if __name__ == "__main__":
    main()
