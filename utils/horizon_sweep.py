"""Horizon curves from the oracle, rather than from displacement.

`dataset-properties.md`'s horizon table answers "how far has a person moved
after t seconds", which bounds methods that discriminate *by displacement* --
its own stated scope.  A trajectory reasoner is not such a method: a stationary
person still emits a weak-but-consistent return that integrates.

So the horizon question is better asked of the oracle, on both of its halves:

  recall side     how long a detection GAP must be bridgeable for the
                  interpolation bound to saturate
  precision side  how long a phantom TRAIL filter must reach for the
                  removable fraction to saturate
  both            the trajectory-duration distributions, which cap the useful
                  horizon independently of either -- a 30 s horizon buys
                  nothing on a person visible for 4 s

One detector pass, cached; both axes then swept analytically over the cached
track structures, so the sweep itself is free.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.datasets.frog_dataset import FROG_Dataset, frog_laser_angles
from follow_the_drow.utils.drow_utils import project_cartesian_from_polar

from motion_analysis import build_tracks, to_world
from temporal_oracle import _detector, _detect_xy


# ---------------------------------------------------------------- collection

def collect_recall(args):
    """One detector pass over annotated frames -> per-trajectory (frame, hit)."""
    ds = FROG_Dataset(split="test", mode="balanced", auto_download=False,
                      verbose=False)
    det = _detector(args.detector)
    angles = frog_laser_angles(720)

    tracks, hz_out = [], None
    for seq in range(len(ds.det_id)):
        t = np.asarray(ds.scan_time[seq], dtype=float)
        if len(t) < 2 or (t[-1] - t[0]) < args.min_seq_seconds:
            continue
        hz = 1.0 / float(np.median(np.diff(t)))
        hz_out = hz if hz_out is None else hz_out
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
                found[(iscan, round(float(w[j][0]), 6),
                       round(float(w[j][1]), 6))] = bool(ok[j])

        for tr in build_tracks(world, scan_index=idx, gate=args.gate):
            seqn = []
            for f in sorted(tr):
                key = (int(f), round(float(tr[f][0]), 6),
                       round(float(tr[f][1]), 6))
                if key in found:
                    seqn.append((int(f), found[key]))
            if seqn:
                tracks.append(seqn)
    return tracks, hz_out


def trail_curve(tracks, lengths):
    """Both sides of a trail-length filter, per horizon.

    `tracks` is [(length_in_frames, n_true_positives, n_false_positives)].
    Returns [(false_positives_removed, true_positives_destroyed)] per L, under
    the same rule the precision oracle uses -- *drop every trail shorter than
    L* -- so the two are directly comparable.

    Trails are dropped whole. A filter cannot keep the person half of a trail
    that merged a person with a phantom, and the bound must charge for that.
    """
    out = []
    for L in lengths:
        fp = sum(t[2] for t in tracks if t[0] < L)
        tp = sum(t[1] for t in tracks if t[0] < L)
        out.append((fp, tp))
    return out


def collect_discrimination(args):
    """The precision oracle asked on POPULATED frames, where it must discriminate.

    `collect_precision` below runs on person-free frames, so every detection is
    a phantom by construction and the filter cannot destroy a true positive.
    That bounds *removal*. A41 measured what happens when a mechanism has to do
    both on real scenes -- recall 79.3% -> 23.2% -- so the one-sided bound was
    not predictive of anything.

    Here trails are built over **every** detection, phantom and person alike,
    exactly as a deployed filter would see them, and each detection is labelled
    by whether it was assigned to an annotated person. Returns per-track
    (length, n_tp, n_fp) plus the totals needed to turn counts into rates.
    """
    ds = FROG_Dataset(split="test", mode=args.mode, auto_download=False,
                      verbose=False)
    det = _detector(args.detector)
    angles = frog_laser_angles(720)
    hz = 1.0 / float(np.median(np.diff(np.asarray(ds.scan_time[0], dtype=float))))

    tracks, n_tp, n_fp, n_gt, n_frames = [], 0, 0, 0, 0
    for seq in range(len(ds.det_id)):
        idx, world, label = [], [], {}
        for det_idx in range(len(ds.det_id[seq])):
            iscan = int(ds.idet2iscan[seq][det_idx])
            g = np.array([project_cartesian_from_polar(r, p)
                          for r, p in ds.det_wp[seq][det_idx]],
                         dtype=np.float64).reshape(-1, 2)
            d = _detect_xy(det, ds.scans[seq][iscan], angles, args.thresh)

            is_tp = np.zeros(len(d), dtype=bool)
            if len(d) and len(g):
                cost = np.linalg.norm(d[:, None, :] - g[None, :, :], axis=-1)
                gated = np.where(cost > args.assoc, 1e6, cost)
                for r, c in zip(*linear_sum_assignment(gated)):
                    if gated[r, c] < 1e6:
                        is_tp[r] = True

            w = to_world(d, ds.odoms[seq][iscan]["xya"])
            idx.append(iscan)
            world.append(w)
            for j in range(len(w)):
                label[(iscan, round(float(w[j][0]), 6),
                       round(float(w[j][1]), 6))] = bool(is_tp[j])
            n_tp += int(is_tp.sum())
            n_fp += int((~is_tp).sum())
            n_gt += len(g)
            n_frames += 1

        for tr in build_tracks(world, scan_index=idx, gate=args.gate,
                               max_gap=args.max_gap):
            frames = sorted(tr)
            tp = fp = 0
            for f in frames:
                key = (int(f), round(float(tr[f][0]), 6), round(float(tr[f][1]), 6))
                if label.get(key, False):
                    tp += 1
                else:
                    fp += 1
            xy = np.array([tr[f] for f in frames], dtype=np.float64)
            # Bounding-box diagonal in world coordinates: how far this thing
            # ranged over its life. `PAPER.md`'s separating claim is that a
            # person eventually displaces and a chair never does -- which is
            # displacement, not trail length, and the two are not the same
            # feature. Trail length was only ever a proxy, and on person-free
            # frames a proxy is never tested against the class it must exclude.
            disp = float(np.hypot(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
            tracks.append((frames[-1] - frames[0] + 1, tp, fp, disp))
    return tracks, n_tp, n_fp, n_gt, n_frames, hz


def report_discrimination(tracks, n_tp, n_fp, n_gt, n_frames, hz, args):
    print(f"\n=== ROLE-C BOUND, ON POPULATED FRAMES: {args.detector} "
          f"@ {args.thresh} / {args.mode} ===")
    print(f"{n_frames} frames, {n_gt} annotated people, {n_tp + n_fp} detections "
          f"({n_tp} matched, {n_fp} phantom), {len(tracks)} trails")
    print(f"baseline recall {n_tp / max(n_gt, 1):.1%}, "
          f"FP/frame {n_fp / n_frames:.3f}")
    print("\n  A perfect acausal filter dropping every trail shorter than L,")
    print("  applied to ALL detections -- so it can destroy true positives too.")
    print(f"\n  {'L':>9} {'frames':>8} {'FP removed':>12} {'TP lost':>10} "
          f"{'recall':>8} {'FP/frame':>10} {'FP per FN':>11}")
    for h in args.trails:
        L = max(2, int(round(h * hz)))
        (fp_rm, tp_lost), = trail_curve(tracks, [L])
        rec = (n_tp - tp_lost) / max(n_gt, 1)
        d_fp, d_fn = fp_rm / n_frames, tp_lost / n_frames
        print(f"  {h:>8.2f}s {L:>8} {fp_rm / max(n_fp, 1):>11.1%} "
              f"{tp_lost / max(n_tp, 1):>9.1%} {rec:>7.1%} "
              f"{(n_fp - fp_rm) / n_frames:>10.3f} "
              f"{(d_fp / d_fn if d_fn else float('inf')):>11.2f}")
    print("\n  `FP per FN` is the exchange rate, comparable with the levers already")
    print("  priced: SORT 3.8, the merge radius 21.9 (CHANGELOG 2026-09-11).")

    if len(tracks) and len(tracks[0]) > 3:
        print("\n  The same filter keyed on DISPLACEMENT instead of trail length:")
        print("  suppress every trail whose world-frame bounding box is smaller")
        print("  than D. A chair never moves; a person eventually does.")
        print("\n  {:>9} {:>12} {:>10} {:>8} {:>10} {:>11}".format(
            "D", "FP removed", "TP lost", "recall", "FP/frame", "FP per FN"))
        for d in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0):
            fp_rm = sum(t[2] for t in tracks if t[3] < d)
            tp_lost = sum(t[1] for t in tracks if t[3] < d)
            d_fp, d_fn = fp_rm / n_frames, tp_lost / n_frames
            print("  {:>8.2f}m {:>11.1%} {:>9.1%} {:>7.1%} {:>10.3f} {:>11.2f}".format(
                d, fp_rm / max(n_fp, 1), tp_lost / max(n_tp, 1),
                (n_tp - tp_lost) / max(n_gt, 1), (n_fp - fp_rm) / n_frames,
                (d_fp / d_fn if d_fn else float("inf"))))


def collect_precision(args):
    """One detector pass over person-free frames -> phantom track frame lists."""
    ds = FROG_Dataset(split="test", mode="transferred", auto_download=False,
                      verbose=False)
    det = _detector(args.detector)
    angles = frog_laser_angles(720)
    hz = 1.0 / float(np.median(np.diff(np.asarray(ds.scan_time[0], dtype=float))))

    tracks, n_frames, run_lens = [], 0, []
    for seq in range(len(ds.det_id)):
        ids = np.asarray(ds.det_id[seq], dtype=np.int64)
        brk = np.nonzero(np.diff(ids) != 1)[0]
        starts = np.concatenate(([0], brk + 1))
        ends = np.concatenate((brk + 1, [len(ids)]))
        best = int(np.argmax(ends - starts))
        run = ids[starts[best]:ends[best]][:args.run_frames]
        if len(run) < 500:
            continue
        world = [to_world(_detect_xy(det, ds.scans[seq][int(i)], angles,
                                     args.thresh),
                          ds.odoms[seq][int(i)]["xya"]) for i in run]
        tracks += build_tracks(world, scan_index=[int(i) for i in run],
                               gate=args.gate, max_gap=0)
        n_frames += len(run)
        run_lens.append(len(run))
    return [sorted(int(f) for f in t) for t in tracks], n_frames, hz, run_lens


# ------------------------------------------------------------------- reports

def _saturation(xs, ys, frac):
    """Smallest x whose y reaches `frac` of the final (saturated) y."""
    target = ys[-1] * frac
    for x, y in zip(xs, ys):
        if y >= target:
            return x
    return xs[-1]


def report_durations(label, lens_s, censor_note=""):
    a = np.asarray(lens_s, dtype=float)
    print(f"\n  {label}: {len(a)} trajectories")
    print(f"    {'median':>8} {'p75':>8} {'p90':>8} {'p95':>8} {'p99':>8} "
          f"{'max':>8}")
    print(f"    {np.median(a):>7.1f}s {np.percentile(a, 75):>7.1f}s "
          f"{np.percentile(a, 90):>7.1f}s {np.percentile(a, 95):>7.1f}s "
          f"{np.percentile(a, 99):>7.1f}s {a.max():>7.1f}s")
    # Observation-weighted: what a model actually spends its time looking at.
    w = a / a.sum()
    print(f"    observation-weighted mean duration "
          f"{float((a * w).sum()):.1f}s")
    for h in (1, 2, 5, 10, 30, 60):
        print(f"    trajectories reaching {h:>3}s: "
              f"{(a >= h).mean():>5.1%}   "
              f"observations on them: {a[a >= h].sum() / a.sum():>5.1%}")
    if censor_note:
        print(f"    {censor_note}")


def report_recall(tracks, hz, args):
    n_pts = sum(len(s) for s in tracks)
    n_det = sum(sum(d for _, d in s) for s in tracks)
    print(f"\n=== RECALL HORIZON: {args.detector} @ {args.thresh} ===")
    print(f"{len(tracks)} ground-truth trajectories, {n_pts} person-observations,"
          f" {hz:.1f} Hz")
    print(f"currently detected {n_det}/{n_pts} = {n_det / n_pts:.1%}"
          f"   (miss {1 - n_det / n_pts:.1%})")

    report_durations("person trajectories",
                     [(s[-1][0] - s[0][0]) / hz for s in tracks])

    print(f"\n  Interpolation bound as a function of the bridgeable gap.")
    print(f"  A miss counts as recoverable only if the SAME trajectory was")
    print(f"  detected both before and after it, within the horizon.")
    print(f"  {'horizon':>9} {'frames':>8} {'recoverable':>12} "
          f"{'recall ceiling':>15} {'gain':>8}")
    xs, gains = [], []
    for h in args.bridges:
        bridge = np.inf if h == np.inf else h * hz
        rec = 0
        for s in tracks:
            df = np.array([f for f, d in s if d])
            if not len(df):
                continue
            for f, d in s:
                if d:
                    continue
                b, a = df[df < f], df[df > f]
                if len(b) and len(a) and (a[0] - b[-1]) <= bridge:
                    rec += 1
        ceil = (n_det + rec) / n_pts
        xs.append(h)
        gains.append(rec / n_pts)
        hl = "inf" if h == np.inf else f"{h:.2f}s"
        fl = "-" if h == np.inf else f"{h * hz:.0f}"
        print(f"  {hl:>9} {fl:>8} {rec / n_pts:>11.1%} {ceil:>14.1%} "
              f"{(ceil - n_det / n_pts) * 100:>7.1f}pp")
    # Saturation is measured on the GAIN, which starts near zero -- not on the
    # ceiling, which starts at the current recall and would saturate trivially.
    fin = [x for x in xs if x != np.inf]
    fg = gains[:len(fin)]
    tgt = gains[-1]  # infinite-horizon gain
    print(f"\n    saturates at: 90% of the {tgt:.1%} recoverable by "
          f"{_saturation(fin, fg, 0.90 * tgt / fg[-1] if fg[-1] else 1):.2f}s, "
          f"95% by {_saturation(fin, fg, 0.95 * tgt / fg[-1] if fg[-1] else 1):.2f}s")


def report_precision(tracks, n_frames, hz, run_lens, args):
    lens = np.array([len(t) for t in tracks])
    n_ph = int(lens.sum())
    print(f"\n=== PRECISION HORIZON: {args.detector} @ {args.thresh} ===")
    print(f"{len(tracks)} phantom trajectories over {n_frames} person-free "
          f"frames, {n_ph} detections ({n_ph / n_frames:.2f}/frame)")

    longest_run_s = max(run_lens) / hz
    report_durations(
        "phantom trails", lens / hz,
        censor_note=(f"RIGHT-CENSORED at the observation run: longest run is "
                     f"{longest_run_s:.0f}s,\n      so no trail longer than "
                     f"that can be observed."))

    print(f"\n  A perfect acausal filter dropping every trail shorter than L.")
    print(f"  {'L':>9} {'frames':>8} {'removed':>9} {'remaining/frame':>16} "
          f"{'irreducible':>12}")
    xs, ys = [], []
    for h in args.trails:
        L = max(2, int(round(h * hz)))
        removed = int(lens[lens < L].sum())
        xs.append(h)
        ys.append(removed / n_ph)
        print(f"  {h:>8.2f}s {L:>8} {removed / n_ph:>8.1%} "
              f"{(n_ph - removed) / n_frames:>16.2f} "
              f"{(n_ph - removed) / n_ph:>11.1%}")
    print(f"\n    saturates at: 90% of the removable fraction by "
          f"{_saturation(xs, ys, 0.90):.2f}s, 95% by {_saturation(xs, ys, 0.95):.2f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", choices=["lfe-peaks", "lfe-ppn"],
                    default="lfe-peaks")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--assoc", type=float, default=0.5)
    ap.add_argument("--gate", type=float, default=0.6)
    ap.add_argument("--min-seq-seconds", type=float, default=40.0)
    ap.add_argument("--run-frames", type=int, default=8000)
    ap.add_argument("--recall", action="store_true")
    ap.add_argument("--precision", action="store_true")
    ap.add_argument("--discrimination", action="store_true",
                    help="the precision oracle on POPULATED frames, where a "
                         "trail filter can destroy true positives as well as "
                         "phantoms. The person-free version bounds removal "
                         "only, and A41 showed that is not predictive")
    ap.add_argument("--mode", default="official",
                    choices=["official", "transferred", "balanced"])
    ap.add_argument("--max-gap", type=int, default=0,
                    help="unmatched frames a trail survives before it breaks")
    ap.add_argument("--cache", type=Path, default=_HERE / ".horizon_cache")
    args = ap.parse_args()
    if not (args.recall or args.precision or args.discrimination):
        args.recall = args.precision = True

    args.bridges = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0,
                    60.0, np.inf]
    args.trails = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0,
                   60.0, 120.0]

    args.cache.mkdir(exist_ok=True)
    tag = f"{args.detector}_{args.thresh}_{args.assoc}_{args.gate}"

    if args.recall:
        f = args.cache / f"recall_{tag}.pkl"
        if f.exists():
            tracks, hz = pickle.loads(f.read_bytes())
            print(f"[cached] {f.name}")
        else:
            tracks, hz = collect_recall(args)
            f.write_bytes(pickle.dumps((tracks, hz)))
        report_recall(tracks, hz, args)

    if args.discrimination:
        tag = f"{args.detector}_{args.thresh}_{args.assoc}_{args.gate}"
        f = args.cache / f"disc_{tag}_{args.max_gap}_{args.mode}.pkl"
        if f.exists():
            payload = pickle.loads(f.read_bytes())
            print(f"[cached] {f.name}")
        else:
            payload = collect_discrimination(args)
            f.write_bytes(pickle.dumps(payload))
        report_discrimination(*payload, args)

    if args.precision:
        f = args.cache / f"prec_{tag}_{args.run_frames}.pkl"
        if f.exists():
            tracks, n_frames, hz, run_lens = pickle.loads(f.read_bytes())
            print(f"[cached] {f.name}")
        else:
            tracks, n_frames, hz, run_lens = collect_precision(args)
            f.write_bytes(pickle.dumps((tracks, n_frames, hz, run_lens)))
        report_precision(tracks, n_frames, hz, run_lens, args)


if __name__ == "__main__":
    main()
