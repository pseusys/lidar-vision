"""Role C, classically: a world-frame persistence map over a static detector.

TODO.md A41. The oracle says a 30 s horizon makes **89.7%** of LFE-Peaks' and
**93.9%** of LFE-PPN's phantom detections rejectable, while only **3.6%** of
real person trajectories last that long. That asymmetry is the whole argument
for a long horizon -- but the oracle is acausal and association-gated, so it
bounds a class of methods rather than naming one.

This names the cheapest one. A phantom is a chair: it sits at a fixed place in
the *world* while the robot drives past. So accumulate, per world cell, an
exponentially-forgetting count of how often a detection has been reported
there, and suppress detections standing on cells with too much history. Two
knobs, no training, O(1) per detection per frame.

**Why classical before learned.** A40 measured a post-processing constant
removing false positives 5.8x more cheaply than SORT. The standing lesson is to
price the non-learned lever before proposing a learned one -- and if this
reaches most of the oracle bound, the paper's claim is proven without any
learned recurrence, which is a stronger result rather than a weaker one.

**Why LFE and not DR-SPAAM.** The oracle bound was measured on LFE-Peaks and
LFE-PPN, so only there does this have a target to be scored against. The
bundled DR-SPAAM weights are the official DROW-dataset config, so DR-SPAAM on
FROG would be a zero-shot transfer rather than its published 75.6%.

Usage
-----
    python persistence_map.py                     # both detectors, both splits
    python persistence_map.py --detector lfe-ppn
"""
import argparse
import pickle
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

from motion_analysis import to_world
from nms_sweep import _frame_metrics


def horizon_to_decay(seconds, hz):
    """Decay whose saturation value 1/(1-d) equals the horizon in frames.

    `decay` is an unreadable number; horizons are what the oracle reports and
    what the paper argues about, so the knob is specified in seconds. A cell
    observed every frame forever saturates at exactly `seconds * hz`, which
    makes the suppression threshold readable in frames of evidence.
    """
    return 1.0 - 1.0 / (seconds * hz)


class PersistenceMap:
    """Exponentially-forgetting per-cell evidence, indexed by world location.

    World-indexed rather than beam-indexed because over a 30 s horizon the
    robot travels ~15 m: a per-beam state at that horizon is a memory of
    wherever the beam now points, not of the object. That is the constraint
    that decides the whole design, and it comes from the odometry rather than
    from preference.
    """

    def __init__(self, cell=0.25, decay=0.999):
        self.cell = float(cell)
        self.decay = float(decay)
        self._v = {}                 # (ix, iy) -> [value, last_frame]

    def _key(self, x, y):
        return (int(np.floor(x / self.cell)), int(np.floor(y / self.cell)))

    def query(self, x, y, frame):
        """Evidence already at this location. Read-only, so a detection can be
        judged by what stood there *before* it arrived."""
        e = self._v.get(self._key(x, y))
        if e is None:
            return 0.0
        return e[0] * (self.decay ** (frame - e[1]))

    def add(self, x, y, frame):
        k = self._key(x, y)
        e = self._v.get(k)
        if e is None:
            self._v[k] = [1.0, frame]
        else:
            e[0] = e[0] * (self.decay ** (frame - e[1])) + 1.0
            e[1] = frame


class BackgroundMap:
    """Static-surface model from raw scan returns, per world cell (A44).

    `PersistenceMap` above counts **detections** and fails (A41): people revisit
    locations, so detection frequency conflates a queue with a chair. This
    counts **returns**, which a chair produces whether or not anything detects
    it, and keys the decision on **hit rate** rather than hit count:

        a chair                seen whenever observed    rate ~1, span long
        someone standing 5 s   seen whenever observed    rate ~1, span SHORT
        a busy corridor cell   many people, briefly      rate LOW, span long

    Background is `rate >= min_rate AND span >= min_span`. Each of the three is
    excluded by a different condition, and a count alone excludes neither of the
    last two -- which is precisely how A41 failed.

    **Why this escapes A45's ceiling.** A45 put ~4.2 FP per FN on filtering
    detection *trails*, and showed the limit is association quality. This uses
    no trails and no association, so that bound does not apply to it.

    **Known approximation.** `rate` is hits over the frames spanned by the
    cell's own first and last sighting, not over the frames in which the cell
    was genuinely observable. A surface occluded for a long stretch is
    penalised. Correcting it needs per-ray free-space carving, which is the
    classical occupancy grid (Elfes 1989) and is ~29k cell updates per frame
    against this version's ~720.
    """

    def __init__(self, cell=0.25):
        self.cell = float(cell)
        self._v = {}                 # (ix, iy) -> [hits, first_frame, last_frame]

    def _key(self, x, y):
        return (int(np.floor(x / self.cell)), int(np.floor(y / self.cell)))

    def observe(self, points, frame):
        """Mark every cell holding a return this frame. Once per cell per frame:
        a person at 2 m subtends five times the beams they do at 9 m, and
        counting each would measure range, not persistence."""
        for k in {self._key(float(x), float(y)) for x, y in points}:
            e = self._v.get(k)
            if e is None:
                self._v[k] = [1, frame, frame]
            elif e[2] != frame:
                e[0] += 1
                e[2] = frame

    def rate(self, x, y):
        e = self._v.get(self._key(x, y))
        return 0.0 if e is None else e[0] / max(e[2] - e[1] + 1, 1)

    def span(self, x, y):
        e = self._v.get(self._key(x, y))
        return 0 if e is None else e[2] - e[1] + 1

    def is_background(self, x, y, min_rate, min_span, radius=0.0):
        """Strongest qualifying cell within `radius`. A detection's centroid
        need not land in the same cell as the surface it stands on."""
        n = int(np.ceil(radius / self.cell))
        for dx in range(-n, n + 1):
            for dy in range(-n, n + 1):
                e = self._v.get(self._key(x + dx * self.cell, y + dy * self.cell))
                if e is None:
                    continue
                span = e[2] - e[1] + 1
                if span >= min_span and e[0] / max(span, 1) >= min_rate:
                    return True
        return False


# ------------------------------------------------------------------ detection

def _detector(name):
    return LFEPeaksDetector() if name == "lfe-peaks" else LFEPPNDetector(score_thresh=0.01)


def collect(name, mode, stride, thresh):
    """One detector pass -> per-frame (world detections, world GT, frame index).

    Everything downstream is a sweep over the map's two knobs, so the detector
    runs once and the grid is free -- the pattern the rest of `utils/` uses.
    """
    ds = FROG_Dataset(split="test", mode=mode, auto_download=False, verbose=False)
    det = _detector(name)
    angles = frog_laser_angles(720)

    seqs = []
    for seq in range(len(ds.det_id)):
        frames = []
        for det_idx in range(0, len(ds.det_id[seq]), stride):
            iscan = int(ds.idet2iscan[seq][det_idx])
            scan = ds.scans[seq][iscan]
            xya = ds.odoms[seq][iscan]["xya"]

            d = np.asarray(det.detect(scan, angles), dtype=np.float64).reshape(-1, 3)
            keep = d[:, 0] >= thresh if len(d) else np.zeros(0, dtype=bool)
            d = d[keep]
            dw = to_world(d[:, 1:3], xya) if len(d) else np.empty((0, 2))

            gt = np.array([project_cartesian_from_polar(r, p)
                           for r, p in ds.det_wp[seq][det_idx]],
                          dtype=np.float64).reshape(-1, 2)
            frames.append((iscan, d[:, 1:3].copy(), dw, gt))
        seqs.append(frames)
    return seqs


# --------------------------------------------------------------------- report

def evaluate(seqs, cell, decay, tau, assoc=0.5):
    """Apply the map causally and score what survives."""
    tp = dup = phantom = close = n_gt = n_frames = n_det = n_kept = 0
    for frames in seqs:
        m = PersistenceMap(cell=cell, decay=decay)
        for iscan, d_robot, d_world, gt in frames:
            keep = []
            for j in range(len(d_world)):
                x, y = d_world[j]
                if m.query(x, y, iscan) <= tau:      # judged on prior evidence
                    keep.append(j)
                m.add(x, y, iscan)                   # then contribute
            n_det += len(d_world)
            n_kept += len(keep)
            n_frames += 1
            n_gt += len(gt)
            t, u, p_, c = _frame_metrics(d_robot[keep], gt, assoc)
            tp += t; dup += u; phantom += p_; close += c
    return dict(frames=n_frames, n_gt=n_gt, det=n_det, kept=n_kept,
                tp=tp, dup=dup, phantom=phantom, close=close)


SPANS = (26, 262, 786, 2620)        # 1 s, 10 s, 30 s, 100 s at 26.2 Hz


def background_pass(name, mode, thresh, cell, radius, seqs):
    """One causal pass, reusing A41's cached detections (A44).

    Per detection, record the strongest nearby background `rate` at each of
    several minimum spans, so `min_rate` sweeps analytically afterwards. The
    map is queried *before* the current frame is observed, so a detection is
    judged against the past and not against itself.
    """
    ds = FROG_Dataset(split="test", mode=mode, auto_download=False, verbose=False)
    angles = frog_laser_angles(720).astype(np.float64)
    sin_a, cos_a = -np.sin(angles), np.cos(angles)

    rows, n_gt, n_frames = [], 0, 0
    for seq, frames in enumerate(seqs):
        m = BackgroundMap(cell=cell)
        for iscan, d_robot, d_world, gt in frames:
            is_tp = np.zeros(len(d_robot), dtype=bool)
            if len(d_robot) and len(gt):
                cost = np.linalg.norm(d_robot[:, None, :] - gt[None, :, :], axis=-1)
                gated = np.where(cost > 0.5, 1e6, cost)
                for r, c in zip(*linear_sum_assignment(gated)):
                    if gated[r, c] < 1e6:
                        is_tp[r] = True

            n = int(np.ceil(radius / cell))
            off = [(dx * cell, dy * cell)
                   for dx in range(-n, n + 1) for dy in range(-n, n + 1)]
            for j in range(len(d_world)):
                x, y = d_world[j]
                best = [0.0] * len(SPANS)
                for dx, dy in off:
                    e = m._v.get(m._key(x + dx, y + dy))
                    if e is None:
                        continue
                    sp = e[2] - e[1] + 1
                    rt = e[0] / max(sp, 1)
                    for si, s_min in enumerate(SPANS):
                        if sp >= s_min and rt > best[si]:
                            best[si] = rt
                rows.append([float(is_tp[j])] + best)

            scan = np.asarray(ds.scans[seq][int(iscan)], dtype=np.float64)
            ok = (scan > 0.05) & (scan < 20.0)
            pts = to_world(np.stack([scan[ok] * sin_a[ok], scan[ok] * cos_a[ok]], 1),
                           ds.odoms[seq][int(iscan)]["xya"])
            m.observe(pts, int(iscan))

            n_gt += len(gt)
            n_frames += 1
    return np.asarray(rows, dtype=np.float64), n_gt, n_frames


def report_background(rows, n_gt, n_frames, cell, radius, rates):
    tp_all = float(rows[:, 0].sum())
    fp_all = float(len(rows) - tp_all)
    print("\n  BACKGROUND MODEL from raw scan returns -- cell {:.2f} m, "
          "query radius {:.2f} m".format(cell, radius))
    print("  baseline recall {:.1%}, FP/frame {:.3f}".format(
        tp_all / max(n_gt, 1), fp_all / n_frames))
    print("\n  {:>9} {:>9} {:>12} {:>10} {:>8} {:>10} {:>11}".format(
        "min_span", "min_rate", "FP removed", "TP lost", "recall", "FP/frame", "FP per FN"))
    for si, s_min in enumerate(SPANS):
        for mr in rates:
            drop = rows[:, 1 + si] >= mr
            fp_rm = float((drop & (rows[:, 0] == 0)).sum())
            tp_lost = float((drop & (rows[:, 0] == 1)).sum())
            d_fp, d_fn = fp_rm / n_frames, tp_lost / n_frames
            print("  {:>8.0f}f {:>9.2f} {:>11.1%} {:>9.1%} {:>7.1%} {:>10.3f} "
                  "{:>11.2f}".format(
                      s_min, mr, fp_rm / max(fp_all, 1), tp_lost / max(tp_all, 1),
                      (tp_all - tp_lost) / max(n_gt, 1),
                      (fp_all - fp_rm) / n_frames,
                      (d_fp / d_fn if d_fn else float("inf"))), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", choices=["lfe-peaks", "lfe-ppn", "both"], default="both")
    ap.add_argument("--stride", type=int, default=1,
                    help="MUST stay 1 for a real measurement: the map is a "
                         "filter and has to see every frame the split "
                         "provides, exactly as a tracker does. Striding it "
                         "starves the accumulator and reads as 0% removed."),
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--cell", type=float, nargs="+", default=[0.25, 0.50])
    ap.add_argument("--horizon", type=float, nargs="+", default=[3.0, 10.0, 30.0, 60.0])
    ap.add_argument("--tau", type=float, nargs="+",
                    default=[10, 30, 100, 200, 400],
                    help="evidence needed to suppress, in frames. A cell "
                         "observed every frame saturates at horizon*hz, so "
                         "786 at 30 s -- and a person standing still for 5 s "
                         "reaches ~130, which is what the grid must bracket"),
    ap.add_argument("--hz", type=float, default=26.2)
    ap.add_argument("--background", action="store_true",
                    help="A44: build the map from raw scan RETURNS instead "
                         "of from detections, and key it on hit rate rather "
                         "than hit count. Uses no trails and no association, "
                         "so A45's ceiling does not bind it")
    ap.add_argument("--radius", type=float, default=0.4,
                    help="how far around a detection to look for background")
    ap.add_argument("--min-rate", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7, 0.85, 0.95])
    ap.add_argument("--cache", type=Path, default=_HERE / ".persist_cache")
    args = ap.parse_args()
    args.cache.mkdir(exist_ok=True)

    names = ["lfe-peaks", "lfe-ppn"] if args.detector == "both" else [args.detector]
    for name in names:
        for mode, label in (("transferred", "person-free"), ("official", "populated")):
            f = args.cache / "{}_{}_s{}_{}.pkl".format(name, mode, args.stride, args.thresh)
            if f.exists():
                seqs = pickle.loads(f.read_bytes())
                print("[cached] {}".format(f.name), flush=True)
            else:
                seqs = collect(name, mode, args.stride, args.thresh)
                f.write_bytes(pickle.dumps(seqs))

            if args.background:
                for cell in args.cell:
                    rows, n_gt, n_frames = background_pass(
                        name, mode, args.thresh, cell, args.radius, seqs)
                    print("\n{}\n=== {} / FROG {} ({}) ===\n{}".format(
                        "=" * 96, name, mode, label, "=" * 96))
                    report_background(rows, n_gt, n_frames, cell, args.radius,
                                      args.min_rate)
                continue

            base = evaluate(seqs, 0.25, 0.0, np.inf)
            print("\n{}\n=== {} / FROG {} ({}) ===\n{}".format(
                "=" * 96, name, mode, label, "=" * 96))
            print("{} frames, {} annotated people, {} detections "
                  "({:.3f}/frame) at score >= {}".format(
                      base["frames"], base["n_gt"], base["det"],
                      base["det"] / base["frames"], args.thresh))
            if mode == "official":
                print("baseline recall {:.1%}, FP/frame {:.3f}".format(
                    base["tp"] / max(base["n_gt"], 1),
                    (base["dup"] + base["phantom"]) / base["frames"]))

            print("\n  {:>6} {:>9} {:>7} {:>9} {:>9} {:>9} {:>9}".format(
                "cell", "horizon", "tau", "kept", "removed", "recall", "FP/fr"))
            for cell in args.cell:
                for h in args.horizon:
                    decay = horizon_to_decay(h, args.hz)
                    for tau in args.tau:
                        r = evaluate(seqs, cell, decay, tau)
                        removed = 1.0 - r["kept"] / max(base["det"], 1)
                        rec = r["tp"] / max(r["n_gt"], 1) if r["n_gt"] else float("nan")
                        print("  {:>6.2f} {:>8.0f}s {:>7.0f} {:>9d} {:>8.1%} "
                              "{:>8.1%} {:>9.3f}".format(
                                  cell, h, tau, r["kept"], removed, rec,
                                  (r["dup"] + r["phantom"]) / r["frames"]), flush=True)


if __name__ == "__main__":
    main()
