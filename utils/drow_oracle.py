"""The oracle analysis on DROW, using DROW's own published detector.

LFE cannot be used here: it is FROG-trained, and on DROW it is a degraded
cross-dataset transfer (memory/interpreting-evaluation.md), so its miss rate
would not be a static detector's miss rate. `DrowDetector` ships the DROW
authors' own published weights, which is the right static baseline for this
dataset.

It is a cutout-based PyTorch model rather than a `detect(scan, angles)` one, so
detections come out of `evaluate_auc`'s shared pipeline -- exactly the same
decoder its AP is computed from.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path("library").resolve()))
sys.path.insert(0, str(Path("utils").resolve()))

from follow_the_drow.datasets import DROW_Dataset
from follow_the_drow.detectors import DrowDetector
from follow_the_drow.utils.drow_utils import (
    laser_angles, laser_minimum, laser_maximum, laser_increment,
    project_cartesian_from_polar,
)
from train import evaluate_auc
from motion_analysis import build_tracks, to_world

ASSOC = 0.5
GATE = 1.25          # 0.5 s between annotations; a walker covers 0.7 m
THRESH = 0.3

ds = DROW_Dataset(verbose=False)
cfg = SimpleNamespace(name="drow", angles_fn=laser_angles, fov_min=laser_minimum,
                      fov_max=laser_maximum, laser_inc=laser_increment)
net = DrowDetector.init(verbose=False)
net.eval()

res = evaluate_auc(net, ds, cfg, eval_r=ASSOC, device="cpu", batch_size=16,
                   return_detections=True)
d = res["detections"]
print(f"DROW3 published weights: wp-AUC {res['wp']:.1%} "
      f"(published 61.9%), {d['n_frames']} annotated frames\n")

# Group detections by frame id, keep those above threshold.
by_frame = {}
keep = d["scores"] >= THRESH
for s_, x_, y_, f_ in zip(d["scores"][keep], d["x"][keep], d["y"][keep],
                          d["frames"][keep]):
    by_frame.setdefault(int(f_), []).append((float(x_), float(y_)))

pairs = d["pairs"]
hz = 1.0 / float(np.median(np.diff(np.asarray(ds.scan_time[0], dtype=float))))

# --------------------------------------------------------------- recall side
tracks_all = []
for seq in range(len(ds.det_id)):
    idx, world, found = [], [], {}
    for fid, (s_, di) in enumerate(pairs):
        if s_ != seq:
            continue
        iscan = int(ds.idet2iscan[seq][di])
        g = np.array([project_cartesian_from_polar(r, p)
                      for r, p in ds.det_wp[seq][di]],
                     dtype=np.float64).reshape(-1, 2)
        ok = np.zeros(len(g), dtype=bool)
        det = np.asarray(by_frame.get(fid, []), dtype=np.float64).reshape(-1, 2)
        if len(g) and len(det):
            cost = np.linalg.norm(g[:, None, :] - det[None, :, :], axis=-1)
            gated = np.where(cost > ASSOC, 1e6, cost)
            for r, c in zip(*linear_sum_assignment(gated)):
                if gated[r, c] < 1e6:
                    ok[r] = True
        w = to_world(g, ds.odoms[seq][iscan]["xya"])
        idx.append(iscan)
        world.append(w)
        for j in range(len(w)):
            found[(iscan, round(float(w[j][0]), 6), round(float(w[j][1]), 6))] = bool(ok[j])
    if not idx:
        continue
    for tr in build_tracks(world, scan_index=idx, gate=GATE):
        seqn = [(int(f), found[(int(f), round(float(tr[f][0]), 6),
                                round(float(tr[f][1]), 6))])
                for f in sorted(tr)
                if (int(f), round(float(tr[f][0]), 6),
                    round(float(tr[f][1]), 6)) in found]
        if seqn:
            tracks_all.append(seqn)

n_pts = sum(len(s) for s in tracks_all)
n_det = sum(sum(x for _, x in s) for s in tracks_all)
print("=== RECALL ORACLE (DROW3, published weights) ===")
print(f"{len(tracks_all)} trajectories, {n_pts} person-observations")
print(f"currently detected {n_det}/{n_pts} = {n_det / n_pts:.1%}  "
      f"(miss {1 - n_det / n_pts:.1%})")

cov = np.array([sum(x for _, x in s) / len(s) for s in tracks_all])
wts = np.array([len(s) for s in tracks_all], dtype=float)
for lab, m in (("Mostly Tracked  (>=80%)", cov >= 0.8),
               ("Partially Tracked", (cov >= 0.2) & (cov < 0.8)),
               ("Mostly Lost     (<20%)", cov < 0.2)):
    print(f"  {lab:<26} {m.sum():>4} ({m.sum() / len(cov):>5.1%})  "
          f"{wts[m].sum() / wts.sum():>5.1%} of observations")

bridge = 2.0 * hz
interp = 0
for s in tracks_all:
    df = np.array([f for f, x in s if x])
    if not len(df):
        continue
    for f, x in s:
        if x:
            continue
        b, a = df[df < f], df[df > f]
        if len(b) and len(a) and (a[0] - b[-1]) <= bridge:
            interp += 1
print(f"  INTERPOLATION bound: recall {n_det / n_pts:.1%} -> "
      f"{(n_det + interp) / n_pts:.1%}  (+{interp / n_pts * 100:.1f} pp)")
print(f"  never detected at all: "
      f"{sum(len(s) for s in tracks_all if not any(x for _, x in s)) / n_pts:.1%}")

# ------------------------------------------------------------ precision side
ph_tracks, n_free = [], 0
for seq in range(len(ds.det_id)):
    run_idx, run_world = [], []
    prev_iscan = None
    for fid, (s_, di) in enumerate(pairs):
        if s_ != seq:
            continue
        iscan = int(ds.idet2iscan[seq][di])
        if len(ds.det_wp[seq][di]) > 0 or (prev_iscan is not None
                                           and iscan - prev_iscan > 5):
            if len(run_idx) >= 4:
                ph_tracks += build_tracks(run_world, scan_index=run_idx,
                                          gate=GATE, max_gap=0)
                n_free += len(run_idx)
            run_idx, run_world = [], []
            prev_iscan = iscan if len(ds.det_wp[seq][di]) == 0 else None
            if len(ds.det_wp[seq][di]) > 0:
                continue
        det = np.asarray(by_frame.get(fid, []), dtype=np.float64).reshape(-1, 2)
        run_idx.append(iscan)
        run_world.append(to_world(det, ds.odoms[seq][iscan]["xya"]))
        prev_iscan = iscan
    if len(run_idx) >= 4:
        ph_tracks += build_tracks(run_world, scan_index=run_idx, gate=GATE, max_gap=0)
        n_free += len(run_idx)

lens = np.array([len(t) for t in ph_tracks]) if ph_tracks else np.array([0])
n_ph = int(lens.sum())
print(f"\n=== PRECISION ORACLE (DROW3) ===")
print(f"{len(ph_tracks)} phantom trajectories over {n_free} person-free annotated "
      f"frames, {n_ph} phantom detections ({n_ph / max(n_free,1):.2f}/frame)")
print(f"  annotated frames are 0.5 s apart, so a trail of L frames is 0.5(L-1) s")
print(f"  {'L (frames)':>11} {'L (s)':>7} {'removed':>9} {'remaining/frame':>16}")
for L in (2, 3, 5, 10, 20):
    removed = int(lens[lens < L].sum())
    print(f"  {L:>11} {0.5 * (L - 1):>7.1f} {removed / max(n_ph,1):>8.1%} "
          f"{(n_ph - removed) / max(n_free,1):>16.2f}")
