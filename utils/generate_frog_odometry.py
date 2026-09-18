"""
Generate pseudo-odometry via ICP-based scan matching, and write it to the
``<h5-stem>_odom.npz`` sidecar file that
FROG_Dataset._load_odom() already knows how to pick up automatically

**Rarely needed now.** FROG does publish real per-recording odometry, all of
it is downloaded by ``FROG_Dataset.download()``, and ``_load_odom()`` prefers
it and raises rather than silently faking when it is absent (TODO.md A14).
This script is the fallback for a recording that genuinely has none.
-- no changes to frog_dataset.py are needed; this script only needs to
produce that file. See library/follow_the_drow/utils/odometry_estimation.py
for the estimator itself and its verification tests
(tests/test_odometry_estimation.py) -- run those first if this file has been
touched.

Usage:
    python generate_frog_odometry.py [--datapath PATH] [--inlier-thresh M]

Writes one ``<stem>_odom.npz`` per ``*.h5`` file found in --datapath,
covering every scan in that file regardless of its train/val/test split
(the sidecar's `ts`/`data` arrays are looked up by timestamp, not split).
"""
import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from follow_the_drow.datasets.frog_dataset import frog_laser_angles, _DATASET_PATH
from follow_the_drow.utils.odometry_estimation import integrate_trajectory


def process_h5(h5_path: Path, inlier_thresh: float, min_inliers: int) -> None:
    with h5py.File(h5_path, "r") as h5:
        scans = h5["scans"][:].astype(np.float32)
        timestamps = h5["timestamps"][:].astype(np.float64)
    scans = np.where(np.isfinite(scans), scans, 10.0)
    angles = frog_laser_angles(scans.shape[1])

    print(f"=== {h5_path.name} ({len(scans)} scans) ===")
    dt = np.diff(timestamps)
    gap_mask = dt > 0.1
    if gap_mask.any():
        print(f"  {gap_mask.sum()} recording gap(s) > 0.1s found "
              f"(max {dt.max():.1f}s) -- these transitions will be forced "
              f"to zero relative motion rather than trusting ICP across them")
    t0 = time.time()
    traj, inliers = integrate_trajectory(
        scans, angles, timestamps=timestamps,
        inlier_thresh=inlier_thresh, min_inliers=min_inliers)
    elapsed = time.time() - t0
    print(f"  integrated in {elapsed:.1f}s ({elapsed / len(scans) * 1000:.2f}ms/frame)")

    step_dist = np.sqrt(np.diff(traj[:, 0]) ** 2 + np.diff(traj[:, 1]) ** 2)
    low_inlier_frac = (inliers[1:] < min_inliers * 1.5).mean()
    print(f"  x range: [{traj[:,0].min():.2f}, {traj[:,0].max():.2f}]  "
          f"y range: [{traj[:,1].min():.2f}, {traj[:,1].max():.2f}]")
    print(f"  step distance: median={np.median(step_dist):.4f}m  "
          f"p99={np.percentile(step_dist, 99):.4f}m  max={step_dist.max():.4f}m")
    print(f"  frames with low inlier count (<{int(min_inliers*1.5)}): {low_inlier_frac:.2%}")
    print(f"  any non-finite: {not np.all(np.isfinite(traj))}")

    out_path = h5_path.with_name(h5_path.stem + "_odom.npz")
    np.savez(out_path, ts=timestamps, data=traj.astype(np.float64))
    print(f"  -> wrote {out_path}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datapath", type=Path, default=_DATASET_PATH,
                        help="Directory containing FROG .h5 files (default: "
                             "the package's own include/FROG-data/).")
    parser.add_argument("--inlier-thresh", type=float, default=0.15,
                        help="ICP inlier distance threshold in metres (default 0.15).")
    parser.add_argument("--min-inliers", type=int, default=10,
                        help="Minimum ICP inlier count to trust a frame's estimate "
                             "(default 10).")
    args = parser.parse_args()

    h5_files = sorted(Path(args.datapath).glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {args.datapath}")
    for h5_path in h5_files:
        process_h5(h5_path, args.inlier_thresh, args.min_inliers)


if __name__ == "__main__":
    main()
