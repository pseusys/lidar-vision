#!/usr/bin/env python3
"""`TODO.md` item 64: the frame-convention check on **real recorded DROW frames**, which rule 4 demands.

`tests/test_upstream_scoring.py` pins that their offset step and ours are the same formula and that their
Cartesian frame is ours rotated 90 degrees. Those are unit tests on synthetic scans, and this repository has
twice shipped a dataset-logic sign error that passed clean synthetic tests and was only caught by looking at
real recorded data. So the claim that cell 2 of the 2x2 is sound -- *their* suppression feeding *our* ground
truth -- is checked here, against real scans and real annotations.

**The question, narrowly.** Every beam the target generator marks as a person is given an oracle vote: the exact
offset to its own annotation, in the beam frame both trees agree on, built by `train.make_targets`, which is the
function training itself uses. A decoder wired correctly must then put a detection on that annotation, measured
in the frame `_process_detections` matches in. Centimetres means the bridge is right. A 90-degree error puts the
detection metres away at DROW's ranges, and no amount of averaging hides it.

This is deliberately not a network: a trained head's own error would mask exactly the error being looked for.

    .venv/Scripts/python.exe utils/check_upstream_convention.py --frames 300
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from takheliped.datasets.drow_dataset import DROW_Dataset
from takheliped.utils.drow_utils import (
    _win2global, laser_angles, project_cartesian_from_polar, votes_to_detections,
)

from train import make_targets

import upstream_scoring as us

# `evaluate_auc`'s own decoder settings, so "ours" here is the path that produced the 48.1%.
V2D = {"blur_sigma": 2.0, "blur_win": 11, "bin_size": 0.1,
       "vote_collect_radius": 0.5, "min_thresh": 1e-3}

WP = 3          # this repository's class coding: 0=bg, 1=wc, 2=wa, 3=wp
CONF_KEEP = 0.5  # oracle confidences are 0 or 1, so any mid threshold separates them


def _gt_xy(annotations) -> np.ndarray:
    """Annotations `(r, phi)` as x-y in this project's frame -- the frame the metric matches in."""
    if len(annotations) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    r = np.array([a[0] for a in annotations], dtype=np.float32)
    phi = np.array([a[1] for a in annotations], dtype=np.float32)
    return np.stack(project_cartesian_from_polar(r, phi), axis=1)


def _ours(scan: np.ndarray, angles: np.ndarray, conf: np.ndarray, votes: np.ndarray) -> np.ndarray:
    """Our decoder: every beam's vote into a blurred grid, exactly as `evaluate_auc` does it."""
    probas = np.zeros((1, len(scan), 4), dtype=np.float32)
    probas[0, :, 0] = 1.0 - conf
    probas[0, :, WP] = conf

    r_new, phi_new = _win2global(scan[None], angles[None], votes[None, :, 0], votes[None, :, 1])
    xs, ys = r_new * -np.sin(phi_new), r_new * np.cos(phi_new)

    kept = [(d[0], d[1]) for d in votes_to_detections(xs, ys, probas, **V2D)[0] if d[2][WP] > CONF_KEEP]
    return np.array(kept, dtype=np.float32).reshape(-1, 2)


def _theirs(scan: np.ndarray, angles: np.ndarray, conf: np.ndarray, votes: np.ndarray) -> np.ndarray:
    """Their decoder, through the bridge under test. Zero-confidence beams vote at themselves; drop them."""
    det_xy, det_scores = us.nms_detections(scan[None], angles, conf, votes, min_dist=0.5)
    return det_xy[det_scores > CONF_KEEP]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=300,
                        help="annotated frames to stride down to (0 = all)")
    parser.add_argument("--vote-collect-radius", type=float, default=0.6,
                        help="make_targets' positive radius, in metres")
    parser.add_argument("--tolerance", type=float, default=0.1,
                        help="a decoded detection this close to its annotation counts as placed, in metres")
    args = parser.parse_args()

    dataset = DROW_Dataset(time_frame_size=1)
    angles = laser_angles(dataset.scans[0].shape[1])

    pairs = [(seq, idx)
             for seq in range(len(dataset.det_id))
             for idx in range(len(dataset.det_id[seq]))]
    if args.frames and len(pairs) > args.frames:
        pairs = pairs[::(len(pairs) + args.frames - 1) // args.frames]

    errors = {"ours": [], "theirs": []}
    frames_used = 0

    for seq, det_idx in pairs:
        annotations = list(dataset.det_wp[seq][det_idx])
        if not annotations:
            continue

        scan = np.asarray(dataset.scans[seq][dataset.idet2iscan[seq][det_idx]], dtype=np.float32)
        gt_per_class = {1: list(dataset.det_wc[seq][det_idx]),
                        2: list(dataset.det_wa[seq][det_idx]),
                        WP: annotations}

        labels, votes = make_targets(scan, angles, gt_per_class,
                                     vote_collect_radius=args.vote_collect_radius, frame="beam")
        conf = (labels == WP).astype(np.float32)
        if not conf.any():
            continue

        votes = votes.astype(np.float32)
        gt_xy = _gt_xy(annotations)
        frames_used += 1

        for name, decoded in (("ours", _ours(scan, angles, conf, votes)),
                              ("theirs", _theirs(scan, angles, conf, votes))):
            for target in gt_xy:
                if len(decoded) == 0:
                    errors[name].append(np.inf)
                else:
                    errors[name].append(float(np.min(np.hypot(*(decoded - target).T))))

    if not frames_used:
        print("No annotated person frames found -- nothing was checked.")
        return 1

    print(f"\n  Oracle votes on {frames_used} real DROW frames, "
          f"{len(errors['ours'])} person annotations\n")
    print(f"  {'decoder':<10}{'median':>10}{'p90':>10}{'max':>10}{'placed':>10}")
    print(f"  {'-' * 50}")

    worst = 0.0
    for name in ("ours", "theirs"):
        err = np.array(errors[name], dtype=np.float64)
        finite = err[np.isfinite(err)]
        placed = float(np.mean(err <= args.tolerance)) * 100.0
        print(f"  {name:<10}{np.median(finite):>9.3f}m{np.percentile(finite, 90):>9.3f}m"
              f"{finite.max():>9.3f}m{placed:>9.1f}%")
        worst = max(worst, float(np.median(finite)))

    verdict = worst <= args.tolerance
    print(f"\n  {'PASS' if verdict else 'FAIL'}: median placement error is "
          f"{'within' if verdict else 'outside'} {args.tolerance:.2f} m for both decoders.")
    print("  A 90-degree frame error would read as metres here, not centimetres.\n")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
