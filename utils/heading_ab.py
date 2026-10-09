#!/usr/bin/env python3
"""Why did the displacement oracle move from 4.20 to 2.58 when the heading-wrap bug was fixed? (`TODO.md` §A item 3.)

Until 2026-09-14 `FROG_Dataset._load_odom` interpolated heading **wrapped**, so a pair of odometry samples straddling +-180
degrees interpolated the long way round: 54 single-frame jumps above 2 degrees on the test recording, up to 144 degrees. Fixing
it changed A45's role-C displacement bound from **4.20 to 2.58** false positives removed per true positive lost -- and the two
obvious explanations both predict the opposite sign:

    inflated boxes     a spurious rotation moves every detection, which *enlarges* a trail's world-frame bounding box, so
                       fewer trails fall under the "box smaller than D" rule -- fewer phantoms removed, not more
    broken trails      the same jumps break association, which *shortens* person trails and shrinks their boxes, so more
                       people are suppressed -- a worse exchange rate, not a better one

So it is measured instead of argued. The same cached detections are re-projected through both pose series, with everything else
held constant -- association gate, assignment rule, bounding-box rule, thresholds -- and the two displacement tables printed
side by side. The fixed column must reproduce `horizon_sweep.py --discrimination`, or this file is measuring something else.

The detections come from `utils/.persist_cache/lfe-peaks_official_s1_0.3.pkl`, which stores each frame's detections in **sensor**
coordinates beside its world ones, so no detector pass is needed.

Usage (from utils/)
-----
    python heading_ab.py
    python heading_ab.py --limit-sessions 5        # smoke run
"""

import argparse
import json
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.datasets.frog_dataset import FROG_Dataset  # noqa: E402
from motion_analysis import build_tracks, to_world  # noqa: E402

CACHE = _HERE / ".persist_cache" / "lfe-peaks_official_s1_0.3.pkl"
ODOM = _HERE.parent / "library" / "follow_the_drow" / "include" / "FROG-data" / "frog_16-41_odom.npz"
DISPLACEMENTS_M = (0.10, 0.20, 0.30, 0.50, 0.80, 1.20, 2.00, 3.00, 5.00)
ASSOC_M = 0.5               # `horizon_sweep --assoc`
GATE_M = 0.6                # `horizon_sweep --gate`
MAX_GAP = 0                 # `horizon_sweep --max-gap`
UNASSIGNED = 1e6


def naive_heading(at: np.ndarray, odom_t: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """The interpolation the loader used until 2026-09-14: linear in the *wrapped* angle, so a crossing takes the long way."""
    return np.interp(at, odom_t, heading)


def unwrapped_heading(at: np.ndarray, odom_t: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Today's interpolation: unwrap, interpolate, wrap back, so a turn across +-pi takes the short way."""
    interpolated = np.interp(at, odom_t, np.unwrap(heading))
    return np.arctan2(np.sin(interpolated), np.cos(interpolated))


def poses(timestamps: np.ndarray, odom: Dict[str, np.ndarray], buggy: bool) -> np.ndarray:
    """`[N, 3]` of x, y and heading at each scan time, with the heading interpolated one way or the other."""
    odom_t, data = odom["ts"].astype(float), odom["data"].astype(float)
    turn = naive_heading if buggy else unwrapped_heading
    return np.stack([np.interp(timestamps, odom_t, data[:, 0]),
                     np.interp(timestamps, odom_t, data[:, 1]),
                     turn(timestamps, odom_t, data[:, 2])], axis=1)


def person_labels(detections: np.ndarray, people: np.ndarray) -> np.ndarray:
    """Which of this frame's detections cover an annotated person, one to one within `ASSOC_M` -- the rule the oracle uses."""
    labels = np.zeros(len(detections), dtype=bool)
    if not len(detections) or not len(people):
        return labels
    cost = np.linalg.norm(detections[:, None, :] - people[None, :, :], axis=-1)
    gated = np.where(cost > ASSOC_M, UNASSIGNED, cost)
    for row, column in zip(*linear_sum_assignment(gated)):
        if gated[row, column] < UNASSIGNED:
            labels[row] = True
    return labels


def trails(frames: List[Tuple[int, np.ndarray, np.ndarray]], pose: np.ndarray, gate: float) -> List[Tuple[float, int, int]]:
    """Per trail: its world-frame bounding-box diagonal, and how many person and phantom detections it holds."""
    index, world, label = [], [], {}
    for scan, sensor, people in frames:
        placed = to_world(sensor, pose[scan]) if len(sensor) else np.empty((0, 2))
        covers = person_labels(sensor, people)
        index.append(scan)
        world.append(placed)
        for j in range(len(placed)):
            label[(scan, round(float(placed[j][0]), 6), round(float(placed[j][1]), 6))] = bool(covers[j])
    out = []
    for track in build_tracks(world, scan_index=index, gate=gate, max_gap=MAX_GAP):
        order = sorted(track)
        person = sum(label.get((int(f), round(float(track[f][0]), 6), round(float(track[f][1]), 6)), False) for f in order)
        xy = np.array([track[f] for f in order], dtype=np.float64)
        box = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
        out.append((box, person, len(order) - person))
    return out


def table(tracked: List[Tuple[float, int, int]], frames: int) -> List[Dict[str, float]]:
    """The displacement oracle: suppress every trail whose box is under D, and price it as false positives per lost person."""
    people = sum(t[1] for t in tracked)
    phantoms = sum(t[2] for t in tracked)
    rows = []
    for limit in DISPLACEMENTS_M:
        removed = sum(t[2] for t in tracked if t[0] < limit)
        lost = sum(t[1] for t in tracked if t[0] < limit)
        rows.append(dict(displacement_m=limit, fp_removed=removed / max(phantoms, 1), tp_lost=lost / max(people, 1),
                         fp_per_frame=(phantoms - removed) / max(frames, 1), exchange=removed / lost if lost else float("inf")))
    return rows


def report(name: str, rows: List[Dict[str, float]]) -> None:
    print(f"\n  {name}")
    print(f"  {'D':>7} {'FP removed':>12} {'TP lost':>10} {'FP/frame':>10} {'FP per FN':>11}")
    for row in rows:
        print(f"  {row['displacement_m']:>6.2f}m {row['fp_removed']:>11.1%} {row['tp_lost']:>9.1%} "
              f"{row['fp_per_frame']:>10.3f} {row['exchange']:>11.2f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit-sessions", type=int, default=0, help="only the first N sessions (smoke runs)")
    ap.add_argument("--gate", type=float, default=GATE_M, help="association gate in metres: a looser gate merges trails and inflates "
                    "every bounding box, which is the other candidate for the recorded 4.20")
    ap.add_argument("--out", type=Path, default=_HERE.parent / "results" / "heading_ab.json")
    args = ap.parse_args()

    if not CACHE.exists():
        raise SystemExit(f"missing {CACHE} -- run `python persistence_map.py` first")
    sessions = pickle.loads(CACHE.read_bytes())
    if args.limit_sessions:
        sessions = sessions[:args.limit_sessions]
    odom = dict(np.load(ODOM))
    ds = FROG_Dataset(split="test", mode="official", auto_download=False, verbose=False)
    print(f"{len(sessions)} sessions, {sum(len(s) for s in sessions)} annotated frames, gate {args.gate:g} m", flush=True)

    collected = {"fixed": [], "buggy": []}
    jumps, frames = 0, 0
    for seq, session in enumerate(sessions):
        stamps = np.asarray(ds.scan_time[seq], dtype=np.float64)
        pose = {kind: poses(stamps, odom, buggy=(kind == "buggy")) for kind in collected}
        disagreement = np.abs(np.angle(np.exp(1j * (pose["buggy"][:, 2] - pose["fixed"][:, 2]))))
        jumps += int((disagreement > np.radians(2.0)).sum())
        rows = [(int(scan), np.asarray(sensor, dtype=np.float64).reshape(-1, 2), np.asarray(people, dtype=np.float64).reshape(-1, 2))
                for scan, sensor, _, people in session]
        frames += len(rows)
        for kind in collected:
            collected[kind].extend(trails(rows, pose[kind], args.gate))

    print(f"{jumps} frames where the two headings disagree by more than 2 degrees "
          f"({100 * jumps / max(frames, 1):.2f}% of the annotated ones)", flush=True)
    result = {}
    for kind, tracked in collected.items():
        result[kind] = dict(trails=len(tracked), people=sum(t[1] for t in tracked), phantoms=sum(t[2] for t in tracked),
                            median_box_m=float(np.median([t[0] for t in tracked])), rows=table(tracked, frames))
        report(f"{kind} heading: {len(tracked)} trails, median box {result[kind]['median_box_m']:.3f} m", result[kind]["rows"])

    best = {kind: max(r["exchange"] for r in result[kind]["rows"] if r["exchange"] != float("inf")) for kind in result}
    print(f"\n  best exchange rate: fixed {best['fixed']:.2f}, buggy {best['buggy']:.2f} "
          f"(recorded: 2.58 after the fix, 4.20 before)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(frames=frames, disagreeing_frames=jumps, best_exchange=best, **result), indent=1),
                        encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
        code = 0
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)
