#!/usr/bin/env python3
"""Where DROW's labels sit on its scans, and what each label radius takes in, measured on the real files.

Model-free, per split (`AGENTS.md` rule 4: a dataset-logic claim is checked on recorded data):

1. **Alignment.** The share of annotated people with a scan return within `NEAR_M` of their centre, at the bearings
   as loaded and at every shift of up to `MAX_SHIFT` beams either way, mirrored or not. A loader that placed labels
   on the wrong beams or on the wrong side of the robot would score best somewhere other than "as loaded, unshifted".
2. **The positive disk.** How many beams a person covers at 0.6 m (every stage 2 on file) and at 0.35 m (DROW's own
   person radius), and how many of those `make_targets` hands to a wheelchair or a walker when every class is a
   target -- at 0.6 m for all of them under the nearest assignment, which is what the multi-class arms trained, and
   at DROW's own radii.
3. **Double labels.** The share of walkers and of wheelchairs with a person annotation within `DOUBLE_M`, which
   DROW's protocol of labelling a walker user twice (person and walker) predicts for the walkers.

    python drow_label_geometry.py
    python drow_label_geometry.py --splits test
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from json import dumps
from pathlib import Path
from typing import Dict

import numpy as np

from takheliped.datasets import DROW_Dataset
from takheliped.utils.drow_utils import laser_angles, laser_increment
from train import make_targets

_HERE = Path(__file__).parent
SPLITS = ("train", "val", "test")
#: A return this close to a label's centre counts as the label sitting on its own object.
NEAR_M = 0.2
MAX_SHIFT = 4
#: Every stage 2 on file, and DROW's own person radius.
PERSON_RADII_M = (0.6, 0.35)
#: DROW's and DR-SPAAM's own radii by DROW's label ids: 1 wheelchair, 2 walker, 3 person.
DROW_RADII = {1: 0.6, 2: 0.4, 3: 0.35}
ADOPTED_RADIUS_M = 0.6
DOUBLE_M = 0.3
WHEELCHAIR, WALKER, PERSON = 1, 2, 3


def polar(labels) -> np.ndarray:
    return np.asarray(labels, np.float64).reshape(-1, 2)


def to_xy(rphi: np.ndarray) -> np.ndarray:
    return np.stack([rphi[:, 0] * -np.sin(rphi[:, 1]), rphi[:, 0] * np.cos(rphi[:, 1])], axis=1)


def nearest_distance(points: np.ndarray, to: np.ndarray) -> np.ndarray:
    """Each of `points`' distance to the nearest of `to`, infinite when `to` is empty."""
    if not len(to):
        return np.full(len(points), np.inf)
    return np.linalg.norm(points[:, None] - to[None], axis=-1).min(axis=1)


def measure(ds: DROW_Dataset) -> Dict[str, object]:
    angles = laser_angles(ds.scans[0].shape[1]).astype(np.float64)
    shifts = range(-MAX_SHIFT, MAX_SHIFT + 1)
    near = {(mirror, shift): 0 for mirror in (False, True) for shift in shifts}
    counts = dict(frames=0, people=0, wheelchairs=0, walkers=0)
    disk = {r: dict(beams=0, taken_at_adopted=0, taken_at_drow=0) for r in PERSON_RADII_M}
    doubles = {"walkers": 0, "wheelchairs": 0}
    for s in range(len(ds.det_id)):
        for d in range(len(ds.det_id[s])):
            scan = np.asarray(ds.scans[s][ds.idet2iscan[s][d]], np.float64)
            ends = np.stack([scan * -np.sin(angles), scan * np.cos(angles)], axis=1)
            people, chairs, walkers = polar(ds.det_wp[s][d]), polar(ds.det_wc[s][d]), polar(ds.det_wa[s][d])
            counts["frames"] += 1
            counts["people"] += len(people)
            counts["wheelchairs"] += len(chairs)
            counts["walkers"] += len(walkers)
            for mirror, shift in near:
                moved = people.copy()
                moved[:, 1] = (-moved[:, 1] if mirror else moved[:, 1]) + shift * laser_increment
                near[(mirror, shift)] += int((nearest_distance(to_xy(moved), ends) < NEAR_M).sum())
            if not len(people):
                continue
            everything = {WHEELCHAIR: [tuple(o) for o in chairs], WALKER: [tuple(o) for o in walkers],
                          PERSON: [tuple(p) for p in people]}
            adopted, _ = make_targets(scan, angles, everything, vote_collect_radius=ADOPTED_RADIUS_M, nearest=True)
            published, _ = make_targets(scan, angles, everything, vote_collect_radius=DROW_RADII, nearest=True)
            for r in PERSON_RADII_M:
                alone, _ = make_targets(scan, angles, {PERSON: everything[PERSON]}, vote_collect_radius=r, nearest=True)
                mine = alone == PERSON
                disk[r]["beams"] += int(mine.sum())
                disk[r]["taken_at_adopted"] += int((mine & (adopted != PERSON) & (adopted > 0)).sum())
                disk[r]["taken_at_drow"] += int((mine & (published != PERSON) & (published > 0)).sum())
            people_xy = to_xy(people)
            doubles["walkers"] += int((nearest_distance(to_xy(walkers), people_xy) < DOUBLE_M).sum())
            doubles["wheelchairs"] += int((nearest_distance(to_xy(chairs), people_xy) < DOUBLE_M).sum())
    best = max(near, key=near.get)
    return dict(**counts,
                alignment={f"{'mirrored' if m else 'as loaded'} {k:+d}": n / max(counts["people"], 1) for (m, k), n in near.items()},
                best_alignment=f"{'mirrored' if best[0] else 'as loaded'} {best[1]:+d}",
                person_disk={str(r): {**v, "beams_per_person": v["beams"] / max(counts["people"], 1)} for r, v in disk.items()},
                walkers_with_a_person=doubles["walkers"] / max(counts["walkers"], 1),
                wheelchairs_with_a_person=doubles["wheelchairs"] / max(counts["wheelchairs"], 1))


def main() -> int:
    ap = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    ap.add_argument("--out", type=Path, default=_HERE.parent / "results" / "drow_label_geometry.json")
    args = ap.parse_args()
    results = {}
    for name in args.splits:
        m = measure(DROW_Dataset(split=name, verbose=False))
        results[name] = m
        print(f"\n{name}: {m['frames']} frames, {m['people']} people, {m['wheelchairs']} wheelchairs, {m['walkers']} walkers")
        print(f"  a return within {NEAR_M} m of a person's centre: as loaded {m['alignment']['as loaded +0']:.1%}, "
              f"best {m['best_alignment']} at {max(m['alignment'].values()):.1%}")
        for r, v in m["person_disk"].items():
            print(f"  person disk {r} m: {v['beams_per_person']:.1f} beams per person; taken by another class "
                  f"{v['taken_at_adopted'] / max(v['beams'], 1):.1%} at 0.6 m for every class, "
                  f"{v['taken_at_drow'] / max(v['beams'], 1):.1%} at DROW's radii")
        print(f"  a person within {DOUBLE_M} m: {m['walkers_with_a_person']:.1%} of walkers, "
              f"{m['wheelchairs_with_a_person']:.1%} of wheelchairs", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
