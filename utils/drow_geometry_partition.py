#!/usr/bin/env python3
"""DROW repartitioned by scan geometry, and the control built the same way: writes `takheliped/datasets/drow_partitions.py`.

DROW's official test recordings are enclosed and its training recordings open -- the median return within 10 m is 1.74 m
on test against 4.33 m on train -- and on held-out val our stage 2's scores stop comparing between scans exactly on
the enclosed ones (`CHANGELOG.md`, 2026-11-09). Retraining on a partition where every split carries DROW's overall
share of enclosed scans asks whether that is the cause (`TODO.md` row 106).

**What is preserved, the owner's condition (2026-11-09): as much of original DROW as possible.**

- **The unit is a place.** A recording's lettered segments (`-a`, `-b`, ...) go together, so no scene crosses a split --
  which the official partition does not guarantee: 26% of its val's people sit in a sister segment of a train recording.
- **The composition is the official one.** The staged, lunch and turning recordings stay in train, as officially; val
  and test hold driving runs only, as officially.
- **The sizes are the official ones**, every split within 5% of its annotated scans.
- **Everything else stays put.** An exact integer program (SciPy's `milp`) moves the fewest annotated scans away from
  the official split that the constraints allow.

**The two partitions.** `geometry-balanced`: every split's share of enclosed scans -- a median return range under
official train's 10th percentile -- within 2 points of DROW's overall share. `geometry-shifted`, the control: the
official test exactly, and train and val made leak-free by the same rule, so it differs from `geometry-balanced` in the
geometry and from the official split in the leak alone.

    python drow_geometry_partition.py
"""

import re
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from drow_locality_property import annotated_recordings, recording_type, scan_statistics
from train_takheliped import DROW, frames_from_dataset

SPLITS = ("train", "val", "test")
SIZE_TOLERANCE = 0.05
SHARE_TOLERANCE = 0.02
#: An enclosed scan's median return range is under this percentile of official train's.
ENCLOSED_PERCENTILE = 10
OUT = Path(__file__).resolve().parents[1] / "library" / "takheliped" / "datasets" / "drow_partitions.py"


@dataclass(frozen=True)
class Group:
    """One place: every segment of one recording, its kind, its annotated and enclosed scans, and where the official
    partition put its annotated scans."""
    name: str
    kind: str
    annotated: int
    enclosed: int
    official: Mapping[str, int]


def solve_partition(groups: Sequence[Group], allowed: Mapping[str, Sequence[str]], targets: Mapping[str, int], *,
                    size_tolerance: float, enclosed_share: Optional[float], share_tolerance: float) -> Dict[str, str]:
    """Each group's split, moving the fewest annotated scans from where the official partition put them, with every
    split's size within `size_tolerance` of `targets` and, given `enclosed_share`, its share of enclosed scans within
    `share_tolerance` of it. A group goes only to a split `allowed` names for it. Raises if nothing satisfies all that."""
    cells = [(g, s) for g in groups for s in allowed[g.name]]
    cost = np.array([g.annotated - g.official.get(s, 0) for g, s in cells], np.float64)
    rows, lower, upper = [], [], []
    for g in groups:
        rows.append([1.0 if c[0] is g else 0.0 for c in cells])
        lower.append(1.0)
        upper.append(1.0)
    for s in SPLITS:
        rows.append([float(g.annotated) if t == s else 0.0 for g, t in cells])
        lower.append((1 - size_tolerance) * targets[s])
        upper.append((1 + size_tolerance) * targets[s])
        if enclosed_share is not None:
            # share - tol <= enclosed / size <= share + tol, both sides multiplied out by the size
            rows.append([g.enclosed - (enclosed_share - share_tolerance) * g.annotated if t == s else 0.0 for g, t in cells])
            lower.append(0.0)
            upper.append(np.inf)
            rows.append([g.enclosed - (enclosed_share + share_tolerance) * g.annotated if t == s else 0.0 for g, t in cells])
            lower.append(-np.inf)
            upper.append(0.0)
    result = milp(cost, constraints=LinearConstraint(np.array(rows), lower, upper), integrality=np.ones(len(cells)), bounds=Bounds(0, 1))
    if result.status != 0:
        raise ValueError(f"no partition satisfies the constraints: {result.message}")
    return {g.name: s for (g, s), x in zip(cells, result.x, strict=True) if x > 0.5}


def place(stem: str) -> str:
    """The recording a segment belongs to. DROW's files are `<name>.bag.csv`, so a stem ends in `.bag`, which has to
    go before the segment letter can be seen -- missed once, it made every segment a place of its own."""
    return re.sub(r"-[a-z]$", "", stem.removesuffix(".bag"))


def measure() -> tuple:
    """Every official recording's stem, split and stats, the groups they form, and the enclosed threshold."""
    rows, train_medians = [], None
    for split_name in SPLITS:
        ds = DROW.load(split_name, None, "official")
        split = frames_from_dataset(ds, DROW)
        medians = scan_statistics(split.scans[split.annotated], split.people_xy, DROW.spacing_deg)["median_range_m"]
        if split_name == "train":
            train_medians = medians
        rec = annotated_recordings(split)
        people = np.array([len(p) for p in split.people_xy])
        for r, f in enumerate(ds.filenames):
            rows.append(dict(stem=Path(f).name, split=split_name, medians=medians[rec == r], people=int(people[rec == r].sum())))
    threshold = float(np.nanpercentile(train_medians, ENCLOSED_PERCENTILE))
    by_place = defaultdict(list)
    for row in rows:
        row["enclosed"] = int(np.sum(row["medians"] < threshold))
        by_place[place(row["stem"])].append(row)
    groups = []
    for name, members in sorted(by_place.items()):
        official = defaultdict(int)
        for m in members:
            official[m["split"]] += len(m["medians"])
        groups.append(Group(name, recording_type(members[0]["stem"]), sum(len(m["medians"]) for m in members),
                            sum(m["enclosed"] for m in members), dict(official)))
    return rows, groups, threshold


def summary(rows, split_of_stem: Mapping[str, str]) -> Dict[str, Dict[str, int]]:
    out = {s: dict(recordings=0, annotated=0, enclosed=0, people=0) for s in SPLITS}
    for row in rows:
        cell = out[split_of_stem[row["stem"]]]
        cell["recordings"] += 1
        cell["annotated"] += len(row["medians"])
        cell["enclosed"] += row["enclosed"]
        cell["people"] += row["people"]
    return out


def main() -> int:
    ap = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rows, groups, threshold = measure()
    official = {row["stem"]: row["split"] for row in rows}
    targets = {s: sum(len(r["medians"]) for r in rows if r["split"] == s) for s in SPLITS}
    share = sum(g.enclosed for g in groups) / sum(g.annotated for g in groups)
    official_test = {place(r["stem"]) for r in rows if r["split"] == "test"}
    designs = {
        "geometry-balanced": ({g.name: SPLITS if g.kind == "driving" else ("train",) for g in groups}, share),
        "geometry-shifted": ({g.name: ("test",) if g.name in official_test else ("train", "val") if g.kind == "driving" else ("train",)
                              for g in groups}, None),
    }
    tables = {"official": {s: sorted(r["stem"] for r in rows if r["split"] == s) for s in SPLITS}}
    summaries = {"official": summary(rows, official)}
    for mode, (allowed, enclosed_share) in designs.items():
        assignment = solve_partition(groups, allowed, targets, size_tolerance=SIZE_TOLERANCE, enclosed_share=enclosed_share,
                                     share_tolerance=SHARE_TOLERANCE)
        split_of_stem = {r["stem"]: assignment[place(r["stem"])] for r in rows}
        tables[mode] = {s: sorted(stem for stem, t in split_of_stem.items() if t == s) for s in SPLITS}
        summaries[mode] = summary(rows, split_of_stem)
        moved = sum(len(r["medians"]) for r in rows if split_of_stem[r["stem"]] != r["split"])
        summaries[mode]["moved_annotated"] = moved
    print(f"enclosed: median return range under {threshold:.3f} m (official train's p{ENCLOSED_PERCENTILE}); "
          f"DROW-wide share {share:.1%}", flush=True)
    for mode, rows_ in summaries.items():
        moved = rows_.get("moved_annotated", 0)
        print(f"\n{mode}: {moved} annotated scans moved from the official split", flush=True)
        for s in SPLITS:
            c = rows_[s]
            print(f"  {s:<6} {c['recordings']:>3} recordings  {c['annotated']:>6} annotated scans  {c['people']:>6} people  "
                  f"{c['enclosed'] / c['annotated']:.1%} enclosed", flush=True)
    header = ('"""DROW\'s partitions, as recording stems per split, and what each split holds. **Generated by\n'
              '`utils/drow_geometry_partition.py` -- do not edit by hand**; that script\'s docstring says how they are built.\n\n'
              f'`official` is the three directories as published. A scan is enclosed when its median return range within 10 m is\n'
              f'under {threshold:.3f} m, official train\'s {ENCLOSED_PERCENTILE}th percentile.\n"""\n\n')
    body = f"ENCLOSED_M = {threshold!r}\n\nDROW_PARTITIONS = {pformat(tables, width=120)}\n\nSUMMARY = {pformat(summaries, width=120)}\n"
    args.out.write_text(header + body, encoding="utf-8")
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
