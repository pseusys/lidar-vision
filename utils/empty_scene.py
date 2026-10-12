#!/usr/bin/env python3
"""What TAKHeLiPeD does where nobody is: FROG `transferred`'s 72,533 person-free frames (`TODO.md` A50 phase 1 item 7).

The FROG benchmark holds no person-free frame at all -- every annotated frame has 3-4 people -- so a per-frame false-positive
count there hides behind a ratio: 1.43 phantoms still reads as 63% precision beside 3.06 real people. `transferred`'s test split
is the complement: the person-free frames of the three raw recordings, where **every detection is a false positive by
construction**. wp-AUC is undefined there (no true positives, so no recall denominator) and the measurement is a count.

The published static detectors emit **1.39** (LFE-Peaks) and **1.94** (LFE-PPN) per frame there, and 88.3% / 77.6% of those
recur within 0.3 m in the next frame against 99.8% for real people -- they are furniture, and no short-window tracker removes
them (`memory/performance-log.md`, A41/A44). This is the same measurement for a detector that carries 10 s of memory, where it
could go either way: the slots see the same chair for a thousand frames, so they either learn it is furniture or remember it.

**The three recordings interleave populated and person-free frames**, and the memory replays them whole, as a robot would. So a
detection on a person-free frame can be a *stale memory of a real person* who has just left -- something a static detector
cannot produce. Every count is therefore also reported over the frames where nobody has been in view for longer than the slots'
fade time, where that explanation is unavailable.

Usage (from utils/)
-----
    python empty_scene.py --stage2 <stage 2 checkpoint> --memory <step3a checkpoint>
    python empty_scene.py --stage2 ... --memory ... --limit-recordings 1 --stride 50    # smoke run
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from takheliped.datasets.frog_dataset import FROG_Dataset  # noqa: E402
from train_takheliped import (  # noqa: E402
    OPERATING_THRESHOLD, Split, detect_device, evaluate, evaluation_segments, load_split, mean_frame_period, object_memory,
)

PERSON_FREE_MODE = "transferred"
POPULATED_SPLIT = "val"             # `transferred`'s val is the populated complement of its test, over the same recordings
RECUR_RADIUS_M = 0.3                # `phantom_analysis.py`'s association radius, so the recurrence shares compare
FADE_HORIZON_S = 10.0               # `SlotRules.fade_time_s`: past it an unmatched slot has retired, so no memory can be stale
THRESHOLDS = (0.9, 0.5, OPERATING_THRESHOLD, 0.2, 0.1)
Frames = Sequence[Tuple[np.ndarray, np.ndarray]]


def phantom_counts(frames: Frames, index: np.ndarray, threshold: float, radius: float, seconds: float) -> Dict[str, float]:
    """How many detections a person-free split holds at `threshold`: per frame, per second, the share of frames holding any, and
    the share that recur within `radius` in the next frame -- the measure that separates furniture from noise.

    `index` is each frame's position in the recordings packed end to end. Only frames that really do follow one another are
    paired for that last column: the person-free frames are interleaved with populated ones the split drops, so two consecutive
    scored frames can be minutes apart, and `evaluate` plays several recordings at once so they do not even arrive in order."""
    order = np.argsort(np.asarray(index), kind="stable")
    index = np.asarray(index)[order]
    kept = [frames[i][1][frames[i][0] >= threshold] for i in order]
    detections = sum(len(xy) for xy in kept)
    recurring = considered = 0
    for i in np.nonzero(np.diff(index) == 1)[0] if len(kept) > 1 else []:
        before, after = kept[i], kept[i + 1]
        if not len(after):
            continue
        considered += len(after)
        if len(before):
            recurring += int((np.linalg.norm(after[:, None, :] - before[None, :, :], axis=-1).min(axis=1) <= radius).sum())
    return dict(threshold=threshold, frames=len(kept), detections=detections,
                per_frame=detections / len(kept) if kept else 0.0,
                per_second=detections / seconds if seconds else 0.0,
                frames_with_any=sum(1 for xy in kept if len(xy)) / len(kept) if kept else 0.0,
                recurs=recurring / considered if considered else float("nan"), recur_of=considered)


def coasting_share(frames: Frames, candidates_per_frame: np.ndarray, threshold: float) -> Dict[str, float]:
    """Which of a person-free split's phantoms the memory *rescored* and which it *coasted* (`TODO.md` §A item 27).

    `evaluate` builds each memory frame as the rescored candidates followed by the coasting reports, so the count of the
    first group is the boundary between them. The distinction decides what items 14, 24, 25 and 26 are about: a rescored
    candidate is a scoring question, while a coasting report is the memory reporting a person with nothing under them.
    """
    kept = [(score >= threshold, int(n)) for (score, _), n in zip(frames, candidates_per_frame, strict=True)]
    from_candidates = sum(int(above[:n].sum()) for above, n in kept)
    from_coasting = sum(int(above[n:].sum()) for above, n in kept)
    total = from_candidates + from_coasting
    return dict(threshold=threshold, from_candidates=from_candidates, from_coasting=from_coasting,
                coasting_share=from_coasting / total if total else 0.0,
                coasting_per_frame=from_coasting / len(kept) if kept else 0.0)


def since_person(populated: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Seconds since a person was last in view, per frame of one recording; `inf` before the first populated frame."""
    return time - np.maximum.accumulate(np.where(populated, time, -np.inf))


def time_since_person(split: Split, populated: np.ndarray) -> np.ndarray:
    """`since_person` per packed frame, computed inside each recording so one recording's people never count for the next."""
    out = np.empty(len(populated), dtype=np.float64)
    for offset, length in zip(split.offsets, split.lengths):
        inside = slice(int(offset), int(offset) + int(length))
        out[inside] = since_person(populated[inside], split.data.time[inside])
    return out


def frame_period(split: Split) -> float:
    """The recordings' seconds per frame, through the shared `frame_period_s`. Detections per second is `frames * this`, never the
    span between scored frames: the person-free frames are interleaved with populated ones the split drops, so a span would bill
    time nobody was asked about."""
    return mean_frame_period(split.data.time, split.offsets, split.lengths)


def populated_frames(split: Split, limit: int) -> np.ndarray:
    """Which packed frames hold at least one person, read from the populated split over the same recordings."""
    ds = FROG_Dataset(split=POPULATED_SPLIT, mode=PERSON_FREE_MODE, auto_download=False, verbose=False, missing_return_m=None)
    sessions = [len(s) for s in ds.scans][:len(split.lengths)] if limit else [len(s) for s in ds.scans]
    if sessions != [int(n) for n in split.lengths]:
        raise ValueError(f"the populated split's recordings are {sessions}, the person-free split's are {list(split.lengths)}; "
                         "they must be the same recordings for the stale-memory split to mean anything")
    mask = np.zeros(int(split.lengths.sum()), dtype=bool)
    for s, ids in enumerate(ds.det_id[:len(sessions)]):
        mask[int(split.offsets[s]) + np.asarray(ids, dtype=np.int64)] = True
    return mask


def report(rows: List[Dict[str, float]], title: str) -> None:
    """One table per detector stage, in the columns `memory/performance-log.md`'s empty-scene table already uses."""
    print(f"\n{title}")
    print(f"  {'threshold':>9}  {'per frame':>9}  {'per second':>10}  {'frames with any':>15}  {'recur next frame':>16}")
    for row in rows:
        print(f"  {row['threshold']:>9.2f}  {row['per_frame']:>9.3f}  {row['per_second']:>10.1f}  {row['frames_with_any']:>14.1%}  "
              f"{row['recurs']:>15.1%}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2", type=Path, required=True, help="the trained stage 1-2 checkpoint whose candidates are counted")
    ap.add_argument("--memory", type=Path, required=True, help="the step3a stage 3 trained on that network's candidates")
    ap.add_argument("--out", type=Path, default=_HERE.parent / "results" / "empty_scene.json")
    ap.add_argument("--stride", type=int, default=1, help="count every Nth person-free frame; 1 is all of them, as the published rows are")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--limit-recordings", type=int, default=0, help="only the first N recordings (smoke runs)")
    args = ap.parse_args()

    device, _ = detect_device()
    split = load_split("test", args.limit_recordings, args.stage2, device, mode=PERSON_FREE_MODE)
    print(f"{PERSON_FREE_MODE} test: {len(split.lengths)} recordings, {int(split.lengths.sum())} frames, "
          f"{int(split.data.annotated.sum())} person-free, {int(split.data.gt_valid.sum())} people in them (it must be 0)", flush=True)
    model = object_memory(torch.load(args.memory, map_location=device, weights_only=False), split.data.features.shape[-1]).to(device)

    frames: Dict[str, list] = {}
    evaluate(model, split, device, args.stride, args.streams, evaluation_segments(split, 0, 0, 0), frames_out=frames, compute_metrics=False)
    index = np.asarray(frames["index"], dtype=np.int64)
    stale = time_since_person(split, populated_frames(split, args.limit_recordings))[index] <= FADE_HORIZON_S
    period = frame_period(split)
    print(f"scored {len(index)} person-free frames, {stale.mean():.1%} of them within {FADE_HORIZON_S:g} s of someone in view", flush=True)

    result = {}
    for name in ("candidates", "memory"):
        rows = [(score, xy) for score, xy, _ in frames[name]]
        beyond = [row for row, recent in zip(rows, stale, strict=True) if not recent]
        result[name] = [phantom_counts(rows, index, t, RECUR_RADIUS_M, len(rows) * args.stride * period) for t in THRESHOLDS]
        result[f"{name}_beyond_fade"] = [phantom_counts(beyond, index[~stale], t, RECUR_RADIUS_M, len(beyond) * args.stride * period)
                                         for t in THRESHOLDS]
        report(result[name], f"{name}: {len(rows)} person-free frames")
        report(result[f"{name}_beyond_fade"], f"{name}: the {len(beyond)} where nobody has been in view for {FADE_HORIZON_S:g} s")
        if name == "memory":
            counts = np.asarray(frames["rescored"])
            result["split"] = [coasting_share(rows, counts, t) for t in THRESHOLDS]
            result["split_beyond_fade"] = [coasting_share(beyond, counts[~stale], t) for t in THRESHOLDS]
            print("\n  where the memory's phantoms come from (item 27)")
            print(f"  {'threshold':>9}  {'rescored':>9}  {'coasted':>9}  {'coasted share':>14}  {'coasted/frame':>14}")
            for row, far in zip(result["split"], result["split_beyond_fade"], strict=True):
                print(f"  {row['threshold']:>9.2f}  {row['from_candidates']:>9d}  {row['from_coasting']:>9d}  "
                      f"{row['coasting_share']:>13.1%}  {row['coasting_per_frame']:>14.4f}   "
                      f"(beyond fade: {far['from_coasting']} coasted, {far['coasting_share']:.1%})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(stage2=str(args.stage2), memory=str(args.memory), mode=PERSON_FREE_MODE, stride=args.stride,
                                        scored_frames=len(index), frame_period_s=period, within_fade=float(stale.mean()), counts=result),
                                   indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
        code = 0
    except SystemExit as leaving:        # argparse's --help and its usage errors: pass their status through
        code = leaving.code if isinstance(leaving.code, int) else (0 if leaving.code is None else 1)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)                          # a finished ROCm process can hang past a normal exit (memory/gotchas.md)
