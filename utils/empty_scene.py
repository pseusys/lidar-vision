#!/usr/bin/env python3
"""What TAKHeLiPeD does where nobody is: any FROG partition's person-free frames.

The FROG benchmark holds no person-free frame at all -- every annotated frame has 3-4 people -- so a per-frame false-positive
count there hides behind a ratio: 1.43 phantoms still reads as 63% precision beside 3.06 real people. `transferred`'s test split
is the complement: the person-free frames of the three raw recordings, where **every detection is a false positive by
construction**. wp-AUC is undefined there (no true positives, so no recall denominator) and the measurement is a count.

The published static detectors emit **1.39** (LFE-Peaks) and **1.94** (LFE-PPN) per frame there, and 88.3% / 77.6% of those
recur within 0.3 m in the next frame against 99.8% for real people -- they are furniture, and no short-window tracker removes
them (`memory/performance-log.md`). This is the same measurement for a detector that carries 10 s of memory, where it
could go either way: the slots see the same chair for a thousand frames, so they either learn it is furniture or remember it.

**The three recordings interleave populated and person-free frames**, and the memory replays them whole, as a robot would. So a
detection on a person-free frame can be a *stale memory of a real person* who has just left -- something a static detector
cannot produce. Every count is therefore also reported over the frames where nobody has been in view for longer than the slots'
fade time, where that explanation is unavailable.

**Which partition, and why that is a flag.** `transferred` is the default and what every published row here was measured on.
`balanced` is the other side of one experiment: it is the only partition with person-free frames in **train** and **val**, so
a detector trained on it has seen a room with nobody in it and a detector trained on `official` or `transferred` never has.
The hypothesis is that this is why stage 2 emits 2.62 phantoms per empty frame, and the two counts on the same recording's
empty frames are the evidence. Only person-free frames are counted whichever partition is asked for, the partition each
checkpoint trained on is read from the checkpoint and reported, and a split with no person-free frame is refused.

Usage (from utils/)
-----
    python empty_scene.py --stage2 <stage 2 checkpoint> --memory <step3a checkpoint>
    python empty_scene.py --stage2 ... --memory ... --frog-mode balanced --split test
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
    partition_note,
)

#: The partition and split every published row here was measured on, and so the defaults: `transferred`'s test is
#: `official`'s person-free complement, 72,533 frames over the three raw recordings.
DEFAULT_PERSON_FREE = ("transferred", "test")
#: Where a split's people are read from when it does not carry them itself. One entry, and it is the reason the table
#: exists rather than a branch: `transferred`'s test split is person-free *by construction*, so it has no annotations
#: at all and when someone was last in view can only come from the populated split over the same three recordings.
PEOPLE_FROM = {("transferred", "test"): "val"}
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
    """Which of a person-free split's phantoms the memory *rescored* and which it *coasted*.

    `evaluate` builds each memory frame as the rescored candidates followed by the coasting reports, so the count of the
    first group is the boundary between them. The distinction decides what the person-free work is about: a rescored
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


def populated_frames(split: Split, mode: str, name: str, limit: int) -> np.ndarray:
    """Which packed frames hold at least one person.

    Read from the split's **own** annotations wherever it has them: `balanced` interleaves populated and person-free
    frames in every split, so nothing else is needed and the answer is exact rather than reconstructed. `PEOPLE_FROM`
    holds the one exception, where the split carries no annotations to read."""
    sibling = PEOPLE_FROM.get((mode, name))
    if sibling is None:
        return split.data.gt_valid.any(axis=1)
    ds = FROG_Dataset(split=sibling, mode=mode, auto_download=False, verbose=False, missing_return_m=None)
    sessions = [len(s) for s in ds.scans][:len(split.lengths)] if limit else [len(s) for s in ds.scans]
    if sessions != [int(n) for n in split.lengths]:
        raise ValueError(f"the populated split's recordings are {sessions}, the person-free split's are {list(split.lengths)}; "
                         "they must be the same recordings for the stale-memory split to mean anything")
    mask = np.zeros(int(split.lengths.sum()), dtype=bool)
    for s, ids in enumerate(ds.det_id[:len(sessions)]):
        mask[int(split.offsets[s]) + np.asarray(ids, dtype=np.int64)] = True
    return mask


def counted_frames(index: np.ndarray, populated: np.ndarray, mode: str, name: str) -> np.ndarray:
    """Which of the scored frames hold nobody, and so are the ones this script may count.

    `evaluate` scores every annotated frame, which on a partition that interleaves them includes populated ones. A
    detection there may be a person, so counting it as a phantom would hide the effect being measured underneath it.

    A partition with no person-free frame at all is **refused** rather than counted: `0.000 phantoms per frame` reads
    as a result, when the truth is that the question was never asked."""
    person_free = ~np.asarray(populated)[np.asarray(index)]
    if not person_free.any():
        raise SystemExit(f"none of the {len(index)} frames scored on FROG {mode} {name} is person-free, so there is nothing "
                         "to count. official is 100% populated in every split; transferred holds person-free frames in "
                         "test only; balanced holds them in all three (19-21%).")
    return person_free


def calibration_override(ignore: bool) -> Dict[str, Tuple[float, float]]:
    """`object_memory` overrides that put a score-calibrated arm on an uncalibrated arm's scale, or none at all.

    Every row this script prints is read at a **fixed threshold**, and `calibrate_rescore.py` fitted a temperature and
    bias into the `official` runs' stage 3 that `balanced`'s does not have. The size of that is measured, not guessed:
    writing the pair in moved the same checkpoints' phantom count by -11.8% at threshold 0.3 and 8x at 0.9. Stage 2
    was never calibrated, so the candidate rows compare as they are and this is only about the memory's.

    `(1.0, 0.0)` is the identity of `(logit + bias) / temperature` and `ObjectMemory`'s own default, which is exactly
    how a checkpoint carrying no fitted pair behaves -- so this neutralises the transform without editing either
    checkpoint, and the published, calibrated rows stay reproducible by leaving the flag off.
    """
    return {"calibration": (1.0, 0.0)} if ignore else {}


def load_person_free(name: str, mode: str, limit: int, stage2, device) -> Tuple[Split, np.ndarray]:
    """FROG `mode`'s `name` split with its cached candidates, and which of its frames hold a person.

    One call for both, so a partition cannot reach the loader without reaching the people as well: every count below
    depends on the two describing the same frames."""
    split = load_split(name, limit, stage2, device, mode=mode)
    return split, populated_frames(split, mode, name, limit)


def report(rows: List[Dict[str, float]], title: str) -> None:
    """One table per detector stage, in the columns `memory/performance-log.md`'s empty-scene table already uses."""
    print(f"\n{title}")
    print(f"  {'threshold':>9}  {'per frame':>9}  {'per second':>10}  {'frames with any':>15}  {'recur next frame':>16}")
    for row in rows:
        print(f"  {row['threshold']:>9.2f}  {row['per_frame']:>9.3f}  {row['per_second']:>10.1f}  {row['frames_with_any']:>14.1%}  "
              f"{row['recurs']:>15.1%}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2", type=Path, required=True, help="the trained stage 1-2 checkpoint whose candidates are counted")
    ap.add_argument("--memory", type=Path, required=True, help="the step3a stage 3 trained on that network's candidates")
    ap.add_argument("--frog-mode", choices=("official", "transferred", "balanced"), default=DEFAULT_PERSON_FREE[0],
                    help="the partition whose person-free frames are counted. `transferred` is the default and what every "
                         "published row here was measured on; `balanced` is the one partition with person-free frames in "
                         "train and val too, so it is the other side of the comparison. The partition the checkpoints were "
                         "trained on is read from them and reported, not assumed")
    ap.add_argument("--split", choices=("train", "val", "test"), default=DEFAULT_PERSON_FREE[1],
                    help="which split of it; its person-free frames are the ones counted, and a split with none is refused")
    ap.add_argument("--no-calibration", action="store_true",
                    help="ignore the post-hoc score calibration in the stage-3 checkpoint, so an arm that was "
                         "calibrated and one that was not can be read at the same thresholds. Off by default, which "
                         "reproduces every published row; needed on both sides of the official-against-balanced "
                         "comparison, since only the official runs carry a fitted pair")
    ap.add_argument("--out", type=Path, default=_HERE.parent / "results" / "empty_scene.json")
    ap.add_argument("--stride", type=int, default=1, help="count every Nth person-free frame; 1 is all of them, as the published rows are")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--limit-recordings", type=int, default=0, help="only the first N recordings (smoke runs)")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    device, _ = detect_device()
    memory_checkpoint = torch.load(args.memory, map_location=device, weights_only=False)
    note = partition_note(torch.load(args.stage2, map_location="cpu"), memory_checkpoint, args.frog_mode)
    print(f"counting on FROG {args.frog_mode} {args.split}; checkpoints trained on {' and '.join(note['trained_on'])}", flush=True)
    if note["cross_partition"]:
        print("  ** CROSS-PARTITION COUNT: these frames are not the ones these weights were trained against.", flush=True)
        print("  Deliberate for the official-against-balanced comparison -- but never quote it beside a", flush=True)
        print("  same-partition count without saying so (AGENTS.md rule 3).", flush=True)

    split, populated = load_person_free(args.split, args.frog_mode, args.limit_recordings, args.stage2, device)
    print(f"{args.frog_mode} {args.split}: {len(split.lengths)} recordings, {int(split.lengths.sum())} frames, "
          f"{int(split.data.annotated.sum())} scored, {int((~populated).sum())} of them person-free", flush=True)
    model = object_memory(memory_checkpoint, split.data.features.shape[-1], **calibration_override(args.no_calibration)).to(device)
    print(f"score calibration: {'ignored, identity' if args.no_calibration else memory_checkpoint.get('calibration') or 'none fitted'}", flush=True)

    frames: Dict[str, list] = {}
    evaluate(model, split, device, args.stride, args.streams, evaluation_segments(split, 0, 0, 0), frames_out=frames, compute_metrics=False)
    index = np.asarray(frames["index"], dtype=np.int64)
    keep = counted_frames(index, populated, args.frog_mode, args.split)
    index = index[keep]
    stale = time_since_person(split, populated)[index] <= FADE_HORIZON_S
    period = frame_period(split)
    print(f"scored {len(keep)} frames, counting the {len(index)} person-free ones, "
          f"{stale.mean():.1%} of them within {FADE_HORIZON_S:g} s of someone in view", flush=True)

    result = {}
    for name in ("candidates", "memory"):
        rows = [(score, xy) for (score, xy, _), counted in zip(frames[name], keep, strict=True) if counted]
        beyond = [row for row, recent in zip(rows, stale, strict=True) if not recent]
        result[name] = [phantom_counts(rows, index, t, RECUR_RADIUS_M, len(rows) * args.stride * period) for t in THRESHOLDS]
        result[f"{name}_beyond_fade"] = [phantom_counts(beyond, index[~stale], t, RECUR_RADIUS_M, len(beyond) * args.stride * period)
                                         for t in THRESHOLDS]
        report(result[name], f"{name}: {len(rows)} person-free frames")
        report(result[f"{name}_beyond_fade"], f"{name}: the {len(beyond)} where nobody has been in view for {FADE_HORIZON_S:g} s")
        if name == "memory":
            counts = np.asarray(frames["rescored"])[keep]
            result["split"] = [coasting_share(rows, counts, t) for t in THRESHOLDS]
            result["split_beyond_fade"] = [coasting_share(beyond, counts[~stale], t) for t in THRESHOLDS]
            print("\n  where the memory's phantoms come from")
            print(f"  {'threshold':>9}  {'rescored':>9}  {'coasted':>9}  {'coasted share':>14}  {'coasted/frame':>14}")
            for row, far in zip(result["split"], result["split_beyond_fade"], strict=True):
                print(f"  {row['threshold']:>9.2f}  {row['from_candidates']:>9d}  {row['from_coasting']:>9d}  "
                      f"{row['coasting_share']:>13.1%}  {row['coasting_per_frame']:>14.4f}   "
                      f"(beyond fade: {far['from_coasting']} coasted, {far['coasting_share']:.1%})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(stage2=str(args.stage2), memory=str(args.memory), mode=args.frog_mode, split=args.split,
                                        partitions=note, calibration=None if args.no_calibration else memory_checkpoint.get("calibration"),
                                        stride=args.stride, scored_frames=int(len(keep)), person_free_frames=len(index),
                                        frame_period_s=period, within_fade=float(stale.mean()), counts=result),
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
