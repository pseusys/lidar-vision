#!/usr/bin/env python3
"""What a slot creation or confirmation rule would remove.

The object memory cuts person-free false positives from **2.622** per frame to **0.740**, and the creation-rule work reads the survivors as a
creation-rule question: they recur in the next frame **92.2%** of the time against the candidates' 84.9% and real people's
99.8%, so a higher creation threshold or a confirmation over several frames might remove them.

**The reading has a problem that this script exists to settle.** A confirmation rule filters slots that are *young*, and
furniture is the opposite of young -- a chair sits in the room for the whole recording, so its slot is among the oldest and
highest-hit-rate in the memory. If the slots behind the surviving phantoms are old, no creation rule reaches them, and all such
a rule buys is a delay on genuinely new people. The measurement is the slot **age** distribution behind each population.

The phantom audit makes the question well-posed: 99.997% of the phantoms at the operating threshold are **rescored candidates**, not
coasting reports, so every one of them has a slot matched to it whose statistics can be read.

**What the delay costs, for scale.** FROG's trajectories run a median 4.5 s -- 180 frames at 40 Hz -- so a 5-frame
confirmation suppresses 2.8% of a median person's frames and delays their first report by 100 ms. The cost is a delay rather
than a loss for anyone who stays in view, which is why the person column below is reported as frames, not as people.

Usage (from utils/)
-----
    python slot_confirmation.py --stage2 <stage 2 checkpoint> --memory <step3a checkpoint>
"""

from typing import Dict

import numpy as np

PHANTOM, PERSON = 0, 1
CONFIRMATION_FRAMES = (1, 2, 3, 5, 8, 13, 21, 40)       # up to one second at FROG's 40 Hz


def slot_age_frames(age_s: np.ndarray, period_s: float) -> np.ndarray:
    """A slot's age in frames, which is the unit a confirmation rule is written in."""
    if period_s <= 0:
        raise ValueError(f"the frame period must be positive, got {period_s}")
    return np.asarray(age_s, dtype=np.float64) / period_s


def confirmation_curve(age_frames: np.ndarray, population: np.ndarray, k: int) -> Dict[str, float]:
    """What "a slot must be `k` frames old before it may report" would do to each population.

    Reported per population rather than in total, because the whole question is whether the rule is selective: removing
    phantoms and people in equal shares is not a filter, it is a threshold on how long the detector has been watching.
    """
    age_frames, population = np.asarray(age_frames, dtype=np.float64), np.asarray(population)
    young = age_frames < k - 1                              # the first frame is free: age 0 satisfies k = 1
    counts = {}
    for name, label in (("phantom", PHANTOM), ("person", PERSON)):
        of_this = population == label
        removed, total = int((young & of_this).sum()), int(of_this.sum())
        counts[f"{name}s_removed" if name == "phantom" else "people_removed"] = removed
        counts[f"{name}s_kept" if name == "phantom" else "people_kept"] = total - removed
        counts[f"{name}_share_removed"] = removed / total if total else 0.0
    return dict(k=k, delay_frames=k - 1, **counts)


def report(rows, period_s: float) -> None:
    print(f"\n  a slot must be k frames old to report, at {1000 * period_s:.1f} ms per frame\n")
    print(f"  {'k':>3} {'delay':>9} {'phantoms cut':>13} {'person frames lost':>19} {'selectivity':>12}")
    for row in rows:
        selective = (row["phantom_share_removed"] / row["person_share_removed"]) if row["person_share_removed"] else float("inf")
        print(f"  {row['k']:>3} {1000 * period_s * row['delay_frames']:>7.0f}ms "
              f"{row['phantom_share_removed']:>12.1%} {row['person_share_removed']:>18.1%} "
              f"{('n/a' if selective == float('inf') else f'{selective:.2f}x'):>12}", flush=True)
    print("\n  selectivity above 1 means the rule removes a larger share of phantoms than of person frames;\n"
          "  at or below 1 it is not a phantom filter, only a delay.")



def populations(frames, gt_radius_m: float, threshold: float):
    """Each reported detection's backing-slot age, labelled phantom or person.

    A detection on a **person-free** frame is a phantom by construction -- that split holds no annotated people at all,
    which is the whole reason it exists (`empty_scene.py`). On a populated frame a detection within `gt_radius_m` of an
    annotated person is that person, and a populated frame's *other* false positives are left out rather than pooled with
    the empty-room ones: `docs/PAPER.md`'s argument is that the two hide each other whenever a ratio mixes them.

    A detection with no backing slot is dropped rather than read as age zero -- it is a coasting report, which has no
    candidate under it, and zero would say "brand new" about a slot that is usually the opposite.
    """
    ages, labels = [], []
    for (score, xy, gt), backing in zip(frames["memory"], frames["slot_age"], strict=True):
        backing = np.asarray(backing, dtype=np.float64)
        keep = np.asarray(score)[:len(backing)] >= threshold
        age, here = backing[keep], np.asarray(xy)[:len(backing)][keep]
        gt = np.asarray(gt).reshape(-1, 2)
        if not len(gt):
            ages.append(age)
            labels.append(np.full(len(age), PHANTOM, dtype=np.int8))
            continue
        is_person = np.linalg.norm(here[:, None, :] - gt[None, :, :], axis=-1).min(axis=1) <= gt_radius_m if len(here) else np.zeros(0, bool)
        ages.append(age[is_person])
        labels.append(np.full(int(is_person.sum()), PERSON, dtype=np.int8))
    ages = np.concatenate(ages) if ages else np.zeros(0)
    labels = np.concatenate(labels) if labels else np.zeros(0, dtype=np.int8)
    known = ~np.isnan(ages)
    return ages[known], labels[known]


def main() -> None:
    import argparse                                                  # noqa: PLC0415 -- only the driver needs these
    import gc
    import json
    import os
    import sys
    from pathlib import Path

    here = Path(__file__).parent
    sys.path.insert(0, str(here.parent / "library"))
    sys.path.insert(0, str(here))
    import torch                                                      # noqa: PLC0415
    from empty_scene import frame_period                             # noqa: PLC0415 -- the same unpacking, already written
    from train_takheliped import (  # noqa: PLC0415
        MATCH_RADIUS_M, OPERATING_THRESHOLD, detect_device, evaluate, evaluation_segments, load_split, object_memory,
    )

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2", type=Path, required=True)
    ap.add_argument("--memory", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=OPERATING_THRESHOLD)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--limit-recordings", type=int, default=0)
    ap.add_argument("--out", type=Path, default=here.parent / "results" / "slot_confirmation.json")
    args = ap.parse_args()

    device, _ = detect_device()
    collected = {}
    # `transferred` test supplies the phantoms -- it is the person-free complement of `official` test over the same three
    # recordings, so every detection there is a false positive by construction -- and `official` test supplies the people,
    # because that is the split every recall number in this project is quoted on. Disjoint frames, same room, and both
    # already have cached candidates, which `transferred`'s populated half does not.
    for label, (split_name, mode) in (("phantoms", ("test", "transferred")), ("people", ("test", "official"))):
        split = load_split(split_name, args.limit_recordings, args.stage2, device, mode=mode)
        model = object_memory(torch.load(args.memory, map_location=device, weights_only=False),
                              split.data.features.shape[-1]).to(device)
        frames = {}
        evaluate(model, split, device, args.stride, args.streams, evaluation_segments(split, 0, 0, 0),
                 frames_out=frames, compute_metrics=False)
        ages, labels = populations(frames, MATCH_RADIUS_M, args.threshold)
        period = frame_period(split)
        collected[label] = (slot_age_frames(ages, period), labels, period)
        print(f"{label}: {len(ages)} reported detections over {len(frames['memory'])} frames, {period * 1000:.1f} ms/frame",
              flush=True)
        # Each split's candidates are about a gigabyte in RAM and only the small arrays above are wanted, so let the first
        # one go before the second is read: holding both exhausted the host once (2026-10-12, `memory/gotchas.md`).
        del split, model, frames
        gc.collect()

    ages = np.concatenate([collected["phantoms"][0][collected["phantoms"][1] == PHANTOM],
                           collected["people"][0][collected["people"][1] == PERSON]])
    labels = np.concatenate([np.full(int((collected["phantoms"][1] == PHANTOM).sum()), PHANTOM, dtype=np.int8),
                             np.full(int((collected["people"][1] == PERSON).sum()), PERSON, dtype=np.int8)])
    period = collected["phantoms"][2]

    for name, label in (("phantom", PHANTOM), ("person", PERSON)):
        of_this = ages[labels == label]
        if len(of_this):
            quartiles = np.percentile(of_this, [25, 50, 75, 95])
            print(f"\n  backing-slot age behind {name}s, in frames: median {quartiles[1]:.0f}, "
                  f"quartiles {quartiles[0]:.0f}/{quartiles[2]:.0f}, p95 {quartiles[3]:.0f}  (n = {len(of_this)})", flush=True)

    rows = [confirmation_curve(ages, labels, k) for k in CONFIRMATION_FRAMES]
    report(rows, period)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(stage2=str(args.stage2), memory=str(args.memory), threshold=args.threshold,
                                        frame_period_s=period, curve=rows), indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}", flush=True)
    sys.stdout.flush()
    os._exit(0)                     # a finished ROCm process can hang past a normal exit (memory/gotchas.md)


if __name__ == "__main__":
    main()
