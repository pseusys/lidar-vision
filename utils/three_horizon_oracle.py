#!/usr/bin/env python3
"""The temporal oracle replayed on *our own* detector, rather than on LFE-Peaks.

`utils/temporal_oracle.py` measured the headroom a perfect temporal model has
over a static one, and found +9.6 points of recall available from people missed
in one frame but detected shortly before or after. That ceiling was measured on
**LFE-Peaks' detections**, so it describes LFE's headroom, not ours: our stage-2
calibration network already detects more people, and a ceiling measured over a
weaker detector's misses says nothing about what is left above a stronger one.

`docs/PROPOSAL.md` §9 asks the question this script answers: of the recall that
was *available* to memory above our own candidates, how much did stage 3
actually take?

Three quantities, all over annotated people ("person-observations"), all at the
operating threshold:

    covered by stage 2   our calibration network's candidates alone
    recoverable          stage-2 misses whose own trajectory was detected both
                         before and after the miss, within `--max-bridge-s`
                         -- the interpolation bound, the credible ceiling
    covered by stage 3   the same people after the object memory rescores the
                         candidates and reports coasting slots

and the cross-tabulation that is the actual result: of the recoverable misses,
how many stage 3 recovered -- and, in the other direction, how many people
stage 2 had found that stage 3 then lost. Both directions matter here, because
at the operating threshold the memory trades recall for precision (87.3% ->
84.7% in `checkpoints_three_horizon/step4_results.json`), so a recovery count
quoted alone would flatter it.

**Read the recall figures as person-observation coverage, not as the `recall`
column of the result tables.** Those come from `_prec_rec_2d` sweeping a
precision-recall curve; this script matches detections to people one frame at a
time with a gated Hungarian assignment at `MATCH_RADIUS_M`. The two agree
closely but not exactly, and the printed header reports both so the difference
stays visible.

Usage
-----
    python three_horizon_oracle.py                      # whole test split
    python three_horizon_oracle.py --limit-recordings 2 # a smoke run
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.detectors.three_horizon import ObjectMemory, SlotRules  # noqa: E402
from motion_analysis import build_tracks, to_world  # noqa: E402
from train_three_horizon import (  # noqa: E402
    MATCH_RADIUS_M, MIN_REPORTED_SCORE, OPERATING_THRESHOLD, Split, candidate_targets, detect_device, evaluation_segments, gather, load_split,
)

BRIDGES_S = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 60.0)

Coverage = Dict[Tuple[int, int], np.ndarray]     # (recording, frame) -> which of that frame's annotated people a detector covers
Track = Tuple[List[Tuple[int, bool, bool]], float]   # one trajectory's (frame, covered by stage 2, covered by stage 3) points, and its rate


@torch.no_grad()
def coverage(model: ObjectMemory, split: Split, device, streams: int, threshold: float, slice_frames: int = 512) -> Tuple[Coverage, Coverage]:
    """Which annotated people each detector covers, per annotated frame: `(recording, frame) -> bool array` over that frame's
    people, once for the candidates alone and once for the memory's report. Plays whole recordings from empty memory, `streams`
    at a time in slices of `slice_frames`, exactly as `train_three_horizon.evaluate` does."""
    model.eval()
    cand_cov, mem_cov = {}, {}
    segments = sorted(evaluation_segments(split, 0, 0, 0), key=lambda seg: -seg[2])
    for group in [segments[i:i + streams] for i in range(0, len(segments), streams)]:
        length = max(seg[2] for seg in group)
        recording = np.array([seg[0] for seg in group])[:, None]
        frame = np.array([seg[1] for seg in group])[:, None] + np.minimum(np.arange(length)[None], np.array([seg[2] for seg in group])[:, None] - 1)
        chosen = np.zeros(frame.shape, dtype=bool)
        for b, (s, first, frames, _) in enumerate(group):
            chosen[b, np.nonzero(split.data.annotated[split.offsets[s] + first:split.offsets[s] + first + frames])[0]] = True
        slots = model.initial_state(len(group), device)
        for t0 in range(0, length, slice_frames):
            t1 = min(t0 + slice_frames, length)
            batch = gather(split, recording.repeat(t1 - t0, 1), frame[:, t0:t1], device)
            for t in range(t0, t1):
                i = t - t0
                out = model.step(batch["xy"][:, i], batch["score"][:, i], batch["valid"][:, i], batch["pose"][:, i], batch["dt"][:, i], slots,
                                 reset=torch.full((len(group),), t == 0, dtype=torch.bool, device=device), cand_features=batch["features"][:, i])
                slots = out.slots
                rows = np.nonzero(chosen[:, t])[0]
                if not len(rows):
                    continue
                cand_p = torch.sigmoid(out.candidate_logit)[rows].cpu().numpy()
                slot_p = torch.sigmoid(out.slot_logit)[rows].cpu().numpy()
                coasting, slot_xy = out.coasting[rows].cpu().numpy(), out.slot_xy[rows].cpu().numpy()
                valid, xy, score = batch["valid"][rows, i].cpu().numpy(), batch["xy"][rows, i].cpu().numpy(), batch["score"][rows, i].cpu().numpy()
                gt_xy, gt_valid = batch["gt_xy"][rows, i].cpu().numpy(), batch["gt_valid"][rows, i].cpu().numpy()
                for j, r in enumerate(rows):
                    gt = gt_xy[j][gt_valid[j]].astype(np.float64)
                    _, c2 = candidate_targets(xy[j], valid[j] & (score[j] >= threshold), gt, MATCH_RADIUS_M)
                    report = coasting[j] & (slot_p[j] >= max(threshold, MIN_REPORTED_SCORE))
                    _, c3 = candidate_targets(np.concatenate([xy[j], slot_xy[j]]),
                                              np.concatenate([valid[j] & (cand_p[j] >= threshold), report]), gt, MATCH_RADIUS_M)
                    cand_cov[(int(group[r][0]), int(frame[r, t]))], mem_cov[(int(group[r][0]), int(frame[r, t]))] = c2, c3
    return cand_cov, mem_cov


def trajectories(split: Split, cand_cov: Coverage, mem_cov: Coverage, gate: float) -> Tuple[List[Track], int]:
    """Annotated people linked into world-frame trajectories, each point carrying whether stage 2 and stage 3 covered it.

    Returns `(points, hz)` per trajectory, `points` being `(frame, covered by stage 2, covered by stage 3)` in frame order, and
    the number of points whose rounded world position collided with another person's in the same frame (the lookup back from
    `build_tracks`' positions to coverage is keyed by position, as `temporal_oracle.py` does; a non-zero count means that key is
    not unique and the coverage of those points may be attributed to the wrong person)."""
    out, collisions = [], 0
    for s, (offset, n) in enumerate(zip(split.offsets, split.lengths)):
        time = split.data.time[offset:offset + n]
        hz = 1.0 / float(np.median(np.diff(time))) if n > 1 else float("nan")
        index, world, look = [], [], {}
        for f in range(int(n)):
            g = int(offset) + f
            if not split.data.annotated[g]:
                continue
            people = split.data.gt_xy[g][split.data.gt_valid[g]].astype(np.float64)
            w = to_world(people, split.data.pose[g]) if len(people) else np.zeros((0, 2))
            index.append(f)
            world.append(w)
            c2, c3 = cand_cov[(s, f)], mem_cov[(s, f)]
            for j in range(len(w)):
                key = (f, round(float(w[j][0]), 6), round(float(w[j][1]), 6))
                collisions += key in look
                look[key] = (bool(c2[j]), bool(c3[j]))
        for track in build_tracks(world, scan_index=index, gate=gate):
            points = [(int(f), *look[key]) for f in sorted(track)
                      if (key := (int(f), round(float(track[f][0]), 6), round(float(track[f][1]), 6))) in look]
            if points:
                out.append((points, hz))
    return out, collisions


def bounds(tracks: List[Track], bridge_s: float) -> Tuple[int, int]:
    """The interpolation bound at `bridge_s` and what stage 3 did with it: how many stage-2 misses sit between two detections of
    their own trajectory no more than `bridge_s` apart, and how many of those stage 3 covered."""
    recoverable = recovered = 0
    for points, hz in tracks:
        found = np.array([f for f, c2, _ in points if c2])
        if not len(found) or not np.isfinite(hz):
            continue
        span = bridge_s * hz
        for f, c2, c3 in points:
            if c2:
                continue
            before, after = found[found < f], found[found > f]
            if len(before) and len(after) and (after[0] - before[-1]) <= span:
                recoverable += 1
                recovered += c3
    return recoverable, recovered


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory", type=Path, default=_HERE.parent / "checkpoints_three_horizon" / "step3a_object_memory.best.pth")
    ap.add_argument("--stage2", type=Path, default=_HERE.parent / "checkpoints_three_horizon" / "step2_calibration.best.pth")
    ap.add_argument("--out", type=Path, default=_HERE.parent / "checkpoints_three_horizon" / "oracle_ours.json")
    ap.add_argument("--slots", type=int, default=0, help="Slot capacity; 0 takes the trained checkpoint's own (default: 0)")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=OPERATING_THRESHOLD)
    ap.add_argument("--gate", type=float, default=0.6, help="Trajectory association gate, metres (default: 0.6)")
    ap.add_argument("--limit-recordings", type=int, default=0)
    args = ap.parse_args()

    device, _ = detect_device()
    checkpoint = torch.load(args.memory, map_location=device, weights_only=False)
    slots = args.slots or int(checkpoint["args"]["slots"])
    split = load_split("test", args.limit_recordings, args.stage2, device)
    model = ObjectMemory(SlotRules(capacity=slots), feature_dim=split.data.features.shape[-1]).to(device)
    model.load_state_dict(checkpoint["model"])
    print(f"memory: {args.memory.name}, {slots} slots, epoch {checkpoint['epoch']}, chunks of {checkpoint['chunk_s']:g} s", flush=True)

    cand_cov, mem_cov = coverage(model, split, device, args.streams, args.threshold)
    tracks, collisions = trajectories(split, cand_cov, mem_cov, args.gate)

    n = sum(len(p) for p, _ in tracks)
    c2 = sum(a for p, _ in tracks for _, a, _ in p)
    c3 = sum(b for p, _ in tracks for _, _, b in p)
    lost = sum(a and not b for p, _ in tracks for _, a, b in p)
    gained = sum(b and not a for p, _ in tracks for _, a, b in p)
    print(f"\n=== ORACLE ON OUR OWN CANDIDATES: threshold {args.threshold}, match {MATCH_RADIUS_M} m, gate {args.gate} m ===")
    print(f"{len(tracks)} ground-truth trajectories, {n} person-observations over {len(cand_cov)} annotated frames"
          f"{f', {collisions} position-key collisions' if collisions else ''}")
    print(f"  covered by stage 2 (candidates): {c2}/{n} = {c2 / n:.1%}   (miss rate {1 - c2 / n:.1%})")
    print(f"  covered by stage 3 (memory):     {c3}/{n} = {c3 / n:.1%}   (miss rate {1 - c3 / n:.1%})")
    print(f"  stage 3 against stage 2: +{gained} recovered, -{lost} lost, net {c3 - c2:+d}")

    coverage_of = np.array([sum(a for _, a, _ in p) / len(p) for p, _ in tracks])
    weight = np.array([len(p) for p, _ in tracks], dtype=float)
    print("\n  trajectory coverage by stage 2 (MT/PT/ML, Li-Huang-Nevatia):")
    for label, mask in (("Mostly Tracked  (>=80% covered)", coverage_of >= 0.8),
                        ("Partially Tracked (20-80%)", (coverage_of >= 0.2) & (coverage_of < 0.8)),
                        ("Mostly Lost     (<20% covered)", coverage_of < 0.2)):
        print(f"    {label:<34} {mask.sum():>5} trajectories ({mask.sum() / len(coverage_of):>5.1%})  "
              f"{weight[mask].sum() / weight.sum():>5.1%} of observations")

    print("\n  INTERPOLATION bound -- a stage-2 miss counts as recoverable only if its own trajectory was detected both before")
    print("  and after it, within the bridge; 'taken' is how many of those stage 3 actually covered:")
    print(f"  {'bridge':>8} {'recoverable':>12} {'ceiling':>9} {'taken':>8} {'of ceiling':>11}")
    per_bridge = {}
    for bridge_s in BRIDGES_S:
        recoverable, recovered = bounds(tracks, bridge_s)
        per_bridge[bridge_s] = dict(recoverable=recoverable, recovered=recovered)
        print(f"  {bridge_s:>6.1f} s {recoverable:>12} {(c2 + recoverable) / n:>8.1%} {recovered:>8} "
              f"{recovered / recoverable if recoverable else float('nan'):>10.1%}")

    never = sum(len(p) for p, _ in tracks if not any(a for _, a, _ in p))
    print(f"\n  Irreducible for interpolation: people on trajectories stage 2 never detected at all -- {never}/{n} = {never / n:.1%}.")
    args.out.write_text(json.dumps(dict(
        memory=str(args.memory), slots=slots, threshold=args.threshold, gate=args.gate, match_radius=MATCH_RADIUS_M,
        trajectories=len(tracks), observations=n, annotated_frames=len(cand_cov), collisions=collisions,
        covered_stage2=c2, covered_stage3=c3, recovered=gained, lost=lost, never_detected=never,
        bridges={f"{k:g}": v for k, v in per_bridge.items()},
    ), indent=1), encoding="utf-8")
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
