#!/usr/bin/env python3
"""Why does stage 3 lose people stage 2 had already found? (`TODO.md` A50 phase 1b.)

`three_horizon_oracle.py` established the shape of the problem: over the FROG `official` test split the memory recovers 2,113
annotated people the candidates missed and **loses 6,076 it had**, for a net -3,963. So stage 3's weakness is not forgetting --
something actively destroys good detections -- and its +3.2 points of AP are bought with precision, not recovered recall.

Three mechanisms could lose a person the candidates covered, and they need different fixes:

    rescoring   a candidate within 0.5 m still exists, but `candidate_logit = logit(score) + rescore_gate * rescore(...)`
                pushed it under the operating threshold. The correction is *signed* and applies to every candidate, with
                nothing constraining it to raise confidence only for remembered objects.
    association the candidate exists and kept its score, but no live slot sits near the person, so the memory had nothing to
                recall -- `manage_slots` matches mutual-nearest within a fixed `gate_m` of the velocity-predicted position.
    retirement  a slot was there and died: unmatched value fades to `floor` over `fade_time_s`, and the slot retires.

This measures which, per person, and in the same pass compares the rescoring delta on candidates that cover a person against
candidates that cover nobody -- the recall-for-precision trade the head is actually making. Everything here is read from values
`ObjectMemory.step` already returns, so it needs no change to the library and cannot disturb a training run.

Usage
-----
    python memory_diagnosis.py
    python memory_diagnosis.py --limit-recordings 2      # smoke run
    python memory_diagnosis.py --stage3-threshold 0.15   # does the loss survive a threshold recalibrated for stage 3?

The last one is the test that separates the two readings of the loss. Stage 3's rescoring shifts the whole score distribution
down -- on the full split the people it *kept* were pushed down harder (median -3.115 logits) than the ones it lost (-1.394) --
so a threshold fixed at stage 2's 0.3 is no longer calibrated for stage 3's output. If lowering only stage 3's threshold
recovers the people, the loss is an artefact of the fixed threshold and the fix is calibration, not architecture.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.detectors.three_horizon import ObjectMemory, SlotRules  # noqa: E402
from train_three_horizon import (  # noqa: E402
    MATCH_RADIUS_M, MIN_REPORTED_SCORE, OPERATING_THRESHOLD, Split, detect_device, evaluation_segments, gather, load_split,
)

PROBABILITY_EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    """The candidates' own logit, the baseline the memory's correction is added to."""
    p = np.clip(p, PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
    return np.log(p) - np.log1p(-p)


def pairs(cand_xy: np.ndarray, usable: np.ndarray, gt_xy: np.ndarray, radius: float) -> Dict[int, int]:
    """Which candidate covers which person: a one-to-one gated assignment, as `candidate_targets` makes it, but keeping the
    pairing rather than only the two boolean masks."""
    out = {}
    idx = np.nonzero(usable)[0]
    if not len(idx) or not len(gt_xy):
        return out
    cost = np.linalg.norm(cand_xy[idx, None, :] - gt_xy[None, :, :], axis=-1)
    gated = np.where(cost > radius, 1e6, cost)
    for i, j in zip(*linear_sum_assignment(gated)):
        if gated[i, j] < 1e6:
            out[int(j)] = int(idx[i])
    return out


@torch.no_grad()
def diagnose(model: ObjectMemory, split: Split, device, streams: int, threshold: float, stage3_threshold: float, slice_frames: int = 512) -> Dict[str, Dict[str, np.ndarray]]:
    """Per annotated person, what the memory did to the candidate covering them; per candidate, the rescoring delta and whether
    it covers anybody. Plays whole recordings from empty memory exactly as `train_three_horizon.evaluate` does."""
    model.eval()
    people = {k: [] for k in ("stage2", "stage3", "raw", "rescored", "delta", "slot_distance", "slot_value", "slot_since_match", "slot_age")}
    candidates = {k: [] for k in ("delta", "raw", "covers_person")}
    totals = {k: 0 for k in ("count", "stage2_detections", "stage2_covered", "stage3_detections", "stage3_covered")}
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
                alive, value = out.slots.alive[rows].cpu().numpy(), out.slots.value[rows].cpu().numpy()
                since, age = out.slots.since_match_s[rows].cpu().numpy(), out.slots.age_s[rows].cpu().numpy()
                valid, xy, score = batch["valid"][rows, i].cpu().numpy(), batch["xy"][rows, i].cpu().numpy(), batch["score"][rows, i].cpu().numpy()
                gt_xy, gt_valid = batch["gt_xy"][rows, i].cpu().numpy(), batch["gt_valid"][rows, i].cpu().numpy()
                for j in range(len(rows)):
                    gt = gt_xy[j][gt_valid[j]].astype(np.float64)
                    delta = logit(cand_p[j]) - logit(score[j])
                    by_stage2 = pairs(xy[j], valid[j] & (score[j] >= threshold), gt, MATCH_RADIUS_M)
                    reported = coasting[j] & (slot_p[j] >= max(stage3_threshold, MIN_REPORTED_SCORE))
                    by_stage3 = pairs(np.concatenate([xy[j], slot_xy[j]]),
                                      np.concatenate([valid[j] & (cand_p[j] >= stage3_threshold), reported]), gt, MATCH_RADIUS_M)
                    totals["count"] += 1
                    totals["stage2_detections"] += int((valid[j] & (score[j] >= threshold)).sum())
                    totals["stage2_covered"] += len(by_stage2)
                    totals["stage3_detections"] += int((valid[j] & (cand_p[j] >= stage3_threshold)).sum() + reported.sum())
                    totals["stage3_covered"] += len(by_stage3)
                    any_person = set(pairs(xy[j], valid[j], gt, MATCH_RADIUS_M).values())
                    for c in np.nonzero(valid[j])[0]:
                        candidates["delta"].append(float(delta[c])), candidates["raw"].append(float(score[j][c]))
                        candidates["covers_person"].append(int(c) in any_person)
                    live = np.nonzero(alive[j])[0]
                    for person in range(len(gt)):
                        people["stage2"].append(person in by_stage2), people["stage3"].append(person in by_stage3)
                        c = by_stage2.get(person)
                        people["raw"].append(float(score[j][c]) if c is not None else float("nan"))
                        people["rescored"].append(float(cand_p[j][c]) if c is not None else float("nan"))
                        people["delta"].append(float(delta[c]) if c is not None else float("nan"))
                        if len(live):
                            nearest = live[int(np.argmin(np.linalg.norm(slot_xy[j][live] - gt[person][None], axis=1)))]
                            people["slot_distance"].append(float(np.linalg.norm(slot_xy[j][nearest] - gt[person])))
                            people["slot_value"].append(float(value[j][nearest])), people["slot_since_match"].append(float(since[j][nearest]))
                            people["slot_age"].append(float(age[j][nearest]))
                        else:
                            for key in ("slot_distance", "slot_value", "slot_since_match", "slot_age"):
                                people[key].append(float("nan"))
    model.train()
    return dict(people={k: np.asarray(v) for k, v in people.items()}, candidates={k: np.asarray(v) for k, v in candidates.items()}, frames=totals)


def summarise(data: Dict, gate_m: float, threshold: float, stage3_threshold: float) -> Dict:
    """Split the people stage 3 lost by mechanism, and report the rescoring delta by whether a candidate covers anybody."""
    p, c = data["people"], data["candidates"]
    lost = p["stage2"] & ~p["stage3"]
    kept = p["stage2"] & p["stage3"]
    gained = ~p["stage2"] & p["stage3"]
    near = p["slot_distance"] <= gate_m

    print(f"\n=== WHAT STAGE 3 DID TO {len(p['stage2']):,} ANNOTATED PEOPLE (stage 2 at {threshold}, stage 3 at {stage3_threshold}) ===")
    print(f"  covered by stage 2 {p['stage2'].sum():,}   by stage 3 {p['stage3'].sum():,}   kept {kept.sum():,}  lost {lost.sum():,}  gained {gained.sum():,}")

    print(f"\n  of the {lost.sum():,} lost, by mechanism:")
    still_there = lost & np.isfinite(p["rescored"])
    pushed_under = still_there & (p["rescored"] < stage3_threshold)
    print(f"    rescoring pushed the candidate under {stage3_threshold}   {pushed_under.sum():>7,} ({pushed_under.sum() / max(lost.sum(), 1):>5.1%})")
    print(f"      their median raw score {np.nanmedian(p['raw'][pushed_under]):.3f} -> rescored {np.nanmedian(p['rescored'][pushed_under]):.3f} "
          f"(median delta {np.nanmedian(p['delta'][pushed_under]):+.3f} logits)")
    print(f"      a live slot sat within {gate_m} m for {np.count_nonzero(near & pushed_under) / max(pushed_under.sum(), 1):.1%} of them, "
          f"median value {np.nanmedian(p['slot_value'][pushed_under]):.3f}, median {np.nanmedian(p['slot_since_match'][pushed_under]):.1f} s since it last matched")
    other = lost & ~pushed_under
    print(f"    lost some other way (the person moved out of the 0.5 m match, or the pairing changed)  {other.sum():>7,} ({other.sum() / max(lost.sum(), 1):>5.1%})")

    print(f"\n  for contrast, the {kept.sum():,} kept: median delta {np.nanmedian(p['delta'][kept]):+.3f} logits, "
          f"a live slot within {gate_m} m for {np.count_nonzero(near & kept) / max(kept.sum(), 1):.1%}")

    covers, misses = c["covers_person"].astype(bool), ~c["covers_person"].astype(bool)
    print(f"\n=== THE TRADE THE RESCORE HEAD MAKES, over {len(c['delta']):,} candidates ===")
    print(f"  covering a person  ({covers.sum():>8,}): median delta {np.median(c['delta'][covers]):+.3f} logits, "
          f"{np.mean(c['delta'][covers] < 0):.1%} pushed down")
    print(f"  covering nobody    ({misses.sum():>8,}): median delta {np.median(c['delta'][misses]):+.3f} logits, "
          f"{np.mean(c['delta'][misses] < 0):.1%} pushed down")
    print("  A head that only suppressed phantoms would push the second group down and leave the first alone.")

    f = data["frames"]
    stage2_fp = (f["stage2_detections"] - f["stage2_covered"]) / max(f["count"], 1)
    stage3_fp = (f["stage3_detections"] - f["stage3_covered"]) / max(f["count"], 1)
    print(f"\n=== WHAT THE RECALL COSTS, over {f['count']:,} annotated frames ===")
    print(f"  stage 2 at {threshold}:        {f['stage2_detections'] / max(f['count'], 1):.3f} detections/frame, covering {f['stage2_covered'] / max(f['count'], 1):.3f} people -> "
          f"**{stage2_fp:.3f} false positives/frame**")
    print(f"  stage 3 at {stage3_threshold}:        {f['stage3_detections'] / max(f['count'], 1):.3f} detections/frame, covering {f['stage3_covered'] / max(f['count'], 1):.3f} people -> "
          f"**{stage3_fp:.3f} false positives/frame**")
    print("  Recall bought below stage 2's own false-positive rate is a real gain; above it is only a different operating point.")

    return dict(stage2_fp_per_frame=float(stage2_fp), stage3_fp_per_frame=float(stage3_fp), frames=int(f["count"]),
                people=int(len(p["stage2"])), stage2=int(p["stage2"].sum()), stage3=int(p["stage3"].sum()),
                kept=int(kept.sum()), lost=int(lost.sum()), gained=int(gained.sum()),
                pushed_under=int(pushed_under.sum()), other=int(other.sum()),
                median_delta_kept=float(np.nanmedian(p["delta"][kept])), median_delta_pushed_under=float(np.nanmedian(p["delta"][pushed_under])),
                median_delta_covering=float(np.median(c["delta"][covers])), median_delta_missing=float(np.median(c["delta"][misses])),
                share_pushed_down_covering=float(np.mean(c["delta"][covers] < 0)), share_pushed_down_missing=float(np.mean(c["delta"][misses] < 0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory", type=Path, default=_HERE.parent / "checkpoints_three_horizon" / "step3a_object_memory.best.pth")
    ap.add_argument("--stage2", type=Path, default=_HERE.parent / "checkpoints_three_horizon" / "step2_calibration.best.pth")
    ap.add_argument("--out", type=Path, default=_HERE.parent / "checkpoints_three_horizon" / "memory_diagnosis.json")
    ap.add_argument("--slots", type=int, default=0, help="slot capacity; 0 takes the trained checkpoint's own")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=OPERATING_THRESHOLD)
    ap.add_argument("--stage3-threshold", type=float, default=None, help="score stage 3 at its own threshold, to test whether the people it "
                                                                        "'loses' are an artefact of a threshold calibrated for stage 2 (default: the same)")
    ap.add_argument("--limit-recordings", type=int, default=0)
    args = ap.parse_args()

    device, _ = detect_device()
    checkpoint = torch.load(args.memory, map_location=device, weights_only=False)
    slots = args.slots or int(checkpoint["args"]["slots"])
    split = load_split("test", args.limit_recordings, args.stage2, device)
    rules = SlotRules(capacity=slots)
    model = ObjectMemory(rules, feature_dim=split.data.features.shape[-1]).to(device)
    model.load_state_dict(checkpoint["model"])
    print(f"memory: {args.memory.name}, {slots} slots, gate {rules.gate_m} m, fade {rules.fade_time_s} s", flush=True)

    stage3_threshold = args.threshold if args.stage3_threshold is None else args.stage3_threshold
    summary = summarise(diagnose(model, split, device, args.streams, args.threshold, stage3_threshold), rules.gate_m, args.threshold, stage3_threshold)
    args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
        code = 0
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)
