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
    python memory_diagnosis.py --slot-speed --cache .three_horizon_cache/<a test cache>.pkl   # is slot velocity corrupted by dt?
    python memory_diagnosis.py --slot-rules --cache .three_horizon_cache/<a test cache>.pkl   # do the association gate and the fade time bind?
    python memory_diagnosis.py --recovery                # of the people stage 2 misses, how many does the memory already hold?

The last one is the test that separates the two readings of the loss. Stage 3's rescoring shifts the whole score distribution
down -- on the full split the people it *kept* were pushed down harder (median -3.115 logits) than the ones it lost (-1.394) --
so a threshold fixed at stage 2's 0.3 is no longer calibrated for stage 3's output. If lowering only stage 3's threshold
recovers the people, the loss is an artefact of the fixed threshold and the fix is calibration, not architecture.

`--slot-speed` needs no trained model (`TODO.md` A51 0a). FROG's stamps come in bunches (38 ms, 38 ms, under 0.1 ms), stage 3
clips each interval to `MIN_DT_S`, and `manage_slots` divides a matched slot's displacement by it. This replays the rule-based
bookkeeping on a candidate cache twice, with each stamped interval and with training's running time step, and sets each slot's
velocity state against a reference that never divides by one interval: its displacement over at least `--window-s`.

`--slot-rules` needs no trained model either (`TODO.md` A51 S1 and S2). `manage_slots`' association gate and fade time need no
retraining to change (`docs/PROPOSAL.md` §6.3), so they are swept first: for each pair, how often a live slot stands on an
annotated person, how many slots match a candidate, how much the memory churns, and -- when a slot is lost and something
reappears where it stood -- how long the gap was. The last one prices what a slot's state would have to survive.

`--recovery` asks the question that decides where the remaining work goes (`TODO.md` A51). The oracle says ~10,000 annotated
people are recoverable -- their own trajectory is detected before and after the gap -- and stage 3 reports about a fifth of them.
For each person no kept detection covers, this measures whether the memory **already holds a live slot on them**. If it does, the
slots are full enough and the failure is in reporting, so the coasting threshold and the rescoring calibration are the levers; the
coasting curve below prices the first of those directly. If it does not, the memory never learned about them, and the lever is
more candidates per frame -- the slot capacity that replay shows is never used.
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
import torch
from scipy.optimize import linear_sum_assignment

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.detectors.three_horizon import ObjectMemory, SlotRules, initial_slots, manage_slots, sensor_to_world, world_to_sensor  # noqa: E402
from train_three_horizon import (  # noqa: E402
    MATCH_RADIUS_M, MAX_DT_S, MIN_DT_S, MIN_REPORTED_SCORE, OPERATING_THRESHOLD, PROBABILITY_EPS, SequenceCandidates, Split, detect_device, evaluation_segments, gather,
    load_split, mean_frame_period, object_memory, pack,
)

SPEED_WINDOW_S = 1.0
RETURN_HORIZON_S = 60.0
GATES_M = (0.3, 0.6, 1.0)
FADE_TIMES_S = (2.0, 10.0, 30.0)
GAP_PERCENTILES = (50, 90, 99)
COAST_THRESHOLDS = (0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01)
FAST_SPEED_M_S = 3.0
PERCENTILES = (50, 90, 99)


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


def windowed_speed(anchor_xy: np.ndarray, anchor_t: np.ndarray, xy: np.ndarray, t, refill: np.ndarray, window_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One frame of each slot's reference speed `[..., K]`: displacement from its anchor over the elapsed time, emitted (NaN
    otherwise) once at least `window_s` has passed, after which the anchor moves to the current position. A refilled slot holds a
    different object, so its anchor restarts there and nothing is emitted. Returns the speed and the new anchor."""
    elapsed = t - anchor_t
    ready = (elapsed >= window_s) & ~refill
    speed = np.where(ready, np.linalg.norm(xy - anchor_xy, axis=-1) / np.maximum(elapsed, window_s), np.nan)
    restart = ready | refill
    return speed, np.where(restart[..., None], xy, anchor_xy), np.where(restart, t, anchor_t)


def label_coasting_reports(report_xy: np.ndarray, report_score: np.ndarray, missed_xy: np.ndarray, radius: float):
    """Split coasting reports into the scores of those that would recover a person no kept detection covers, and the scores of
    those that would only add a false positive. One report per person, nearest first, as the benchmark matches."""
    if not len(report_xy):
        return np.zeros(0), np.zeros(0)
    hit = np.zeros(len(report_xy), dtype=bool)
    if len(missed_xy):
        cost = np.linalg.norm(report_xy[:, None, :] - missed_xy[None, :, :], axis=-1)
        gated = np.where(cost > radius, 1e6, cost)
        for i, j in zip(*linear_sum_assignment(gated)):
            if gated[i, j] < 1e6:
                hit[i] = True
    return report_score[hit], report_score[~hit]


def covered_people(slot_xy: np.ndarray, alive: np.ndarray, gt_xy: np.ndarray, gt_valid: np.ndarray, radius: float) -> int:
    """How many annotated people of one frame have a live slot within `radius`. An upper bound on recall from memory, not recall: it
    counts a slot standing on a person however it got there (`docs/PROPOSAL.md` §5.5 counts the same quantity)."""
    people, slots = gt_xy[gt_valid], slot_xy[alive]
    if not len(people) or not len(slots):
        return 0
    return int((np.linalg.norm(people[:, None, :] - slots[None, :, :], axis=-1).min(axis=1) <= radius).sum())


def retirement_gap(spawn_xy: np.ndarray, lost_xy: np.ndarray, lost_t: np.ndarray, now_t: float, radius: float, horizon_s: float) -> float:
    """Seconds since the most recent slot lost within `radius` of where this slot has just spawned, or NaN if none within
    `horizon_s`. FROG has no identities, so a short gap is evidence that the same object returned, not proof."""
    if not len(lost_xy):
        return float("nan")
    near = (np.linalg.norm(lost_xy - spawn_xy[None, :], axis=-1) <= radius) & (now_t - lost_t <= horizon_s)
    return float(now_t - lost_t[near].max()) if near.any() else float("nan")


@torch.no_grad()
def recovery(model: ObjectMemory, split: Split, device, streams: int, threshold: float, slice_frames: int = 512) -> Dict[str, object]:
    """Per annotated person that no kept detection covers: does a live slot sit on them, and is it coasting? Plus every coasting
    report's score, split by whether it would recover such a person, and the slots actually in use."""
    missed, held, held_coasting, people, covered, frames = 0, 0, 0, 0, 0, 0
    hit_scores, false_scores, in_use, held_value = [], [], [], []
    value_hit, value_false = [], []
    segments = sorted(evaluation_segments(split, 0, 0, 0), key=lambda seg: -seg[2])
    for group in [segments[i:i + streams] for i in range(0, len(segments), streams)]:
        length = max(seg[2] for seg in group)
        recording = np.array([seg[0] for seg in group])[:, None]
        frame = np.array([seg[1] for seg in group])[:, None] + np.minimum(np.arange(length)[None], np.array([seg[2] for seg in group])[:, None] - 1)
        chosen = np.zeros(frame.shape, dtype=bool)
        for b, (sq, first, n, _) in enumerate(group):
            chosen[b, np.nonzero(split.data.annotated[split.offsets[sq] + first:split.offsets[sq] + first + n])[0]] = True
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
                in_use.extend(out.slots.alive.sum(dim=1).cpu().numpy().tolist())
                if not len(rows):
                    continue
                cand_p = torch.sigmoid(out.candidate_logit)[rows].cpu().numpy()
                slot_p = torch.sigmoid(out.slot_logit)[rows].cpu().numpy()
                xy, valid = batch["xy"][rows, i].cpu().numpy(), batch["valid"][rows, i].cpu().numpy()
                slot_xy, alive = out.slot_xy[rows].cpu().numpy(), out.slots.alive[rows].cpu().numpy()
                coasting, value = out.coasting[rows].cpu().numpy(), out.slots.value[rows].cpu().numpy()
                gt_xy, gt_valid = batch["gt_xy"][rows, i].cpu().numpy(), batch["gt_valid"][rows, i].cpu().numpy()
                for j in range(len(rows)):
                    frames += 1
                    gt = gt_xy[j][gt_valid[j]].astype(np.float64)
                    people += len(gt)
                    kept = valid[j] & (cand_p[j] >= threshold)
                    taken = set(pairs(xy[j], kept, gt, MATCH_RADIUS_M))
                    covered += len(taken)
                    gaps = np.array([g for k, g in enumerate(gt) if k not in taken]).reshape(-1, 2)
                    missed += len(gaps)
                    for person in gaps:                     # does the memory already hold this person?
                        near = alive[j] & (np.linalg.norm(slot_xy[j] - person[None], axis=1) <= MATCH_RADIUS_M)
                        if near.any():
                            held += 1
                            held_value.append(float(value[j][near].max()))
                            held_coasting += int((near & coasting[j]).any())
                    reports = coasting[j] & alive[j]
                    hits, falses = label_coasting_reports(slot_xy[j][reports], slot_p[j][reports], gaps, MATCH_RADIUS_M)
                    hit_scores.append(hits), false_scores.append(falses)
                    # the same reports ranked by the rule's own accumulated value instead of the learned head, to see which knows more
                    hits, falses = label_coasting_reports(slot_xy[j][reports], value[j][reports], gaps, MATCH_RADIUS_M)
                    value_hit.append(hits), value_false.append(falses)
    return dict(people=people, covered=covered, missed=missed, held=held, held_coasting=held_coasting, frames=frames,
                hit_scores=np.concatenate(hit_scores) if hit_scores else np.zeros(0),
                false_scores=np.concatenate(false_scores) if false_scores else np.zeros(0),
                in_use=np.asarray(in_use), held_value=np.asarray(held_value),
                value_hit=np.concatenate(value_hit) if value_hit else np.zeros(0),
                value_false=np.concatenate(value_false) if value_false else np.zeros(0))


def summarise_recovery(data: Dict[str, object], threshold: float) -> Dict:
    """Whether the memory already holds the people it fails to report, and what reporting more of them would cost."""
    missed, held, frames = data["missed"], data["held"], max(data["frames"], 1)
    print(f"\n=== OF {data['people']:,} ANNOTATED PEOPLE, STAGE 3 AT {threshold} COVERS {data['covered']:,} ===")
    print(f"  missed {missed:,}; of those a live slot already sits on **{held:,} ({held / max(missed, 1):.1%})**, "
          f"and it is coasting for {data['held_coasting']:,} ({data['held_coasting'] / max(missed, 1):.1%})")
    if len(data["held_value"]):
        print(f"  those slots carry a median value of {np.median(data['held_value']):.3f}")
    print(f"  slots in use per frame: median {np.median(data['in_use']):.0f}, p99 {np.percentile(data['in_use'], 99):.0f}, "
          f"max {data['in_use'].max():.0f} of {data['in_use'].max() and 256}")

    hits, false = data["hit_scores"], data["false_scores"]
    print(f"\n=== WHAT REPORTING MORE COASTING SLOTS WOULD COST ({len(hits):,} reports land on a missed person, {len(false):,} on nobody) ===")
    print("  coast threshold   people recovered   share of the missed   false positives added per frame")
    rows = []
    for tau in COAST_THRESHOLDS:
        recovered, added = int((hits >= tau).sum()), int((false >= tau).sum())
        rows.append(dict(threshold=tau, recovered=recovered, added_per_frame=added / frames))
        print(f"  {tau:>15.2f}   {recovered:>16,}   {recovered / max(missed, 1):>19.1%}   {added / frames:>31.3f}")
    print("\n=== THE SAME REPORTS RANKED BY THE RULE'S ACCUMULATED VALUE, NOT THE LEARNED HEAD ===")
    print("  value threshold   people recovered   false positives added per frame   FP per person")
    value_rows = []
    for tau in COAST_THRESHOLDS:
        recovered, added = int((data["value_hit"] >= tau).sum()), int((data["value_false"] >= tau).sum())
        value_rows.append(dict(threshold=tau, recovered=recovered, added_per_frame=added / frames))
        print(f"  {tau:>15.2f}   {recovered:>16,}   {added / frames:>31.3f}   {added / max(recovered, 1):>13.1f}")

    return dict(threshold=threshold, people=int(data["people"]), covered=int(data["covered"]), missed=int(missed), held=int(held),
                value_curve=value_rows,
                held_coasting=int(data["held_coasting"]), frames=int(frames),
                slots_in_use_median=float(np.median(data["in_use"])), slots_in_use_p99=float(np.percentile(data["in_use"], 99)),
                slots_in_use_max=int(data["in_use"].max()), curve=rows)


@torch.no_grad()
def replay_slot_rules(split: Split, device, streams: int, rules: SlotRules, slice_frames: int = 512) -> Dict[str, float]:
    """Replay `manage_slots` under one set of rules over whole recordings: the share of annotated people a live slot stands on, the
    share whose slot also matched a candidate this frame, matches and churn per annotated frame, slots in use, and the gaps between
    losing a slot and a new one spawning where it stood."""
    people, covered, matched_cover, matches, spawns, losses, annotated_frames = 0, 0, 0, 0, 0, 0, 0
    in_use, gaps = [], []
    segments = sorted(evaluation_segments(split, 0, 0, 0), key=lambda seg: -seg[2])
    for group in [segments[i:i + streams] for i in range(0, len(segments), streams)]:
        length = max(seg[2] for seg in group)
        lengths = np.array([seg[2] for seg in group])
        recording = np.array([seg[0] for seg in group])[:, None]
        frame = np.array([seg[1] for seg in group])[:, None] + np.minimum(np.arange(length)[None], lengths[:, None] - 1)
        slots = initial_slots(len(group), rules, 1, device)
        was_alive = np.zeros((len(group), rules.capacity), bool)
        position = np.zeros((len(group), rules.capacity, 2))
        lost_xy = [np.zeros((0, 2)) for _ in group]
        lost_t = [np.zeros(0) for _ in group]
        for t0 in range(0, length, slice_frames):
            t1 = min(t0 + slice_frames, length)
            batch = gather(split, recording.repeat(t1 - t0, 1), frame[:, t0:t1], device)
            times = split.data.time[split.offsets[recording] + frame[:, t0:t1]]
            for t in range(t0, t1):
                i = t - t0
                world = sensor_to_world(batch["xy"][:, i], batch["pose"][:, i])
                slots, events = manage_slots(slots, world, batch["score"][:, i], batch["valid"][:, i], batch["dt"][:, i], rules)
                alive, refill = slots.alive.cpu().numpy(), (events.source >= 0).cpu().numpy()
                near = (torch.cdist(world_to_sensor(slots.position, batch["pose"][:, i]), batch["gt_xy"][:, i])
                        .masked_fill(~batch["gt_valid"][:, i, None, :], float("inf")).min(dim=2).values <= MATCH_RADIUS_M).cpu().numpy()
                slot_xy = world_to_sensor(slots.position, batch["pose"][:, i]).cpu().numpy()
                gt_xy, gt_valid = batch["gt_xy"][:, i].cpu().numpy(), batch["gt_valid"][:, i].cpu().numpy()
                matched = events.matched.cpu().numpy()
                lost = was_alive & (~alive | refill)
                for b in range(len(group)):
                    if t >= lengths[b]:
                        continue
                    now = float(times[b, i])
                    if lost[b].any():
                        keep = now - lost_t[b] <= RETURN_HORIZON_S
                        lost_xy[b] = np.concatenate([lost_xy[b][keep], position[b][lost[b]]])
                        lost_t[b] = np.concatenate([lost_t[b][keep], np.full(int(lost[b].sum()), now)])
                        losses += int(lost[b].sum())
                    for k in np.nonzero(refill[b])[0]:
                        gap = retirement_gap(slots.position[b, k].cpu().numpy(), lost_xy[b], lost_t[b], now, MATCH_RADIUS_M, RETURN_HORIZON_S)
                        if not np.isnan(gap):
                            gaps.append(gap)
                    spawns += int(refill[b].sum())
                    in_use.append(int(alive[b].sum()))
                    if not batch["annotated"][b, i]:
                        continue
                    annotated_frames += 1
                    people += int(gt_valid[b].sum())
                    covered += covered_people(slot_xy[b], alive[b], gt_xy[b], gt_valid[b], MATCH_RADIUS_M)
                    matched_cover += covered_people(slot_xy[b], alive[b] & matched[b], gt_xy[b], gt_valid[b], MATCH_RADIUS_M)
                    matches += int((alive[b] & matched[b]).sum())
                position = slots.position.cpu().numpy()
                was_alive = alive
    frames = max(annotated_frames, 1)
    gaps = np.asarray(gaps)
    return dict(gate_m=rules.gate_m, fade_time_s=rules.fade_time_s, people=people, covered_share=covered / max(people, 1),
                matched_share=matched_cover / max(people, 1), matches_per_frame=matches / frames, spawns_per_frame=spawns / frames,
                losses_per_frame=losses / frames, slots_p50=float(np.percentile(in_use, 50)), slots_p99=float(np.percentile(in_use, 99)),
                returns_per_frame=len(gaps) / frames, gap_share_over_fade=float(np.mean(gaps > rules.fade_time_s)) if len(gaps) else float("nan"),
                **{f"gap_p{q}": float(np.percentile(gaps, q)) if len(gaps) else float("nan") for q in GAP_PERCENTILES})


def summarise_slot_rules(rows: List[Dict[str, float]]) -> Dict:
    """The sweep as one table: coverage and churn per (gate, fade), then the return gaps that price surviving state."""
    print(f"\n=== SLOT RULES over {rows[0]['people']:,} annotated people: a live slot within {MATCH_RADIUS_M} m ===")
    print("  gate   fade    people covered   also matched   matches/frame   spawns/frame   losses/frame   slots p50/p99")
    for r in rows:
        print(f"  {r['gate_m']:.2f} m  {r['fade_time_s']:>4.0f} s   {r['covered_share']:>13.1%}   {r['matched_share']:>12.1%}   "
              f"{r['matches_per_frame']:>13.2f}   {r['spawns_per_frame']:>12.2f}   {r['losses_per_frame']:>12.2f}   "
              f"{r['slots_p50']:>6.0f}/{r['slots_p99']:.0f}")
    print(f"\n=== WHEN A SLOT IS LOST AND SOMETHING RETURNS WHERE IT STOOD (within {MATCH_RADIUS_M} m, {RETURN_HORIZON_S:g} s) ===")
    print("  gate   fade    returns/frame   gap p50 / p90 / p99   share beyond the fade time")
    for r in rows:
        print(f"  {r['gate_m']:.2f} m  {r['fade_time_s']:>4.0f} s   {r['returns_per_frame']:>13.2f}   "
              + " / ".join(f"{r[f'gap_p{q}']:5.2f}" for q in GAP_PERCENTILES) + f" s   {r['gap_share_over_fade']:>10.1%}")
    return dict(match_radius_m=MATCH_RADIUS_M, return_horizon_s=RETURN_HORIZON_S, rows=rows)


@torch.no_grad()
def replay_slot_speeds(split: Split, device, streams: int, rules: SlotRules, stamped: bool, window_s: float,
                       slice_frames: int = 512) -> Dict[str, np.ndarray]:
    """Replay `manage_slots` over whole recordings, with each frame's own stamped interval (clipped to `MIN_DT_S`, as training did
    before 2026-09-17) or with training's running time step (`frame_periods`). For every
    live slot matched on an annotated frame, the speed of its velocity state; for every live slot, its `windowed_speed`; each
    tagged by whether an annotated person stands within the match radius. Also the matches per annotated frame."""
    out = {k: [] for k in ("state_speed", "state_near", "window_speed", "window_near")}
    matches, annotated_frames = 0, 0
    segments = sorted(evaluation_segments(split, 0, 0, 0), key=lambda seg: -seg[2])
    for group in [segments[i:i + streams] for i in range(0, len(segments), streams)]:
        length = max(seg[2] for seg in group)
        lengths = np.array([seg[2] for seg in group])
        recording = np.array([seg[0] for seg in group])[:, None]
        frame = np.array([seg[1] for seg in group])[:, None] + np.minimum(np.arange(length)[None], lengths[:, None] - 1)
        slots = initial_slots(len(group), rules, 1, device)
        anchor_xy, anchor_t = np.zeros((len(group), rules.capacity, 2)), np.zeros((len(group), rules.capacity))
        for t0 in range(0, length, slice_frames):
            t1 = min(t0 + slice_frames, length)
            batch = gather(split, recording.repeat(t1 - t0, 1), frame[:, t0:t1], device)
            times = split.data.time[split.offsets[recording] + frame[:, t0:t1]]
            for t in range(t0, t1):
                i = t - t0
                flat = split.offsets[recording[:, 0]] + frame[:, t]
                interval = np.where(frame[:, t] > np.array([seg[1] for seg in group]), split.data.time[flat] - split.data.time[np.maximum(flat - 1, 0)], MIN_DT_S)
                dt = torch.from_numpy(interval.clip(MIN_DT_S, MAX_DT_S).astype(np.float32)).to(device) if stamped else batch["dt"][:, i]
                world = sensor_to_world(batch["xy"][:, i], batch["pose"][:, i])
                slots, events = manage_slots(slots, world, batch["score"][:, i], batch["valid"][:, i], dt, rules)
                gt = batch["gt_xy"][:, i]
                dist = torch.cdist(world_to_sensor(slots.position, batch["pose"][:, i]), gt).masked_fill(~batch["gt_valid"][:, i, None, :], float("inf"))
                near = (dist.min(dim=2).values <= MATCH_RADIUS_M).cpu().numpy()
                alive, matched, refill = slots.alive.cpu().numpy(), events.matched.cpu().numpy(), (events.source >= 0).cpu().numpy()
                inside = (t < lengths)[:, None]
                speed, anchor_xy, anchor_t = windowed_speed(anchor_xy, anchor_t, slots.position.cpu().numpy(), times[:, i, None], refill | (t == 0), window_s)
                keep = inside & alive & np.isfinite(speed)
                out["window_speed"].append(speed[keep]), out["window_near"].append(near[keep])
                annotated = inside[:, 0] & batch["annotated"][:, i].cpu().numpy()
                chosen = annotated[:, None] & alive & matched
                out["state_speed"].append(slots.velocity.norm(dim=-1).cpu().numpy()[chosen]), out["state_near"].append(near[chosen])
                matches += int(chosen.sum())
                annotated_frames += int(annotated.sum())
    result = {k: np.concatenate(v) for k, v in out.items()}
    result["matches_per_frame"] = np.array(matches / max(annotated_frames, 1))
    return result


def summarise_slot_speeds(runs: Dict[str, Dict[str, np.ndarray]], clipped_share: float, period_s: float) -> Dict:
    """Velocity-state speeds against the windowed reference, near people and elsewhere, per replay."""
    print(f"\n=== SLOT SPEEDS: mean frame period {period_s * 1000:.2f} ms ({1 / period_s:.1f} Hz); "
          f"{clipped_share:.1%} of stamped intervals clipped to {MIN_DT_S * 1000:.0f} ms ===")
    summary = dict(period_s=period_s, clipped_share=clipped_share)
    for name, r in runs.items():
        row = dict(matches_per_frame=float(r["matches_per_frame"]))
        print(f"\n  dt {name}: {row['matches_per_frame']:.2f} slot matches per annotated frame")
        for is_near, label in ((True, "near a person"), (False, "elsewhere")):
            for kind in ("state", "window"):
                speeds = r[f"{kind}_speed"][r[f"{kind}_near"] == is_near]
                stats = dict(count=int(len(speeds)), fast_share=float(np.mean(speeds > FAST_SPEED_M_S)) if len(speeds) else float("nan"),
                             **{f"p{q}": float(np.percentile(speeds, q)) if len(speeds) else float("nan") for q in PERCENTILES})
                row[f"{kind}_{'near' if is_near else 'elsewhere'}"] = stats
                print(f"    {label:<14} {'velocity state' if kind == 'state' else 'windowed':<15} n {stats['count']:>10,}   "
                      + "   ".join(f"p{q} {stats[f'p{q}']:6.2f}" for q in PERCENTILES) + f" m/s   over {FAST_SPEED_M_S:g} m/s {stats['fast_share']:6.1%}")
        summary[name] = row
    return summary


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
    ap.add_argument("--slot-speed", action="store_true", help="replay slot bookkeeping only, and check slot velocity against a windowed reference")
    ap.add_argument("--slot-rules", action="store_true", help="sweep the association gate and fade time over the same replay, with the return gaps")
    ap.add_argument("--recovery", action="store_true", help="of the people stage 3 fails to report, how many does the memory already hold, and what would "
                                                            "reporting more coasting slots cost")
    ap.add_argument("--cache", type=Path, default=None, help="with --slot-speed: a candidate cache pickle to read directly instead of --stage2's")
    ap.add_argument("--window-s", type=float, default=SPEED_WINDOW_S, help="reference speed window, seconds")
    args = ap.parse_args()

    device, _ = detect_device()
    if args.slot_speed or args.slot_rules:
        if args.cache:
            seqs = [SequenceCandidates(**fields_of) for fields_of in pickle.loads(args.cache.read_bytes())]
            split = pack(seqs[:args.limit_recordings] if args.limit_recordings else seqs)
        else:
            split = load_split("test", args.limit_recordings, args.stage2, device)
        capacity = args.slots or SlotRules().capacity
        if args.slot_rules:
            rows = [replay_slot_rules(split, device, args.streams, SlotRules(capacity=capacity, gate_m=gate, fade_time_s=fade))
                    for gate in GATES_M for fade in FADE_TIMES_S]
            args.out.write_text(json.dumps(summarise_slot_rules(rows), indent=1), encoding="utf-8")
            print(f"\nwrote {args.out}", flush=True)
            return
        rules = SlotRules(capacity=capacity)
        period = mean_frame_period(split.data.time, split.offsets, split.lengths)
        intervals = np.concatenate([np.diff(split.data.time[o:o + n]) for o, n in zip(split.offsets, split.lengths)])
        runs = {name: replay_slot_speeds(split, device, args.streams, rules, stamped, args.window_s) for name, stamped in (("stamped", True), ("running average", False))}
        summary = summarise_slot_speeds(runs, float(np.mean(intervals < MIN_DT_S)), period)
        args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}", flush=True)
        return
    checkpoint = torch.load(args.memory, map_location=device, weights_only=False)
    slots = args.slots or int(checkpoint["args"]["slots"])
    split = load_split("test", args.limit_recordings, args.stage2, device)
    rules = SlotRules(capacity=slots)
    model = object_memory(checkpoint, split.data.features.shape[-1], capacity=slots).to(device)
    print(f"memory: {args.memory.name}, {slots} slots, gate {rules.gate_m} m, fade {rules.fade_time_s} s", flush=True)

    stage3_threshold = args.threshold if args.stage3_threshold is None else args.stage3_threshold
    if args.recovery:
        summary = summarise_recovery(recovery(model, split, device, args.streams, stage3_threshold), stage3_threshold)
        args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}", flush=True)
        return
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
