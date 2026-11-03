#!/usr/bin/env python3
"""The tracking literature's baselines and metrics against TAKHeLiPeD's object memory.

What each experiment decides, and what was predicted before it ran, is `memory/experiments/2026-09-tracking-claims.md`. Every run
replays the same cached stage-2 candidates the memory was trained and scored on (the no-temporal stage 2, `step2_static`), tunes
on FROG `official` val and scores the test split once.

    rule      E15: the memory's own bookkeeping (`manage_slots`) with CBMOT's score update in place of the learned head
    track     E14: SORT, ByteTrack, and ByteTrack with CBMOT's track confidence, tuned by coordinate descent on val
    arms      the memory checkpoints and the published LFE detectors, their frames saved for the report
    report    AP with block-bootstrap intervals, GOSPA (E16), the gain by stratum (E20) and first-detection latency (E21)
    latency   E22: the proposed detector streamed one scan at a time, stage by stage, on the GPU and the CPU
    merge     the LFE detectors' merge radius on every saved output, chosen on val, with the double reports dissected
    summary   the reports of several replays (`<out>/s*/report.json`, one per stage-2 seed) lined up by arm

Usage (from utils/)
-----
    python tracking_baselines.py rule
    python tracking_baselines.py track --workers 8
    python tracking_baselines.py arms
    python tracking_baselines.py report
    python tracking_baselines.py latency --device cpu --threads 4
    python tracking_baselines.py merge --workers 6
    python tracking_baselines.py arms --stage2 <stage 2> --out <dir> --arms E13:a60_e13_static_s0      # a memory other than the default
    python tracking_baselines.py summary --out ../checkpoints_three_horizon/a61

The module itself imports no torch: the replays, GOSPA and the precision-recall arithmetic are numpy, so `track` can fan its
configurations out over worker processes without paying for a torch import in each, which the commit limit forbids on this machine
(`memory/gotchas.md`). Everything that needs torch imports it inside the function that does.
"""

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import auc as sklearn_auc

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from takheliped.utils.tracking import CBMOT_RULES, ByteTracker, odom_xya_delta_to_tracker_frame  # noqa: E402

CHECKPOINTS = _HERE.parent / "checkpoints_three_horizon"
STAGE2 = CHECKPOINTS / "step2_static" / "step2_calibration.best.pth"
OUT_DIR = CHECKPOINTS / "a56"
MATCH_RADIUS_M = 0.5                 # the benchmark's association distance, `train_takheliped.MATCH_RADIUS_M`
OPERATING_THRESHOLD = 0.3
SCORED_STRIDE = 5                     # every 5th annotated frame, as every published stage-3 number here
BLOCK_S = 30.0                        # bootstrap block: long enough that people rarely cross one, short enough for ~100 per recording
RESAMPLES = 1000
GOSPA_C_M, GOSPA_P = 0.5, 1
THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.05, 1.0, 0.05)) + (0.97, 0.99)   # GOSPA's optimum sits high: it charges a miss and a phantom alike

# E15: the memory's bookkeeping with CBMOT's update. `gain = 1` is CBMOT's multiplication rule exactly; the others generalise it.
RULE_GAINS = (0.1, 0.3, 1.0)
RULE_FADES_S = (2.0, 10.0, 30.0)
RULE_UNMATCHED = (1.0, 0.5)
RULE_COAST = (None, 0.5, 0.8)

# E14: fixed rows from the papers, then coordinate descent from TRACK_START over TRACK_AXES, two passes.
BYTETRACK_PUBLISHED = dict(two_stage=True, high_thresh=0.5, low_thresh=0.1, new_track_thresh=0.6, match_radius=0.5, max_age_s=1.0,
                           score_mode="detection", score_rule="multiplication", score_decay_per_s=0.0, unmatched_scale=0.0,
                           coast_s=0.0, output="tracks")
TRACK_START = dict(two_stage=True, high_thresh=0.5, low_thresh=0.1, new_track_thresh=None, match_radius=0.5, max_age_s=1.0,
                   score_mode="track", score_rule="multiplication", score_decay_per_s=0.0, unmatched_scale=1.0, coast_s=0.0,
                   output="rescored")
TRACK_AXES = (("high_thresh", (0.1, 0.2, 0.3, 0.5, 0.7)), ("low_thresh", (0.01, 0.05, 0.1)), ("match_radius", (0.3, 0.5, 0.8)),
              ("max_age_s", (0.25, 1.0, 3.0)), ("score_decay_per_s", (0.0, 0.1, 0.5, 2.0)), ("unmatched_scale", (0.0, 0.5, 1.0)),
              ("coast_s", (0.0, 0.5, 2.0)), ("score_rule", CBMOT_RULES))
TRACK_PASSES = 2

ARMS = {"E13": [f"a51_e67_recurrent_s{s}" for s in range(3)],
        "B0": [f"a51_b0_short_s{s}" for s in range(3)],
        "E10": [f"a51_e10_norecurrent_s{s}" for s in range(2)]}

Frames = List[Tuple[np.ndarray, np.ndarray, np.ndarray]]      # per frame: scores, xy, annotated people


def parse_arms(items: Sequence[str]) -> Dict[str, List[str]]:
    """`--arms` items `ARM:RUN` -> {arm: [runs, in order]}, the shape of `ARMS`."""
    arms: Dict[str, List[str]] = {}
    for item in items:
        arm, sep, run = item.partition(":")
        if not (sep and arm and run):
            raise ValueError(f"--arms takes ARM:RUN items, e.g. E13:a60_e13_static_s0, not {item!r}")
        arms.setdefault(arm, []).append(run)
    return arms


def arm_name(method: str) -> str:
    """A method's name without its run, `E13:a60_e13_static_s0` -> `E13`, so one arm's seeds line up across reports."""
    return method.split(":")[0]


def seed_stats(values: Sequence[float]) -> Dict[str, object]:
    return dict(seeds=list(values), mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"))


def summarise(reports: Sequence[Dict]) -> Dict[str, object]:
    """Several `report` outputs -- one replay per stage-2 seed -- lined up by arm: AP, recall at the candidates'
    false-positive rate, GOSPA, first-frame detection and strata recall as seed lists with mean and sd, and each bootstrap pair's
    differences with how many of its intervals sit above zero."""
    ap, recall, gospa, first = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    boot = defaultdict(lambda: defaultdict(list))
    strata = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in reports:
        for m, row in r["ap"].items():
            ap[arm_name(m)].append(row["ap"])
            recall[arm_name(m)].append(row["matched_fp_recall"])
        for pair, b in r.get("bootstrap", {}).items():
            a, other = pair.split(" - ", 1)
            for k in ("difference", "low", "high"):
                boot[f"{arm_name(a)} - {arm_name(other)}"][k].append(b[k])
        for m, g in r.get("gospa", {}).items():
            gospa[arm_name(m)].append(g["at_val_best"]["gospa"])
        for m, lat in r.get("latency", {}).items():
            first[arm_name(m)].append(lat["first_frame"])
        for axis, values in r.get("strata", {}).items():
            for value, cells in values.items():
                for m, c in cells.items():
                    strata[axis][value][arm_name(m)].append(c["recall"])
    return dict(ap={m: seed_stats(v) for m, v in ap.items()}, matched_fp_recall={m: seed_stats(v) for m, v in recall.items()},
                bootstrap={pair: dict(**{k: list(v) for k, v in b.items()}, above_zero=int(sum(low > 0 for low in b["low"])))
                           for pair, b in boot.items()},
                gospa={m: seed_stats(v) for m, v in gospa.items()}, first_frame={m: seed_stats(v) for m, v in first.items()},
                strata={axis: {value: {m: seed_stats(v) for m, v in cells.items()} for value, cells in values.items()} for axis, values in strata.items()},
                best_classical=[r.get("best_classical") for r in reports])


# ---------------------------------------------------------------- metrics

def gospa(pred_xy: np.ndarray, gt_xy: np.ndarray, c: float = GOSPA_C_M, p: int = GOSPA_P) -> Tuple[float, float, int, int]:
    """GOSPA with alpha = 2 between predicted and annotated points (Rahmathullah, Garcia-Fernandez & Svensson, *FUSION* 2017):
    `(total, localisation, missed, false)`, where pairs closer than `c` cost their distance to the power `p` and every missed or
    false target costs `c^p / 2`. Not normalised by cardinality, unlike OSPA, so every phantom costs the same however many people
    are present -- the accounting `docs/PAPER.md` argues AP lacks."""
    pred, gt = np.asarray(pred_xy, np.float64).reshape(-1, 2), np.asarray(gt_xy, np.float64).reshape(-1, 2)
    n, m = len(pred), len(gt)
    if n == 0 or m == 0:
        return (c ** p / 2 * (n + m)) ** (1.0 / p), 0.0, m, n
    d = cdist(pred, gt)
    rows, cols = linear_sum_assignment(np.minimum(d, c) ** p)
    paired = d[rows, cols] < c
    loc = float(np.sum(d[rows, cols][paired] ** p))
    assigned = int(paired.sum())
    missed, false = m - assigned, n - assigned
    return float((loc + c ** p / 2 * (missed + false)) ** (1.0 / p)), loc, missed, false


def gospa_frames(frames: Frames, threshold: float, c: float = GOSPA_C_M, p: int = GOSPA_P) -> Dict[str, float]:
    """Mean per-frame GOSPA of the detections at or above `threshold`, with its parts per frame."""
    parts = np.array([gospa(xy[np.asarray(s) >= threshold], g, c, p) for s, xy, g in frames], dtype=np.float64).reshape(-1, 4)
    return dict(gospa=float(parts[:, 0].mean()), localisation=float(parts[:, 1].mean()), missed=float(parts[:, 2].mean()),
                false=float(parts[:, 3].mean()), frames=len(frames))


def safe_auc(recs: np.ndarray, precs: np.ndarray) -> float:
    """`train._safe_auc`, repeated here so this module needs no torch."""
    mask = ~np.isnan(recs) & ~np.isnan(precs)
    r, p = recs[mask], precs[mask]
    if len(r) < 2:
        return float("nan")
    order = np.argsort(r)
    return float(sklearn_auc(r[order], p[order]))


class PrecisionRecall:
    """The precision-recall curve of `train_takheliped.score_detections`, kept as per-detection true-positive increments.

    `drow_utils._prec_rec_2d` accepts detections in descending score order and, at each one, re-solves the largest one-to-one
    matching within `radius` in that detection's frame. A frame's match count only ever depends on its own accepted detections, so
    each detection's increment is fixed once; the curve is then a running sum. That reproduces `score_detections`' AP exactly (a
    test pins it), and lets a bootstrap re-weight frames without matching again.
    """

    def __init__(self, frames: Frames, radius: float = MATCH_RADIUS_M):
        sizes = [len(s) for s, _, _ in frames]
        scores = np.concatenate([np.asarray(s, np.float32) for s, _, _ in frames]) if frames else np.zeros(0, np.float32)
        xy = np.concatenate([np.asarray(x, np.float32).reshape(-1, 2) for _, x, _ in frames]) if frames else np.zeros((0, 2), np.float32)
        det_frame = np.repeat(np.arange(len(frames)), sizes)
        gts = [np.asarray(g, np.float32).reshape(-1, 2) for _, _, g in frames]
        self.gt_count = np.array([len(g) for g in gts], dtype=np.int64)
        order = np.argsort(scores, kind="mergesort")[::-1]
        increment = np.zeros(len(scores), np.int64)
        accepted, matched = defaultdict(list), np.zeros(len(frames), np.int64)
        for idx in order:
            f = det_frame[idx]
            accepted[f].append(idx)
            g = gts[f]
            if not len(g) or matched[f] == len(g):
                continue
            if cdist(g, xy[idx:idx + 1]).min() > radius:      # a detection with no annotation in reach cannot grow the matching
                continue
            too_far = radius < cdist(g, xy[accepted[f]])
            rows, cols = linear_sum_assignment(too_far)
            now = int(np.sum(~too_far[rows, cols]))
            increment[idx], matched[f] = now - matched[f], now
        self.thresholds = scores[order]
        self.increment = increment[order]
        self.frame = det_frame[order]

    def curve(self, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """`(recall, precision, true positives, detections, annotated people)` down the thresholds, each frame counted `weights`
        times. Thresholds whose frame has weight zero are dropped."""
        if weights is None:
            tp, n = np.cumsum(self.increment), np.arange(1, len(self.increment) + 1)
            people = float(self.gt_count.sum())
            keep = slice(None)
        else:
            w = np.asarray(weights, np.float64)[self.frame]
            tp, n = np.cumsum(self.increment * w), np.cumsum(w)
            people = float((self.gt_count * np.asarray(weights, np.float64)).sum())
            keep = w > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            precs = (tp / n).astype(np.float32)                 # `_prec_rec_2d` fills float32 arrays, and AP must match it exactly
            recs = (tp / people).astype(np.float32) if people > 0 else np.full(len(tp), np.nan, np.float32)
        return recs[keep], precs[keep], tp[keep], n[keep], people

    def ap(self, weights: Optional[np.ndarray] = None) -> float:
        recs, precs, _, _, _ = self.curve(weights)
        return safe_auc(recs, precs)

    def at_threshold(self, threshold: float, frames: int) -> Dict[str, float]:
        """Precision, recall and errors per frame at the lowest listed threshold not under `threshold`, read off the curve as
        `score_detections` does."""
        recs, precs, tp, n, people = self.curve()
        idx = np.nonzero(self.thresholds >= threshold)[0]
        i = int(idx[-1]) if len(idx) else 0
        return dict(precision=float(precs[i]), recall=float(recs[i]), fp_per_frame=float(n[i] - tp[i]) / frames,
                    fn_per_frame=float(people - tp[i]) / frames)

    def at_fp_budget(self, budget: float, frames: int) -> Dict[str, float]:
        """The operating point spending at most `budget` false positives per frame: threshold and recall, as `score_detections`'
        `own_*` fields -- the "recall at matched FP" every stage-3 arm reports."""
        recs, _, tp, n, _ = self.curve()
        within = np.nonzero((n - tp) / frames <= budget + 1e-9)[0]
        if not len(within):
            return dict(threshold=float("nan"), recall=float("nan"), fp_per_frame=float("nan"))
        j = int(within[-1])
        return dict(threshold=float(self.thresholds[j]), recall=float(recs[j]), fp_per_frame=float(n[j] - tp[j]) / frames)


def block_ids(recording: np.ndarray, time_s: np.ndarray, block_s: float = BLOCK_S) -> np.ndarray:
    """A bootstrap block per scored frame: consecutive `block_s` stretches of one recording, never spanning two."""
    recording, time_s = np.asarray(recording), np.asarray(time_s, np.float64)
    start = {r: time_s[recording == r].min() for r in np.unique(recording)}
    slot = np.floor((time_s - np.array([start[r] for r in recording])) / block_s).astype(np.int64)
    _, ids = np.unique(np.stack([recording, slot], axis=1), axis=0, return_inverse=True)
    return ids.reshape(-1)


def paired_bootstrap(frames_a: Frames, frames_b: Frames, blocks: np.ndarray, resamples: int = RESAMPLES, seed: int = 0) -> Dict[str, float]:
    """AP(a) - AP(b) on the same frames, with a 95% interval from resampling whole blocks with replacement. The interval covers
    which frames happened to be recorded, not training seeds: a learned arm's seeds each get their own."""
    a, b = PrecisionRecall(frames_a), PrecisionRecall(frames_b)
    unique, inverse = np.unique(np.asarray(blocks), return_inverse=True)
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    for k in range(resamples):
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))[inverse].astype(np.float64)
        diffs[k] = a.ap(weights) - b.ap(weights)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return dict(difference=float(a.ap() - b.ap()), low=float(low), high=float(high), a=float(a.ap()), b=float(b.ap()), resamples=resamples)


# ---------------------------------------------------------------- replays

def stride_frames(split, stride: int = SCORED_STRIDE) -> np.ndarray:
    """Packed indices of every `stride`-th annotated frame of each recording, the frames `train_takheliped.evaluate` scores."""
    picked = [np.nonzero(split.data.annotated[o:o + n])[0][::stride] + o for o, n in zip(split.offsets, split.lengths)]
    return np.concatenate(picked).astype(np.int64) if picked else np.zeros(0, np.int64)


def recording_of(split, index: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(split.offsets), np.asarray(index), side="right") - 1


def replay_tracker(split, factory: Callable[[], ByteTracker], output: str = "tracks", keep: Optional[set] = None) -> Tuple[Frames, List[int]]:
    """A tracker from `factory` over every frame of every recording of `split`'s candidates, with ego-motion from the poses exactly
    as `train_takheliped.replay_sort` applies it; on each annotated frame in `keep` (all when None), its reported tracks
    (`output="tracks"`) or every candidate as the tracker rescored it (`"rescored"`). Returns the frames and their packed indices."""
    d = split.data
    frames, index = [], []
    for offset, n in zip(split.offsets, split.lengths):
        tracker = factory()
        previous = None
        for f in range(int(offset), int(offset + n)):
            x, y, theta = (float(v) for v in d.pose[f])
            if previous is None:
                dtheta, dxy = 0.0, (0.0, 0.0)
            else:
                dtheta, dxy = theta - previous[2], odom_xya_delta_to_tracker_frame(x - previous[0], y - previous[1], theta)
            previous = (x, y, theta)
            valid = d.valid[f]
            reported = tracker.step([(float(sc), float(px), float(py)) for sc, (px, py) in zip(d.score[f][valid], d.xy[f][valid])],
                                    dtheta, dxy, dt=float(split.dt[f]))
            if d.annotated[f] and (keep is None or f in keep):
                rows = np.array(tracker.rescored if output == "rescored" else reported, dtype=np.float64).reshape(-1, 3)
                frames.append((rows[:, 0], rows[:, 1:], d.gt_xy[f][d.gt_valid[f]]))
                index.append(f)
    return frames, index


def replay_rule_scorer(split, device, streams: int, rules, unmatched_scale: float, coast_floor: Optional[float],
                       keep: Optional[set] = None, slice_frames: int = 512) -> Tuple[Frames, List[int]]:
    """E15: `manage_slots` -- the memory's own association, value rule, spawning and retirement -- over every recording, with no
    learned part. A candidate matched to a slot takes the slot's new value (with `gain = 1`, CBMOT's multiplication rule); every
    other candidate keeps `unmatched_scale` times its own score; with `coast_floor`, a live unmatched slot at or above it reports
    itself at its faded value, as a tracker coasting a lost track. Frames are returned in packed order."""
    import torch
    from takheliped.detectors.takheliped import initial_slots, manage_slots, sensor_to_world, world_to_sensor
    from train_takheliped import evaluation_segments, gather

    out: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    d = split.data
    segments = sorted(evaluation_segments(split, 0, 0, 0), key=lambda seg: -seg[2])
    with torch.no_grad():
        for group in [segments[i:i + streams] for i in range(0, len(segments), streams)]:
            length = max(seg[2] for seg in group)
            lengths = np.array([seg[2] for seg in group])
            recording = np.array([seg[0] for seg in group])[:, None]
            frame = np.array([seg[1] for seg in group])[:, None] + np.minimum(np.arange(length)[None], lengths[:, None] - 1)
            packed = np.asarray(split.offsets)[recording] + frame
            slots = initial_slots(len(group), rules, 1, device)
            for t0 in range(0, length, slice_frames):
                t1 = min(t0 + slice_frames, length)
                batch = gather(split, recording.repeat(t1 - t0, 1), frame[:, t0:t1], device)
                for t in range(t0, t1):
                    i = t - t0
                    pose = batch["pose"][:, i]
                    slots, events = manage_slots(slots, sensor_to_world(batch["xy"][:, i], pose), batch["score"][:, i], batch["valid"][:, i],
                                                 batch["dt"][:, i], rules)
                    rows = [b for b in range(len(group)) if t < lengths[b] and d.annotated[packed[b, t]] and (keep is None or int(packed[b, t]) in keep)]
                    if not rows:
                        continue
                    value, matched, which = slots.value.cpu().numpy(), events.matched.cpu().numpy(), events.match_index.cpu().numpy()
                    alive, refill = slots.alive.cpu().numpy(), (events.source >= 0).cpu().numpy()
                    slot_xy = world_to_sensor(slots.position, pose).cpu().numpy()
                    for b in rows:
                        g = int(packed[b, t])
                        valid = d.valid[g]
                        score = d.score[g].astype(np.float64) * unmatched_scale
                        score[which[b][matched[b]]] = value[b][matched[b]]
                        scores, xy = score[valid], d.xy[g][valid].astype(np.float64)
                        if coast_floor is not None:
                            coast = alive[b] & ~matched[b] & ~refill[b] & (value[b] >= coast_floor)
                            scores, xy = np.concatenate([scores, value[b][coast]]), np.concatenate([xy, slot_xy[b][coast]])
                        out[g] = (scores, xy, d.gt_xy[g][d.gt_valid[g]])
    index = sorted(out)
    return [out[g] for g in index], index


# ---------------------------------------------------------------- strata

def occlusion_share(ranges: np.ndarray, angles: np.ndarray, person_xy: np.ndarray, person_radius_m: float = 0.25, margin_m: float = 0.5) -> float:
    """The share of the beams spanning a person's width at their range that return something at least `margin_m` in front of them:
    0 visible, 1 hidden. Missing returns never occlude."""
    x, y = float(person_xy[0]), float(person_xy[1])
    r, phi = float(np.hypot(x, y)), float(np.arctan2(-x, y))
    half = float(np.arcsin(min(1.0, person_radius_m / max(r, person_radius_m))))
    window = np.abs(np.asarray(angles) - phi) <= half
    if not window.any():
        return 0.0
    beams = np.asarray(ranges, np.float64)[window]
    return float(np.mean(np.isfinite(beams) & (beams > 0) & (beams < r - margin_m)))


def first_detection_delays(tracks: Sequence[Sequence[Tuple[int, bool]]]) -> List[Optional[int]]:
    """Per trajectory of `(frame, covered)` points: frames from its first annotation to its first covered point, None if never."""
    delays = []
    for points in tracks:
        points = sorted(points)
        first = points[0][0]
        covered = [f for f, c in points if c]
        delays.append(covered[0] - first if covered else None)
    return delays


# ---------------------------------------------------------------- merging an output that has no merge step

def merge_detections(scores: np.ndarray, xy: np.ndarray, radius: Optional[float]) -> np.ndarray:
    """Which detections `lfe_detector._merge_nearby` keeps, as a mask in input order: in descending score order, ties in input
    order, a detection is dropped when it lies within `radius` of one already kept. `None` keeps everything; a radius of 0 still
    merges detections at the same point, as `_merge_nearby` does. Vectorised, since the radius sweep runs it ~10^6 times."""
    scores = np.asarray(scores)
    keep = np.ones(len(scores), bool)
    if radius is None or len(scores) < 2:
        return keep
    order = np.argsort(-scores, kind="stable")
    points = np.asarray(xy, np.float64).reshape(-1, 2)[order]
    dist = cdist(points, points)
    alive = np.ones(len(order), bool)
    for i in range(len(order)):
        if alive[i]:
            alive[i + 1:] &= dist[i, i + 1:] > radius
    keep[order[~alive]] = False
    return keep


def merged(frames: Frames, radius: Optional[float]) -> Frames:
    out = []
    for s, xy, g in frames:
        k = merge_detections(s, xy, radius)
        out.append((np.asarray(s)[k], np.asarray(xy).reshape(-1, 2)[k], g))
    return out


def drop_duplicate_annotations(gt: np.ndarray) -> np.ndarray:
    """Annotated people `[G, 2]` without FROG's copied labels: a row goes when it equals an earlier row exactly
    (`memory/data-model.md`)."""
    gt = np.asarray(gt).reshape(-1, 2)
    return gt[[j for j in range(len(gt)) if not (j and np.all(gt[:j] == gt[j], axis=1).any())]]


def double_report_anatomy(frames: Frames, candidate_frames: Frames, threshold: float, radius: float = MATCH_RADIUS_M) -> Dict:
    """How many people have two or more detections at or above `threshold` within `radius`, and where the two best came from: a
    stage-2 candidate (at one of `candidate_frames`' positions) or a slot or track reported anywhere else. People with a copied
    label are left out, since a second detection is the only way to score the copy."""
    kinds = {"candidate+candidate": 0, "candidate+slot": 0, "slot+slot": 0}
    people = doubled = 0
    separation = []
    for (s, xy, gt), (_, cand_xy, _) in zip(frames, candidate_frames, strict=True):
        gt = np.asarray(gt, np.float64).reshape(-1, 2)
        solo = np.all(gt[:, None] == gt[None], axis=-1).sum(1) == 1
        people += int(solo.sum())
        keep = np.asarray(s) >= threshold
        if not solo.any() or keep.sum() < 2:
            continue
        kept_xy, kept_s = np.asarray(xy, np.float64).reshape(-1, 2)[keep], np.asarray(s)[keep]
        cand_xy = np.asarray(cand_xy, np.float64).reshape(-1, 2)
        is_candidate = cdist(kept_xy, cand_xy).min(1) < 1e-6 if len(cand_xy) else np.zeros(len(kept_xy), bool)
        for near in cdist(gt[solo], kept_xy) <= radius:
            idx = np.nonzero(near)[0]
            if len(idx) < 2:
                continue
            doubled += 1
            a, b = idx[np.argsort(-kept_s[idx], kind="stable")[:2]]
            kinds[("slot+slot", "candidate+slot", "candidate+candidate")[int(is_candidate[a]) + int(is_candidate[b])]] += 1
            separation.append(float(np.linalg.norm(kept_xy[a] - kept_xy[b])))
    return dict(people=people, doubled=doubled, rate=doubled / people if people else float("nan"), kinds=kinds,
                separation_median_m=float(np.median(separation)) if separation else float("nan"))


# ---------------------------------------------------------------- shared plumbing

def slim(split) -> SimpleNamespace:
    """The fields a replay reads, as plain arrays, picklable without importing torch."""
    d = split.data
    data = SimpleNamespace(**{k: getattr(d, k) for k in ("xy", "score", "valid", "pose", "time", "annotated", "gt_xy", "gt_valid")})
    return SimpleNamespace(data=data, offsets=np.asarray(split.offsets), lengths=np.asarray(split.lengths), dt=np.asarray(split.dt))


def save(path: Path, frames: Frames, index: Sequence[int], **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(dict(frames=frames, index=np.asarray(index, np.int64), meta=meta)))


def load(path: Path) -> Dict:
    return pickle.loads(Path(path).read_bytes())


def subset(frames: Frames, index: Sequence[int], wanted: np.ndarray) -> Frames:
    """The frames at packed indices `wanted`, in that order."""
    at = {int(g): k for k, g in enumerate(index)}
    missing = [int(g) for g in wanted if int(g) not in at]
    if missing:
        raise ValueError(f"{len(missing)} wanted frames were not saved, first {missing[:3]}")
    return [frames[at[int(g)]] for g in wanted]


def frame_period(split) -> float:
    spans = [(split.data.time[o + n - 1] - split.data.time[o], n - 1) for o, n in zip(split.offsets, split.lengths) if n > 1]
    return float(sum(e for e, _ in spans) / sum(f for _, f in spans))


def tracker_factory(config: Dict, period_s: float) -> Callable[[], ByteTracker]:
    """A `ByteTracker` from one E14 configuration, durations in seconds turned into frames at `period_s`."""
    kwargs = dict(two_stage=config["two_stage"], high_thresh=config["high_thresh"], low_thresh=config["low_thresh"],
                  new_track_thresh=config.get("new_track_thresh"), match_radius=config["match_radius"],
                  max_age=max(1, int(round(config["max_age_s"] / period_s))), min_hits=config.get("min_hits", 1),
                  score_mode=config["score_mode"], score_rule=config["score_rule"], score_decay_per_s=config["score_decay_per_s"],
                  coast_frames=int(round(config["coast_s"] / period_s)), unmatched_scale=config["unmatched_scale"])
    return lambda: ByteTracker(**kwargs)


def config_key(config: Dict) -> str:
    return json.dumps(config, sort_keys=True)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------- E14 workers (no torch)

_WORKER: Dict[str, object] = {}


def _worker_init(path: str) -> None:
    state = pickle.loads(Path(path).read_bytes())
    _WORKER.update(split=state["split"], keep=set(state["keep"].tolist()), period=state["period"])


def _worker_ap(config: Dict) -> Tuple[str, float, float]:
    t0 = time.time()
    frames, _ = replay_tracker(_WORKER["split"], tracker_factory(config, _WORKER["period"]), config["output"], _WORKER["keep"])
    return config_key(config), PrecisionRecall(frames).ap(), time.time() - t0


# ---------------------------------------------------------------- merge workers (no torch)

MERGE_RADII = (None, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)


def merge_outputs(arms: Dict[str, List[str]]) -> Dict[str, str]:
    """Every output the merge sweep touches, by name, with the stem of its saved frames."""
    outputs = {"candidates": "candidates", "E14": "track_bytetrack_cbmot"}
    outputs.update({f"{arm}:{run}": f"arm_{arm}_{run}" for arm, runs in arms.items() for run in runs})
    return outputs


def _merge_init(out: str, scored: np.ndarray, blocks: np.ndarray, budget: float) -> None:
    _WORKER.update(out=Path(out), scored=scored, blocks=blocks, budget=budget)


def _merge_frames(stem: str, split_name: str, dedup: bool = False) -> Frames:
    """Saved frames: all of val's (they are its scored stride already), test's scored stride."""
    saved = load(_WORKER["out"] / f"{stem}_{split_name}.pkl")
    frames = saved["frames"] if split_name == "val" else subset(saved["frames"], saved["index"], _WORKER["scored"])
    return [(s, xy, drop_duplicate_annotations(g)) for s, xy, g in frames] if dedup else frames


def _merge_sweep(job: Tuple[str, str]) -> Dict:
    """One output: AP on val at every radius, the best (no merge wins a tie), then test at every radius, with the operating point
    at the candidates' false positives and the double-report anatomy there, unmerged and at the chosen radius."""
    name, stem = job
    t0 = time.time()
    val_ap = {str(r): PrecisionRecall(merged(_merge_frames(stem, "val"), r)).ap() for r in MERGE_RADII}
    best = max(MERGE_RADII, key=lambda r: val_ap[str(r)])
    test, dedup = _merge_frames(stem, "test"), _merge_frames(stem, "test", dedup=True)
    candidates = _merge_frames("candidates", "test")
    result = dict(name=name, stem=stem, val_ap=val_ap, radius=best, test_ap={}, test_ap_dedup={}, matched_fp={}, anatomy={})
    for r in MERGE_RADII:
        frames = merged(test, r)
        pr = PrecisionRecall(frames)
        result["test_ap"][str(r)] = pr.ap()
        if r is None or r == best:
            point = pr.at_fp_budget(_WORKER["budget"], len(test))
            result["matched_fp"][str(r)] = point
            result["anatomy"][str(r)] = double_report_anatomy(frames, candidates, point["threshold"])
            result["test_ap_dedup"][str(r)] = PrecisionRecall(merged(dedup, r)).ap()
    result["seconds"] = time.time() - t0
    return result


def _merge_boot(job: Tuple[str, Tuple[str, Optional[float]], Tuple[str, Optional[float]], bool]) -> Tuple[str, Dict]:
    label, (stem_a, radius_a), (stem_b, radius_b), dedup = job
    a, b = merged(_merge_frames(stem_a, "test", dedup), radius_a), merged(_merge_frames(stem_b, "test", dedup), radius_b)
    return label, paired_bootstrap(a, b, _WORKER["blocks"], RESAMPLES, seed=0)


# ---------------------------------------------------------------- subcommands

def run_rule(args) -> None:
    """E15: tune the rule scorer on val over the full grid, then score its best configuration on val and test once."""
    from takheliped.detectors.takheliped import SlotRules
    from train_takheliped import detect_device, load_split

    device, _ = detect_device()
    tuning_path = args.out / "rule_tuning.json"
    tuning = json.loads(tuning_path.read_text()) if tuning_path.exists() else {}
    val = load_split("val", 0, args.stage2, device)
    keep_val = set(stride_frames(val).tolist())
    for gain in RULE_GAINS:
        for fade in RULE_FADES_S:
            for unmatched in RULE_UNMATCHED:
                for coast in RULE_COAST:
                    config = dict(gain=gain, fade_time_s=fade, unmatched_scale=unmatched, coast_floor=coast)
                    key = config_key(config)
                    if key in tuning:
                        continue
                    t0 = time.time()
                    frames, _ = replay_rule_scorer(val, device, args.streams, SlotRules(gain=gain, fade_time_s=fade), unmatched, coast, keep_val)
                    tuning[key] = PrecisionRecall(frames).ap()
                    tuning_path.write_text(json.dumps(tuning, indent=1))
                    log(f"rule {key}: val AP {tuning[key]:.4f} ({time.time() - t0:.0f} s)")
    best = json.loads(max(tuning, key=tuning.get))
    log(f"rule best on val: {best}, val AP {tuning[config_key(best)]:.4f}")
    rules = SlotRules(gain=best["gain"], fade_time_s=best["fade_time_s"])
    frames, index = replay_rule_scorer(val, device, args.streams, rules, best["unmatched_scale"], best["coast_floor"], keep_val)
    save(args.out / "rule_val.pkl", frames, index, config=best)
    del val
    test = load_split("test", 0, args.stage2, device)
    frames, index = replay_rule_scorer(test, device, args.streams, rules, best["unmatched_scale"], best["coast_floor"])
    save(args.out / "rule_test.pkl", frames, index, config=best)
    log(f"rule test AP (every {SCORED_STRIDE}th annotated frame): {PrecisionRecall(subset(frames, index, stride_frames(test))).ap():.4f}")


def run_track(args) -> None:
    """E14: the fixed rows, then coordinate descent on val in parallel worker processes, then val and test for each row."""
    from train_takheliped import load_split

    tuning_path = args.out / "track_tuning.json"
    tuning = json.loads(tuning_path.read_text()) if tuning_path.exists() else {}
    val = load_split("val", 0, args.stage2, "cpu")
    period = frame_period(val)
    keep_val = stride_frames(val)
    shared = args.out / "track_val_split.pkl"
    shared.write_bytes(pickle.dumps(dict(split=slim(val), keep=keep_val, period=period)))

    def evaluate_all(configs: List[Dict]) -> None:
        todo = [c for c in configs if config_key(c) not in tuning]
        if not todo:
            return
        with ProcessPoolExecutor(max_workers=min(args.workers, len(todo)), initializer=_worker_init, initargs=(str(shared),)) as pool:
            for key, ap, seconds in pool.map(_worker_ap, todo):
                tuning[key] = ap
                tuning_path.write_text(json.dumps(tuning, indent=1))
                log(f"track {key}: val AP {ap:.4f} ({seconds:.0f} s)")

    rows = {"sort_m1": dict(BYTETRACK_PUBLISHED, two_stage=False, high_thresh=0.0, low_thresh=0.0, new_track_thresh=0.0, min_hits=1,
                            max_age_s=period),
            "sort_m3": dict(BYTETRACK_PUBLISHED, two_stage=False, high_thresh=0.0, low_thresh=0.0, new_track_thresh=0.0, min_hits=3,
                            max_age_s=period),
            "bytetrack": dict(BYTETRACK_PUBLISHED)}
    evaluate_all(list(rows.values()))
    best = dict(TRACK_START)
    for sweep in range(TRACK_PASSES):
        for name, values in TRACK_AXES:
            candidates = [dict(best, **{name: v}) for v in values]
            evaluate_all(candidates)
            best = max(candidates, key=lambda c: tuning[config_key(c)])
            log(f"pass {sweep + 1}, {name}: best {best[name]} -> val AP {tuning[config_key(best)]:.4f}")
    rows["bytetrack_cbmot"] = best
    (args.out / "track_rows.json").write_text(json.dumps(rows, indent=1))
    keep_val_set = set(keep_val.tolist())
    for name, config in rows.items():
        frames, index = replay_tracker(val, tracker_factory(config, period), config["output"], keep_val_set)
        save(args.out / f"track_{name}_val.pkl", frames, index, config=config)
    del val
    test = load_split("test", 0, args.stage2, "cpu")
    for name, config in rows.items():
        frames, index = replay_tracker(test, tracker_factory(config, frame_period(test)), config["output"])
        save(args.out / f"track_{name}_test.pkl", frames, index, config=config)
        log(f"track {name} test AP: {PrecisionRecall(subset(frames, index, stride_frames(test))).ap():.4f}")
    shared.unlink(missing_ok=True)


def run_arms(args) -> None:
    """Every memory checkpoint's frames on test (every annotated frame) and val (the scored stride), and the published LFE
    detectors' on the same frames."""
    import torch
    from takheliped.datasets.frog_dataset import FROG_Dataset, frog_laser_angles
    from takheliped.detectors import LFEPeaksDetector, LFEPPNDetector
    from train_takheliped import LFE_MISSING_RETURN_M, detect_device, evaluate, evaluation_segments, load_split, object_memory

    device, _ = detect_device()
    for split_name, stride in (("test", 1), ("val", SCORED_STRIDE)):
        split = load_split(split_name, 0, args.stage2, device)
        candidates_path = args.out / f"candidates_{split_name}.pkl"
        for arm, runs in args.arms.items():
            for run in runs:
                path = args.out / f"arm_{arm}_{run}_{split_name}.pkl"
                checkpoint_path = CHECKPOINTS / run / "step3a_object_memory.best.pth"
                if path.exists() or not checkpoint_path.exists():
                    continue
                t0 = time.time()
                model = object_memory(torch.load(checkpoint_path, map_location=device, weights_only=False), split.data.features.shape[-1]).to(device)
                frames = {}
                evaluate(model, split, device, stride, args.streams, evaluation_segments(split, 0, 0, 0), frames_out=frames, compute_metrics=False)
                order = np.argsort(frames["index"], kind="stable")
                index = [frames["index"][k] for k in order]
                memory_frames = [frames["memory"][k] for k in order]
                ap = PrecisionRecall(subset(memory_frames, index, stride_frames(split))).ap()
                save(path, memory_frames, index, run=run, ap=ap)
                if not candidates_path.exists():
                    save(candidates_path, [frames["candidates"][k] for k in order], index)
                log(f"{arm} {run} {split_name}: memory AP {ap:.4f} on the scored stride ({time.time() - t0:.0f} s)")
        keep = stride_frames(split, 1 if split_name == "test" else SCORED_STRIDE)
        ds = None
        for name, detector in (("LFE-Peaks", lambda: LFEPeaksDetector(peak_height=0.01)), ("LFE-PPN", lambda: LFEPPNDetector(score_thresh=0.01))):
            path = args.out / f"lfe_{name}_{split_name}.pkl"
            if path.exists():
                continue
            if ds is None:
                ds = FROG_Dataset(split=split_name, mode="official", auto_download=False, verbose=False, missing_return_m=LFE_MISSING_RETURN_M)
                angles = frog_laser_angles(720)
            t0 = time.time()
            det = detector()
            recording = recording_of(split, keep)
            frames = []
            for g, s in zip(keep, recording):
                c = np.asarray(det.detect(ds.scans[s][int(g - split.offsets[s])], angles), np.float64).reshape(-1, 3)
                frames.append((c[:, 0], c[:, 1:], split.data.gt_xy[g][split.data.gt_valid[g]]))
            save(path, frames, keep, detector=name)
            log(f"{name} {split_name}: {len(frames)} frames ({time.time() - t0:.0f} s)")
        del split


def report_focus(table, best_classical) -> List[str]:
    """The methods the per-person tables compare: the candidates, the strongest classical pipeline, and every arm.

    An arm is any key carrying its `label:run` separator, whatever the label. While this read `startswith("E13:")` it
    dropped the frozen arm from the strata and the first-detection latency altogether.
    """
    return [m for m in ("candidates", best_classical) if m] + [m for m in table if ":" in m]


def run_report(args) -> None:
    """The baseline tables: AP with intervals, GOSPA, strata, latency."""
    from takheliped.datasets.frog_dataset import FROG_Dataset, frog_laser_angles
    from motion_analysis import build_tracks, to_world
    from train_takheliped import candidate_targets, load_split

    test = load_split("test", 0, args.stage2, "cpu")
    scored = stride_frames(test)
    report: Dict[str, object] = {}

    methods: Dict[str, Dict[str, Path]] = {"candidates": dict(test=args.out / "candidates_test.pkl", val=args.out / "candidates_val.pkl")}
    for name in ("LFE-Peaks", "LFE-PPN"):
        methods[name] = dict(test=args.out / f"lfe_{name}_test.pkl", val=args.out / f"lfe_{name}_val.pkl")
    for name in ("sort_m1", "sort_m3", "bytetrack", "bytetrack_cbmot"):
        methods[name] = dict(test=args.out / f"track_{name}_test.pkl", val=args.out / f"track_{name}_val.pkl")
    methods["rule_cbmot"] = dict(test=args.out / "rule_test.pkl", val=args.out / "rule_val.pkl")
    for arm, runs in args.arms.items():
        for run in runs:
            methods[f"{arm}:{run}"] = dict(test=args.out / f"arm_{arm}_{run}_test.pkl", val=args.out / f"arm_{arm}_{run}_val.pkl")
    methods = {m: p for m, p in methods.items() if p["test"].exists()}
    saved = {m: load(p["test"]) for m, p in methods.items()}
    frames5 = {m: subset(s["frames"], s["index"], scored) for m, s in saved.items()}
    n5 = len(scored)

    # AP, operating points and recall at the candidates' false-positive rate
    curves = {m: PrecisionRecall(f) for m, f in frames5.items()}
    budget = curves["candidates"].at_threshold(OPERATING_THRESHOLD, n5)["fp_per_frame"]
    table = {}
    for m, pr in curves.items():
        table[m] = dict(ap=pr.ap(), **{f"at_0.3_{k}": v for k, v in pr.at_threshold(OPERATING_THRESHOLD, n5).items()},
                        **{f"matched_fp_{k}": v for k, v in pr.at_fp_budget(budget, n5).items()})
    report["ap"] = table
    for arm in args.arms:
        seeds = [table[m]["ap"] for m in table if m.startswith(f"{arm}:")]
        if seeds:
            report.setdefault("arm_means", {})[arm] = dict(seeds=seeds, mean=float(np.mean(seeds)), sd=float(np.std(seeds, ddof=1)) if len(seeds) > 1 else float("nan"))
    log("AP, every 5th annotated test frame:")
    for m, row in table.items():
        log(f"  {m:28s} AP {row['ap']:.4f}  recall at matched FP {row['matched_fp_recall']:.4f}  FP/frame at 0.3 {row['at_0.3_fp_per_frame']:.3f}")

    # paired block bootstrap against the strongest classical pipeline and against the candidates
    classical = [m for m in ("bytetrack_cbmot", "rule_cbmot", "bytetrack", "sort_m1") if m in table]
    best_classical = max(classical, key=lambda m: table[m]["ap"]) if classical else None
    blocks = block_ids(recording_of(test, scored), test.data.time[scored])
    pairs = [(m, "candidates") for m in table if m != "candidates" and not m.startswith("LFE")]
    if best_classical:
        pairs += [(m, best_classical) for m in table if ":" in m]
    report["bootstrap"] = {}
    for a, b in pairs:
        r = paired_bootstrap(frames5[a], frames5[b], blocks, args.resamples, seed=0)
        report["bootstrap"][f"{a} - {b}"] = r
        log(f"  {a} - {b}: {100 * r['difference']:+.2f} pp [{100 * r['low']:+.2f}, {100 * r['high']:+.2f}]")
    report["best_classical"] = best_classical

    # E16: GOSPA, each method at the threshold that minimises it on val, and at 0.3
    report["gospa"] = {}
    for m, paths in methods.items():
        if not paths["val"].exists():
            continue
        v = load(paths["val"])
        chosen = min(THRESHOLDS, key=lambda th: gospa_frames(v["frames"], th)["gospa"])
        report["gospa"][m] = dict(threshold=chosen, at_val_best=gospa_frames(frames5[m], chosen), at_0_3=gospa_frames(frames5[m], OPERATING_THRESHOLD))
        g = report["gospa"][m]["at_val_best"]
        log(f"  GOSPA {m:28s} at {chosen:.2f}: {g['gospa']:.3f} (localisation {g['localisation']:.3f}, missed {g['missed']:.3f}, false {g['false']:.3f})")

    # E20 and E21: per annotated person, on every annotated test frame
    ds = FROG_Dataset(split="test", mode="official", auto_download=False, verbose=False, missing_return_m=None)
    angles = frog_laser_angles(720)
    d = test.data
    every = stride_frames(test, 1)
    thresholds = {m: table[m]["matched_fp_threshold"] for m in table}
    focus = report_focus(table, best_classical)
    covered = {m: {} for m in focus}
    attributes = {}
    for m in focus:
        for g, (scores, xy, gt) in zip(saved[m]["index"], saved[m]["frames"]):
            if int(g) not in covered[m] and len(gt):
                keep = np.asarray(scores) >= thresholds[m]
                covered[m][int(g)] = candidate_targets(np.asarray(xy)[keep], np.ones(int(keep.sum()), bool), np.asarray(gt, np.float64), MATCH_RADIUS_M)[1]
    tracks_by_point = {}
    for s, (offset, n) in enumerate(zip(test.offsets, test.lengths)):
        frames_here = [int(g) for g in every if offset <= g < offset + n]
        world = [to_world(d.gt_xy[g][d.gt_valid[g]].astype(np.float64), d.pose[g]) for g in frames_here]
        for k, track in enumerate(build_tracks(world, scan_index=frames_here, gate=0.6)):
            fs = np.array(sorted(track))
            ts, ws = d.time[fs].astype(np.float64), np.array([track[f] for f in fs])
            lo, hi = np.searchsorted(ts, ts - 0.5), np.searchsorted(ts, ts + 0.5, side="right") - 1
            span = ts[hi] - ts[lo]
            with np.errstate(invalid="ignore", divide="ignore"):
                speeds = np.where(span > 0, np.linalg.norm(ws[hi] - ws[lo], axis=1) / span, np.nan)
            for f, w, age, speed in zip(fs, ws, ts - ts[0], speeds):
                people = to_world(d.gt_xy[f][d.gt_valid[f]].astype(np.float64), d.pose[f])
                j = int(np.argmin(np.linalg.norm(people - w[None], axis=1)))
                tracks_by_point[(int(f), j)] = (s, k, float(age), float(speed))
    for g, r in zip(every, recording_of(test, every)):
        g = int(g)
        people = d.gt_xy[g][d.gt_valid[g]].astype(np.float64)
        scan = ds.scans[r][g - int(test.offsets[r])]
        for j, p in enumerate(people):
            s, k, age, speed = tracks_by_point.get((g, j), (None, None, float("nan"), float("nan")))
            attributes[(g, j)] = dict(range=float(np.hypot(*p)), occluded=occlusion_share(scan, angles, p), age=age, speed=speed, track=(s, k))

    def stratum(a: Dict) -> Dict[str, str]:
        rng = a["range"]
        return dict(range="0-2" if rng < 2 else "2-4" if rng < 4 else "4-6" if rng < 6 else "6-8" if rng < 8 else "8+",
                    occlusion="visible" if a["occluded"] == 0 else "partial" if a["occluded"] <= 0.5 else "occluded",
                    motion="unknown" if not np.isfinite(a["speed"]) else "still" if a["speed"] < 0.2 else "slow" if a["speed"] < 0.5 else "moving",
                    age="unknown" if not np.isfinite(a["age"]) else "<0.5 s" if a["age"] < 0.5 else "0.5-2 s" if a["age"] < 2 else "2-10 s" if a["age"] < 10 else ">10 s")

    strata = {}
    for (g, j), a in attributes.items():
        for axis, value in stratum(a).items():
            cell = strata.setdefault(axis, {}).setdefault(value, {m: [0, 0] for m in focus})
            for m in focus:
                hit = covered[m].get(g)
                cell[m][0] += int(hit is not None and bool(hit[j]))
                cell[m][1] += 1
    report["strata"] = {axis: {value: {m: dict(recall=h / n if n else float("nan"), people=n) for m, (h, n) in cells.items()}
                               for value, cells in values.items()} for axis, values in strata.items()}
    for axis, values in report["strata"].items():
        log(f"  strata by {axis}:")
        for value, cells in values.items():
            log("    " + f"{value:10s} " + "  ".join(f"{m.split(':')[-1]} {c['recall']:.3f}" for m, c in cells.items()) + f"  (n={next(iter(cells.values()))['people']})")

    by_track = defaultdict(list)
    for (g, j), a in attributes.items():
        if a["track"][0] is not None:
            by_track[a["track"]].append((g, j))
    period = frame_period(test)
    report["latency"] = {}
    for m in focus:
        delays = first_detection_delays([[(g, bool(covered[m].get(g) is not None and covered[m][g][j])) for g, j in points] for points in by_track.values()])
        found = [x for x in delays if x is not None]
        report["latency"][m] = dict(trajectories=len(delays), never=sum(x is None for x in delays), median_frames=float(np.median(found)) if found else float("nan"),
                                    median_s=float(np.median(found)) * period if found else float("nan"), first_frame=float(np.mean([x == 0 for x in found])) if found else float("nan"),
                                    within_0_5_s=float(np.mean([x * period <= 0.5 for x in found])) if found else float("nan"))
        r = report["latency"][m]
        log(f"  latency {m:28s} median {r['median_s']:.3f} s, on the first frame {r['first_frame']:.3f}, within 0.5 s {r['within_0_5_s']:.3f}, never {r['never']}/{r['trajectories']}")

    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    log(f"wrote {args.out / 'report.json'}")


def run_latency(args) -> None:
    """E22: stage 2 with candidate decoding, then the memory, one scan at a time on one stream, timed stage by stage."""
    import torch
    from takheliped.datasets.frog_dataset import FROG_Dataset, frog_laser_angles
    from takheliped.detectors.takheliped import free_range_profile
    from train_takheliped import (MAX_RANGE_M, SECTOR_BEAMS, TOP_P, beam_frame_votes, calibration_network, decode_candidates,
                                  detect_device, laser_angles, object_memory)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cpu") if args.device == "cpu" else detect_device()[0]
    synchronise = str(device).startswith("cuda")
    angles = frog_laser_angles(720)
    net = calibration_network(torch.load(args.stage2, map_location="cpu", weights_only=False), angles).to(device).eval()
    checkpoint = torch.load(args.memory, map_location="cpu", weights_only=False)
    ds = FROG_Dataset(split="test", mode="official", auto_download=False, verbose=False, missing_return_m=None)
    s = int(np.argmax([len(x) for x in ds.scans]))
    scans, poses = ds.scans[s][:args.frames + 1], np.asarray(ds.odoms[s]["xya"])[:args.frames + 1]
    slots, memory = None, None
    timings = {"stage2": [], "decode": [], "memory": []}

    def tick():
        if synchronise:
            torch.cuda.synchronize()
        return time.perf_counter()

    with torch.no_grad():
        for i, (scan, pose) in enumerate(zip(scans, poses)):
            t0 = tick()
            logit, votes, features = net(torch.from_numpy(np.asarray(scan, np.float32)[None]).to(device))
            t1 = tick()
            offsets = beam_frame_votes(votes[0].cpu().numpy(), angles, net.vote_frame)
            candidates, weights = decode_candidates(np.asarray(scan), angles, torch.sigmoid(logit)[0].cpu().numpy(), offsets, TOP_P, net.max_range_m)
            pooled = weights @ features[0].T.cpu().numpy()
            if memory is None:
                memory = object_memory(checkpoint, pooled.shape[1]).to(device).eval()
                slots = memory.initial_state(1, device)
            k = len(candidates)
            xy, score, valid, feat = np.zeros((1, TOP_P, 2), np.float32), np.zeros((1, TOP_P), np.float32), np.zeros((1, TOP_P), bool), np.zeros((1, TOP_P, pooled.shape[1]), np.float32)
            xy[0, :k], score[0, :k], valid[0, :k], feat[0, :k] = candidates[:, 1:], candidates[:, 0], True, pooled
            t2 = tick()
            free = free_range_profile(torch.from_numpy(np.asarray(scan, np.float32)[None]), 720 // SECTOR_BEAMS, MAX_RANGE_M).to(device)
            out = memory.step(torch.from_numpy(xy).to(device), torch.from_numpy(score).to(device), torch.from_numpy(valid).to(device),
                              torch.from_numpy(np.asarray(pose, np.float32)[None]).to(device), torch.full((1,), 0.025, device=device), slots,
                              cand_features=torch.from_numpy(feat).to(device), free_range=free, angles=laser_angles(device))
            slots = out.slots
            t3 = tick()
            if i:
                timings["stage2"].append(1000 * (t1 - t0))
                timings["decode"].append(1000 * (t2 - t1))
                timings["memory"].append(1000 * (t3 - t2))
    total = np.array(timings["stage2"]) + np.array(timings["decode"]) + np.array(timings["memory"])
    result = {k: dict(median=float(np.median(v)), p99=float(np.percentile(v, 99)), mean=float(np.mean(v))) for k, v in timings.items()}
    result["total"] = dict(median=float(np.median(total)), p99=float(np.percentile(total, 99)), mean=float(np.mean(total)))
    result.update(device=str(device), threads=torch.get_num_threads(), frames=len(total), memory_checkpoint=str(args.memory), stage2_checkpoint=str(args.stage2))
    path = args.out / f"latency_{'cpu' if args.device == 'cpu' else 'gpu'}{'_t' + str(args.threads) if args.threads else ''}.json"
    path.write_text(json.dumps(result, indent=1))
    log(f"{device}, {torch.get_num_threads()} threads: " + "  ".join(f"{k} {v['median']:.2f} ms (p99 {v['p99']:.2f})" for k, v in result.items() if isinstance(v, dict)))


def run_merge(args) -> None:
    """`lfe_detector._merge_nearby`'s merge on every output's saved frames, the radius chosen per output on val and test
    scored once; block bootstraps against the unmerged output and between outputs, on the official labels and without the copies."""
    from train_takheliped import load_split

    test = load_split("test", 0, args.stage2, "cpu")
    scored = stride_frames(test)
    blocks = block_ids(recording_of(test, scored), test.data.time[scored])
    del test
    candidates = load(args.out / "candidates_test.pkl")
    budget = PrecisionRecall(subset(candidates["frames"], candidates["index"], scored)).at_threshold(OPERATING_THRESHOLD, len(scored))["fp_per_frame"]
    del candidates
    log(f"matched false-positive budget: {budget:.3f} per frame (the candidates at {OPERATING_THRESHOLD})")
    outputs = {name: stem for name, stem in merge_outputs(args.arms).items() if (args.out / f"{stem}_test.pkl").exists()}
    result: Dict[str, object] = dict(budget=budget, radii=[str(r) for r in MERGE_RADII], outputs={}, bootstrap={})
    with ProcessPoolExecutor(max_workers=min(args.workers, len(outputs)), initializer=_merge_init,
                             initargs=(str(args.out), scored, blocks, budget)) as pool:
        for r in pool.map(_merge_sweep, list(outputs.items())):
            result["outputs"][r["name"]] = r
            best, a = str(r["radius"]), r["anatomy"]
            log(f"{r['name']:28s} radius {best:>4s} (val {100 * r['val_ap'][best]:.2f} against {100 * r['val_ap']['None']:.2f})  "
                f"test {100 * r['test_ap']['None']:.2f} -> {100 * r['test_ap'][best]:.2f}  "
                f"doubled {100 * a['None']['rate']:.1f}% -> {100 * a[best]['rate']:.1f}%  {a['None']['kinds']}  ({r['seconds']:.0f} s)")
        radius = {name: r["radius"] for name, r in result["outputs"].items()}
        learned = [name for name in outputs if ":" in name]
        jobs = []
        for dedup, tag in ((False, ""), (True, ", duplicates dropped")):
            jobs += [(f"{n} merged - unmerged{tag}", (outputs[n], radius[n]), (outputs[n], None), dedup) for n in outputs]
            jobs += [(f"{n} - E14, both merged{tag}", (outputs[n], radius[n]), (outputs["E14"], radius["E14"]), dedup) for n in learned]
            jobs.append((f"E14 - candidates, both merged{tag}", (outputs["E14"], radius["E14"]), (outputs["candidates"], radius["candidates"]), dedup))
        for label, r in pool.map(_merge_boot, jobs):
            result["bootstrap"][label] = r
            log(f"  {label}: {100 * r['difference']:+.2f} pp [{100 * r['low']:+.2f}, {100 * r['high']:+.2f}]")
    (args.out / "merge.json").write_text(json.dumps(result, indent=1, default=float))
    log(f"wrote {args.out / 'merge.json'}")


def run_summary(args) -> None:
    """Every `<out>/s*/report.json` lined up by arm (`summarise`)."""
    paths = sorted(args.out.glob("s*/report.json"))
    if not paths:
        raise FileNotFoundError(f"no {args.out}/s*/report.json to summarise")
    summary = dict(reports=[str(p) for p in paths], **summarise([json.loads(p.read_text()) for p in paths]))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    log(f"{len(paths)} reports; AP over seeds, every 5th annotated test frame:")
    for m, s in summary["ap"].items():
        log(f"  {m:20s} {100 * s['mean']:.2f} (sd {100 * s['sd']:.2f})  " + " / ".join(f"{100 * v:.2f}" for v in s["seeds"]))
    for pair, b in summary["bootstrap"].items():
        log(f"  {pair}: " + "  ".join(f"{100 * d:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]" for d, lo, hi in zip(b["difference"], b["low"], b["high"]))
            + f"  ({b['above_zero']} of {len(b['low'])} above zero)")
    log(f"wrote {args.out / 'summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("rule", "track", "arms", "report", "latency", "merge", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--stage2", type=Path, default=STAGE2)
        p.add_argument("--out", type=Path, default=OUT_DIR)
        if name in ("arms", "report", "merge"):
            p.add_argument("--arms", nargs="+", default=None, metavar="ARM:RUN",
                           help="memory runs to replay or read, e.g. E13:a60_e13_static_s0; unset is the default set")
        if name in ("rule", "arms"):
            p.add_argument("--streams", type=int, default=64)
        if name == "track":
            p.add_argument("--workers", type=int, default=8)
        if name == "merge":
            p.add_argument("--workers", type=int, default=6)
        if name == "report":
            p.add_argument("--resamples", type=int, default=RESAMPLES)
        if name == "latency":
            p.add_argument("--memory", type=Path, default=CHECKPOINTS / ARMS["E13"][0] / "step3a_object_memory.best.pth")
            p.add_argument("--device", choices=("auto", "cpu"), default="auto")
            p.add_argument("--frames", type=int, default=2000)
            p.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if hasattr(args, "arms"):
        args.arms = parse_arms(args.arms) if args.arms else ARMS
    args.out.mkdir(parents=True, exist_ok=True)
    log(f"tracking_baselines {args.command}")
    dict(rule=run_rule, track=run_track, arms=run_arms, report=run_report, latency=run_latency, merge=run_merge, summary=run_summary)[args.command](args)


if __name__ == "__main__":
    try:
        main()
        code = 0
    except SystemExit as stop:                 # --help and argument errors keep their own exit code
        code = stop.code if isinstance(stop.code, int) else 0
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)                          # a finished ROCm process can hang past a normal exit (memory/gotchas.md)
