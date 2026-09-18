#!/usr/bin/env python3
"""Train the three-horizon detector (`docs/PROPOSAL.md` §8, §10) with the training budgets of `TODO.md` A43.

`step1` -- stage 3 alone on saved LFE-Peaks candidates: detections down to a 0.01 score, the best 64 per frame, on every frame
of FROG `official` train, val and test, cached once. Recordings play as parallel streams, trained in chunks with truncated
backpropagation over a chunk-length curriculum, selected on val AP over warmed-up windows, and scored on the whole test
recording against the candidates alone: AP, false positives and negatives per frame at 0.3, and the exchange rate.

`step2` -- stages 1-2, training phase A: 35-frame clips ending at annotated frames, DROW head loss, selected on val AP, and
scored on every annotated test frame with clip histories in order and shuffled.

Usage (from utils/)
-----
    python train_three_horizon.py step1
    python train_three_horizon.py step2
    python train_three_horizon.py step3a --stage2 ../checkpoints_three_horizon/step2_calibration.best.pth
    python train_three_horizon.py step4 --stage2 ../checkpoints_three_horizon/step2_calibration.best.pth --memory ../checkpoints_three_horizon/step3a_object_memory.best.pth --joint ../checkpoints_three_horizon/step3b_joint.best.pth
    python train_three_horizon.py step3b --stage2 ../checkpoints_three_horizon/step2_calibration.best.pth --memory ../checkpoints_three_horizon/step3a_object_memory.best.pth
    python train_three_horizon.py step1 --chunk-s 1 --max-epochs 1 --min-epochs 0 --limit-recordings 3 --val-windows 4   # smoke run
    python train_three_horizon.py step2 --max-epochs 0.02 --eval-every 0.01 --min-epochs 0 --val-frames 64 --test-stride 500   # smoke run
"""
import argparse
import ctypes
import json
import math
import os
import pickle
import sys
import time
import traceback
from ast import literal_eval
from dataclasses import dataclass, fields
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import binary_cross_entropy_with_logits

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.datasets.frog_dataset import FROG_Dataset, frog_laser_angles
from follow_the_drow.detectors import LFEPeaksDetector
from follow_the_drow.detectors.three_horizon import (
    MAX_RANGE_M, CalibrationNetwork, CalibrationState, MemoryOutput, ObjectMemory, SlotRules, Slots, TemporalCache, load_calibration_weights, render_prior, sanitize_ranges,
    world_to_sensor,
)
from follow_the_drow.utils.tracking import SortTracker, odom_xya_delta_to_tracker_frame
from follow_the_drow.utils.drow_utils import _prec_rec_2d, _win2global, project_cartesian_from_polar, votes_to_detections
from train import _safe_auc, compute_loss, detect_device, make_targets

VOTE_DECODING = dict(blur_sigma=2.0, blur_win=11, bin_size=0.1, vote_collect_radius=0.5, min_thresh=1e-3)
MATCH_RADIUS_M = 0.5
CANDIDATE_FLOOR = 0.01
TOP_P = 64
OPERATING_THRESHOLD = 0.3
MIN_REPORTED_SCORE = 1e-3
FOCAL_GAMMA = 2.0
FRAME_PERIOD_S = 1.0 / 26.2
LFE_MISSING_RETURN_M = 10.0     # the FROG loader's legacy clamp, kept for LFE-Peaks' candidates only
MIN_DT_S = 1e-3
MAX_DT_S = 1.0
GRAD_CLIP = 1.0
EPOCH_TOLERANCE = 0.01
TEST_STRIDES = (1, 5)
CACHE_DIR = _HERE / ".three_horizon_cache"
SPLITS = ("train", "val", "test")
PRIOR_CHANNELS = 2
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def candidate_targets(cand_xy: np.ndarray, cand_valid: np.ndarray, gt_xy: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """One-to-one gated Hungarian match of valid candidates to annotated people within `radius`:
    (which candidates are people, which people a candidate covers)."""
    target = np.zeros(len(cand_xy), dtype=bool)
    covered = np.zeros(len(gt_xy), dtype=bool)
    idx = np.nonzero(cand_valid)[0]
    if len(idx) and len(gt_xy):
        cost = np.linalg.norm(cand_xy[idx, None, :] - gt_xy[None, :, :], axis=-1)
        gated = np.where(cost > radius, 1e6, cost)
        for i, j in zip(*linear_sum_assignment(gated)):
            if gated[i, j] < 1e6:
                target[idx[i]] = covered[j] = True
    return target, covered


@dataclass
class SequenceCandidates:
    """One recording, per frame: candidates `[T, P]` (detection-frame xy, score, validity, person target), robot pose `[T, 3]`,
    timestamp, whether the frame is annotated, and annotated people `[T, G]` (xy, validity, covered by a candidate)."""
    xy: np.ndarray
    score: np.ndarray
    valid: np.ndarray
    target: np.ndarray
    pose: np.ndarray
    time: np.ndarray
    annotated: np.ndarray
    gt_xy: np.ndarray
    gt_valid: np.ndarray
    gt_covered: np.ndarray
    features: np.ndarray


def build_candidates(ds, candidates_of, top_p: int, radius: float, recordings: int = 0) -> List[SequenceCandidates]:
    """Candidates for every recording of `ds`: `candidates_of(s)` gives recording `s`'s per-frame candidates `(score, x, y)` `[K, 3]`
    with the detector's features `[K, C]`; the `top_p` best per frame are kept, with annotations attached."""
    out = []
    for s in range(min(recordings, len(ds.scans)) if recordings else len(ds.scans)):
        n = len(ds.scans[s])
        annotated = np.zeros(n, dtype=bool)
        people = [np.empty((0, 2))] * n
        for d in range(len(ds.det_id[s])):
            i = int(ds.idet2iscan[s][d])
            annotated[i] = True
            people[i] = np.array([project_cartesian_from_polar(r, phi) for r, phi in ds.det_wp[s][d]], dtype=np.float64).reshape(-1, 2)
        g = max(1, max(len(p) for p in people))
        frames = candidates_of(s)
        seq = SequenceCandidates(
            xy=np.zeros((n, top_p, 2), np.float32), score=np.zeros((n, top_p), np.float32), valid=np.zeros((n, top_p), bool),
            target=np.zeros((n, top_p), bool), pose=np.asarray(ds.odoms[s]["xya"], np.float32), time=np.asarray(ds.scan_time[s], np.float64),
            annotated=annotated, gt_xy=np.zeros((n, g, 2), np.float32), gt_valid=np.zeros((n, g), bool), gt_covered=np.zeros((n, g), bool),
            features=np.zeros((n, top_p, frames[0][1].shape[1] if frames else 0), np.float16),
        )
        for i, (c, f) in enumerate(frames):
            order = np.argsort(-c[:, 0], kind="stable")[:top_p]
            c, f = c[order], f[order]
            seq.xy[i, :len(c)], seq.score[i, :len(c)], seq.valid[i, :len(c)], seq.features[i, :len(c)] = c[:, 1:], c[:, 0], True, f
            m = len(people[i])
            seq.gt_xy[i, :m], seq.gt_valid[i, :m] = people[i], True
            if annotated[i]:
                seq.target[i], seq.gt_covered[i, :m] = candidate_targets(seq.xy[i], seq.valid[i], people[i], radius)
        out.append(seq)
    return out


@dataclass
class Split:
    """Every recording of a split packed end to end: `SequenceCandidates` fields concatenated over frames, people padded to
    one width, with each recording's first frame offset and length."""
    data: SequenceCandidates
    offsets: np.ndarray
    lengths: np.ndarray


def pack(seqs: Sequence[SequenceCandidates]) -> Split:
    g = max(s.gt_xy.shape[1] for s in seqs)
    widened = []
    for s in seqs:
        pad = g - s.gt_xy.shape[1]
        widened.append(SequenceCandidates(
            s.xy, s.score, s.valid, s.target, s.pose, s.time, s.annotated,
            np.pad(s.gt_xy, ((0, 0), (0, pad), (0, 0))), np.pad(s.gt_valid, ((0, 0), (0, pad))), np.pad(s.gt_covered, ((0, 0), (0, pad))), s.features,
        ))
    data = SequenceCandidates(**{f.name: np.concatenate([getattr(s, f.name) for s in widened]) for f in fields(SequenceCandidates)})
    lengths = np.array([len(s.time) for s in seqs])
    return Split(data, np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths)


class StreamSampler:
    """Streams through whole recordings for truncated backpropagation. Each of `batch` streams plays a recording from its first
    frame, then takes the next one from a shared shuffled queue, so every recording is played once per pass."""

    def __init__(self, lengths: Sequence[int], batch: int, rng: np.random.Generator):
        self.lengths, self.rng = list(lengths), rng
        self.queue: List[int] = []
        self.current = [self._next_recording() for _ in range(batch)]
        self.position = [0] * batch

    def _next_recording(self) -> int:
        if not self.queue:
            self.queue = [int(i) for i in self.rng.permutation(len(self.lengths))]
        return self.queue.pop()

    def next_chunk(self, frames: int, reset_prob: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`[batch, frames]` recording index, frame index, and whether memory starts empty at that frame: at every recording's
        first frame, and at a chunk's first frame with probability `reset_prob`."""
        b = len(self.current)
        seq, frame, reset = np.zeros((b, frames), np.int64), np.zeros((b, frames), np.int64), np.zeros((b, frames), bool)
        for s in range(b):
            chunk_reset = self.rng.random() < reset_prob
            for t in range(frames):
                if self.position[s] >= self.lengths[self.current[s]]:
                    self.current[s], self.position[s] = self._next_recording(), 0
                seq[s, t], frame[s, t] = self.current[s], self.position[s]
                reset[s, t] = self.position[s] == 0 or (t == 0 and chunk_reset)
                self.position[s] += 1
        return seq, frame, reset


class EarlyStopping:
    """Stop once `patience_evals` evaluations pass without a new best, never before `min_epochs`, always at `max_epochs` or
    `max_hours`."""

    def __init__(self, patience_evals: int, min_epochs: float, max_epochs: float, max_hours: float):
        self.patience_evals, self.min_epochs, self.max_epochs, self.max_hours = patience_evals, min_epochs, max_epochs, max_hours
        self.best = -math.inf
        self.since_best = 0

    def update(self, metric: float) -> bool:
        """Record one evaluation; True if it is a new best."""
        if metric > self.best:
            self.best, self.since_best = metric, 0
            return True
        self.since_best += 1
        return False

    def should_stop(self, epoch: float, hours: float) -> bool:
        return epoch >= self.max_epochs or hours >= self.max_hours or (epoch >= self.min_epochs and self.since_best >= self.patience_evals)


class BestCheckpoint:
    """The best checkpoint of a whole run, whatever epoch or chunk-length stage produced it: `offer` saves only a strictly better,
    defined metric, so an optimizer restart or a worse later stage can never overwrite a better model."""

    def __init__(self, path: Path):
        self.path = path
        self.best = -math.inf
        self.saved = False

    def offer(self, metric: float, checkpoint: dict) -> bool:
        if not metric > self.best:
            return False
        self.best = metric
        torch.save(checkpoint, self.path)
        self.saved = True
        return True


class PeriodicCheckpoint:
    """Checkpoints kept every `epochs` of training whatever they score, alongside `BestCheckpoint`'s single best, so test AP can be
    measured against training duration instead of only at the run's best (`TODO.md` A50 phase 1 item 4). `epochs` of 0 keeps none.
    An evaluation landing a little early still counts, within `EPOCH_TOLERANCE` of the interval, because iterations rarely divide an
    epoch exactly -- step 2's land at 0.2499, 0.4998 and so on."""

    def __init__(self, directory: Path, stem: str, epochs: float):
        self.directory, self.stem, self.epochs = directory, stem, epochs
        self.due = epochs
        self.saved: List[Path] = []

    def offer(self, epoch: float, checkpoint: dict) -> Optional[Path]:
        """Save `checkpoint` when this evaluation reaches the next multiple of `epochs`; the path written, or None. Intervals that
        pass together save once, not once each."""
        if not self.epochs or epoch < self.due - self.epochs * EPOCH_TOLERANCE:
            return None
        path = self.directory / f"{self.stem}.epoch{epoch:.2f}.pth"
        torch.save(checkpoint, path)
        self.saved.append(path)
        self.due = (math.floor(epoch / self.epochs + EPOCH_TOLERANCE) + 1) * self.epochs
        return path


def plateau_schedule(optimizer: torch.optim.Optimizer, factor: float, patience_evals: int, min_lr: float) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Cut the learning rate by `factor` once `patience_evals` evaluations in a row fail to improve val AP, never below `min_lr`;
    step it with each evaluation's AP."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=factor, patience=patience_evals, min_lr=min_lr)


def coast_targets(slot_xy: torch.Tensor, coasting: torch.Tensor, gt_xy: torch.Tensor, gt_missed: torch.Tensor, radius: float) -> torch.Tensor:
    """Which coasting slots `[B, K]` should report a person: mutual nearest pairs with annotated people no candidate covers,
    within `radius`."""
    if gt_xy.shape[1] == 0:
        return torch.zeros_like(coasting)
    dist = torch.cdist(slot_xy, gt_xy).masked_fill(~(coasting[:, :, None] & gt_missed[:, None, :]), float("inf"))
    nearest_gt = dist.argmin(dim=2)
    mutual = dist.argmin(dim=1).gather(1, nearest_gt) == torch.arange(slot_xy.shape[1], device=slot_xy.device)[None]
    return coasting & mutual & (dist.gather(2, nearest_gt[..., None])[..., 0] <= radius)


def exchange_rate(fp_base: float, fn_base: float, fp_new: float, fn_new: float) -> float:
    """False positives removed per false negative added, against a baseline (`docs/PAPER.md`); unbounded if none are added."""
    added = fn_new - fn_base
    return math.inf if added <= 0 else (fp_base - fp_new) / added


def candidate_pooling_weights(det_xy: np.ndarray, vote_xy: np.ndarray, prob: np.ndarray, radius: float) -> np.ndarray:
    """Weights `[D, N]` that turn per-beam features `[N, C]` into each detection's features: the probability-weighted mean over the
    beams whose votes `[N, 2]` land within `radius` of it; a zero row where no beam with a non-zero probability does."""
    weight = (np.linalg.norm(det_xy[:, None, :] - vote_xy[None, :, :], axis=-1) <= radius) * prob[None, :]
    total = weight.sum(axis=1, keepdims=True)
    return weight / np.where(total > 0, total, 1.0)


def voting_beams(scan: np.ndarray, prob: np.ndarray, max_range_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """A frame's ranges as the network read them, and its per-beam probabilities with every beam at the range limit silenced: with
    nothing within range there is no surface for a person to stand on (`TODO.md` A50; +0.28 to +0.39 pp test AP on all six
    seed-matched checkpoints, 2026-09-17)."""
    anchored = anchor_ranges(scan, max_range_m)
    return anchored, np.where(anchored < max_range_m, prob, 0.0)


def decode_candidates(scan: np.ndarray, angles: np.ndarray, prob: np.ndarray, votes: np.ndarray, top_p: int, max_range_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """One frame's best `top_p` candidates `(score, x, y)` `[K, 3]` from raw ranges `[N]`, per-beam probabilities and votes `[N, 2]`,
    through DROW's vote grid, with each candidate's pooling weights over the beams `[K, N]`; beams at the range limit cast no votes."""
    clean, prob = voting_beams(scan, prob, max_range_m)
    candidates = np.array(decode_votes(clean, angles, prob, votes), dtype=np.float64).reshape(-1, 3)
    candidates = candidates[np.argsort(-candidates[:, 0], kind="stable")[:top_p]]
    r, phi = _win2global(clean, angles, votes[:, 0], votes[:, 1])
    vote_xy = np.stack([r * -np.sin(phi), r * np.cos(phi)], axis=1)
    return candidates, candidate_pooling_weights(candidates[:, 1:], vote_xy, prob, VOTE_DECODING["vote_collect_radius"])


def clip_length(coarse_lags: Sequence[int]) -> int:
    """Frames a training clip must hold so that no temporal tap reads before its start: the coarse convolution's oldest tap and the
    current frame. A shorter clip silently repeats its first frame in training and in clip-based scoring while streaming inference
    reads the real frames (`memory/gotchas.md`)."""
    return max(coarse_lags) + 1


def calibration_network(checkpoint: dict, angles: np.ndarray, **overrides) -> CalibrationNetwork:
    """The checkpoint's weights in a network shaped the way they were trained.

    The coarse lag list changes a layer's input width and `max_range_m` changes what the input means, so both are read from the
    checkpoint's `args` (stored as strings); a checkpoint written before a flag existed falls back to the constructor's own default
    rather than to a second copy of it here. `overrides` (e.g. `prior_channels`) go straight to the constructor. Weights load through
    `load_calibration_weights`, which folds checkpoints from before the fine convolution's removal.
    """
    args = checkpoint.get("args", {})
    parameters = signature(CalibrationNetwork).parameters
    shape = {"coarse_lags": parameters["coarse_lags"].default if args.get("coarse_lags") is None else literal_eval(args["coarse_lags"]),
             "max_range_m": parameters["max_range_m"].default if args.get("max_range_m") is None else float(args["max_range_m"])}
    net = CalibrationNetwork(torch.from_numpy(angles), **shape, **overrides)
    load_calibration_weights(net, checkpoint["model"])
    return net


def clip_frames(index: int, recording_start: int, length: int) -> np.ndarray:
    """Global frame indices of the `length`-frame clip ending at `index`, history clamped at the recording's first frame."""
    return np.maximum(np.arange(index - length + 1, index + 1), recording_start)


def anchor_ranges(scans: np.ndarray, max_range_m: float) -> np.ndarray:
    """Ranges as the network read them (`sanitize_ranges`), for everything that anchors on a beam's endpoint -- vote decoding and
    training targets. Raw scans there would put a missing return's votes at infinity, or at FROG train/val's 61.0 m, while the
    network saw the range limit."""
    return sanitize_ranges(torch.from_numpy(np.asarray(scans, np.float32)), max_range_m).numpy()


@torch.no_grad()
def network_candidates(net: CalibrationNetwork, scans: np.ndarray, poses: np.ndarray, device) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stream one recording `[T, N]` at `poses` `[T, 3]` through the calibration network and decode each frame's candidates
    `(score, x, y)` with DROW's vote grid, each with the decoder features pooled from the beams whose votes land near it."""
    net.eval()
    angles = net.angles.cpu().numpy()
    state, out = None, []
    for scan, pose in zip(scans, poses):
        logit, votes, features, state = net.step(torch.from_numpy(np.asarray(scan, np.float32)[None]).to(device),
                                                 torch.from_numpy(np.asarray(pose, np.float32)[None]).to(device), state)
        candidates, weights = decode_candidates(np.asarray(scan), angles, torch.sigmoid(logit)[0].cpu().numpy(), votes[0].cpu().numpy(), TOP_P, net.max_range_m)
        out.append((candidates, weights @ features[0].T.cpu().numpy()))
    return out


def decode_votes(scan: np.ndarray, angles: np.ndarray, prob: np.ndarray, votes: np.ndarray) -> List[Tuple[float, float, float]]:
    """Per-beam person probabilities and votes -> detections `(score, x, y)` in the detection frame, through DROW's vote grid
    with `train.evaluate_auc`'s settings."""
    r, phi = _win2global(scan[None], angles[None], votes[None, :, 0], votes[None, :, 1])
    probas = np.zeros((1, len(scan), 4), np.float32)
    probas[0, :, 3] = prob
    return [(float(p[3]), float(x), float(y)) for x, y, p in votes_to_detections(r * -np.sin(phi), r * np.cos(phi), probas, **VOTE_DECODING)[0]]


# ---------------------------------------------------------------- data on the device

def gather(split: Split, seq: np.ndarray, frame: np.ndarray, device) -> Dict[str, torch.Tensor]:
    """Frames `[B, L]` of `split` as tensors, with seconds since each stream's previous frame."""
    flat = split.offsets[seq] + frame
    d = split.data
    dt = np.where(frame > 0, d.time[flat] - d.time[np.maximum(flat - 1, 0)], FRAME_PERIOD_S).clip(MIN_DT_S, MAX_DT_S)
    arrays = dict(xy=d.xy[flat], score=d.score[flat], valid=d.valid[flat], target=d.target[flat], pose=d.pose[flat],
                  annotated=d.annotated[flat], gt_xy=d.gt_xy[flat], gt_valid=d.gt_valid[flat], gt_covered=d.gt_covered[flat],
                  features=d.features[flat].astype(np.float32), dt=dt.astype(np.float32))
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in arrays.items()}


def detach(slots: Slots) -> Slots:
    return Slots(**{f.name: getattr(slots, f.name).detach() for f in fields(Slots)})


def focal_loss(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ce = binary_cross_entropy_with_logits(logit, target.float(), reduction="none")
    p = torch.sigmoid(logit)
    return (ce * (1.0 - torch.where(target, p, 1.0 - p)) ** FOCAL_GAMMA * mask).sum()


# ---------------------------------------------------------------- scoring

def score_detections(frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, float]:
    """AP at 0.5 m over frames of (scores, xy, annotated people), plus precision, recall and false positives and negatives per frame
    at 0.3, read off the same precision-recall curve (`tracker_sweep.score_populated`). No frames -- a window with nothing annotated --
    gives undefined metrics."""
    if not frames:
        return dict(ap=float("nan"), precision=float("nan"), recall=float("nan"), fp_per_frame=float("nan"), fn_per_frame=float("nan"), frames=0, people=0)
    d_s, d_xy, d_f, g_xy, g_f = [], [], [], [], []
    for f, (scores, xy, gt) in enumerate(frames):
        d_s.append(scores), d_xy.append(xy), d_f.append(np.full(len(scores), f)), g_xy.append(gt), g_f.append(np.full(len(gt), f))
    g_f = np.concatenate(g_f)
    recs, precs, threshs = _prec_rec_2d(np.concatenate(d_s).astype(np.float32), np.concatenate(d_xy).astype(np.float32), np.concatenate(d_f),
                                         np.concatenate(g_xy).astype(np.float32), g_f, np.full(len(g_f), MATCH_RADIUS_M, np.float32))
    idx = np.nonzero(np.asarray(threshs) >= OPERATING_THRESHOLD)[0]
    i = int(idx[-1]) if len(idx) else 0
    n_gt, n = len(g_f), len(frames)
    tp = float(recs[i]) * n_gt
    fp = tp * (1 - float(precs[i])) / float(precs[i]) if precs[i] > 0 else float("nan")
    return dict(ap=_safe_auc(recs, precs), precision=float(precs[i]), recall=float(recs[i]), fp_per_frame=fp / n, fn_per_frame=(n_gt - tp) / n,
                frames=n, people=n_gt)


def evaluation_segments(split, windows: int, window_frames: int, warmup_frames: int) -> List[Tuple[int, int, int, int]]:
    """(recording, first frame, frames, warm-up frames before scoring): whole recordings when `windows` is 0, otherwise about
    `windows` evenly spaced windows of warm-up plus `window_frames`, shared among the recordings long enough to hold one in
    proportion to their length."""
    span = warmup_frames + window_frames
    room = {s: int(n) - span for s, n in enumerate(split.lengths) if n >= span}
    if not windows or not room:
        return [(s, 0, int(n), 0) for s, n in enumerate(split.lengths)]
    total = sum(r + 1 for r in room.values())
    return [(s, int(first), span, warmup_frames) for s, r in room.items() for first in np.linspace(0, r, max(1, round(windows * (r + 1) / total))).astype(int)]


@torch.no_grad()
def evaluate(model: ObjectMemory, split: Split, device, stride: int, streams: int, segments: List[Tuple[int, int, int, int]],
             slice_frames: int = 512) -> Dict[str, Dict[str, float]]:
    """Play each segment (recording, first frame, frames, warm-up) from empty memory, `streams` at a time, and score every
    `stride`-th annotated frame after its warm-up: the candidates alone, and the candidates rescored plus coasting slots.
    Frames are gathered `slice_frames` at a time, so memory is bounded by streams x slice rather than by the longest recording."""
    model.eval()
    raw, memory = [], []
    segments = sorted(segments, key=lambda seg: -seg[2])
    for group in [segments[i:i + streams] for i in range(0, len(segments), streams)]:
        length = max(seg[2] for seg in group)
        recording = np.array([seg[0] for seg in group])[:, None]
        frame = np.array([seg[1] for seg in group])[:, None] + np.minimum(np.arange(length)[None], np.array([seg[2] for seg in group])[:, None] - 1)
        chosen = np.zeros(frame.shape, dtype=bool)
        for b, (s, first, frames, warmup) in enumerate(group):
            scored = np.nonzero(split.data.annotated[split.offsets[s] + first + warmup:split.offsets[s] + first + frames])[0] + warmup
            chosen[b, scored[::stride]] = True
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
                valid, xy, coasting = batch["valid"][rows, i].cpu().numpy(), batch["xy"][rows, i].cpu().numpy(), out.coasting[rows].cpu().numpy()
                slot_xy, score = out.slot_xy[rows].cpu().numpy(), batch["score"][rows, i].cpu().numpy()
                gt = [batch["gt_xy"][r, i][batch["gt_valid"][r, i]].cpu().numpy() for r in rows]
                for j in range(len(rows)):
                    report = coasting[j] & (slot_p[j] >= MIN_REPORTED_SCORE)
                    raw.append((score[j][valid[j]], xy[j][valid[j]], gt[j]))
                    memory.append((np.concatenate([cand_p[j][valid[j]], slot_p[j][report]]), np.concatenate([xy[j][valid[j]], slot_xy[j][report]]), gt[j]))
    model.train()
    base, new = score_detections(raw), score_detections(memory)
    new["exchange_rate"] = exchange_rate(base["fp_per_frame"], base["fn_per_frame"], new["fp_per_frame"], new["fn_per_frame"])
    return dict(candidates=base, memory=new)


# ---------------------------------------------------------------- training

def load_split(name: str, limit: int, stage2: Optional[Path], device) -> Split:
    """Cached candidates of every recording of FROG `official` `name`: LFE-Peaks' when `stage2` is None, otherwise the trained
    calibration network's at `stage2`, with its decoder features."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    source = "lfe-peaks" if stage2 is None else f"network-{stage2.stem}-{int(stage2.stat().st_mtime)}-limit-silenced"
    path = CACHE_DIR / f"{source}_official_{name}_f{CANDIDATE_FLOOR}_p{TOP_P}{f'_first{limit}' if limit else ''}.pkl"
    if path.exists():
        seqs = [SequenceCandidates(**fields_of) for fields_of in pickle.loads(path.read_bytes())]
    else:
        t0 = time.time()
        # LFE-Peaks keeps the loader's legacy clamp it was always run with; the network reads raw scans and sanitises them itself
        ds = FROG_Dataset(split=name, mode="official", auto_download=False, verbose=False, missing_return_m=LFE_MISSING_RETURN_M if stage2 is None else None)
        angles = frog_laser_angles(720)
        if stage2 is None:
            detector = LFEPeaksDetector(peak_height=CANDIDATE_FLOOR)

            def candidates_of(s):
                return [(c := np.asarray(detector.detect(scan, angles), dtype=np.float64).reshape(-1, 3), np.zeros((len(c), 0))) for scan in ds.scans[s]]
        else:
            net = calibration_network(torch.load(stage2, map_location="cpu"), angles).to(device)

            def candidates_of(s):
                return network_candidates(net, ds.scans[s], ds.odoms[s]["xya"], device)
        seqs = build_candidates(ds, candidates_of, TOP_P, MATCH_RADIUS_M, recordings=limit)
        # plain dicts of arrays, so the cache loads from any module, not only from this script run as __main__
        path.write_bytes(pickle.dumps([{f.name: getattr(s, f.name) for f in fields(SequenceCandidates)} for s in seqs]))
        print(f"built {path.name}: {len(seqs)} recordings, {sum(len(s.time) for s in seqs)} frames, {time.time() - t0:.0f} s", flush=True)
    return pack(seqs)


def train_memory(args) -> None:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device, _ = detect_device()
    splits = {name: load_split(name, args.limit_recordings, args.stage2, device) for name in SPLITS}
    for name, split in splits.items():
        print(f"{name}: {len(split.lengths)} recordings, {split.lengths.sum()} frames, {int(split.data.annotated.sum())} annotated", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best = BestCheckpoint(args.out_dir / f"{args.mode}_object_memory.best.pth")

    model = ObjectMemory(SlotRules(capacity=args.slots), feature_dim=splits["train"].data.features.shape[-1]).to(device)
    print(f"stage 3 parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    start = time.time()
    history = []
    frames_per_epoch = int(splits["train"].lengths.sum())

    for chunk_s in args.chunk_s:
        length = max(2, round(chunk_s / FRAME_PERIOD_S))
        streams = min(args.max_streams, max(1, args.tokens // length))
        iterations = math.ceil(frames_per_epoch / (streams * length))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs * iterations)
        stopper = EarlyStopping(args.patience, args.min_epochs, args.max_epochs, args.max_hours)
        sampler = StreamSampler(splits["train"].lengths, streams, rng)
        slots = model.initial_state(streams, device)
        print(f"\n== chunks of {chunk_s:g} s: {length} frames x {streams} streams, {iterations} iterations per epoch", flush=True)
        epoch = 0
        while True:
            t_epoch, losses = time.time(), []
            for _ in range(iterations):
                seq, frame, reset = sampler.next_chunk(length, args.reset_prob)
                batch = gather(splits["train"], seq, frame, device)
                reset_t = torch.from_numpy(reset).to(device)
                slots = detach(slots)
                loss = torch.zeros((), device=device)
                for t in range(length):
                    out = model.step(batch["xy"][:, t], batch["score"][:, t], batch["valid"][:, t], batch["pose"][:, t], batch["dt"][:, t], slots, reset=reset_t[:, t],
                                     cand_features=batch["features"][:, t])
                    slots = out.slots
                    annotated = batch["annotated"][:, t, None]
                    missed = batch["gt_valid"][:, t] & ~batch["gt_covered"][:, t]
                    loss = loss + focal_loss(out.candidate_logit, batch["target"][:, t], batch["valid"][:, t] & annotated)
                    coast = coast_targets(out.slot_xy, out.coasting, batch["gt_xy"][:, t], missed, MATCH_RADIUS_M)
                    loss = loss + focal_loss(out.slot_logit, coast, out.coasting & annotated)
                loss = loss / batch["annotated"].sum().clamp(min=1)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                schedule.step()
                losses.append(loss.item())
            epoch += 1
            train_s = time.time() - t_epoch
            val = evaluate(model, splits["val"], device, args.val_stride, args.max_streams,
                           evaluation_segments(splits["val"], args.val_windows, round(args.val_window_s / FRAME_PERIOD_S), round(args.val_warmup_s / FRAME_PERIOD_S)))
            improved = stopper.update(val["memory"]["ap"])      # patience restarts with every chunk length
            checkpoint = dict(model=model.state_dict(), chunk_s=chunk_s, epoch=epoch, val=val, args={k: str(v) for k, v in vars(args).items()})
            if improved:
                torch.save(checkpoint, args.out_dir / f"{args.mode}_object_memory.{chunk_s:g}s.best.pth")
            overall = best.offer(val["memory"]["ap"], checkpoint)   # the run's best does not
            hours = (time.time() - start) / 3600
            row = dict(chunk_s=chunk_s, epoch=epoch, loss=float(np.mean(losses)), val_ap=val["memory"]["ap"], val_ap_candidates=val["candidates"]["ap"],
                       val_fp=val["memory"]["fp_per_frame"], val_fn=val["memory"]["fn_per_frame"], gate=model.rescore_gate.item(), train_s=train_s,
                       eval_s=time.time() - t_epoch - train_s, hours=hours, stage_best=improved, best=overall,
                       gpu_gb=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan"))
            history.append(row)
            marker = "  best" if overall else ("  stage best" if improved else "")
            print(f"  epoch {epoch:>3}  loss {row['loss']:.4f}  val AP {row['val_ap']:.2%} (candidates {row['val_ap_candidates']:.2%})  "
                  f"FP {row['val_fp']:.3f} FN {row['val_fn']:.3f}  gate {row['gate']:+.3f}  train {train_s:.0f} s  eval {row['eval_s']:.0f} s  "
                  f"{hours:.2f} h  GPU {row['gpu_gb']:.1f} GB{marker}", flush=True)
            if stopper.should_stop(epoch, hours):
                break
        if best.saved:
            model.load_state_dict(torch.load(best.path, map_location=device)["model"])     # the next stage resumes from the run's best
        if (time.time() - start) / 3600 >= args.max_hours:
            break

    results = dict(history=history, test={})
    for stride in TEST_STRIDES:
        results["test"][f"stride_{stride}"] = evaluate(model, splits["test"], device, stride, args.max_streams, evaluation_segments(splits["test"], 0, 0, 0))
        r = results["test"][f"stride_{stride}"]
        print(f"\ntest, every {stride} annotated frame: candidates AP {r['candidates']['ap']:.2%} P {r['candidates']['precision']:.1%} R {r['candidates']['recall']:.1%} "
              f"FP {r['candidates']['fp_per_frame']:.3f} FN {r['candidates']['fn_per_frame']:.3f}"
              f"  |  memory AP {r['memory']['ap']:.2%} P {r['memory']['precision']:.1%} R {r['memory']['recall']:.1%} "
              f"FP {r['memory']['fp_per_frame']:.3f} FN {r['memory']['fn_per_frame']:.3f}  exchange rate {r['memory']['exchange_rate']:.2f}", flush=True)
    (args.out_dir / f"{args.mode}_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {args.out_dir / f'{args.mode}_results.json'}", flush=True)


# ---------------------------------------------------------------- phase A: stages 1-2 on clips

@dataclass
class FrameSplit:
    """Every frame of a split end to end: ranges `[F, N]`, pose `[F, 3]`, the first frame of its recording `[F]`, and the
    annotated frames with their people as (range, bearing) and as detection-frame xy."""
    scans: np.ndarray
    poses: np.ndarray
    start: np.ndarray
    annotated: np.ndarray
    people_polar: List[np.ndarray]
    people_xy: List[np.ndarray]
    time: np.ndarray


def load_frames(name: str) -> FrameSplit:
    return frames_from_dataset(FROG_Dataset(split=name, mode="official", auto_download=False, verbose=False, missing_return_m=None))


def frames_from_dataset(ds) -> FrameSplit:
    """Pack every recording of `ds` end to end, annotations placed on their own scans through `idet2iscan`."""
    scans, poses, start, annotated, polar, times = [], [], [], [], [], []
    offset = 0
    for s in range(len(ds.scans)):
        n = len(ds.scans[s])
        scans.append(np.asarray(ds.scans[s], np.float32))
        times.append(np.asarray(ds.scan_time[s], np.float64))
        poses.append(np.asarray(ds.odoms[s]["xya"], np.float32))
        start.append(np.full(n, offset))
        for d in range(len(ds.det_id[s])):
            annotated.append(offset + int(ds.idet2iscan[s][d]))
            polar.append(np.asarray(ds.det_wp[s][d], np.float64).reshape(-1, 2))
        offset += n
    xy = [np.array([project_cartesian_from_polar(r, phi) for r, phi in p], dtype=np.float64).reshape(-1, 2) for p in polar]
    return FrameSplit(np.concatenate(scans), np.concatenate(poses), np.concatenate(start), np.array(annotated), polar, xy, time=np.concatenate(times))


@dataclass
class JointState:
    """What the whole detector carries between frames: stage 2's caches, stage 3's slots, and the slots' last person logits."""
    calibration: CalibrationState
    slots: Slots
    slot_logit: torch.Tensor


@dataclass
class JointOutput:
    """One frame of the whole detector: stage 2's per-beam logits `[B, N]` and votes, the decoded candidates `[B, P]` with their pooled
    features, the memory prior drawn into stage 2 `[B, 2, S]`, and stage 3's output."""
    logit: torch.Tensor
    votes: torch.Tensor
    cand_xy: torch.Tensor
    cand_score: torch.Tensor
    cand_valid: torch.Tensor
    cand_features: torch.Tensor
    prior: torch.Tensor
    memory: MemoryOutput


def joint_step(net: CalibrationNetwork, memory: ObjectMemory, ranges: torch.Tensor, pose: torch.Tensor, dt: torch.Tensor,
               state: Optional[JointState]) -> Tuple[JointOutput, JointState]:
    """One frame `[B, N]` of the whole detector for streams that start together (`state` None starts all of them from empty memory):
    the slots' predicted positions drawn into stage 2's bottleneck, stage 2, candidates decoded with DROW's vote grid and their
    features pooled from stage 2's decoder -- the path by which stage 3's loss reaches stage 2 -- and stage 3."""
    b, n = ranges.shape
    device = ranges.device
    sectors = n // 8
    if state is None:
        calibration, slots = None, memory.initial_state(b, device)
        prior = torch.zeros(b, 2, sectors, device=device)
    else:
        calibration, slots = state.calibration, state.slots
        predicted = world_to_sensor(slots.position + slots.velocity * dt[:, None, None], pose)
        prior = render_prior(predicted, slots.value, torch.sigmoid(state.slot_logit), slots.alive, net.angles, sectors)
    logit, votes, features, calibration = net.step(ranges, pose, calibration, prior)
    angles = net.angles.cpu().numpy()
    prob, votes_np, scans = torch.sigmoid(logit).detach().cpu().numpy(), votes.detach().cpu().numpy(), ranges.cpu().numpy()
    xy, score = np.zeros((b, TOP_P, 2), np.float32), np.zeros((b, TOP_P), np.float32)
    valid, weights = np.zeros((b, TOP_P), bool), np.zeros((b, TOP_P, n), np.float32)
    for i in range(b):
        candidates, w = decode_candidates(scans[i], angles, prob[i], votes_np[i], TOP_P, net.max_range_m)
        k = len(candidates)
        xy[i, :k], score[i, :k], valid[i, :k], weights[i, :k] = candidates[:, 1:], candidates[:, 0], True, w
    cand_xy, cand_score, cand_valid = (torch.from_numpy(a).to(device) for a in (xy, score, valid))
    cand_features = torch.einsum("bpn,bcn->bpc", torch.from_numpy(weights).to(device), features)
    out = memory.step(cand_xy, cand_score, cand_valid, pose, dt, slots, cand_features=cand_features)
    return JointOutput(logit, votes, cand_xy, cand_score, cand_valid, cand_features, prior, out), JointState(calibration, out.slots, out.slot_logit)


def clip_batch(split: FrameSplit, frames: np.ndarray, length: int, device, rng: np.random.Generator = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ranges `[B, length, N]` and poses of the clips ending at global `frames`; with `rng`, each clip's history is shuffled."""
    index = np.stack([clip_frames(int(f), int(split.start[f]), length) for f in frames])
    if rng is not None:
        index[:, :-1] = rng.permuted(index[:, :-1], axis=1)
    return torch.from_numpy(split.scans[index]).to(device), torch.from_numpy(split.poses[index]).to(device)


@torch.no_grad()
def evaluate_calibration(net: CalibrationNetwork, split: FrameSplit, positions: np.ndarray, length: int, device, batch: int, shuffle_seed: int = None) -> Dict[str, float]:
    """AP and operating point of the network on annotated frames `positions` of `split`; `shuffle_seed` shuffles clip histories."""
    net.eval()
    angles = net.angles.cpu().numpy()
    rng = None if shuffle_seed is None else np.random.default_rng(shuffle_seed)
    frames = []
    for chunk in np.array_split(positions, math.ceil(len(positions) / batch)):
        ranges, poses = clip_batch(split, split.annotated[chunk], length, device, rng)
        logit, votes, _ = net(ranges, poses)
        prob, votes = torch.sigmoid(logit).cpu().numpy(), votes.cpu().numpy()
        for j, a in enumerate(chunk):
            scan, p = voting_beams(split.scans[split.annotated[a]], prob[j], net.max_range_m)
            dets = np.array(decode_votes(scan, angles, p, votes[j]), dtype=np.float64).reshape(-1, 3)
            frames.append((dets[:, 0], dets[:, 1:], split.people_xy[a]))
    net.train()
    return score_detections(frames)


def train_calibration(args) -> None:
    if args.clip_frames is None:
        args.clip_frames = clip_length(args.coarse_lags)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device, _ = detect_device()
    splits = {name: load_frames(name) for name in SPLITS}
    for name, split in splits.items():
        print(f"{name}: {len(split.scans)} frames, {len(split.annotated)} annotated", flush=True)
    angles = frog_laser_angles(720)
    net = CalibrationNetwork(torch.from_numpy(angles), coarse_lags=args.coarse_lags, dropout=args.dropout, max_range_m=args.max_range_m).to(device)
    if args.init:
        load_calibration_weights(net, torch.load(args.init, map_location=device)["model"])
        print(f"starting from {args.init}", flush=True)
    print(f"stages 1-2 parameters: {sum(p.numel() for p in net.parameters()):,}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best = BestCheckpoint(args.out_dir / "step2_calibration.best.pth")
    periodic = PeriodicCheckpoint(args.out_dir, "step2_calibration", args.checkpoint_every)

    train_split, val_split = splits["train"], splits["val"]
    iterations = math.ceil(len(train_split.annotated) / args.batch)
    eval_every = max(1, round(args.eval_every * iterations))
    val_positions = np.unique(np.linspace(0, len(val_split.annotated) - 1, min(args.val_frames, len(val_split.annotated))).astype(int))
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.schedule == "plateau":
        schedule = plateau_schedule(optimizer, args.lr_factor, args.lr_patience, args.min_lr)
    else:
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.max_epochs * iterations))
    stopper = EarlyStopping(args.patience, args.min_epochs, args.max_epochs, args.max_hours)
    start, step, history, losses, done = time.time(), 0, [], [], False
    t_block = time.time()
    while not done:
        for chunk in np.array_split(rng.permutation(len(train_split.annotated)), iterations):
            frames = train_split.annotated[chunk]
            ranges, poses = clip_batch(train_split, frames, args.clip_frames, device)
            logit, votes, _ = net(ranges, poses)
            targets = [make_targets(anchor_ranges(train_split.scans[f], net.max_range_m), angles, {3: [tuple(p) for p in train_split.people_polar[c]]}) for c, f in zip(chunk, frames)]
            labels = torch.from_numpy(np.stack([t[0] for t in targets])).to(device)
            vote_targets = torch.from_numpy(np.stack([t[1] for t in targets])).to(device)
            loss, _, _ = compute_loss(logit.reshape(-1, 1), votes.reshape(-1, 2), labels.reshape(-1), vote_targets.reshape(-1, 2))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimizer.step()
            if args.schedule == "cosine":
                schedule.step()
            losses.append(loss.item())
            step += 1
            if step % eval_every:
                continue
            epoch, train_s = step / iterations, time.time() - t_block
            val = evaluate_calibration(net, val_split, val_positions, args.clip_frames, device, args.eval_batch)
            improved = stopper.update(val["ap"])
            if args.schedule == "plateau":
                schedule.step(val["ap"])
            checkpoint = dict(model=net.state_dict(), epoch=epoch, val=val, args={k: str(v) for k, v in vars(args).items()})
            best.offer(val["ap"], checkpoint)
            periodic.offer(epoch, checkpoint)
            hours = (time.time() - start) / 3600
            row = dict(epoch=epoch, loss=float(np.mean(losses)), val_ap=val["ap"], val_fp=val["fp_per_frame"], val_fn=val["fn_per_frame"], lr=optimizer.param_groups[0]["lr"], train_s=train_s,
                       eval_s=time.time() - t_block - train_s, hours=hours, best=improved,
                       gpu_gb=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan"))
            history.append(row)
            print(f"  epoch {epoch:6.2f}  loss {row['loss']:.4f}  val AP {row['val_ap']:.2%}  FP {row['val_fp']:.3f} FN {row['val_fn']:.3f}  lr {row['lr']:.1e}  "
                  f"train {train_s:.0f} s  eval {row['eval_s']:.0f} s  {hours:.2f} h  GPU {row['gpu_gb']:.1f} GB{'  best' if improved else ''}", flush=True)
            losses, t_block = [], time.time()
            if stopper.should_stop(epoch, hours):
                done = True
                break

    if best.saved:
        net.load_state_dict(torch.load(best.path, map_location=device)["model"])
    else:
        print("no validation score was ever defined, so no checkpoint was selected: testing the last weights", flush=True)
    test = splits["test"]
    positions = np.arange(0, len(test.annotated), args.test_stride)
    results = dict(history=history, test=dict(
        ordered=evaluate_calibration(net, test, positions, args.clip_frames, device, args.eval_batch),
        shuffled_history=evaluate_calibration(net, test, positions, args.clip_frames, device, args.eval_batch, shuffle_seed=args.seed)))
    for name, r in results["test"].items():
        print(f"\ntest ({name}, every {args.test_stride} annotated frame): AP {r['ap']:.2%}  FP {r['fp_per_frame']:.3f}  FN {r['fn_per_frame']:.3f}", flush=True)
    (args.out_dir / "step2_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {args.out_dir / 'step2_results.json'}", flush=True)


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs; the request ends with the process and changes no power setting."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


# ---------------------------------------------------------------- phase C: the whole detector, with feedback

def recording_bounds(start: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """First frame and length of each recording, from every frame's recording-start index."""
    offsets = np.unique(start)
    return offsets, np.diff(np.append(offsets, len(start)))


def joint_frame_loss(joint: JointOutput, scans: np.ndarray, angles: np.ndarray, people_polar: Sequence[Optional[np.ndarray]],
                     people_xy: Sequence[Optional[np.ndarray]]) -> torch.Tensor:
    """One frame's loss for `B` streams: stage 2's DROW head loss on its per-beam logits and votes, plus stage 3's focal losses on the
    rescored candidates and on coasting slots near people no candidate covers. A stream whose frame is not annotated (None) adds nothing."""
    device = joint.logit.device
    loss = torch.zeros((), device=device)
    for i, (polar, xy) in enumerate(zip(people_polar, people_xy)):
        if polar is None:
            continue
        labels, vote_targets = make_targets(scans[i], angles, {3: [tuple(p) for p in polar]})
        stage2, _, _ = compute_loss(joint.logit[i].reshape(-1, 1), joint.votes[i], torch.from_numpy(labels).to(device), torch.from_numpy(vote_targets).to(device))
        target, covered = candidate_targets(joint.cand_xy[i].detach().cpu().numpy(), joint.cand_valid[i].cpu().numpy(), xy, MATCH_RADIUS_M)
        stage3 = focal_loss(joint.memory.candidate_logit[i], torch.from_numpy(target).to(device), joint.cand_valid[i])
        g = max(len(xy), 1)
        gt, missed = torch.zeros(1, g, 2, device=device), torch.zeros(1, g, dtype=torch.bool, device=device)
        gt[0, :len(xy)] = torch.from_numpy(np.asarray(xy, np.float32).reshape(-1, 2)).to(device)
        missed[0, :len(xy)] = torch.from_numpy(~covered).to(device)
        coast = coast_targets(joint.memory.slot_xy[i:i + 1], joint.memory.coasting[i:i + 1], gt, missed, MATCH_RADIUS_M)[0]
        loss = loss + stage2 + stage3 + focal_loss(joint.memory.slot_logit[i], coast, joint.memory.coasting[i])
    return loss


def detach_joint(state: JointState) -> JointState:
    coarse = state.calibration.coarse
    cache = TemporalCache([f.detach() for f in coarse.features], [q.detach() for q in coarse.poses])
    return JointState(CalibrationState(cache), detach(state.slots), state.slot_logit.detach())


@torch.no_grad()
def evaluate_joint(net: CalibrationNetwork, memory: ObjectMemory, split: FrameSplit, segments: List[Tuple[int, int, int, int]], stride: int,
                   device, streams: int) -> Dict[str, Dict[str, float]]:
    """Play each segment (recording, first frame, frames, warm-up) through the whole detector from empty memory, `streams` at a time,
    and score every `stride`-th annotated frame after its warm-up: stage 2's candidates alone (the memory's feedback already in them),
    and the candidates rescored plus coasting slots."""
    net.eval(), memory.eval()
    offsets, _ = recording_bounds(split.start)
    people_at = {int(f): i for i, f in enumerate(split.annotated)}
    raw, full = [], []
    segments = sorted(segments, key=lambda seg: -seg[2])
    for group in [segments[i:i + streams] for i in range(0, len(segments), streams)]:
        length = max(seg[2] for seg in group)
        frame = np.array([[offsets[s] + first + min(t, frames - 1) for t in range(length)] for s, first, frames, _ in group])
        chosen = np.zeros(frame.shape, dtype=bool)
        for b, (s, first, frames, warmup) in enumerate(group):
            scored = [t for t in range(warmup, frames) if int(offsets[s] + first + t) in people_at]
            chosen[b, scored[::stride]] = True
        state = None
        for t in range(length):
            f = frame[:, t]
            dt = (split.time[f] - split.time[np.maximum(f - 1, 0)] if t else np.full(len(f), FRAME_PERIOD_S)).clip(MIN_DT_S, MAX_DT_S)
            joint, state = joint_step(net, memory, torch.from_numpy(split.scans[f]).to(device), torch.from_numpy(split.poses[f]).to(device),
                                      torch.from_numpy(dt.astype(np.float32)).to(device), state)
            rows = np.nonzero(chosen[:, t])[0]
            if not len(rows):
                continue
            cand_p, slot_p = torch.sigmoid(joint.memory.candidate_logit)[rows].cpu().numpy(), torch.sigmoid(joint.memory.slot_logit)[rows].cpu().numpy()
            valid, xy, score = joint.cand_valid[rows].cpu().numpy(), joint.cand_xy[rows].cpu().numpy(), joint.cand_score[rows].cpu().numpy()
            coasting, slot_xy = joint.memory.coasting[rows].cpu().numpy(), joint.memory.slot_xy[rows].cpu().numpy()
            for j, b in enumerate(rows):
                gt = split.people_xy[people_at[int(f[b])]]
                report = coasting[j] & (slot_p[j] >= MIN_REPORTED_SCORE)
                raw.append((score[j][valid[j]], xy[j][valid[j]], gt))
                full.append((np.concatenate([cand_p[j][valid[j]], slot_p[j][report]]), np.concatenate([xy[j][valid[j]], slot_xy[j][report]]), gt))
    net.train(), memory.train()
    base, new = score_detections(raw), score_detections(full)
    new["exchange_rate"] = exchange_rate(base["fp_per_frame"], base["fn_per_frame"], new["fp_per_frame"], new["fn_per_frame"])
    return dict(candidates=base, memory=new)


def train_joint(args) -> None:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device, _ = detect_device()
    splits = {name: first_recordings(load_frames(name), args.limit_recordings) for name in SPLITS}
    angles = frog_laser_angles(720)
    net = calibration_network(torch.load(args.stage2, map_location="cpu"), angles, prior_channels=PRIOR_CHANNELS).to(device)
    memory = ObjectMemory(SlotRules(capacity=args.slots), feature_dim=net.head.in_channels).to(device)
    memory.load_state_dict(torch.load(args.memory, map_location=device)["model"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best = BestCheckpoint(args.out_dir / "step3b_joint.best.pth")

    train = splits["train"]
    offsets, lengths = recording_bounds(train.start)
    people_at = {int(f): i for i, f in enumerate(train.annotated)}
    length = max(2, round(args.chunk_s / FRAME_PERIOD_S))
    iterations = math.ceil(len(train.scans) / length)
    eval_every = max(1, round(args.eval_every * iterations))
    val = splits["val"]
    val_segments = evaluation_segments(SimpleNamespace(lengths=recording_bounds(val.start)[1]), args.val_windows,
                                       round(args.val_window_s / FRAME_PERIOD_S), round(args.val_warmup_s / FRAME_PERIOD_S))
    parameters = list(net.parameters()) + list(memory.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.max_epochs * iterations))
    stopper = EarlyStopping(args.patience, args.min_epochs, args.max_epochs, args.max_hours)
    sampler = StreamSampler(lengths, 1, rng)
    print(f"joint: {length} frames per chunk, {iterations} chunks per epoch, evaluating every {eval_every}", flush=True)
    state, start, step, history, losses, t_block = None, time.time(), 0, [], [], time.time()
    while True:
        seq, frame, reset = sampler.next_chunk(length, args.reset_prob)
        state = None if state is None else detach_joint(state)
        loss, annotated = torch.zeros((), device=device), 0
        for t in range(length):
            if reset[0, t]:
                state = None
            f = int(offsets[seq[0, t]] + frame[0, t])
            dt = FRAME_PERIOD_S if frame[0, t] == 0 else float(np.clip(train.time[f] - train.time[f - 1], MIN_DT_S, MAX_DT_S))
            joint, state = joint_step(net, memory, torch.from_numpy(train.scans[f][None]).to(device), torch.from_numpy(train.poses[f][None]).to(device),
                                      torch.tensor([dt], dtype=torch.float32, device=device), state)
            i = people_at.get(f)
            if i is not None:
                loss = loss + joint_frame_loss(joint, anchor_ranges(train.scans[f][None], net.max_range_m), angles, [train.people_polar[i]], [train.people_xy[i]])
                annotated += 1
        if annotated:
            optimizer.zero_grad(set_to_none=True)
            (loss / annotated).backward()
            torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
            optimizer.step()
            losses.append(loss.item() / annotated)
        schedule.step()
        step += 1
        if step % eval_every:
            continue
        epoch, train_s = step / iterations, time.time() - t_block
        result = evaluate_joint(net, memory, val, val_segments, args.val_stride, device, args.val_windows)
        improved = stopper.update(result["memory"]["ap"])
        overall = best.offer(result["memory"]["ap"], dict(net=net.state_dict(), memory=memory.state_dict(), epoch=epoch, val=result,
                                                          args={k: str(v) for k, v in vars(args).items()}))
        hours = (time.time() - start) / 3600
        row = dict(epoch=epoch, loss=float(np.mean(losses)) if losses else float("nan"), val_ap=result["memory"]["ap"], val_ap_candidates=result["candidates"]["ap"],
                   val_fp=result["memory"]["fp_per_frame"], val_fn=result["memory"]["fn_per_frame"], train_s=train_s, eval_s=time.time() - t_block - train_s,
                   hours=hours, best=overall, gpu_gb=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan"))
        history.append(row)
        print(f"  epoch {epoch:6.3f}  loss {row['loss']:.4f}  val AP {row['val_ap']:.2%} (stage 2 with feedback {row['val_ap_candidates']:.2%})  "
              f"FP {row['val_fp']:.3f} FN {row['val_fn']:.3f}  train {train_s:.0f} s  eval {row['eval_s']:.0f} s  {hours:.2f} h  GPU {row['gpu_gb']:.1f} GB"
              f"{'  best' if overall else ''}", flush=True)
        losses, t_block = [], time.time()
        if stopper.should_stop(epoch, hours):
            break

    if best.saved:
        checkpoint = torch.load(best.path, map_location=device)
        net.load_state_dict(checkpoint["net"])
        memory.load_state_dict(checkpoint["memory"])
    else:
        print("no validation score was ever defined, so no checkpoint was selected: testing the last weights", flush=True)
    test = splits["test"]
    result = evaluate_joint(net, memory, test, evaluation_segments(SimpleNamespace(lengths=recording_bounds(test.start)[1]), 0, 0, 0), 1, device, args.val_windows)
    print(f"\ntest, every annotated frame: stage 2 with feedback AP {result['candidates']['ap']:.2%} P {result['candidates']['precision']:.1%} "
          f"R {result['candidates']['recall']:.1%} FP {result['candidates']['fp_per_frame']:.3f} FN {result['candidates']['fn_per_frame']:.3f}"
          f"  |  whole detector AP {result['memory']['ap']:.2%} P {result['memory']['precision']:.1%} R {result['memory']['recall']:.1%} "
          f"FP {result['memory']['fp_per_frame']:.3f} FN {result['memory']['fn_per_frame']:.3f}  exchange rate {result['memory']['exchange_rate']:.2f}", flush=True)
    (args.out_dir / "step3b_results.json").write_text(json.dumps(dict(history=history, test=result), indent=1), encoding="utf-8")
    print(f"wrote {args.out_dir / 'step3b_results.json'}", flush=True)


def first_recordings(split: FrameSplit, n: int) -> FrameSplit:
    """The first `n` recordings of a split with their annotations, for smoke runs; the whole split when `n` is 0 or covers it."""
    offsets, _ = recording_bounds(split.start)
    if not n or n >= len(offsets):
        return split
    end = int(offsets[n])
    keep = split.annotated < end
    return FrameSplit(split.scans[:end], split.poses[:end], split.start[:end], split.annotated[keep],
                      [q for q, k in zip(split.people_polar, keep) if k], [q for q, k in zip(split.people_xy, keep) if k], time=split.time[:end])


# ---------------------------------------------------------------- step 4: ablations that need no training

def replay_sort(split: Split, min_hits: int, match_radius: float = MATCH_RADIUS_M) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """SORT (`tracking.SortTracker`, as `tracker_sweep.py` runs it) over every frame of every recording's cached candidates, with
    ego-motion from the poses; its reported tracks on the annotated frames, as (scores, xy, annotated people)."""
    d = split.data
    frames = []
    for offset, n in zip(split.offsets, split.lengths):
        tracker = SortTracker(match_radius=match_radius, min_hits=min_hits, max_age=1)
        previous = None
        for f in range(int(offset), int(offset + n)):
            x, y, theta = (float(v) for v in d.pose[f])
            if previous is None:
                dtheta, dxy = 0.0, (0.0, 0.0)
            else:
                dtheta, dxy = theta - previous[2], odom_xya_delta_to_tracker_frame(x - previous[0], y - previous[1], theta)
            previous = (x, y, theta)
            valid = d.valid[f]
            tracks = tracker.step([(float(sc), float(px), float(py)) for sc, (px, py) in zip(d.score[f][valid], d.xy[f][valid])], dtheta, dxy)
            if d.annotated[f]:
                reported = np.array(tracks, dtype=np.float64).reshape(-1, 3)
                frames.append((reported[:, 0], reported[:, 1:], d.gt_xy[f][d.gt_valid[f]]))
    return frames


def run_ablations(args) -> None:
    device, _ = detect_device()
    angles = frog_laser_angles(720)
    results = dict(sort={}, memory_slots={}, joint_slots={})
    candidates = load_split("test", args.limit_recordings, args.stage2, device)
    raw = score_detections([(d_s[v], d_xy[v], gt[gv]) for d_s, d_xy, v, gt, gv, a in
                            zip(candidates.data.score, candidates.data.xy, candidates.data.valid, candidates.data.gt_xy, candidates.data.gt_valid, candidates.data.annotated) if a])
    results["candidates"] = raw
    print(f"stage 2 candidates alone: AP {raw['ap']:.2%}  P {raw['precision']:.1%}  R {raw['recall']:.1%}  FP {raw['fp_per_frame']:.3f}  FN {raw['fn_per_frame']:.3f}", flush=True)
    for min_hits in (1, 3, 5):
        r = score_detections(replay_sort(candidates, min_hits))
        r["exchange_rate"] = exchange_rate(raw["fp_per_frame"], raw["fn_per_frame"], r["fp_per_frame"], r["fn_per_frame"])
        results["sort"][min_hits] = r
        print(f"SORT min_hits {min_hits}: AP {r['ap']:.2%}  P {r['precision']:.1%}  R {r['recall']:.1%}  FP {r['fp_per_frame']:.3f}  FN {r['fn_per_frame']:.3f}  "
              f"exchange rate {r['exchange_rate']:.2f}", flush=True)

    whole = evaluation_segments(candidates, 0, 0, 0)
    memory_state = torch.load(args.memory, map_location=device)["model"]
    joint_state = torch.load(args.joint, map_location=device)
    test = first_recordings(load_frames("test"), args.limit_recordings)
    test_segments = evaluation_segments(SimpleNamespace(lengths=recording_bounds(test.start)[1]), 0, 0, 0)
    for slots in args.slots:
        memory = ObjectMemory(SlotRules(capacity=slots), feature_dim=candidates.data.features.shape[-1]).to(device)
        memory.load_state_dict(memory_state)
        r = evaluate(memory, candidates, device, 1, args.streams, whole)["memory"]
        results["memory_slots"][slots] = r
        print(f"stage 3 on stage 2's candidates, {slots} slots: AP {r['ap']:.2%}  P {r['precision']:.1%}  R {r['recall']:.1%}  "
              f"FP {r['fp_per_frame']:.3f}  FN {r['fn_per_frame']:.3f}", flush=True)
        net = calibration_network(dict(model=joint_state["net"], args=joint_state.get("args", {})), angles, prior_channels=PRIOR_CHANNELS).to(device)
        memory.load_state_dict(joint_state["memory"])
        r = evaluate_joint(net, memory, test, test_segments, 1, device, args.streams)["memory"]
        results["joint_slots"][slots] = r
        print(f"whole detector with feedback, {slots} slots: AP {r['ap']:.2%}  P {r['precision']:.1%}  R {r['recall']:.1%}  "
              f"FP {r['fp_per_frame']:.3f}  FN {r['fn_per_frame']:.3f}", flush=True)
    (args.out_dir / "step4_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {args.out_dir / 'step4_results.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("step1", "step2", "step3a", "step3b"):
        p = sub.add_parser(name)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--weight-decay", type=float, default=1e-2)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out-dir", type=Path, default=_HERE.parent / "checkpoints_three_horizon")
    for step1 in (sub.choices["step1"], sub.choices["step3a"]):
        step1.add_argument("--chunk-s", type=float, nargs="+", default=[1.0, 10.0, 30.0], help="chunk-length curriculum, seconds")
        step1.add_argument("--max-epochs", type=float, default=100, help="per chunk length; patience usually ends a stage first")
        step1.add_argument("--patience", type=int, default=8, help="evaluations without a new best val AP")
        step1.add_argument("--min-epochs", type=float, default=5, help="per chunk length")
        step1.add_argument("--max-hours", type=float, default=6.0, help="for the whole run")
        step1.add_argument("--reset-prob", type=float, default=0.2, help="chance a chunk starts from empty memory")
        step1.add_argument("--slots", type=int, default=256)
        step1.add_argument("--tokens", type=int, default=1048, help="streams x frames per iteration; ~3-4 GB of GPU memory at 256 slots")
        step1.add_argument("--max-streams", type=int, default=32)
        step1.add_argument("--val-stride", type=int, default=10, help="score every Nth annotated frame of each val window")
        step1.add_argument("--val-windows", type=int, default=64, help="evenly spaced val windows per evaluation, streamed in parallel")
        step1.add_argument("--val-window-s", type=float, default=60.0, help="scored length of each val window")
        step1.add_argument("--val-warmup-s", type=float, default=30.0, help="memory warm-up before each val window is scored")
        step1.add_argument("--limit-recordings", type=int, default=0, help="use only the first N recordings of each split (smoke runs)")
    sub.choices["step1"].set_defaults(stage2=None)
    sub.choices["step3a"].add_argument("--stage2", type=Path, required=True, help="trained stage 1-2 checkpoint whose candidates stage 3 learns from")
    step3b = sub.choices["step3b"]
    step3b.set_defaults(lr=1e-4)
    step3b.add_argument("--stage2", type=Path, required=True, help="trained stage 1-2 checkpoint")
    step3b.add_argument("--memory", type=Path, required=True, help="stage 3 trained on that network's candidates (step3a)")
    step3b.add_argument("--slots", type=int, default=256)
    step3b.add_argument("--chunk-s", type=float, default=10.0, help="seconds of recording per backpropagation chunk, one stream")
    step3b.add_argument("--max-epochs", type=float, default=3)
    step3b.add_argument("--eval-every", type=float, default=0.25, help="epochs between evaluations")
    step3b.add_argument("--patience", type=int, default=8, help="evaluations without a new best val AP")
    step3b.add_argument("--min-epochs", type=float, default=0.5)
    step3b.add_argument("--max-hours", type=float, default=24.0)
    step3b.add_argument("--reset-prob", type=float, default=0.2, help="chance a chunk starts from empty memory")
    step3b.add_argument("--val-stride", type=int, default=10, help="score every Nth annotated frame of each val window")
    step3b.add_argument("--val-windows", type=int, default=16, help="val windows per evaluation, streamed together")
    step3b.add_argument("--val-window-s", type=float, default=60.0)
    step3b.add_argument("--val-warmup-s", type=float, default=30.0)
    step4 = sub.add_parser("step4")
    step4.add_argument("--stage2", type=Path, required=True)
    step4.add_argument("--memory", type=Path, required=True, help="step3a's stage 3")
    step4.add_argument("--joint", type=Path, required=True, help="step3b's whole detector")
    step4.add_argument("--slots", type=int, nargs="+", default=[64, 128, 256])
    step4.add_argument("--streams", type=int, default=16, help="recordings played at once")
    step4.add_argument("--limit-recordings", type=int, default=0, help="only the first N test recordings (smoke runs)")
    step4.add_argument("--out-dir", type=Path, default=_HERE.parent / "checkpoints_three_horizon")
    step3b.add_argument("--limit-recordings", type=int, default=0, help="only the first N recordings of each split (smoke runs)")
    step2 = sub.choices["step2"]
    step2.add_argument("--init", type=Path, default=None, help="start from this checkpoint's weights (a fresh optimizer)")
    step2.add_argument("--schedule", choices=("plateau", "cosine"), default="plateau", help="plateau: cut the rate when val AP stalls")
    step2.add_argument("--lr-factor", type=float, default=0.3)
    step2.add_argument("--lr-patience", type=int, default=4, help="evaluations without improvement before a cut (4 = one epoch)")
    step2.add_argument("--min-lr", type=float, default=1e-5)
    step2.add_argument("--clip-frames", type=int, default=None, help="frames per training clip; default max(coarse_lags) + 1, "
                                                                     "the history the oldest tap reads")
    step2.add_argument("--dropout", type=float, default=0.0, help="dropout on the head's input; 0 reproduces every result so far, "
                                                                 "`docs/RESEARCH.md` §5.4 intends 0.1 for this project's own architectures")
    step2.add_argument("--checkpoint-every", type=float, default=0.0, help="epochs between checkpoints kept whatever they score, "
                                                                          "so test AP can be measured against training duration; 0 keeps only the best")
    step2.add_argument("--max-range-m", type=float, default=MAX_RANGE_M, help="the model's range: a reading that is missing, non-positive or at or "
                                                                             "beyond it reads as this value, whatever the dataset's encoding (FROG is annotated to 10 m; "
                                                                             "DROW has 7.3%% of its labels beyond 10 m)")
    step2.add_argument("--coarse-lags", type=int, nargs="+", default=[0, 8, 16, 24, 32], help="frames the coarse temporal convolution reaches back, per sector; "
                                                                                             "`--coarse-lags 0` alone removes it (A50 phase 2)")
    step2.add_argument("--batch", type=int, default=32)
    step2.add_argument("--eval-batch", type=int, default=64)
    step2.add_argument("--max-epochs", type=float, default=20)
    step2.add_argument("--eval-every", type=float, default=0.25, help="epochs between evaluations")
    step2.add_argument("--patience", type=int, default=12, help="evaluations without a new best val AP")
    step2.add_argument("--min-epochs", type=float, default=2)
    step2.add_argument("--max-hours", type=float, default=24.0)
    step2.add_argument("--val-frames", type=int, default=2000, help="evenly spaced annotated val frames per evaluation")
    step2.add_argument("--test-stride", type=int, default=1, help="score every Nth annotated test frame at the end")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    keep_awake()
    # a ROCm process on Windows can hang in interpreter teardown, after success or failure alike, holding memory and blocking
    # the next run: always leave through os._exit, with the exit code the launcher checks
    try:
        {"step2": train_calibration, "step3b": train_joint, "step4": run_ablations}.get(args.mode, train_memory)(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(), sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
