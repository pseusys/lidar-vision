#!/usr/bin/env python3
"""The four evaluation-only slot experiments a write-up about the object memory needs (`TODO.md` A51 phase 5).

None of them trains anything: each replays a trained stage 3 over the FROG `official` test split and changes one thing about how
it is run or scored.

    horizon    empty the memory every N seconds and score the result. The project's claim is that a long horizon carries the
               signal, and the slots' own horizon has never been measured -- only the coarse convolution's (E1).
    odometry   corrupt the poses with a drift that grows over time, and score the room-frame model against the sensor-frame
               ablation. Room coordinates are the design's central claim and they are bought with odometry, which drifts on a
               real robot (E2).
    crowding   score by how many people the frame holds. The design argues memory pays where detection fails, which is crowds;
               a flat curve would contradict the story even with a good average (E3).
    cost       milliseconds per frame against slot count, which also prices the design's only super-linear term, the O(K^2)
               slot-to-slot exchange (E4).

Usage
-----
    python slot_experiments.py horizon --memory <step3a checkpoint> --stage2 <stage 2 checkpoint>
    python slot_experiments.py odometry --memory ... --stage2 ... --sensor-memory <a D1 checkpoint>
    python slot_experiments.py crowding --memory ... --stage2 ...
    python slot_experiments.py cost --memory ... --stage2 ...
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from train_three_horizon import (  # noqa: E402
    MATCH_RADIUS_M, MIN_REPORTED_SCORE, OPERATING_THRESHOLD, Split, detect_device, evaluate, evaluation_segments, gather,
    load_split, memory_ms_per_frame, object_memory, score_detections,
)

RESET_INTERVALS_S = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, None)
DRIFTS_M_PER_S = (0.0, 0.005, 0.02, 0.05, 0.1)
HEADING_DEG_PER_S = 0.5
CROWD_BUCKETS = ((0, 1), (2, 3), (4, 5), (6, 99))
SLOT_COUNTS = (64, 128, 256, 512, 720)
TIMED_FRAMES = 500


def load(args, device):
    """The trained memory and the candidates it was trained on."""
    split = load_split("test", args.limit_recordings, args.stage2, device)
    checkpoint = torch.load(args.memory, map_location=device, weights_only=False)
    return object_memory(checkpoint, split.data.features.shape[-1]).to(device), split


def horizon(args, device) -> Dict:
    """AP against how often the memory is emptied: the slots' own temporal horizon."""
    model, split = load(args, device)
    segments = evaluation_segments(split, 0, 0, 0)
    rows = []
    for interval in RESET_INTERVALS_S:
        r = evaluate(model, split, device, args.stride, args.streams, segments, reset_every_s=interval)
        rows.append(dict(reset_every_s=interval, ap=r["memory"]["ap"] * 100, own_recall=r["memory"]["own_recall"] * 100,
                         candidates_ap=r["candidates"]["ap"] * 100))
        label = "never" if interval is None else f"{interval:g} s"
        print(f"  reset every {label:>7}: memory AP {rows[-1]['ap']:.2f}%  recall at matched FP {rows[-1]['own_recall']:.2f}%", flush=True)
    return dict(rows=rows)


def odometry(args, device) -> Dict:
    """AP against odometry drift, for the room-frame model and, when given, the sensor-frame ablation that does not use poses."""
    models = {"room": load(args, device)}
    if args.sensor_memory:
        model, split = models["room"]
        sensor = object_memory(torch.load(args.sensor_memory, map_location=device, weights_only=False), split.data.features.shape[-1]).to(device)
        models["sensor"] = (sensor, split)
    rows = []
    for name, (model, split) in models.items():
        segments = evaluation_segments(split, 0, 0, 0)
        for drift in DRIFTS_M_PER_S:
            r = evaluate(model, split, device, args.stride, args.streams, segments, drift_m_per_s=drift,
                         heading_deg_per_s=HEADING_DEG_PER_S if drift else 0.0)
            rows.append(dict(frame=name, drift_m_per_s=drift, ap=r["memory"]["ap"] * 100, own_recall=r["memory"]["own_recall"] * 100))
            print(f"  {name:>6} frame, drift {drift:5.3f} m/s: memory AP {rows[-1]['ap']:.2f}%  recall {rows[-1]['own_recall']:.2f}%", flush=True)
    return dict(rows=rows, heading_deg_per_s=HEADING_DEG_PER_S)


@torch.no_grad()
def crowding(args, device) -> Dict:
    """AP by how many annotated people the frame holds, for the candidates and for the memory."""
    model, split = load(args, device)
    model.eval()
    buckets: Dict[str, Dict[str, List]] = {f"{lo}-{hi}": dict(raw=[], memory=[]) for lo, hi in CROWD_BUCKETS}
    segments = sorted(evaluation_segments(split, 0, 0, 0), key=lambda seg: -seg[2])
    for group in [segments[i:i + args.streams] for i in range(0, len(segments), args.streams)]:
        length = max(seg[2] for seg in group)
        recording = np.array([seg[0] for seg in group])[:, None]
        frame = np.array([seg[1] for seg in group])[:, None] + np.minimum(np.arange(length)[None], np.array([seg[2] for seg in group])[:, None] - 1)
        chosen = np.zeros(frame.shape, dtype=bool)
        for b, (sq, first, n, _) in enumerate(group):
            scored = np.nonzero(split.data.annotated[split.offsets[sq] + first:split.offsets[sq] + first + n])[0]
            chosen[b, scored[::args.stride]] = True
        slots = model.initial_state(len(group), device)
        for t0 in range(0, length, 512):
            t1 = min(t0 + 512, length)
            batch = gather(split, recording.repeat(t1 - t0, 1), frame[:, t0:t1], device)
            for t in range(t0, t1):
                i = t - t0
                out = model.step(batch["xy"][:, i], batch["score"][:, i], batch["valid"][:, i], batch["pose"][:, i], batch["dt"][:, i], slots,
                                 reset=torch.full((len(group),), t == 0, dtype=torch.bool, device=device), cand_features=batch["features"][:, i])
                slots = out.slots
                rows = np.nonzero(chosen[:, t])[0]
                if not len(rows):
                    continue
                cand_p, slot_p = torch.sigmoid(out.candidate_logit)[rows].cpu().numpy(), torch.sigmoid(out.slot_logit)[rows].cpu().numpy()
                valid, xy = batch["valid"][rows, i].cpu().numpy(), batch["xy"][rows, i].cpu().numpy()
                slot_xy, coasting = out.slot_xy[rows].cpu().numpy(), out.coasting[rows].cpu().numpy()
                score = batch["score"][rows, i].cpu().numpy()
                gt_xy, gt_valid = batch["gt_xy"][rows, i].cpu().numpy(), batch["gt_valid"][rows, i].cpu().numpy()
                for j in range(len(rows)):
                    gt = gt_xy[j][gt_valid[j]]
                    name = next((f"{lo}-{hi}" for lo, hi in CROWD_BUCKETS if lo <= len(gt) <= hi), None)
                    if name is None:
                        continue
                    report = coasting[j] & (slot_p[j] >= MIN_REPORTED_SCORE)
                    buckets[name]["raw"].append((score[j][valid[j]], xy[j][valid[j]], gt))
                    buckets[name]["memory"].append((np.concatenate([cand_p[j][valid[j]], slot_p[j][report]]),
                                                    np.concatenate([xy[j][valid[j]], slot_xy[j][report]]), gt))
    rows = []
    for name, parts in buckets.items():
        base, new = score_detections(parts["raw"]), score_detections(parts["memory"])
        rows.append(dict(people=name, frames=base["frames"], candidates_ap=base["ap"] * 100, memory_ap=new["ap"] * 100,
                         gain=(new["ap"] - base["ap"]) * 100))
        print(f"  {name:>5} people: {base['frames']:>6,} frames, candidates {rows[-1]['candidates_ap']:.2f}%  "
              f"memory {rows[-1]['memory_ap']:.2f}%  gain {rows[-1]['gain']:+.2f}", flush=True)
    return dict(rows=rows)


def cost(args, device) -> Dict:
    """Milliseconds per frame against slot count, one stream, as on the robot."""
    _, split = load(args, device)
    checkpoint = torch.load(args.memory, map_location=device, weights_only=False)
    rows = []
    for slots in SLOT_COUNTS:
        model = object_memory(checkpoint, split.data.features.shape[-1], capacity=slots).to(device)
        ms = memory_ms_per_frame(model, split, device, frames=TIMED_FRAMES)
        rows.append(dict(slots=slots, ms_per_frame=ms))
        print(f"  {slots:>4} slots: {ms:.2f} ms per frame", flush=True)
    return dict(rows=rows, timed_frames=TIMED_FRAMES)


def output_path(args) -> Path:
    """Where this run's results go: `--out` as given, or one file per experiment under `--out-dir`."""
    return args.out if args.out is not None else args.out_dir / f"a51_{args.experiment}.json"


EXPERIMENTS = dict(horizon=horizon, odometry=odometry, crowding=crowding, cost=cost)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=tuple(EXPERIMENTS))
    ap.add_argument("--memory", type=Path, required=True)
    ap.add_argument("--stage2", type=Path, required=True)
    ap.add_argument("--sensor-memory", type=Path, default=None, help="odometry only: a sensor-frame (D1) checkpoint to compare against")
    ap.add_argument("--out-dir", type=Path, default=_HERE.parent / "checkpoints_three_horizon")
    ap.add_argument("--out", type=Path, default=None, help="the results file; the default is one file per experiment under --out-dir, "
                    "which a second run on another checkpoint overwrites")
    ap.add_argument("--stride", type=int, default=5, help="score every Nth annotated frame")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--limit-recordings", type=int, default=0)
    args = ap.parse_args()

    device, _ = detect_device()
    print(f"{args.experiment}: {args.memory.parent.name}", flush=True)
    result = EXPERIMENTS[args.experiment](args, device)
    out = output_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(memory=str(args.memory), **result), indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
        code = 0
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)
