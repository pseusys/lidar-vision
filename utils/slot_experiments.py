#!/usr/bin/env python3
"""The four evaluation-only slot experiments a write-up about the object memory needs.

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

from takheliped.detectors.takheliped import ObjectMemory  # noqa: E402
from train_takheliped import (  # noqa: E402
    DT_MODES, MIN_REPORTED_SCORE, decimate, detect_device, evaluate, evaluation_segments, gather,
    load_split, memory_ms_per_frame, object_memory, score_detections,
)

RESET_INTERVALS_S = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, None)
DRIFTS_M_PER_S = (0.0, 0.005, 0.02, 0.05, 0.1)
HEADING_DEG_PER_S = 0.5
CROWD_BUCKETS = ((0, 1), (2, 3), (4, 5), (6, 99))
SLOT_COUNTS = (64, 128, 256, 512, 720)
TIMED_FRAMES = 500
# The width sweep, bracketing the adopted 0.6 m: half it, and on up to a width at which every slot is about equally near,
# which is the global-state arm approached from the other direction.
SIGMAS_M = (0.3, 0.45, 0.6, 0.9, 1.5, 3.0)
#: The slot-budget arm. Brackets the adopted 256 in both directions, because a sweep that only shrinks cannot show 256 is too
#: small. FROG holds at most a few dozen people in view, so the low end is where the interesting answer lives.
CAPACITIES = (32, 64, 128, 256, 512)
# Four shapes share a 20% drop so gap shape is separated from how much data was lost; the last raises the share
# to 50% at the same shape. Bursts stay under 40 frames because MAX_DT_S is 1.0 s and a longer gap saturates the clamp.
FLAKY_PATTERNS = (
    ("none", lambda s: np.ones(len(s.data.time), bool)),
    ("periodic-5", lambda s: periodic_keep(s, 5)),                  # 20%, one-frame gaps: dt alternates 25/50 ms
    ("bernoulli-0.2", lambda s: bernoulli_keep(s, 0.2, seed=0)),    # 20%, random gaps, mostly one frame
    ("burst-8/40", lambda s: burst_keep(s, 8, 40)),                 # 20%, 0.2 s outages once a second
    ("burst-20/100", lambda s: burst_keep(s, 20, 100)),             # 20%, 0.5 s outages every 2.5 s
    ("burst-20/40", lambda s: burst_keep(s, 20, 40)),               # 50%, 0.5 s outages every second
)


def _per_recording(split, mark) -> np.ndarray:
    """A keep mask over packed frames, built one recording at a time; `mark(n)` returns that recording's own mask.

    Never across a boundary: recordings are packed end to end, so a pattern applied to the flat array would treat the
    seam between two of them as an outage. Each recording's **first frame is always kept** -- it carries no elapsed time
    anyway, so dropping it only shortens the warm-up without creating a gap to measure.
    """
    keep = np.zeros(len(split.data.time), bool)
    for offset, n in zip(split.offsets, split.lengths):
        inside = mark(int(n))
        inside[0] = True
        keep[int(offset):int(offset) + int(n)] = inside
    return keep


def periodic_keep(split, every: int) -> np.ndarray:
    """Drop every `every`-th frame of each recording -- the mildest shape, and on its own probably too mild.

    At 40 Hz it makes `dt` alternate between one and two periods, 25 and 50 ms. Worth having as the floor of the sweep
    precisely because a mechanism that cannot be separated here needs a real outage to show itself.
    """
    return _per_recording(split, lambda n: (np.arange(n) % every) != every - 1)


def bernoulli_keep(split, p: float, seed: int) -> np.ndarray:
    """Drop each frame independently with probability `p`: a noisy link rather than a stalled sensor."""
    def mark(n):
        return np.random.default_rng([seed, n]).random(n) >= p
    return _per_recording(split, mark)


def burst_keep(split, length: int, period: int) -> np.ndarray:
    """Drop `length` consecutive frames at the end of every `period`-frame block: a periodic sensor dropout.

    This is the shape that matters. A burst of 20 frames is half a second of nothing, which is where a learned decay
    with a 1-60 s horizon actually has to decide whether a slot still exists -- and `MAX_DT_S` is 1.0 s, so a sweep
    must stay under 40 frames or the gap saturates the clamp instead of being measured.
    """
    def mark(n):
        return (np.arange(n) % period) < period - length
    return _per_recording(split, mark)


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


def sigma(args, device) -> Dict:
    """AP against `sigma_m`, the width of the recall attention's distance bias.

    **Replayed, not retrained**, which is minutes instead of hours: `step` reads `self.sigma_m` on every call, so a
    trained memory can be played at other widths with its weights untouched -- exactly how `cost` replays it at other
    slot counts.

    **Why this parameter.** The global-state ablation settled that addressing is the entire mechanism (one global state captures 5% of the
    slots' gain), which makes `sigma_m` the number that defines the thing that matters. It has never been varied in any
    form; an earlier sweep covered `gate_m`, the *association* gate inside `manage_slots`, which is a different parameter in a
    different component -- and that is why the learned-width question was closed and this is its sharper replacement.

    **Read a flat curve weakly.** The weights adapted to 0.6, so a replay understates how much the width matters; a
    sharp curve is strong evidence and a flat one only says the trained model is locally insensitive. Train an arm only
    if the replay looks live.
    """
    model, split = load(args, device)
    segments = evaluation_segments(split, 0, 0, 0)
    rows = []
    for width in SIGMAS_M:
        model.sigma_m = width
        r = evaluate(model, split, device, args.stride, args.streams, segments)
        rows.append(dict(sigma_m=width, ap=r["memory"]["ap"] * 100, candidates_ap=r["candidates"]["ap"] * 100,
                         gain=(r["memory"]["ap"] - r["candidates"]["ap"]) * 100, own_recall=r["memory"]["own_recall"] * 100))
        print(f"  sigma {width:4.2f} m: memory AP {rows[-1]['ap']:.2f}%  candidates {rows[-1]['candidates_ap']:.2f}%"
              f"  gain {rows[-1]['gain']:+.2f} pp", flush=True)
    return dict(rows=rows, trained_at=ObjectMemory().sigma_m)


def native_timing(split) -> Dict:
    """How uniform FROG actually is, both as stamped and as the model is fed it.

    The two differ on purpose and the difference is the item's premise: the raw stamps come in bunches of roughly 38 ms,
    38 ms and under 0.1 ms, so `frame_periods` averages them over `DT_AVERAGE_FRAMES`. Whatever jitter the
    stamps carry, what reaches `exp(-dt * rate)` is that average -- so this is the number that says whether the
    mechanism has ever had anything to act on.
    """
    stamped = np.concatenate([np.diff(split.data.time[int(o):int(o) + int(n)]) for o, n in zip(split.offsets, split.lengths)])
    fed = np.concatenate([split.dt[int(o) + 1:int(o) + int(n)] for o, n in zip(split.offsets, split.lengths)])
    return dict(stamped_median_ms=float(np.median(stamped)) * 1e3,
                stamped_p05_ms=float(np.percentile(stamped, 5)) * 1e3,
                stamped_p95_ms=float(np.percentile(stamped, 95)) * 1e3,
                stamped_under_1ms_share=float((stamped < 1e-3).mean()),
                dt_median_ms=float(np.median(fed)) * 1e3,
                dt_p05_ms=float(np.percentile(fed, 5)) * 1e3,
                dt_p95_ms=float(np.percentile(fed, 95)) * 1e3)


def flaky(args, device) -> Dict:
    """AP against dropped frames, in both `dt` modes: the only way FROG can test the dt-aware design.

    **The metric is the gain over the arm's own candidates on the same decimated stream, not its raw AP.** Dropping
    frames removes information, which moves the scored population's difficulty for every arm alike and for the
    candidates too, so a raw-AP curve would mostly measure "fewer frames is worse". The candidates are history-free --
    stage 2 sees one scan -- so their AP on a decimated stream is that population's intrinsic difficulty, and
    subtracting it leaves what the memory contributed. It doubles as the harness's control: a pattern where the
    candidate AP moves far from the undecimated run is a pattern whose population changed, not a result.

    Four shapes at a constant 20% share separate **gap shape** from **how much data was lost**; `burst-20/40` then
    raises the share to 50% at the same shape.
    """
    model, split = load(args, device)
    native = native_timing(split)
    print(f"  FROG as recorded: stamped intervals median {native['stamped_median_ms']:.2f} ms, "
          f"5-95% {native['stamped_p05_ms']:.2f}-{native['stamped_p95_ms']:.2f}, "
          f"{native['stamped_under_1ms_share'] * 100:.1f}% under 1 ms (bunched stamps)", flush=True)
    print(f"  what the model is actually fed: averaged dt median {native['dt_median_ms']:.2f} ms, "
          f"5-95% {native['dt_p05_ms']:.2f}-{native['dt_p95_ms']:.2f} -- the spread the dt-aware rule has had to work with",
          flush=True)
    rows = []
    for name, pattern in FLAKY_PATTERNS:
        keep = pattern(split)
        # "none" is bit-identical in both modes by construction (every gap is one frame), and the tests pin that.
        for dt_mode in (("exact",) if name == "none" else DT_MODES):
            thin = decimate(split, keep, dt_mode)
            r = evaluate(model, thin, device, args.stride, args.streams, evaluation_segments(thin, 0, 0, 0))
            rows.append(dict(pattern=name, dt_mode=dt_mode, kept=float(keep.mean()),
                             ap=r["memory"]["ap"] * 100, candidates_ap=r["candidates"]["ap"] * 100,
                             gain=(r["memory"]["ap"] - r["candidates"]["ap"]) * 100,
                             own_recall=r["memory"]["own_recall"] * 100))
            print(f"  {name:>13} {dt_mode:>8} dt, {keep.mean() * 100:5.1f}% kept: memory AP {rows[-1]['ap']:.2f}%"
                  f"  candidates {rows[-1]['candidates_ap']:.2f}%  gain {rows[-1]['gain']:+.2f} pp", flush=True)
    return dict(rows=rows, native_timing=native)


def output_path(args) -> Path:
    """Where this run's results go: `--out` as given, or one file per experiment under `--out-dir`."""
    return args.out if args.out is not None else args.out_dir / f"a51_{args.experiment}.json"


def slots(args, device) -> Dict:
    """AP against the slot count, the last number in the design nobody has priced.

    **Replayed, not retrained.** `object_memory(..., capacity=k)` rebuilds the trained memory at another slot count
    with its weights untouched -- the same trick `cost` uses to price the O(K) terms and `sigma` uses for the recall
    width -- so the whole sweep costs minutes rather than one arm per point.

    **Why it matters beyond tidiness.** Both of stage 3's costly terms are O(K): the recall attention over slots and
    `manage_slots`' distance matrix. `--dim 64` already took 28% off the memory phase by halving the state *width*;
    capacity is the other axis, and if the model is insensitive below 256 it is free CPU on the axis that decides
    deployment.

    **How to read it.** A flat curve is weak evidence -- the weights adapted to 256, so a replay understates how much
    capacity matters. A **cliff** is strong: a capacity the model genuinely needs cannot be removed without one. Train
    an arm only if a smaller count looks live.
    """
    model, split = load(args, device)
    segments = evaluation_segments(split, 0, 0, 0)
    checkpoint = torch.load(args.memory, map_location="cpu", weights_only=False)
    rows = []
    for capacity in CAPACITIES:
        model = object_memory(checkpoint, split.data.features.shape[-1], capacity=capacity).to(device).eval()
        r = evaluate(model, split, device, args.stride, args.streams, segments)
        rows.append(dict(capacity=capacity, ap=r["memory"]["ap"] * 100, candidates_ap=r["candidates"]["ap"] * 100,
                         gain=(r["memory"]["ap"] - r["candidates"]["ap"]) * 100))
        print(f"  slots {capacity:4d}: memory AP {rows[-1]['ap']:.2f}%  candidates {rows[-1]['candidates_ap']:.2f}%"
              f"  gain {rows[-1]['gain']:+.2f}", flush=True)
    # No `memory` key: `main` injects one, and returning it here raised "dict() got multiple values" *after* the
    # whole sweep had run -- eight minutes of work lost to the write, not the compute.
    return dict(rows=rows, stride=args.stride)


EXPERIMENTS = dict(horizon=horizon, odometry=odometry, crowding=crowding, cost=cost, flaky=flaky, sigma=sigma,
                   slots=slots)


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
    except SystemExit as leaving:        # argparse's --help and its usage errors: pass their status through
        code = leaving.code if isinstance(leaving.code, int) else (0 if leaving.code is None else 1)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)                          # a finished ROCm process can hang past a normal exit (memory/gotchas.md)
