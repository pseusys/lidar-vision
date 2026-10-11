#!/usr/bin/env python3
"""Every detector's CPU latency, on one host under one protocol (`TODO.md` §A item 29).

`AGENTS.md` rule 2b makes speed half of every accuracy answer, and `memory/performance-log.md`'s speed table could not honour
it: its CPU rows were taken around 2026-09-10 by a different script than the one that timed TAKHeLiPeD, at an unstated thread
count and with no percentile. It satisfied its own stated rule -- one machine -- while breaking the one it needed, **one
protocol**.

**The protocol.** The same FROG scans, in order, one at a time, batch 1, a stated thread count, after an untimed warm-up. What
is timed is **scan in, detections out**, split into the three phases a deployment pays for separately:

    preprocess   building the detector's input: the per-beam cutouts, or the aligned raw scan
    network      the forward pass
    decode       turning per-beam scores and votes into detections (DROW's vote grid, as `evaluate.py` uses)

A detector whose released form does all three inside one call -- the two LFE ONNX graphs, and the C++ `AlgorithmicDetector` --
is timed as one phase and its row says so. Each row carries the median, the 99th percentile and the share of FROG's **25 ms**
frame period, which is the real one (item 1; the stamped median said 26.2 Hz and made every budget 1.53x too generous).

TAKHeLiPeD is not re-timed here: `tracking_baselines.py latency` already measures its three stages on this host at 4 threads,
batch 1, over 2,000 frames, which is this protocol, and that result is merged in rather than reimplemented.

Usage (from utils/)
-----
    python cpu_latency.py                       # every detector, 2,000 frames, 4 threads
    python cpu_latency.py --frames 100          # a smoke run
    python cpu_latency.py --detectors lfe_peaks drow
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

FROG_PERIOD_MS = 25.0           # FROG's real frame period: one scan every 25 ms (item 1, A49)
WARMUP = 5
HEATMAP_HEIGHT, HEATMAP_DISTANCE = 0.3, 5      # `evaluate_auc`'s own peak rule for a heatmap head
V2D = dict(blur_sigma=2.0, blur_win=11, bin_size=0.1, vote_collect_radius=0.5, min_thresh=1e-3)   # its vote-grid settings
SELF_CONTAINED = ("algorithmic", "lfe_peaks", "lfe_ppn")    # released forms that preprocess, run and decode in one call
ORDER = ("algorithmic", "lfe_peaks", "lfe_ppn", "fullscan_tcn", "temporal_unet", "spacetime_cnn", "drow", "drspaam", "li2former")
TAKHELIPED_LATENCY = _HERE.parent / "checkpoints_three_horizon" / "a61" / "latency_cpu_t4.json"


def summarise(durations: Sequence[float]) -> Dict[str, Optional[float]]:
    """One row from a list of per-frame milliseconds: the middle, the tail, and what each costs of a frame period."""
    if not len(durations):
        return dict(frames=0, median=None, p99=None, mean=None, budget_median=None, budget_p99=None)
    ordered = sorted(durations)
    p99 = ordered[min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))]
    middle = median(ordered)
    return dict(frames=len(ordered), median=middle, p99=p99, mean=mean(ordered),
                budget_median=middle / FROG_PERIOD_MS, budget_p99=p99 / FROG_PERIOD_MS)


def time_detector(name: str, frames: int, threads: int) -> Dict[str, Dict[str, Optional[float]]]:
    """One detector over `frames` consecutive scans, phase by phase."""
    import torch                                                    # noqa: PLC0415 -- only the timing path needs it
    from takheliped.datasets.frog_dataset import FROG_Dataset, frog_laser_angles  # noqa: PLC0415
    from takheliped.detectors import DETECTOR_REGISTRY  # noqa: PLC0415
    from train import _extract_input  # noqa: PLC0415

    torch.set_num_threads(threads)
    angles = frog_laser_angles(720)
    cfg = _config(angles)
    ds = FROG_Dataset(split="test", mode="official", auto_download=False, verbose=False)
    sequence = int(np.argmax([len(s) for s in ds.scans]))
    net = DETECTOR_REGISTRY[name].init(verbose=False) if hasattr(DETECTOR_REGISTRY[name], "init") else DETECTOR_REGISTRY[name]()
    if hasattr(net, "eval"):
        net.eval()
    window = getattr(net, "num_scans", getattr(ds, "time_frame", 1))
    phases: Dict[str, List[float]] = {"preprocess": [], "network": [], "decode": []}

    with torch.no_grad():
        for index in range(min(frames + WARMUP, len(ds.scans[sequence]))):
            scan = ds.scans[sequence][index]
            timed = index >= WARMUP
            if name in SELF_CONTAINED:
                # `AlgorithmicDetector` takes one scan and that scan's pose, as `forward_all` calls it; the ONNX detectors
                # take the scan and the laser's angles. Passing either the wrong shape crashes the C++ extension outright.
                start = time.perf_counter()
                if hasattr(net, "detect"):
                    net.detect(scan, angles)
                else:
                    net.forward_one(np.asarray(scan, np.float64), np.asarray(ds.odoms[sequence][index]["xya"], np.float64))
                if timed:
                    phases["network"].append(1000 * (time.perf_counter() - start))
                continue
            scans_hist, odoms_hist = ds.get_scan(sequence, index, window, dtime=1)
            start = time.perf_counter()
            # `forward_one` is numpy in, numpy out on every trainable detector here, so the tensor `_extract_input` builds is
            # handed back as an array rather than passed through -- pybind11 and `from_numpy` both refuse a Tensor.
            x = _extract_input(net, scan, scans_hist, odoms_hist, angles, cfg, "cpu").cpu().numpy()
            built = time.perf_counter()
            confs, votes = net.forward_one(x)
            ran = time.perf_counter()
            decode(net, scan, angles, confs, votes)
            done = time.perf_counter()
            if timed:
                phases["preprocess"].append(1000 * (built - start))
                phases["network"].append(1000 * (ran - built))
                phases["decode"].append(1000 * (done - ran))
    return {phase: summarise(values) for phase, values in phases.items() if values}


def decode(net, scan, angles, confs, votes) -> None:
    """Turn per-beam outputs into detections the way `evaluate.py` does for this head: a vote grid, or peaks."""
    from scipy.signal import find_peaks                             # noqa: PLC0415
    from takheliped.utils.drow_utils import _win2global, votes_to_detections    # noqa: PLC0415

    if getattr(net, "head_type", "drow") == "heatmap":
        find_peaks(np.asarray(confs).reshape(-1), height=HEATMAP_HEIGHT, distance=HEATMAP_DISTANCE)
        return
    # A vote head decodes in two steps, as `evaluate_auc` does it: each beam's offset becomes a global point, then the points
    # are voted into detections. `votes_to_detections` takes those points and the probabilities -- not the scan and the votes.
    scan, votes = np.asarray(scan, np.float64)[None], np.asarray(votes)[None]
    radius, bearing = _win2global(scan, angles[None], votes[:, :, 0], votes[:, :, 1])
    votes_to_detections(radius * -np.sin(bearing), radius * np.cos(bearing), np.asarray(confs)[None], **V2D)


def _config(angles):
    """The FROG laser geometry `_extract_input` needs."""
    from types import SimpleNamespace                               # noqa: PLC0415
    from takheliped.datasets.frog_dataset import FROG_Dataset, frog_laser_angles  # noqa: PLC0415
    return SimpleNamespace(name="frog", angles_fn=frog_laser_angles, fov_min=FROG_Dataset.LASER_MIN_ANGLE,
                           fov_max=FROG_Dataset.LASER_MAX_ANGLE, laser_inc=FROG_Dataset.LASER_INCREMENT)


def takheliped_row() -> Optional[Dict[str, Dict[str, Optional[float]]]]:
    """TAKHeLiPeD's three stages, from the measurement that already took them under this protocol."""
    if not TAKHELIPED_LATENCY.exists():
        return None
    d = json.loads(TAKHELIPED_LATENCY.read_text())
    out = {}
    for phase, key in (("preprocess", None), ("network", "stage2"), ("decode", "decode"), ("memory", "memory")):
        if key and key in d:
            out[phase] = dict(frames=d["frames"], median=d[key]["median"], p99=d[key]["p99"], mean=d[key]["mean"],
                              budget_median=d[key]["median"] / FROG_PERIOD_MS, budget_p99=d[key]["p99"] / FROG_PERIOD_MS)
    return out


def total_row(phases: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    """The phases added: what one frame costs end to end."""
    present = [p for p in phases.values() if p.get("median") is not None]
    if not present:
        return dict(median=None, p99=None, budget_median=None, budget_p99=None)
    total, tail = sum(p["median"] for p in present), sum(p["p99"] for p in present)
    return dict(median=total, p99=tail, budget_median=total / FROG_PERIOD_MS, budget_p99=tail / FROG_PERIOD_MS)


def report(rows: Dict[str, Dict], threads: int) -> None:
    print(f"\nCPU, {threads} threads, batch 1, FROG test scans in order, against a {FROG_PERIOD_MS:g} ms frame period\n")
    print(f"  {'detector':<16} {'pre':>7} {'network':>8} {'decode':>7} {'memory':>7} {'total':>8} {'p99':>8} {'budget':>8}")
    for name, phases in rows.items():
        total = total_row(phases)
        cells = {p: (phases.get(p, {}) or {}).get("median") for p in ("preprocess", "network", "decode", "memory")}
        line = f"  {name:<16}" + "".join(f"{v:>8.2f}" if v is not None else f"{'-':>8}" for v in cells.values())
        print(f"{line}{total['median']:>8.2f}{total['p99']:>8.2f}{total['budget_median']:>7.0%}"
              if total["median"] is not None else f"{line}{'-':>8}{'-':>8}{'-':>8}", flush=True)
    print("\n  'pre' is building the input; a detector whose released form does everything in one call reports it all as network.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=2000, help="timed scans per detector, after a warm-up")
    ap.add_argument("--threads", type=int, default=4, help="torch threads, stated in the output because it changes the answer")
    ap.add_argument("--detectors", nargs="+", default=list(ORDER))
    ap.add_argument("--out", type=Path, default=_HERE.parent / "results" / "cpu_latency.json")
    args = ap.parse_args()

    rows: Dict[str, Dict] = {}
    for name in args.detectors:
        print(f"timing {name} ...", flush=True)
        try:
            rows[name] = time_detector(name, args.frames, args.threads)
        except BaseException as failure:                            # one detector must not take the table with it
            print(f"  {name}: could not be timed -- {type(failure).__name__}: {failure}", flush=True)
            rows[name] = {}
    takheliped = takheliped_row()
    if takheliped:
        rows["takheliped"] = takheliped

    report(rows, args.threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(threads=args.threads, frames=args.frames, period_ms=FROG_PERIOD_MS,
                                        rows={k: dict(phases=v, total=total_row(v)) for k, v in rows.items()}), indent=1),
                        encoding="utf-8")
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
