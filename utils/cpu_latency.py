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

**The total is a per-frame sum, percentiled afterwards**, not the phases' percentiles added: the frame that is slow to
preprocess is rarely the frame that is slow to decode, so adding each phase's own tail describes a frame that never happened.
On this host that mistake read TAKHeLiPeD's tail as 27.90 ms against a measured 25.32 -- 112% of the budget against 101%.

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
TENSOR_IN = ("li2former",)      # the one detector whose `forward_one` takes a tensor; the rest take numpy and refuse one
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


def time_detector(name: str, frames: int, threads: int) -> Dict[str, Dict]:
    """One detector over `frames` consecutive scans: each phase, and what the whole frame cost."""
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
            # `forward_one` is numpy in, numpy out on every trainable detector here *except* Li2Former, whose own takes the
            # tensor -- `from_numpy` refuses a Tensor on one side and `unsqueeze` an array on the other, so ask which.
            x = network_input(name, _extract_input(net, scan, scans_hist, odoms_hist, angles, cfg, "cpu"))
            built = time.perf_counter()
            confs, votes = net.forward_one(x)
            ran = time.perf_counter()
            decode(net, scan, angles, confs, votes)
            done = time.perf_counter()
            if timed:
                phases["preprocess"].append(1000 * (built - start))
                phases["network"].append(1000 * (ran - built))
                phases["decode"].append(1000 * (done - ran))
    return dict(phases={phase: summarise(values) for phase, values in phases.items() if values},
                total=per_frame_total(phases))


def network_input(name: str, tensor):
    """What this detector's `forward_one` accepts: Li2Former the tensor, everyone else an array of it."""
    return tensor if name in TENSOR_IN else tensor.cpu().numpy()


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


def takheliped_row() -> Optional[Dict[str, Dict]]:
    """TAKHeLiPeD's three stages, from the measurement that already took them under this protocol.

    That run timed the whole frame as well as its stages, so its own total travels with the row rather than being
    rebuilt from the stages here -- it is the measured end-to-end tail, which no arithmetic on the stages recovers.
    """
    if not TAKHELIPED_LATENCY.exists():
        return None
    d = json.loads(TAKHELIPED_LATENCY.read_text())

    def phase(key):
        return dict(frames=d["frames"], median=d[key]["median"], p99=d[key]["p99"], mean=d[key]["mean"],
                    budget_median=d[key]["median"] / FROG_PERIOD_MS, budget_p99=d[key]["p99"] / FROG_PERIOD_MS)

    return dict(phases={name: phase(key) for name, key in
                        (("network", "stage2"), ("decode", "decode"), ("memory", "memory")) if key in d},
                total=phase("total"))


def per_frame_total(timings: Dict[str, Sequence[float]]) -> Dict[str, Optional[float]]:
    """What a frame costs end to end: the phases added **per frame**, then summarised.

    Not the phases' percentiles added. A deployment pays all the phases on one frame, and the frame that is slow to
    preprocess is rarely the frame that is slow to decode, so adding each phase's own tail describes a frame that never
    happened and overstates the budget. The medians come out nearly the same either way; the tails do not.
    """
    columns = [list(values) for values in timings.values() if len(values)]
    if not columns:
        return summarise([])
    lengths = {len(column) for column in columns}
    if len(lengths) > 1:
        raise ValueError(f"phases recorded different frame counts: {sorted(lengths)}")
    return summarise([sum(frame) for frame in zip(*columns)])


def pending(requested: Sequence[str], done: Dict[str, Dict], redo: bool = False) -> List[str]:
    """Which of the requested detectors still need timing, in the order asked.

    A row that exists but holds no timing is **not** done: the only way to get one is a detector that could not be
    timed, and the likeliest reason it is in the file is a harness fault since fixed.
    """
    return [name for name in requested
            if redo or (done.get(name) or {}).get("total", {}).get("median") is None]


def merge(done: Dict[str, Dict], fresh: Dict[str, Dict]) -> Dict[str, Dict]:
    """The table after this run: every row it timed, over every row already there. The output is the table, not a slice."""
    return {**done, **fresh}


def total_of(row: Dict) -> Dict[str, Optional[float]]:
    """A row's total: the one it was measured with, or nothing -- never a total reconstructed from its phases."""
    return row.get("total") or summarise([])


DISPLAY = dict(algorithmic="AlgorithmicDetector *(rule-based, C++)*", lfe_peaks="LFE-Peaks", lfe_ppn="LFE-PPN",
               fullscan_tcn="FullScanTCN", temporal_unet="TemporalUNet", spacetime_cnn="SpaceTimeCNN",
               drow="DrowDetector (DROW3)", drspaam="DrSpaamDetector (DR-SPAAM)", li2former="Li2FormerDetector",
               takheliped="**TAKHeLiPeD**, whole detector")


def markdown(rows: Dict[str, Dict], threads: int) -> str:
    """The same table as `report`, as `memory/performance-log.md` wants it -- cheapest first, and emitted rather than typed."""
    def cost(item):
        return total_of(item[1]).get("median") or float("inf")

    out = [f"**CPU, {threads} threads, batch 1, 2,000 consecutive FROG test scans, against FROG's 25 ms frame period.**", "",
           "| detector | pre | network | decode | memory | **total** | p99 | p99 vs budget |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in sorted(rows.items(), key=cost):
        total, phases = total_of(row), row.get("phases", {})
        cells = [f"{(phases.get(p) or {}).get('median'):.2f}" if (phases.get(p) or {}).get("median") is not None else "--"
                 for p in ("preprocess", "network", "decode", "memory")]
        if total.get("median") is None:
            out.append(f"| {DISPLAY.get(name, name)} | " + " | ".join(cells) + f" | not timed | -- | {row.get('failed', '--')} |")
            continue
        out.append(f"| {DISPLAY.get(name, name)} | " + " | ".join(cells) +
                   f" | **{total['median']:.2f} ms** | {total['p99']:.2f} | **{total['budget_p99']:.0%}** |")
    return "\n".join(out)


def report(rows: Dict[str, Dict], threads: int) -> None:
    print(f"\nCPU, {threads} threads, batch 1, FROG test scans in order, against a {FROG_PERIOD_MS:g} ms frame period\n")
    print(f"  {'detector':<16} {'pre':>7} {'network':>8} {'decode':>7} {'memory':>7} {'total':>8} {'p99':>8} {'budget':>8}")
    for name, row in rows.items():
        total, phases = total_of(row), row.get("phases", {})
        cells = {p: (phases.get(p, {}) or {}).get("median") for p in ("preprocess", "network", "decode", "memory")}
        line = f"  {name:<16}" + "".join(f"{v:>8.2f}" if v is not None else f"{'-':>8}" for v in cells.values())
        print(f"{line}{total['median']:>8.2f}{total['p99']:>8.2f}{total['budget_p99']:>7.0%}"
              if total["median"] is not None else f"{line}{'-':>8}{'-':>8}{'-':>8}", flush=True)
    print("\n  'pre' is building the input; a detector whose released form does everything in one call reports it all as "
          "network.\n  'total' and 'p99' are per-frame costs; 'budget' is the p99's share of the frame period.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=2000, help="timed scans per detector, after a warm-up")
    ap.add_argument("--threads", type=int, default=4, help="torch threads, stated in the output because it changes the answer")
    ap.add_argument("--detectors", nargs="+", default=list(ORDER))
    ap.add_argument("--out", type=Path, default=_HERE.parent / "results" / "cpu_latency.json")
    ap.add_argument("--redo", action="store_true", help="re-time detectors the output already holds, instead of skipping them")
    args = ap.parse_args()

    rows: Dict[str, Dict] = json.loads(args.out.read_text())["rows"] if args.out.exists() else {}
    queue = pending(args.detectors, rows, args.redo)
    if skipped := [n for n in args.detectors if n not in queue]:
        print(f"already timed, skipping: {', '.join(skipped)} (--redo to repeat them)", flush=True)

    def persist():
        """After every detector, because the slowest three take 40-55 minutes each and an interruption must not cost them."""
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(threads=args.threads, frames=args.frames, period_ms=FROG_PERIOD_MS, rows=rows),
                                       indent=1), encoding="utf-8")
        args.out.with_suffix(".md").write_text(markdown(rows, args.threads) + "\n", encoding="utf-8")

    for name in queue:
        print(f"timing {name} ...", flush=True)
        try:
            rows = merge(rows, {name: time_detector(name, args.frames, args.threads)})
        except BaseException as failure:                            # one detector must not take the table with it
            print(f"  {name}: could not be timed -- {type(failure).__name__}: {failure}", flush=True)
            rows = merge(rows, {name: dict(phases={}, total=summarise([]), failed=f"{type(failure).__name__}: {failure}")})
        persist()
    takheliped = takheliped_row()
    if takheliped:
        rows = merge(rows, dict(takheliped=takheliped))

    report(rows, args.threads)
    persist()
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
