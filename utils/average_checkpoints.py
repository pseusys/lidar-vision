#!/usr/bin/env python3
"""Weight averaging for stage 2, part of its recipe since the owner's call of 2026-10-06: average a run's periodic
checkpoints (`step2 --checkpoint-every 0.25`) over the window that scores best on val, and save the average as
`<run>/step2_calibration.averaged.pth`, which loads wherever `step2_calibration.best.pth` does.

Averaging along one training trajectory (stochastic weight averaging, Izmailov et al., UAI 2018) gained in 15 of 18 stage-2 runs, +0.66 pp
test AP on average, and on every seed of the adopted recipe (+0.11 to +0.36; best stage 2 82.20%). The window is chosen on val's
2,000-frame sample from `WINDOWS`, exactly as those numbers were; each run's val score per window goes to `<run>/averaging.json`.

**The val split has to be the run's own.** Until 2026-11-06 this script took the run directory and nothing else,
so it inherited none of the flags that produced the run: it built the network on FROG `official`'s 720-beam grid
and loaded FROG `official` val whatever the checkpoints were trained on. Nothing complains -- an aggregator-free
calibration network is fully convolutional over beams, so a 360-beam state dict loads into a 720-angle network and
a DROW network loads into a FROG one -- and three runs were ranked on the wrong data before the printed val figure
sitting 12 pp below its peers gave it away (`memory/gotchas.md`). Pass the same `--dataset`, `--frog-mode` and
`--beam-stride` the run's stage 2 ran with.

Usage (from utils/)
-----
    python average_checkpoints.py ../checkpoints_three_horizon/<run> [...]
    python average_checkpoints.py ../checkpoints_three_horizon/<run> --dataset drow            # ranked on DROW val
    python average_checkpoints.py ../checkpoints_three_horizon/<run> --beam-stride 2           # ranked at 0.5 deg
    python average_checkpoints.py ../checkpoints_three_horizon/<run> --rank-only               # re-rank, keep the weights
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from duration_curve import checkpoints  # noqa: E402
from train_takheliped import (DATASETS, DEFAULT_DATASET, DEFAULT_MODE, calibration_network,  # noqa: E402
                              dataset_spec, detect_device, evaluate_calibration, load_frames)

WINDOWS: Tuple[Tuple[float, float], ...] = ((0.5, 1.0), (0.75, 1.25), (0.5, 1.5), (1.0, 2.0), (0.5, 2.0), (0.25, 2.25))
VAL_FRAMES = 2000
EVAL_BATCH = 64
EPOCH_MATCH = 1e-6
AVERAGED_NAME = "step2_calibration.averaged.pth"
RANKING_NAME = "averaging.json"
#: A `--rank-only` pass writes here instead. `averaging.json` is the record of what a run's window **was**
#: chosen on, and for the three runs ranked on the wrong val split it is the evidence that found the defect --
#: so a re-ranking pass must not overwrite the thing it is being compared against.
RANK_ONLY_NAME = "averaging.rank_only.json"


def window_members(epochs: Sequence[float], window: Tuple[float, float]) -> List[float]:
    """The epochs inside `window`, both ends included."""
    lo, hi = window
    return [e for e in sorted(epochs) if lo - EPOCH_MATCH <= e <= hi + EPOCH_MATCH]


def average_window(run: Path, window: Tuple[float, float]) -> Dict[str, object]:
    """A checkpoint whose weights are the mean of `run`'s periodic checkpoints inside `window`, with the first member's `args`."""
    members = dict(checkpoints(run, window_members([e for e, _ in checkpoints(run, [])], window)))
    loaded = {e: torch.load(p, map_location="cpu", weights_only=False) for e, p in members.items()}
    first = loaded[min(loaded)]
    state = {k: torch.stack([c["model"][k].float() for c in loaded.values()]).mean(0) if v.is_floating_point() else v for k, v in first["model"].items()}
    return {**first, "model": state, "epoch": f"mean {window[0]:.2f}-{window[1]:.2f}", "averaged_epochs": sorted(loaded)}


def build_parser() -> argparse.ArgumentParser:
    """The runs, plus the three flags that decide what val means and the one that keeps old weights in place."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="run directories holding periodic checkpoints")
    ap.add_argument("--dataset", choices=tuple(DATASETS), default=DEFAULT_DATASET,
                    help="the dataset the runs trained on; a window ranked on another dataset's val is not ranked at all")
    ap.add_argument("--frog-mode", choices=("official", "transferred", "balanced"), default=DEFAULT_MODE,
                    help="FROG's partition, which decides what val holds")
    ap.add_argument("--beam-stride", type=int, default=1,
                    help="keep every Nth beam; must match the runs' own, or the network is ranked at a spacing it never saw")
    ap.add_argument("--rank-only", action="store_true",
                    help="write averaging.json and leave the averaged checkpoint alone -- for re-ranking a run whose "
                         "stage 3 is already trained on the weights that file holds")
    return ap


def ranking_path(run: Path, rank_only: bool) -> Path:
    """Where a pass records each window's val score: its own file when it is only re-ranking."""
    return run / (RANK_ONLY_NAME if rank_only else RANKING_NAME)


def main() -> None:
    args = build_parser().parse_args()
    device, _ = detect_device()
    spec = dataset_spec(args)
    angles = spec.angles()
    val = load_frames("val", args.frog_mode, spec)
    positions = np.unique(np.linspace(0, len(val.annotated) - 1, VAL_FRAMES).astype(int))
    for run in args.runs:
        epochs = [e for e, _ in checkpoints(run, [])]
        scores = {}
        for window in WINDOWS:
            if len(window_members(epochs, window)) > 1:
                net = calibration_network(average_window(run, window), angles).to(device)
                scores[f"{window[0]:.2f}-{window[1]:.2f}"] = evaluate_calibration(net, val, positions, device, EVAL_BATCH)["ap"]
        chosen = max(scores, key=scores.get)
        if not args.rank_only:
            torch.save(average_window(run, tuple(float(x) for x in chosen.split("-"))), run / AVERAGED_NAME)
        ranking_path(run, args.rank_only).write_text(
            json.dumps(dict(val_ap=scores, chosen=chosen, dataset=spec.name, frog_mode=args.frog_mode,
                            beam_stride=spec.decimate), indent=1), encoding="utf-8")
        where = "ranked only" if args.rank_only else f"-> {AVERAGED_NAME}"
        print(f"{run.name}: epochs {chosen} chosen on {spec.name} val ({scores[chosen]:.2%}) {where}", flush=True)
    print("ranked every run" if args.rank_only else "wrote every run's averaged checkpoint", flush=True)


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
