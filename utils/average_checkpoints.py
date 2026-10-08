#!/usr/bin/env python3
"""Weight averaging for stage 2, part of its recipe since the owner's call of 2026-10-06 (`TODO.md` A59): average a run's periodic
checkpoints (`step2 --checkpoint-every 0.25`) over the window that scores best on val, and save the average as
`<run>/step2_calibration.averaged.pth`, which loads wherever `step2_calibration.best.pth` does.

Averaging along one training trajectory (stochastic weight averaging, Izmailov et al., UAI 2018) gained in 15 of 18 A59 runs, +0.66 pp
test AP on average, and on every seed of the adopted recipe (+0.11 to +0.36; best stage 2 82.20%). The window is chosen on val's
2,000-frame sample from `WINDOWS`, exactly as those numbers were; each run's val score per window goes to `<run>/averaging.json`.

Usage (from utils/)
-----
    python average_checkpoints.py ../checkpoints_three_horizon/<run> [...]
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
from follow_the_drow.datasets.frog_dataset import frog_laser_angles  # noqa: E402
from train_three_horizon import calibration_network, clip_length, detect_device, evaluate_calibration, load_frames  # noqa: E402

WINDOWS: Tuple[Tuple[float, float], ...] = ((0.5, 1.0), (0.75, 1.25), (0.5, 1.5), (1.0, 2.0), (0.5, 2.0), (0.25, 2.25))
VAL_FRAMES = 2000
EVAL_BATCH = 64
EPOCH_MATCH = 1e-6
AVERAGED_NAME = "step2_calibration.averaged.pth"


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="run directories holding periodic checkpoints")
    args = ap.parse_args()
    device, _ = detect_device()
    angles = frog_laser_angles(720)
    val = load_frames("val")
    positions = np.unique(np.linspace(0, len(val.annotated) - 1, VAL_FRAMES).astype(int))
    for run in args.runs:
        epochs = [e for e, _ in checkpoints(run, [])]
        scores = {}
        for window in WINDOWS:
            if len(window_members(epochs, window)) > 1:
                net = calibration_network(average_window(run, window), angles).to(device)
                scores[f"{window[0]:.2f}-{window[1]:.2f}"] = evaluate_calibration(net, val, positions, clip_length(net.coarse.lags), device, EVAL_BATCH)["ap"]
        chosen = max(scores, key=scores.get)
        torch.save(average_window(run, tuple(float(x) for x in chosen.split("-"))), run / AVERAGED_NAME)
        (run / "averaging.json").write_text(json.dumps(dict(val_ap=scores, chosen=chosen), indent=1), encoding="utf-8")
        print(f"{run.name}: epochs {chosen} chosen on val ({scores[chosen]:.2%}) -> {AVERAGED_NAME}", flush=True)
    print("wrote every run's averaged checkpoint", flush=True)


if __name__ == "__main__":
    try:
        main()
        code = 0
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)
