#!/usr/bin/env python3
"""Test AP against training duration, from a run's periodic checkpoints (`TODO.md` A50 phase 1 item 4).

Stage 2's validation AP peaks around epoch 1 and falls for every epoch after, which was read as overfitting -- but until
`step2 --checkpoint-every` existed, no checkpoint survived past the best one, so nobody had checked whether **test** AP falls
with it. `BestCheckpoint` keeps only the best; `PeriodicCheckpoint` keeps one per interval, and this scores them through the
protocol that produced every other stage-2 number (every annotated test frame, 35-frame clips, batch 64).

Run over two runs at once and the curves answer two questions together:

    does test AP fall after the validation peak?   whether stage 2 overfits on test at all, not only on validation
    is one curve higher, or flatter?               whether an option helps, and whether it helps at the peak or throughout

It answered both on 2026-09-16: test AP peaks at epoch 1.00 and loses 3.58 points by epoch 4, while precision climbs and
recall falls, so the network grows steadily more conservative past the AP-optimal point; and dropout 0.1 lifts every point of
the curve without flattening its decline.

Dropout holds no parameters and `evaluate_calibration` scores in eval mode, so the network is rebuilt at the default 0.0
whatever the checkpoint was trained at -- only the weights differ.

Usage
-----
    python duration_curve.py --dir ../checkpoints_three_horizon/step2_control ../checkpoints_three_horizon/step2_dropout01
    python duration_curve.py --dir ../checkpoints_three_horizon/step2_dropout01 --epochs 0.75   # one checkpoint
    python duration_curve.py --dir ../checkpoints_three_horizon/step2_control --epochs          # every one kept
    python duration_curve.py --dir <runs> --split val --epochs -1 --include-best                # each run's selected weights, on another split

`--split` exists for a specific question (`TODO.md` A50 phase 2): FROG `official`'s val ranks the temporal ablations in almost
the reverse order to its test, and val is three recordings sampled 2,000 frames at a time while test is one recording scored on
every annotated frame. Scoring the same weights on every annotated *val* frame separates "the temporal path does not generalise"
from "the single test recording is the outlier". `--epochs -1` matches no periodic checkpoint, so with `--include-best` it
scores only the weights each run actually selected.
"""

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.datasets.frog_dataset import frog_laser_angles  # noqa: E402
from train_three_horizon import SPLITS, calibration_network, clip_length, detect_device, evaluate_calibration, load_frames  # noqa: E402

EPOCH_IN_NAME = re.compile(r"\.epoch([0-9.]+)\.pth$")
EVAL_BATCH = 64             # its --eval-batch
EPOCH_MATCH = 1e-6
DEFAULT_EPOCHS = (0.25, 1.0, 2.0, 3.0, 4.0)     # enough to place the peak and the decline; each checkpoint costs ~3.5 minutes


def checkpoints(directory: Path, wanted: Sequence[float]) -> List[Tuple[float, Path]]:
    """A run's periodic checkpoints as `(epoch, path)` in epoch order, keeping only the `wanted` epochs when any are given.
    Epochs a run never reached are simply absent, so a shorter run yields a shorter curve rather than an error."""
    found = []
    for path in directory.glob("step2_calibration.epoch*.pth"):
        match = EPOCH_IN_NAME.search(path.name)
        if match and (not wanted or any(abs(float(match.group(1)) - epoch) < EPOCH_MATCH for epoch in wanted)):
            found.append((float(match.group(1)), path))
    return sorted(found)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, nargs="+", required=True, help="run directories holding periodic checkpoints")
    ap.add_argument("--epochs", type=float, nargs="*", default=list(DEFAULT_EPOCHS), help="which epochs to score; pass empty to score every one kept")
    ap.add_argument("--split", choices=SPLITS, default="test", help="which FROG `official` split to score on (default: test)")
    ap.add_argument("--include-best", action="store_true", help="also score `step2_calibration.best.pth`, the weights the run selected")
    ap.add_argument("--out", type=Path, default=_HERE.parent / "checkpoints_three_horizon" / "duration_curve.json")
    args = ap.parse_args()

    device, _ = detect_device()
    angles = frog_laser_angles(720)
    split = load_frames(args.split)
    positions = np.arange(0, len(split.annotated), 1)
    print(f"{args.split}: {len(split.annotated)} annotated frames\n", flush=True)

    results = {}
    for directory in args.dir:
        selected = [(f"{epoch:.2f}", path) for epoch, path in checkpoints(directory, args.epochs)]
        best = directory / "step2_calibration.best.pth"
        if args.include_best and best.exists():
            selected.append(("best", best))
        print(f"=== {directory.name}: {len(selected)} checkpoints ===", flush=True)
        results[directory.name] = {}
        for label, path in selected:
            net = calibration_network(torch.load(path, map_location="cpu", weights_only=False), angles).to(device)
            r = evaluate_calibration(net, split, positions, clip_length(net.coarse.lags), device, EVAL_BATCH)
            results[directory.name][label] = r
            print(f"  {label:>5}  {args.split} AP {r['ap']:.2%}  P {r['precision']:.1%}  R {r['recall']:.1%}  "
                  f"FP {r['fp_per_frame']:.3f}  FN {r['fn_per_frame']:.3f}", flush=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
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
