#!/usr/bin/env python3
"""
Ablation: does the T-frame temporal window actually help on FROG, or is it
dead weight?

Motivation: LFE-Peaks/LFE-PPN are single-frame (no temporal input at all) yet
match or beat FullScanTCN/TemporalUNet on FROG, and SpaceTimeCNN's lead is
plausibly explained by raw capacity/joint fusion rather than motion modeling
(Section 8.5.1). Two concrete reasons to suspect the temporal signal itself
is compromised on FROG specifically:

  1. FROG ships no odometry at all (confirmed: its .h5 files have no pose/
     odom field) — aligned_raw_scan() silently falls back to zero-motion,
     so every "temporal" window is actually raw *unaligned* stacking: beam i
     at frame t-k is compared to beam i at frame t with no correction for
     however much the robot rotated in between.
  2. FROG runs at 40 Hz (vs. the ~10 Hz T=5 was originally justified against
     as "half a gait cycle", Section 5.4) — so T=5 covers only ~100-125 ms
     of real time on FROG, not the ~500 ms the design rationale assumes.

This script trains each of the 3 proposed architectures twice on FROG,
identical in every way except one flag:
  - control:      normal T=5 window (--zero-history not set)
  - zero_history: same T=5 architecture, but every historical frame (all but
                  the current/last one) is zeroed before feature extraction
                  (--zero-history) — so the model can only ever learn from
                  single-frame information, with the architecture itself
                  (parameter count, receptive field) held fixed.

If zero_history scores close to control, the current temporal input isn't
contributing much (consistent with reasons 1/2 above). If control clearly
wins, the temporal window is doing real work despite those issues.

Smoke-test scope: --subsample 0.2 (fast directional signal, not a final
number) — a full run would take the original 3.3-8.9h/model x2 variants.

Usage
-----
  python ablation_zero_history.py
  python ablation_zero_history.py --skip temporal_unet
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import _default_args, train_model  # noqa: E402
from train_all import _run_in_subprocess, _train_worker, _fmt  # noqa: E402

_DETECTORS = ["spacetime_cnn", "fullscan_tcn", "temporal_unet"]
_VARIANTS = ["control", "zero_history"]


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir",   type=Path, default=Path("../checkpoints_frog_ablation"))
    p.add_argument("--subsample", type=float, default=0.2)
    p.add_argument("--epochs",    type=int,   default=30)
    p.add_argument("--patience",  type=int,   default=5)
    p.add_argument("--skip",      nargs="*",  default=[], choices=_DETECTORS)
    return p.parse_args()


def main():
    cli = _parse()
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [d for d in _DETECTORS if d not in set(cli.skip)]

    print(f"\n{'='*60}\n  FROG temporal-window ablation — {len(todo)} model(s) x 2 variants\n"
          f"  subsample={cli.subsample}\n{'='*60}\n")

    results = {}  # (det, variant) -> (final_val_loss, best_val_auc_wp)
    for det in todo:
        for variant in _VARIANTS:
            print(f"\n{'='*60}\n  {det}  [{variant}]\n{'='*60}\n")
            args = _default_args(
                detector=det, dataset="frog",
                train_split="train", val_split="val",
                epochs=cli.epochs, patience=cli.patience,
                lr=1e-3, weight_decay=1e-4, dropout=0.5,
                lr_schedule="plateau", subsample=cli.subsample, batch_size=4,
                auc_every=5, frame_cache=True,
                out=cli.out_dir / f"{det}_{variant}.pth",
                zero_history=(variant == "zero_history"),
            )
            status, payload = _run_in_subprocess(_train_worker, (args,))
            if status == "error":
                print(f"\n  [ERROR] {det}[{variant}] failed: {payload}\n")
                results[(det, variant)] = (None, None)
                continue
            history = payload
            final_val = history["val_loss"][-1] if history["val_loss"] else None
            best_auc = (max(history["val_auc_wp"]) if history.get("val_auc_wp") else None)
            results[(det, variant)] = (final_val, best_auc)
            print(f"\n  Done: {det}[{variant}]  final val_loss={_fmt(final_val)}  "
                  f"best val AUC(wp)={_fmt(best_auc, '.1%') if best_auc is not None else 'n/a'}\n")

    print(f"\n{'='*60}\n  Summary (subsample={cli.subsample})\n{'='*60}\n")
    print(f"  {'Model':<16}  {'control AUC(wp)':>16}  {'zero-hist AUC(wp)':>18}  {'delta':>8}")
    print(f"  {'-'*16}  {'-'*16}  {'-'*18}  {'-'*8}")
    for det in todo:
        _, c_auc = results.get((det, "control"), (None, None))
        _, z_auc = results.get((det, "zero_history"), (None, None))
        c_s = _fmt(c_auc, ".1%") if c_auc is not None else "n/a"
        z_s = _fmt(z_auc, ".1%") if z_auc is not None else "n/a"
        d_s = _fmt(c_auc - z_auc, "+.1%") if (c_auc is not None and z_auc is not None) else "n/a"
        print(f"  {det:<16}  {c_s:>16}  {z_s:>18}  {d_s:>8}")
    print()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
