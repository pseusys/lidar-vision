#!/usr/bin/env python3
"""
Fine-tune the FROG-trained proposed architectures on DROW, instead of
training DROW from a random init.

Motivation: DROW-native training from scratch (checkpoints_drow/) overfits
almost immediately — spacetime_cnn's val_loss gets worse every epoch after
epoch 1, because DROW's train split has ~17.8x fewer person-annotation
instances than FROG's (17,665 frames x 1.35 people/frame vs 108,356 frames x
3.93 people/frame). Initialising from the FROG checkpoint lets the model
start from features learned on FROG's much larger effective dataset, with a
low LR so DROW only lightly adapts those features instead of re-learning
from noise.

Same config as checkpoints_drow/ (train_all.py's defaults: batch_size=4,
patience=5, lr_schedule=plateau, dropout=0.5, epochs=30) except:
  - init_weights = the matching FROG .best.pth checkpoint
  - lr           = 1e-4 instead of 1e-3 (10x lower, standard fine-tune practice)

Usage
-----
  python finetune_on_drow.py
  python finetune_on_drow.py --skip temporal_unet
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import _default_args, train_model  # noqa: E402
from train_all import _run_in_subprocess, _train_worker, _fmt  # noqa: E402

_DETECTORS = ["spacetime_cnn", "fullscan_tcn", "temporal_unet"]


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frog-dir", type=Path, default=Path("../checkpoints"),
                   help="Directory with FROG-trained <detector>.best.pth files")
    p.add_argument("--out-dir",  type=Path, default=Path("../checkpoints_drow_ft"))
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--epochs",   type=int,   default=30)
    p.add_argument("--patience", type=int,   default=5)
    p.add_argument("--skip",     nargs="*",  default=[], choices=_DETECTORS)
    return p.parse_args()


def main():
    cli = _parse()
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [d for d in _DETECTORS if d not in set(cli.skip)]

    print(f"\n{'='*60}\n  Fine-tuning {len(todo)} model(s) on DROW from FROG init\n{'='*60}\n")

    summaries = []
    for det in todo:
        frog_ckpt = cli.frog_dir / f"{det}.best.pth"
        print(f"\n{'='*60}\n  {det}  (init from {frog_ckpt})\n{'='*60}\n")
        if not frog_ckpt.exists():
            print(f"  [SKIP] {frog_ckpt} not found\n")
            continue

        args = _default_args(
            detector=det, dataset="drow",
            train_split="train", val_split="val",
            epochs=cli.epochs, patience=cli.patience,
            lr=cli.lr, weight_decay=1e-4, dropout=0.5,
            lr_schedule="plateau", subsample=1.0, batch_size=4,
            auc_every=5, frame_cache=True,
            out=cli.out_dir / f"{det}.pth",
            init_weights=frog_ckpt,
        )

        status, payload = _run_in_subprocess(_train_worker, (args,))
        if status == "error":
            print(f"\n  [ERROR] {det} failed: {payload}\n")
            summaries.append((det, None, None))
            continue

        history = payload
        final_val = history["val_loss"][-1] if history["val_loss"] else None
        best_auc = (max(history["val_auc_wp"]) if history.get("val_auc_wp") else None)
        summaries.append((det, final_val, best_auc))
        print(f"\n  Done: {det}  final val_loss={_fmt(final_val)}  "
              f"best val AUC(wp)={_fmt(best_auc, '.1%') if best_auc is not None else 'n/a'}\n")

    print(f"\n{'='*60}\n  Summary\n{'='*60}\n")
    print(f"  {'Model':<16}  {'val_loss':>10}  {'best val AUC(wp)':>18}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*18}")
    for det, vl, auc in summaries:
        vl_s  = _fmt(vl) if vl is not None else "  n/a  "
        auc_s = _fmt(auc, ".1%") if auc is not None else "n/a"
        print(f"  {det:<16}  {vl_s:>10}  {auc_s:>18}")
    print()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
