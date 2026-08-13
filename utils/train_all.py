#!/usr/bin/env python3
"""
Train all custom full-scan detectors sequentially.

DROW and DR-SPAAM use published pre-trained weights and are excluded.
Calls train_model() from train.py for each detector and collects the final
val-loss and AUC results in a summary table.

spacetime_cnn, fullscan_tcn and temporal_unet are all non-recursive
(CNN/TCN/U-Net only, no RNN/attention) and DirectML-compatible — no
force_cpu juggling needed.

Usage
-----
  # All four models, FROG, 30 epochs, early stopping with patience 5
  python train_all.py

  # Override common settings
  python train_all.py --epochs 50 --patience 10 --dataset drow --out-dir runs/

  # Quick smoke-test (5 % of data)
  python train_all.py --epochs 3 --subsample 0.05

  # Skip a specific model
  python train_all.py --skip li2former
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

# train.py is in the same directory — add it to the path if needed
sys.path.insert(0, str(Path(__file__).parent))
from train import (  # noqa: E402
    _default_args, train_model,
    _setup_datasets, _build_model, load_checkpoint,
    evaluate_auc, detect_device,
)


def _train_worker(args, result_queue) -> None:
    """Run train_model() in a child process and report the outcome back.

    Each detector trains in its own OS process so its full memory footprint
    (LidarFrameDataset's per-frame cache is several GB for FROG full-scan
    models) is guaranteed to be returned to the OS when the process exits —
    Python-level refcounting frees the objects, but the CPython/PyTorch
    allocators on Windows don't reliably hand freed heap pages back to the
    OS within a still-running process, which otherwise starves the next
    detector's training of memory.
    """
    try:
        result_queue.put(("ok", train_model(args)))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


def _eval_worker(det, ckpt_path, dataset, train_split, test_split,
                 batch_size, force_cpu, result_queue) -> None:
    """Run evaluate_auc() for one checkpoint in a child process.

    Same rationale as _train_worker: evaluate_auc on FROG's full,
    un-subsampled test split (120k+ frames) transiently allocates ~1-2 GB
    even after trimming redundant buffers (see evaluate_auc).  Running every
    detector's test-eval back-to-back in one process lets that pressure
    compound across calls — observed in practice as a hard crash (access
    violation, not a catchable MemoryError) partway through the last
    detector's pass.  Re-loading the dataset per worker costs a few seconds;
    evaluate_auc itself takes minutes, so the overhead is negligible.
    """
    try:
        test_args = _default_args(
            dataset=dataset, train_split=train_split, val_split=test_split,
        )
        _, test_ds, cfg = _setup_datasets(test_args)
        if test_ds is None:
            result_queue.put(("error", f"no data for split '{test_split}'"))
            return
        dev = "cpu" if force_cpu else detect_device()[0]
        model_args = _default_args(detector=det, force_cpu=force_cpu)
        net = _build_model(model_args).to(dev)
        load_checkpoint(ckpt_path, net)
        aucs = evaluate_auc(net, test_ds, cfg, device=dev, batch_size=batch_size)
        result_queue.put(("ok", aucs))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


def _train_in_process(args):
    """Run train_model() directly, no subprocess.

    li2former's training crashes with an access violation (Windows exit code
    3221225477) when run inside a multiprocessing.Process on this machine,
    even at batch_size=1 where the identical call succeeds when run directly
    — this looks like a torch-directml / Windows-spawn interaction bug, not
    a memory issue (memory pressure is what the subprocess isolation exists
    to avoid for the other three models). li2former runs last in
    _ALL_DETECTORS, so running it in-process here doesn't reintroduce the
    cross-model memory compounding the subprocess isolation was added for.
    """
    try:
        return ("ok", train_model(args))
    except Exception as exc:
        return ("error", str(exc))


def _eval_in_process(det, ckpt_path, dataset, train_split, test_split,
                     batch_size, force_cpu):
    """In-process counterpart to _eval_worker(), for the same reason as
    _train_in_process() — see its docstring."""
    try:
        test_args = _default_args(
            dataset=dataset, train_split=train_split, val_split=test_split,
        )
        _, test_ds, cfg = _setup_datasets(test_args)
        if test_ds is None:
            return ("error", f"no data for split '{test_split}'")
        dev = "cpu" if force_cpu else detect_device()[0]
        model_args = _default_args(detector=det, force_cpu=force_cpu)
        net = _build_model(model_args).to(dev)
        load_checkpoint(ckpt_path, net)
        aucs = evaluate_auc(net, test_ds, cfg, device=dev, batch_size=batch_size)
        return ("ok", aucs)
    except Exception as exc:
        return ("error", str(exc))


# Detectors whose subprocess isolation must be bypassed — see
# _train_in_process()'s docstring.
_RUN_IN_PROCESS = {"li2former"}


def _run_in_subprocess(target, args):
    """Run target(*args, result_queue) in a child process; return (status, payload).

    Joins before draining the queue: payloads here (history dicts, AUC dicts)
    are small enough to never fill the pipe buffer, so joining first means a
    hard crash (OOM-kill, segfault) is detected via exitcode instead of
    hanging forever on a .get() that will never be satisfied.
    """
    result_queue = mp.Queue()
    proc = mp.Process(target=target, args=(*args, result_queue))
    proc.start()
    proc.join()
    if not result_queue.empty():
        return result_queue.get()
    return ("error", f"worker process exited with code {proc.exitcode} "
                      "(crashed or was killed, likely OOM) before reporting a result")

# Canonical training order (DROW and DR-SPAAM use published weights, not trained;
# every detector below has no published weights and is trained from scratch)
_ALL_DETECTORS = [
    "spacetime_cnn",
    "fullscan_tcn",
    "temporal_unet",
    "li2former",
]

# li2former's attention ops crash the DirectML device outright at the default
# batch size ("DML allocator out of memory" -> "The GPU device instance has
# been suspended") — this is a memory-pressure issue, not a hard operator
# incompatibility: batch_size=1 trains fine on DirectML (with some ops
# falling back to CPU per-op, same as temporal_unet's unsupported ops).
# CPU-only training was measured at ~1.5 fr/s (days per epoch at full scale)
# vs. ~8.6 fr/s on DirectML at batch_size=1 — GPU-with-small-batch, not CPU,
# is the right fallback here.
_BATCH_SIZE_OVERRIDES = {"li2former": 1}


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train all detectors sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",      choices=["drow", "frog", "jrdb"], default="frog")
    p.add_argument("--train-split",  default="train")
    p.add_argument("--val-split",    default="val")
    p.add_argument("--test-split",   default="test",
                   help="Split used for final test evaluation (default: test; '' to skip)")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--patience",     type=int,   default=5,
                   help="Early-stopping patience in epochs; 0=disabled (default: 5)")
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.5)
    p.add_argument("--lr-schedule",  default="plateau",
                   choices=["none", "cosine", "plateau"])
    p.add_argument("--subsample",    type=float, default=1.0)
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--auc-every",    type=int,   default=5,
                   help="Compute AUC every N epochs during training (default: 5)")
    p.add_argument("--no-frame-cache", dest="frame_cache",
                   action="store_false",
                   help="Recompute frame preprocessing on every access instead of "
                        "caching it in RAM (~2.6 GB/model for FROG raw_scan). "
                        "Trades training time for memory.")
    p.add_argument("--out-dir",      type=Path,  default=Path("checkpoints"),
                   help="Directory for per-model checkpoint files (default: checkpoints/)")
    p.add_argument("--skip",         nargs="*",  default=[],
                   metavar="DETECTOR",
                   choices=_ALL_DETECTORS,
                   help="Detectors to skip")
    return p.parse_args()


def _fmt(val, fmt=".4f"):
    return f"{val:{fmt}}" if val is not None else "  n/a  "


def main():
    cli = _parse()
    cli.out_dir.mkdir(parents=True, exist_ok=True)

    skip = set(cli.skip)
    todo = [d for d in _ALL_DETECTORS if d not in skip]

    print(f"\n{'='*60}")
    print(f"  Training {len(todo)} full-scan model(s) on {cli.dataset.upper()} "
          f"— {cli.epochs} epochs, patience={cli.patience}")
    print(f"  Models : {', '.join(todo)}")
    print(f"  Output : {cli.out_dir.resolve()}")
    print(f"{'='*60}\n")

    test_split = cli.test_split.strip() or None

    summaries = []   # list of (detector, final_val_loss, best_auc_agnostic, elapsed_s)

    for det in todo:
        print(f"\n{'='*60}")
        print(f"  [{todo.index(det)+1}/{len(todo)}]  {det}")
        print(f"{'='*60}\n")

        args = _default_args(
            detector=det,
            dataset=cli.dataset,
            train_split=cli.train_split,
            val_split=cli.val_split,
            epochs=cli.epochs,
            patience=cli.patience,
            lr=cli.lr,
            weight_decay=cli.weight_decay,
            dropout=cli.dropout,
            lr_schedule=cli.lr_schedule,
            subsample=cli.subsample,
            batch_size=_BATCH_SIZE_OVERRIDES.get(det, cli.batch_size),
            auc_every=cli.auc_every,
            frame_cache=cli.frame_cache,
            out=cli.out_dir / f"{det}.pth",
        )

        t0 = time.perf_counter()
        if det in _RUN_IN_PROCESS:
            status, payload = _train_in_process(args)
        else:
            status, payload = _run_in_subprocess(_train_worker, (args,))
        if status == "error":
            print(f"\n  [ERROR] {det} failed: {payload}\n")
            summaries.append((det, None, None, None))
            continue
        history = payload
        elapsed = time.perf_counter() - t0

        final_val  = history["val_loss"][-1]  if history["val_loss"]  else None
        best_auc   = max(history["val_auc_agnostic"], default=None)
        summaries.append((det, final_val, best_auc, elapsed))

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    test_aucs = {}   # det -> {"agnostic": float, "wc": float, "wa": float, "wp": float}

    if test_split:
        print(f"\n{'='*60}")
        print(f"  TEST EVALUATION  (split='{test_split}')")
        print(f"{'='*60}\n")

        # One-time check that the split has data at all.  The real per-model
        # evaluation below reloads the dataset itself, inside its own
        # subprocess (see _eval_worker) — running all 4 models' full,
        # un-subsampled test-eval passes back-to-back in *this* process is
        # what caused the cross-call memory compounding / hard crash this
        # subprocess isolation fixes.
        test_args = _default_args(
            dataset=cli.dataset,
            train_split=cli.train_split,
            val_split=test_split,   # reuse val_split slot to load the test split
        )
        _, test_ds, _cfg = _setup_datasets(test_args)
        has_test_data = test_ds is not None
        del test_ds, _cfg

        if not has_test_data:
            print(f"  [WARN] No data found for split '{test_split}' — skipping test eval.\n")
        else:
            for det, vl, *_ in summaries:
                ckpt = cli.out_dir / f"{det}.pth"
                if not ckpt.exists():
                    print(f"  [{det}] checkpoint not found, skipping.")
                    test_aucs[det] = None
                    continue

                if det in _RUN_IN_PROCESS:
                    status, payload = _eval_in_process(
                        det, ckpt, cli.dataset, cli.train_split, test_split,
                        cli.batch_size, False,
                    )
                else:
                    status, payload = _run_in_subprocess(_eval_worker, (
                        det, ckpt, cli.dataset, cli.train_split, test_split,
                        cli.batch_size, False,
                    ))
                if status == "error":
                    print(f"  [{det}] eval failed: {payload}")
                    test_aucs[det] = None
                    continue
                aucs = payload
                test_aucs[det] = aucs
                print(f"  {det:<28}  agnostic={aucs['agnostic']:.1%}"
                      f"  wc={aucs['wc']:.1%}"
                      f"  wa={aucs['wa']:.1%}"
                      f"  wp={aucs['wp']:.1%}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    # NB: this table reports the 'agnostic' (any-class) AUC, not 'wp'
    # (person-only) — the metric every table in docs/RESEARCH.md uses. The
    # two can diverge substantially on DROW (which has wc/wa/wp as distinct
    # classes, unlike FROG's person-only annotations) — don't compare this
    # column's numbers directly against RESEARCH.md's tables; re-run
    # evaluate.py for the wp figure instead.
    hdr = f"  {'Model':<28}  {'Val loss':>9}  {'Val AUC(any)':>12}  {'Test AUC(any)':>13}  {'Time':>8}"
    print(hdr)
    print(f"  {'-'*28}  {'-'*9}  {'-'*12}  {'-'*13}  {'-'*8}")
    for det, vl, best_auc, elapsed in summaries:
        t_str  = f"{elapsed/60:.1f} min" if elapsed is not None else "  FAILED"
        va_str = f"{best_auc:.1%}"       if best_auc is not None   else "    n/a"
        v_str  = _fmt(vl)                if vl is not None         else "  FAILED"
        ta     = test_aucs.get(det)
        ta_str = f"{ta['agnostic']:.1%}" if ta is not None         else "    n/a"
        print(f"  {det:<28}  {v_str:>9}  {va_str:>12}  {ta_str:>13}  {t_str:>8}")
    print()


if __name__ == "__main__":
    main()
