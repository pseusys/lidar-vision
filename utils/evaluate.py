#!/usr/bin/env python3
"""
Comprehensive evaluation and throughput benchmark for all person detectors.

All inference runs on CPU for reproducible, device-independent results.

Sections
--------
1. Dataset verification  --verify : FoV coverage + annotation-alignment stats
2. Algorithmic evaluation          : precision / recall on the test set
3. NN model AUC          --drow / --drspaam / --spacetime-cnn / --fullscan-tcn / ...
                                   : load checkpoint, compute AUC on CPU
4. Throughput benchmark  --bench   : forward / backward timing on synthetic data

Usage
-----
  # Evaluate DROW and DR-SPAAM using bundled published weights (no path needed)
  python evaluate.py --dataset drow --drow --drspaam

  # Evaluate trained models on FROG test set, also benchmark
  python evaluate.py --dataset frog --split test
                     --drspaam   checkpoints/drspaam.pth
                     --fullscan-tcn checkpoints/fstcn.pth
                     --bench

  # Benchmark only (no dataset needed)
  python evaluate.py --bench --no-eval

  # Dataset verification only
  python evaluate.py --dataset drow --verify --no-bench
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.detectors import (
    AlgorithmicDetector,
    DrowDetector,
    DrSpaamDetector,
    Li2FormerDetector,
    SpaceTimeCNNDetector,
    FullScanTCNDetector,
    TemporalUNetDetector,
    LFEPeaksDetector,
    LFEPPNDetector,
)
from follow_the_drow.utils.drow_utils import (
    laser_angles, laser_minimum, laser_maximum, laser_increment,
    project_cartesian_from_polar, standard_cartesian_to_project_cartesian, cutout, raw_scan,
    _prec_rec_2d,
)
from train import (
    _make_optimizer, _build_model, _default_args,
    evaluate_auc, load_checkpoint, _safe_auc,
)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset setup
# ──────────────────────────────────────────────────────────────────────────────

def _subsample_dataset(dataset, stride: int) -> None:
    """Systematically thin every sequence to every `stride`-th detection
    index, in place (det_id, det_wc/wa/wp, idet2iscan kept aligned).

    Why a stride and not a smaller dataset to begin with: FROG annotates
    every scan at 40 Hz (unlike DROW, where only ~5% of scans are labeled
    to begin with), so consecutive FROG frames are highly temporally
    correlated near-duplicates for AP purposes — evaluating all of them
    has sharply diminishing statistical value per frame, it just multiplies
    wall-clock time. A systematic stride decorrelates the evaluated sample
    (rather than e.g. truncating to a contiguous prefix, which would just
    narrow the time window covered) while cutting eval time by ~stride.
    """
    if stride <= 1:
        return
    for seq in range(len(dataset.det_id)):
        n = len(dataset.det_id[seq])
        keep = list(range(0, n, stride))
        old_idet2iscan = dataset.idet2iscan[seq]
        dataset.det_id[seq] = [dataset.det_id[seq][i] for i in keep]
        dataset.det_wc[seq] = [dataset.det_wc[seq][i] for i in keep]
        dataset.det_wa[seq] = [dataset.det_wa[seq][i] for i in keep]
        dataset.det_wp[seq] = [dataset.det_wp[seq][i] for i in keep]
        dataset.idet2iscan[seq] = {new_i: old_idet2iscan[old_i]
                                    for new_i, old_i in enumerate(keep)}


def _load_dataset(args):
    """Return (dataset, cfg)."""
    if args.dataset == "frog":
        from follow_the_drow.datasets import FROG_Dataset, frog_laser_angles
        print(f"Loading FROG dataset (split='{args.split}') ...")
        ds = FROG_Dataset(split=args.split)
        cfg = SimpleNamespace(
            name="frog",
            angles_fn=frog_laser_angles,
            fov_min=FROG_Dataset.LASER_MIN_ANGLE,
            fov_max=FROG_Dataset.LASER_MAX_ANGLE,
            laser_inc=FROG_Dataset.LASER_INCREMENT,
        )
    elif args.dataset == "jrdb":
        from follow_the_drow.datasets import JRDB_Dataset, jrdb_laser_angles
        print(f"Loading JRDB dataset (split='{args.split}') ...")
        print("  NOTE: JRDB requires manual download — see JRDB_Dataset docstring.")
        ds = JRDB_Dataset(split=args.split)
        cfg = SimpleNamespace(
            name="jrdb",
            angles_fn=jrdb_laser_angles,
            fov_min=JRDB_Dataset.LASER_MIN_ANGLE,
            fov_max=JRDB_Dataset.LASER_MAX_ANGLE,
            laser_inc=JRDB_Dataset.LASER_INCREMENT,
        )
    else:
        from follow_the_drow.datasets import DROW_Dataset
        print("Loading DROW test set ...")
        ds = DROW_Dataset()
        cfg = SimpleNamespace(
            name="drow",
            angles_fn=laser_angles,
            fov_min=laser_minimum,
            fov_max=laser_maximum,
            laser_inc=laser_increment,
        )
    n_frames = sum(len(d) for d in ds.det_id)
    print(f"  {len(ds.scan_id)} sequence(s), {n_frames} annotated frames\n")

    if getattr(args, "eval_stride", 1) > 1:
        _subsample_dataset(ds, args.eval_stride)
        n_kept = sum(len(d) for d in ds.det_id)
        print(f"  Subsampled to every {args.eval_stride}-th frame: "
              f"{n_kept} frames evaluated (of {n_frames})\n")

    return ds, cfg


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset verification
# ──────────────────────────────────────────────────────────────────────────────

def verify_dataset(dataset, cfg):
    """Print FoV coverage and annotation-alignment statistics."""
    fov_min, fov_max = cfg.fov_min, cfg.fov_max
    angles_fn = cfg.angles_fn

    # --- FoV coverage ---
    counts = {k: {"total": 0, "in_fov": 0} for k in ("wc", "wa", "wp")}
    for seq in range(len(dataset.det_id)):
        for det in range(len(dataset.det_id[seq])):
            for label, anns in [("wc", dataset.det_wc[seq][det]),
                                 ("wa", dataset.det_wa[seq][det]),
                                 ("wp", dataset.det_wp[seq][det])]:
                for r, phi in anns:
                    counts[label]["total"] += 1
                    if fov_min <= phi <= fov_max:
                        counts[label]["in_fov"] += 1

    print("=== Annotation field-of-view coverage ===")
    print(f"  Laser FoV: {np.degrees(fov_min):.1f} deg ... "
          f"{np.degrees(fov_max):.1f} deg  "
          f"({np.degrees(fov_max - fov_min):.1f} deg total)")
    print(f"  {'Class':<6} {'Total':>6}  {'In-FoV':>8}  {'Out-of-FoV':>12}")
    print("  " + "-" * 38)
    for label, s in counts.items():
        n, k = s["total"], s["in_fov"]
        if n == 0:
            print(f"  {label:<6} {'0':>6}")
        else:
            print(f"  {label:<6} {n:>6}  {k/n:>7.1%}  {(n-k)/n:>11.1%}")
    print()

    # --- Annotation alignment ---
    dists = {"wc": [], "wa": [], "wp": []}
    for seq in range(len(dataset.det_id)):
        angles = angles_fn(dataset.scans[seq].shape[1])
        for det in range(len(dataset.det_id[seq])):
            iscan = dataset.idet2iscan[seq][det]
            scan  = dataset.scans[seq][iscan]
            pts   = np.stack([scan * -np.sin(angles),
                               scan *  np.cos(angles)], axis=1)
            for label, anns in [("wc", dataset.det_wc[seq][det]),
                                 ("wa", dataset.det_wa[seq][det]),
                                 ("wp", dataset.det_wp[seq][det])]:
                for r, phi in anns:
                    gx, gy = project_cartesian_from_polar(r, phi)
                    dists[label].append(
                        float(np.min(np.linalg.norm(pts - [[gx, gy]], axis=1))))

    print("=== Annotation -> nearest LiDAR point distance ===")
    print(f"  {'Class':<6} {'Count':>6}  {'Mean':>7}  {'Median':>7}"
          f"  {'<0.3m':>7}  {'<0.5m':>7}")
    print("  " + "-" * 52)
    for label, d_list in dists.items():
        if not d_list:
            print(f"  {label:<6} {'0':>6}")
            continue
        d = np.array(d_list)
        print(f"  {label:<6} {len(d):>6}  {d.mean():>6.3f}m  "
              f"{np.median(d):>6.3f}m  "
              f"{(d < 0.3).mean():>6.1%}  {(d < 0.5).mean():>6.1%}")
    print()


def _as_rp_array(x) -> np.ndarray:
    """Normalize one class's per-frame annotation list to an (N, 2) array.

    Per-class annotation containers are inconsistently typed across
    datasets (DROW: object-dtype arrays; FROG: plain Python lists, empty
    `[]` for the classes it doesn't have). Concatenating them with `+`
    silently switches from list concatenation to numpy elementwise
    addition whenever a plain list meets a numpy array, and crashes on
    shape mismatch — normalize first instead.
    """
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1, 2) if arr.size else np.empty((0, 2))


def _all_classes_rp(dataset, seq: int, det_idx: int) -> np.ndarray:
    """Concatenate wc+wa+wp (r, phi) annotations for one frame, robustly."""
    return np.concatenate([
        _as_rp_array(dataset.det_wc[seq][det_idx]),
        _as_rp_array(dataset.det_wa[seq][det_idx]),
        _as_rp_array(dataset.det_wp[seq][det_idx]),
    ], axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Algorithmic detector evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_algorithmic(dataset, cfg, eval_r: float = 0.5) -> dict:
    """Run AlgorithmicDetector on the full dataset; return precision/recall."""
    from scipy.spatial.distance import cdist

    print("Running AlgorithmicDetector (CPU) ...")
    total_gt = total_det = total_match = total_fp = 0

    for seq in range(len(dataset.det_id)):
        algo = AlgorithmicDetector(verbose=False)
        for det in range(len(dataset.det_id[seq])):
            iscan = dataset.idet2iscan[seq][det]
            scans, odoms = dataset.get_scan(seq, iscan, algo.time_frame)
            det_xy = np.array(algo.forward_one(scans[-1], odoms[-1]["xya"]))
            if det_xy.ndim == 1:
                det_xy = det_xy.reshape(-1, 2)
            if det_xy.size > 0:
                det_xy = np.column_stack(standard_cartesian_to_project_cartesian(det_xy[:, 0], det_xy[:, 1]))

            all_ann = _all_classes_rp(dataset, seq, det)
            gt_xy = (np.array([project_cartesian_from_polar(r, p) for r, p in all_ann])
                     if len(all_ann) else np.empty((0, 2)))

            n_gt  = len(gt_xy)
            n_det = len(det_xy)
            n_match = n_fp = 0
            if n_det > 0 and n_gt > 0:
                matched = np.min(cdist(det_xy, gt_xy), axis=1) < eval_r
                n_match = int(matched.sum())
                n_fp    = n_det - n_match
            elif n_det > 0:
                n_fp = n_det

            total_gt    += n_gt
            total_det   += n_det
            total_match += n_match
            total_fp    += n_fp

    recall    = total_match / total_gt  if total_gt  > 0 else float("nan")
    precision = total_match / total_det if total_det > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else float("nan"))

    print(f"  GT annotations : {total_gt}")
    print(f"  Detections     : {total_det}")
    print(f"  Near-GT hits   : {total_match}")
    print(f"  False positives: {total_fp}")
    print(f"  Recall         : {recall:.1%}")
    print(f"  Precision      : {precision:.1%}")
    print(f"  F1             : {f1:.1%}")
    print()

    return dict(recall=recall, precision=precision, f1=f1,
                total_gt=total_gt, total_det=total_det)


# ──────────────────────────────────────────────────────────────────────────────
# 3. NN model AUC evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_nn_model(det_name: str, weights_path: Path,
                  dataset, cfg, eval_r: float = 0.5,
                  batch_size: int = 16) -> dict:
    """Load checkpoint and compute AUC on CPU."""
    if det_name == "drspaam" and weights_path == DrSpaamDetector.DEFAULT_WEIGHTS:
        net = DrSpaamDetector.load_published()
    else:
        args = _default_args(detector=det_name, dataset=cfg.name)
        net  = _build_model(args)
        load_checkpoint(weights_path, net)
    net.eval()
    print(f"Evaluating {det_name} on CPU (eval batch_size={batch_size}) ...")
    aucs = evaluate_auc(net, dataset, cfg, eval_r=eval_r, device="cpu",
                         batch_size=batch_size)
    print(f"  Agnostic (any) : {aucs['agnostic']:.1%}")
    print(f"  Wheelchair(wc) : {aucs['wc']:.1%}")
    print(f"  Walker    (wa) : {aucs['wa']:.1%}")
    print(f"  Person    (wp) : {aucs['wp']:.1%}")
    print()
    return aucs


def eval_lfe_model(
    det_name:    str,
    detector,
    dataset,
    cfg,
    eval_r:      float = 0.5,
) -> dict:
    """
    Evaluate an LFE-style detector (raw scan input, ONNX) and compute AUC.

    LFE detectors operate on full raw scan vectors rather than cutouts, so
    they bypass the standard evaluate_auc pipeline and use their own loop —
    but they still route through the shared `_prec_rec_2d` PR-curve code
    every other model uses, via the same per-frame (score, x, y) + GT
    layout `_process_detections` builds. An earlier version of this function
    computed recall as (found true positives) / (found true positives) —
    i.e. against its own detections rather than against the total ground
    truth — which silently ignored frames where the detector found nothing
    at all near a real person, structurally inflating AUC for any detector
    that fires sparingly. `_prec_rec_2d` (below) counts every GT annotation,
    including ones with zero matching detections, as a false negative.

    Returns the same dict as eval_nn_model:
      {"agnostic": float, "wc": float, "wa": float, "wp": float}
    """
    n_beams = dataset.scans[0].shape[1]
    angles  = cfg.angles_fn(n_beams)

    det_scores_wp,  det_xy_wp,  det_frame_wp  = [], [], []
    det_scores_any, det_xy_any, det_frame_any = [], [], []
    gt_xy_wp,  gt_frame_wp,  gt_r_wp  = [], [], []
    gt_xy_any, gt_frame_any, gt_r_any = [], [], []

    fid = 0
    for seq in range(len(dataset.det_id)):
        for det_idx in range(len(dataset.det_id[seq])):
            iscan = dataset.idet2iscan[seq][det_idx]
            scan  = dataset.scans[seq][iscan]

            # Process this frame's GT even when detections is empty — a
            # frame with a real person and zero detections is a miss
            # (false negative), not a frame to skip.
            detections = detector.detect(scan, angles)
            for score, x, y in detections:
                det_scores_wp.append(score);  det_xy_wp.append((x, y));  det_frame_wp.append(fid)
                det_scores_any.append(score); det_xy_any.append((x, y)); det_frame_any.append(fid)

            for r, p in _as_rp_array(dataset.det_wp[seq][det_idx]):
                gx, gy = project_cartesian_from_polar(r, p)
                gt_xy_wp.append((gx, gy)); gt_frame_wp.append(fid); gt_r_wp.append(eval_r)

            for r, p in _all_classes_rp(dataset, seq, det_idx):
                gx, gy = project_cartesian_from_polar(r, p)
                gt_xy_any.append((gx, gy)); gt_frame_any.append(fid); gt_r_any.append(eval_r)

            fid += 1

    def _auc(det_scores, det_xy, det_frame, gt_xy, gt_frame, gt_r):
        if not det_scores or not gt_xy:
            return float("nan")
        recs, precs, _ = _prec_rec_2d(
            np.array(det_scores, dtype=np.float32),
            np.array(det_xy, dtype=np.float32),
            np.array(det_frame),
            np.array(gt_xy, dtype=np.float32),
            np.array(gt_frame),
            np.array(gt_r, dtype=np.float32),
        )
        return _safe_auc(recs, precs)

    aucs = {
        "agnostic": _auc(det_scores_any, det_xy_any, det_frame_any,
                          gt_xy_any, gt_frame_any, gt_r_any),
        "wc":       float("nan"),   # LFE is class-agnostic
        "wa":       float("nan"),
        "wp":       _auc(det_scores_wp, det_xy_wp, det_frame_wp,
                          gt_xy_wp, gt_frame_wp, gt_r_wp),
    }

    print(f"Evaluating {det_name} (ONNX) on CPU ...")
    print(f"  Agnostic (any) : {aucs['agnostic']:.1%}")
    print(f"  Person    (wp) : {aucs['wp']:.1%}  (wc/wa: n/a — class-agnostic)")
    print()
    return aucs


# ──────────────────────────────────────────────────────────────────────────────
# 4. Throughput benchmark (CPU only)
# ──────────────────────────────────────────────────────────────────────────────

def _fake_scans(n_beams: int, T: int):
    scans  = np.random.uniform(0.5, 10.0, (T, n_beams)).astype(np.float32)
    odoms  = [{"xya": np.array([0.0, 0.0, i * 0.01], dtype=np.float32)}
               for i in range(T)]
    return scans, odoms


def _fake_input(input_mode: str, n_beams: int, T: int, n_samp: int = 48) -> torch.Tensor:
    if input_mode == "cutout":
        return torch.randn(n_beams, T, n_samp)
    return torch.randn(n_beams, T, 1)


def _fake_targets(n_beams: int):
    return (torch.randint(0, 4, (n_beams,)),
            torch.randn(n_beams, 2))


def _time_fn(fn, warmup: int, iters: int) -> Tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(times[2:])
    return float(arr.mean()), float(arr.std())


def bench_preprocessing(n_beams: int, T: int,
                        warmup: int, iters: int) -> Dict[str, Tuple[float, float]]:
    scans, odoms = _fake_scans(n_beams, T)
    c_m, c_s = _time_fn(
        lambda: cutout(scans, odoms, n_beams, nsamp=48),
        warmup, iters)
    r_m, r_s = _time_fn(
        lambda: raw_scan(scans),
        warmup, iters)
    return {"cutout": (c_m, c_s), "raw_scan": (r_m, r_s)}


def bench_model(model: torch.nn.Module, input_mode: str,
                n_beams: int, T: int,
                warmup: int, iters: int) -> Dict[str, Tuple[float, float]]:
    """Benchmark one model on CPU (eval + train pass)."""
    model = model.to("cpu")
    n_samp = getattr(model, "N_SAMP", 48)
    x = _fake_input(input_mode, n_beams, T, n_samp)
    labels, vote_tgts = _fake_targets(n_beams)
    opt = _make_optimizer(model.parameters(), lr=1e-3,
                          weight_decay=1e-4, using_dml=False)

    # heatmap-head models (e.g. TemporalUNetDetector with head="heatmap")
    # return a 1-tuple (heatmap,), not the (logits, vpred) 2-tuple every
    # other head returns — branch on it instead of assuming 2 outputs.
    _head_type = getattr(model, "head_type", "drow")

    model.eval()
    def run_eval():
        with torch.no_grad():
            out = model(x)
        _ = out[0].reshape(-1)[0].item()

    e_m, e_s = _time_fn(run_eval, warmup, iters)

    model.train()
    def run_train():
        out = model(x)
        if _head_type == "heatmap":
            heatmap = out[0]
            loss = F.binary_cross_entropy_with_logits(
                heatmap.reshape(-1), (labels > 0).float())
        else:
            logits, vpred = out
            pos  = labels > 0
            lv   = (F.mse_loss(vpred[pos], vote_tgts[pos])
                    if pos.any() else vpred.new_tensor(0.0))
            if logits.shape[-1] == 1:
                # Binary logit output (e.g. Li2Former) — use BCE
                lc = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), (labels > 0).float())
            else:
                lc = F.cross_entropy(logits, labels)
            loss = lc + 0.02 * lv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        _ = loss.item()

    t_m, t_s = _time_fn(run_train, warmup, iters)
    return {"eval": (e_m, e_s), "train": (t_m, t_s)}


def _build_bench_models(T: int) -> List[Tuple[str, torch.nn.Module, str]]:
    return [
        ("DrowDetector",
            DrowDetector(dropout=0.5, time_frame_size=T, verbose=False),
            "cutout"),
        ("DrSpaamDetector",
            DrSpaamDetector(dropout=0.5, num_scans=T),
            "cutout"),
        ("Li2FormerDetector",
            Li2FormerDetector(dropout=0.5, num_scans=T),
            "cutout"),
        ("SpaceTimeCNN",
            SpaceTimeCNNDetector(n_time=T),
            "raw_scan"),
        ("FullScanTCN",
            FullScanTCNDetector(n_time=T),
            "raw_scan"),
        ("TemporalUNet",
            TemporalUNetDetector(n_time=T, head="heatmap"),
            "raw_scan"),
    ]


def run_benchmark(args):
    N  = args.n_beams
    T  = args.time_frame
    WU = args.warmup
    IT = args.iters

    print(f"Config: N_beams={N}, T={T}, warmup={WU}, iters={IT}, device=CPU\n")

    # Preprocessing
    print("=" * 60)
    print("  PREPROCESSING  (CPU)")
    print("=" * 60)
    prep = bench_preprocessing(N, T, WU, IT)
    for name, (m, s) in prep.items():
        fps = 1000.0 / m if m > 0 else 0
        print(f"  {name:<22} {m:6.1f} +-{s:4.1f} ms   {fps:6.0f} sc/s")
    print()

    models = _build_bench_models(T)
    col_n  = 24

    for mode_label, mode_key in [("EVAL  (no_grad forward)", "eval"),
                                  ("TRAIN (fwd + bwd + opt.step)", "train")]:
        print("=" * 60)
        print(f"  {mode_label}")
        print("=" * 60)
        print(f"  {'Model':<{col_n}}  {'ms/scan':>12}  {'sc/s':>8}")
        print("  " + "-" * 48)
        for name, model, input_mode in models:
            try:
                result = bench_model(model, input_mode, N, T, WU, IT)
                m, s   = result[mode_key]
                fps    = 1000.0 / m if m > 0 else 0
                print(f"  {name:<{col_n}}  {m:6.1f} +-{s:4.1f} ms  {fps:6.0f} sc/s")
            except Exception as exc:
                print(f"  {name:<{col_n}}  ERROR: {exc}")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def _pct(v):
    return f"{v:.1%}" if not np.isnan(v) else "  n/a  "


def print_summary(algo_stats: Optional[dict],
                  nn_results: Dict[str, dict]):
    if not algo_stats and not nn_results:
        return
    print("=" * 72)
    print("  EVALUATION SUMMARY")
    print("=" * 72)
    col = 26
    print(f"  {'Model':<{col}}  {'AUC(any)':>9}  {'AUC(wc)':>8}"
          f"  {'AUC(wa)':>8}  {'AUC(wp)':>8}")
    print("  " + "-" * 68)

    if algo_stats:
        rec  = _pct(algo_stats["recall"])
        prec = _pct(algo_stats["precision"])
        f1   = _pct(algo_stats["f1"])
        print(f"  {'AlgorithmicDetector':<{col}}  {'n/a':>9}  {'n/a':>8}"
              f"  {'n/a':>8}  {'n/a':>8}")
        print(f"    recall={rec}  prec={prec}  F1={f1}")

    for name, aucs in nn_results.items():
        print(f"  {name:<{col}}  "
              f"{_pct(aucs['agnostic']):>9}  "
              f"{_pct(aucs['wc']):>8}  "
              f"{_pct(aucs['wa']):>8}  "
              f"{_pct(aucs['wp']):>8}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

# PyTorch cutout/full-scan models evaluated via evaluate_auc
_NN_MODELS = {
    "drow":          "drow",
    "drspaam":       "drspaam",
    "li2former":     "li2former",
    "spacetime-cnn": "spacetime_cnn",
    "fullscan-tcn":  "fullscan_tcn",
    "temporal-unet": "temporal_unet",
}

# ONNX-based models evaluated via eval_lfe_model
_LFE_MODELS = {
    "lfe-peaks": ("lfe_peaks", LFEPeaksDetector),
    "lfe-ppn":   ("lfe_ppn",   LFEPPNDetector),
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation + benchmark for all person detectors (CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset
    parser.add_argument("--dataset", choices=["drow", "frog", "jrdb"], default="drow",
                        help="Dataset to evaluate on (default: drow)")
    parser.add_argument("--split",   choices=["test", "train", "val"],
                        default="test",
                        help="Dataset split (default: test; DROW always uses test)")
    parser.add_argument("--eval-r",  type=float, default=0.5,
                        help="Detection radius for AUC / precision-recall (default: 0.5 m)")
    parser.add_argument("--eval-stride", type=int, default=1,
                        help="Evaluate every Nth annotated frame instead of all of "
                             "them (systematic subsample, default: 1 = no "
                             "subsampling). Recommended for FROG (every scan "
                             "annotated at 40 Hz, so adjacent frames are highly "
                             "correlated near-duplicates for AP purposes) — leave "
                             "at 1 for DROW, which is already sparsely labeled "
                             "(~5% of scans) in the source data.")
    parser.add_argument("--eval-batch-size", type=int, default=16,
                        help="Forward-pass batch size for the AUC evaluation loop "
                             "(default: 16). evaluate_auc()'s own default is 1 "
                             "(single-frame), which is fine for DROW's smaller test "
                             "split but impractically slow on FROG's ~120k-frame "
                             "test split (~3 fr/s at batch_size=1, ~11h/model).")

    # What to run
    parser.add_argument("--verify",  action="store_true",
                        help="Print dataset verification stats (FoV + alignment)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip all evaluation (dataset + models)")
    parser.add_argument("--no-bench", action="store_true",
                        help="Skip throughput benchmark")

    # NN model checkpoints
    # --drow and --drspaam accept an optional path; when given without a path
    # they use the bundled published paper weights (downloaded at pip install time).
    parser.add_argument("--drow", nargs="?", type=Path,
                        const=DrowDetector.DEFAULT_WEIGHTS, default=None,
                        metavar="WEIGHTS",
                        help="Checkpoint for drow; omit value to use bundled paper weights")
    parser.add_argument("--drspaam", nargs="?", type=Path,
                        const=DrSpaamDetector.DEFAULT_WEIGHTS, default=None,
                        metavar="WEIGHTS",
                        help="Checkpoint for drspaam; omit value to use bundled paper weights")
    for flag in [f for f in _NN_MODELS if f not in ("drow", "drspaam")]:
        parser.add_argument(f"--{flag}", type=Path, default=None, metavar="WEIGHTS",
                            help=f"Checkpoint for {flag} (enables AUC evaluation)")

    # LFE ONNX models — accept optional path; default to bundled ONNX weights
    parser.add_argument("--lfe-peaks", nargs="?", type=Path,
                        const=LFEPeaksDetector.DEFAULT_WEIGHTS, default=None,
                        metavar="ONNX",
                        help="LFE-Peaks ONNX model; omit value to use bundled weights")
    parser.add_argument("--lfe-ppn", nargs="?", type=Path,
                        const=LFEPPNDetector.DEFAULT_WEIGHTS, default=None,
                        metavar="ONNX",
                        help="LFE-PPN ONNX model; omit value to use bundled weights")

    # Benchmark options
    parser.add_argument("--n-beams",    type=int, default=450,
                        help="Beams per scan for benchmark (450=DROW, 720=FROG; default: 450)")
    parser.add_argument("--time-frame", type=int, default=5,
                        help="Temporal window T for benchmark (default: 5)")
    parser.add_argument("--warmup",     type=int, default=5,
                        help="Warm-up iterations before timing (default: 5)")
    parser.add_argument("--iters",      type=int, default=20,
                        help="Timed iterations per model (default: 20)")

    args = parser.parse_args()

    # Collect which PyTorch NN models to evaluate
    nn_weights = {
        _NN_MODELS[flag]: getattr(args, flag.replace("-", "_"))
        for flag in _NN_MODELS
        if getattr(args, flag.replace("-", "_")) is not None
    }

    # Collect which LFE ONNX models to evaluate
    lfe_weights = {
        det_name: (cls, getattr(args, flag.replace("-", "_")))
        for flag, (det_name, cls) in _LFE_MODELS.items()
        if getattr(args, flag.replace("-", "_")) is not None
    }

    run_eval  = not args.no_eval
    run_bench = not args.no_bench

    # ── Load dataset (only if needed) ─────────────────────────────────────────
    dataset = cfg = None
    if run_eval:
        dataset, cfg = _load_dataset(args)

    algo_stats = None
    nn_results = {}

    if run_eval and dataset is not None:
        # 1. Dataset verification
        if args.verify:
            print("=== Dataset Verification ===\n")
            verify_dataset(dataset, cfg)

        # 2. Algorithmic detector
        print("=== AlgorithmicDetector ===\n")
        algo_stats = eval_algorithmic(dataset, cfg, eval_r=args.eval_r)

        # 3. NN models (cutout / full-scan, PyTorch)
        for det_name, weights_path in nn_weights.items():
            print(f"=== {det_name} ===\n")
            try:
                nn_results[det_name] = eval_nn_model(
                    det_name, weights_path, dataset, cfg, eval_r=args.eval_r,
                    batch_size=args.eval_batch_size)
            except Exception as exc:
                print(f"  [ERROR] {exc}\n")

        # 4. LFE ONNX models
        for det_name, (cls, onnx_path) in lfe_weights.items():
            print(f"=== {det_name} ===\n")
            try:
                detector = cls(onnx_path=onnx_path)
                nn_results[det_name] = eval_lfe_model(
                    det_name, detector, dataset, cfg, eval_r=args.eval_r)
            except Exception as exc:
                print(f"  [ERROR] {exc}\n")

        # Summary
        print_summary(algo_stats, nn_results)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    if run_bench:
        print("=== Throughput Benchmark (CPU) ===\n")
        run_benchmark(args)


if __name__ == "__main__":
    main()
