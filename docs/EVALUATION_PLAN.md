# Training & Evaluation Plan

> A self-contained, sequential runbook for training the models this repo has no published weights for, evaluating every applicable (detector × dataset) pair, benchmarking inference speed, and filling in the `TBD` cells in [docs/RESEARCH.md](RESEARCH.md) §8.1/§8.2. Written to be handed to a dispatch job or run unattended — every step states its exact command, its success criterion, and what to do if it fails. No prior conversation context is assumed.

## 0. What this plan produces

Two filled-in tables in `docs/RESEARCH.md`:
- §8.1 — Average Precision per class, per detector, per dataset (DROW / FROG / JRDB)
- §8.2 — ms/frame inference speed, per detector, CPU (and GPU if available)

Plus four trained checkpoints in `checkpoints/` (gitignored — these stay local, they are not committed).

## 1. Prerequisites

Run these once, in order. Do not proceed to Phase 1 until every check below passes.

```bash
cd utils
pip install -e ../library                # installs the package + downloads bundled DROW/DR-SPAAM/LFE weights + FROG dataset (~2 GB)
pip install -r requirements.txt

# Sanity check: the package must import cleanly, including the compiled C++ binding.
python -c "import follow_the_drow; from follow_the_drow.detectors import AlgorithmicDetector; print('OK')"
```

**If the import check fails with `ModuleNotFoundError: No module named 'follow_the_drow.cpp_binding'`**: the C++ pybind11 extension didn't build. Run `make build-lib` from the repo root, or re-run `pip install -e ../library` — the extension is built automatically by `Pybind11Extension` in `library/setup.py`. Do not proceed until this import succeeds; every phase below depends on it.

**GPU check (optional but strongly recommended for Phase 1 — see time estimates below):**

```bash
python -c "from follow_the_drow.utils.torch_utils import detect_device; print(detect_device())"
```

Reports `(cuda, False)`, `(privateuseone, True)` (DirectML), or `(cpu, False)`. Training on CPU works but is slow — see the time estimates in Phase 1. If on Windows with an AMD/Intel/NVIDIA GPU and no CUDA, install DirectML: `pip install torch-directml`. For NVIDIA GPUs on Windows, training inside WSL2 gives full CUDA throughput instead of DirectML — see the README's GPU section if that applies.

**JRDB check (determines whether the JRDB column is filled or marked `n/a`):**

```bash
ls ../library/follow_the_drow/include/JRDB-data 2>/dev/null && echo "JRDB present" || echo "JRDB absent — JRDB column will be n/a"
```

JRDB requires free manual registration at jrdb.erc.monash.edu — it is not auto-downloaded. **If absent, do not block on it.** Skip every JRDB-dataset step below and mark that column `n/a` in the final tables, consistent with the existing `n/a` footnote convention in docs/RESEARCH.md §8.1.

## 2. Applicability matrix

Which (detector, dataset) pairs to actually run — do not run combinations marked "skip," they either don't apply or aren't part of the primary comparison.

| Detector | Needs training? | DROW | FROG | JRDB |
|---|---|---|---|---|
| `algorithmic` | No | run | run | run (if present) |
| `drow` | No — bundled weights | run | run | run (if present) |
| `drspaam` | No — bundled weights | run | run | run (if present) |
| `li2former` | **Yes — once, on FROG** | run | run | run (if present) |
| `lfe_peaks` | No — bundled weights | *skip (see note)* | run | *skip (see note)* |
| `lfe_ppn` | No — bundled weights | *skip (see note)* | run | *skip (see note)* |
| `spacetime_cnn` | **Yes — once, on FROG** | run | run | run (if present) |
| `fullscan_tcn` | **Yes — once, on FROG** | run | run | run (if present) |
| `temporal_unet` | **Yes — once, on FROG** | run | run | run (if present) |

**LFE on DROW/JRDB note:** `LFEPeaksDetector`/`LFEPPNDetector` were trained on FROG's 720-beam, 0.25°/beam geometry. Running them on DROW (450 beams, 0.5°/beam) or JRDB (541 beams) requires zero-padding that does not correct for the different angular resolution — this repo's own known-inexact extrapolation (docs/RESEARCH.md §2.1.1/§3.4). **Skip these two cells for the primary table.** If you want them anyway for completeness, run them and record the numbers in RESEARCH.md's §8.1 footnote area (already reserved for "degraded cross-dataset" numbers), not the primary table cells — do not present them as directly comparable to the other rows.

**Training dataset choice:** all four trainable models are trained **once, on FROG only** (densest annotations, no sparse-label noise, already cached locally) — never per-dataset. The resulting single checkpoint per model is then *evaluated* across all three datasets (cross-dataset transfer), the same way LFE already is. This is the recommendation already made in docs/RESEARCH.md §8.4; this plan implements it.

## 3. Phase 1 — Train the four models with no published weights

One command trains all four, sequentially, with early stopping:

```bash
cd utils
python train_all.py --dataset frog --epochs 30 --patience 5 --out-dir ../checkpoints 2>&1 | tee ../checkpoints/train_all.log
```

This trains `spacetime_cnn`, `fullscan_tcn`, `temporal_unet`, `li2former` in that order (in this repo's fixed canonical order — `temporal_unet` was added to `train_all.py`'s detector list as part of the fix that made this plan possible; if you're running against an older checkout where `train_all.py`'s `_ALL_DETECTORS` doesn't include `"temporal_unet"`, pull latest or add it before running). Each model trains in its own subprocess (`train_all.py`'s existing design — avoids cross-model memory compounding), and `temporal_unet` automatically gets its documented default head (`heatmap`) — you do not need to pass `--head` explicitly for this run.

**Time estimate:** FROG's train split is ≈20k frames. On a discrete GPU (CUDA or DirectML), expect roughly 5–20 minutes per model per epoch depending on hardware, so 30 epochs × 4 models could be several hours; early stopping (`--patience 5`) typically cuts this well short of 30 epochs once validation AUC plateaus. On CPU-only, this could take a full day or more — strongly prefer a GPU for this phase. If you only have a narrow dispatch time window, run a smoke test first to catch config errors cheaply before committing to the full run:

```bash
python train_all.py --dataset frog --epochs 3 --subsample 0.05 --out-dir /tmp/smoke_test
```

**Success criterion:** `train_all.py`'s printed summary table (also in `checkpoints/train_all.log`) has no `FAILED` rows, and these four files exist:

```
checkpoints/spacetime_cnn.best.pth
checkpoints/fullscan_tcn.best.pth
checkpoints/temporal_unet.best.pth
checkpoints/li2former.best.pth
```

(`train_all.py` also writes `checkpoints/<detector>.pth`, the final-epoch weights — use the `.best.pth` files for evaluation, they're the best-validation-AUC checkpoint from early stopping.)

**If a specific model fails (OOM, crash, etc.):** retrain it individually rather than re-running the whole batch:

```bash
# Example: retrying just fullscan_tcn, with RAM-constrained settings
python train.py --detector fullscan_tcn --dataset frog --epochs 30 --patience 5 \
    --lr-schedule plateau --auc-every 5 --out ../checkpoints/fullscan_tcn.pth \
    --no-frame-cache   # trades training time for ~2.6 GB less RAM if this is what OOM'd
```

**If interrupted partway through a model** (dispatch timeout, etc.): resume from the last checkpoint rather than restarting that model from scratch:

```bash
python train.py --detector temporal_unet --dataset frog --resume ../checkpoints/temporal_unet.pth --epochs 30 --patience 5 --out ../checkpoints/temporal_unet.pth
```

## 4. Phase 2 — Accuracy evaluation

Run once per dataset. Each command evaluates every applicable detector on that dataset in one pass and prints a summary table (`evaluate.py`'s `print_summary`). `--verify` additionally prints dataset FoV/annotation sanity stats, useful to sanity-check nothing is misconfigured before trusting the AUC numbers.

```bash
cd utils
mkdir -p ../results

# DROW
python evaluate.py --dataset drow --split test --verify --no-bench \
    --drow --drspaam \
    --spacetime-cnn ../checkpoints/spacetime_cnn.best.pth \
    --fullscan-tcn ../checkpoints/fullscan_tcn.best.pth \
    --temporal-unet ../checkpoints/temporal_unet.best.pth \
    --li2former ../checkpoints/li2former.best.pth \
    2>&1 | tee ../results/eval_drow.log

# FROG (also includes LFE — trained on this exact dataset)
python evaluate.py --dataset frog --split test --verify --no-bench \
    --drow --drspaam --lfe-peaks --lfe-ppn \
    --spacetime-cnn ../checkpoints/spacetime_cnn.best.pth \
    --fullscan-tcn ../checkpoints/fullscan_tcn.best.pth \
    --temporal-unet ../checkpoints/temporal_unet.best.pth \
    --li2former ../checkpoints/li2former.best.pth \
    2>&1 | tee ../results/eval_frog.log

# JRDB — only if the Prerequisites check found JRDB-data present
python evaluate.py --dataset jrdb --split test --verify --no-bench \
    --drow --drspaam \
    --spacetime-cnn ../checkpoints/spacetime_cnn.best.pth \
    --fullscan-tcn ../checkpoints/fullscan_tcn.best.pth \
    --temporal-unet ../checkpoints/temporal_unet.best.pth \
    --li2former ../checkpoints/li2former.best.pth \
    2>&1 | tee ../results/eval_jrdb.log
```

**Success criterion:** each log ends with a populated `EVALUATION SUMMARY` table (from `print_summary`) with an AUC percentage in every column for every detector passed — not an `ERROR` line. `AlgorithmicDetector` reports precision/recall/F1 instead of AUC (it has no confidence scores) — that's expected, not a failure.

**If a detector's AUC evaluation errors out:** the most common cause is a `--head` mismatch for `temporal_unet` if you trained with an explicit non-default `--head drow` in Phase 1 — the evaluation call will auto-resolve to `heatmap` by default (matching the training default), so only worry about this if Phase 1 was run with an explicit override. Otherwise, check the checkpoint path is correct and the file isn't from a failed/partial training run (Phase 1's success criterion).

## 5. Phase 3 — Efficiency benchmark

`evaluate.py --bench` times forward (eval mode) and forward+backward (train mode) passes on synthetic data — no dataset needed, so this is independent of Phase 2 and can run in parallel with it if you have spare capacity. Run once per dataset's native beam count, since receptive-field-relative cost scales with `N_beams`:

```bash
cd utils
python evaluate.py --bench --no-eval --n-beams 450 --time-frame 5 2>&1 | tee ../results/bench_drow_450beams.log   # DROW scale
python evaluate.py --bench --no-eval --n-beams 720 --time-frame 5 2>&1 | tee ../results/bench_frog_720beams.log   # FROG scale
python evaluate.py --bench --no-eval --n-beams 541 --time-frame 5 2>&1 | tee ../results/bench_jrdb_541beams.log   # JRDB scale — run regardless of whether JRDB data is present, this is synthetic
```

This covers **6 of the 8 rows** in RESEARCH.md §8.2: `DrowDetector`, `DrSpaamDetector`, `Li2FormerDetector`, `SpaceTimeCNN`, `FullScanTCN`, `TemporalUNet`. It does **not** cover `AlgorithmicDetector` (C++ rule-based, not a `torch.nn.Module` — architecturally outside this harness) or `LFEPeaksDetector`/`LFEPPNDetector` (ONNX Runtime inference, also not a `torch.nn.Module`). For those two rows, either leave them `TBD` in the final table (acceptable — the primary comparison this research makes is among the CNN/TCN architectures, not against the classical/ONNX baselines' raw speed), or time them manually:

```python
# Optional manual timing — AlgorithmicDetector, run from utils/ with the package installed
import time
from follow_the_drow.detectors import AlgorithmicDetector
from follow_the_drow.datasets import FROG_Dataset

ds = FROG_Dataset(split="test")
algo = AlgorithmicDetector(verbose=False)
scans_hist, odoms_hist = ds.get_scan(0, ds.idet2iscan[0][0], ds.time_frame)
# warm-up
for _ in range(5):
    algo.forward_one(scans_hist[-1], odoms_hist[-1]["xya"])
t0 = time.perf_counter()
N = 100
for _ in range(N):
    algo.forward_one(scans_hist[-1], odoms_hist[-1]["xya"])
print(f"{(time.perf_counter() - t0) / N * 1000:.2f} ms/frame")
```

```python
# Optional manual timing — LFE-Peaks / LFE-PPN, run from utils/
import time
from follow_the_drow.detectors import LFEPeaksDetector
import numpy as np

det = LFEPeaksDetector()
scan = np.random.uniform(0.2, 10.0, 720).astype("float32")
angles = np.linspace(-np.pi/2, np.pi/2, 720)
for _ in range(5):
    det.detect(scan, angles)   # warm-up
t0 = time.perf_counter()
N = 100
for _ in range(N):
    det.detect(scan, angles)
print(f"{(time.perf_counter() - t0) / N * 1000:.2f} ms/frame")
```

**Success criterion:** each `--bench` log has a `ms/scan` value (not `ERROR:`) for all six covered models, in both the `EVAL` and `TRAIN` sections.

## 6. Phase 4 — Transcribe results into docs/RESEARCH.md

Replace the `TBD` cells in [docs/RESEARCH.md](RESEARCH.md):

- **§8.1** (accuracy): from each `results/eval_<dataset>.log`'s summary table, take the `wp` (person-class) AUC/precision-recall column — that's what every other row in the existing table already reports (person class, 0.5 m radius). Leave a cell `n/a` only where the applicability matrix in Section 2 above says `skip`.
- **§8.2** (efficiency): from `results/bench_<dataset>_*beams.log`, use the `EVAL` section's `ms/scan` (that's inference-time cost — the number relevant to a deployed robot, not the `TRAIN` section's forward+backward number, which measures training cost instead).
- Update the footnote under §8.1 that currently reads "train once²" — once checkpoints exist, change it to state the actual FROG-trained-then-cross-evaluated numbers are now filled in.
- Update §8.4's "not yet written" framing — the orchestration described in that section is exactly this plan; replace the paragraph describing the not-yet-built script with a pointer to this file (`docs/EVALUATION_PLAN.md`) and a note that it's been run (with the date).

## 7. Re-running / partial re-runs

Every phase is independent and re-runnable on its own:
- Phase 1 is idempotent per model — re-running `train_all.py` retrains everything; to redo just one model, use the individual `train.py --detector X` command in Phase 1's troubleshooting note.
- Phase 2 is a pure read of existing checkpoints — safe to re-run anytime, e.g. after fixing one model's checkpoint, without repeating the others.
- Phase 3 doesn't depend on checkpoints at all (synthetic data, freshly-initialized weights) — safe to run before, during, or after the other phases.

If this plan is dispatched with a time limit and doesn't complete: whatever `checkpoints/*.best.pth` and `results/*.log` files exist are valid, resumable progress. A follow-up run should skip Phase 1 for any model whose `.best.pth` already exists (or explicitly retrain it if you suspect it's stale/corrupt), then continue from Phase 2.
