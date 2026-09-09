# Commands

*keywords:* commands, invocation, flags, entrypoint, train.py, evaluate.py, render_video.py, pytest, make

Every routine invocation, with the flags actually used.
[`../README.md`](../README.md) carries the short list for humans; this is the full reference agents copy from.

## Prelude

```bash
.venv312/Scripts/activate                 # Windows; or call .venv312/Scripts/python.exe directly
pip install ./library                     # installs the package, downloads ~2 GB of bundled datasets/weights on first run
pip install -r utils/requirements.txt     # research-script dependencies (tqdm, pytorch-lightning, scikit-learn, pytest)
```

Do not use bare `python3` or `make venv` — see `gotchas.md`.

## Install and build

```bash
pip install ./library                                       # library + downloads (DROW, FROG, bundled DROW/DR-SPAAM/LFE weights)
make build-lib                                               # cmake + make in library/cpp_core/build; installs system-wide only if run as root
python -c "import follow_the_drow; from follow_the_drow.detectors import AlgorithmicDetector; print('OK')"   # confirms the C++ extension built
```

## Training (`utils/train.py`, run from `utils/`)

```bash
python train.py --detector spacetime_cnn --dataset frog --epochs 30 --patience 5   # trains one detector
python train_all.py --dataset frog --epochs 30 --patience 5 --out-dir ../checkpoints   # spacetime_cnn, fullscan_tcn, temporal_unet, li2former, in sequence
python train.py --detector temporal_unet --dataset frog --resume ../checkpoints/temporal_unet.pth --epochs 30   # resume an interrupted run
python train.py --detector spacetime_cnn --dataset frog --init-weights ../checkpoints/spacetime_cnn.best.pth --lr 1e-4   # fine-tune from another checkpoint's weights only (fresh optimiser/epoch count, unlike --resume)
```

`--detector` choices: `algorithmic` (eval only, not trainable), `spacetime_cnn`, `fullscan_tcn`, `temporal_unet`, `li2former`.
`--dataset {drow,frog,jrdb}`, `--dtime N` (temporal-stride ablation), `--no-align-scans` (disable odometry-based rotation correction), `--no-frame-cache` (trade time for ~2.6 GB less RAM on FROG's full train split).
Run `python train.py --help` for the full, current flag list — it changes more often than this file does.

## Evaluation (`utils/evaluate.py`, run from `utils/`)

```bash
python evaluate.py --dataset drow --drow --drspaam                                   # published weights, no training needed
python evaluate.py --dataset frog --lfe-peaks --lfe-ppn                              # bundled ONNX weights
python evaluate.py --dataset frog --split test --spacetime-cnn ../checkpoints/spacetime_cnn.best.pth --bench
python evaluate.py --dataset drow --verify --no-bench                                # dataset FoV/annotation sanity stats only, no models
python evaluate.py --bench --no-eval --n-beams 720 --time-frame 5                     # synthetic-data speed benchmark, no dataset needed
```

`--eval-batch-size N` (verified numerically equivalent to `batch_size=1`, much faster) and `--eval-stride N` (systematic frame subsample — see `interpreting-evaluation.md` before trusting a strided FROG number) both cut a full-test-split run from hours to minutes.
`evaluate.py` flags mirror the detector registry: `--drow [WEIGHTS]`, `--drspaam [WEIGHTS]`, `--spacetime-cnn WEIGHTS`, `--fullscan-tcn WEIGHTS`, `--temporal-unet WEIGHTS`, `--li2former WEIGHTS`, `--lfe-peaks [WEIGHTS]`, `--lfe-ppn [WEIGHTS]` (omit `WEIGHTS` to use bundled/published weights where available).
Run `python evaluate.py --help` for the full, current flag list.

## Rendering

```bash
python render_video.py --dataset frog                                    # MP4, native annotation-rate playback
python render_video.py --dataset drow --split train --seq 0 --fps 5
python render_video.py --no-algo --drow weights_drow.pth --max-frames 300
```

## Odometry (`utils/generate_frog_odometry.py`, run from `utils/`)

```bash
python generate_frog_odometry.py                                # writes <h5-stem>_odom.npz sidecars via ICP scan-matching, for any session that has no real official odometry file
python generate_frog_odometry.py --inlier-thresh 0.15 --min-inliers 10
```

Only needed as a fallback — `FROG_Dataset.download()` fetches the real, official per-session odometry first; see `data-model.md`.

## Checks

```bash
.venv312/Scripts/python.exe -m pytest tests -q            # 63 tests, ~6s, CPU-only, no GPU or dataset needed — run after every edit
python memory/scripts/verify_memory.py                    # docs: links, indexes, markdown rules
python memory/scripts/verify_memory.py --strict            # same, but style warnings fail the run too
```

There is no typecheck/parse step faster than the test suite itself — 6 seconds already is the cheap check here.
The test suite runs by hand only; there is no CI job invoking `pytest` yet (`TODO.md`).

## Linting

```bash
ruff check --config memory/scripts/ruff.toml memory/scripts/   # the automation scripts in memory/scripts/ only
actionlint                                                       # workflows, plus shellcheck on run: blocks
npx markdownlint-cli2 "**/*.md"                                  # markdown, if Node is available
```

No linter is configured yet for this project's own Python (`library/`, `utils/`), C++, or the `entrypoint.sh`/`Dockerfile` under `deploy/docker/` — see `coding-guidelines.md` and `TODO.md`.

## Commands whose output is easy to misread

- **`evaluate.py`'s AUC number without checking which regime produced it.** The same "DROW" column can mean a published-weights zero-shot transfer from FROG, a from-scratch DROW-native retrain, or a fine-tuned-from-FROG checkpoint — three different numbers that have all appeared for the same architecture.
  Check `interpreting-evaluation.md` before quoting one.
- **`train_all.py`'s printed summary table** — historically mislabeled the agnostic (any-class) AUC column as "Test AUC" instead of person-class `wp` AUC; confirm which column a number came from before repeating it (see `AGENTS.md` rule 2b).
- **A `--eval-stride N` run's timing** looks like a full-dataset evaluation but only touched `1/N` of the frames — real for a quick check, not directly comparable to a `stride=1` number without accounting for it.

## Platform notes

- Windows Git Bash: redirect long-running command output to a file and read the file, never `tail`/truncate it unread (`AGENTS.md` I2) — this project's training/evaluation logs are routinely tens of MB and hours long.
- `make` targets assume a POSIX shell (`SHELL = /bin/bash` in the Makefile) — run them from Git Bash, not PowerShell directly.
