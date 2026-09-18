# Commands

*keywords:* commands, invocation, flags, entrypoint, train.py, evaluate.py, render_video.py, pytest, make

Every routine invocation, with the flags actually used.
[`../README.md`](../README.md) carries the short list for humans; this is the full reference agents copy from.

## Prelude

**`.venv` is the one environment** (Python 3.12, PyTorch 2.9.1 on AMD ROCm 7.2.1), built from the root `requirements.txt`, whose header carries the full recipe:

```bash
py -3.12 -m venv .venv                                                           # 3.12 explicitly: a bare `python` is 3.14 on this machine
.venv/Scripts/python.exe -m pip install --no-cache-dir -r requirements.txt       # from the repo root; installs ./library too; needs AMD driver 26.2.2
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"     # ROCm exposes the GPU as torch.cuda
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
python train.py --detector spacetime_cnn --dataset frog --dtime 10               # defaults are now --epochs 100 --patience 15
python train_all.py --dataset frog --out-dir ../checkpoints                      # spacetime_cnn, fullscan_tcn, temporal_unet, li2former, in sequence
python train.py --detector temporal_unet --dataset frog --resume ../checkpoints/temporal_unet.best.pth --epochs 40   # resume: runs 40 MORE epochs, not "up to 40 total"
python train.py --detector spacetime_cnn --dataset frog --init-weights ../checkpoints/spacetime_cnn.best.pth --lr 1e-4   # fine-tune from another checkpoint's weights only (fresh optimiser/epoch count, unlike --resume)
```

**The three-horizon detector** (`TODO.md` A43) trains with its own script, because it streams recordings rather than sampling frames:

```bash
python train_three_horizon.py step2        # stages 1-2, phase A; clips of max(coarse_lags) + 1 = 33 frames; --max-range-m sets the model range (default 10 m); 24 h cap
python train_three_horizon.py step3a --stage2 ../checkpoints_three_horizon/step2_calibration.best.pth     # stage 3 on stage 2's candidates (builds their cache first)
python train_three_horizon.py step3b --stage2 ../checkpoints_three_horizon/step2_calibration.best.pth --memory ../checkpoints_three_horizon/step3a_object_memory.best.pth   # whole detector with feedback
python train_three_horizon.py step4 --stage2 ... --memory ... --joint ../checkpoints_three_horizon/step3b_joint.best.pth   # SORT and slot-count ablations, no training
python scan_anomalies.py --dataset frog    # raw-scan noise report: no-return encodings, invalid runs, spikes, jitter, timestamps (also drow; A49)
python three_horizon_oracle.py             # the interpolation bound on OUR candidates: recall headroom above stage 2, and how much of it stage 3 takes (A50 phase 1)
python duration_curve.py --dir ../checkpoints_three_horizon/step2_control ../checkpoints_three_horizon/step2_dropout01   # test AP against training duration, from `step2 --checkpoint-every`'s checkpoints (A50 phase 1)
python train_three_horizon.py step1        # stage 3 on cached LFE-Peaks candidates, the memory in isolation (dropped from the plan 2026-09-15)
```

Each writes `../checkpoints_three_horizon/<step>_*.best.pth` (the run's best, whatever epoch or stage produced it) and `<step>_results.json`; every budget flag is in `--help`, defaults as in `TODO.md` A43. `--limit-recordings N` makes a smoke run. Launch long runs as a detached process, not as a child of an agent session, which ends with the session.

`--frog-mode {official,transferred,balanced}` selects the FROG partition (default `official`, the published benchmark).
It is **stored in the checkpoint**, so `evaluate.py` defaults to the partition the weights were trained under and refuses to score two differently-partitioned checkpoints in one pass.
`transferred` shares `official`'s train and val exactly, so one checkpoint serves both; `balanced` does not, and its val contains a recording the published baselines trained on.

`--detector` choices: `algorithmic` (eval only, not trainable), `spacetime_cnn`, `fullscan_tcn`, `temporal_unet`, `li2former`.
`--dataset {drow,frog,jrdb}`, `--dtime N` (temporal stride — **also stored in the checkpoint**, so `evaluate.py` picks it up automatically), `--no-align-scans` (disable odometry-based rotation correction), `--no-frame-cache` (trade time for ~2.6 GB less RAM on FROG's full train split).

**Resuming is safe, with one gotcha.** `.best.pth` is written atomically (temp + rename) on every wp-AUC improvement and carries weights, optimizer state, epoch and the `val_wp_auc` it was selected on — so a crashed run loses at most one epoch, and resuming will not overwrite a better checkpoint with a worse one.
The gotcha is arithmetic: `--epochs N` with `--resume` runs N *further* epochs, counted from the checkpoint's own epoch, not up to N in total.

**Early stopping and model selection run on val wp-AUC**, not `val_loss` (which swings 3x with batch composition — `interpreting-evaluation.md`).
`--patience` counts epochs without a wp-AUC improvement; `--auc-max-frames` (default 2000) caps the per-epoch gate to a strided subsample of the val split, while the reported number always uses all of it.
`--epochs 100 --patience 15` are the defaults, roughly matching LFE-Peaks/LFE-PPN's own recipe.
Run `python train.py --help` for the full, current flag list — it changes more often than this file does.

## Evaluation (`utils/evaluate.py`, run from `utils/`)

```bash
python evaluate.py --dataset drow --drow --drspaam                                   # published weights, no training needed
python evaluate.py --dataset frog --lfe-peaks --lfe-ppn                              # bundled ONNX weights
python evaluate.py --dataset frog --split test --spacetime-cnn ../checkpoints/spacetime_cnn.best.pth --bench   # --eval-dtime now defaults to the checkpoint's own value
python evaluate.py --dataset drow --verify --no-bench                                # dataset FoV/annotation sanity stats only, no models
python evaluate.py --bench --no-eval --n-beams 720 --time-frame 5                     # synthetic-data speed benchmark, no dataset needed
```

`--eval-zero-history` blanks every frame of the temporal window but the current one, at eval time only — the one-pass measurement of whether a trained model actually uses its history (`TODO.md` A21). `train.py --zero-history` is the matching train-time ablation.

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

## Dataset-property analysis (run from `utils/`)

The measurements behind [`dataset-properties.md`](dataset-properties.md). All model-free except `--phantoms`, all minutes rather than hours.

```bash
python motion_analysis.py --dataset frog --people        # how long people stay still (the horizon question)
python motion_analysis.py --dataset drow --people        # the same, in a care facility rather than a museum
python motion_analysis.py --people --gate-floor 1.0      # sensitivity: the conclusion must survive the gate
python motion_analysis.py --phantoms                     # the same measure on a detector's false positives (FROG only)
python phantom_analysis.py                               # phantom persistence + phantom-vs-duplicate split
python motion_analysis.py --headroom                     # of the people a static model misses, how many move?
python temporal_oracle.py                                # upper bound on what any trajectory reasoning could buy
python horizon_sweep.py --detector lfe-peaks            # how long a horizon each oracle half actually needs
python horizon_sweep.py --discrimination                # the role-C bound on POPULATED frames, where a filter can destroy people too (A45)
python range_probe.py                                   # recall vs range, and the sub-threshold miss budget
python repr_probe.py                                    # scale vs capacity, 1-NN over four window parameterisations
python nms_sweep.py --skip fullscan_tcn lfe_ppn         # range-aware merge radius r(d) = a + b*d (A38)
python nms_sweep.py --skip fullscan_tcn lfe_ppn --score-by operating-point   # the same radius, scored on recall/duplicates/close-pairs instead of AP (A40)
python persistence_map.py                               # role C: world-location persistence over detections (A41)
python persistence_map.py --background                  # role C: background model from raw scan returns (A44)
python slot_budget.py                                   # how many object-memory slots K stage 3 needs, per retire window (A43)
python slot_budget.py --memory                          # value-ranked slot eviction vs unlimited K and a person-aware oracle (A43)
```

**`nms_sweep.py --score-by` picks the question, and the two answers disagree.**
`ap` (the default) sweeps against average precision; `operating-point` sweeps against what Tier 0 measured — recall, duplicates, phantoms and close-pair misses at a fixed threshold.
AP prefers a *larger* merge radius than the operating-point view does, because a duplicate is a full false positive while a close-pair miss costs only one missed person (`rejected-ideas.md`, `TODO.md` A40).
Quote which one a radius came from.

**`range_probe.py` and `repr_probe.py` both cache** (`utils/.range_cache/`, `utils/.repr_cache/`), so only the first invocation per configuration pays for the detector pass or the window build.
`range_probe.py`'s recall figures are at a **single operating point** (`--thresh`, default 0.3), which is not AP — see `static-detector-diagnosis.md` before quoting one.

**`--gate-floor` is the knob to distrust.** Trajectory association needs a gate wide enough to absorb annotation jitter, and one sized from motion alone is far too tight at 26 Hz (2.5 m/s over 38 ms is 9.6 cm). Too tight and trajectories shatter into singletons, which biases every statistic *toward* stationary people — the direction that happens to flatter this project's hypothesis. Check any conclusion at 0.4, 0.6 and 1.0 before believing it.

**`--min-seq-seconds` filters by duration, not frame count**, because 1,000 frames is 38 s of FROG and 100 s of DROW, and DROW labels ~486 frames per 18-minute session.

## Informational-capacity proxy (`utils/informational_capacity.py`, run from `utils/`)

```bash
python informational_capacity.py   # full (T, dtime) grid, CPU-only, no training, ~a few minutes
```

No flags — edit `T_VALUES`/`DTIME_VALUES` at the top of the file to change the grid.
Methodology, citations and the current grid: `informational-capacity-proxy.md`.

## Checks

```bash
.venv/Scripts/python.exe -m pytest tests -q               # 347 tests, ~30s, CPU-only, no GPU or dataset needed — run after every edit
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
