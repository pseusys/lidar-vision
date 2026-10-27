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
python -c "import takheliped; from takheliped.detectors import AlgorithmicDetector; print('OK')"   # confirms the C++ extension built
```

## Training (`utils/train.py`, run from `utils/`)

```bash
python train.py --detector spacetime_cnn --dataset frog --dtime 10               # defaults are now --epochs 100 --patience 15
python train_all.py --dataset frog --out-dir ../checkpoints                      # spacetime_cnn, fullscan_tcn, temporal_unet, li2former, in sequence
python train.py --detector temporal_unet --dataset frog --resume ../checkpoints/temporal_unet.best.pth --epochs 40   # resume: runs 40 MORE epochs, not "up to 40 total"
python train.py --detector spacetime_cnn --dataset frog --init-weights ../checkpoints/spacetime_cnn.best.pth --lr 1e-4   # fine-tune from another checkpoint's weights only (fresh optimiser/epoch count, unlike --resume)
```

**TAKHeLiPeD**, the three-horizon detector (`TODO.md` A43, A51), trains with its own script, because it streams recordings rather than sampling frames:

```bash
python train_takheliped.py step2 --checkpoint-every 0.25   # stages 1-2 on single frames, A59's recipe by default (nms-mean, mirror, nearest labels, dropout 0.1); then average_checkpoints.py; --max-range-m sets the model range (default 10 m)
python train_takheliped.py step3a --stage2 ../checkpoints_three_horizon/step2_calibration.best.pth     # stage 3 on stage 2's candidates (builds their cache first)
python train_takheliped.py step3b --stage2 ../checkpoints_three_horizon/step2_calibration.best.pth --memory ../checkpoints_three_horizon/step3a_object_memory.best.pth   # whole detector with feedback
python train_takheliped.py step4 --stage2 ... --memory ... --joint ../checkpoints_three_horizon/step3b_joint.best.pth   # SORT and slot-count ablations, no training
python scan_anomalies.py --dataset frog    # raw-scan noise report: no-return encodings, invalid runs, spikes, jitter, timestamps (also drow; A49)
python slot_experiments.py horizon --memory <step3a ckpt> --stage2 <step2 ckpt>    # how long a memory the model uses: AP against how often it is emptied (A51 E1)
python slot_experiments.py odometry --memory ... --stage2 ... --sensor-memory <D1 ckpt>  # AP against odometry drift, room frame vs sensor frame (A51 E2)
python slot_experiments.py crowding --memory ... --stage2 ...                    # AP by people per frame, candidates vs memory (A51 E3)
python slot_experiments.py cost --memory ... --stage2 ...                        # ms per frame against slot count, 64-720 (A51 E4)
python slot_experiments.py odometry --memory ... --stage2 ... --out ../results/<name>.json   # --out names the results file; without it every run of a mode overwrites the same a51_<mode>.json, which destroyed E2's and E4's numbers once
python empty_scene.py --stage2 <step2 ckpt> --memory <step3a ckpt>               # detections per frame on FROG transferred's 72,533 person-free frames, where every one is a false positive (A50 phase 1 item 7)
python score_memory.py --memory ... --stage2 ... --out <json>                    # score an interrupted run's best checkpoint (A51)
python judge_arms.py                                                            # every A51 arm's deltas and verdicts against a control's three-sigma bars; stdlib only, safe beside a running job
python train_takheliped.py step3a --stage2 ... --visibility                   # give the coasting head the free space the laser saw past each slot (A51 S2); rebuilds the candidate cache once
python tracking_baselines.py rule                                               # A56 E15: the memory's bookkeeping with a rule score, 54-config grid on val, then test (~2.5 h)
python tracking_baselines.py track --workers 6                                  # A56 E14: SORT, ByteTrack, ByteTrack + CBMOT by coordinate descent on val, then test (~35 min); never beside another torch job
python tracking_baselines.py arms                                               # every memory checkpoint's frames, test (every annotated frame) and val, plus LFE (~1 h)
python tracking_baselines.py report                                             # AP with block-bootstrap intervals, GOSPA, strata, first-detection latency -> a56/report.json (~15 min)
python tracking_baselines.py latency [--device cpu --threads 4]                 # A56 E22: the whole detector streamed one scan at a time, stage by stage
python tracking_baselines.py merge --workers 6                                  # A58: merge radius per saved output, chosen on val, bootstrapped, double reports dissected -> a56/merge.json (~10 min, needs arms + track)
python tracking_baselines.py {rule,track,arms,report} --stage2 <step2 ckpt> --out <dir> --arms E13:<run>   # A61: the same replays on another stage 2 and memory; one --out per stage-2 seed, --arms on arms/report/merge
python tracking_baselines.py summary --out ../checkpoints_three_horizon/a61    # A61: <out>/s*/report.json lined up by arm, with each bootstrap pair's intervals above zero counted -> <out>/summary.json
python tracker_sweep.py --tracker byte | byte-cbmot                             # ByteTrack (with CBMOT confidence) on the published LFE detectors
python train_takheliped.py step3a --stage2 ... --train-drift-m-per-s 0.05    # train with odometry drifting at a rate drawn per stream, the repair for D1's conditional win (A51 S3/E11)
python train_takheliped.py step3b --stage2 ... --memory ... --oracle-prior    # feed stage 2 a ground-truth prior: the ceiling on what feedback could buy (A51 E12b)
python train_takheliped.py step3b --stage2 ... --memory ... --no-rescore      # drop the rescoring correction, leaving the prior as memory's only route to the loss (A51 E12a)
python verify_oracle_prior.py --joint <step3b ckpt>                             # is E12's oracle prior real on recorded frames, and did the feedback weights move? needs the machine to itself (A51 E12)
python measure_prior_share.py --runs a51_e12b_oracle a51_e12a_norescore        # what fraction of stage 2's bottleneck the feedback prior carries; explains E12's null (A51 E12)
python judge_arms.py --control a51_b0_short --json ../results/a51_arms.json      # the same against the older control, written out as data
python memory_diagnosis.py --slot-speed --cache .takheliped_cache/<test cache>.pkl   # slot velocity against a >= 1 s displacement reference, stamped dt vs the trainer's running average; no model (A51 0a)
python memory_diagnosis.py --recovery --stage2 ... --memory ... --out <json>     # who stage 3 misses, whether a slot holds them, and whether a candidate exists at all (feedback's headroom; A51 E12)
python takheliped_oracle.py             # the interpolation bound on OUR candidates: recall headroom above stage 2, and how much of it stage 3 takes (A50 phase 1)
python duration_curve.py --dir ../checkpoints_three_horizon/step2_control ../checkpoints_three_horizon/step2_dropout01   # test AP against training duration, from `step2 --checkpoint-every`'s checkpoints (A50 phase 1); pass --out, or it overwrites duration_curve.json; ~3.5 min per checkpoint
python stage2_probe.py --stage2 <step2 ckpts> --stride 5                        # A59: the same stage-2 outputs through 7 decoders (vote grid, DR-SPAAM's NMS, LFE's peaks...), and recall by range and crowding against LFE at matched FP (~5 min per checkpoint)
python stage2_probe.py --stage2 <step2 ckpts> --split val --stride 25 --out ../results/stage2_probe_val.json   # A59: the decoders' AP on val, where a decoder is chosen
python train_takheliped.py step2 --label-assign last --no-mirror --dropout 0   # A59 ablation: the recipe before 2026-10-06; the defaults are nearest-person labels, mirroring and dropout 0.1, decoded by nms-mean
python train_takheliped.py step2 --input-features lfe | --channels 8 16 32   # A59 ladder F1 / W: LFE's input alone, LFE's widths; the probe and averaging rebuild either from the checkpoint's args
python train_takheliped.py step2 --batch 8 --max-epochs 5 --min-epochs 5 --schedule exponential --lr 1e-3 --min-lr 1e-6 --weight-decay 0 --select last   # A59 ladder R: DR-SPAAM's protocol; tests the last weights, kept as step2_calibration.last.pth beside the best
python coarse_e2e.py                                                            # A60's verdict: the six E13 detectors' AP, AP@0.3, recall at a shared FP budget and first-second recall, and drop/keep for the coarse convolution; re-scores from its saved frames, since the coarse checkpoints no longer load (code at e76aa94)
python average_checkpoints.py ../checkpoints_three_horizon/<run> [...]         # stage 2's weight averaging (recipe since 2026-10-06): val-chosen window of a `--checkpoint-every 0.25` run -> <run>/step2_calibration.averaged.pth (~2 min per run)
python train_takheliped.py step1        # stage 3 on cached LFE-Peaks candidates, the memory in isolation (dropped from the plan 2026-09-15)
```

Every step takes `--resume`, **verified end to end on 2026-09-24** by killing a run at epoch 3 and relaunching it: it continued at epoch 4 with its validation curve and gate unbroken, then finished and wrote its results. It continues from `<step>_resume.pth` in `--out-dir` -- written after each evaluation, so a crash costs the time since it rather than the whole run. Relaunch the identical command with `--resume` added; it is off by default, so a fresh run never picks up an old state by accident.

**Long queues run through `utils/run_queue.py`** (A57), which replaced the eight hand-copied PowerShell launchers. A queue is a committed Python module that builds `Job`s and calls `main`:

```bash
python ../checkpoints_three_horizon/<queue>.py --list          # the queue in dependency order, and which jobs are already done
python <queue>.py --steps ladder_probe a61_summary             # only these jobs; the others count as satisfied
python <queue>.py                                              # everything whose dependencies are met and whose outputs are missing
```

Each job names the files that prove it finished, so a finished job is skipped and an interrupted one is retried from its own `--resume` state, up to three attempts. A job whose log carries a training guard (`not finite at chunk`, `candidates' val AP moved`, `the training loss is`) stops after one attempt, because retrying cannot fix a run that is wrong rather than interrupted, and **a job whose dependency failed is recorded `BLOCKED` and never starts** -- the mistake the copied launchers made, since they waited for an idle machine and a stopped chain makes the machine idle. Progress goes to `checkpoints_three_horizon/progress.txt` in the `QUEUE <job> ...` format the watchers poll. Launch it detached through the Task Scheduler (`long-running-tasks.md`).

Each writes `../checkpoints_three_horizon/<step>_*.best.pth` (the run's best, whatever epoch or stage produced it) and `<step>_results.json`; every budget flag is in `--help`, defaults as in `TODO.md` A43.
**There is no `--max-hours`** (removed 2026-10-09, A50 phase 1 item 6): a run is bounded by `--max-epochs` and `--patience`, and the wall-clock cap only ever truncated a healthy run -- it ended A51's first full-curriculum step 3a inside its first chunk-length stage. `--limit-recordings N` makes a smoke run. Launch long runs as a detached process, not as a child of an agent session, which ends with the session.

`--frog-mode {official,transferred,balanced}` selects the FROG partition (default `official`, the published benchmark).

**The ablation flags, and which script scores each.** Every one of these is recorded in the checkpoint and restored from it
(`network_shape` for stage 2, `object_memory` for stage 3), so a checkpoint rebuilds into the network it was trained as and
one written before a flag existed falls back to the adopted default.

| flag | step | item | arms |
| --- | --- | --- | --- |
| `--recurrence selective\|gru\|mingru` | `step3a` | 16 | which rule updates a slot's state; **only `selective` sees `dt`**, which is the comparison |
| `--vote-frame beam\|sensor` | `step2` | 18 | which frame the head regresses the offset to a person's centre in |
| `--block convnext\|residual` | `step2` | 19 | which block carries the U-Net's depth |
| `--channels 64 128 256` | `step2` | 19 | the widening half of item 19; no new code was needed |
| `--gather-width-m`, `--gather-samples` | `step2` | 15 | the metric-width first layer -- **answered, it costs** |

Their queues: `queue_item16.py` (the recurrences, each with its own CPU latency job) and `queue_items1819.py` (three stage-2
arms at three seeds, each followed by its weight average, then one `stage2_probe.py` pass over all twelve checkpoints).

**The adopted architecture, as of 2026-10-27** -- slots + minGRU, **no aggregator, no curriculum, `dim 64`**
(`TODO.md` items 38, 40, 41, 49). Stage 2 needs `--no-aggregator`; stage 3 needs these six flags and nothing else:

```powershell
cd utils
.\..\.venv\Scripts\python.exe train_takheliped.py step2 --checkpoint-every 0.25 --no-aggregator --seed 0 --out-dir ../checkpoints_three_horizon/<name>
.\..\.venv\Scripts\python.exe average_checkpoints.py ../checkpoints_three_horizon/<name>          # stage 3 loads the AVERAGED weights
.\..\.venv\Scripts\python.exe train_takheliped.py step3a --stage2 ../checkpoints_three_horizon/<name>/step2_calibration.averaged.pth --no-exchange --no-learned-match --no-coast --recurrence mingru --chunk-s 10 --dim 64 --seed 0 --out-dir ../checkpoints_three_horizon/<arm>
```

**The defaults are deliberately NOT the adopted recipe**, and flipping them would be a mistake: `network_shape` falls
back to a constructor default for any flag a checkpoint predates, so defaulting `aggregator=False` would make every
older stage-2 checkpoint rebuild without it and refuse its own weights.

**The matched CPU latency sweep** (`TODO.md` item 29's protocol), which has now been needed four times and got
done wrongly once:

```powershell
cd utils
$s2 = "../checkpoints_three_horizon/a69_no_aggregator_s0/step2_calibration.averaged.pth"
foreach ($r in 1..3) { foreach ($a in @("dim128:a70_baseline_s0", "dim64:a71_dim64_s0")) {
  $n, $d = $a -split ":"
  .\..\.venv\Scripts\python.exe tracking_baselines.py latency --device cpu --threads 4 `
    --stage2 $s2 --memory "../checkpoints_three_horizon/$d/step3a_object_memory.best.pth" `
    --out "../results/lat_${n}_r${r}"
} }
```

**Four things make it valid, and dropping any one invalidates it.** The arms must be **interleaved within one
session**, because rows from different sessions drift ~3% and that is how item 16's wrong `gru_cpu` row (23.87 ms
against a true 20.57) was produced. The host must be **idle** -- do not run the sweep while anything trains, and do
not run anything else while the sweep runs. Each run needs **its own `--out`**, because the tool names its file
`latency_cpu_t4.json` from the device and thread count only and silently overwrites otherwise. And the **`stage2`
phase is read first**: it is byte-identical work across arms, so its spread *is* the session's noise floor, and it also
catches ordering bias -- in the 2026-10-27 sweep it showed `dim 64` running 0.31 ms slower on identical work purely
because it went second.

**Never** use `step3a_results.json`'s `memory_ms_per_frame` instead. It is timed on whatever device trained the arm,
which is the GPU here.

**`queue_freeze.py` is the three-seed freeze** (`TODO.md` item 47), prepared and **not launched** -- it trains stage-2
seeds 1 and 2 (seed 0 already exists as `a69_no_aggregator_s0`), averages each, then runs the three stage-3 arms, about
13 h. `--list` shows what it would do without running anything. It deliberately contains **no latency job**: item 29's
protocol is a matched, interleaved sweep on an idle host, and a sequential queue runs each job in its own session,
which is how item 16's wrong `gru_cpu` row (23.87 ms against a true 20.57) was produced.
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
.venv/Scripts/python.exe -m pytest tests -q               # 533 tests, ~60s, CPU-only, no GPU or dataset needed — run after every edit
python memory/scripts/machine_is_free.py && python -m pytest tests -q   # the guard first: the suite has killed two live training runs
python memory/scripts/verify_memory.py                    # docs: links, indexes, markdown rules
python memory/scripts/verify_memory.py --strict            # same, but style warnings fail the run too
```

There is no typecheck/parse step faster than the test suite itself — a minute already is the cheap check here.
The test suite runs by hand only; there is no CI job invoking `pytest` yet (`TODO.md`).

## Linting

```bash
ruff check library/ utils/                                  # the project's own Python: the bug-shaped rules, clean as of 2026-10-11
ruff check --config memory/scripts/ruff.toml memory/scripts/   # the automation scripts in memory/scripts/ only
actionlint                                                       # workflows, plus shellcheck on run: blocks
npx markdownlint-cli2 "**/*.md"                                  # markdown, if Node is available
```

No linter is configured yet for this project's own Python (`library/`, `utils/`), C++, or the `entrypoint.sh`/`Dockerfile` under `deploy/docker/` — see `coding-guidelines.md` and `TODO.md`.
`ruff` itself is installed in `.venv` since 2026-10-02 (`python -m ruff`); until the project has a config, `python -m ruff check --isolated --line-length 320 --select E,F,B <files>` catches real errors without fighting the codebase's `typing.List` style.

## Commands whose output is easy to misread

- **`evaluate.py`'s AUC number without checking which regime produced it.** The same "DROW" column can mean a published-weights zero-shot transfer from FROG, a from-scratch DROW-native retrain, or a fine-tuned-from-FROG checkpoint — three different numbers that have all appeared for the same architecture.
  Check `interpreting-evaluation.md` before quoting one.
- **`train_all.py`'s printed summary table** — historically mislabeled the agnostic (any-class) AUC column as "Test AUC" instead of person-class `wp` AUC; confirm which column a number came from before repeating it (see `AGENTS.md` rule 2b).
- **A `--eval-stride N` run's timing** looks like a full-dataset evaluation but only touched `1/N` of the frames — real for a quick check, not directly comparable to a `stride=1` number without accounting for it.

## Platform notes

- Windows Git Bash: redirect long-running command output to a file and read the file, never `tail`/truncate it unread (`AGENTS.md` I2) — this project's training/evaluation logs are routinely tens of MB and hours long.
- `make` targets assume a POSIX shell (`SHELL = /bin/bash` in the Makefile) — run them from Git Bash, not PowerShell directly.
