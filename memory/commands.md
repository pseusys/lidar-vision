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
python train_all.py --dataset frog --out-dir ../checkpoints                      # spacetime_cnn, li2former, in sequence
python train.py --detector spacetime_cnn --dataset frog --resume ../checkpoints/spacetime_cnn.best.pth --epochs 40   # resume: runs 40 MORE epochs, not "up to 40 total"
python train.py --detector spacetime_cnn --dataset frog --init-weights ../checkpoints/spacetime_cnn.best.pth --lr 1e-4   # fine-tune from another checkpoint's weights only (fresh optimiser/epoch count, unlike --resume)

# Li2Former's OWN network. `--lr 1e-4` is required: the 1e-3 default kills their regression head in two epochs.
# Batch 1 was forced by their BatchFormer attending across B x N_beams tokens; with it off (the omni-dimension modes cannot
# run it) that constraint is gone and a larger batch may fit -- unmeasured, so batch 1 is kept for comparability.
python train.py --detector li2former_official --dataset drow --batch-size 1 --epochs 12 --patience 15 --lr 1e-4 --no-frame-cache --out ../checkpoints_drow/li2former_official_parallel.pth
python queue_li2former_drow.py --list                                                    # the DROW validation leg as a chain
python queue_li2former_drow_resume.py --list                                            # the 12-epoch resume leg at lr 1e-4, arm A
python queue_upstream_grid.py --list                                                    # their decoder and their metric, one pass per radius

# DROW3 at T=1 on DR-SPAAM's own DROW recipe (their cfgs/base_drow_drow_cfg.yaml, flag for flag): the known answer
# DROW *training* is checked against, 66.6% on DROW test in their Table I. Last weights, scored by evaluate.py.
python train.py --detector drow --dataset drow --time-frame 1 --drow-cutout drspaam --person-only --vote-radius 0.35 \
    --no-class-balance --vote-loss euclidean --vote-weight 1 --amsgrad --weight-decay 0 --lr 1e-3 \
    --lr-schedule exponential --min-lr 1e-6 --epochs 40 --batch-size 8 --dropout 0.5 --patience 0 \
    --out ../checkpoints_drow/drow_t1_drspaam_recipe.pth
python evaluate.py --dataset drow --split test --drow ../checkpoints_drow/drow_t1_drspaam_recipe.pth --no-bench --out ../results/drow_t1_drspaam_recipe_test.json
python queue_drow_validation.py --list      # that run, plus the scoring-path check, the label geometry and the fit-or-generalise scores

# Is their vote head alive? 8 beams, CPU, no dataset -- safe to run beside a training job.
# Their terminal ReLU dies within two epochs; this is what proves it, and what arm B has to disprove.
.venv/Scripts/python.exe -I utils/diagnose_vote_head.py                                   # fresh weights
.venv/Scripts/python.exe -I utils/diagnose_vote_head.py checkpoints_drow/li2former_official.best.pth
```

**The first optimizer step of any ROCm run takes 230-370 s of kernel compilation and prints nothing** -- per process and per
batch shape, so a shape change is another silent gap. Read a training log with `tr '\r' '\n'`, and only `-u` runs put tqdm's
bar in a redirected log at all (`gotchas.md`).

**TAKHeLiPeD**, the three-horizon detector, trains with its own script, because it streams recordings rather than sampling frames:

```bash
python train_takheliped.py step2 --checkpoint-every 0.25   # stages 1-2 on single frames, the adopted recipe by default (nms-mean, mirror, nearest labels, dropout 0.1); then average_checkpoints.py; --max-range-m sets the model range (default 10 m)
python train_takheliped.py step3a --stage2 ../checkpoints_three_horizon/step2_calibration.best.pth     # stage 3 on stage 2's candidates (builds their cache first)
python train_takheliped.py step3b --stage2 ../checkpoints_three_horizon/step2_calibration.best.pth --memory ../checkpoints_three_horizon/step3a_object_memory.best.pth   # whole detector with feedback
#   step3b takes no --label-* flags: it reads the beam target, its radius and the dropout out of the stage-2
#   checkpoint and prints them, because fine-tuning has to minimise the objective those weights already minimised.
python train_takheliped.py step4 --stage2 ... --memory ... --joint ../checkpoints_three_horizon/step3b_joint.best.pth   # SORT and slot-count ablations, no training
python scan_anomalies.py --dataset frog    # raw-scan noise report: no-return encodings, invalid runs, spikes, jitter, timestamps (also drow)

# Another FROG partition. `--frog-mode` reaches every step that loads frames and is **stored in the checkpoint**,
# so `evaluate.py` scores the partition the weights were trained on and refuses to put two in one pass. It also
# keys the stage-3 candidate cache, and a resume file written under another partition is refused rather than used.
python train_takheliped.py step2 --frog-mode balanced --no-aggregator --label-sigma-m 0.3 --seed 0 --out-dir ../checkpoints_three_horizon/balanced_stage2_s0 --resume
python queue_balanced.py --list                     # the whole one-seed chain plus the published pair on balanced's test

# Score a trained detector on a partition it did NOT train on -- the like-for-like comparison, no training.
# Stride 5 by default, because the frozen 86.29% is a stride-5 number; the partition mismatch is printed and
# stored in the output rather than refused, since that mismatch is the whole point of the mode.
python train_takheliped.py score --stage2 ../checkpoints_three_horizon/balanced_stage2_s0/step2_calibration.averaged.pth \r
    --memory ../checkpoints_three_horizon/balanced_stage3_s0/step3a_object_memory.best.pth \r
    --frog-mode official --split test --out ../results/score_balanced_on_official.json   # ~7 min on the CPU

# Is the partition what the loader claims? Real recordings, not the synthetic fixtures the unit tests use.
.venv/Scripts/python.exe -I utils/check_balanced_partition.py                 # all three splits, ~10 min
.venv/Scripts/python.exe -I utils/check_balanced_partition.py --splits val    # one at a time, for memory
python slot_experiments.py horizon --memory <step3a ckpt> --stage2 <step2 ckpt>    # how long a memory the model uses: AP against how often it is emptied
python slot_experiments.py odometry --memory ... --stage2 ... --sensor-memory <D1 ckpt>  # AP against odometry drift, room frame vs sensor frame
python slot_experiments.py crowding --memory ... --stage2 ...                    # AP by people per frame, candidates vs memory
python slot_experiments.py cost --memory ... --stage2 ...                        # ms per frame against slot count, 64-720
python slot_experiments.py odometry --memory ... --stage2 ... --out ../results/<name>.json   # --out names the results file; without it every run of a mode overwrites the same a51_<mode>.json, which destroyed E2's and E4's numbers once
python empty_scene.py --stage2 <step2 ckpt> --memory <step3a ckpt>               # detections per frame on FROG transferred's 72,533 person-free frames, where every one is a false positive
python empty_scene.py --stage2 ... --memory ... --frog-mode balanced --split test   # the same count on another partition's person-free frames; only those frames are counted and a split with none is refused (official is 100% populated)
python empty_scene.py --stage2 ... --memory ... --no-calibration                  # ignore the stage-3 checkpoint's fitted (temperature, bias) so a calibrated arm and an uncalibrated one are read at the same thresholds; off by default, which reproduces the published rows
python queue_person_free.py --list                                                    # both arms of the person-free comparison on balanced test's person-free frames: the balanced-trained pair, the official-trained one at matched scale, then the official one as published (~45 min)
python score_memory.py --memory ... --stage2 ... --out <json>                    # score an interrupted run's best checkpoint
python calibrate_rescore.py --memory ... --stage2 ... --out <json> [--frames <npz>]  # fit (temperature, bias) on val so a fixed threshold means something (~65 min/seed)
python calibrate_rescore.py --memory ... --apply <json>                          # store an already-fitted pair in the checkpoint, re-checking its gates; seconds, no re-measurement
python queue_score_calibration.py --list                                                    # the three-seed chain, run via run_queue.py so the ROCm teardown hang cannot block the next seed
python queue_phantom_column.py --list                                                    # the phantom column re-measured on the calibrated checkpoints; empty_scene.py unchanged, the key comes from the checkpoint
python queue_frozen_tracking_baselines.py --list                                                    # the frozen candidates:'s rule/track/arms/report at three seeds on the frozen candidates, then summary (~13 h); `report` is blocked if `rule` fails
python judge_arms.py                                                            # every stage-3 arm's deltas and verdicts against a control's three-sigma bars; stdlib only, safe beside a running job
python train_takheliped.py step3a --stage2 ... --visibility                   # give the coasting head the free space the laser saw past each slot; rebuilds the candidate cache once
python tracking_baselines.py rule                                               # E15: the memory's bookkeeping with a rule score, 54-config grid on val, then test (~2.5 h)
python tracking_baselines.py track --workers 6                                  # E14: SORT, ByteTrack, ByteTrack + CBMOT by coordinate descent on val, then test (~35 min); never beside another torch job
python tracking_baselines.py arms                                               # every memory checkpoint's frames, test (every annotated frame) and val, plus LFE (~1 h)
python tracking_baselines.py report                                             # AP with block-bootstrap intervals, GOSPA, strata, first-detection latency -> a56/report.json (~15 min)
python tracking_baselines.py latency [--device cpu --threads 4]                 # E22: the whole detector streamed one scan at a time, stage by stage
python tracking_baselines.py merge --workers 6                                  # merge radius per saved output, chosen on val, bootstrapped, double reports dissected -> a56/merge.json (~10 min, needs arms + track)
python tracking_baselines.py {rule,track,arms,report} --stage2 <step2 ckpt> --out <dir> --arms E13:<run>   # the same replays on another stage 2 and memory; one --out per stage-2 seed, --arms on arms/report/merge
python tracking_baselines.py summary --out ../checkpoints_three_horizon/a61    # <out>/s*/report.json lined up by arm, with each bootstrap pair's intervals above zero counted -> <out>/summary.json
python tracker_sweep.py --tracker byte | byte-cbmot                             # ByteTrack (with CBMOT confidence) on the published LFE detectors
python train_takheliped.py step3a --stage2 ... --train-drift-m-per-s 0.05    # train with odometry drifting at a rate drawn per stream, the repair for D1's conditional win
python train_takheliped.py step3b --stage2 ... --memory ... --oracle-prior    # feed stage 2 a ground-truth prior: the ceiling on what feedback could buy
python train_takheliped.py step3b --stage2 ... --memory ... --no-rescore      # drop the rescoring correction, leaving the prior as memory's only route to the loss
python verify_oracle_prior.py --joint <step3b ckpt>                             # is E12's oracle prior real on recorded frames, and did the feedback weights move? needs the machine to itself
python measure_prior_share.py --runs a51_e12b_oracle a51_e12a_norescore        # what fraction of stage 2's bottleneck the feedback prior carries; explains E12's null
python judge_arms.py --control a51_b0_short --json ../results/a51_arms.json      # the same against the older control, written out as data
python memory_diagnosis.py --slot-speed --cache .takheliped_cache/<test cache>.pkl   # slot velocity against a >= 1 s displacement reference, stamped dt vs the trainer's running average; no model
python memory_diagnosis.py --recovery --stage2 ... --memory ... --out <json>     # who stage 3 misses, whether a slot holds them, and whether a candidate exists at all (feedback's headroom)
python takheliped_oracle.py             # the interpolation bound on OUR candidates: recall headroom above stage 2, and how much of it stage 3 takes
python duration_curve.py --dir ../checkpoints_three_horizon/step2_control ../checkpoints_three_horizon/step2_dropout01   # test AP against training duration, from `step2 --checkpoint-every`'s checkpoints; pass --out, or it overwrites duration_curve.json; ~3.5 min per checkpoint
python stage2_probe.py --stage2 <step2 ckpts> --stride 5                        # the same stage-2 outputs through 7 decoders (vote grid, DR-SPAAM's NMS, LFE's peaks...), and recall by range and crowding against LFE at matched FP (~5 min per checkpoint)
python stage2_probe.py --stage2 <step2 ckpts> --split val --stride 25 --out ../results/stage2_probe_val.json   # the decoders' AP on val, where a decoder is chosen
python train_takheliped.py step2 --label-assign last --no-mirror --dropout 0   # the recipe before 2026-10-06; the defaults are nearest-person labels, mirroring and dropout 0.1, decoded by nms-mean
python train_takheliped.py step2 --input-features lfe | --channels 8 16 32   # ladder F1 / W: LFE's input alone, LFE's widths; the probe and averaging rebuild either from the checkpoint's args
python train_takheliped.py step2 --batch 8 --max-epochs 5 --min-epochs 5 --schedule exponential --lr 1e-3 --min-lr 1e-6 --weight-decay 0 --select last   # ladder R: DR-SPAAM's protocol; tests the last weights, kept as step2_calibration.last.pth beside the best
python coarse_e2e.py                                                            # the coarse convolution's verdict: the six E13 detectors' AP, AP@0.3, recall at a shared FP budget and first-second recall, and drop/keep for the coarse convolution; re-scores from its saved frames, since the coarse checkpoints no longer load (code at e76aa94)
python average_checkpoints.py ../checkpoints_three_horizon/<run> [...]         # stage 2's weight averaging (recipe since 2026-10-06): val-chosen window of a `--checkpoint-every 0.25` run -> <run>/step2_calibration.averaged.pth (~2 min per run)
# PASS THE RUN'S OWN DATASET FLAGS. Until 2026-11-06 this script ranked every window on FROG `official` val
# whatever the run trained on, and three runs chose their window on the wrong data (`memory/gotchas.md`).
python average_checkpoints.py ../checkpoints_three_horizon/<run> --dataset drow        # ranked on DROW val
python average_checkpoints.py ../checkpoints_three_horizon/<run> --beam-stride 2       # ranked at 0.5 deg
python average_checkpoints.py ../checkpoints_three_horizon/<run> --frog-mode balanced  # ranked on `balanced` val
python average_checkpoints.py ../checkpoints_three_horizon/<run> --dataset drow --rank-only   # re-rank and report only: writes averaging.rank_only.json, leaves the averaged checkpoint (and the old averaging.json) alone
python train_takheliped.py step1        # stage 3 on cached LFE-Peaks candidates, the memory in isolation (dropped from the plan 2026-09-15)
```

Every step takes `--resume`, **verified end to end on 2026-09-24** by killing a run at epoch 3 and relaunching it: it continued at epoch 4 with its validation curve and gate unbroken, then finished and wrote its results. It continues from `<step>_resume.pth` in `--out-dir` -- written after each evaluation, so a crash costs the time since it rather than the whole run. Relaunch the identical command with `--resume` added; it is off by default, so a fresh run never picks up an old state by accident.

**Long queues run through `utils/run_queue.py`**, which replaced the eight hand-copied PowerShell launchers. A queue is a committed Python module that builds `Job`s and calls `main`:

```bash
python ../checkpoints_three_horizon/<queue>.py --list          # the queue in dependency order, and which jobs are already done
python <queue>.py --steps ladder_probe a61_summary             # only these jobs; the others count as satisfied
python <queue>.py                                              # everything whose dependencies are met and whose outputs are missing
```

Each job names the files that prove it finished, so a finished job is skipped and an interrupted one is retried from its own `--resume` state, up to three attempts. A job whose log carries a training guard (`not finite at chunk`, `candidates' val AP moved`, `the training loss is`) stops after one attempt, because retrying cannot fix a run that is wrong rather than interrupted, and **a job whose dependency failed is recorded `BLOCKED` and never starts** -- the mistake the copied launchers made, since they waited for an idle machine and a stopped chain makes the machine idle. Progress goes to `checkpoints_three_horizon/progress.txt` in the `QUEUE <job> ...` format the watchers poll. Launch it detached through the Task Scheduler (`long-running-tasks.md`).

Each writes `../checkpoints_three_horizon/<step>_*.best.pth` (the run's best, whatever epoch or stage produced it) and `<step>_results.json`; every budget flag is in `--help`, defaults as in `--help`.
**There is no `--max-hours`** (removed 2026-10-09): a run is bounded by `--max-epochs` and `--patience`, and the wall-clock cap only ever truncated a healthy run -- it ended the first full-curriculum step 3a inside its first chunk-length stage. `--limit-recordings N` makes a smoke run. Launch long runs as a detached process, not as a child of an agent session, which ends with the session.

`--frog-mode {official,transferred,balanced}` selects the FROG partition (default `official`, the published benchmark).

**The ablation flags, and which script scores each.** Every one of these is recorded in the checkpoint and restored from it
(`network_shape` for stage 2, `object_memory` for stage 3), so a checkpoint rebuilds into the network it was trained as and
one written before a flag existed falls back to the adopted default.

| flag | step | item | arms |
| --- | --- | --- | --- |
| `--recurrence selective\|gru\|mingru` | `step3a` | 16 | which rule updates a slot's state; **only `selective` sees `dt`**, which is the comparison |
| `--vote-frame beam\|sensor` | `step2` | 18 | which frame the head regresses the offset to a person's centre in |
| `--block convnext\|residual` | `step2` | 19 | which block carries the U-Net's depth |
| `--channels 64 128 256` | `step2` | 19 | the widening half of the shape sweep; no new code was needed |
| `--gather-width-m`, `--gather-samples` | `step2` | 15 | the metric-width first layer -- **answered on FROG, it costs 1.42 pp**; reopened once on DROW, where a beam covers twice the arc |
| `--head binary\|multiclass`, `--class-norm softmax\|sigmoid` | `step2` | 84 | whether the head emits one person logit or a score per annotated class, and how a multi-class one is normalised. The vote regression is shared either way |
| `--eval-max-range-m` | `step2`, `score` | 82, 86 | score a finished pass again inside a tighter radius, both readings. Scoring only -- `--max-range-m` is what shapes the network |

Their queues: `queue_selective_vs_gru.py` (the recurrences, each with its own CPU latency job) and `queue_vote_frame_and_block.py` (three stage-2
arms at three seeds, each followed by its weight average, then one `stage2_probe.py` pass over all twelve checkpoints).

**The adopted architecture, as of 2026-10-27** -- slots + minGRU, **no aggregator, no curriculum, `dim 64`**
Stage 2 needs `--no-aggregator`; stage 3 needs these six flags and nothing else:

```powershell
cd utils
.\..\.venv\Scripts\python.exe train_takheliped.py step2 --checkpoint-every 0.25 --no-aggregator --seed 0 --out-dir ../checkpoints_three_horizon/<name>
.\..\.venv\Scripts\python.exe average_checkpoints.py ../checkpoints_three_horizon/<name>          # stage 3 loads the AVERAGED weights
.\..\.venv\Scripts\python.exe train_takheliped.py step3a --stage2 ../checkpoints_three_horizon/<name>/step2_calibration.averaged.pth --no-exchange --no-learned-match --no-coast --recurrence mingru --chunk-s 10 --dim 64 --seed 0 --out-dir ../checkpoints_three_horizon/<arm>
```

**The defaults are deliberately NOT the adopted recipe**, and flipping them would be a mistake: `network_shape` falls
back to a constructor default for any flag a checkpoint predates, so defaulting `aggregator=False` would make every
older stage-2 checkpoint rebuild without it and refuse its own weights.

**The matched CPU latency sweep**, which has now been needed four times and got
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
session**, because rows from different sessions drift ~3% and that is how the wrong `gru_cpu` row (23.87 ms
against a true 20.57) was produced. The host must be **idle** -- do not run the sweep while anything trains, and do
not run anything else while the sweep runs. Each run needs **its own `--out`**, because the tool names its file
`latency_cpu_t4.json` from the device and thread count only and silently overwrites otherwise. And the **`stage2`
phase is read first**: it is byte-identical work across arms, so its spread *is* the session's noise floor, and it also
catches ordering bias -- in the 2026-10-27 sweep it showed `dim 64` running 0.31 ms slower on identical work purely
because it went second.

**`calibrate_rescore.py` is post-hoc and monotone, and that is the whole safety argument**. It fits
`calibrated = (logit + bias) / temperature` on **val** by binary NLL, so **AP, recall at matched false positives and
every other rank-based number cannot move** -- only the fixed-threshold columns do, which is the decalibration the diagnosis
phase 1b measured and never fixed. Four things to know before using it:

- **It refuses `--split test`**, because a transform chosen on the split it is reported on is not a measurement.
- **The fitted pair lives in the checkpoint's top-level `calibration` key**, not in `args`: it is fitted after training,
  so recording it as a training argument would claim something that never happened. `object_memory` reads it back and
  **falls back to the identity `(1.0, 0.0)`**, so every checkpoint written before the key scores bit-identically.
- **It is applied to the candidate logit only**, never to the coasting head's. Without that head `_coasting` pins the
  slot logit at `SILENT_COAST_LOGIT`, and a bias there would un-silence the coasting head that was removed.
- **A write refuses when the label cross-check disagrees** with the reported metric by more than 2 pp, or when AP moved
  by more than `AP_TOLERANCE` (1e-6). `_prec_rec_2d` is deliberately *not* refactored to supply the labels, because
  every published number in this repo rests on it, so the two must be compared rather than assumed equal — **at the
  operating threshold on both sides**, which is the only place they are comparable. All three freeze seeds agreed to
  ~0.001.
- **The labels follow the metric's rule, not `candidate_targets`'.** `greedy_labels` accepts detections in descending
  score order, one-to-one within the match radius, exactly as `_prec_rec_2d` accepts its prefix. `candidate_targets`
  minimises **distance** and is score-blind, so it gives a person to a candidate 0.1 m away scoring 0.05 over one
  0.3 m away scoring 0.9 — labelling the metric's own true positive a negative. That noise is anti-correlated with the
  score, which is the one kind a temperature fit cannot survive. It cost the first attempt (`CHANGELOG.md`).
- **Use `--apply <json>` to store a pair, not a fresh `--write`.** Writing is two floats; re-running the fit to do it
  redoes ~55 min of scoring per seed. **Where the hour goes:** ~8 min of GPU replay, then `score_detections` **twice**
  (uncalibrated and calibrated) over 2.1 M detections. So `--frames` buys reproducibility — a re-fit needs no GPU, no
  checkpoints and no 1 GB candidate cache — and **not** wall clock.
- **Run the seeds through `run_queue.py`** (`utils/queue_score_calibration.py`), never a shell loop: the script's `os._exit` still
  leaves the process lingering, and in a loop that hang blocks the next seed. The runner caps it at one
  `EXIT_GRACE_SECONDS`.

**`--state-grad-clip FLOAT`** bounds each frame's gradient at the carried slot state, in the
training loop rather than the model, so the forward is bit-identical. **It is off by default and must stay off** for
any arm being compared with one already measured: `a74_fixed256_s0`'s 85.73% trained without it. It is needed only by
`--iterations > 1`, whose residual update compounds 2.1x per frame and overflows float32 about 60 frames into a chunk.
The epoch line then reports `clipped N/M frames`; a high share means the estimator was modified heavily and the arm's
number needs that said out loud. **Calibrate the bound before trusting an arm that uses it**: replay a chunk through a
healthy arm with the hook counting only, read the p100 per-frame state-gradient norm, and set the bound an order of
magnitude above it. No value is baked in, and the 10.0 used in the gradient probe was arbitrary. `GRAD_CLIP` cannot substitute -- it runs after `backward` has already finished.

**Pricing one flag against another** uses the same latency sweep, one `--memory` per arm, which is how `iterations`
1-against-3 was priced at 19.65 against 31.45 ms total (2026-10-28). Any two memory checkpoints of the same feature
width are interchangeable here, because only the arithmetic per frame is being timed, not the accuracy -- so arms
trained on different stage 2s can still be compared, provided **one** `--stage2` is passed to both so the `stage2`
phase stays byte-identical.

**Never** use `step3a_results.json`'s `memory_ms_per_frame` instead. It is timed on whatever device trained the arm,
which is the GPU here.

**`queue_freeze.py` is the three-seed freeze**, prepared and **not launched** -- it trains stage-2
seeds 1 and 2 (seed 0 already exists as `a69_no_aggregator_s0`), averages each, then runs the three stage-3 arms, about
13 h. `--list` shows what it would do without running anything. It deliberately contains **no latency job**: the CPU protocol's
protocol is a matched, interleaved sweep on an idle host, and a sequential queue runs each job in its own session,
which is how the wrong `gru_cpu` row (23.87 ms against a true 20.57) was produced.
It is **stored in the checkpoint**, so `evaluate.py` defaults to the partition the weights were trained under and refuses to score two differently-partitioned checkpoints in one pass.
`transferred` shares `official`'s train and val exactly, so one checkpoint serves both; `balanced` does not, and its val contains a recording the published baselines trained on.

`--detector` choices: `algorithmic` (eval only, not trainable), `drow` (DROW3's cutout network, trainable since 2026-11-08 so DR-SPAAM's own recipe can be re-run as a known answer), `spacetime_cnn`, `li2former`, `li2former_official`.
**`li2former` and `li2former_official` are different architectures, and only the second one's numbers may go in a Li2Former
row**: the first is this project's reading of the paper's prose (4.70 M parameters), the second wraps the authors' released
network (16.85 M as trained here, 29.47 M as their `__init__` builds it), downloaded into the gitignored `include/` tree by
`library/setup.py` and never vendored because it is GPL-3.0 (`detector-architectures.md`).
`li2former_official` takes no `--dropout` or `--time-frame`: it is built from their configuration, with **three stated
departures from their `ros_li2former.yaml`, all required to reach the paper's model** -- `MODE: PARALLEL` instead of their
deployment setting `TEMPORAL` (which never calls the spatial encoder, leaving 22.07 M parameters without a gradient),
`add_bf=False`, because their `SERIAL`/`PARALLEL` branches cannot run the auxiliary stream they hardcode on, and
**`NUM_CTS: 720` instead of their robot's 90**, which is the cap on their position encoding and so on the number of beams
a scan may have: DROW has 450, JRDB 541, FROG 720, and under 90 the omni-dimension modes die on the first batch with
`The size of tensor a (450) must match the size of tensor b (90)`. It changes no parameter -- the encoding is a buffer,
closed-form in the position -- and `ConvBackbone` ignores the value entirely.
**It needs `--lr 1e-4`**, not `train.py`'s `1e-3` default: that default is tuned for this repo's small CNNs and killed the
regression head in two epochs on a 29.5 M pre-LN transformer.
(`fullscan_tcn` and `temporal_unet` were retired on 2026-10-31; their code last existed at commit `98b0716`.)
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
python evaluate.py --dataset drow --split test --eval-r 0.5 --upstream-grid --no-bench --li2former-official ../checkpoints_drow/li2former_official_parallel_resumed.best.pth
```

`--eval-zero-history` blanks every frame of the temporal window but the current one, at eval time only — the one-pass measurement of whether a trained model actually uses its history. `train.py --zero-history` is the matching train-time ablation.

`--eval-batch-size N` (verified numerically equivalent to `batch_size=1`, much faster) and `--eval-stride N` (systematic frame subsample — see `interpreting-evaluation.md` before trusting a strided FROG number) both cut a full-test-split run from hours to minutes.
`evaluate.py` flags mirror the detector registry: `--drow [WEIGHTS]`, `--drspaam [WEIGHTS]`, `--spacetime-cnn WEIGHTS`, `--fullscan-tcn WEIGHTS`, `--temporal-unet WEIGHTS`, `--li2former WEIGHTS`, `--li2former-official WEIGHTS`, `--lfe-peaks [WEIGHTS]`, `--lfe-ppn [WEIGHTS]` (omit `WEIGHTS` to use bundled/published weights where available).
`--out FILE` also writes each detector's four AUCs as JSON, which is what lets a queue tell a finished evaluation from an interrupted one.
`--upstream-grid` runs inference **once** and scores that one pass four ways -- our decoder and Li2Former's, crossed with our wp-AUC and their `calPrecisionRecall` AP -- so a difference between the cells lands on a named half of the measurement rather than on "the harness". It needs the upstream tree in `include/`, costs one CPU pass per radius (~2 h on DROW test), and its baseline cell reproduces `--eval-r`'s own number to 1e-9. The 2026-11-03 answer: the whole decode-and-score path is worth **+0.1 pp at `d = 0.5` and +0.5 pp at `d = 0.3`** (`performance-log.md`).
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
python horizon_sweep.py --discrimination                # the role-C bound on POPULATED frames, where a filter can destroy people too
python range_probe.py                                   # recall vs range, and the sub-threshold miss budget
python repr_probe.py                                    # scale vs capacity, 1-NN over four window parameterisations
python nms_sweep.py --skip lfe_ppn         # range-aware merge radius r(d) = a + b*d
python nms_sweep.py --skip lfe_ppn --score-by operating-point   # the same radius, scored on recall/duplicates/close-pairs instead of AP
python persistence_map.py                               # role C: world-location persistence over detections
python persistence_map.py --background                  # role C: background model from raw scan returns
python slot_budget.py                                   # how many object-memory slots K stage 3 needs, per retire window
python slot_budget.py --memory                          # value-ranked slot eviction vs unlimited K and a person-aware oracle
```

**`nms_sweep.py --score-by` picks the question, and the two answers disagree.**
`ap` (the default) sweeps against average precision; `operating-point` sweeps against what Tier 0 measured — recall, duplicates, phantoms and close-pair misses at a fixed threshold.
AP prefers a *larger* merge radius than the operating-point view does, because a duplicate is a full false positive while a close-pair miss costs only one missed person (`rejected-ideas.md`).
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
.venv/Scripts/python.exe -m pytest tests -q               # 1,218 tests, ~105 s, CPU-only, no GPU or dataset needed — run after every edit
python memory/scripts/machine_is_free.py && python -m pytest tests -q   # the guard first: the suite has killed two live training runs
python memory/scripts/verify_memory.py                    # docs: links, indexes, markdown rules
python memory/scripts/verify_memory.py --strict            # same, but style warnings fail the run too
.venv/Scripts/python.exe -I utils/check_upstream_convention.py   # their decoder against real DROW frames and oracle votes, not synthetic ones (rule 4)
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

## Another dataset

The dataset is a flag, and it defaults to `frog`, so every command on this page is unchanged by its existence.

```bash
# DROW, natively, at its own sensor limit -- the two values the dataset sets, and nothing else.
# `--max-range-m` is **step2's flag alone**: it shapes the network, so `network_shape` restores it from the
# stage-2 checkpoint and `step3a` rejects it. Passing it there stopped a chain three times in nine seconds.
python train_takheliped.py step2  --dataset drow --max-range-m 29.99 --no-aggregator --label-sigma-m 0.3
python train_takheliped.py step3a --dataset drow --stage2 <averaged> --chunk-s 10

# FROG read at DROW's 0.5 deg spacing: every second beam, 360 of them, 45 sectors. Unlike the range limit
# this one **is** every stage's flag, because `dataset_spec` reads it from the command line rather than
# from the checkpoint -- so a stage that omits it trains at 0.25 deg on 0.5 deg weights and nothing objects.
python train_takheliped.py step2  --beam-stride 2 ...
python train_takheliped.py step3a --beam-stride 2 --stage2 <averaged> ...

# Stop training the non-person classes as background: the beams a wheelchair or a walker covers leave the
# classification loss, neither positive nor negative. Stage 2's alone, and a no-op on a one-class dataset.
python train_takheliped.py step2 --dataset drow --max-range-m 29.99 --ignore-other-classes

# FROG's stage 2 with DROW's supervision: only the training frames in DROW's pattern keep their labels (bursts of 20 at
# every 5th scan, one per 400 scans of each recording, 5% of FROG's frames); val and test are untouched.
python train_takheliped.py step2 --annotation-pattern drow --checkpoint-every 0.25 --no-aggregator --label-sigma-m 0.3 --seed 0 --out-dir ../checkpoints_three_horizon/frog_drow_pattern_s0

# The cutout's depth tunnel on the ten beam features: every depth delta clipped to DROW3's 1 m.
python train_takheliped.py step2 --dataset drow --max-range-m 29.99 --input-features tunnel ...

# One label radius per class, in the spec's class order (DROW: pedestrian wheelchair walker); one value still means
# every class. These are DROW's and DR-SPAAM's own radii, and a beam inside two disks goes to the one it is deepest in.
python train_takheliped.py step2 --dataset drow --max-range-m 29.99 --head multiclass --label-radius 0.35 0.6 0.4

# Stage 2 alone on any split, through the pass step2 scores its own test split with -- train beside test is
# fit against generalisation. Without --memory, `score` takes no stage 3.
python train_takheliped.py score --dataset drow --stage2 ../checkpoints_three_horizon/drow_stage2_s0/step2_calibration.best.pth --split train --test-strides 1 --out ../results/fit_drow_stage2_s0_train.json

# Is the path every TAKHeLiPeD DROW number came through the same as evaluate.py's, on the real files? FrameSplit
# against DROW_Dataset on every split, then DROW3's own detections re-scored on FrameSplit's labels (CPU, ~30 min).
.venv/Scripts/python.exe -I utils/check_drow_scoring.py
# Where DROW's labels sit on its scans, what a 0.6 m and a 0.35 m person disk take in, and how often a walker or a
# wheelchair has a person on top of it. Model-free, minutes.
.venv/Scripts/python.exe -I utils/drow_label_geometry.py

# the published schedule on DROW's stage 2 (40 epochs on a fixed decay, last weights), two seeds; then the levers
# that need no code, two seeds each (DROW's person radius, LFE's input, DROW's three radii on the multi-class head)
python queue_drow_schedule.py --list
python queue_drow_levers.py --list
# drastic capacity: our stage 2 at 15x and 58x fewer parameters and an LFE-shaped proxy, each with its train/test ratio
python queue_drow_capacity.py --list
# DROW's person radius, 0.35 m against 0.6 m, seeds 2-4 of both, so each has five
python queue_drow_radius_seeds.py --list

# Score a finished run a second time against a restricted label set, without retraining or replaying twice.
# Prints and stores both readings: labels truncated alone, and labels with the far detections dropped too.
python train_takheliped.py score --dataset drow --stage2 <averaged> --memory <step3a best> \
    --eval-max-range-m 10 --test-strides 1 5 --out ../results/drow_within_10m.json

# the whole DROW chain, with the two stage-2 arms that make its number readable
python queue_drow.py --list
python queue_drow.py

# the seed-and-sigmoid queue: the item-88 re-run, a second seed on the headline and on the sigmoid arm,
# then one capped stage 3 off the fixed arm (~5.5 h, of which 70 min is the candidate cache)
python queue_drow_seeds.py --list
python queue_drow_seeds.py

# the head queue: the FROG regression first, then DROW's classes as classes (softmax and sigmoid), the range
# span from both sides, the metric gather and the kernel arc -- and one stage 3 off the softmax arm (~9 h)
python queue_drow_heads.py --list
python queue_drow_heads.py

# a multi-class stage 2 on DROW: three columns plus the shared vote, scored on the person class as always
python train_takheliped.py step2 --dataset drow --max-range-m 29.99 --head multiclass --class-norm softmax \
    --checkpoint-every 0.25 --no-aggregator --label-sigma-m 0.3 --seed 0 --out-dir ../checkpoints_three_horizon/<name>

# the handling queue: the 10 m column, the ignore-region arm, and the FROG 0.5 deg chain beside them
python queue_drow_handling.py --list
python queue_drow_handling.py

# the resolution-cost queue: the CPU timing the resolution answer was missing, the three re-ranks, and DROW's
# first chunk length that spans a second annotation burst. The timing jobs run first because the protocol is an
# idle host, so nothing touching the GPU may precede them.
python queue_resolution_cost.py --list
python queue_resolution_cost.py

# the CPU sweep, told which sensor it is timing -- without --beam-stride it feeds a 360-beam network 720 beams
python tracking_baselines.py latency --device cpu --threads 4 --frames 2000 --beam-stride 2     --stage2 <0.5 deg averaged> --memory <0.5 deg step3a best> --out ../results/latency_resolution_half
```

Every run prints its dataset's parameter table before it loads anything, and stores `dataset` and
`beam_stride` in each checkpoint. A run whose range limit cannot represent some of its labels says so, with
the count and the furthest one -- which is the only warning that a forgotten `--max-range-m` on DROW is
training against 8% of its labels blind.

`evaluate.py --dataset drow` honours `--split` as of 2026-11-05; before that it always loaded the test set.
