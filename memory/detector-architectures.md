# Detector architectures — how each one turns a scan into detections

*keywords:* detector, cutout, full-scan, TAKHeLiPeD, three-horizon, calibration network, object memory, SpaceTimeCNN, FullScanTCN, TemporalUNet, DrowDetector, DrSpaamDetector, LFE, AlgorithmicDetector, Aycard, DETECTOR_REGISTRY, source of truth

The full research write-up — literature review, citations, and the fidelity audit against each paper — is [`../docs/RESEARCH.md`](../docs/RESEARCH.md).
This file is the short, operational version: what an agent needs to predict a detector's behaviour without reading that whole document.

## The core idea

Every detector in this project answers the same question — for each beam in a 2D LiDAR scan, is there a person/wheelchair/walker there, and where exactly — but takes one of two structurally different routes to it.
**Cutout-based** detectors (`DrowDetector`, `DrSpaamDetector`, `Li2FormerDetector`) extract a fixed-width polar window around each beam and run a shared per-beam CNN over it independently, with no or only local cross-beam communication.
**Full-scan** detectors (`SpaceTimeCNNDetector`, `FullScanTCNDetector`, `TemporalUNetDetector`, and the ONNX-only `LFEPeaksDetector`/`LFEPPNDetector`) process the entire raw scan at once, with dilated convolutions standing in for the cutout's fixed window.
The research question this project started from: can a full-scan, non-recursive (no RNN, no attention) design match or beat the cutout-based state of the art, while being far cheaper per frame.
**That framing was superseded on 2026-09-14.** `TAKHeLiPeD` (below) keeps the full-scan half and deliberately lifts the non-recursive constraint, because the evidence that separates a chair from a standing person only exists at a horizon of tens of seconds, which no feed-forward window can reach (`../docs/PAPER.md`).

## Lifecycle

1. **Ingest** one or `T` consecutive scans (`get_scan()` on the active dataset, or `LiveDataset.push_measure()` for ROS).
2. **Align** (optional, `--align-scans`, default on) — historical frames are rotated by the accumulated odometry delta into the current frame, via `aligned_raw_scan()` (full-scan) or the cutout's own built-in rotation (cutout-based).
   With zero or absent odometry this is a no-op, not an error.
3. **Extract input** — a `(N_beams, T, N_SAMP)` cutout tensor, or a `(N_beams, T, 1)` raw-range tensor.
4. **Forward pass** — per-beam classification logits + a 2D vote offset (the DROW-style head), or a per-beam heatmap value (the LFE-style head, also available on Architectures A/B via `--head heatmap`).
5. **Cluster** votes/heatmap peaks into `(x, y, class, confidence)` detections (`votes_to_detections()` / `find_peaks`).
6. **(optional) Track** — `SimpleTracker` (training/eval side, `library/follow_the_drow/utils/tracking.py`) or `PersonTracker` (the ROS node) turns the per-frame detection list into one persistent, ID-stable position, predicting through missed frames up to `max_age` and requiring `min_hits` real matches before reporting a track at all.

Every stage after preprocessing is shared by every detector; only the extraction (step 3) and the head (step 4) differ by family.

## TAKHeLiPeD — the current detector

**T**(emporal) **A**(daptive) **K**(nee-)**He**(ight) **Li**(dar) **Pe**(rson) **D**(etector), named 2026-09-23 — *TA-KHé-Li-PeD*.
Everything written before that date calls it **the three-horizon detector**, and the code still spells it `three_horizon`: `library/follow_the_drow/detectors/three_horizon.py`, `utils/train_three_horizon.py`, `checkpoints_three_horizon/`.
It is the only detector here that is **stateful across frames**, so it does not go through `DETECTOR_REGISTRY`, `train.py` or `evaluate.py` at all — it has its own trainer with its own steps (`commands.md`).

It breaks the lifecycle above in two places: there is no `T`-scan window (one scan per call), and there is no post-hoc tracker (the memory replaces it, and beats it — `performance-log.md`).

| Stage | What it is | Parameters | Cost |
| --- | --- | --- | --- |
| 1, features | 10 pose-invariant per-beam features from one raw scan; `sanitize_ranges` maps non-finite, non-positive and beyond-`max_range_m` readings to `max_range_m`, so no dataset-specific preprocessing is needed | — | — |
| 2, calibration network | 1D ConvNeXt U-Net over **one scan**, a 3-tap convolution across bottleneck sectors and a global max; a DROW vote head, decoded by `nms-mean` into 64 candidates, sub-threshold ones included. No state, no pose: its coarse temporal convolution was removed 2026-10-08 (A60) | 0.98 M | 8.75 ms + 2.77 ms decoding, CPU, 4 threads |
| 3, object memory | 256 slots in **room** coordinates; cross-attention association, slot self-attention, and a Δt-aware selective diagonal recurrence (Mamba-style, decay 1-60 s). Rescores candidates and coasts briefly-missed people | 0.31 M | ~4.4 ms/frame |

Only stage 3 carries state between frames: its slots, ~138 KB per stream (the coarse convolution's ~1.8 MB cache went with it), and the per-frame cost does not grow with how far back the memory reaches. The whole detector streams at 20.8 ms per frame (p99 24.1) on this machine's CPU at 4 threads, inside FROG's 25 ms scan period (`performance-log.md`).
**`manage_slots` runs under `no_grad`** — association, value accumulation, spawn, replacement and retirement are fixed rules; only what reads the slots learns.

**What A51 settled about stage 3 (2026-09-29/30), and what the write-up should propose.**

- **Keep the learned slot state.** It is the only part whose removal has ever cost anything: −1.50 pp AP, −1.67 pp recall at matched false positives, −14.08 pp coasting precision, all past their bars (E10).
- **Drop both attention mechanisms.** Rule-based association plus the per-slot recurrence ties the full model on every metric at three seeds, at **173,579 parameters against 306,443 (−43%)** and **3.72 ms against 4.53 (−18%)** (E6, E7, E13). This is the architecture to propose; the full model is the ablation.
- **Drop the feedback path.** `render_prior` and `prior_channels` contribute 0.00 pp in three configurations, because the prior carries 0.012–0.045% of `aggregate`'s input and its exclusive ceiling is 1.91 pp of recall anyway (E12). Dropping it also drops the `step3b` joint stage, which already scored worse than `step3a` (78.84% against 80.10%) — this session localised that loss to the **rescoring head's** drift on stage 2, not the prior.
- **Still there, still conditional:** room coordinates win only below ~0.02 m/s of odometry drift (D1, E2); training with jittered poses is the untried repair (`../TODO.md` E11).
Design rationale: [`../docs/PROPOSAL.md`](../docs/PROPOSAL.md). Evidence for each slot decision, or an explicit note that there is none: [`slot-design-evidence.md`](slot-design-evidence.md). Numbers: [`performance-log.md`](performance-log.md).

**Stage 2's recipe, decided 2026-10-06 (owner, `../TODO.md` A59):** decode with `nms-mean` (NMS over the beams' votes, each detection at the probability-weighted mean of the votes it suppresses), train with mirror augmentation, and assign each training beam to its nearest person within 0.6 m. Together these take the no-temporal stage 2 from 77.1% to ~81.5% on FROG test; a second median channel (F3) was tested and dropped (`rejected-ideas.md`). **With dropout 0.1 and weight averaging, also adopted 2026-10-06** (`step2 --dropout 0.1 --checkpoint-every 0.25`, then `utils/average_checkpoints.py`), the no-temporal stage 2 reaches **82.20%**. **They are the trainer's defaults since 2026-10-06** (`decode_frame`, `step2 --label-assign nearest --mirror --dropout 0.1`; averaging is its own step); every stage-3 number until A59 phase 3a rests on the old recipe -- vote grid, no mirror, last-listed labels.

**What A59 found about stage 2 (2026-10-05)** (`static-detector-diagnosis.md`, last section):

- **Its decoder understates it.** The official DR-SPAAM decoder (`nms_predicted_center`, 0.5 m) on the same outputs scores +2.33 to +3.03 pp higher on all six stage-2 checkpoints.
  The vote grid stays the default until a decoder is chosen on val and stage 3 is retrained on the new candidates.
- **Its gain over LFE is far and isolated people.** Through the vote grid it also loses close pairs (60% recall against LFE's 83%) and 1-4 m, but both are the vote grid merging neighbours: through `nms-mean` (chosen on val on all six) it reaches 82% of close pairs and beats LFE at every range, at **80.07%** AP.
- **The coarse convolution: dropped by A60's rule and removed (2026-10-08)** -- end to end, on the new recipe, three seeds each, the default architecture scores 85.01% against the no-temporal 85.87%, and +0.59 pp on people in their first second against a +2 pp bar (`../TODO.md` A60). Its code is at commit e76aa94 (`rejected-ideas.md`). Before that measurement it already earned nothing measured: a tie at stage 2 under either decoder, the same recall in every range and crowding bin, at +20.6% parameters, ~3.3x the training time, a 1.8 MB streaming cache and odometry inside stage 2.
  The end-to-end test, with the rule that drops it, is `../TODO.md` A60; until it runs, the no-temporal stage 2 -- one frame per call -- is the one every stage-3 result uses.

**Older architecture names in the logs.** Between 2026-09-17 and 2026-10-08 the runs called "no-fine" or "default" kept the coarse convolution and "static" or "no-temporal" did not (`--coarse-lags 0`); directory names keep the old words. Only the no-temporal kind still loads (its one-lag `coarse.conv` folds into `context`), and it is what stage 2 now is; the others are refused with a pointer to e76aa94.

## The three earlier full-scan architectures — superseded

Preceded TAKHeLiPeD and still in `DETECTOR_REGISTRY`; every accuracy number they carried is retracted (`performance-log.md`), so they are provenance, not results.

| Detector | Beam communication | Temporal fusion order | Cost | Best FROG wp-AUC |
| --- | --- | --- | --- | --- |
| `SpaceTimeCNNDetector` (A) | local, joint with time from layer 1 | joint | ~9-12 ms/frame | retracted (`performance-log.md`) |
| `FullScanTCNDetector` (B) | local, growing via dilation | time-then-space (causal TCN first) | ~2-3 ms/frame | retracted |
| `TemporalUNetDetector` (C) | global (bottleneck) + local (skips) | joint (T input channels) | ~2-3 ms/frame | retracted |

A vs. B is a deliberate ablation: does fusing space and time jointly from the first layer beat collapsing time first and mixing space afterward?
The accuracy ordering that used to sit in this table (A > B > C) is **retracted pending re-measurement** — every FROG number behind it carries at least one of the four 2026-09-10 measurement bugs, and A18 in particular hit A and B but not C, so the three were not even measured the same way.
The speed ordering (B ~ C < A) is unaffected: it is a timing measurement, not an accuracy one.

**All three set `BEAM_BATCH = True` and take `(B, N_beams, T, C)`.**
Every one of them convolves along the beam axis, so a flattened `(B*N_beams, T, C)` minibatch splices B frames into one long scan and lets GroupNorm statistics, the SE gate and the convolutions themselves leak between frames that share a batch.
A and B did exactly that until 2026-09-10; C never could, because it pools over the beam axis.
Measured before the fix, real val split at `dtime`=10: beam-level person AUC 0.9644 (batch 1) -> 0.9696 (4) -> 0.9765 (16) -> 0.9810 (64), because evaluation batches are consecutive frames and a bigger batch leaked more temporal context.
`tests/test_batch_independence.py` now asserts a frame's output is identical alone or inside a batch of 16, for every model and both heads.
Old checkpoints still load — shapes are unchanged — but compute different outputs.
Two earlier designs (`FullScanTransformerDetector`, `FullScanCNNDetector`) used global self-attention and a GRU respectively and were removed for violating the non-recursive constraint; see `rejected-ideas.md` for why they were also worse, not merely disqualified on principle.

## Source of truth

`DETECTOR_REGISTRY` (`library/follow_the_drow/detectors/__init__.py`) is the authoritative list of what detectors exist and their canonical `--detector`/CLI key.
A detector class not in this dict cannot be trained or evaluated through `train.py`/`evaluate.py`, regardless of whether the class itself exists — absence from the registry means "not wired up," not "does not exist."
**TAKHeLiPeD is the standing exception** and is deliberately absent: it is stateful across frames, so it does not fit the registry's stateless `(scan, T) -> detections` contract, and it is trained and scored by `utils/train_three_horizon.py` instead.
`library/follow_the_drow/utils/torch_utils.py` looks like it should own device selection and does not — see `gotchas.md`.

## Components

- **`three_horizon.py`** — TAKHeLiPeD: `beam_features`/`sanitize_ranges`, `CalibrationNetwork` (single-frame; `fold_legacy_calibration` carries older checkpoints over), `ObjectMemory` + `Slots`/`SlotRules`/`manage_slots`, and `calibration_network()`/`object_memory()`, which rebuild either stage the way its checkpoint was trained. Not in `DETECTOR_REGISTRY` — it is stateful and has its own trainer.
- **`architectures.py`** — `DrSpaamDetector` (cutout + auto-regressive spatial attention).
- **`drow_detector.py`** — `DrowDetector` (cutout + fixed temporal sum); the only detector the ROS deployment currently runs (`deployment.md`).
- **`full_scan.py`** — `SpaceTimeCNNDetector`, `FullScanTCNDetector`, `TemporalUNetDetector`; this project's own novel designs.
- **`li2former.py`** — `Li2FormerDetector` (cutout + temporal Transformer); no published weights exist, and training has never completed on this project's hardware (`gotchas.md` I4).
- **`lfe_detector.py`** — `LFEPeaksDetector`/`LFEPPNDetector`, ONNX inference only, single-scan (no temporal window at all); the closest prior art to Architectures A-C.
- **`algorithmic_detector.py`** — wraps the C++ `AlgorithmicDetector` (rule-based leg+chest clustering + tracking); the only detector with no confidence score, so it reports F1/precision/recall instead of AUC.
  **The algorithm is prof. O. Aycard's, not this project's** (`cpp_core/sources/detector.cpp`, "Written by O. Aycard") — credit it as his wherever it is reported; only the binding, the wrapper and the harness are ours (`performance-log.md` footnote 0).
- **`odometry_estimation.py`** — trimmed-ICP scan matching, used only as a fallback when a dataset has no real odometry file; see `data-model.md` for when that fallback actually triggers.
- **`tracking.py`** — `SimpleTracker`, the SORT-style post-hoc tracker described in the lifecycle above.

## Tunables

| Tunable | Value | Read in | What it does |
| --- | --- | --- | --- |
| `time_frame` (`T`) | 5 (default) | `train.py --time-frame`, dataset `get_scan()` | Number of scans in the temporal window. |
| `dtime` | 1 (default) | `train.py --dtime`, `get_scan()` | Real-time stride between the `T` frames — `dtime=5` was FROG's measured accuracy peak on the pre-fix pipeline, since retracted (`performance-log.md`); a wider re-sweep (`{1,5,10,15,20,25}`) on the now-fixed real-odometry pipeline is in progress (`TODO.md` A1), since the old "wider regresses" finding (`rejected-ideas.md`) was caused by uncompensated ego-motion that real odometry now corrects for. |
| `align_scans` | on (default) | `train.py --no-align-scans`, `_extract_input()` | Whether historical frames are odometry-rotation-corrected before extraction. |
| `SimpleTracker.match_radius` | 0.5 m (fine stride) | `tracking.py`, `evaluate_auc(tracker_kwargs=...)` | Max distance to associate a detection with a predicted track position — a single global value, known not to generalize across invocation rates (`rejected-ideas.md`). |
| `SimpleTracker.min_hits` | 3 (default) | same | Real matches required before a track is reported at all. |
| `LFEPeaksDetector.merge_radius` / `LFEPPNDetector.nms_radius` | **0.30 m** both | `lfe_detector.py`'s `_merge_nearby()` | How close two centroids must be to count as one person. Calibrated against each detector's published AP at *both* association distances, not taken from the paper's sentence — which implies the 0.8 m person diameter and is ~3.5pp low for LFE-PPN and ~6pp low for LFE-Peaks. Changing either invalidates every false-positive number in the repo; that has already happened once (`CHANGELOG.md` 2026-09-11). |
| `merge_radius_slope` / `nms_radius_slope` | **0.0** (default = the single scalar above) | same | Makes the radius range-dependent, `r(d) = radius + slope * d`, evaluated at the pair's mean range and floored at zero. Exists because Tier 0 measured one scalar failing in both directions at 0.30 m — ~5 pp of LFE-Peaks' recall spent merging adjacent people, while LFE-PPN emits 1.609 duplicates per frame (`static-detector-diagnosis.md`, `TODO.md` A38). `0.0` is bit-identical to the old behaviour and is guarded by `tests/test_merge_radius.py`. |
| session-split threshold | 300 s | `frog_dataset.py`'s `_load_h5()` | Any real timestamp gap wider than this starts a new sequence, so temporal windows never cross it — chosen to isolate FROG's one genuine ~24.6h recording-session boundary without splitting on the 61 sub-90s in-session pauses. |
| `dropout` | **0.5** (`train.py --dropout`, `_default_args()`) | `_build_model()` | Applied uniformly to every trainable detector — `drow`, `drspaam`, `li2former`, and all three full-scan models alike — via `_build_model()`'s single `dr = getattr(args, "dropout", 0.5)`, which overrides each class's own constructor default (`full_scan.py`'s three classes default to 0.1 when instantiated directly, bypassing `train.py`). §5.4 of `docs/RESEARCH.md` documents 0.1 as the *design intent* for the novel full-scan architectures specifically — that intent was never wired into `train.py`'s actual default, so every full-scan training run through the CLI has used 0.5, not 0.1 (verified by reading `_build_model()` and `_default_args()` directly, 2026-09-10 — an earlier memory note claiming "0.1, light regularisation" was wrong, based on checking only the class constructor, not the CLI override). 0.5 exactly matches the official DR-SPAAM/DROW training recipe's own `dropout: 0.5` (`VisualComputingInstitute/2D_lidar_person_detection`'s `base_dr_spaam_drow_cfg.yaml`, fetched directly 2026-09-10) — so this project's dropout is not under-regularized relative to the papers it ports. |
| `weight_decay` | **1e-4** (`train.py --weight-decay`) | `_make_optimizer()` | The official DR-SPAAM/DROW repo's own optimizer (`dr_spaam/pipeline/optim.py`) uses plain `Adam(..., amsgrad=True)` with **no weight decay at all** — this project's default is already *more* regularized on this axis than the papers it ports, not less. Neither this project nor the official recipe (`augment_data: False` in the same yaml) applies data augmentation by default. |
