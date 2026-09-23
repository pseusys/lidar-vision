# Detector architectures — how each one turns a scan into detections

*keywords:* detector, cutout, full-scan, TAKHeLiPeD, three-horizon, calibration network, object memory, SpaceTimeCNN, FullScanTCN, TemporalUNet, DrowDetector, DrSpaamDetector, LFE, DETECTOR_REGISTRY, source of truth

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
| 2, calibration network | 1D ConvNeXt U-Net over the whole scan, causal temporal convolutions per sector at the bottleneck with their caches odometry-re-aligned; a DROW vote head decodes 64 candidates, sub-threshold ones included | 1.18 M | ~10.4 ms/frame with decoding |
| 3, object memory | 256 slots in **room** coordinates; cross-attention association, slot self-attention, and a Δt-aware selective diagonal recurrence (Mamba-style, decay 1-60 s). Rescores candidates, coasts briefly-missed people, optionally renders slots back into stage 2 | 0.31 M | ~4.4 ms/frame |

`step()` carries `(CalibrationState, Slots)` between frames, ~1.8 MB, and the per-frame cost does not grow with how far back the memory reaches.
**`manage_slots` runs under `no_grad`** — association, value accumulation, spawn, replacement and retirement are fixed rules; only what reads the slots learns.
Design rationale: [`../docs/PROPOSAL.md`](../docs/PROPOSAL.md). Evidence for each slot decision, or an explicit note that there is none: [`slot-design-evidence.md`](slot-design-evidence.md). Numbers: [`performance-log.md`](performance-log.md).

**Two architecture names appear in the logs.** With `--fine-lags` removed on 2026-09-17, the runs formerly called "no-fine" are the **default** architecture (coarse convolution only) and "static" is **no-temporal** (`--coarse-lags 0`); directory names keep the old words. The B0 chain and the A51 slot ablations sit on the no-temporal stage 2, by the owner's call.

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

- **`three_horizon.py`** — TAKHeLiPeD: `beam_features`/`sanitize_ranges`, `CalibrationNetwork` + `CalibrationState`, `ObjectMemory` + `Slots`/`SlotRules`/`manage_slots`, and `calibration_network()`/`object_memory()`, which rebuild either stage the way its checkpoint was trained. Not in `DETECTOR_REGISTRY` — it is stateful and has its own trainer.
- **`architectures.py`** — `DrSpaamDetector` (cutout + auto-regressive spatial attention).
- **`drow_detector.py`** — `DrowDetector` (cutout + fixed temporal sum); the only detector the ROS deployment currently runs (`deployment.md`).
- **`full_scan.py`** — `SpaceTimeCNNDetector`, `FullScanTCNDetector`, `TemporalUNetDetector`; this project's own novel designs.
- **`li2former.py`** — `Li2FormerDetector` (cutout + temporal Transformer); no published weights exist, and training has never completed on this project's hardware (`gotchas.md` I4).
- **`lfe_detector.py`** — `LFEPeaksDetector`/`LFEPPNDetector`, ONNX inference only, single-scan (no temporal window at all); the closest prior art to Architectures A-C.
- **`algorithmic_detector.py`** — wraps the C++ `AlgorithmicDetector` (rule-based leg+chest clustering + tracking); the only detector with no confidence score, so it reports F1/precision/recall instead of AUC.
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
