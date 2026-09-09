# Detector architectures — how each one turns a scan into detections

*keywords:* detector, cutout, full-scan, SpaceTimeCNN, FullScanTCN, TemporalUNet, DrowDetector, DrSpaamDetector, LFE, DETECTOR_REGISTRY, source of truth

The full research write-up — literature review, citations, and the fidelity audit against each paper — is [`../docs/RESEARCH.md`](../docs/RESEARCH.md).
This file is the short, operational version: what an agent needs to predict a detector's behaviour without reading that whole document.

## The core idea

Every detector in this project answers the same question — for each beam in a 2D LiDAR scan, is there a person/wheelchair/walker there, and where exactly — but takes one of two structurally different routes to it.
**Cutout-based** detectors (`DrowDetector`, `DrSpaamDetector`, `Li2FormerDetector`) extract a fixed-width polar window around each beam and run a shared per-beam CNN over it independently, with no or only local cross-beam communication.
**Full-scan** detectors (`SpaceTimeCNNDetector`, `FullScanTCNDetector`, `TemporalUNetDetector`, and the ONNX-only `LFEPeaksDetector`/`LFEPPNDetector`) process the entire raw scan at once, with dilated convolutions standing in for the cutout's fixed window.
The research question this project exists to answer: can a full-scan, non-recursive (no RNN, no attention) design match or beat the cutout-based state of the art, while being far cheaper per frame.

## Lifecycle

1. **Ingest** one or `T` consecutive scans (`get_scan()` on the active dataset, or `LiveDataset.push_measure()` for ROS).
2. **Align** (optional, `--align-scans`, default on) — historical frames are rotated by the accumulated odometry delta into the current frame, via `aligned_raw_scan()` (full-scan) or the cutout's own built-in rotation (cutout-based).
   With zero or absent odometry this is a no-op, not an error.
3. **Extract input** — a `(N_beams, T, N_SAMP)` cutout tensor, or a `(N_beams, T, 1)` raw-range tensor.
4. **Forward pass** — per-beam classification logits + a 2D vote offset (the DROW-style head), or a per-beam heatmap value (the LFE-style head, also available on Architectures A/B via `--head heatmap`).
5. **Cluster** votes/heatmap peaks into `(x, y, class, confidence)` detections (`votes_to_detections()` / `find_peaks`).
6. **(optional) Track** — `SimpleTracker` (training/eval side, `library/follow_the_drow/utils/tracking.py`) or `PersonTracker` (the ROS node) turns the per-frame detection list into one persistent, ID-stable position, predicting through missed frames up to `max_age` and requiring `min_hits` real matches before reporting a track at all.

Every stage after preprocessing is shared by every detector; only the extraction (step 3) and the head (step 4) differ by family.

## The three proposed architectures

| Detector | Beam communication | Temporal fusion order | Cost | Best FROG wp-AUC |
| --- | --- | --- | --- | --- |
| `SpaceTimeCNNDetector` (A) | local, joint with time from layer 1 | joint | ~9-12 ms/frame | **highest of the three** (`performance-log.md`) |
| `FullScanTCNDetector` (B) | local, growing via dilation | time-then-space (causal TCN first) | ~2-3 ms/frame | lower than A, higher speed |
| `TemporalUNetDetector` (C) | global (bottleneck) + local (skips) | joint (T input channels) | ~2-3 ms/frame | lowest of the three |

A vs. B is a deliberate ablation: does fusing space and time jointly from the first layer beat collapsing time first and mixing space afterward?
A wins on accuracy, B and C win on speed — consistent with the general video-CNN finding that joint processing usually edges out factorized designs on accuracy while factorized designs win on speed.
Two earlier designs (`FullScanTransformerDetector`, `FullScanCNNDetector`) used global self-attention and a GRU respectively and were removed for violating the non-recursive constraint; see `rejected-ideas.md` for why they were also worse, not merely disqualified on principle.

## Source of truth

`DETECTOR_REGISTRY` (`library/follow_the_drow/detectors/__init__.py`) is the authoritative list of what detectors exist and their canonical `--detector`/CLI key.
A detector class not in this dict cannot be trained or evaluated through `train.py`/`evaluate.py`, regardless of whether the class itself exists — absence from the registry means "not wired up," not "does not exist."
`library/follow_the_drow/utils/torch_utils.py` looks like it should own device selection and does not — see `gotchas.md`.

## Components

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
| session-split threshold | 300 s | `frog_dataset.py`'s `_load_h5()` | Any real timestamp gap wider than this starts a new sequence, so temporal windows never cross it — chosen to isolate FROG's one genuine ~24.6h recording-session boundary without splitting on the 61 sub-90s in-session pauses. |
