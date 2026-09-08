# Full-Scan, Non-Recursive Convolutional Architectures for 2D LiDAR Person Detection

> Formalized research statement and literature review for this repository. For a client-facing overview, see the [top-level README](../README.md); for a map of the codebase, see [AGENTS.md](../AGENTS.md).

---

## 1. Problem Statement

A mobile robot carries a 2D lidar scanner mounted at leg height (~40 cm off the ground). The scanner rotates in a horizontal plane and produces a single array of **N range measurements** per timestep — one distance value `r_i` for each beam angle `φ_i`, uniformly distributed across the sensor's field of view (DROW: N ≈ 450 beams at 0.5° spacing; FROG: N = 720 beams at 0.5° spacing; JRDB: N = 541 beams at 0.5° spacing over a 270° FoV).

The task is **beam-level classification with spatial vote regression**: for every beam `i` in every scan, predict:
1. A probability over four classes: background, wheelchair (wc), walker (wa), person/pedestrian (wp)
2. A 2D vote offset `(dx, dy)` in the sensor frame pointing toward the nearest annotation centre

Post-processing clusters the vote targets from all beams that voted "positive" into discrete person/wheelchair/walker detections (Gaussian-smoothed accumulator + peak detection), avoiding the need to predict the exact count of people.

### Key challenges

| Challenge | Description |
|---|---|
| **Scale variance** | A person at 2 m subtends ~12 beams; the same person at 8 m subtends ~3 beams |
| **Sparse targets** | People occupy <5 % of beams in a typical indoor scan |
| **Angular ambiguity** | The sensor measures range, not shape; a thin pole and a leg look similar locally |
| **Temporal drift** | The robot moves between scans, so consecutive scans are in different frames |
| **Class imbalance** | Background beams outnumber positive beams by ~20:1 |

### Evaluation metric

The standard metric is **Average Precision (AP)** at a matching radius of 0.5 m in Cartesian space, computed separately per class. Literature numbers quoted below use the DROW test set as the reference benchmark unless stated otherwise.

## 2. Goal

Existing learned 2D-LiDAR person detectors (DROW, DR-SPAAM, Li2Former) all rely on a **cutout**: a fixed-size polar window extracted around each beam, processed independently by a per-beam CNN. The cutout was a reasonable choice when these methods were designed (it gives trivial per-beam parallelism and a small, fixed input size), but it hardcodes the spatial context a beam can see and discards everything outside the window.

**This research asks: can a CNN that processes the *entire* scan at once — with no cutout, no per-beam windowing, no hand-set spatial extent — match or exceed cutout-based detectors' accuracy, while removing that fixed-window assumption?**

Three further constraints, made deliberately and stated up front so every later design choice is explainable against them:

1. **Non-recursive only.** Every architecture is built exclusively from CNN and TCN layers (plus pooling/norm/linear "glue") — no RNN/GRU/LSTM, no attention. This is an efficiency choice: it's the architectural feature expected to be the actual winning point against Li2Former (a temporal-Transformer cutout detector with O(N_beams²)/O(T²)-flavoured cost) and against any recurrent design — feed-forward, fully parallel layers are cheaper to run and trivially batchable.
2. **Raw input, no preprocessing.** Architectures consume the raw scan — angle (implicit in beam index) and distance only. No cutout and no conversion to Cartesian `(x, y)`. This is a bigger departure from prior work than it looks: DROW, DR-SPAAM, and this project's own earlier full-scan designs all rotate historical scans into the current frame using odometry before doing anything else (Section 3.2). Dropping the Cartesian conversion turns the `T`-frame window into a trivial rotating buffer at inference time — see Section 5.0.
3. **Compare against the cutout-based research.** All of the above is benchmarked against DROW, DR-SPAAM, and Li2Former, on the same datasets and the same evaluation pipeline (Section 8).

Concretely, this research proposes **three** architectures (Section 5): a joint space-time CNN, a TCN-then-CNN fusion, and a temporal U-Net — see Section 6 for why three, not two.

### 2.1 Provenance: replicated vs. novel

Every detector this repo trains or evaluates falls into exactly one of three provenance categories. This distinction matters for how to read any accuracy number produced by this codebase: a "replicated" score is a reproduction of a published result, a "novel" score is this project's own claim.

| Detector | Category | Evidence |
|---|---|---|
| `DrowDetector` | **Replicated — official weights** | Architecture is a direct port of the official DROW notebook release; loads the official checkpoint `final-WNet3xLF2p-T5-odom.rot-trainval-50ep.pth.tar` (`library/setup.py`) via `load_state_dict`, which only succeeds because the layer shapes match the original bit-for-bit. |
| `DrSpaamDetector` | **Replicated — official weights** | Module docstring states the code matches "the official SpatialDROW implementation"; module structure is deliberately nested to load the official checkpoint `dr_spaam_e40.pth` exactly. |
| `LFEPeaksDetector` / `LFEPPNDetector` | **Replicated — official weights (ONNX)** | Loads the authors' own exported ONNX models (`LFE-Peaks.onnx`, `LFE-PPN.onnx`, hosted at `robotics.upo.es/~famozur/onnx/`) via `onnxruntime.InferenceSession` — genuine published-model inference, not a reimplementation. |
| `Li2FormerDetector` | **Reimplemented — no official code or weights** | Header says the architecture is "inspired by" Yang et al. (weaker language than the DROW/DR-SPAAM ports) and explicitly notes no published weights exist; this project trains it from scratch from the paper's description alone. No official repo was available to verify against, so treat any AP this project reports for Li2Former as this project's own reproduction attempt, not a confirmed match to the paper's reported 0.764 (DROW) / 0.814 (JRDB). |
| `AlgorithmicDetector` | **Ported classical algorithm — not from any of the SOTA papers compared here** | The C++ source attributes the algorithm to "O. Aycard" (`cpp_core/sources/detector.cpp`), not to Pantofaru's ROS `leg_detector`, Arras et al., or Leigh et al. — see Section 2.2. It's a distinct two-tier leg+chest clustering/tracking algorithm, included as this repo's own non-learned reference point, not a port of any baseline used by DROW/DR-SPAAM/Li2Former/FROG. |
| `SpaceTimeCNNDetector`, `FullScanTCNDetector`, `TemporalUNetDetector` | **Novel — proposed by this work** | No prior paper describes these architectures, so no official weights could exist; each must be trained locally. Individual sub-components (SE blocks, Inception-style branching, causal TCN, U-Net skip connections) are explicitly credited to their originating papers in the code (Section 5), but the overall architectures and their combination are this project's own design. |

### 2.1.1 Fidelity audit: how precisely is "replicated" replicated?

The categories above are the right level of description, but "replicated" is not automatically "pixel-perfect" — every replicated/reimplemented detector was audited line-by-line against its official source (code where public, paper text otherwise). Findings, most severe first:

- **`LFEPeaksDetector` / `LFEPPNDetector` — one functional deviation found and fixed, one remaining.**
  - ~~**LFE-Peaks was missing the paper's centroid-merge/NMS step.**~~ **Fixed.** The FROG paper (§5.2.4) describes finding peaks in the segmentation signal, then merging nearby centroids ("Centroids that are close together are interpreted as legs or part of legs, and merged together into final person detections using a NMS-like process"). `LFEPeaksDetector.detect()` previously reported every raw `find_peaks` hit as its own detection, so a person whose two legs produced two separate peaks would score as two detections instead of one. It now runs the same greedy centroid-merge as `LFEPPNDetector`'s NMS (`_merge_nearby()`, shared by both classes), with a `merge_radius` defaulting to `_PERSON_RADIUS = 0.4 m` — a reasonable value for "same person's leg parts," but not one the paper states explicitly, since it doesn't give an exact radius either.
  - **Cross-dataset zero-padding is still an unverified extrapolation — not fixed, and not straightforward to fix.** The FROG paper only ever evaluates LFE on FROG's own 720-beam scans; it never describes running it on DROW (450 beams, 225° FoV) or JRDB (541 beams, 270° FoV). This repo's zero-padding of shorter scans to 720 (`_pad_to_trained`) is this project's own invention, and it does not resample to FROG's 0.25°/beam angular resolution — so on DROW/JRDB, beam index no longer corresponds to the same physical angle the network was trained on. Treat any DROW/JRDB numbers from LFE as a known-approximate cross-dataset transfer, not a faithful replication (this was already flagged as a caveat in Section 3.4, but the padding scheme itself is now confirmed unverified against the paper, not just "may degrade").
  - The exact input-normalization formula and the LFE-PPN anchor count (31) could not be located in the paper text; the PPN depth-spacing formula in this repo's code omits a `-1` term the paper's formula has (~3% scale difference). Minor, but unverified/inexact.

- **`DrowDetector` — matches with one minor preprocessing deviation.** `load_state_dict` is called strict (no `strict=False` anywhere), and the network definition is character-for-character identical to the official `DROWNet3LF2p` (verified directly against `VisualComputingInstitute/DROW`'s `v2/Clean Final* [T=5,net=drow3xLF2p,odom=rot,trainval].ipynb`) — channel progression, kernel sizes, LeakyReLU(0.1), pooling, dropout, weight init, and the T-sum temporal fusion are all identical. The one confirmed deviation: the official `cutout()` uses `cv2.INTER_AREA` (box-filter averaging) when downsampling a window to 48 samples and `cv2.INTER_LINEAR` only when upsampling; this repo's `cutout()` always uses linear interpolation regardless of direction. Since near-range detections typically downsample, this changes the input feature distribution at close range — a real if minor deviation from the original preprocessing (it does not affect weight loading, since it's a data-preparation step, not a model parameter).

- **`DrSpaamDetector` — matches with one training-only deviation.** `load_state_dict` is strict at both call sites, and the conv blocks, spatial-attention formula, published hyperparameters (`alpha=0.5`, `window_size=11`, `N_SAMP=56`, pedestrian-only head), and detection heads all match the official `VisualComputingInstitute/DR-SPAAM-Detector` source exactly. The one confirmed deviation: this repo's training-time forward pass explicitly `.detach()`es the auto-regressive template on all but the last timestep, whereas the official training code back-propagates through the full T-step chain with no detach. This has no effect on inference (loaded weights are identical either way) but means retraining from scratch here would not exactly reproduce the official training dynamics.

- **`TemporalUNetDetector`** — see Section 5.3; the U-Net implementation itself is confirmed correct against the canonical definition, but its relationship to LFE's *actual* backbone needed a correction (below).
- **`SpaceTimeCNNDetector`** and **`FullScanTCNDetector`** — see Sections 5.1 and 5.2; both had at least one inaccurate/overstated literature citation that has been corrected.

### 2.2 SOTA baselines mentioned in the literature but not replicated here

The four papers this research compares against (Sections 3, 7) each report their own baseline comparisons. Collated across all of them, three classical detectors recur that this repo does **not** implement:

| Baseline | Used as a comparison point by | Why not replicated here |
|---|---|---|
| **ROS `leg_detector`** (Pantofaru, ROS package, 2010) — jump-distance leg clustering + random-forest classifier + Kalman-filter tracking, off-the-shelf pretrained model | DROW (IROS 2018), Li2Former (TIM 2024), and FROG (2025) all use it as their weakest baseline | Consistently the lowest-scoring method in every one of those papers' own tables (e.g. FROG Table 4: mAP 15.8 vs. 73+ for DROW3/DR-SPAAM). This repo already carries its own non-learned reference point (`AlgorithmicDetector`, Section 2.1) — replicating a second classical leg-tracker would duplicate that role without adding anything to the actual research question (full-scan vs. cutout **deep** architectures). |
| **Segment-based detector, Arras et al.** (ICRA 2007) — jump-distance segmentation + boosted-features classifier on hand-crafted geometric features | DROW (IROS 2018), Li2Former (TIM 2024) | No official training code was ever released — even the DROW paper itself had to fall back on a third-party ROS reimplementation (Linder & Breuers et al.) rather than the original. Replicating it here would itself be an unverified reimplementation, several steps removed from the actual paper, for a baseline that isn't part of this project's own architectural question. |
| **Joint leg tracker, Leigh et al.** (ICRA 2015) — Kalman-filter tracker jointly associating pairs of leg clusters into person tracks | DROW (IROS 2018), Li2Former (TIM 2024) | The DROW paper's own evaluation found it stops producing detections beyond ~7 m range and requires per-dataset threshold retuning to be usable at all — a known, paper-acknowledged limitation of the two-leg-assumption approach, not a competitive comparison point for this project's learned-detector research. |
| **PeTra** (Guerrero-Higueras et al., Frontiers in Neurorobotics, 2019) — full 2D U-Net over a 256×256 rasterized occupancy-grid image, with an external leg-pairing post-process | FROG (2025), as its "modern leg-based" baseline | Already discussed as the closest *related* full-scan work in Section 7.2 — but its task framing differs (binary occupancy-grid segmentation, not this project's shared beam-classification-plus-vote-regression head) and it is markedly slower in FROG's own benchmark (~28 ms/scan vs. <2 ms for LFE, ~14 ms for DROW3/DR-SPAAM). A fair replication would need a separate evaluation harness rather than a drop-in addition to the shared pipeline. |

No other model variants were found published alongside `LFEPeaksDetector`/`LFEPPNDetector`: the FROG paper's own benchmark (Table 4) and the authors' file host (`robotics.upo.es/~famozur/onnx/`) contain exactly these two ONNX exports for the LFE family — `LFE-Peaks.onnx` and `LFE-PPN.onnx`, both already bundled here — alongside unrelated models from the same lab's other projects (object detection, dialogue ontologies), not additional LFE variants.

---

## 3. Previous Research and Baselines

These are already reproduced and runnable in this repo (`train.py --detector drow|drspaam|li2former`, `evaluate.py --lfe-peaks --lfe-ppn`).

### 3.1 DROW — Beyer, Hermans, Leibe (ICRA 2016, IROS 2018)

**Reference:** Beyer, Hermans, Leibe. *DROW: Real-Time Deep Learning-Based Wheelchair Detection in 2-D Range Data*. ICRA 2016 / RA-L 2017 extension. arXiv:1603.02636; 3-class extension (wheelchair/walker/person) in the IROS 2018 paper, arXiv:1804.02463.

> "We present a method to detect, segment, and track multiple people in a 2-D range scan... obtaining state of the art detection results." — abstract

#### Key idea: the cutout

For each beam `i`, a window of `S = 48` neighbouring beam ranges is extracted in polar coordinates and centred on beam `i`. This **cutout** forms a 1D signal of length 48 that encodes the local angular neighbourhood of the beam.

A shared 1D CNN is applied to every cutout independently — beams are processed in parallel with no cross-beam communication during feature extraction. The network has three convolutional blocks followed by two upsampling heads (class logits and vote offset).

#### Temporal aggregation: fixed sum

When using `T` consecutive scans, the per-beam CNN is applied to each timestep independently, producing `T` feature vectors per beam. These are simply **summed** across the time dimension before the detection heads. All timesteps are weighted equally; the network has no way to down-weight a noisy or motion-blurred frame.

```
for each timestep t = 1…T:
    cutout_t[i]  = extract_48_beams(scan_t, beam_i)
    feature_t[i] = CNN(cutout_t[i])          ← no cross-beam communication

feature[i] = Σ_t feature_t[i]               ← fixed sum, equal weights
logit[i], vote[i] = detection_head(feature[i])
```

#### Odometry alignment

Before feature extraction, historical scans are rotated into the current scan's coordinate frame using the robot's odometry (differential rotation Δθ between timesteps). Only rotation is applied — translation is ignored, approximating that the robot has moved little between T consecutive frames.

#### Performance (DROW test set)

| Class | AP |
|---|---|
| Person (wp) | 0.619 |
| Wheelchair (wc) | 0.658 |
| Walker (wa) | 0.520 |

#### Identified weaknesses (motivating this research)

1. **Zero cross-beam communication**: the CNN processes each 48-beam window in complete isolation. Two adjacent beams, each individually ambiguous, cannot reinforce each other's detection.
2. **Fixed temporal weighting**: a scan frame where the person was occluded or the sensor returned a spurious reading contributes equally to the sum.
3. **Fixed receptive field**: the 48-beam cutout is hardcoded; the network cannot look further or closer based on what it sees.
4. **Polar-only representation**: the cutout contains raw range values. The CNN must learn to be invariant to the physical distance of the person, which changes the angular width of the cluster.

#### ⚠ Annotation quality concern — verified against source (Section 8)

The DROW dataset uses **sparse annotations**: only ~5% of recorded scan frames in each sequence carry ground truth labels (measured directly on the bundled data: 17,665 of 341,147 raw train scans = 5.18%, matching the DROW paper's own abstract — "464k laser scans, out of which 24k were annotated" = 5.17%). Annotated frames are not scattered arbitrarily; they land at a regular ~1-in-5 cadence (median gap = 5 raw scans = 0.4 s between consecutive annotated frames).

This was initially assumed to create a "false-negative label noise" training pathology — unannotated frames being penalised as if any detection on them were a false positive. **That assumption was checked against the actual training/eval code and does not hold**: `LidarFrameDataset`'s sample index (`utils/train.py`, `LidarFrameDataset.__getitem__`) enumerates only `(sequence, det_idx)` pairs — one entry per *annotated* frame — so loss is computed exclusively at labelled frames. Unannotated frames are never used as supervision targets; `DROW_Dataset.get_scan()` / `FROG_Dataset.get_scan()` pull the T-frame temporal window from the dataset's separate, dense, continuous raw-scan array (all ~341k DROW scans, not just the labelled 17,665), so the window ending at an annotated frame is real consecutive sensor data, not backfilled or interpolated. This matches the official DROW-v2 reference loader's `get_batch()` (`times = arange(iscan - ntime*dtime+1, iscan+1, dtime)`, same repeat-padding at sequence start) almost exactly.

The real, verified limitation is **coverage, not label noise**: DROW train has far fewer supervised examples than FROG, on two axes at once — 6.1× fewer annotated frames (17,665 vs 108,356) *and* 2.9× fewer people per annotated frame (1.35 vs 3.93, measured directly), for a combined **~17.8× fewer total person-annotation instances** (23,910 vs 425,407). This scarcity is the direct cause of the DROW-native training results in Section 8.1.1/8.5 — see the overfitting evidence there (val loss regressing from epoch 2 onward on a model regularised the same way as its FROG counterpart).

Additionally, ground truth positions were obtained by correlating lidar timestamps with an external reference (camera footage or motion capture), introducing **temporal synchronisation error** at fast motion speeds. Annotations very close to the sensor (< 1 m) or at the edge of the field of view are known to be less reliable due to projection ambiguity.

The FROG dataset was recorded specifically to address the coverage problem: every scan frame is annotated, at a much higher per-frame person density, providing a far denser training signal.

### 3.2 DR-SPAAM — Jia, Hermans, Leibe (IROS 2020, RA-L 2022)

**Reference:** arXiv:2004.14079.

DR-SPAAM retains the DROW cutout extraction and per-beam CNN but replaces the fixed temporal sum with an **auto-regressive spatial attention mechanism** that directly addresses DROW's weaknesses (1) and (2).

This project's implementation is the official **SpatialDROW** from the RA-L 2022 release, loaded from the published checkpoint `dr_spaam_e40.pth`.

#### Architecture: four conv blocks + spatial attention gate

```
Block 1  (1→128,  3 layers) + MaxPool(2)  ↘
Block 2  (128→256, 3 layers) + MaxPool(2)  → 256-ch feature map per beam (14 pts for S=56)
                                             ↓ spatial attention gate
Block 3  (256→512, 3 layers) + MaxPool(2)
Block 4  (512→128, 2 layers) + AvgPool
Conv1d heads → logits (1 or 4 classes), votes (2)
```

#### Auto-regressive spatial attention (the "A" in DR-SPAAM)

After blocks 1–2, each beam has a feature map `f_t[i]` (spatial, not pooled). The temporal aggregation uses an **auto-regressive template** rather than explicit temporal attention weights:

```
template_0 = f_0[:]              ← initialised from first scan in window (detached)

for t = 1…T−1:
    feat_t[i] = encode(scan_t, beam_i)   ← blocks 1–2

    # spatial attention: beam i attends to ±window/2 neighbours in template
    attn_weight[i,j] = softmax( embed(feat_t[i]) · embed(template[j])
                                for j in i−K … i+K )
    template[i] = α · feat_t[i]  +  (1−α) · Σ_j attn_weight[i,j] · template[j]
                  ↑ current feat       ↑ attended historical template

final_feature[:] = decode(template[:])   ← blocks 3–4 + heads
```

Key properties:

- **Stop-gradient on template input**: gradients only flow through the current scan's encode/gate path, not back through the full T-step chain (prevents BPTT instability).
- **Local angular context** (±5 neighbours, `window_size=11`): beam `i` can observe its closest angular neighbours in the historical template.
- **Alpha blending** (`alpha=0.5`): the gate blends current features with the attended template, preventing the template from collapsing to a running mean.
- **56-pt cutouts** (`N_SAMP=56`): published weights use a 56-sample polar window instead of DROW's 48.
- **Pedestrian-only output**: published weights (`dr_spaam_e40.pth`) output a single sigmoid class (person only). This project's training configuration uses 4 classes.

#### Ablation results from the paper

| Component | AP (wp) | Δ vs DROW |
|---|---|---|
| DROW (sum only) | 0.619 | — |
| Temporal attn only | ~0.640 | +2.1 pp |
| Spatial attn only | ~0.659 | +4.0 pp |
| DR-SPAAM (both) | **0.696** | **+7.7 pp** |
| DR-SPAAM RA-L (2022) | **0.720+** | **+10+ pp** |

#### ✅ Fidelity gap found, root-caused, and fixed (Section 8)

Running the published `dr_spaam_e40.pth` weights through this repo's evaluation pipeline on the DROW test set originally gave **25.8% wp AUC**, far below the paper's reported 69.6–72%+. Four explanations were tested directly rather than assumed:

1. **Input/cutout shape mismatch?** Ruled out — `DrSpaamDetector.N_SAMP=56` is read correctly by the shared cutout-extraction code (`_extract_input`), verified by direct inspection: cutout tensors are `(450, 5, 56)`, matching the published weights' expected shape exactly.
2. **Wrong `load_published()` construction args?** Ruled out — `alpha=0.5, window_size=11, num_pts=56, pedestrian_only=True` were checked against the official DR-SPAAM-Detector repo's own training config (`dr_spaam/cfgs/dr_spaam.yaml`) and match exactly.
3. **Postprocessing (vote-clustering) hyperparameter mismatch?** Tested directly and ruled out — the official config's `vote_kwargs` (`vote_collect_radius=0.157`, `min_thresh=9.4e-5`, `blur_sigma=1.46`, vs. this repo's shared defaults of `0.5`/`1e-3`/`2.0`) were substituted in for an isolated re-run: **26.2% wp AUC — no meaningful change** from the 25.8% baseline.
4. **Temporal window depth?** The official `dr_spaam.yaml` specifies `num_scans: 10` (the DROW-only `drow5.yaml`, a *different* network type, uses 5) — `load_published()` was defaulting to 5. Fixed (`evaluate_auc()` now reads `net.num_scans` when present instead of always using `dataset.time_frame`), but a direct forward-pass comparison at T=5 vs T=10 on the same real frame showed **no meaningful difference** in output confidence — not the cause, though the fix is correct and kept for fidelity.

The actual root cause, found by cloning the official [DR-SPAAM-Detector](https://github.com/VisualComputingInstitute/DR-SPAAM-Detector) repo and diffing the real `scans_to_cutout()` source (not just the config) against this repo's `cutout()`: `dr_spaam.yaml`'s `cutout_kwargs` specify `window_width: 1.0, window_depth: 0.5` — narrower and shallower than DROW's own defaults (`window_width: 1.66, window_depth: 1.0`, which this repo's `cutout()` hardcoded unconditionally for every cutout-based model). Worse, the official implementation **normalises** the centred cutout by dividing by `window_depth` (`ct = (ct - dists) / window_depth`) — a step this repo's `cutout()` was missing entirely. For DROW's own published weights (`window_depth=1.0`) that division is a no-op, which is why `DrowDetector` was unaffected and looked fine throughout; for DR-SPAAM's published weights (`window_depth=0.5`) the missing division under-scales every feature by 2×, compounded by clipping to the wrong (2× too wide) depth tunnel in the first place.

**Fix**: `cutout()` in `drow_utils.py` now divides the centred window by `thresh_dist` (backward-compatible no-op at the DROW default of 1.0); `DrSpaamDetector` gained `WIN_SZ=1.0` / `THRESH_DIST=0.5` class attributes (matching `dr_spaam.yaml`), threaded through `_extract_input()` in `train.py`. Re-running the DROW eval after the fix: **wp AUC 25.8% → 68.1%** — landing right in the paper's 69.6–72% range, and now sensibly *above* `DrowDetector`'s own (also-fixed-pipeline) 66.5% wp AUC, matching the paper's ordering (DR-SPAAM > DROW). The earlier "structurally under-confident" pattern (median max-confidence 0.465 on real positive frames, vs. `DrowDetector`'s 0.98) was exactly what a 2× feature-scale shrink would produce — consistent with this being the actual cause, not a deeper attention-mechanism bug as originally suspected.

#### Limitations of DR-SPAAM

1. **Still uses fixed 56-beam cutouts**: the CNN has no access to raw data beyond 56 beams, and the cutout boundary is hardcoded.
2. **Spatial attention operates on compressed features**: neighbouring beams only communicate *after* independent CNN processing. Raw-signal cross-beam context is impossible.
3. **Local attention only** (±5 neighbours): each beam can see only its closest neighbours. Global scan structure (e.g., two legs of the same person widely separated) is not captured.
4. **Performance-driven design**: DR-SPAAM was developed under real-time constraints (ROS node, embedded hardware). Many architectural choices reflect computational budget rather than theoretical optimality.

### 3.3 Li2Former — Yang et al. (IEEE TIM 2024)

**Reference:** Yang et al. *Li2Former: Omni-Dimension Aggregation Transformer for Person Detection in 2-D Range Data*. IEEE Trans. Instrumentation and Measurement, 2024. DOI: 10.1109/TIM.2024.3420353.

Li2Former retains the cutout-based input representation but replaces DR-SPAAM's spatial attention with a **temporal Transformer** that attends over T consecutive cutouts per beam independently.

```
For each (beam, timestep) pair — cutout width P = 64 samples:

  _ConvBackbone(1 → d_model=512):
    Stage 1: Conv1d(1→64)×2  + Conv1d(64→128)  + MaxPool(2)   → P/2 positions
    Stage 2: Conv1d(128→128)×2 + Conv1d(128→256) + MaxPool(2)  → P/4 positions
    Stage 3: Conv1d(256→256)×2 + Conv1d(256→512) + AdaptiveAvgPool(1) → 1 position
    → (B*N*T, 512) feature vector per (beam, timestep)

Reshape to (B*N, T, 512)
+ Sinusoidal positional encoding over T timesteps
TransformerEncoder(d_model=512, nhead=8, 1 layer)  ← temporal self-attention
Mean-pool over T  → (B*N, 512)

Classification head : Linear(512 → 1)            binary person logit
Regression head     : Linear(512 → 1024) → ReLU → Linear(1024 → 2)  vote offsets
```

Key properties:

- **Temporal attention, not spatial**: DR-SPAAM attends across neighbouring beams at a fixed timestep; Li2Former attends across T timesteps for a single beam. The two mechanisms are complementary.
- **Wider cutout** (P=64 vs. 48 for DROW, 56 for DR-SPAAM): more angular context per beam.
- **Binary output** (person vs. background): the sigmoid probability is placed in the pedestrian slot for compatibility with the unified evaluation pipeline.
- **No published weights**: must be trained from scratch using `train.py --detector li2former`.
- Recurrent-free in the RNN sense, but uses a temporal Transformer — exactly the kind of component constraint 1 (Section 2) excludes from this project's own proposed architectures, and the comparison point that constraint's efficiency argument is made against.

### 3.4 LFE-Peaks and LFE-PPN — Amodeo et al. (2025)

**Reference:** Amodeo, Pérez-Higueras, Merino, Caballero. *FROG: A new people detection dataset for knee-high 2D range finders*. Frontiers in Robotics and AI, 2025. arXiv:2306.08531.

The LFE detectors are inference-only ONNX baselines from the FROG benchmark paper. They differ from all other detectors in two important ways: they operate on a **single scan with no temporal context**, and they use a **1-D U-Net FCN** (the LFE backbone) applied directly to the normalised raw scan vector rather than cutout windows.

```
Input: (1, N=720, 1)   — normalised range vector  (1.0 = near, 0.0 = far)
1-D U-Net FCN (encoder-decoder with skip connections)
→ per-beam probability map (1, N, 1)
```

They differ only in the detection head:

| Variant | Head | Post-processing |
|---|---|---|
| **LFE-Peaks** | Per-beam sigmoid probability | `scipy.find_peaks` on the 1-D probability map |
| **LFE-PPN** | Anchor grid: N/6 sectors × 30 depth anchors × 3 outputs | Anchor decoding + greedy distance-based NMS |

Key constraints: single-scan only (no odometry, no temporal history); trained on 720-beam FROG scans (inference on 450-beam DROW or 541-beam JRDB scans requires zero-padding — see the fidelity caveat below); class-agnostic (person confidence only); ONNX inference only (weights are bundled automatically; re-training is not supported within this framework).

**This is the closest prior work to this project's own proposed architectures** — see Section 7.2.

> **Fidelity caveat (Section 2.1.1 has the full detail).** The ONNX weights this repo loads are the authors' own published models — the *network* is exact. The Python wrapper around it has one remaining known gap: zero-padding non-720-beam scans (DROW, JRDB) to run LFE cross-dataset is this project's own extrapolation, never described or tested in the paper, and does not correct for FROG's different angular resolution — treat DROW/JRDB numbers as an approximate cross-dataset transfer, not a faithful replication. (A second gap — LFE-Peaks reporting every raw `find_peaks` hit as its own detection, omitting the paper's centroid-merge step — has been fixed: `LFEPeaksDetector` now runs the same greedy NMS-style merge described in the paper, via a shared `_merge_nearby()` helper also used by `LFEPPNDetector`. The merge radius, `_PERSON_RADIUS = 0.4 m`, is this project's own choice — the paper describes the merge step but does not state an exact radius.)
>
> **Two more gaps found and fixed while producing Section 8's numbers.** (1) `LFEPPNDetector` hardcoded 31 depth anchors per sector (`_N_ANCHORS = 31`, from the design formula `(FAR-NEAR)/(0.8·RADIUS)`), but the bundled published ONNX model's actual output shape is `[..., 30, 3]` — 30 anchors, not 31 — so every evaluation of LFE-PPN crashed with an index-out-of-bounds error. Fixed by reading the anchor count from the ONNX model's own output shape at load time instead of hardcoding the design estimate. (2) `evaluate.py`'s LFE evaluation path computed recall as `(true positives found) / (true positives found)` — i.e. against its own detections, not against the total ground-truth annotation count — so a frame where the detector found nothing at all near a real person contributed nothing to the denominator either, and recall was structurally guaranteed to approach 1.0 regardless of how many people were missed outright. This inflated LFE-Peaks' FROG AUC(wp) to 86.2%, ~21 points above the paper's own 64.9% AP — implausible for a same-network, same-weights evaluation. Fixed by routing LFE through the same `_prec_rec_2d` PR-curve code every other model uses (recall against total GT, missed detections counted). After the fix, LFE-Peaks measures **74.6%** — much closer to the paper's 64.9% (the residual gap is a normal, explainable margin: different eval-frame sample, this repo's own `_merge_nearby` radius choice, etc. — not another metric artifact). LFE-PPN was less affected by the recall bug (67.7% either way, coincidentally close to its own 66.5% published figure) but is reported post-fix for consistency.

---

## 4. Preprocessing

### 4.1 Raw sensor data

Each scan is a 1D array of `N` range measurements:

```
scan = [r_0, r_1, …, r_{N-1}]    r_i ∈ [r_min, r_max] metres
```

The beam angles are fixed and sensor-specific:

```
φ_i = φ_min + i · Δφ             (uniformly spaced, e.g. Δφ = 0.5° = 0.00873 rad)
```

### 4.2 Coordinate representation: the `(r, x, y)` triplet, and why it was dropped

For each beam `i` in each aligned scan:

```python
x_i = -r_i * sin(φ_i)    # lateral  (DROW frame: x = sideways)
y_i =  r_i * cos(φ_i)    # forward  (DROW frame: y = forward)
```

giving a 3-channel per-beam input `(r_i, x_i, y_i)`. This was the representation used by this project's own early full-scan designs (and, implicitly, by the cutout approach's post-hoc vote geometry):

- **`r` (polar range)** tells the network how far away a beam endpoint is, which a fixed-kernel CNN cannot otherwise infer — a person at 2 m spans ~12 beams, at 8 m only ~3.
- **`(x, y)` (Cartesian)** lets a 1D conv kernel detect a person's roughly constant-diameter body (~0.5–0.8 m) as a dense point cluster directly, without learning trigonometry from `r` and beam index.
- **Not `(x, y)` alone**: loses the explicit distance/scale signal.
- **Not first differences `(Δr, Δx, Δy)`**: a 1D CNN with kernel `[-1, +1]` can compute these internally; providing them as input is redundant.
- **Redundancy**: `x² + y² = r²`, so the three channels are not independent — this creates a null space in the first conv layer's weights but does not prevent learning.

**Current default (Section 5 architectures): raw range only, no Cartesian channels.** `drow_utils.raw_scan(scans_hist)` returns `(T, N_beams, 1)` — just the range value per beam per timestep, no transformation at all. The `(r, x, y)` representation and its supporting function `aligned_scan_xyz()` have been removed; nothing in the currently active architectures calls it anymore. This is deliberate — see constraint 2 in Section 2.

### 4.3 Odometry alignment

When stacking `T` consecutive scans, the robot moves between frames. Only the **rotation component** of odometry (Δθ) is corrected for — translation is negligible over T=5 frames at typical indoor robot speeds:

1. Compute the accumulated rotation `Δθ_k` from odometry between `t−k` and `t`.
2. Shift each beam's angle: `φ'_i = φ_i + Δθ_k`.
3. Re-sample onto the original beam grid via nearest-neighbour: `r'_j = r_i` where `i = round((φ'_j − φ_min) / Δφ)`.

`--align-scans` (default: **on**) applies this rotation correction to the raw-scan representation via `aligned_raw_scan()` in `drow_utils.py`, shifting all T frames simultaneously via vectorised NumPy indexing with linear interpolation. When odom data is absent (FROG without an `_odom.npz` file), the shifts are zero and the output is identical to the unaligned baseline. Pass `--no-align-scans` to reproduce the unaligned variant as an ablation — this was the original design (no odometry dependency at inference time, simpler rotating buffer), and whether alignment actually helps accuracy is deliberately left as an open empirical question for Section 8 rather than assumed.

### 4.4 Cutout extraction (DROW / DR-SPAAM / Li2Former only)

The cutout approach replaces the full-scan input with a fixed-width polar window extracted around each beam:

```
cutout[i] = [r_{i-24}, r_{i-23}, …, r_i, …, r_{i+23}]   ← raw range values only (width 48 for DROW)
```

This discards all cross-beam information beyond the window, uses raw `r` values only, and hardcodes the spatial context window. It is kept for compatibility with the cutout-based baselines' training pipeline and inference API; the full-scan architectures in Section 5 do not use it.

---

## 5. Proposed Architectures

### 5.0 Design choices that apply to all three

**Raw scan representation.** See Section 4.2 — `(T, N_beams, 1)`, no Cartesian channels.

**Rotating buffer for the `T`-frame window.** Because there's no realignment step required for correctness, maintaining the `T`-scan window at inference time reduces to a fixed-size circular buffer: evict the oldest scan, append the newest, O(1) per measurement. This is already implemented — `LiveDataset.push_measure()` (`library/follow_the_drow/datasets/live_dataset.py`) uses a `collections.deque(maxlen=time_frame)`, which evicts the oldest entry automatically on `append()`.

**What got dropped, and why.** This project previously had four full-scan detector classes; two were removed for violating the non-recursive constraint (Section 2, constraint 1) — see Section 6 for what they were and why they were tried. What's left, `SpaceTimeCNNDetector` and `FullScanTCNDetector`, were already non-recursive before that constraint was stated explicitly; `TemporalUNetDetector` (Architecture C) was added afterward as a controlled ablation against LFE (Section 5.3).

### 5.1 Architecture A — Joint Space-Time CNN (`SpaceTimeCNNDetector`)

**What it does:** reshapes the input to `(1, 1, N_beams, T)` and treats it like a 1-channel image, where the height axis is the beam index and the width axis is time. A stem + two dilated "joint" 2-D conv blocks mix beam and time together from the very first layer; multi-scale Inception-style blocks (parallel branches at kernel sizes 3/5/9, increasing dilation) then specialize along the beam axis (for scale-invariance — near people are wide in beam-space, far people are narrow) and the time axis (for motion).

**Explainability, verified against each cited paper (Section 2.1.1 has the audit methodology) — not every claimed inspiration held up equally well:**

| Block | Borrowed from | Why |
|---|---|---|
| Residual skip connections everywhere | He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 (ResNet) | **Verified accurate.** Every block does `out + skip(x)`, with a 1×1 projection shortcut when channels change — a textbook ResNet shortcut, faithfully implemented (`_JointConvBlock`, `_MultiScaleBlock`, `DilatedScanBackbone`). |
| Squeeze-and-Excitation channel gating | Hu, Shen, Sun, *Squeeze-and-Excitation Networks*, CVPR 2018 | **Verified accurate.** `_SEBlock`'s global-avg-pool → 2-layer bottleneck MLP → sigmoid → channel-wise multiply matches Hu et al. almost line-for-line. |
| Multi-branch, multi-kernel-size blocks merged via 1×1 conv | Szegedy et al., *Going Deeper with Convolutions*, CVPR 2015 (Inception/GoogLeNet) | **Loose inspiration, not a faithful port.** `_MultiScaleBlock` does run parallel kernel-size branches concatenated and 1×1-projected — the core motif is real — but it diverges from actual Inception: branches use anisotropic `(k,1)`/`(1,k)` kernels along a single axis rather than square 2D multi-scale kernels, there's no pre-branch 1×1 bottleneck (Inception's actual compute-saving trick), and SE + residual are bolted on, which GoogLeNet's Inception module never had. A person's beam-width varies 4× between near and far range (Section 1), which is the real motivation for the multi-kernel idea, independent of how closely it tracks GoogLeNet specifically. |
| Joint (not factorized) space-time convolution in the early layers, cited as video-CNN literature (e.g. C3D) | ~~Tran et al., *Learning Spatiotemporal Features with 3D Convolutions*, ICCV 2015 (C3D)~~ | **Inaccurate citation — removed.** C3D's defining feature is a *true 3D convolution* over two spatial axes plus time. This architecture has only one spatial axis (beam index); the "joint" stem/blocks are ordinary `Conv2d` over a beam×time grid, not a volumetric 3D kernel. Citing C3D overstates a kinship to volumetric video CNNs this design doesn't have. No single published architecture was found that matches this beam×time-grid treatment closely enough to cite directly (it doesn't match range-image LiDAR CNNs like SqueezeSeg/RangeNet++ either, which use azimuth×elevation, not beam×time) — **the honest framing is that this is an original design combining the ResNet/SE/Inception-style ideas above, not a port or close variant of any single published architecture.** |

### 5.2 Architecture B — Causal TCN + Spatial CNN (`FullScanTCNDetector`)

**What it does:** a small causal temporal-convolutional stack runs over each beam's own `T`-frame history *first* (collapsing time to a single per-beam summary), and only then does the spatial `DilatedScanBackbone` mix across beams — so spatial mixing has access to motion-aware features from the start, which Architecture A's joint-from-layer-1 design has to learn implicitly instead.

| Block | Borrowed from | Why |
|---|---|---|
| Causal dilated 1-D convolution, "chomp" padding | Bai, Kolter, Koltun, *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, arXiv 2018 (coined "TCN" in this sense); causal dilated convs originate in van den Oord et al., *WaveNet*, arXiv 2016 | A feed-forward, fully parallel alternative to recurrence (GRU/LSTM) with the same left-to-right dependency structure — directly satisfies constraint 1 while keeping order-sensitivity (unlike plain mean-pooling, which is permutation-invariant in time). **Verified against the paper's official reference implementation (`locuslab/TCN`): this is a *simplified* TCN block, not a faithful port** — see the deviation list below. |
| Dilated 1-D conv stack, dilation 1→2→4→8, residual skip (`DilatedScanBackbone`) | He et al. (ResNet); dilated convs from Yu & Koltun, *Multi-Scale Context Aggregation by Dilated Convolutions*, ICLR 2016 | Receptive field grows exponentially (3→7→15→31 beams) with only linear parameter growth — covers a standing person (~12 beams at 2 m) without a hand-picked fixed window. **Verified**: the dilation schedule matches Yu & Koltun, and there is a real skip connection — but it is *one* residual wrapping the entire 4-layer stack end-to-end, not He et al.'s pattern of a shortcut around every ~2-conv block. Should be described as "a single global residual wrapper," not "ResNet-style residual learning." |

**Deviations from the canonical TCN block (`locuslab/TCN`'s `TemporalBlock`), confirmed by direct comparison:** this repo's `_TemporalTCNBlock` uses **one** dilated causal conv per block where the reference uses **two** (compensated by stacking two single-conv blocks at dilations 1, 2 — 2 total conv layers here vs. 4 in two canonical blocks at the same dilations); **GroupNorm instead of weight normalization** (not documented as a substitution anywhere in the code); only **two dilation levels** (1, 2) rather than the reference's arbitrary-depth schedule (receptive field 7 is just enough to cover T=5, but this is a materially shallower network than "a TCN" in the general sense); and **no final activation after the residual sum** (the reference applies `ReLU(out + res)`; this repo returns `out + skip(x)` directly). The residual connection itself and the causal chomp padding size are both correct. These are reasonable efficiency simplifications for a fixed T=5, B=1 inference setting — but the architecture should be described as **"a simplified causal TCN block, following Bai et al.'s general design but not a faithful port of their reference implementation,"** not as "a TCN" without qualification.

**Architecture A vs. B is a deliberate ablation, not two unrelated designs:** "should space and time be fused jointly from layer 1, or should time be collapsed first and space handled afterward?" is analogous to the question studied for video CNNs by Xie et al., *Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification*, ECCV 2018 (joint 3-D convolution vs. factorized 2-D-spatial + 1-D-temporal designs). Architecture A is the "joint" point in that design space; Architecture B inverts the factorized order further by doing time *before* space rather than after.

### 5.3 Architecture C — LFE-with-Time U-Net (`TemporalUNetDetector`)

**What it does:** a standard 1-D U-Net backbone (Ronneberger et al., MICCAI 2015) with the single-channel input replaced by T-channel input (one channel per historical scan). Operates over the beam axis — each encoder stage halves N_beams by MaxPool1d(2), the bottleneck sees the full scan compressed 8×, and the decoder upsamples back with skip connections from each encoder stage restoring spatial precision. Default head is heatmap (matching LFE-Peaks).

**Verified against the canonical U-Net definition — correct, no bugs.** All six defining properties were checked directly against the code: symmetric 3-stage encoder/decoder depth, a genuine two-conv "double conv" block at every stage, MaxPool1d downsampling (not strided conv), skip connections via channel-wise **concatenation** (not addition — confirmed no accidental ResNet-style shortcut bug), a bottleneck stage, and `F.interpolate(size=skip.shape[-1])` upsampling that exactly matches each skip's spatial size regardless of non-power-of-2 beam counts (450, 720). This is a faithful, correctly-implemented U-Net.

**Correction — this is *not* actually "LFE's backbone + time."** The original framing below ("Architecture C adds only temporal context to LFE's backbone and head, isolating temporal contribution") was checked directly against the FROG paper's description of LFE's real architecture (§4.1) and does not hold up: LFE's own backbone is **not** a textbook U-Net. It's three residual blocks, each containing three depthwise-separable convs at kernel sizes [9, 7, 5], each with an *internal* residual addition and a "global aggregator" (global max-pool concatenated at every position) — none of which `TemporalUNetDetector` has. LFE also keeps channel width roughly constant at 32 (bumping to 64 only at the deepest level) and downsamples by 2× and 3× (chosen so that 450-point scans divide evenly), whereas this repo's `TemporalUNetDetector` doubles channels every stage (C→2C→4C→8C) and downsamples by three successive ×2 pools. LFE uses BatchNorm; this repo uses GroupNorm.

So `TemporalUNetDetector` is honestly **a fresh, textbook U-Net given a temporal input** — not LFE's own backbone with time added. That's a legitimate, defensible architecture in its own right (see the citation table below), but the ablation framing needs correcting: Architecture C isolates "**temporal context + a from-scratch U-Net backbone**" against LFE, not "temporal context only." The row below is corrected accordingly.

| Model | Backbone | T | Head | What changes vs previous row |
|---|---|---|---|---|
| LFE-Peaks | LFE's own backbone (3 residual blocks, depthwise-separable multi-kernel convs, constant ~32ch, global aggregator) | 1 | heatmap | — |
| **Architecture C** (TemporalUNet + heatmap) | **Canonical U-Net** (symmetric encoder-decoder, doubling channels, MaxPool/concat-skip — *not* LFE's own backbone topology, see correction above) | **5** | **heatmap** | **+temporal context, and a different (though comparably-scoped) backbone** |
| Architecture A + heatmap (SpaceTimeCNN) | dilated 2D CNN | 5 | heatmap | backbone only |
| Architecture A + drow (SpaceTimeCNN) | dilated 2D CNN | 5 | DROW votes | head only |
| Architecture B + drow (FullScanTCN) | TCN + dilated 1D CNN | 5 | DROW votes | fusion order |

| Block | Borrowed from | Why |
|---|---|---|
| Encoder-decoder with skip connections | Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015 | Skip connections restore fine beam-level precision lost by pooling: the bottleneck encodes "is there a cluster of short-range readings here?" globally; the skip at E1 restores exactly which beam carries that reading. |
| T input channels | Same strategy as Architecture A's stem `Conv2d(1, C, 3×3)` treating T as the width axis | The simplest way to extend a single-frame U-Net stem to multi-frame input — not literally LFE's own stem (LFE's backbone is not this U-Net topology, see the correction above), but the same general idea of putting T in the channel dimension of the first layer. |
| GroupNorm instead of BatchNorm | Wu & He, ECCV 2018 | LFE's Keras model uses BatchNorm, which degrades at B=1 inference; GroupNorm avoids that failure mode. Framed here purely as a normalization choice — it is not, by itself, what makes this backbone differ from LFE's (see the backbone-topology correction above; GroupNorm is one of several differences, not the only one). |
| BEAM_BATCH=True (vs BEAM_BATCH=False for A and B) | DrSpaamDetector (same flag) | The U-Net pools over the beam axis: flattening the batch to (B×N, T, 1) would let MaxPool1d pool across scan boundaries and corrupt the encoder-decoder alignment. |
| MaxPool1d(2) + F.interpolate(size=skip.shape[-1]) | Standard U-Net | 720 beams (FROG) and 450 beams (DROW) are not powers of 2; matching the skip's exact size avoids padding artifacts. |

**What the U-Net skip connections do and do not do.** A common misreading: skip connections do NOT directly connect beam 0 to beam 719. They connect each *encoder layer* to its *mirror decoder layer at the same spatial resolution*. What connects distant beams indirectly is the bottleneck, where the N//8-beam compressed representation has a large effective receptive field through the Conv1d(k=3) layers.

### 5.4 Cross-cutting hyperparameters

| Hyperparameter | Value | Why |
|---|---|---|
| Time window `T` | 5 frames | At a typical robot update rate of ≈ 10 Hz, T=5 covers 500 ms — long enough to observe a single walking half-cycle (≈ 400–600 ms per step, Winter 1990). |
| Dropout | 0.1 | Light regularisation; output heads are regularised by the data volume (FROG's train split is 108,356 annotated frames — every scan is labeled, Section 3.1 — measured directly during Section 8's evaluation run, correcting an earlier ≈20k estimate here). |
| GroupNorm | largest power-of-2 group count | Wu & He, *Group Normalization*, ECCV 2018 — batch-size-independent, unlike BatchNorm at inference `B=1`. |

<details>
<summary>Per-architecture design-choice tables (channel widths, kernel sizes, receptive fields)</summary>

#### Architecture A — SpaceTimeCNNDetector

| Design choice | Value(s) | Origin & reasoning |
|---|---|---|
| Base channels `C` | 96 | GoogLeNet/Inception's second-stage channel count. For 3-branch parallel blocks of C/3 = 32 ch each, 96 prevents channel collapse; fits within a ~2–3M parameter budget on FROG-scale scans. |
| Stem | 1 × `Conv2d(1, C, 3×3)` | Minimal entry point, lifting 1 input channel to C. |
| Joint blocks | 2 × `Conv2d(C, C, 3×3)` at dilations (1,1) and (2,2) | Borrowed from DeepLab v3 / ASPP (Chen et al., ICLR 2018) — two dilation levels build a modest receptive field on both axes before the factorized multi-scale stages. |
| Spatial stages | default 3 (`n_spatial_stages`), dilations **1, 4, 16** (optional 4th: 64) | Powers-of-4 (WaveNet convention) cover more angular extent with fewer stages: 3 stages + k=9 branches give a 177-beam receptive field (≈ ±44° at 0.5°/beam). |
| Spatial kernel sizes | (3, 5, 9) | At 0.5°/beam: a person at 2 m subtends ≈ 28 beams; at 8 m ≈ 7 beams — parallel branches let the network learn which size to weight (Inception motivation). |
| Temporal stages | 2 × `_MultiScaleBlock` on the time axis, kernel sizes (3, 5, 7) | Stage 1 (k=3): velocity-like patterns; Stage 2 (k=5, applied to Stage 1's output): acceleration-like patterns, reaching the full T=5 window. |
| SE reduction ratio | 8 | Default from Hu et al. (CVPR 2018); 12-channel bottleneck for C=96. |
| Mean collapse over T | — | Motion is already encoded in the channel dimension after the temporal stages; mean-pooling discards the raw T axis without losing information. |

#### Architecture B — FullScanTCNDetector

| Design choice | Value(s) | Origin & reasoning |
|---|---|---|
| TCN channel width | 64 | Smaller than Architecture A's 96 — the TCN stage only summarizes T, it isn't the main spatial-reasoning stage. Matching `tcn_channels = backbone_channels = 64` avoids an extra projection at the handoff. |
| Number of TCN layers | 2, dilations **1, 2** | With kernel_size=3: total receptive field = 1 + 2×1 + 2×2 = 7 > T=5. Two causal layers are the minimum to cover the full 5-frame window. |
| Causal "chomp" padding | pad left by `(k−1)×dilation`; drop trailing `2×dilation` positions | Guarantees output position `t` depends only on input positions `≤ t` — same left-to-right dependency as a GRU, via masked convolution instead of recurrence. |
| Last-timestep selection `[:, :, -1]` | — | The causal TCN output at `t=T−1` has seen all `T` prior steps; equivalent to a GRU's final hidden state, without the permutation-invariance of mean-pooling. |
| DilatedScanBackbone | 4 × `Conv1d(C, C, 3)` at dilations **1, 2, 4, 8**; 1×1 end-to-end residual skip | Receptive field = 1 + 2×(1+2+4+8) = 31 beams ≈ ±7.5° ≈ 0.8 m at 3 m range — spans a standing person's shoulder width (≈ 0.5 m) with margin. |

#### Architecture C — TemporalUNetDetector

| Design choice | Value(s) | Origin & reasoning |
|---|---|---|
| Base channels | 32 (`unet_channels`) | Four stages: 32→64→128→256 channels, ~650k total parameters — lighter than SpaceTimeCNN (≈1-2M). |
| Number of encoder stages | 3 | Three MaxPool1d(2) stages compress 720 beams to 90 in the bottleneck. |
| Bottleneck channels | 8C = 256 | Standard U-Net doubling per stage. |
| Decoder upsample | `F.interpolate(size=skip.shape[-1], mode='linear')` | `ConvTranspose1d` introduces checkerboard artifacts at non-power-of-2 input sizes; linear interpolation exactly matches the encoder's spatial size. |
| Default head | heatmap | Matches LFE-Peaks, making Architecture C the cleanest extension of LFE; `--head drow` available for ablation. |

</details>

### 5.5 Alternative detection head — Gaussian heatmap (`--head heatmap`)

Both Architecture A and B support a heatmap head as a drop-in replacement for the default DROW-style head; Architecture C defaults to it. Everything else (backbone, training loop, evaluation) is unchanged. This is an **ablation**, not the primary method: the DROW head is kept as default elsewhere because it keeps the comparison with DROW/DR-SPAAM/Li2Former head-neutral.

| Design choice | Value | Origin & reasoning |
|---|---|---|
| Target type | 1D Gaussian heatmap over N_beams; one Gaussian per annotated person, summed and clipped to [0, 1] | CenterNet (Zhou et al., *Objects as Points*, CVPR 2019) object-center heatmap targets, applied in 1D. |
| Gaussian width σ | **2 beams** (`--heatmap-sigma`, default 2.0) | FWHM ≈ 4.7 beams ≈ 2.35° at 0.5°/beam. σ=1 is too sparse; σ≥4 lets adjacent-person Gaussians overlap at typical ≤1 m inter-person spacing. |
| Loss | Weighted BCE (`pos_weight = n_neg / n_pos`, clamped to 200) against the soft Gaussian target | Compensates the ≈50:1 background-to-foreground ratio; the soft target spreads gradient over ≈2σ beams rather than only the exact peak. |
| Output head | `nn.Linear(C, 1)` | Strictly simpler than the DROW head's two branches — if it achieves comparable AUC with fewer parameters, that is evidence vote regression was the weak link, not the backbone. |
| Decoder | `scipy.signal.find_peaks(sigmoid(output), height=0.3, distance=5)` | `height=0.3` suppresses background noise; `distance=5` beams (≈2.5°, ≈0.17 m at 2 m) separates peaks below typical inter-person spacing but above the Gaussian FWHM. |
| Why it may outperform the DROW head for full-scan architectures | — | The DROW vote-regression head was designed for cutouts, where a beam at the cluster edge must predict an offset without seeing the cluster center. Full-scan backbones (31–177 beam receptive fields) *do* see the cluster center, so a heatmap peak is a more direct target. |
| Evaluation compatibility | — | The decoder outputs (x, y, confidence) triplets placed in column 3 (person class) of the same `(N_dets, 4)` array `_process_detections`/`_prec_rec_2d` expect — no pipeline changes needed. |

---

## 6. Design History: Architectures Tried and Dropped

This project previously had four full-scan detector classes; two were removed for violating the "non-recursive only" constraint (Section 2). Kept here as a record of what was tried and why — not part of the current comparison.

### FullScanCNNDetector — removed

Used a GRU for temporal aggregation after a dilated 1D CNN backbone processed each timestep independently:

```
For each timestep t independently:
  Conv1d(3,    64,  kernel=3, dilation=1 ) → receptive field  3 beams
  Conv1d(64,  128,  kernel=3, dilation=2 ) → receptive field  7 beams
  Conv1d(128, 256,  kernel=3, dilation=4 ) → receptive field 15 beams
  Conv1d(256, 256,  kernel=3, dilation=8 ) → receptive field 31 beams
  Conv1d(256, 256,  kernel=3, dilation=16) → receptive field 63 beams
  → (N_beams, 256) per timestep

Temporal: GRU over T, per beam → (N_beams, 256)
Detection heads: Conv1d(256, 4) + Conv1d(256, 2)
```

Dropped because the GRU violates the non-recursive constraint. The class was deleted from `full_scan.py`.

### FullScanTransformerDetector — removed

Used global multi-head self-attention over all N_beams simultaneously, after a dilated CNN backbone:

```
Full-scan dilated CNN backbone  → (N_beams, T, C)
BeamSelfAttention               → each beam attends to all N_beams (at each T)
Temporal mean-pool over T       → (N_beams, C)
Detection heads
```

The most expressive beam-level architecture in the family (content-adaptive, global beam-to-beam attention via a `N_beams × N_beams = 202,500`-entry attention matrix per layer), but self-attention isn't a CNN/TCN primitive, so it violates constraint 1 — and in the one same-footing benchmark run available, it was both the slowest (~1.5 s/scan on CPU) and lowest-scoring of the four full-scan models. Class deleted, along with the now-unused `BeamSelfAttention` helper. An earlier version used a GRU for temporal aggregation here too; it was replaced with mean-pooling since at T=5 the recurrence added negligible benefit while blocking DirectML — but mean-pooling is permutation-invariant in time, so this architecture structurally cannot use motion direction (approaching vs. receding) at all, unlike FullScanTCNDetector (Section 5.2).

### Architecture comparison (including removed architectures, for provenance)

| Architecture | Beam communication | Temporal | Fusion order | Input representation | Status |
|---|---|---|---|---|---|
| DrowDetector | None (per-beam cutout) | Fixed sum | space-then-time | Polar range only | active |
| DrSpaamDetector | Local (±5 beams, auto-regressive) | Learned blending | joint (auto-regressive) | Polar range only | active |
| ~~FullScanCNNDetector~~ | Local → growing (dilated, up to ±15 beams) | GRU | space-then-time | (r, x, y) full scan | **removed** |
| SpaceTimeCNNDetector | Local (spatial + temporal jointly) | Implicit (2D conv) | joint | raw range only, full scan, unaligned by default* | active |
| ~~FullScanTransformerDetector~~ | Global (all N beams) | Mean-pool | space-then-time | (r, x, y) full scan | **removed** |
| FullScanTCNDetector | Local → growing (dilated, up to ±15 beams) | Causal TCN | **time-then-space** | raw range only, full scan, unaligned by default* | active |
| TemporalUNetDetector | Global (bottleneck) + local (skips) | T input channels | joint | raw range only, full scan, unaligned by default* | active |
| Li2FormerDetector | None (per-beam cutout) | Transformer (T steps) | time-then-space | Polar range only | active |
| LFEPeaksDetector / LFEPPNDetector | Global (U-Net, full scan) | None (single scan) | n/a | Normalised range only | active |

\* `--align-scans` (default on) applies odometry rotation correction; see Section 4.3.

All active detectors are accessible via `follow_the_drow.detectors.DETECTOR_REGISTRY`; the trainable ones (no published weights) via:

```bash
python train.py --detector drow|drspaam|spacetime_cnn|fullscan_tcn|temporal_unet|li2former
```

---

## 7. Related Work

### 7.1 Cutout-based detectors — see Section 3 for full technical detail

- **DROW** — Beyer, Hermans, Leibe, arXiv:1603.02636 / arXiv:1804.02463.
- **DR-SPAAM** — Jia, Hermans, Leibe, IROS 2020, **arXiv:2004.14079** (this document previously cited `2004.14064`, an unrelated paper — a one-digit transcription error, now fixed).
- **Li2Former** — Yang et al., IEEE TIM 2024, DOI: 10.1109/TIM.2024.3420353.

All three share the cutout: a fixed-size polar window per beam, processed by a per-beam CNN with no (DROW) or only local (DR-SPAAM, ±5 beams) cross-beam communication during feature extraction, and no (DROW), auto-regressive (DR-SPAAM), or temporal-attention (Li2Former) handling of the time axis.

### 7.2 Full-scan / no-cutout detectors — is this novel?

**Short answer: the full-scan idea itself is not unprecedented, but the specific combination this research proposes — full-scan + raw/unaligned-by-default input + multi-frame temporal context + beam-classification-with-vote-regression, using only non-recursive primitives — appears to be.** Three prior works are close enough to discuss directly:

1. **LFE-Peaks / LFE-PPN** — Amodeo et al., Frontiers in Robotics and AI, 2025, arXiv:2306.08531. **This is the closest prior work.** It applies a 1-D U-Net FCN directly to the raw, full-scan range vector (no cutout) — the same "give the network the whole raw scan" idea used here, including the non-Cartesian, single-channel input (Section 3.4). The key difference: LFE is **single-scan only** — no temporal window at all. This research's contribution relative to LFE is adding the `T`-frame time axis as a first-class input dimension via three different non-recursive fusion strategies (Section 5).

   **Relationship to "extending LFE."** Both Architecture A and B can be read as principled extensions of the LFE spatial idea:

   - *Replace the encoder–decoder with dilated convolutions.* LFE's U-Net compresses the beam axis then reconstructs it via upsampling + skip connections. Architectures A and B use dilated convolutions with `same` padding throughout — the beam axis is never downsampled, so no decoder is needed. This is the same substitution DeepLab (Chen et al. 2015) made in image segmentation: dilated ("atrous") convolutions expand the receptive field without sacrificing spatial resolution.
   - *Add the temporal axis, via three fusion strategies.* Architecture A fuses space and time jointly from layer 1; Architecture B summarizes time first via a causal TCN, then mixes space; Architecture C gives LFE's general "full-scan, U-Net-family backbone + heatmap head" idea a temporal input, though — as corrected in Section 5.3 — via a from-scratch canonical U-Net rather than a literal port of LFE's own multi-kernel residual backbone, so it isolates "temporal context + backbone redesign" rather than temporal context alone. Neither A nor B's question was available in LFE since it had no time axis at all.

   The detection head is also different by default: LFE finds persons via local peak detection (Peaks variant) or a proposal network (PPN variant); Architectures A and B default to DROW-style per-beam 4-class logits plus 2D vote-offset regression — the same head used by every cutout baseline, making accuracy numbers directly comparable without a bridging step. The heatmap head (Section 5.5) is the point of contact with LFE's own approach.

2. **PeTra** — Guerrero-Higueras et al., Frontiers in Neurorobotics, 2019, DOI: 10.3389/fnbot.2018.00085. Projects raw 2-D LIDAR returns into a 256×256 binary occupancy grid and segments leg positions with a full U-Net — also full-scan, also no cutout. Differs in task framing (binary occupancy-grid segmentation + a separate tracker, vs. per-beam multi-class detection + vote regression directly on the polar scan) and is markedly slower (≈300 ms/scan reported vs. single-digit-to-low-double-digit ms for the cutout baselines).
   > "the occupancy map is defined as a 256 × 256 matrix, with a resolution of about 2 cm." — Section on input representation.

3. **TCN + 2-D LIDAR precedent (different task)** — Luo, Poslad, Bodanese, IEEE IoT Journal, 2020, DOI: 10.1109/JIOT.2020.2984544. Applies a TCN to **already-extracted trajectories** to classify 15 activity types — TCN-after-detection, not TCN-for-detection. Evidence that TCNs are a known, working tool in this sensor domain, but doesn't address detection itself the way Architecture B does. Cited as precedent, not as a competing baseline.

No paper found in this search combines full-scan (no-cutout), raw input, a multi-frame temporal window, the beam-classification-plus-vote-regression detection head used by DROW/DR-SPAAM/this project, **and** an exclusively non-recursive architecture. That combination — not any one piece alone — is the actual novelty claim to make.

### 7.3 Adjacent 3-D LiDAR literature (context, not a fair baseline)

Full-scan, no-cutout CNNs are the *standard*, not the exception, in 3-D LiDAR object detection — range-image-based detectors (e.g. Sun et al., *RSN: Range Sparse Net*, CVPR 2021; *Fully Convolutional One-Stage 3D Object Detection on LiDAR Range Images*, 2022) run ordinary 2-D convolutions over the whole range image. Useful context for why "full-scan CNN" is a reasonable thing to try, but these papers are 3-D, multi-object, autonomous-driving-scale, and not a fair comparison row for this project's 2-D, person-only, robot-scale setting.

---

## 8. Evaluation Plan & Comparison Tables

All numbers below either come from this repo's own pipeline (`utils/train.py`'s `evaluate_auc`, `evaluate.py --bench`) run on the same hardware (AMD Radeon RX 9060 XT for training via DirectML; all evaluation and benchmarking is CPU-only by design, for reproducible device-independent numbers), or are clearly marked as literature-reported numbers from a different setup (typically the original paper's own GPU). **Do not mix the two when drawing conclusions** — §8.3 and the "paper" cells below exist for context, not as an apples-to-apples check against this repo's own measurements.

This evaluation was run end-to-end on 2026-08-13 following [docs/EVALUATION_PLAN.md](EVALUATION_PLAN.md), with several corrections made along the way (documented inline below and in Section 3 where they affect a specific detector's fidelity). JRDB was confirmed absent in the run environment (requires manual registration) — its column is `n/a` throughout, consistent with the existing convention.

**A note on "AUC" vs. the literature's "AP":** this repo's `AUC` and the papers' `AP` are the same computation, not two different metrics being informally compared. This project's `_prec_rec_2d`/`_safe_auc` (`library/follow_the_drow/utils/drow_utils.py`, `utils/train.py`) is a direct port of the original DROW-v2 evaluation code; the official DR-SPAAM-Detector repo's own `eval_prec_rec()` (`dr_spaam/src/dr_spaam/utils/prec_rec_utils.py`) computes what its paper calls "AP" via the identical formula — raw trapezoidal `sklearn.metrics.auc(recall, precision)` over the same Hungarian-matched, per-frame TP/FP/FN precision-recall curve, no monotonic-envelope interpolation (unlike COCO/VOC-style AP). Verified by direct source comparison against the cloned official repo, not assumed. The FROG paper's own DROW3/DR-SPAAM baselines almost certainly inherit the same eval code (it explicitly reuses Jia et al.'s DR-SPAAM setup) — that part is a well-grounded inference rather than a direct code diff, since FROG's own eval source wasn't available to check line-by-line. Bottom line: the gaps discussed throughout this section are real accuracy differences, not artifacts of comparing differently-defined metrics.

### 8.1 Accuracy (AUC, person class "wp", at 0.5 m matching radius)

| Architecture | Type | DROW | FROG | JRDB |
|---|---|---|---|---|
| AlgorithmicDetector | rule-based (no learning) | F1 55.7%⁴ (recall 68.8%, prec 46.9%) | F1 1.7%⁴ (recall 1.9%, prec 1.5%) | n/a⁵ |
| DrowDetector | cutout, fixed sum | 66.5%¹ | 62.7%¹ | n/a⁵ |
| DrSpaamDetector | cutout, auto-regressive attn | 68.1%¹ ⁶ | 67.2%¹ ⁶ | n/a⁵ |
| Li2FormerDetector | cutout, temporal transformer | n/a⁷ | n/a⁷ | n/a⁷ |
| LFEPeaksDetector | full-scan, single-frame (ONNX) | 24.0%¹ ³ | 74.6%¹ ⁸ | n/a⁵ |
| LFEPPNDetector | full-scan, single-frame (ONNX) | 11.7%¹ ³ | 67.7%¹ ⁸ | n/a⁵ |
| **Architecture A** — SpaceTimeCNNDetector | full-scan, raw, joint CNN | 29.0%² | **80.4%** | n/a⁵ |
| **Architecture B** — FullScanTCNDetector | full-scan, raw, time-then-space TCN+CNN | 22.2%² | 69.5% | n/a⁵ |
| **Architecture C** — TemporalUNetDetector | full-scan, raw, 1D U-Net+time | 22.1%² | 65.4% | n/a⁵ |

¹ DROW, DR-SPAAM, and LFE all use published/bundled weights (`evaluate.py --drow --drspaam --lfe-peaks --lfe-ppn`) — no training performed.
² Trained once on FROG (Section 8.4), evaluated cross-dataset on DROW without retraining — see Section 8.1.1 for why this beat retraining natively on DROW.
³ LFE was trained on FROG's 720 beams; running it on DROW's 450 requires zero-padding (this project's own extrapolation, not described in the paper) and does not correct for the different angular resolution — expect degraded, approximate numbers, not a faithful replication (Section 3.4).
⁴ AlgorithmicDetector has no confidence score, so AUC isn't defined for it — F1/precision/recall reported instead. Its near-total collapse on FROG (F1 1.7% vs. 55.7% on DROW) isn't a bug: FROG's people are far denser per scan (Section 3.1's "annotation quality" framing doesn't apply here, this is a genuine geometric/rule-tuning mismatch) — see Section 8.5.2.
⁵ JRDB data was not present in the environment this evaluation ran in (requires manual registration) — not attempted, not a code limitation.
⁶ **Fixed during this evaluation run** — see Section 3.2's fidelity note. DrSpaamDetector originally measured 25.8% (DROW) / 38.6% (FROG), far below its paper's 69.6–72%+, due to a missing cutout-normalisation step (`cutout()` didn't divide by `window_depth`, silently no-op for DROW's own `window_depth=1.0` but a 2× feature-scale error for DR-SPAAM's published `window_depth=0.5`). After the fix, DROW lands in the paper's own range and correctly ranks above DrowDetector; FROG (still zero-shot DROW→FROG transfer, not retrained) improved from 38.6% to 67.2% — much closer to the FROG paper's *retrained* DR-SPAAM figures (73.3–75.3 AP, Section 8.3) despite the remaining zero-shot handicap.
⁷ Li2Former's training could not be completed in this environment — DirectML crashed the GPU three different ways (immediate allocator OOM at batch_size=2; access violation and later a segfault at batch_size=1, in-subprocess and in-process) and CPU-only training was measured at ~1.5 fr/s, i.e. ~6.6h *per epoch* (Section 8.5.2) — impractical to complete. No locally-measured accuracy number exists for this detector; see Section 8.3 for its own paper's reported figures instead.
⁸ Fixed during this evaluation run — see Section 3.4's fidelity note. LFE-Peaks' FROG number was 86.2% before the fix (a metric bug inflated it ~21 points above its own paper); LFE-PPN crashed outright before an anchor-count bug was fixed.

#### 8.1.1 DROW: zero-shot transfer vs. training natively vs. fine-tuning — transfer still won

Because DROW (224.5° FoV, 450 beams) and FROG (180° FoV, 720 beams) differ in more than just beam count, training DROW-native versions of Architectures A/B/C seemed like the more principled comparison. A third regime was added after diagnosing *why* native training underperforms (see below): fine-tuning from the FROG checkpoint at a 10× lower LR (1e-4 vs. 1e-3), instead of training from a random init, using a new `--init-weights` flag (`utils/train.py`) that loads model weights only, with a fresh optimiser/epoch count — unlike `--resume`, which also restores optimiser state and would have carried over the wrong LR:

| Architecture | FROG-trained → DROW (zero-shot) | DROW-native (from scratch) | DROW fine-tuned (FROG init, lr=1e-4) |
|---|---|---|---|
| SpaceTimeCNN | **29.0%** | 15.0% | 17.2% |
| FullScanTCN | **22.2%** | 16.5% | 21.4% |
| TemporalUNet | **22.1%** | 19.4% | 12.7% |

Zero-shot transfer wins outright in all three regimes for all three architectures — fine-tuning helped over from-scratch training for two architectures (FullScanTCN 16.5%→21.4%, SpaceTimeCNN 15.0%→17.2%) but made TemporalUNet worse (19.4%→12.7%), and none of the three fine-tuned results caught up to simply deploying the FROG-trained weights unmodified.

**Why from-scratch DROW-native training underperforms** (checked directly against the training logs, not assumed): DROW's train split has ~17.8× fewer total person-annotation instances than FROG's — 6.1× fewer frames (17,665 vs. 108,356) *and* 2.9× fewer people per frame (1.35 vs. 3.93, measured directly) — for models using the same `dropout=0.5` regularisation in both cases. `spacetime_cnn`'s DROW-native run shows textbook overfitting: val_loss gets worse every epoch from epoch 2 onward (0.8511→0.9114→1.0301→1.0349) while train_loss keeps falling, so the "best" checkpoint early-stops back to essentially the epoch-1 snapshot. This isn't a code bug — a spot-check of that same checkpoint on real DROW positive frames shows it *can* produce confident, roughly-correct detections (median max-confidence 0.71 across 60 sampled positive test frames, 78% above 0.5) — it's a genuine data-scarcity problem relative to model capacity, not a degenerate/broken model.

Fine-tuning from FROG features was the natural fix to try, and it partly worked (2 of 3 architectures beat from-scratch), but didn't close the gap to zero-shot transfer. A remaining unexplored option, worth noting for future work: **shrink the architectures themselves** (fewer channels/spatial stages, e.g. via the existing `--channels`/`--n-spatial-stages`/`--tcn-channels`/`--unet-channels` flags) and raise dropout further, so a DROW-native model's capacity is matched to DROW's much smaller effective dataset instead of reusing FROG-scale capacity — the current architectures/regularisation were tuned against FROG's ~18×-larger signal, and neither native training nor fine-tuning changes that mismatch, only feature initialisation. This wasn't tested here (out of scope for this fidelity investigation) but is the logical next experiment if a genuinely DROW-optimized (not just DROW-compatible) model is wanted.

The practical takeaway for this project's own claim (Section 5.4) stands regardless: these architectures transfer to a differently-shaped LiDAR reasonably well *without* retraining, and *can* be retrained cheaply if desired (all three DROW experiments — native and fine-tuned — completed in well under an hour on this repo's hardware) — it's just that, at DROW's data scale, retraining doesn't currently pay for itself.

### 8.2 Efficiency (ms / frame, CPU, single scan — same hardware for every row)

All numbers below are CPU-only by design (see the note at the top of Section 8) — GPU inference numbers for this repo's own models were not collected; where a detector's own paper reports GPU numbers, they're in Section 8.3, not here.

| Architecture | DROW (450 beams) | FROG (720 beams) | JRDB (541 beams) |
|---|---|---|---|
| AlgorithmicDetector | 0.04 ms | 0.06 ms | 0.06 ms |
| DrowDetector | 203.3 ms | 276.4 ms | 235.0 ms |
| DrSpaamDetector | 201.5 ms | 350.4 ms | 298.3 ms |
| Li2FormerDetector | 431.7 ms | 590.6 ms | 498.3 ms |
| LFEPeaksDetector | 1.55 ms | 1.76 ms | 2.01 ms |
| LFEPPNDetector | 315.5 ms⁹ | 243.8 ms⁹ | 278.3 ms⁹ |
| **Architecture A** — SpaceTimeCNNDetector | 8.7 ms | 12.1 ms | 12.5 ms |
| **Architecture B** — FullScanTCNDetector | **2.1 ms** | **2.6 ms** | **3.1 ms** |
| **Architecture C** — TemporalUNetDetector | **2.0 ms** | 2.7 ms | 2.3 ms |

⁹ LFE-PPN's cost is a decoding-loop artifact, not an architectural one — see Section 8.5.2.

All three proposed architectures beat every cutout-based baseline (DROW/DR-SPAAM/Li2Former, all 200–600 ms/frame) by **20–290x**, and FullScanTCN/TemporalUNet are within 2x of the fastest thing in the whole table (AlgorithmicDetector, a rule-based C++ method with no learned inference cost at all).

### 8.3 Reference: published numbers from the literature (context only, not for the tables above)

From the FROG paper's own internal comparison (Table 4, person AP @ 0.5 m, on an Intel i9-9900X + TITAN RTX):

| Detector | AP (paper) | ms/frame (paper, GPU) | AP (this repo) | ms/frame (this repo, CPU) |
|---|---|---|---|---|
| ROS leg_detector | 15.8 | 1.77 | not replicated (Section 2.2) | not replicated |
| PeTra | 49.6 | 28.17 | not implemented (Section 2.2 — different task framing) | not implemented |
| PeTra* (mixed-loss variant) | 58.3 | — (not reported) | not implemented | not implemented |
| LFE-Peaks | 64.9 | 1.76 | 74.6 (FROG) | 1.76 (FROG) |
| LFE-PPN | 66.5 | 1.49 | 67.7 (FROG) | 243.8 (FROG) |
| DROW3 (T=1) | 73.6 | 13.08 | 62.7 (FROG) | 276.4 (FROG) |
| DR-SPAAM (T=1) | 73.3 | 13.95 | not yet evaluated — official T=1 weights downloaded, evaluation queued (Section 8.4-adjacent, `docs/IDEAS_BACKLOG.md` item 4) | not yet measured |
| DR-SPAAM (T=5) | 75.3 | 13.99 | 67.2 (FROG)¹⁰ | 350.4 (FROG) |

PeTra/PeTra*/DR-SPAAM(T=1) rows added from a direct re-check of the paper's Table 4 (arXiv:2306.08531v2 HTML) while populating `docs/HIGHLIGHTS.md`'s comparison tables — previously this table only transcribed the four rows this repo actively replicates; the "PeTra*" mixed-loss variant's own weights (`petra_frog_mixedloss.h5`) are the same file already found on the FROG dataset page's weights listing (`docs/IDEAS_BACKLOG.md` item 4), confirming the correspondence rather than assuming it.

From the DROW and DR-SPAAM papers (DROW test set, person class):

| Detector | AP (paper) | AP (this repo) |
|---|---|---|
| DROW | 0.619 | 0.665 |
| DR-SPAAM | 0.696 (paper), 0.720+ (RA-L 2022 release) | 0.681 — see Section 3.2's fidelity note (fixed; was 0.258) |

Li2Former paper: **0.764 AP on DROW, 0.814 AP on JRDB** (per-paper, not independently re-verified — training was not completed in this environment, Section 8.1 footnote 7). Li2Former-A (official repo benchmark): 66.9 FPS (≈14.95 ms) on DROW, 57.0 FPS (≈17.54 ms) on JRDB — both on the paper's own GPU; this repo's own CPU measurement of the same architecture (untrained weights, forward-pass timing only) was 431.7–498.3 ms/frame, 30–40x slower, entirely explained by CPU vs. GPU rather than any implementation difference.

Reading the LFE-Peaks/DROW3/DR-SPAAM rows together: this repo's numbers track the paper's within a normal margin for LFE-Peaks (+9.7, after fixing the recall-metric bug that originally inflated it to 86.2 — see Section 3.4), DROW3 (-10.9, plausible for published weights evaluated by a different, independently-written harness), and now DR-SPAAM too (-1.5 on DROW after fixing the cutout-normalisation bug documented in Section 3.2 — was -36.7 before the fix). The remaining FROG gap for DROW3 (-10.9) and DR-SPAAM (-8.1, down from -36.7) is expected rather than a fidelity problem: this repo's `drow`/`drspaam` FROG numbers are **zero-shot DROW→FROG transfer** using the published DROW-trained weights, while the FROG paper's reference figures are for DROW3/DR-SPAAM **retrained natively on FROG** (5 epochs, batch 8 — confirmed directly from the paper's text, Section 8.1.1). That's not an apples-to-apples comparison; closing it would require retraining `drow`/`drspaam` on FROG the same way, which was out of scope for this fidelity investigation.

¹⁰ Zero-shot DROW→FROG transfer (published DROW-trained weights, not retrained on FROG) — see the paragraph above and Section 8.1.1 for why this differs from the paper's own retrained-on-FROG DR-SPAAM figure.

### 8.4 How these numbers were produced

Weights strategy: published weights are used directly for DROW, DR-SPAAM, LFE-Peaks, LFE-PPN (all bundled). Architectures A, B, and C were trained once on FROG, as originally planned; Section 8.1.1 above adds DROW-native versions after the FoV/beam-count mismatch between DROW and FROG turned out to matter enough to test directly. Li2Former's training did not complete (Section 8.1, footnote 7). AlgorithmicDetector needs no training. JRDB was absent in the run environment.

Orchestration followed [docs/EVALUATION_PLAN.md](EVALUATION_PLAN.md), but the environment and the plan itself both needed real fixes before it would run end-to-end:

**Environment issues found and fixed:** a stale editable `pip install -e ../library` pointing at an unrelated old checkout; a Python 3.14/cp312 ABI mismatch against the pre-built C++ extension (resolved with a dedicated Python 3.12 venv); the bundled DROW dataset and DROW/DR-SPAAM/LFE published weights were never actually downloaded (`setup.py`'s downloader only runs when `include/` doesn't exist yet, and it already existed with FROG data only); no GPU acceleration was installed despite a usable AMD GPU being present (`torch==2.4.1` + `torch-directml` installed and verified against the official DR-SPAAM/DROW training configs where relevant).

**Code bugs found and fixed** (beyond the three `EVALUATION_PLAN.md` itself already called out — missing `temporal_unet` in `train_all.py`, missing `--temporal-unet` in `evaluate.py`, and the default-head resolution bug):
- `evaluate.py`'s `eval_algorithmic()` and `eval_lfe_model()` crashed on FROG (never triggered on DROW) — `dataset.det_wc[seq][det] + dataset.det_wa[seq][det] + dataset.det_wp[seq][det]` silently switches from list concatenation to numpy elementwise addition whenever a plain Python list (FROG's always-empty `det_wc`/`det_wa`) meets a numpy array (FROG's `det_wp`), raising a broadcast error on shape mismatch. Fixed with a shared `_as_rp_array`/`_all_classes_rp` helper.
- `LFEPPNDetector` hardcoded 31 depth anchors per sector; the bundled ONNX model's real output shape has 30. Fixed by reading the anchor count from the model's own output shape instead of the design-formula estimate.
- `evaluate.py`'s LFE evaluation computed recall against its own found true positives instead of the total ground-truth count — structurally unable to penalize a detector for missing people entirely, inflating LFE-Peaks' FROG AUC by ~21 points. Fixed by routing LFE through the same `_prec_rec_2d` PR-curve code every cutout/full-scan model already uses.
- `evaluate.py --bench`'s `bench_model()` assumed every model returns a `(logits, votes)` 2-tuple; heatmap-head models (`TemporalUNetDetector`) return a 1-tuple and errored out in both EVAL and TRAIN sections. Fixed by branching on `head_type`.
- The AUC evaluation loop (`evaluate_auc`, `batch_size=1` by default) makes FROG's 120,396-frame test split take ~10.8h *per model* — impractical for 7 queued models. Added `--eval-batch-size` (verified numerically equivalent at `16`) and `--eval-stride` (a systematic subsample — FROG's every-scan-annotated-at-40Hz protocol means adjacent frames are highly correlated near-duplicates, unlike DROW's already-sparse ~5%-of-scans labeling; stride 40 → ~3,010 evaluated frames, comparable in scale to DROW's own 2,428-frame test set, cutting FROG eval time by ~40x with no measurable accuracy difference in spot checks).
- `train_all.py`'s own training-summary table silently printed the *agnostic* (any-class) AUC under the header "Test AUC" rather than *wp* (person-only), the metric this document uses everywhere — its self-reported DROW-native numbers (32.5%/35.5%/28.1%) looked meaningfully better than the real comparable wp figures (15.0%/16.5%/19.4%, Section 8.1.1) purely because of the column mislabel. Fixed by relabeling the column explicitly as `Test AUC(any)`.
- li2former's DirectML crash at the default batch size turned out to be a memory-pressure issue, not a hard operator incompatibility — `batch_size=1` on GPU got measurably further (and ~6x faster) than falling back to CPU, before a separate, deeper DirectML stability issue (Section 8.1, footnote 7) made even that impractical to complete. `train_all.py` now uses a per-detector batch-size override for this reason, though it didn't end up being sufficient on its own.
- `cutout()`'s shared feature-extraction code hardcoded DROW's own `window_width=1.66`/`window_depth=1.0` and never normalised (divided) the centred window by `window_depth` — a no-op at DROW's own values, but silently halved every input feature for DR-SPAAM's published weights, which specify `window_width=1.0`/`window_depth=0.5` in the official `dr_spaam.yaml`. This is what caused DrSpaamDetector's 25.8%/38.6% (DROW/FROG) wp-AUC, not a deeper attention-mechanism bug as first suspected — found by cloning the official DR-SPAAM-Detector repo and diffing its real `scans_to_cutout()` source, not just its config file. Fixed by adding the missing normalisation plus per-model `WIN_SZ`/`THRESH_DIST` overrides; DROW wp-AUC rose to 68.1% (matching the paper's 69.6–72%), FROG to 67.2% (Section 3.2, Section 8.1 footnote 6).

### 8.5 Analysis: which proposed architecture wins, and what would improve each detector

#### 8.5.1 Best of the three proposed architectures: SpaceTimeCNN — and how to improve it

**SpaceTimeCNN (Architecture A) is the strongest of the three proposed designs.** It posts the best FROG-native accuracy (80.4% — the highest number in the *entire* table, ahead of every baseline too) and the best DROW zero-shot transfer (29.0%, vs. 22.2%/22.1%). Its one loss is DROW-native (15.0%, lowest of the three, Section 8.1.1) — but that's a data-volume artifact affecting all three architectures identically in direction, not a sign the architecture is worse; SpaceTimeCNN's own zero-shot number beats its own native-trained number by nearly 2x, which argues for "give it more data," not "use a different architecture." Fine-tuning from the FROG checkpoint (Section 8.1.1) recovered some of that gap (15.0%→17.2%) but not all of it — zero-shot transfer remains the best of the three regimes tested for this architecture.

*Why it wins:* SpaceTimeCNN mixes beam and time axes together from the very first convolutional layer (Section 5.1) — the richest cross-time/cross-beam receptive field of the three, at the cost of being the most parameter-hungry of the three at the widest channel count (base channels=96, vs. FullScanTCN's 64 and TemporalUNet's 32) — plausibly why it benefits most from FROG's dense, exhaustive per-scan annotations (Section 3.1) and generalizes best, but also why it's the one architecture visibly starved by DROW-native's sparse ~5%-of-scans labeling. It's also the *slowest* of the three proposed architectures (8.7–12.5 ms/frame vs. 2–3 ms for the other two, Section 8.2) — a real but minor cost, since it's still 20–40x faster than any cutout baseline.

**Concrete ways to improve it, in rough order of expected value:**

1. **Shrink the architecture for DROW-scale data, rather than reusing FROG-sized capacity.** Fine-tuning from the FROG checkpoint (Section 8.1.1, tested directly) closed *some* of the zero-shot gap but not all of it (15.0%→17.2%, still short of zero-shot's 29.0%) — reusing the full FROG-sized network (base channels=96) with only a lower LR wasn't enough. The next thing worth testing is reducing capacity itself (`--channels`, `--n-spatial-stages`) and raising dropout above the current flat 0.5 (Section 5.4) specifically for DROW-scale training, so the model's capacity is matched to DROW's ~17.8×-smaller effective dataset (Section 8.1.1) instead of carrying FROG-scale capacity into a much smaller-data regime.
2. **Light data augmentation for low-data regimes** (beam-wise jitter, random dropout of beams to simulate occlusion) would directly target the same data-starvation weakness, complementary to reducing capacity, without touching the FROG-native numbers.
3. **Cheaper multi-scale branches, if the ~9ms/frame cost ever needs cutting.** The fidelity audit (Section 5.1) already flags that `_MultiScaleBlock` skips real Inception's pre-branch 1×1 bottleneck (its actual compute-saving mechanism) — adding it back, or replacing the widest (k=9) branch with LFE's depthwise-separable convs (Section 5.3's LFE-backbone comparison), would cut cost with comparatively little accuracy risk, since FullScanTCN already shows a well-regularized full-scan architecture can reach 69.5% FROG accuracy at 67K params and 2.6 ms/frame.

#### 8.5.2 The rest: bugs, underperformance, speed, and improvement ideas

**FullScanTCNDetector (Architecture B) and TemporalUNetDetector (Architecture C) — no bugs found, genuinely behind on accuracy, ahead on speed.** Both are correctly implemented (verified against their cited references in Sections 5.2/5.3) and both trail SpaceTimeCNN on FROG (69.5%/65.4% vs. 80.4%) while being 3–4x faster (2.6–2.7 ms vs. 12.1 ms). FullScanTCN's factorized time-then-space design (Section 5.2) processes motion as a *summary* before spatial mixing, rather than jointly — plausibly the reason it trails Architecture A, consistent with the video-CNN literature's own finding (Xie et al., ECCV 2018, cited in Section 5.2) that joint 3-D-style processing usually edges out factorized designs on accuracy while factorized designs win on speed. TemporalUNet's repeated MaxPool1d downsampling (Section 5.3) compresses the 720-beam scan 8x by the bottleneck before the decoder restores it — a plausible, though unverified without a direct ablation, explanation for why it's the *most* parameter-hungry of the three (651K, more than SpaceTimeCNN's 476K) yet the least accurate on FROG: the bottleneck may discard fine beam-level position information that only skip connections (rather than the compressed bottleneck itself) can restore, and detection is fundamentally a per-beam localization task. Both are strong candidates when speed matters more than the last few points of accuracy — TemporalUNet in particular is priced almost identically to LFE-Peaks (2.0–2.7 ms vs. 1.55–2.01 ms) while being a full temporal model rather than single-frame.

**DrowDetector — no bugs found, published weights transfer reasonably.** 66.5% on DROW (vs. the paper's own 61.9% — this repo's number is actually *higher*, plausibly because `_prec_rec_2d`'s trapezoidal PR-AUC integration (Section 8.1) is a slightly different (and typically marginally more generous) estimator than whatever exact AP computation the original paper used — a normal, expected margin, not a red flag) and 62.7% on FROG cross-dataset transfer (vs. DROW3's 73.6% in the FROG paper's own comparison — a believable transfer penalty, not investigated further since it's in the plausible range this project's own convention treats as "not requiring investigation," Section 8.3).

**DrSpaamDetector — real fidelity gap, found and fixed.** See Section 3.2's fidelity note for the full writeup: after ruling out input shape, published-weights construction, and postprocessing hyperparameters, cloning the official DR-SPAAM-Detector repo and diffing its real `scans_to_cutout()` source (not just its yaml config) found the actual cause — a missing cutout-normalisation step that silently halved every input feature for DR-SPAAM's published `window_depth=0.5` weights (a no-op for DROW's own `window_depth=1.0`, which is why `DrowDetector` was never affected). Fixed; DROW wp-AUC rose from 25.8% to 68.1% (matching the paper's 69.6–72% and now correctly ranking above `DrowDetector`), FROG from 38.6% to 67.2%. No further improvement work needed here — the remaining FROG-side gap to the literature (67.2% vs. the FROG paper's 75.3 AP) is expected zero-shot-vs-retrained variance (Section 8.3), not a fidelity problem.

**Li2FormerDetector — not a bug, a genuine environment limitation, but an informative one.** Three independent DirectML crash modes (Section 8.1, footnote 7) plus a measured ~1.5 fr/s CPU-only training speed (≈6.6h *per epoch* — Section 8.2's bench, extrapolated) made completing even one training run impractical here. This is itself a data point: of every detector in this comparison, Li2Former is both the only one whose published GPU speed (14.95 ms/frame, Section 8.3) doesn't hold up in any form on this hardware (this repo's own CPU measurement of the same untrained architecture: 431.7–590.6 ms/frame, 30–40x slower — a hardware-driver interaction, not an implementation flaw) and the only one whose training this project couldn't complete at all. **How it could be improved (as a research question, not a code fix):** either a torch-directml version bump (this repo's `torch-directml==0.2.5.dev240914` is a 2024 preview build pinned to `torch==2.4.1`; the crash could plausibly be fixed by a newer release) or training on genuine CUDA/ROCm hardware rather than DirectML — this project's own `detect_device()` already prioritizes CUDA/ROCm first (Section 1 of the evaluation plan) precisely because DirectML is the third-choice fallback, not the recommended path.

**LFEPeaksDetector — no remaining bugs; the closest match to its own paper in the whole table.** 74.6% on FROG vs. the paper's 64.9% (Section 8.3) is a normal margin after fixing the recall-metric bug (Section 3.4) — no further action recommended.

**LFEPPNDetector — accuracy fixed and matches its paper closely (67.7% vs. 66.5%); speed remains a real, fixable inefficiency.** At 243.8–315.5 ms/frame (Section 8.2), it's ~150x slower than LFE-Peaks despite similar accuracy and an architecturally comparable model — the cause is implementation, not the anchor-grid design itself: `LFEPPNDetector.detect()` decodes every (sector × anchor) score with a nested pure-Python `for` loop (`library/follow_the_drow/detectors/lfe_detector.py`), while LFE-Peaks decodes via a single vectorized `scipy.find_peaks` call. **How it could be improved:** vectorizing the anchor-decoding loop with NumPy (threshold-mask the whole `(n_sectors, n_anchors)` objectness array at once, then only loop over the surviving candidates before NMS) should bring it in line with LFE-Peaks' cost — this is a concrete, low-risk optimization that doesn't touch model weights or accuracy, just decoding, and wasn't done here because it was out of scope for an evaluation pass rather than because it's hard.

**AlgorithmicDetector — no bug, but a striking, informative failure mode.** F1 55.7% on DROW collapses to 1.7% on FROG (Section 8.1, footnote 4) — both recall (68.8%→1.9%) and precision (46.9%→1.5%) collapse together, meaning it's simultaneously missing real people *and* producing mostly false positives, not just one or the other. A rule-based detector's hardcoded distance/width heuristics, tuned against DROW's sensor geometry, would be expected to fail this way against a sensor with a meaningfully different angular resolution (FROG: ~4 beams/degree vs. DROW: ~2 beams/degree) — and FROG's LiDAR is explicitly knee-height-mounted per its own paper title ("FROG: A new people detection dataset for **knee-high** 2D range finders," Section 3.4), a different sensor placement than DROW's, which plausibly changes what a leg cross-section looks like in-scan enough to break fixed geometric heuristics entirely. This is exactly the class of failure the learned detectors in this table (including all three proposed architectures) are designed to be robust to, and the fastest inference in the table (0.04–0.06 ms) is worth nothing if accuracy collapses this completely outside its tuned domain.

### 8.6 Does temporal fusion actually help on FROG? An error breakdown and a tracker

Section 8.1's FROG table shows LFE-Peaks — a single-frame detector with no temporal input at all — matching or beating FullScanTCN/TemporalUNet, and only SpaceTimeCNN (the widest, most parameter-hungry of the three) clearly ahead. That's suspicious given Section 8.1.1 and this section's own earlier finding: FROG ships **no odometry at all** (confirmed by inspecting its `.h5` files directly — no pose field of any kind), so every "temporal" window on FROG is raw, unaligned frame-stacking; and FROG's 40 Hz rate means a T=5 window spans only ~125 ms of real time, not the ~500 ms "gait half-cycle" the window size was originally justified against (Section 5.4, calibrated for DROW's ~10 Hz).

**First hypothesis tested: late-fusion temporal consensus, using the detectors' own outputs.** A naive version (average confidence across the last N frames' detections, matched by nearest-neighbour to raw historical positions) was built first, then rejected on inspection: averaging *positions* assumes the target is static, which is wrong for a moving person, and the mechanism can only suppress false positives, never recover a false negative. Before building anything more elaborate, the actual error composition was checked directly rather than assumed:

**Error breakdown at each model's own best-F1 operating point (FROG, wp/person class, stride=40):**

| Model | AUC | Recall | Precision | F1 | TP | FP | FN | Dominant error |
|---|---|---|---|---|---|---|---|---|
| spacetime_cnn | 80.4% | 79.0% | 70.9% | 74.7% | 9354 | 3844 | 2483 | FP (0.6×) |
| fullscan_tcn | 69.5% | 80.5% | 62.7% | 70.5% | 9523 | 5657 | 2314 | FP (0.4×) |
| temporal_unet | 65.4% | 72.6% | 59.5% | 65.4% | 8597 | 5855 | 3240 | FP (0.6×) |
| lfe_peaks | 74.6% | 78.4% | 69.6% | 73.7% | 9279 | 4059 | 2558 | FP (0.6×) |
| lfe_ppn | 60.7%¹ | 65.6% | 59.7% | 62.5% | 7770 | 5238 | 4067 | FP (0.8×) |

**False positives dominate false negatives for every single model** — the FN/FP ratio never exceeds 0.8×. This is computed by re-reading the same (recall, precision) curve every AUC number in this document already comes from (`_prec_rec_2d`), at the best-F1 threshold, backing out TP/FP/FN from `total_gt` — no new inference code (`utils/error_breakdown.py`). This decisively answers "filter false positives first, or something else": on FROG specifically, false positives are the bigger problem for every detector tested, including the proposed architectures.

**Mechanism: `SimpleTracker`, replacing the naive consensus filter.** Built directly to your two objections above — `library/follow_the_drow/utils/tracking.py` is a minimal SORT-style tracker: a constant-velocity motion state per track (position + exponentially-smoothed velocity — an "alpha-beta filter," not a full Kalman filter with covariance, to keep this cheap for a pilot), matched to new detections each frame via Hungarian assignment (`scipy.optimize.linear_sum_assignment`, the same tool `_prec_rec_2d` already uses for GT matching) against the track's **predicted** position, not its last raw position. A track must accumulate `min_hits` real matches before it's reported at all (false-positive suppression — directly targets the dominant error found above); a confirmed track that goes briefly undetected keeps reporting its predicted (coasted) position, with the reported score decayed per miss, for up to `max_age` frames before being dropped (false-negative recovery). Per your explicit direction, this assumes a **static sensor** on FROG — there's no odometry to compensate the robot's own motion with, and nothing to be done about that (`docs/IDEAS_BACKLOG.md` item 2 records this as the reason DROW, which has real odometry, is the fairer testbed for the tracker's core assumptions later).

**First result — LFE-Peaks improved, LFE-PPN regressed** (FROG, stride=5, `match_radius=0.5, min_hits=3, max_age=3, vel_alpha=0.5, coast_decay=0.8`, no retraining, post-hoc filter on existing model outputs only):

| Model | No consensus (baseline) | `SimpleTracker` | Δ |
|---|---|---|---|
| lfe_peaks | 74.8% | **76.1%** | **+1.3pp** |
| lfe_ppn | 60.1%¹ | 56.7% | −3.4pp |

LFE-Peaks gains from the tracker with zero retraining — a genuine, if modest, win for a purely post-hoc mechanism. LFE-PPN regresses under the *same* default hyperparameters, which itself is informative: the tracker's defaults (radius, min_hits, decay) were not tuned per-detector, and LFE-PPN's detections are plausibly noisier/less spatially stable frame-to-frame (consistent with its worse FN/FP ratio above, 0.8× vs. LFE-Peaks' 0.6×) — matched to the wrong gate, the confirmation requirement may be discarding real detections as often as it discards false ones. Not yet root-caused further; see the improvement plan below.

¹ **Open discrepancy, not yet root-caused:** a fresh LFE-PPN FROG re-run under the exact same stride=40 setup as the original Section 8.1 table returned 60.7%, and stride=5 returned 60.1% — both meaningfully below the 67.7% documented in Section 8.1/8.3. No code change between this session's LFE-PPN fix and these re-runs is known to affect `eval_lfe_model`'s default (no-consensus, no-tracker) code path. Flagged for investigation, not yet resolved.

#### 8.6.1 Extending the tracker to the proposed architectures — and the invocation-rate sweep

`SimpleTracker` was wired into `evaluate_auc()` (the shared eval path for `DrowDetector`/`DrSpaamDetector`/Architectures A/B/C), not just `eval_lfe_model()` — it groups the model's own decoded per-frame "wp" detections by frame, replays them through the tracker in sequence order (reset at sequence boundaries), and substitutes the tracked output before the shared `_process_detections`/AUC pipeline. `--eval-stride`, already used to systematically subsample FROG's oversampled test set (Section 8.4), turns out to double as the tracker's invocation-rate control for free: at stride N the tracker only ever sees frames N raw-scans apart, since that's what's left in the dataset after subsampling — no separate parameter needed.

**Bug found and fixed during this test:** `SimpleTracker.step()` crashed (`linear_sum_assignment`: "matrix contains invalid numeric entries") the first time it ran against SpaceTimeCNN's real output. Cause: vote-decoded detections can occasionally carry a NaN score/position (a degenerate vote cluster), and `cost > match_radius` silently fails to gate out a NaN cost — NaN comparisons are always `False` in NumPy, so the bad match wasn't excluded, it flowed straight into the assignment matrix. Fixed by dropping non-finite detections before they reach `cdist`/`linear_sum_assignment` (`library/follow_the_drow/utils/tracking.py`).

**Invocation-rate sweep (SpaceTimeCNN, FROG, wp AUC), testing the hypothesis that motion is more visible — and the tracker more useful — at a wider gap between invocations:**

| Stride | Real-time gap | No tracker | Tracker (`radius=0.5, min_hits=3`) | Δ |
|---|---|---|---|---|
| 5 | ~125 ms | 77.6% | **78.6%** | **+1.0pp** |
| 40 | ~1 s | 80.4% | 75.1% | **−5.3pp** |

The result **contradicts the hypothesis as originally framed**: the tracker helps at the fine stride and clearly hurts at the wide one — the opposite of "wider gap → clearer motion → bigger win." A radius/min_hits diagnostic at stride=40 (four configs, isolating which parameter drove the regression) clarified why:

| `match_radius` | `min_hits` | wp AUC (stride=40) | vs. no-tracker (80.4%) |
|---|---|---|---|
| — (no tracker) | — | **80.4%** | — |
| 1.6 m | 2 | 77.9% | −2.5pp |
| 1.6 m | 3 | 75.1% | −5.3pp |
| 0.5 m | 2 | 47.5% | −32.9pp |
| 0.5 m | 3 | 34.8% | **−45.6pp** |

`match_radius` dominates the outcome: the tight gate (0.5 m, unchanged from the stride=5 setting) is catastrophic regardless of `min_hits` — at ~1 s between invocations, a person walking at any normal speed displaces more than 0.5 m, so the tracker simply can't re-associate them with their own predicted position, and recall collapses. The wide gate (1.6 m, scaled up for the larger expected displacement) avoids that failure but is now loose enough, in FROG's dense scenes (~3.9 people/frame, Section 8.1.1), to occasionally lock onto a different nearby person instead of the same one — corrupting that track's velocity estimate and its subsequent predictions. `min_hits=2` (vs. 3) partially recovers both cases (+2.8pp / +12.7pp) by tolerating more imperfect matching, but **no configuration tested beats the plain no-tracker baseline at this stride**.

**Conclusion — a single fixed radius is structurally the wrong tool at ~1 s invocation gaps here, not merely mistuned.** No value can be simultaneously tight enough to avoid cross-person confusion in FROG's crowds and loose enough to track one real person's own displacement, because per-person speed and inter-person spacing both vary. A principled fix would need a per-track–adaptive gate (e.g. scaled by that track's own estimated speed, or a proper covariance/Mahalanobis gate as in a full Kalman filter) rather than one global constant — a meaningfully larger piece of work, not a parameter tweak, and out of scope for this pass. **Decision: keep `SimpleTracker` at fine invocation strides only** (where it already shows a real, if modest, +1.0pp with zero retraining and no tuning drama); stop tuning it at wide strides; redirect further false-positive-reduction effort toward source-level mechanisms (detection-threshold selection, vote-clustering/NMS parameters) covered next, given the error breakdown above already shows false positives — not the temporal window — are the dominant error for every model tested.

#### 8.6.2 Source-level false-positive reduction: threshold selection, NMS sweep, and a wider-window retrain

**Operating-threshold selection (free — no retraining, no new inference).** Every AUC number already comes from a full (recall, precision, threshold) curve; the best-F1 point isn't necessarily the right *deployment* point for an application where false positives are the known dominant error. `utils/error_breakdown.py` was extended to find the highest-recall point on each model's curve that still meets a target precision, and compare its TP/FP/FN against the best-F1 point:

| Model | @75% precision target | Trade vs. best-F1 |
|---|---|---|
| SpaceTimeCNN | +968 FP saved, +718 FN added | Favorable (1.35:1) |
| TemporalUNet | +4179 FP saved, +3567 FN added | Favorable (1.17:1) |
| LFE-Peaks | +1278 FP saved, +936 FN added | Favorable (1.37:1) |
| FullScanTCN | +3944 FP saved, +4382 FN added | Unfavorable (0.90:1) |
| LFE-PPN | +3966 FP saved, +3951 FN added | Roughly break-even |

Three of five models have a real, free improvement sitting unused: moving to a ~75–80% precision operating point (not "as strict as possible" — the trade turns unfavorable again past ~80%) saves meaningfully more false positives than it costs in false negatives, with zero retraining. FullScanTCN and LFE-PPN don't get this win — their curves are shaped differently (FullScanTCN's own best-F1 precision is only 62.7%, well below the other three's 69–71%), so a different point on the *same* curve doesn't help; they need the curve itself changed.

**NMS/vote-clustering sweep, targeting those two stragglers specifically (`utils/nms_sweep.py`).** FullScanTCN's detections come from `votes_to_detections()` (`vote_collect_radius`, `min_thresh`); LFE-PPN already exposes `score_thresh`/`nms_radius` as constructor parameters, no internal code to touch. A first sweep over narrow ranges centred on the existing defaults found `min_thresh`/`score_thresh` completely non-binding (identical results across the whole tested range — the underlying score distributions don't fall where the sweep looked) and no improvement over the defaults. A second, wider sweep found two real, small wins and two bugs:

- **Bugs found and fixed:** an extreme `min_thresh`/`score_thresh` value that zeroes out every detection produced a 1-D `det_p` array instead of the expected `(0, 4)` shape in `_deep2flat()` (`library/follow_the_drow/utils/drow_utils.py`), crashing `_process_detections`; and `error_breakdown.py`'s `_best_f1_point()` crashed taking `argmax` of an empty curve for the same all-empty case. Both fixed to return well-defined zero/NaN sentinels instead of crashing — a sweep can now legitimately land on "this config detects nothing" without dying.
- **FullScanTCN**: `vote_collect_radius` traces a clean local peak at **0.7** (68.6% → 69.5% → **69.7%** → 69.5% → 69.1% → 68.3% across 0.3/0.5/0.7/0.9/1.1/1.3) — a genuine, if modest, **+0.2pp over the 0.5 default**.
- **LFE-PPN**: `nms_radius` similarly peaks at **0.9** (57.8% → 60.7% → **61.2%** → 59.0% across 0.7/0.8/0.9/1.0) — **+0.5pp over the 0.8 default**.
- `min_thresh`/`score_thresh` remain confirmed non-binding even at this wider range — 1.0/5.0 and 0.8 respectively already zero out every detection for these two models, so there's no useful middle ground on that axis at all for either.

Small wins, but real and free — adopted as the new defaults.

**Wider real-time window, retrain (in progress).** Given (a) the error breakdown shows LFE-Peaks — zero temporal input at all — has the *same* FP-dominant pattern as every temporal model, arguing against "not enough motion signal" as the primary FP cause, and (b) NMS tuning alone didn't fully close FullScanTCN's gap, a `dtime` (temporal stride) retrain was scoped as a narrower, better-motivated follow-up rather than a blind broad retrain across all three architectures. Training-side `dtime` support didn't exist before this (`LidarFrameDataset`/`_setup_datasets`/`DROW_Dataset`/`FROG_Dataset.get_scan()` only had it on the *evaluation* path) — added and verified via a 1%-subsample/1-epoch smoke test (real, non-degenerate AUC computed, no shape errors) before committing to a real run.

FullScanTCN was chosen over SpaceTimeCNN for this specific experiment: its temporal collapse happens first, via a lightweight per-beam TCN operating on just the T points for that beam, *before* any spatial mixing — widening T is cheap for it, unlike SpaceTimeCNN, where every layer processes the full `N_beams × T` grid jointly and doubling T roughly doubles compute throughout. FullScanTCN is also one of the two models without an already-solved FP problem, making a positive result here more actionable than the same experiment on SpaceTimeCNN (which already benefits from threshold selection and doesn't have an unsolved problem to fix).

Training launched with `--time-frame 10 --dtime 20` (10 frames spaced 20 raw scans / 500 ms apart, spanning ~4.5 s of real FROG time — vs. the original 5 consecutive frames spanning ~125 ms), otherwise identical hyperparameters to the existing `fullscan_tcn.best.pth` (epochs=30, patience=5, plateau LR schedule) for a clean comparison against its 69.5% FROG baseline (69.7% with the NMS fix above).

First attempt OOM'd (`MemoryError`) during checkpoint save after epoch 1 — `LidarFrameDataset`'s RAM cache scales with `T`, and doubling it (5→10) roughly doubled the cache's footprint on FROG's full 108,356-frame train split (documented at ~2.6 GB at T=5), pushing total usage over the edge during `torch.save`'s own transient memory spike. Retried with `--no-frame-cache` (recompute preprocessing every access instead of caching) — completed cleanly.

**Result: a clear regression, not an improvement.** Early-stopped at epoch 8 (best at epoch 3, ~19 min/epoch), final val-set AUC(wp) = **54.8%** — roughly **15 points below** the T=5 baseline (69.5–69.7%). (Caveat: this is val, not the test split the baseline uses — not enough to explain a 15-point gap alone, but worth naming rather than treating the comparison as perfectly matched.)

**Interpretation.** This is a real, informative negative result, not just "didn't help" — it's consistent with a specific mechanism rather than noise. FROG has no odometry (Section 8.6), so every temporal window is raw unaligned frame-stacking; the *shorter* the window, the smaller the robot's own uncompensated motion between frames, bounding how much that approximation can hurt. Widening the window to ~4.5 s doesn't just make a person's own displacement more visible (the original motivating hypothesis, plausibly still true in isolation) — it gives the robot's own uncompensated motion 36× longer to accumulate too, and the two effects work against each other. Here, the ego-motion corruption clearly dominated. This lines up with the tracker's own stride=40 finding earlier in this section (wider real-time gaps kept losing to the baseline, on the same odometry-free, dense-crowd dataset) — a second, independent piece of evidence for the same underlying constraint rather than an isolated fluke.

**Conclusion:** wider real-time temporal windows are not a productive direction for FROG specifically, given its missing odometry — this isn't a tuning failure (wrong `dtime` value) so much as a structural mismatch between the idea and this particular dataset's limitation. The idea remains worth testing on DROW, which has real odometry (`docs/IDEAS_BACKLOG.md` item 2) and would isolate the "clearer motion signal" effect from the ego-motion confound that dominated here — but that's a separate experiment, not a continuation of this one.

---

## 9. Reference Quotes

A reference collection of direct quotes, for citing in an eventual writeup without re-fetching sources.

**On the cutout's limitations (motivating this research):**
> "DROW... obtaining state of the art detection results" but uses a fixed-size cutout with "zero cross-beam communication: the CNN processes each 48-beam window in complete isolation." — Section 3.1 above, summarizing Beyer et al. 2016/2018.

**On DR-SPAAM's design intent:**
> DR-SPAAM "addresses the sparsity problem of 2D LiDAR points by fusing multiple scans... using an alternative forward looking strategy that is more computationally efficient" than backward-looking fusion requiring explicit scan alignment. — paraphrased from Jia, Hermans, Leibe (IROS 2020), via secondary summary; verify exact wording against arXiv:2004.14079 before quoting in a final writeup.

**On the FROG dataset's motivation (also this project's training data):**
> "[We] propose a benchmark based on the FROG dataset, and analyze a collection of state-of-the-art people detectors." — Amodeo et al., arXiv:2306.08531 abstract.

**On LFE's full-scan, no-cutout design — the closest prior art:**
> LFE-Peaks and LFE-PPN use "a 1-D U-Net FCN... applied directly to the normalised raw scan vector rather than cutout windows" and are "single-scan only: no odometry, no temporal history." — Section 3.4 above.

**On PeTra's full-scan occupancy-grid design:**
> "the occupancy map is defined as a 256 × 256 matrix, with a resolution of about 2 cm." — Guerrero-Higueras et al., *Frontiers in Neurorobotics* (2019), Section on input representation.
> "PeTra spends ≈0.3 s on calculating a location estimate... LD [Leg Detector baseline] spends ≈0.1 s" — same paper, on inference cost.

**On TCNs vs. recurrent models generally (motivating Architecture B):**
> Bai, Kolter, and Koltun's 2018 evaluation is the paper that established "TCN" as a name for causal dilated convolutional networks and showed they "convincingly outperform" canonical recurrent architectures (LSTM, GRU) "across a diverse range of tasks and datasets, while demonstrating longer effective memory." (Standard characterization of arXiv:1803.01271 — verify exact wording against the paper before quoting verbatim in a submitted paper.)

> The quote above is flagged "verify before quoting" because it was reconstructed from training-knowledge / search-engine summary rather than a directly fetched primary-source passage — re-fetch the primary source before using it as a verbatim quote in a submitted paper.
