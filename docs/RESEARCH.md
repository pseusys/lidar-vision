# Full-Scan, Non-Recursive Convolutional Architectures for 2D LiDAR Person Detection

> Formalized research statement and literature review for this repository. For a client-facing overview, see the [top-level README](../README.md); for a map of the codebase, see [AGENTS.md](../AGENTS.md).

> **This document's title names the earlier line of work, not the current one.**
> Since 2026-09-14 the project's contribution is **TAKHeLiPeD** â€” **T**(emporal) **A**(daptive) **K**(nee-)**He**(ight) **Li**(dar) **Pe**(rson) **D**(etector), *TA-KHÃ©-Li-PeD*, called *the three-horizon detector* in everything written before 2026-09-23 â€” which deliberately lifts constraint 1 of Â§2 in favour of an object-level recurrent memory, and holds the current result (83.3% AP on FROG against DR-SPAAM's published 75.6%; [`../memory/performance-log.md`](../memory/performance-log.md)).
> Its design is [`PROPOSAL.md`](PROPOSAL.md) and the measurements it is argued from are [`PAPER.md`](PAPER.md).
> **What stays authoritative here**: Â§1 (the problem and the metric), Â§3 (the published baselines and the fidelity audit against each paper), Â§4 (preprocessing), Â§7 (related work). Â§2's constraints, Â§5's three architectures and Â§6's design history are kept as provenance â€” every accuracy number they carried is retracted (`../memory/performance-log.md`). Rewriting this document around TAKHeLiPeD is [`../TODO.md`](../TODO.md) Â§A.

---

## 1. Problem Statement

A mobile robot carries a 2D lidar scanner mounted at leg height (~40 cm off the ground). The scanner rotates in a horizontal plane and produces a single array of **N range measurements** per timestep â€” one distance value `r_i` for each beam angle `Ï†_i`, uniformly distributed across the sensor's field of view (DROW: N â‰ˆ 450 beams at 0.5Â° spacing; FROG: N = 720 beams at 0.5Â° spacing; JRDB: N = 541 beams at 0.5Â° spacing over a 270Â° FoV).

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

**This research asks: can a CNN that processes the *entire* scan at once â€” with no cutout, no per-beam windowing, no hand-set spatial extent â€” match or exceed cutout-based detectors' accuracy, while removing that fixed-window assumption?**

Three further constraints, made deliberately and stated up front so every later design choice is explainable against them:

1. **Non-recursive only.** Every architecture is built exclusively from CNN and TCN layers (plus pooling/norm/linear "glue") â€” no RNN/GRU/LSTM, no attention. This is an efficiency choice: it's the architectural feature expected to be the actual winning point against Li2Former (a temporal-Transformer cutout detector with O(N_beamsÂ²)/O(TÂ²)-flavoured cost) and against any recurrent design â€” feed-forward, fully parallel layers are cheaper to run and trivially batchable.
2. **Raw input, no preprocessing.** Architectures consume the raw scan â€” angle (implicit in beam index) and distance only. No cutout and no conversion to Cartesian `(x, y)`. This is a bigger departure from prior work than it looks: DROW, DR-SPAAM, and this project's own earlier full-scan designs all rotate historical scans into the current frame using odometry before doing anything else (Section 3.2). Dropping the Cartesian conversion turns the `T`-frame window into a trivial rotating buffer at inference time â€” see Section 6.1.
3. **Compare against the cutout-based research.** All of the above is benchmarked against DROW, DR-SPAAM, and Li2Former, on the same datasets and the same evaluation pipeline (Section 8).

> **Lifted for one model, 2026-09-14.** The A43 architecture (`docs/PROPOSAL.md`) lifts constraint 1 for an object-level recurrent memory confined to its slots, and constraint 2 in favour of beam-local Cartesian input, as settled with the owner on measurements made after these constraints were written (`TODO.md` A42, A43). `SpaceTimeCNN`, `FullScanTCN` and `TemporalUNet` keep both constraints.

Concretely, this research proposes **three** architectures (Section 5): a joint space-time CNN, a TCN-then-CNN fusion, and a temporal U-Net â€” see Section 6 for why three, not two.

### 2.1 Provenance: replicated vs. novel

Every detector this repo trains or evaluates falls into exactly one of three provenance categories. This distinction matters for how to read any accuracy number produced by this codebase: a "replicated" score is a reproduction of a published result, a "novel" score is this project's own claim.

| Detector | Category | Evidence |
|---|---|---|
| `DrowDetector` | **Replicated â€” official weights** | Architecture is a direct port of the official DROW notebook release; loads the official checkpoint `final-WNet3xLF2p-T5-odom.rot-trainval-50ep.pth.tar` (`library/setup.py`) via `load_state_dict`, which only succeeds because the layer shapes match the original bit-for-bit. |
| `DrSpaamDetector` | **Replicated â€” official weights** | Module docstring states the code matches "the official SpatialDROW implementation"; module structure is deliberately nested to load the official checkpoint `dr_spaam_e40.pth` exactly. |
| `LFEPeaksDetector` / `LFEPPNDetector` | **Replicated â€” official weights (ONNX)** | Loads the authors' own exported ONNX models (`LFE-Peaks.onnx`, `LFE-PPN.onnx`, hosted at `robotics.upo.es/~famozur/onnx/`) via `onnxruntime.InferenceSession` â€” genuine published-model inference, not a reimplementation. |
| `Li2FormerDetector` | **Reimplemented â€” no official code or weights** | Header says the architecture is "inspired by" Yang et al. (weaker language than the DROW/DR-SPAAM ports) and explicitly notes no published weights exist; this project trains it from scratch from the paper's description alone. No official repo was available to verify against, so treat any AP this project reports for Li2Former as this project's own reproduction attempt, not a confirmed match to the paper's reported 0.764 (DROW) / 0.814 (JRDB). |
| `AlgorithmicDetector` | **Prof. O. Aycard's classical algorithm, ported â€” not this work, and not from any of the SOTA papers compared here** | The algorithm is **due to prof. O. Aycard**, whose C++ implementation (`cpp_core/sources/detector.cpp`, header comment "Person detector using 2 lidar data / Written by O. Aycard") is what this repo builds and calls; **it must be credited to him in any write-up, and only the pybind11 binding, the Python wrapper and the evaluation harness around it are this project's**. It is a distinct two-tier leg+chest clustering/tracking algorithm â€” not Pantofaru's ROS `leg_detector`, not Arras et al., not Leigh et al. (Section 2.2) â€” included as this repo's non-learned reference point rather than as a port of any baseline used by DROW/DR-SPAAM/Li2Former/FROG. A formal bibliographic reference for it is still to be obtained from the author (`../TODO.md` Â§A). |
| `SpaceTimeCNNDetector`, `FullScanTCNDetector`, `TemporalUNetDetector` | **Novel â€” proposed by this work** | No prior paper describes these architectures, so no official weights could exist; each must be trained locally. Individual sub-components (SE blocks, Inception-style branching, causal TCN, U-Net skip connections) are explicitly credited to their originating papers in the code (Section 5), but the overall architectures and their combination are this project's own design. |

### 2.1.1 Fidelity audit: how precisely is "replicated" replicated?

The categories above are the right level of description, but "replicated" is not automatically "pixel-perfect" â€” every replicated/reimplemented detector was audited line-by-line against its official source (code where public, paper text otherwise). Findings, most severe first:

- **`LFEPeaksDetector` / `LFEPPNDetector` â€” one functional deviation found and fixed, one remaining.**
  - ~~**LFE-Peaks was missing the paper's centroid-merge/NMS step.**~~ **Fixed.** The FROG paper (Â§5.2.4) describes finding peaks in the segmentation signal, then merging nearby centroids ("Centroids that are close together are interpreted as legs or part of legs, and merged together into final person detections using a NMS-like process"). `LFEPeaksDetector.detect()` previously reported every raw `find_peaks` hit as its own detection, so a person whose two legs produced two separate peaks would score as two detections instead of one. It now runs the same greedy centroid-merge as `LFEPPNDetector`'s NMS (`_merge_nearby()`, shared by both classes), with a `merge_radius` defaulting to `_PERSON_RADIUS = 0.4 m` â€” a reasonable value for "same person's leg parts," but not one the paper states explicitly, since it doesn't give an exact radius either.
  - **Cross-dataset zero-padding is still an unverified extrapolation â€” not fixed, and not straightforward to fix.** The FROG paper only ever evaluates LFE on FROG's own 720-beam scans; it never describes running it on DROW (450 beams, 225Â° FoV) or JRDB (541 beams, 270Â° FoV). This repo's zero-padding of shorter scans to 720 (`_pad_to_trained`) is this project's own invention, and it does not resample to FROG's 0.25Â°/beam angular resolution â€” so on DROW/JRDB, beam index no longer corresponds to the same physical angle the network was trained on. Treat any DROW/JRDB numbers from LFE as a known-approximate cross-dataset transfer, not a faithful replication (this was already flagged as a caveat in Section 3.4, but the padding scheme itself is now confirmed unverified against the paper, not just "may degrade").
  - ~~The exact input-normalization formula and the LFE-PPN anchor count (31) could not be located in the paper text; the PPN depth-spacing formula in this repo's code omits a `-1` term the paper's formula has (~3% scale difference).~~ **Settled empirically, 2026-09-10, by reading `lfe_ppn.onnx` and scoring the plausible conventions against the published 66.5% AP** (`CHANGELOG.md`): 30 anchors (the model's own output shape), endpoint-inclusive `linspace(0.2, 10.0, 30)` â€” i.e. the `-1` form is right, 68.1% against the step form's 67.5% â€” and **both offset channels normalised by their own anchor cell's extent**, which is what the decoder had wrong (raw metres 50.5% against anchor-spacing 65.8-68.5%). Channel 0 of `grid` is already sigmoided inside the graph.

- **`DrowDetector` â€” matches with one minor preprocessing deviation.** `load_state_dict` is called strict (no `strict=False` anywhere), and the network definition is character-for-character identical to the official `DROWNet3LF2p` (verified directly against `VisualComputingInstitute/DROW`'s `v2/Clean Final* [T=5,net=drow3xLF2p,odom=rot,trainval].ipynb`) â€” channel progression, kernel sizes, LeakyReLU(0.1), pooling, dropout, weight init, and the T-sum temporal fusion are all identical. The one confirmed deviation: the official `cutout()` uses `cv2.INTER_AREA` (box-filter averaging) when downsampling a window to 48 samples and `cv2.INTER_LINEAR` only when upsampling; this repo's `cutout()` always uses linear interpolation regardless of direction. Since near-range detections typically downsample, this changes the input feature distribution at close range â€” a real if minor deviation from the original preprocessing (it does not affect weight loading, since it's a data-preparation step, not a model parameter).

- **`DrSpaamDetector` â€” matches with one training-only deviation.** `load_state_dict` is strict at both call sites, and the conv blocks, spatial-attention formula, published hyperparameters (`alpha=0.5`, `window_size=11`, `N_SAMP=56`, pedestrian-only head), and detection heads all match the official `VisualComputingInstitute/DR-SPAAM-Detector` source exactly. The one confirmed deviation: this repo's training-time forward pass explicitly `.detach()`es the auto-regressive template on all but the last timestep, whereas the official training code back-propagates through the full T-step chain with no detach. This has no effect on inference (loaded weights are identical either way) but means retraining from scratch here would not exactly reproduce the official training dynamics.

- **`TemporalUNetDetector`** â€” see Section 6.4; the U-Net implementation itself is confirmed correct against the canonical definition, but its relationship to LFE's *actual* backbone needed a correction (below).
- **`SpaceTimeCNNDetector`** and **`FullScanTCNDetector`** â€” see Sections 5.1 and 5.2; both had at least one inaccurate/overstated literature citation that has been corrected.

### 2.2 SOTA baselines mentioned in the literature but not replicated here

The four papers this research compares against (Sections 3, 7) each report their own baseline comparisons. Collated across all of them, three classical detectors recur that this repo does **not** implement:

| Baseline | Used as a comparison point by | Why not replicated here |
|---|---|---|
| **ROS `leg_detector`** (Pantofaru, ROS package, 2010) â€” jump-distance leg clustering + random-forest classifier + Kalman-filter tracking, off-the-shelf pretrained model | DROW (IROS 2018), Li2Former (TIM 2024), and FROG (2025) all use it as their weakest baseline | Consistently the lowest-scoring method in every one of those papers' own tables (e.g. FROG Table 4: mAP 15.8 vs. 73+ for DROW3/DR-SPAAM). This repo already carries a non-learned reference point, prof. O. Aycard's `AlgorithmicDetector` (Section 2.1) â€” replicating a second classical leg-tracker would duplicate that role without adding anything to the actual research question (full-scan vs. cutout **deep** architectures). **Reconsidered and declined 2026-10-11** (`TODO.md` Â§A item 33), on the owner's proposal to prefer it precisely because it is the official, better-maintained ROS answer: see below. |
| **Segment-based detector, Arras et al.** (ICRA 2007) â€” jump-distance segmentation + boosted-features classifier on hand-crafted geometric features | DROW (IROS 2018), Li2Former (TIM 2024) | No official training code was ever released â€” even the DROW paper itself had to fall back on a third-party ROS reimplementation (Linder & Breuers et al.) rather than the original. Replicating it here would itself be an unverified reimplementation, several steps removed from the actual paper, for a baseline that isn't part of this project's own architectural question. |
| **Joint leg tracker, Leigh et al.** (ICRA 2015) â€” Kalman-filter tracker jointly associating pairs of leg clusters into person tracks | DROW (IROS 2018), Li2Former (TIM 2024) | The DROW paper's own evaluation found it stops producing detections beyond ~7 m range and requires per-dataset threshold retuning to be usable at all â€” a known, paper-acknowledged limitation of the two-leg-assumption approach, not a competitive comparison point for this project's learned-detector research. |
| **PeTra** (Guerrero-Higueras et al., Frontiers in Neurorobotics, 2019) â€” full 2D U-Net over a 256Ã—256 rasterized occupancy-grid image, with an external leg-pairing post-process | FROG (2025), as its "modern leg-based" baseline | Already discussed as the closest *related* full-scan work in Section 7.2 â€” but its task framing differs (binary occupancy-grid segmentation, not this project's shared beam-classification-plus-vote-regression head) and it is markedly slower in FROG's own benchmark (~28 ms/scan vs. <2 ms for LFE, ~14 ms for DROW3/DR-SPAAM). A fair replication would need a separate evaluation harness rather than a drop-in addition to the shared pipeline. |

**The `leg_detector` question, reopened and settled 2026-10-11.** The owner asked whether it should *replace* Aycard's
detector rather than merely be skipped, on the reasonable grounds that it is the detector a ROS user would actually reach for
and is better explored and maintained than an algorithm written for one lab. Three findings decided it, and the resolution was
to **cite it rather than measure it**:

1. **It is not the rule-based counterpart it looks like.** `leg_detector` extracts hand-crafted features from laser clusters â€”
   point count, width, linearity, circularity, boundary regularity â€” and feeds them to a **trained OpenCV random forest shipped
   with the package**. It is a classical-*ML* baseline. Aycard's detector is the only genuinely rule-based point in this
   comparison, so substituting one for the other would not swap like for like; it would delete a category.
2. **A number for it here would be zero-shot, and this repo has a rule about that.** Its shipped forest was trained on the
   authors' own data at their own sensor height, so a FROG score would be cross-regime transfer reported beside in-domain
   numbers â€” the exact mixing `AGENTS.md` rule 3 exists to prevent, requiring the caveat restated at every mention.
3. **Obtaining one costs days, not hours.** It is a ROS 1 catkin package, so it means either replaying FROG as `LaserScan`
   messages inside the Docker image and capturing its output, or porting its feature extraction and forest and verifying the
   port against the original. "Maintained" also cuts the other way: ROS 1 is end-of-life and this detector has no ROS 2
   successor in the `people` stack.

So it is named here as the standard ROS answer and as the weakest baseline in three of the four papers this project compares
against, with no row of our own â€” **revisited only alongside a real-ROS deployment of TAKHeLiPeD**, where its own stack is
available and the comparison would sit in its natural setting instead of being ported into ours.

No other model variants were found published alongside `LFEPeaksDetector`/`LFEPPNDetector`: the FROG paper's own benchmark (Table 4) and the authors' file host (`robotics.upo.es/~famozur/onnx/`) contain exactly these two ONNX exports for the LFE family â€” `LFE-Peaks.onnx` and `LFE-PPN.onnx`, both already bundled here â€” alongside unrelated models from the same lab's other projects (object detection, dialogue ontologies), not additional LFE variants.

---

## 3. Previous Research and Baselines

These are already reproduced and runnable in this repo (`train.py --detector drow|drspaam|li2former`, `evaluate.py --lfe-peaks --lfe-ppn`).

### 3.1 DROW â€” Beyer, Hermans, Leibe (ICRA 2016, IROS 2018)

**Reference:** Beyer, Hermans, Leibe. *DROW: Real-Time Deep Learning-Based Wheelchair Detection in 2-D Range Data*. ICRA 2016 / RA-L 2017 extension. arXiv:1603.02636; 3-class extension (wheelchair/walker/person) in the IROS 2018 paper, arXiv:1804.02463.

> "We present a method to detect, segment, and track multiple people in a 2-D range scan... obtaining state of the art detection results." â€” abstract

#### Key idea: the cutout

For each beam `i`, a window of `S = 48` neighbouring beam ranges is extracted in polar coordinates and centred on beam `i`. This **cutout** forms a 1D signal of length 48 that encodes the local angular neighbourhood of the beam.

A shared 1D CNN is applied to every cutout independently â€” beams are processed in parallel with no cross-beam communication during feature extraction. The network has three convolutional blocks followed by two upsampling heads (class logits and vote offset).

#### Temporal aggregation: fixed sum

When using `T` consecutive scans, the per-beam CNN is applied to each timestep independently, producing `T` feature vectors per beam. These are simply **summed** across the time dimension before the detection heads. All timesteps are weighted equally; the network has no way to down-weight a noisy or motion-blurred frame.

```
for each timestep t = 1â€¦T:
    cutout_t[i]  = extract_48_beams(scan_t, beam_i)
    feature_t[i] = CNN(cutout_t[i])          â† no cross-beam communication

feature[i] = Î£_t feature_t[i]               â† fixed sum, equal weights
logit[i], vote[i] = detection_head(feature[i])
```

#### Odometry alignment

Before feature extraction, historical scans are rotated into the current scan's coordinate frame using the robot's odometry (differential rotation Î”Î¸ between timesteps). Only rotation is applied â€” translation is ignored, approximating that the robot has moved little between T consecutive frames.

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

#### âš  Annotation quality concern â€” verified against source (Section 8)

The DROW dataset uses **sparse annotations**: only ~5% of recorded scan frames in each sequence carry ground truth labels (measured directly on the bundled data: 17,665 of 341,147 raw train scans = 5.18%, matching the DROW paper's own abstract â€” "464k laser scans, out of which 24k were annotated" = 5.17%). Annotated frames are not scattered arbitrarily; they land at a regular ~1-in-5 cadence (median gap = 5 raw scans = 0.4 s between consecutive annotated frames).

This was initially assumed to create a "false-negative label noise" training pathology â€” unannotated frames being penalised as if any detection on them were a false positive. **That assumption was checked against the actual training/eval code and does not hold**: `LidarFrameDataset`'s sample index (`utils/train.py`, `LidarFrameDataset.__getitem__`) enumerates only `(sequence, det_idx)` pairs â€” one entry per *annotated* frame â€” so loss is computed exclusively at labelled frames. Unannotated frames are never used as supervision targets; `DROW_Dataset.get_scan()` / `FROG_Dataset.get_scan()` pull the T-frame temporal window from the dataset's separate, dense, continuous raw-scan array (all ~341k DROW scans, not just the labelled 17,665), so the window ending at an annotated frame is real consecutive sensor data, not backfilled or interpolated. This matches the official DROW-v2 reference loader's `get_batch()` (`times = arange(iscan - ntime*dtime+1, iscan+1, dtime)`, same repeat-padding at sequence start) almost exactly.

The real, verified limitation is **coverage, not label noise**: DROW train has far fewer supervised examples than FROG, on two axes at once â€” 6.1Ã— fewer annotated frames (17,665 vs 108,356) *and* 2.9Ã— fewer people per annotated frame (1.35 vs 3.93, measured directly), for a combined **~17.8Ã— fewer total person-annotation instances** (23,910 vs 425,407). This scarcity is the direct cause of the DROW-native training results in Section 8.1.1/8.5 â€” see the overfitting evidence there (val loss regressing from epoch 2 onward on a model regularised the same way as its FROG counterpart).

Additionally, ground truth positions were obtained by correlating lidar timestamps with an external reference (camera footage or motion capture), introducing **temporal synchronisation error** at fast motion speeds. Annotations very close to the sensor (< 1 m) or at the edge of the field of view are known to be less reliable due to projection ambiguity.

The FROG dataset was recorded specifically to address the coverage problem: every scan frame is annotated, at a much higher per-frame person density, providing a far denser training signal.

### 3.2 DR-SPAAM â€” Jia, Hermans, Leibe (IROS 2020, RA-L 2022)

**Reference:** arXiv:2004.14079.

DR-SPAAM retains the DROW cutout extraction and per-beam CNN but replaces the fixed temporal sum with an **auto-regressive spatial attention mechanism** that directly addresses DROW's weaknesses (1) and (2).

This project's implementation is the official **SpatialDROW** from the RA-L 2022 release, loaded from the published checkpoint `dr_spaam_e40.pth`.

#### Architecture: four conv blocks + spatial attention gate

```
Block 1  (1â†’128,  3 layers) + MaxPool(2)  â†˜
Block 2  (128â†’256, 3 layers) + MaxPool(2)  â†’ 256-ch feature map per beam (14 pts for S=56)
                                             â†“ spatial attention gate
Block 3  (256â†’512, 3 layers) + MaxPool(2)
Block 4  (512â†’128, 2 layers) + AvgPool
Conv1d heads â†’ logits (1 or 4 classes), votes (2)
```

#### Auto-regressive spatial attention (the "A" in DR-SPAAM)

After blocks 1â€“2, each beam has a feature map `f_t[i]` (spatial, not pooled). The temporal aggregation uses an **auto-regressive template** rather than explicit temporal attention weights:

```
template_0 = f_0[:]              â† initialised from first scan in window (detached)

for t = 1â€¦Tâˆ’1:
    feat_t[i] = encode(scan_t, beam_i)   â† blocks 1â€“2

    # spatial attention: beam i attends to Â±window/2 neighbours in template
    attn_weight[i,j] = softmax( embed(feat_t[i]) Â· embed(template[j])
                                for j in iâˆ’K â€¦ i+K )
    template[i] = Î± Â· feat_t[i]  +  (1âˆ’Î±) Â· Î£_j attn_weight[i,j] Â· template[j]
                  â†‘ current feat       â†‘ attended historical template

final_feature[:] = decode(template[:])   â† blocks 3â€“4 + heads
```

Key properties:

- **Stop-gradient on template input**: gradients only flow through the current scan's encode/gate path, not back through the full T-step chain (prevents BPTT instability).
- **Local angular context** (Â±5 neighbours, `window_size=11`): beam `i` can observe its closest angular neighbours in the historical template.
- **Alpha blending** (`alpha=0.5`): the gate blends current features with the attended template, preventing the template from collapsing to a running mean.
- **56-pt cutouts** (`N_SAMP=56`): published weights use a 56-sample polar window instead of DROW's 48.
- **Pedestrian-only output**: published weights (`dr_spaam_e40.pth`) output a single sigmoid class (person only). This project's training configuration uses 4 classes.

#### Ablation results from the paper

| Component | AP (wp) | Î” vs DROW |
|---|---|---|
| DROW (sum only) | 0.619 | â€” |
| Temporal attn only | ~0.640 | +2.1 pp |
| Spatial attn only | ~0.659 | +4.0 pp |
| DR-SPAAM (both) | **0.696** | **+7.7 pp** |
| DR-SPAAM RA-L (2022) | **0.720+** | **+10+ pp** |

#### âœ… Fidelity gap found, root-caused, and fixed (Section 8)

Running the published `dr_spaam_e40.pth` weights through this repo's evaluation pipeline on the DROW test set originally gave **25.8% wp AUC**, far below the paper's reported 69.6â€“72%+. Four explanations were tested directly rather than assumed:

1. **Input/cutout shape mismatch?** Ruled out â€” `DrSpaamDetector.N_SAMP=56` is read correctly by the shared cutout-extraction code (`_extract_input`), verified by direct inspection: cutout tensors are `(450, 5, 56)`, matching the published weights' expected shape exactly.
2. **Wrong `load_published()` construction args?** Ruled out â€” `alpha=0.5, window_size=11, num_pts=56, pedestrian_only=True` were checked against the official DR-SPAAM-Detector repo's own training config (`dr_spaam/cfgs/dr_spaam.yaml`) and match exactly.
3. **Postprocessing (vote-clustering) hyperparameter mismatch?** Tested directly and ruled out â€” the official config's `vote_kwargs` (`vote_collect_radius=0.157`, `min_thresh=9.4e-5`, `blur_sigma=1.46`, vs. this repo's shared defaults of `0.5`/`1e-3`/`2.0`) were substituted in for an isolated re-run: **26.2% wp AUC â€” no meaningful change** from the 25.8% baseline.
4. **Temporal window depth?** The official `dr_spaam.yaml` specifies `num_scans: 10` (the DROW-only `drow5.yaml`, a *different* network type, uses 5) â€” `load_published()` was defaulting to 5. Fixed (`evaluate_auc()` now reads `net.num_scans` when present instead of always using `dataset.time_frame`), but a direct forward-pass comparison at T=5 vs T=10 on the same real frame showed **no meaningful difference** in output confidence â€” not the cause, though the fix is correct and kept for fidelity.

The actual root cause, found by cloning the official [DR-SPAAM-Detector](https://github.com/VisualComputingInstitute/DR-SPAAM-Detector) repo and diffing the real `scans_to_cutout()` source (not just the config) against this repo's `cutout()`: `dr_spaam.yaml`'s `cutout_kwargs` specify `window_width: 1.0, window_depth: 0.5` â€” narrower and shallower than DROW's own defaults (`window_width: 1.66, window_depth: 1.0`, which this repo's `cutout()` hardcoded unconditionally for every cutout-based model). Worse, the official implementation **normalises** the centred cutout by dividing by `window_depth` (`ct = (ct - dists) / window_depth`) â€” a step this repo's `cutout()` was missing entirely. For DROW's own published weights (`window_depth=1.0`) that division is a no-op, which is why `DrowDetector` was unaffected and looked fine throughout; for DR-SPAAM's published weights (`window_depth=0.5`) the missing division under-scales every feature by 2Ã—, compounded by clipping to the wrong (2Ã— too wide) depth tunnel in the first place.

**Fix**: `cutout()` in `drow_utils.py` now divides the centred window by `thresh_dist` (backward-compatible no-op at the DROW default of 1.0); `DrSpaamDetector` gained `WIN_SZ=1.0` / `THRESH_DIST=0.5` class attributes (matching `dr_spaam.yaml`), threaded through `_extract_input()` in `train.py`. Re-running the DROW eval after the fix: **wp AUC 25.8% â†’ 68.1%** â€” landing right in the paper's 69.6â€“72% range, and now sensibly *above* `DrowDetector`'s own (also-fixed-pipeline) 66.5% wp AUC, matching the paper's ordering (DR-SPAAM > DROW). The earlier "structurally under-confident" pattern (median max-confidence 0.465 on real positive frames, vs. `DrowDetector`'s 0.98) was exactly what a 2Ã— feature-scale shrink would produce â€” consistent with this being the actual cause, not a deeper attention-mechanism bug as originally suspected.

#### Limitations of DR-SPAAM

1. **Still uses fixed 56-beam cutouts**: the CNN has no access to raw data beyond 56 beams, and the cutout boundary is hardcoded.
2. **Spatial attention operates on compressed features**: neighbouring beams only communicate *after* independent CNN processing. Raw-signal cross-beam context is impossible.
3. **Local attention only** (Â±5 neighbours): each beam can see only its closest neighbours. Global scan structure (e.g., two legs of the same person widely separated) is not captured.
4. **Performance-driven design**: DR-SPAAM was developed under real-time constraints (ROS node, embedded hardware). Many architectural choices reflect computational budget rather than theoretical optimality.

### 3.3 Li2Former â€” Yang et al. (IEEE TIM 2024)

**Reference:** Yang et al. *Li2Former: Omni-Dimension Aggregation Transformer for Person Detection in 2-D Range Data*. IEEE Trans. Instrumentation and Measurement, 2024. DOI: 10.1109/TIM.2024.3420353.

Li2Former retains the cutout-based input representation but replaces DR-SPAAM's spatial attention with a **temporal Transformer** that attends over T consecutive cutouts per beam independently.

```
For each (beam, timestep) pair â€” cutout width P = 64 samples:

  _ConvBackbone(1 â†’ d_model=512):
    Stage 1: Conv1d(1â†’64)Ã—2  + Conv1d(64â†’128)  + MaxPool(2)   â†’ P/2 positions
    Stage 2: Conv1d(128â†’128)Ã—2 + Conv1d(128â†’256) + MaxPool(2)  â†’ P/4 positions
    Stage 3: Conv1d(256â†’256)Ã—2 + Conv1d(256â†’512) + AdaptiveAvgPool(1) â†’ 1 position
    â†’ (B*N*T, 512) feature vector per (beam, timestep)

Reshape to (B*N, T, 512)
+ Sinusoidal positional encoding over T timesteps
TransformerEncoder(d_model=512, nhead=8, 1 layer)  â† temporal self-attention
Mean-pool over T  â†’ (B*N, 512)

Classification head : Linear(512 â†’ 1)            binary person logit
Regression head     : Linear(512 â†’ 1024) â†’ ReLU â†’ Linear(1024 â†’ 2)  vote offsets
```

Key properties:

- **Temporal attention, not spatial**: DR-SPAAM attends across neighbouring beams at a fixed timestep; Li2Former attends across T timesteps for a single beam. The two mechanisms are complementary.
- **Wider cutout** (P=64 vs. 48 for DROW, 56 for DR-SPAAM): more angular context per beam.
- **Binary output** (person vs. background): the sigmoid probability is placed in the pedestrian slot for compatibility with the unified evaluation pipeline.
- **No published weights**: must be trained from scratch using `train.py --detector li2former`.
- Recurrent-free in the RNN sense, but uses a temporal Transformer â€” exactly the kind of component constraint 1 (Section 2) excludes from this project's own proposed architectures, and the comparison point that constraint's efficiency argument is made against.

### 3.4 LFE-Peaks and LFE-PPN â€” Amodeo et al. (2025)

**Reference:** Amodeo, PÃ©rez-Higueras, Merino, Caballero. *FROG: A new people detection dataset for knee-high 2D range finders*. Frontiers in Robotics and AI, 2025. arXiv:2306.08531.

The LFE detectors are inference-only ONNX baselines from the FROG benchmark paper. They differ from all other detectors in two important ways: they operate on a **single scan with no temporal context**, and they use a **1-D U-Net FCN** (the LFE backbone) applied directly to the normalised raw scan vector rather than cutout windows.

```
Input: (1, N=720, 1)   â€” normalised range vector  (1.0 = near, 0.0 = far)
1-D U-Net FCN (encoder-decoder with skip connections)
â†’ per-beam probability map (1, N, 1)
```

They differ only in the detection head:

| Variant | Head | Post-processing |
|---|---|---|
| **LFE-Peaks** | Per-beam sigmoid probability | `scipy.find_peaks` on the 1-D probability map |
| **LFE-PPN** | Anchor grid: N/6 sectors Ã— 30 depth anchors Ã— 3 outputs | Anchor decoding + greedy distance-based NMS |

**LFE-Peaks' graph, read out of the released ONNX file** (`TODO.md` Â§A item 7, 2026-10-10 -- the port attempt was dropped, but the trace it produced is the architecture, which the paper's prose does not give). `lfe_peaks.onnx` holds **29 convolutions and 53,258 parameter
values**, and walking its dataflow gives the architecture exactly -- which the published paper's prose does not:

    stem        a 1x1 lift of the raw scan to 32 channels, kept for the first residual, beside a kernel-9 convolution on the
                single raw channel and a 1x1 into 32 channels, with a folded batch norm
    block 1     depthwise 7 -> pointwise 1 -> kernel 5, each followed by a folded norm, added to the lift
    blocks 2-3  the beams' maximum taken and broadcast back, concatenated to **64** channels, through a depthwise 9 (groups 64)
                and a pointwise back to 32, then the same three convolutions as a residual branch
    blocks 4-5  the same, but the concatenation is first projected 64 -> 32 by a 1x1 and then widened again to **128**
    head        1x1 to one channel, a bias, a sigmoid

A port built to that order reproduced the graph's convolution inventory **exactly** -- every shape, kernel and grouping in
sequence -- which is the structural claim that matters. **What stopped it** was the dataflow of blocks 4 and 5: the graph
concatenates the projected branch back in a way that reaches 128 channels through an `Expand` whose semantics did not fall out
of the node list, and guessing it would have produced a network we could not call LFE's. **Two things a future attempt should
know:** the released weights are *not* needed -- they only verify the port, and training from scratch under their recipe
reproducing ~65% verifies it better -- and the remaining work is reading `Expand`/`Concat` shapes at nodes 104-118 and 148-164
with a shape inference pass rather than from `node.input` alone.

Key constraints: single-scan only (no odometry, no temporal history); trained on 720-beam FROG scans (inference on 450-beam DROW or 541-beam JRDB scans requires zero-padding â€” see the fidelity caveat below); class-agnostic (person confidence only); ONNX inference only (weights are bundled automatically; re-training is not supported within this framework).

**This is the closest prior work to this project's own proposed architectures** â€” see Section 7.2.

> **Fidelity caveat (Section 2.1.1 has the full detail).** The ONNX weights this repo loads are the authors' own published models â€” the *network* is exact. The Python wrapper around it has one remaining known gap: zero-padding non-720-beam scans (DROW, JRDB) to run LFE cross-dataset is this project's own extrapolation, never described or tested in the paper, and does not correct for FROG's different angular resolution â€” treat DROW/JRDB numbers as an approximate cross-dataset transfer, not a faithful replication. (A second gap â€” LFE-Peaks reporting every raw `find_peaks` hit as its own detection, omitting the paper's centroid-merge step â€” has been fixed: `LFEPeaksDetector` now runs the same greedy NMS-style merge described in the paper, via a shared `_merge_nearby()` helper also used by `LFEPPNDetector`. The merge radius, `_PERSON_RADIUS = 0.4 m`, is this project's own choice â€” the paper describes the merge step but does not state an exact radius.)
>
> **Two more gaps found and fixed while producing Section 8's numbers.** (1) `LFEPPNDetector` hardcoded 31 depth anchors per sector (`_N_ANCHORS = 31`, from the design formula `(FAR-NEAR)/(0.8Â·RADIUS)`), but the bundled published ONNX model's actual output shape is `[..., 30, 3]` â€” 30 anchors, not 31 â€” so every evaluation of LFE-PPN crashed with an index-out-of-bounds error. Fixed by reading the anchor count from the ONNX model's own output shape at load time instead of hardcoding the design estimate. (2) `evaluate.py`'s LFE evaluation path computed recall as `(true positives found) / (true positives found)` â€” i.e. against its own detections, not against the total ground-truth annotation count â€” so a frame where the detector found nothing at all near a real person contributed nothing to the denominator either, and recall was structurally guaranteed to approach 1.0 regardless of how many people were missed outright. This inflated LFE-Peaks' FROG AUC(wp) to 86.2%, ~21 points above the paper's own 64.9% AP â€” implausible for a same-network, same-weights evaluation. Fixed by routing LFE through the same `_prec_rec_2d` PR-curve code every other model uses (recall against total GT, missed detections counted). After the fix, LFE-Peaks measures **74.6%** â€” much closer to the paper's 64.9% (the residual gap is a normal, explainable margin: different eval-frame sample, this repo's own `_merge_nearby` radius choice, etc. â€” not another metric artifact). LFE-PPN was less affected by the recall bug (67.7% either way, coincidentally close to its own 66.5% published figure) but is reported post-fix for consistency.

---

### 3.5 Memory inside a detector, and detector-plus-tracker: what already exists (searched 2026-09-28)

Searched before claiming novelty for TAKHeLiPeD's object memory. The claim that survives the search is narrower than "memory
inside the model", and is stated at the end of this section.

**A recurrent memory inside a 2D-LiDAR person detector already exists: DR-SPAAM's own template.** Its auto-regressive half keeps
backbone features as a template and updates them recurrently as each scan arrives (Section 3.2). That memory is **per beam and
spatial**, not keyed to objects, and spans about 0.15 s. So "in-network temporal state for 2D range data" is prior art; the object
keying and the horizon are not.
It is not the only one: **Deep Tracking** (Ondruska & Posner, AAAI 2016) and **Deep Tracking on the Move** (Dequaire et al., IJRR 2018) carry a learned recurrent memory over raw 2D laser scans for seconds, through occlusion, keyed by grid cell rather than by object (Â§7.4.1).

**Object-level memory fused into detection is prior art in other modalities.**

- **MAD, Agro et al., CVPR 2025 (Waabi).** A transformer fuses detector proposals with proposals from a **memory bank of past
  beliefs**, storing explicit trajectory forecasts so an object's pose can be queried at any timestamp. On Waymo Open and
  Argoverse 2 it adds +2.5 to +7.6 AP over the detectors it wraps. Its motivation is word for word ours: detectors see only a
  0.1-0.3 s window, and memory fills what occlusion and a narrow field of view remove. Differences: 3D automotive LiDAR and
  camera, a forecast-carrying memory bank, and no embedded-cost constraint.
- **Tokmakov et al., ICCV 2021, "Learning to Track with Object Permanence".** CenterTrack plus a spatio-temporal **ConvGRU**
  memory over the whole history, trained end to end for detection and tracking, explicitly to localise fully occluded objects.
  Video RGB, scene-level recurrent state rather than per-object slots.

**Detector plus tracker on 2D LiDAR, including on an embedded platform, also exists, and its components are named.**
"Autonomous Navigation in Dynamic Human Environments with an Embedded 2D LiDAR-based Person Tracker" (arXiv 2412.15000, 2024)
runs **DR-SPAAM, not retrained**, followed by **Norfair** (SORT generalised to several points per detection), developed against
JRDB and evaluated on three own recordings against motion capture: **MOTA 85.45%**, MOTP under 0.2 m, at **20 Hz on a Jetson
Xavier NX**. It is strictly **tracking-by-detection**: the detector runs, the tracker associates, and the paper describes **no
feedback from tracker to detector**. DR-SPAAM's own release has likewise been paired with SORT and trajectory prediction in
deployments.

**That modularity is the gap this project occupies, and it is measurable.** A tracker that reports each detection with the detector's own
score can remove a false positive but never lift a person above the threshold. On FROG, **99.1% of the people TAKHeLiPeD's
memory "loses" at a shared threshold were pushed under it by rescoring while a live slot sat on them**
(`../memory/slot-design-evidence.md`), and rescoring before the threshold beats SORT on *identical* candidates at every setting.
SORT cannot reach those people by construction -- not for want of seeing them (this repository's replay feeds it every candidate down to 0.01), but because it only filters.
**The tracking literature does have a way to lift them:** CBMOT's score refinement raises a detection's score from its track's confidence (Benbarka et al., IROS 2021), and ByteTrack's second association pass is built for low-score detections (Zhang et al., ECCV 2022).
**Re-measured 2026-10-09 on the adopted stage-2 recipe (`TODO.md` A61):** the tuned tracker lifts the candidates from 82.19% to 83.51% and the memory to 85.87%, +2.36 over the tracker at every seed, though at one matched false-positive rate the two now find the same people. **Measured 2026-10-03 (Â§7.4.5), old recipe:** ByteTrack with CBMOT's track confidence, tuned on val, lifts the same candidates from 77.66% to 80.96% AP -- about half of what the memory adds -- and the proposed memory is **+2.90 pp above it**, every seed's block-bootstrap interval above zero.
**JRDB** ships image-plane (2D) and point-cloud (3D) person *tracking* benchmarks beside its detection ones, but none on its 2D laser (Â§7.4.1). The older
laser-based people-tracking literature (Arras, Luber, Linder and the SPENCER work; Leigh et al., ICRA 2015) is substantial.
So a reviewer will not accept "nobody combines a learned 2D-LiDAR detector with tracking" -- they do, modularly, and on embedded
hardware.

**What is left, and it is worth stating precisely.** For knee-height 2D LiDAR person detection, TAKHeLiPeD contributes an
**object-keyed memory inside the detector that acts before the threshold**, at **constant per-frame cost** (4.3-4.6 ms, flat from
64 to 720 slots), exploiting a **measured ~10 s horizon** (`../memory/slot-design-evidence.md` E1) rather than DR-SPAAM's ~0.15 s
template or MAD's forecast bank. Three further findings sharpen it, all from A51 phase 5-6 (2026-09-29/30).

**The learned per-slot state is the mechanism, not decoration.** Removing it and rescoring from the bookkeeping alone -- value,
age, time since match, displacement, hit rate -- costs, at two seeds, 1.40 pp of AP, 1.30 pp at 0.3 m, **1.60 pp of recall at
matched false positives (1.9x its bar)** and **13.80 pp of coasting precision (1.7x its bar)**. Three of the four clear their
three-sigma bars; **AP, at 1.40 against a 1.45 bar, does not**, so the claim is carried by recall and coasting rather than by
AP. It is still the only ablation of this design that has ever cost anything. Before it, "the state is learned rather than hand-coded" was the weakest of the claims separating this work from
tracking-by-detection; it is now the strongest, and the coasting collapse says why -- with no current measurement, only carried
state separates a genuinely occluded person from a stale slot.

**The attention usually assumed necessary for a slot memory is not needed.** Rule-based association plus a linear-time per-slot
recurrence ties the full model on every metric at three seeds, with 43% fewer parameters and 18% less time per frame.

**The feedback path does not work here, and the measurement says why.** Rendering the slots back into stage 2 contributes
+0.0012 / -0.0022 / +0.02 pp across three configurations, including a ground-truth prior with the rescoring head removed so
nothing could compete with it. The cause is capacity, measured rather than argued: the prior carries **0.012-0.045% of that
layer's input magnitude** -- 2 channels against 256, non-zero in 2% of sectors, at the most downsampled layer. The ceiling is
small in any case: only **1.91% of annotated people** are missed with *no* candidate for rescoring to raise while a live slot
holds them, and every one of those is already coasting. **So the paper claims rescoring before the threshold, not feedback**,
and reports the feedback result as a measured negative with its ceiling stated.
The accurate name for the mechanism, from the tracking literature, is a **learned, causal, object-keyed track-before-detect over a detector's sub-threshold candidates** (Â§7.4.1); "acts before the threshold" alone is track-before-detect's defining idea, not a new one.

**What the write-up must therefore do:** cite 2412.15000 and JRDB's tracking benchmark rather than claiming the combination is
untried; position against MAD and Tokmakov as memory-augmented detection in other modalities; and keep the comparison against a
*properly tuned* detector-plus-tracker baseline, which is the SORT sweep this project already ran on its own candidates.
SORT is not the strongest such baseline: the one a tracking reviewer will ask for is ByteTrack, with CBMOT's score refinement, on the same candidates including the sub-threshold ones -- now run, and the comparison to quote (Â§7.4.5).

## 4. Preprocessing

### 4.1 Raw sensor data

Each scan is a 1D array of `N` range measurements:

```
scan = [r_0, r_1, â€¦, r_{N-1}]    r_i âˆˆ [r_min, r_max] metres
```

The beam angles are fixed and sensor-specific:

```
Ï†_i = Ï†_min + i Â· Î”Ï†             (uniformly spaced, e.g. Î”Ï† = 0.5Â° = 0.00873 rad)
```

### 4.2 Coordinate representation: the `(r, x, y)` triplet, and why it was dropped

For each beam `i` in each aligned scan:

```python
x_i = -r_i * sin(Ï†_i)    # lateral  (DROW frame: x = sideways)
y_i =  r_i * cos(Ï†_i)    # forward  (DROW frame: y = forward)
```

giving a 3-channel per-beam input `(r_i, x_i, y_i)`. This was the representation used by this project's own early full-scan designs (and, implicitly, by the cutout approach's post-hoc vote geometry):

- **`r` (polar range)** tells the network how far away a beam endpoint is, which a fixed-kernel CNN cannot otherwise infer â€” a person at 2 m spans ~12 beams, at 8 m only ~3.
- **`(x, y)` (Cartesian)** lets a 1D conv kernel detect a person's roughly constant-diameter body (~0.5â€“0.8 m) as a dense point cluster directly, without learning trigonometry from `r` and beam index.
- **Not `(x, y)` alone**: loses the explicit distance/scale signal.
- **Not first differences `(Î”r, Î”x, Î”y)`**: a 1D CNN with kernel `[-1, +1]` can compute these internally; providing them as input is redundant.
- **Redundancy**: `xÂ² + yÂ² = rÂ²`, so the three channels are not independent â€” this creates a null space in the first conv layer's weights but does not prevent learning.

**Current default (Section 5 architectures): raw range only, no Cartesian channels.** `drow_utils.raw_scan(scans_hist)` returns `(T, N_beams, 1)` â€” just the range value per beam per timestep, no transformation at all. The `(r, x, y)` representation and its supporting function `aligned_scan_xyz()` have been removed; nothing in the currently active architectures calls it anymore. This is deliberate â€” see constraint 2 in Section 2.

### 4.3 Odometry alignment

When stacking `T` consecutive scans, the robot moves between frames. Only the **rotation component** of odometry (Î”Î¸) is corrected for â€” translation is negligible over T=5 frames at typical indoor robot speeds:

1. Compute the accumulated rotation `Î”Î¸_k` from odometry between `tâˆ’k` and `t`.
2. Shift each beam's angle: `Ï†'_i = Ï†_i + Î”Î¸_k`.
3. Re-sample onto the original beam grid via nearest-neighbour: `r'_j = r_i` where `i = round((Ï†'_j âˆ’ Ï†_min) / Î”Ï†)`.

`--align-scans` (default: **on**) applies this rotation correction to the raw-scan representation via `aligned_raw_scan()` in `drow_utils.py`, shifting all T frames simultaneously via vectorised NumPy indexing with linear interpolation. When odom data is absent (FROG without an `_odom.npz` file), the shifts are zero and the output is identical to the unaligned baseline. Pass `--no-align-scans` to reproduce the unaligned variant as an ablation â€” this was the original design (no odometry dependency at inference time, simpler rotating buffer), and whether alignment actually helps accuracy is deliberately left as an open empirical question for Section 8 rather than assumed.

### 4.4 Cutout extraction (DROW / DR-SPAAM / Li2Former only)

The cutout approach replaces the full-scan input with a fixed-width polar window extracted around each beam:

```
cutout[i] = [r_{i-24}, r_{i-23}, â€¦, r_i, â€¦, r_{i+23}]   â† raw range values only (width 48 for DROW)
```

This discards all cross-beam information beyond the window, uses raw `r` values only, and hardcodes the spatial context window. It is kept for compatibility with the cutout-based baselines' training pipeline and inference API; the full-scan architectures in Section 5 do not use it.

---

## 5. The Proposed Architecture: TAKHeLiPeD

**T**(emporal) **A**(daptive) **K**(nee-)**He**(ight) **Li**(dar) **Pe**(rson) **D**(etector), named 2026-09-23. Frozen
2026-10-18, after every architectural alternative this project could test had been tested; what each one cost is in
[`../memory/detector-architectures.md`](../memory/detector-architectures.md) and the measurements behind them in
[`../memory/slot-design-evidence.md`](../memory/slot-design-evidence.md). **Mechanism only here** -- results live in
[`../memory/performance-log.md`](../memory/performance-log.md), history in [`../CHANGELOG.md`](../CHANGELOG.md).

### 5.0 What it is, and why it does not fit the pipeline the rest of this document describes

Every detector in Section 3 is a function of `T` consecutive scans: stack a window, predict, discard the window. The
standard way to add persistence on top is a tracker -- detect per frame, associate across frames, smooth. TAKHeLiPeD does
neither. **It takes one scan per call and carries an explicit object memory as state**, so what persists is not a smoothed
track but a set of slots the network reads and writes, and there is no post-hoc association stage at all.

That makes it the standing exception to this project's own shared-pipeline structure, and it is a deliberate one. The
argument is in Section 7.4 and rests on one measurement: a frozen detector plus a *tuned published tracker* --
ByteTrack with CBMOT confidence, swept on validation -- gains **+1.32 pp** over its own candidates, while the memory gains
**+3.82 pp**, with every seed's bootstrap interval above zero. If a tracker could do this, the memory would not be worth
its parameters.

**Three stages, one of which is not learned:**

| stage | reads | writes | learned? |
| --- | --- | --- | --- |
| 1, features | one raw scan | 10 pose-invariant per-beam channels | no |
| 2, calibration network | those channels | a person logit and a centre vote per beam | yes |
| 3, object memory | the decoded candidates and the robot's pose | rescored candidates, and the slots carried to the next call | yes, except the bookkeeping |

### 5.1 Stage 1 -- per-beam features, and the one thing they deliberately do not do

Ten channels per beam: the range, the range minus a local median over an 11-degree window, and each neighbour's endpoint
in the beam's own frame relative to the beam's endpoint (`beam_features`). All pose-invariant, so a turning robot does not
change them.

**`sanitize_ranges` is the whole of this project's dataset-specific preprocessing, and it is one rule**: anything that
means "nothing within `max_range_m`" -- non-finite, non-positive, or at or past the limit -- becomes `max_range_m` itself.
FROG writes a missing return as `+inf` or 61.0 m, DROW as 29.96 m; both arrive as 10.0 m and no loader needs to know which
dataset it is reading. That was the owner's constraint, not an implementation convenience: **the model should be feedable
with raw data**, with noise handled inside the model rather than in a per-dataset cleaning step.

### 5.2 Stage 2 -- a single-scan ConvNeXt U-Net

A 1-D U-Net over the beam axis, built from ConvNeXt blocks, with widths (32, 64, 128) at `N`, `N/2` and `N/4` beams and a
bottleneck over `N/8` sectors; a 3-tap convolution across bottleneck sectors; a DROW-style head giving each beam a
person logit and a 2-D vote toward the person's centre. 960,323 parameters.
**It reads one scan. It has no state and no pose**, which is what makes the temporal claim in Section 7.4 attributable to
stage 3 rather than shared between them.

| component | borrowed from | how faithfully |
| --- | --- | --- |
| ConvNeXt block: depthwise `k=7`, norm, 1x1 widen by 4, GELU, 1x1 narrow, residual | Liu et al., *A ConvNet for the 2020s*, CVPR 2022 | **Faithful in 1-D.** The block order, the inverted bottleneck and the expansion ratio are as published; the only change is `Conv1d` over beams in place of `Conv2d` over pixels, and `GroupNorm(1, C)` in place of LayerNorm, which is the same normalisation computed over the same axes for a 1-D tensor. |
| Encoder-decoder with skip connections at each scale | Ronneberger, Fischer, Brox, *U-Net*, MICCAI 2015 | **Faithful in structure**, with strided convolutions for downsampling rather than max-pooling and `interpolate` + concatenate on the way up. |
| Per-beam classification plus a 2-D vote to the person's centre | Beyer et al., *DROW*, RA-L 2018 / *Deep Person Detection in 2D Range Data*, IROS 2018 | **The head is DROW's**, and so is the voting formulation -- this is the one piece of the proposed detector that is deliberately not novel, because it is what makes the comparison against DROW3 and DR-SPAAM an architecture comparison rather than a head comparison. What differs is what feeds it: a whole scan rather than a per-beam cutout. |
| ~~Whole-scan global max concatenated to every bottleneck sector~~ -- **removed 2026-10-25** | -- | **Measured and dropped** (`TODO.md` item 40). It was the one component here with no source claimed and the one a cutout detector cannot have at all, so it was the natural candidate for "what does seeing the whole scan buy?" The answer is nothing measurable: **82.24% against the three control seeds' 82.19%**, and the whole detector on its candidates scores 86.20% against 86.40%, inside the 1.32 pp bar, for 16,384 parameters. The full-scan input still matters -- a fixed-beam kernel over a whole scan beats a resampled cutout by 1.42 pp (item 15) -- but *explicit* whole-scan context does not. `--aggregator` still builds it, because older checkpoints load with it. |
| Weight averaging over a validation-chosen window of periodic checkpoints | Izmailov et al., *Averaging Weights Leads to Wider Optima and Better Generalization*, UAI 2018 (SWA) | **Faithful in spirit, simplified in practice.** SWA's cyclical learning-rate schedule is not used; the average is taken over consecutive periodic checkpoints from a single plateau-scheduled trajectory, and the window is chosen on validation. It gained in 15 of 18 A59 runs, +0.66 pp mean. |

**A fixed-beam first layer, on measured grounds.** The obvious objection is that a person subtends 42 beams at 2 m and 9 at
9 m, so a fixed `k=7` kernel is scale-wrong nearly everywhere -- which is exactly what DROW's range-adaptive cutout fixes,
and what Tier 0's 1-NN probe priced at **+2.5 pp**, concentrated at long range. Giving stage 2 that adaptive read was built
and measured (`TODO.md` item 15): it scored **1.42 pp worse**. The probe measured what a *capacity-free* classifier gains
from scale normalisation; a trained network of a million parameters evidently learns it, and paying a resample to hand it
over costs. **The honest statement is that the fixed window is not a simplification we are defending but a result.**

### 5.3 Stage 3 -- an object memory of slots, in room coordinates

256 slots. Each carries a world position and velocity, where it started, its age, the time since it last matched, a recent
match rate, an accumulated **value**, and a learned state vector. Per call: the robot's pose puts the candidates into room
coordinates, rule-based bookkeeping decides which slot each candidate belongs to and which slots live or die, a
minGRU updates each surviving slot's state, and a rescoring head adjusts every candidate's score
using the slot matched to it. 173,442 parameters.

**`dt` reaches the bookkeeping, not the state update.** The recurrence was a `dt`-aware selective cell until
2026-10-23 and is now minGRU, which takes a *step*. Elapsed seconds still drive the rule-based half -- slot
positions, velocities, ages, time since match, and the value fade -- so the design is still rate-aware; what it no
longer claims is a *learned* function of elapsed time. Section 5.3's table says why, and `TODO.md` steps 1-2 (DROW at
12.5 Hz against FROG's 40.01) are where the distinction can finally be measured.

**The bookkeeping is not learned, and that is the design.** Spawning, matching, eviction and retirement run under
`no_grad` as explicit rules, because they are discrete decisions with no gradient to give. Only the slot state, the
recurrence and the rescoring head are trained. A51's S4 tested whether a learned association would do better and found
**96.9-97.0%** of annotated people already had a matching slot at every rule setting -- there was nothing left to recover.

| component | borrowed from | how faithfully |
| --- | --- | --- |
| A fixed set of slots, each a vector, competing to explain the current observations | Locatello et al., *Object-Centric Learning with Slot Attention*, NeurIPS 2020 | **The abstraction, not the mechanism.** Slot Attention assigns observations to slots by iterated competitive soft attention within a frame; here assignment is a hand-coded one-to-one match in metric space across frames, and the slots persist between calls rather than being re-initialised per input. The debt is the idea that a scene is a set of addressable object states, not the attention routine. |
| Per-slot recurrence: **minGRU** | Feng et al., *Were RNNs All We Needed?*, arXiv 2024 | **Faithful.** The minimal GRU: one input-dependent forget gate, no reset gate, no dependence of the gate on the previous state. It is used as published, per slot, with the slot's own statistics concatenated to its input. **Adopted 2026-10-23 over the bespoke alternative described in the row below**, on the grounds that the two tie and a citable cell is more defensible than a re-implemented one. |
| ~~`dt`-aware selective diagonal recurrence, decay learned per slot over 1-60 s~~ -- **not adopted**, kept as `--recurrence selective` | Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv 2023 | **Measured against minGRU and the plain GRU, and it ties** (`TODO.md` item 16: 0.52 pp across five runs against a 1.32 pp bar). What it took from Mamba was the shape -- a diagonal state whose decay is a *function of the input* -- plus a real-time term that is not in Mamba at all, `exp(-dt * rate)` in seconds rather than per step. **The real-time term has never acted on FROG, and that is the honest statement:** `frame_periods` feeds stage 3 a running average of stamped intervals whose 5-95% spread is 24.50-25.53 ms, so the input the mechanism exists to read is a constant to within 1%, and A51's D3's 1.04 pp for fixing the rate priced the *learned decay*, not the `dt`. Item 37 then found the dt-aware rule the **most** fragile of the three under dropped frames (-1.16 pp against minGRU's -0.40). It stays principled and untested: a 12.5 Hz dataset is the test, and `TODO.md` step 2 is it. |
| Slots in **room** coordinates rather than the sensor frame | -- | **No source claimed**, and it is the choice the whole design rests on: a static object stays put in room coordinates and drifts with the robot in sensor coordinates, so "has this thing been here for a minute?" is answerable in one frame and not the other. It needs odometry, which is the cost. |

**What stage 3 does not have, each removed by its own measurement**: no cross-attention association (E7), no slot-to-slot
self-attention (E6, the only O(K^2) term), no coasting head (`TODO.md` item 24 -- it fired 2/0/0 times across 72,533
person-free frames and removing it cost nothing measurable), and no bound on the rescoring correction (O1: a +-4 logit
tanh silences the head rather than stabilising it).

### 5.4 Decoding, and the one parameter that was swept properly

Stage 2's per-beam logits and votes become candidates through **`nms-mean`**: beams in probability order, each kept beam's
vote a detection, every vote within 0.5 m suppressed, and each detection placed at the probability-weighted mean of the
votes it suppressed. That last clause is the difference from the official DR-SPAAM `nms_predicted_center`, which keeps the
best beam's own vote.

The radius and the decoder family were both swept at matched false positives over three seeds (`TODO.md` items 13, 17,
34): 0.4 m buys close pairs and loses far range, 0.6 m does the reverse, a range-split decoder is dominated by both its
parents, and a union of two decoders adds nothing a single threshold can exploit. **Overall recall moves less than the
seed spread across the whole family**, so 0.5 m is reported as the middle of a flat optimum rather than a tuned value.

### 5.5 What a reader should take from this section

The architecture is **one scan in, a set of object states carried out**, and its two defensible novelties are the
whole-scan context in stage 2 and the room-frame slot memory in stage 3. Everything else is either borrowed and cited
above, or was tried and removed with a number attached. The parts that look like simplifications -- a fixed-beam first
layer, hand-coded association, no attention anywhere in the memory -- are each the surviving side of a measured
comparison, and Section 6 is where the alternatives went.

## 6. Design History: Architectures Tried and Dropped

### The three full-scan architectures this project proposed before TAKHeLiPeD

**Superseded, and kept for provenance.** Sections 5.0-5.5 of this document used to propose these three; they are the
project's own first line of work, nothing has been trained on them since 2026-09, and `TODO.md` item 32 keeps
`SpaceTimeCNNDetector` so the decisions made on it stay reproducible while removing the other two.

**One of them shaped everything that followed.** The 2026-09-10 history ablation on `SpaceTimeCNN` found it was a
five-frame *averager*: shuffling its temporal window cost **0.5 pp** while freezing it cost 4.1. A `T`-frame architecture
that does not care about the order of its frames is not a temporal model, and that single result is why this project
rebuilt around an explicit memory instead of a wider window -- and why the proposed detector reads **one** scan. Its
single-frame score of **63.0%** is what A35's acceptance criteria were written against.

The citation audits below were done against each cited paper and several did not survive; they are kept verbatim because
the method is reusable and because a retracted citation is worth more on the record than off it.

#### 6.1 Design choices that apply to all three

**Raw scan representation.** See Section 4.2 â€” `(T, N_beams, 1)`, no Cartesian channels.

**Rotating buffer for the `T`-frame window.** Because there's no realignment step required for correctness, maintaining the `T`-scan window at inference time reduces to a fixed-size circular buffer: evict the oldest scan, append the newest, O(1) per measurement. This is already implemented â€” `LiveDataset.push_measure()` (`library/takheliped/datasets/live_dataset.py`) uses a `collections.deque(maxlen=time_frame)`, which evicts the oldest entry automatically on `append()`.

**What got dropped, and why.** This project previously had four full-scan detector classes; two were removed for violating the non-recursive constraint (Section 2, constraint 1) â€” see the two removed classes below for what they were and why they were tried. What's left, `SpaceTimeCNNDetector` and `FullScanTCNDetector`, were already non-recursive before that constraint was stated explicitly; `TemporalUNetDetector` (Architecture C) was added afterward as a controlled ablation against LFE (Section 6.4).

#### 6.2 Architecture A â€” Joint Space-Time CNN (`SpaceTimeCNNDetector`)

**What it does:** reshapes the input to `(1, 1, N_beams, T)` and treats it like a 1-channel image, where the height axis is the beam index and the width axis is time. A stem + two dilated "joint" 2-D conv blocks mix beam and time together from the very first layer; multi-scale Inception-style blocks (parallel branches at kernel sizes 3/5/9, increasing dilation) then specialize along the beam axis (for scale-invariance â€” near people are wide in beam-space, far people are narrow) and the time axis (for motion).

**Explainability, verified against each cited paper (Section 2.1.1 has the audit methodology) â€” not every claimed inspiration held up equally well:**

| Block | Borrowed from | Why |
|---|---|---|
| Residual skip connections everywhere | He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 (ResNet) | **Verified accurate.** Every block does `out + skip(x)`, with a 1Ã—1 projection shortcut when channels change â€” a textbook ResNet shortcut, faithfully implemented (`_JointConvBlock`, `_MultiScaleBlock`, `DilatedScanBackbone`). |
| Squeeze-and-Excitation channel gating | Hu, Shen, Sun, *Squeeze-and-Excitation Networks*, CVPR 2018 | **Verified accurate.** `_SEBlock`'s global-avg-pool â†’ 2-layer bottleneck MLP â†’ sigmoid â†’ channel-wise multiply matches Hu et al. almost line-for-line. |
| Multi-branch, multi-kernel-size blocks merged via 1Ã—1 conv | Szegedy et al., *Going Deeper with Convolutions*, CVPR 2015 (Inception/GoogLeNet) | **Loose inspiration, not a faithful port.** `_MultiScaleBlock` does run parallel kernel-size branches concatenated and 1Ã—1-projected â€” the core motif is real â€” but it diverges from actual Inception: branches use anisotropic `(k,1)`/`(1,k)` kernels along a single axis rather than square 2D multi-scale kernels, there's no pre-branch 1Ã—1 bottleneck (Inception's actual compute-saving trick), and SE + residual are bolted on, which GoogLeNet's Inception module never had. A person's beam-width varies 4Ã— between near and far range (Section 1), which is the real motivation for the multi-kernel idea, independent of how closely it tracks GoogLeNet specifically. |
| Joint (not factorized) space-time convolution in the early layers, cited as video-CNN literature (e.g. C3D) | ~~Tran et al., *Learning Spatiotemporal Features with 3D Convolutions*, ICCV 2015 (C3D)~~ | **Inaccurate citation â€” removed.** C3D's defining feature is a *true 3D convolution* over two spatial axes plus time. This architecture has only one spatial axis (beam index); the "joint" stem/blocks are ordinary `Conv2d` over a beamÃ—time grid, not a volumetric 3D kernel. Citing C3D overstates a kinship to volumetric video CNNs this design doesn't have. No single published architecture was found that matches this beamÃ—time-grid treatment closely enough to cite directly (it doesn't match range-image LiDAR CNNs like SqueezeSeg/RangeNet++ either, which use azimuthÃ—elevation, not beamÃ—time) â€” **the honest framing is that this is an original design combining the ResNet/SE/Inception-style ideas above, not a port or close variant of any single published architecture.** |

#### 6.3 Architecture B â€” Causal TCN + Spatial CNN (`FullScanTCNDetector`)

**What it does:** a small causal temporal-convolutional stack runs over each beam's own `T`-frame history *first* (collapsing time to a single per-beam summary), and only then does the spatial `DilatedScanBackbone` mix across beams â€” so spatial mixing has access to motion-aware features from the start, which Architecture A's joint-from-layer-1 design has to learn implicitly instead.

| Block | Borrowed from | Why |
|---|---|---|
| Causal dilated 1-D convolution, "chomp" padding | Bai, Kolter, Koltun, *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, arXiv 2018 (coined "TCN" in this sense); causal dilated convs originate in van den Oord et al., *WaveNet*, arXiv 2016 | A feed-forward, fully parallel alternative to recurrence (GRU/LSTM) with the same left-to-right dependency structure â€” directly satisfies constraint 1 while keeping order-sensitivity (unlike plain mean-pooling, which is permutation-invariant in time). **Verified against the paper's official reference implementation (`locuslab/TCN`): this is a *simplified* TCN block, not a faithful port** â€” see the deviation list below. |
| Dilated 1-D conv stack, dilation 1â†’2â†’4â†’8, residual skip (`DilatedScanBackbone`) | He et al. (ResNet); dilated convs from Yu & Koltun, *Multi-Scale Context Aggregation by Dilated Convolutions*, ICLR 2016 | Receptive field grows exponentially (3â†’7â†’15â†’31 beams) with only linear parameter growth â€” covers a standing person (~12 beams at 2 m) without a hand-picked fixed window. **Verified**: the dilation schedule matches Yu & Koltun, and there is a real skip connection â€” but it is *one* residual wrapping the entire 4-layer stack end-to-end, not He et al.'s pattern of a shortcut around every ~2-conv block. Should be described as "a single global residual wrapper," not "ResNet-style residual learning." |

**Deviations from the canonical TCN block (`locuslab/TCN`'s `TemporalBlock`), confirmed by direct comparison:** this repo's `_TemporalTCNBlock` uses **one** dilated causal conv per block where the reference uses **two** (compensated by stacking two single-conv blocks at dilations 1, 2 â€” 2 total conv layers here vs. 4 in two canonical blocks at the same dilations); **GroupNorm instead of weight normalization** (not documented as a substitution anywhere in the code); only **two dilation levels** (1, 2) rather than the reference's arbitrary-depth schedule (receptive field 7 is just enough to cover T=5, but this is a materially shallower network than "a TCN" in the general sense); and **no final activation after the residual sum** (the reference applies `ReLU(out + res)`; this repo returns `out + skip(x)` directly). The residual connection itself and the causal chomp padding size are both correct. These are reasonable efficiency simplifications for a fixed T=5, B=1 inference setting â€” but the architecture should be described as **"a simplified causal TCN block, following Bai et al.'s general design but not a faithful port of their reference implementation,"** not as "a TCN" without qualification.

**Architecture A vs. B is a deliberate ablation, not two unrelated designs:** "should space and time be fused jointly from layer 1, or should time be collapsed first and space handled afterward?" is analogous to the question studied for video CNNs by Xie et al., *Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification*, ECCV 2018 (joint 3-D convolution vs. factorized 2-D-spatial + 1-D-temporal designs). Architecture A is the "joint" point in that design space; Architecture B inverts the factorized order further by doing time *before* space rather than after.

#### 6.4 Architecture C â€” LFE-with-Time U-Net (`TemporalUNetDetector`)

**What it does:** a standard 1-D U-Net backbone (Ronneberger et al., MICCAI 2015) with the single-channel input replaced by T-channel input (one channel per historical scan). Operates over the beam axis â€” each encoder stage halves N_beams by MaxPool1d(2), the bottleneck sees the full scan compressed 8Ã—, and the decoder upsamples back with skip connections from each encoder stage restoring spatial precision. Default head is heatmap (matching LFE-Peaks).

**Verified against the canonical U-Net definition â€” correct, no bugs.** All six defining properties were checked directly against the code: symmetric 3-stage encoder/decoder depth, a genuine two-conv "double conv" block at every stage, MaxPool1d downsampling (not strided conv), skip connections via channel-wise **concatenation** (not addition â€” confirmed no accidental ResNet-style shortcut bug), a bottleneck stage, and `F.interpolate(size=skip.shape[-1])` upsampling that exactly matches each skip's spatial size regardless of non-power-of-2 beam counts (450, 720). This is a faithful, correctly-implemented U-Net.

**Correction â€” this is *not* actually "LFE's backbone + time."** The original framing below ("Architecture C adds only temporal context to LFE's backbone and head, isolating temporal contribution") was checked directly against the FROG paper's description of LFE's real architecture (Â§4.1) and does not hold up: LFE's own backbone is **not** a textbook U-Net. It's three residual blocks, each containing three depthwise-separable convs at kernel sizes [9, 7, 5], each with an *internal* residual addition and a "global aggregator" (global max-pool concatenated at every position) â€” none of which `TemporalUNetDetector` has. LFE also keeps channel width roughly constant at 32 (bumping to 64 only at the deepest level) and downsamples by 2Ã— and 3Ã— (chosen so that 450-point scans divide evenly), whereas this repo's `TemporalUNetDetector` doubles channels every stage (Câ†’2Câ†’4Câ†’8C) and downsamples by three successive Ã—2 pools. LFE uses BatchNorm; this repo uses GroupNorm.

So `TemporalUNetDetector` is honestly **a fresh, textbook U-Net given a temporal input** â€” not LFE's own backbone with time added. That's a legitimate, defensible architecture in its own right (see the citation table below), but the ablation framing needs correcting: Architecture C isolates "**temporal context + a from-scratch U-Net backbone**" against LFE, not "temporal context only." The row below is corrected accordingly.

| Model | Backbone | T | Head | What changes vs previous row |
|---|---|---|---|---|
| LFE-Peaks | LFE's own backbone (3 residual blocks, depthwise-separable multi-kernel convs, constant ~32ch, global aggregator) | 1 | heatmap | â€” |
| **Architecture C** (TemporalUNet + heatmap) | **Canonical U-Net** (symmetric encoder-decoder, doubling channels, MaxPool/concat-skip â€” *not* LFE's own backbone topology, see correction above) | **5** | **heatmap** | **+temporal context, and a different (though comparably-scoped) backbone** |
| Architecture A + heatmap (SpaceTimeCNN) | dilated 2D CNN | 5 | heatmap | backbone only |
| Architecture A + drow (SpaceTimeCNN) | dilated 2D CNN | 5 | DROW votes | head only |
| Architecture B + drow (FullScanTCN) | TCN + dilated 1D CNN | 5 | DROW votes | fusion order |

| Block | Borrowed from | Why |
|---|---|---|
| Encoder-decoder with skip connections | Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015 | Skip connections restore fine beam-level precision lost by pooling: the bottleneck encodes "is there a cluster of short-range readings here?" globally; the skip at E1 restores exactly which beam carries that reading. |
| T input channels | Same strategy as Architecture A's stem `Conv2d(1, C, 3Ã—3)` treating T as the width axis | The simplest way to extend a single-frame U-Net stem to multi-frame input â€” not literally LFE's own stem (LFE's backbone is not this U-Net topology, see the correction above), but the same general idea of putting T in the channel dimension of the first layer. |
| GroupNorm instead of BatchNorm | Wu & He, ECCV 2018 | LFE's Keras model uses BatchNorm, which degrades at B=1 inference; GroupNorm avoids that failure mode. Framed here purely as a normalization choice â€” it is not, by itself, what makes this backbone differ from LFE's (see the backbone-topology correction above; GroupNorm is one of several differences, not the only one). |
| BEAM_BATCH=True (vs BEAM_BATCH=False for A and B) | DrSpaamDetector (same flag) | The U-Net pools over the beam axis: flattening the batch to (BÃ—N, T, 1) would let MaxPool1d pool across scan boundaries and corrupt the encoder-decoder alignment. |
| MaxPool1d(2) + F.interpolate(size=skip.shape[-1]) | Standard U-Net | 720 beams (FROG) and 450 beams (DROW) are not powers of 2; matching the skip's exact size avoids padding artifacts. |

**What the U-Net skip connections do and do not do.** A common misreading: skip connections do NOT directly connect beam 0 to beam 719. They connect each *encoder layer* to its *mirror decoder layer at the same spatial resolution*. What connects distant beams indirectly is the bottleneck, where the N//8-beam compressed representation has a large effective receptive field through the Conv1d(k=3) layers.

#### 6.5 Cross-cutting hyperparameters

| Hyperparameter | Value | Why |
|---|---|---|
| Time window `T` | 5 frames | At a typical robot update rate of â‰ˆ 10 Hz, T=5 covers 500 ms â€” long enough to observe a single walking half-cycle (â‰ˆ 400â€“600 ms per step, Winter 1990). |
| Dropout | 0.1 | Light regularisation; output heads are regularised by the data volume (FROG's train split is 108,356 annotated frames â€” every scan is labeled, Section 3.1 â€” measured directly during Section 8's evaluation run, correcting an earlier â‰ˆ20k estimate here). |
| GroupNorm | largest power-of-2 group count | Wu & He, *Group Normalization*, ECCV 2018 â€” batch-size-independent, unlike BatchNorm at inference `B=1`. |

<details>
<summary>Per-architecture design-choice tables (channel widths, kernel sizes, receptive fields)</summary>

#### Architecture A â€” SpaceTimeCNNDetector

| Design choice | Value(s) | Origin & reasoning |
|---|---|---|
| Base channels `C` | 96 | GoogLeNet/Inception's second-stage channel count. For 3-branch parallel blocks of C/3 = 32 ch each, 96 prevents channel collapse; fits within a ~2â€“3M parameter budget on FROG-scale scans. |
| Stem | 1 Ã— `Conv2d(1, C, 3Ã—3)` | Minimal entry point, lifting 1 input channel to C. |
| Joint blocks | 2 Ã— `Conv2d(C, C, 3Ã—3)` at dilations (1,1) and (2,2) | Borrowed from DeepLab v3 / ASPP (Chen et al., ICLR 2018) â€” two dilation levels build a modest receptive field on both axes before the factorized multi-scale stages. |
| Spatial stages | default 3 (`n_spatial_stages`), dilations **1, 4, 16** (optional 4th: 64) | Powers-of-4 (WaveNet convention) cover more angular extent with fewer stages: 3 stages + k=9 branches give a 177-beam receptive field (â‰ˆ Â±44Â° at 0.5Â°/beam). |
| Spatial kernel sizes | (3, 5, 9) | At 0.5Â°/beam: a person at 2 m subtends â‰ˆ 28 beams; at 8 m â‰ˆ 7 beams â€” parallel branches let the network learn which size to weight (Inception motivation). |
| Temporal stages | 2 Ã— `_MultiScaleBlock` on the time axis, kernel sizes (3, 5, 7) | Stage 1 (k=3): velocity-like patterns; Stage 2 (k=5, applied to Stage 1's output): acceleration-like patterns, reaching the full T=5 window. |
| SE reduction ratio | 8 | Default from Hu et al. (CVPR 2018); 12-channel bottleneck for C=96. |
| Mean collapse over T | â€” | Motion is already encoded in the channel dimension after the temporal stages; mean-pooling discards the raw T axis without losing information. |

#### Architecture B â€” FullScanTCNDetector

| Design choice | Value(s) | Origin & reasoning |
|---|---|---|
| TCN channel width | 64 | Smaller than Architecture A's 96 â€” the TCN stage only summarizes T, it isn't the main spatial-reasoning stage. Matching `tcn_channels = backbone_channels = 64` avoids an extra projection at the handoff. |
| Number of TCN layers | 2, dilations **1, 2** | With kernel_size=3: total receptive field = 1 + 2Ã—1 + 2Ã—2 = 7 > T=5. Two causal layers are the minimum to cover the full 5-frame window. |
| Causal "chomp" padding | pad left by `(kâˆ’1)Ã—dilation`; drop trailing `2Ã—dilation` positions | Guarantees output position `t` depends only on input positions `â‰¤ t` â€” same left-to-right dependency as a GRU, via masked convolution instead of recurrence. |
| Last-timestep selection `[:, :, -1]` | â€” | The causal TCN output at `t=Tâˆ’1` has seen all `T` prior steps; equivalent to a GRU's final hidden state, without the permutation-invariance of mean-pooling. |
| DilatedScanBackbone | 4 Ã— `Conv1d(C, C, 3)` at dilations **1, 2, 4, 8**; 1Ã—1 end-to-end residual skip | Receptive field = 1 + 2Ã—(1+2+4+8) = 31 beams â‰ˆ Â±7.5Â° â‰ˆ 0.8 m at 3 m range â€” spans a standing person's shoulder width (â‰ˆ 0.5 m) with margin. |

#### Architecture C â€” TemporalUNetDetector

| Design choice | Value(s) | Origin & reasoning |
|---|---|---|
| Base channels | 32 (`unet_channels`) | Four stages: 32â†’64â†’128â†’256 channels, ~650k total parameters â€” lighter than SpaceTimeCNN (â‰ˆ1-2M). |
| Number of encoder stages | 3 | Three MaxPool1d(2) stages compress 720 beams to 90 in the bottleneck. |
| Bottleneck channels | 8C = 256 | Standard U-Net doubling per stage. |
| Decoder upsample | `F.interpolate(size=skip.shape[-1], mode='linear')` | `ConvTranspose1d` introduces checkerboard artifacts at non-power-of-2 input sizes; linear interpolation exactly matches the encoder's spatial size. |
| Default head | heatmap | Matches LFE-Peaks, making Architecture C the cleanest extension of LFE; `--head drow` available for ablation. |

</details>

#### 6.6 Alternative detection head â€” Gaussian heatmap (`--head heatmap`)

Both Architecture A and B supported a heatmap head as a drop-in replacement for the default DROW-style head; Architecture C defaulted to it. TAKHeLiPeD's head is DROW's (Section 5.2) and has no heatmap variant. Everything else (backbone, training loop, evaluation) is unchanged. This is an **ablation**, not the primary method: the DROW head is kept as default elsewhere because it keeps the comparison with DROW/DR-SPAAM/Li2Former head-neutral.

| Design choice | Value | Origin & reasoning |
|---|---|---|
| Target type | 1D Gaussian heatmap over N_beams; one Gaussian per annotated person, summed and clipped to [0, 1] | CenterNet (Zhou et al., *Objects as Points*, CVPR 2019) object-center heatmap targets, applied in 1D. |
| Gaussian width Ïƒ | **2 beams** (`--heatmap-sigma`, default 2.0) | FWHM â‰ˆ 4.7 beams â‰ˆ 2.35Â° at 0.5Â°/beam. Ïƒ=1 is too sparse; Ïƒâ‰¥4 lets adjacent-person Gaussians overlap at typical â‰¤1 m inter-person spacing. |
| Loss | Weighted BCE (`pos_weight = n_neg / n_pos`, clamped to 200) against the soft Gaussian target | Compensates the â‰ˆ50:1 background-to-foreground ratio; the soft target spreads gradient over â‰ˆ2Ïƒ beams rather than only the exact peak. |
| Output head | `nn.Linear(C, 1)` | Strictly simpler than the DROW head's two branches â€” if it achieves comparable AUC with fewer parameters, that is evidence vote regression was the weak link, not the backbone. |
| Decoder | `scipy.signal.find_peaks(sigmoid(output), height=0.3, distance=5)` | `height=0.3` suppresses background noise; `distance=5` beams (â‰ˆ2.5Â°, â‰ˆ0.17 m at 2 m) separates peaks below typical inter-person spacing but above the Gaussian FWHM. |
| Why it may outperform the DROW head for full-scan architectures | â€” | The DROW vote-regression head was designed for cutouts, where a beam at the cluster edge must predict an offset without seeing the cluster center. Full-scan backbones (31â€“177 beam receptive fields) *do* see the cluster center, so a heatmap peak is a more direct target. |
| Evaluation compatibility | â€” | The decoder outputs (x, y, confidence) triplets placed in column 3 (person class) of the same `(N_dets, 4)` array `_process_detections`/`_prec_rec_2d` expect â€” no pipeline changes needed. |

---


This project previously had four full-scan detector classes; two were removed for violating the "non-recursive only" constraint (Section 2). Kept here as a record of what was tried and why â€” not part of the current comparison.

### FullScanCNNDetector â€” removed

Used a GRU for temporal aggregation after a dilated 1D CNN backbone processed each timestep independently:

```
For each timestep t independently:
  Conv1d(3,    64,  kernel=3, dilation=1 ) â†’ receptive field  3 beams
  Conv1d(64,  128,  kernel=3, dilation=2 ) â†’ receptive field  7 beams
  Conv1d(128, 256,  kernel=3, dilation=4 ) â†’ receptive field 15 beams
  Conv1d(256, 256,  kernel=3, dilation=8 ) â†’ receptive field 31 beams
  Conv1d(256, 256,  kernel=3, dilation=16) â†’ receptive field 63 beams
  â†’ (N_beams, 256) per timestep

Temporal: GRU over T, per beam â†’ (N_beams, 256)
Detection heads: Conv1d(256, 4) + Conv1d(256, 2)
```

Dropped because the GRU violates the non-recursive constraint. The class was deleted from `full_scan.py`.

### FullScanTransformerDetector â€” removed

Used global multi-head self-attention over all N_beams simultaneously, after a dilated CNN backbone:

```
Full-scan dilated CNN backbone  â†’ (N_beams, T, C)
BeamSelfAttention               â†’ each beam attends to all N_beams (at each T)
Temporal mean-pool over T       â†’ (N_beams, C)
Detection heads
```

The most expressive beam-level architecture in the family (content-adaptive, global beam-to-beam attention via a `N_beams Ã— N_beams = 202,500`-entry attention matrix per layer), but self-attention isn't a CNN/TCN primitive, so it violates constraint 1 â€” and in the one same-footing benchmark run available, it was both the slowest (~1.5 s/scan on CPU) and lowest-scoring of the four full-scan models. Class deleted, along with the now-unused `BeamSelfAttention` helper. An earlier version used a GRU for temporal aggregation here too; it was replaced with mean-pooling since at T=5 the recurrence added negligible benefit while blocking DirectML â€” but mean-pooling is permutation-invariant in time, so this architecture structurally cannot use motion direction (approaching vs. receding) at all, unlike FullScanTCNDetector (Section 6.3).

### Architecture comparison (including removed architectures, for provenance)

| Architecture | Beam communication | Temporal | Fusion order | Input representation | Status |
|---|---|---|---|---|---|
| DrowDetector | None (per-beam cutout) | Fixed sum | space-then-time | Polar range only | active |
| DrSpaamDetector | Local (Â±5 beams, auto-regressive) | Learned blending | joint (auto-regressive) | Polar range only | active |
| ~~FullScanCNNDetector~~ | Local â†’ growing (dilated, up to Â±15 beams) | GRU | space-then-time | (r, x, y) full scan | **removed** |
| SpaceTimeCNNDetector | Local (spatial + temporal jointly) | Implicit (2D conv) | joint | raw range only, full scan, unaligned by default* | active |
| ~~FullScanTransformerDetector~~ | Global (all N beams) | Mean-pool | space-then-time | (r, x, y) full scan | **removed** |
| FullScanTCNDetector | Local â†’ growing (dilated, up to Â±15 beams) | Causal TCN | **time-then-space** | raw range only, full scan, unaligned by default* | active |
| TemporalUNetDetector | Global (bottleneck) + local (skips) | T input channels | joint | raw range only, full scan, unaligned by default* | active |
| Li2FormerDetector | None (per-beam cutout) | Transformer (T steps) | time-then-space | Polar range only | active |
| LFEPeaksDetector / LFEPPNDetector | Global (U-Net, full scan) | None (single scan) | n/a | Normalised range only | active |

\* `--align-scans` (default on) applies odometry rotation correction; see Section 4.3.

All active detectors are accessible via `takheliped.detectors.DETECTOR_REGISTRY`; the trainable ones (no published weights) via:

```bash
python train.py --detector drow|drspaam|spacetime_cnn|fullscan_tcn|temporal_unet|li2former
```

---

## 7. Related Work

### 7.1 Cutout-based detectors â€” see Section 3 for full technical detail

- **DROW** â€” Beyer, Hermans, Leibe, arXiv:1603.02636 / arXiv:1804.02463.
- **DR-SPAAM** â€” Jia, Hermans, Leibe, IROS 2020, **arXiv:2004.14079** (this document previously cited `2004.14064`, an unrelated paper â€” a one-digit transcription error, now fixed).
- **Li2Former** â€” Yang et al., IEEE TIM 2024, DOI: 10.1109/TIM.2024.3420353.

All three share the cutout: a fixed-size polar window per beam, processed by a per-beam CNN with no (DROW) or only local (DR-SPAAM, Â±5 beams) cross-beam communication during feature extraction, and no (DROW), auto-regressive (DR-SPAAM), or temporal-attention (Li2Former) handling of the time axis.

### 7.2 Full-scan / no-cutout detectors â€” is this novel?

**Short answer: the full-scan idea itself is not unprecedented, but the specific combination this research proposes â€” full-scan + raw/unaligned-by-default input + multi-frame temporal context + beam-classification-with-vote-regression, using only non-recursive primitives â€” appears to be.** Three prior works are close enough to discuss directly:

1. **LFE-Peaks / LFE-PPN** â€” Amodeo et al., Frontiers in Robotics and AI, 2025, arXiv:2306.08531. **This is the closest prior work.** It applies a 1-D U-Net FCN directly to the raw, full-scan range vector (no cutout) â€” the same "give the network the whole raw scan" idea used here, including the non-Cartesian, single-channel input (Section 3.4). The key difference: LFE is **single-scan only** â€” no temporal window at all. This research's contribution relative to LFE is adding the `T`-frame time axis as a first-class input dimension via three different non-recursive fusion strategies (Section 5).

   **Relationship to "extending LFE."** Both Architecture A and B can be read as principled extensions of the LFE spatial idea:

   - *Replace the encoderâ€“decoder with dilated convolutions.* LFE's U-Net compresses the beam axis then reconstructs it via upsampling + skip connections. Architectures A and B use dilated convolutions with `same` padding throughout â€” the beam axis is never downsampled, so no decoder is needed. This is the same substitution DeepLab (Chen et al. 2015) made in image segmentation: dilated ("atrous") convolutions expand the receptive field without sacrificing spatial resolution.
   - *Add the temporal axis, via three fusion strategies.* Architecture A fuses space and time jointly from layer 1; Architecture B summarizes time first via a causal TCN, then mixes space; Architecture C gives LFE's general "full-scan, U-Net-family backbone + heatmap head" idea a temporal input, though â€” as corrected in Section 6.4 â€” via a from-scratch canonical U-Net rather than a literal port of LFE's own multi-kernel residual backbone, so it isolates "temporal context + backbone redesign" rather than temporal context alone. Neither A nor B's question was available in LFE since it had no time axis at all.

   The detection head is also different by default: LFE finds persons via local peak detection (Peaks variant) or a proposal network (PPN variant); Architectures A and B default to DROW-style per-beam 4-class logits plus 2D vote-offset regression â€” the same head used by every cutout baseline, making accuracy numbers directly comparable without a bridging step. The heatmap head (Section 6.6) is the point of contact with LFE's own approach.

2. **PeTra** â€” Guerrero-Higueras et al., Frontiers in Neurorobotics, 2019, DOI: 10.3389/fnbot.2018.00085. Projects raw 2-D LIDAR returns into a 256Ã—256 binary occupancy grid and segments leg positions with a full U-Net â€” also full-scan, also no cutout. Differs in task framing (binary occupancy-grid segmentation + a separate tracker, vs. per-beam multi-class detection + vote regression directly on the polar scan) and is markedly slower (â‰ˆ300 ms/scan reported vs. single-digit-to-low-double-digit ms for the cutout baselines).
   > "the occupancy map is defined as a 256 Ã— 256 matrix, with a resolution of about 2 cm." â€” Section on input representation.

3. **TCN + 2-D LIDAR precedent (different task)** â€” Luo, Poslad, Bodanese, IEEE IoT Journal, 2020, DOI: 10.1109/JIOT.2020.2984544. Applies a TCN to **already-extracted trajectories** to classify 15 activity types â€” TCN-after-detection, not TCN-for-detection. Evidence that TCNs are a known, working tool in this sensor domain, but doesn't address detection itself the way Architecture B does. Cited as precedent, not as a competing baseline.

No paper found in this search combines full-scan (no-cutout), raw input, a multi-frame temporal window, the beam-classification-plus-vote-regression detection head used by DROW/DR-SPAAM/this project, **and** an exclusively non-recursive architecture. That combination â€” not any one piece alone â€” is the actual novelty claim to make.

### 7.3 Adjacent 3-D LiDAR literature (context, not a fair baseline)

Full-scan, no-cutout CNNs are the *standard*, not the exception, in 3-D LiDAR object detection â€” range-image-based detectors (e.g. Sun et al., *RSN: Range Sparse Net*, CVPR 2021; *Fully Convolutional One-Stage 3D Object Detection on LiDAR Range Images*, 2022) run ordinary 2-D convolutions over the whole range image. Useful context for why "full-scan CNN" is a reasonable thing to try, but these papers are 3-D, multi-object, autonomous-driving-scale, and not a fair comparison row for this project's 2-D, person-only, robot-scale setting.

### 7.4 Person tracking: what this project must cite, and what it suggests (surveyed 2026-09-28)

TAKHeLiPeD's stage 3 associates candidates with slots, keeps an existence score per slot, coasts through misses and retires a slot when its score fades.
That is a tracker in structure, even though it is scored as a detector, so the literature it will be judged against is multi-object tracking (MOT) as much as person detection.
Until this survey the repository cited only SORT, the MOTR family, CenterTrack, MAD, Tokmakov et al. 2021 and the classical laser trackers of Â§2.2.
Venues below were checked against abstracts, proceedings pages or code, not re-read in full â€” the standard of `PROPOSAL.md` Â§13.

#### 7.4.1 Six claims the tracking literature qualifies

| claim, as this repository states it | what the literature says | consequence |
| --- | --- | --- |
| "A post-hoc tracker only ever sees detections that survived the threshold" (Â§3.5, `../TODO.md` A51, `../memory/dataset-properties.md`) | Not even true of this repository's SORT baseline, which is fed every candidate down to 0.01 (`replay_sort`, `tracker_sweep.py`) -- SORT sees them but only filters, reporting each with its own score. **CBMOT** (below) raises a detection's score from its track's confidence, which is the capability the memory is credited with. **ByteTrack** (Zhang et al., *ECCV* 2022, arXiv 2110.06864) associates *every* box, matching low-score detections to still-unmatched tracks in a second pass, precisely to recover occluded people. **OC-SORT** (Cao et al., *CVPR* 2023) re-associates lost tracks from their last real observation | SORT is not the strongest post-hoc baseline. The one that tests "memory inside the detector" is ByteTrack with CBMOT's score refinement on the *same* 64 candidates, sub-threshold ones included (E14), plus CBMOT's rule on the memory's own bookkeeping (E15) |
| The slot value rule is this project's design (`PROPOSAL.md` Â§5.5) | `v <- v + gain * (1 - v) * s` is **CBMOT**'s "multiplication" score update `c = 1 - (1 - c)(1 - s)` generalised by a gain (Benbarka, SchrÃ¶der & Zell, *IROS* 2021, arXiv 2107.04327); CBMOT also decays unmatched tracklets (by a constant per step, where TAKHeLiPeD decays exponentially in seconds) and deletes them on score rather than on a miss count. Track existence as a recursively updated probability goes back to **IPDA** (MuÅ¡icki, Evans & StankoviÄ‡, *IEEE Trans. Automatic Control* 1994) | Cite CBMOT and IPDA for the value rule, and keep "ours" only for value-ranked *replacement*, which never fires on FROG. D2 (latest score instead, +0.60 pp) is CBMOT's count-based-against-confidence-based comparison, where confidence-based won by up to +1.83 AMOTA on nuScenes â€” a contrast worth one sentence in the write-up |
| In-network temporal state for 2D range data exists only as DR-SPAAM's template (Â§3.5, before this survey) | **Deep Tracking** (Ondruska & Posner, *AAAI* 2016) and **Deep Tracking on the Move** (Dequaire et al., *IJRR* 2018, from a moving platform) run a recurrent network over raw 2D laser scans and learn to keep occluded objects in memory. `PROPOSAL.md` Â§12 already lists them as an alternative, but Â§3.5's novelty argument did not | The surviving claim narrows again: **object-keyed**, and scored as person detection on a public benchmark. Their memory is an occupancy grid keyed by place, the addressing `../memory/rejected-ideas.md` shows failing when it is hand-built |
| Memory acting *before* the threshold is what separates TAKHeLiPeD from a tracker (Â§3.5, `PAPER.md`) | This is the defining idea of **track-before-detect** (TBD): integrate unthresholded evidence over time and declare a weak target only once its track is convincing (particle-filter TBD, Salmond & Birch, *American Control Conference* 2001; a learned TBD, Fishel et al., *ICASSP* 2024). In video detection, **Seq-NMS** (Han et al., arXiv 1602.08465, 2016) rescores weak detections from high-scoring linked ones | Name stage 3 a **learned, causal, object-keyed track-before-detect over a detector's sub-threshold candidates**. That is accurate, citable and still distinctive; "before the threshold" alone is not |
| JRDB "ships 2D and 3D person tracking benchmarks" (Â§3.5 and `../TODO.md` A51, before this survey) | JRDB's 2D tracking benchmark is **image-plane** boxes in panoramic video; its 3D benchmark is **point clouds**. It has no 2D-laser benchmark ([jrdb.erc.monash.edu/benchmark](https://jrdb.erc.monash.edu/benchmark/)); tracking is ranked by MOTA and also scored with OSPA, CLEAR-MOT, HOTA and identity metrics | A 2D-laser tracking result on JRDB means projecting the 3D tracks and their identities to the ground plane on train/val, outside the leaderboard; say so wherever it is quoted (A49) |
| AP conceals a constant absolute false-positive rate, so a metric that charges for it is needed (`PAPER.md`) | **GOSPA** (Rahmathullah, GarcÃ­a-FernÃ¡ndez & Svensson, *FUSION* 2017, arXiv 1601.05585) is an established metric on finite sets of *point* targets that decomposes into localisation error, missed targets and false targets, without normalising by cardinality â€” the per-frame FP/FN accounting `PAPER.md` builds by hand. **OSPA** and **OSPA(2)** (Nguyen, Rezatofighi et al., *TPAMI* 2023) are JRDB's own | Cite GOSPA as the metric the argument calls for, and report it beside AP (E16); it needs no identities, so it works on FROG |

#### 7.4.2 Laser people tracking: the lineage to cite

| work | what it does | where it applies here |
| --- | --- | --- |
| Arras, Grzonka, Luber & Burgard, *ICRA* 2008 | multi-hypothesis leg tracker: legs tracked separately, people as leg pairs, **occlusion probabilities adapted to the situation** | A51 O2: an occluded person should coast and a vanished phantom should not (E19) |
| Luber, Tipaldi & Arras, *IJRR* 2011, "Place-dependent people tracking" | learns per place where people appear and disappear, as priors for the tracker | the working form of place-keyed memory: it pays when a platform revisits places for long, which is exactly `../memory/rejected-ideas.md`'s reopen condition for role C |
| Luber & Arras, *RSS* 2013; Linder, Breuers, Leibe & Arras, *ICRA* 2016 (SPENCER) | social-group tracking; laser and RGB-D tracking in dense airport crowds, released as a ROS framework | group interaction is what the slot exchange was meant to learn, and E6 found it earns nothing on FROG â€” a contrast to state. SPENCER is also the ready-made deployment baseline for `../TODO.md` Â§E |
| Leigh, Pineau, Olmedo & Zhang, *ICRA* 2015 | leg tracker, the classical baseline of DROW and Li2Former | already cited (Â§2.2) |
| Guerrero-Higueras et al., *Frontiers in Neurorobotics* 2019 (PeTra) | its title is a *tracking* paper: U-Net leg segmentation plus leg-pair tracking | already cited as a detector (Â§7.2); cite it as a tracker too |
| Aguirre & GarcÃ­a-Silvente, *Neural Computing and Applications* 35, 2023 | leg detector trained on camera auto-labels, plus a Kalman tracker, reported above Leigh's | self-supervision (E18) |
| Jia, Steinweg, Hermans & Leibe, *ICRA* 2021 | 2D-LiDAR detectors trained on camera pseudo-labels, close to manual-label performance | self-supervision (E18); same group and code base as DR-SPAAM |
| Plozza et al., *SAS* 2024 (arXiv 2412.15000) | DR-SPAAM + Norfair, MOTA 85.45%, 20 Hz on a Jetson Xavier NX | already cited (Â§3.5); its recordings are the transfer test (`../TODO.md` A51) |
| Arreghini, Nava, Carlotti, Paolillo & Giusti, arXiv 2607.21309 (July 2026) | planar-LiDAR person detection and 2D pose from factorised space-time blocks over **30 scans (3 s at 10 Hz)**, odometry-reprojected; a Kalman filter tracks in field tests; **65.9 / 66.4% AP on FROG** at d = 0.5 m, 2.9-3.4 ms on a robot CPU, ~40k parameters, trained by cross-modal self-supervision on the authors' own data | the newest published FROG detector, and a temporal one: a 3 s window lands at LFE's level, which agrees with `PAPER.md`'s claim that a window is not the lever. Confirm its training regime (zero-shot or FROG-trained) before it enters `../memory/performance-log.md` |
| Wang, Thorpe, Thrun, Hebert & Durrant-Whyte, *IJRR* 2007 (SLAMMOT); Vu & Aycard, *ICRA* 2009 | detection and tracking of moving objects (DATMO) from a moving laser platform, separating static from dynamic structure | the classical frame for room-coordinate slots (D1) and odometry drift (E2, E11). Vu & Aycard is prof. Aycard's own laser DATMO work: context for `../TODO.md` A54, **not** the reference for `AlgorithmicDetector` |

#### 7.4.3 Tracking methods closest to stage 3

| work | mechanism | relation to TAKHeLiPeD |
| --- | --- | --- |
| Tracktor (Bergmann, Meinhardt & Leal-TaixÃ©, *ICCV* 2019) | the detector's own regression head carries last frame's boxes forward, so the detector *is* the tracker | a detector-in-the-loop precedent for E12's feedback, beside CenterTrack (cited) |
| Detect to Track and Track to Detect (Feichtenhofer, Pinz & Zisserman, *ICCV* 2017) | joint detection and cross-frame regression; tracks rescore detections | the video-detection precedent for "tracking raises detection AP" |
| MOTRv2 (Zhang, Wang & Zhang, *CVPR* 2023) | a pretrained detector's proposals become the queries of an end-to-end tracker | the transformer form of stage 2's candidates feeding stage 3 |
| SimTrack (Luo, Yang & Yuille, *ICCV* 2021); PnPNet (Liang et al., *CVPR* 2020) | joint detection and tracking from raw point clouds, association, birth and death read off one map; tracking in the loop of an end-to-end LiDAR network | "memory inside the detector" in 3D LiDAR, beside MAD (cited) |
| 3D-MAN (Yang et al., *CVPR* 2021); PF-Track (Pang et al., *CVPR* 2023) | a memory bank of past proposals attended by the detector; past and future reasoning over object queries, cutting ID switches by 90% | same; PF-Track's future reasoning is what stage 3's constant-velocity prediction does by rule |
| Tokmakov, Jabri, Li & Gaidon, *ICML* 2022 | object permanence learned **self-supervised**, by a random walk along memory | slots trained on unlabelled recordings (E18) |
| ImmortalTracker (Wang et al., arXiv 2111.13672) | never terminates a track; predicts through darkness | E1's "never emptied" is already its best row (83.88%); an immortal-slot arm costs one evaluation |
| SimpleTrack (Pang et al., *ECCV Workshops* 2022) | decomposes 3D MOT into pre-processing, association, motion model and life cycle; two-stage association | the vocabulary for stage 3's fixed rules in a write-up |
| GNN-PMB (Liu et al., arXiv 2206.10255) | Poisson multi-Bernoulli filter on LiDAR detections, with an existence probability per track | the random-finite-set statement of a slot's value |
| DeepMOT (Xu et al., *CVPR* 2020) | a differentiable Hungarian proxy trains trackers on MOTA and MOTP | the relaxation A51 D5 names for making spawn and eviction trainable |
| SchÃ¶ller, Aravantinos, Lay & Knoll, *RA-L* 2020 | a constant-velocity model beats learned pedestrian predictors | justifies stage 3's constant-velocity slot prediction |

#### 7.4.4 Metrics

| metric | needs identities | fits |
| --- | --- | --- |
| GOSPA (*FUSION* 2017) | no | FROG and DROW, every frame; splits into localisation, missed and false |
| OSPA; OSPA(2) (*TPAMI* 2023) | OSPA no; OSPA(2) yes | JRDB's own |
| CLEAR-MOT: MOTA, MOTP (Bernardin & Stiefelhagen, 2008) | yes | JRDB (ranked by MOTA), Plozza et al. |
| HOTA (Luiten et al., *IJCV* 2021) | its DetA no; AssA yes | DetA is a detection score; AssA needs identities |
| IDF1 (Ristani et al., *ECCV Workshops* 2016) | yes | as above |

FROG carries no identities, so only the identity-free rows apply to this project's headline dataset.

#### 7.4.5 Experiments this suggests

Proposed 2026-09-28 as `../TODO.md` A56, approved 2026-10-02, and **E14-E16 and E20-E22 run 2026-10-02/03**; the predictions and the scored outcome are in `../memory/experiments/2026-09-tracking-claims.md`. In one paragraph: SORT and ByteTrack as published lose to the bare candidates (77.40 / 76.49% against 77.66%); ByteTrack with CBMOT's track confidence, tuned on val, gains +3.30 pp (80.96%), but only with a confidence decay, since persistent phantoms otherwise saturate as people do; the proposed memory is +2.90 pp above that, every seed's interval above zero, and its edge over the tuned rule sits on occluded people and new trajectories; GOSPA ranks it best by a smaller margin (0.502 against 0.530); and the whole streaming detector runs at 12.9 ms per frame on the GPU and 19.5 ms on four CPU threads. E19 ran as A51 S2; E17 and E18 wait for JRDB (A49).

| # | what | cost | what it decides |
| --- | --- | --- | --- |
| E14 | ByteTrack, and OC-SORT's re-update, on the same stage-2 candidates with the sub-threshold ones included, scored by AP with a track score | evaluation only | whether the memory's +5.68 pp exceeds a 2022 post-hoc tracker given the same sub-threshold evidence â€” the first comparison a tracking reviewer will ask for |
| E15 | CBMOT's score refinement: rescore each candidate from its slot's rule value, which `manage_slots` already computes, with no learned head | evaluation only | the hand-coded twin of E10; with E14, whether the learned part adds anything to rules published in 2021-2022 |
| E16 | GOSPA per frame beside AP, for the published detectors, stage 2 and stage 3 | evaluation only, same pass as E14 | puts `PAPER.md`'s false-positive argument into an established metric |
| E17 | score the slots *as a tracker* (HOTA, MOTA, IDF1, OSPA(2)) where identities exist: JRDB's 3D tracks projected to the ground plane, and Plozza et al.'s mocap if they share it | needs A49's JRDB loader | a second result: a detector whose memory is also a competitive tracker, against DR-SPAAM + Norfair and DR-SPAAM + ByteTrack |
| E18 | self-supervised slot training, or label densification from slot tracks (Tokmakov 2022; Jia 2021; Qi et al., *CVPR* 2021, offboard auto-labelling) | a research branch | whether DROW's ~95% unlabelled scans and FROG's extra recordings can train the memory |
| E19 | an occlusion input for the coasting head: whether a nearer return lies on the slot's bearing (after Arras 2008) | one training run | O2 found the head cannot tell a remembered person from a remembered phantom; occlusion explains a person's absence and not a chair's |

E14 and E15 came first because they could falsify the headline claim at no training cost; they narrowed it instead (above).

#### 7.4.6 Who claims a *detection* gain from tracking or object memory (checked 2026-09-29)

The owner's question: does any of this literature claim that tracking raises detection accuracy, as TAKHeLiPeD's +5.68 pp over its own candidates does?
**Yes, outside 2D LiDAR; no one inside it.**
Tracking-by-detection papers (ByteTrack, OC-SORT, CBMOT, Plozza et al., PeTra, Leigh, Arras, SPENCER) report tracking metrics only (MOTA, HOTA, IDF1, localisation error), never the detector's AP.
The claims come from memory-augmented and multi-frame *detection*:

| work | data | causal | claimed detection gain |
| --- | --- | --- | --- |
| MAD (Agro et al., *CVPR* 2025) | 3D LiDAR and camera; Waymo, AV2 | yes | **+2.5 to +7.6 AP** over the detectors it wraps, all classes |
| MoDAR (Li, Qi et al., *CVPR* 2023, arXiv 2306.03206) | 3D LiDAR, Waymo; a Kalman tracker plus MultiPath++ forecasts fed back as virtual points | online: 9.1 s of past; offline: past and future | **pedestrian AP, online: +1.5 to +3.4** (L1/L2, three detectors, Table 1); the headline +7.2 / +11.1 L2 mAPH (online / offline) is mostly heading, which point detection does not score |
| Detect to Track and Track to Detect (*ICCV* 2017) | video, ImageNet VID | no: tubes linked over the whole video | 74.2 -> 79.8 mAP (+5.6) over its frame-level R-FCN |
| 3D Auto Labeling (Qi et al., *CVPR* 2021); Seq-NMS (2016) | 3D LiDAR; video | no | "significant gains", on par with human labels; weak detections rescored from linked strong ones |
| DR-SPAAM (*IROS* 2020) | 2D LiDAR | yes, ~0.15 s template | +1.9 AP on FROG (5 scans against 1): temporal, not tracking |
| Arreghini et al. (arXiv 2607.21309, 2026) | 2D LiDAR | yes, 3 s window | 65.9 / 66.4% FROG AP; its Kalman tracker only smooths field demonstrations |

What this means for the write-up:

- **TAKHeLiPeD would be the first 2D-LiDAR detector to report a detection gain from object-level memory**, as far as this search reaches: +5.68 pp AP at 0.5 m over its own candidates, three seeds.
- **The within-model gain is the defensible number, not 83.3% against 75.6%.** The latter is ours measured here against DR-SPAAM's published figure (`../memory/interpreting-evaluation.md`), and 75.6% is DR-SPAAM's, not DROW3's 73.9%.
- **Cross-domain magnitudes do not compare.** MAD's and MoDAR's gains are 3D-box AP on Waymo and AV2; ours is point AP on FROG. Cite them as the same phenomenon in other sensors, never as a smaller number.
- **E14 decides what the gain is attributed to, not whether it exists.** If ByteTrack on the same candidates recovers most of it, the finding "tracking raises 2D-LiDAR detection AP by several points" is still unclaimed on this sensor; the novelty then moves from the learned memory to the finding and its cheapest mechanism.
  **Decided 2026-10-03 (Â§7.4.5): both halves.** A tuned ByteTrack + CBMOT recovers 3.30 of the memory's 6.20 pp and the learned state the remaining 2.90, so the write-up reports a detection gain from tracking on 2D LiDAR, which no one has published, and a learned memory that clearly beats the best rule.

#### 7.4.7 Classical trackers on the published detectors

The same trackers on LFE-Peaks' and LFE-PPN's detections (`../utils/tracker_sweep.py --tracker byte / byte-cbmot`, A56), read as `../docs/PAPER.md` reads SORT: false positives removed per false negative added, populated frames, against the untouched detector.

| | LFE-Peaks | LFE-PPN |
| --- | --- | --- |
| SORT, `min_hits` 3 | 4.1 (AP 63.7%) | 10.4 (AP 68.6%) |
| ByteTrack, `min_hits` 1 | 4.4 (AP 62.7%) | **27.6** (AP 68.0%) |
| ByteTrack, `min_hits` 10 | 3.4 (AP 60.9%) | 17.8 (AP 67.2%) |
| merge radius 0.30 -> 0.40 m (`PAPER.md`) | **21.9** | **~63** |

ByteTrack trades far better than SORT on LFE-PPN's dense phantom field and about as well on LFE-Peaks, and the merge radius still dominates both: `PAPER.md`'s conclusion that no post-hoc stage beats a post-processing constant survives a 2022 tracker. Untuned CBMOT confidence (no decay) collapses AP on both detectors (49.8-51.2% and 36.6-39.4%), for the saturation reason above.

---

## 8. Evaluation Results

The full evaluation history â€” every number measured, the bugs found while producing them, and the tuning sweeps run since â€” has moved out of this document.

This section used to hold that material inline; splitting it out is what keeps this document usable as a citation-grade research write-up rather than a running lab notebook.

- **The standing comparison against published SOTA** (accuracy per architecture per dataset, ms/frame): [`../memory/performance-log.md`](../memory/performance-log.md).
- **Individual runs and the `dtime`/NMS/threshold/tracker sweeps**: [`../CHANGELOG.md`](../CHANGELOG.md).
- **How to read a number correctly** â€” which regime it's from, what it doesn't correct for, what it's not comparable to: [`../memory/interpreting-evaluation.md`](../memory/interpreting-evaluation.md).
- **What was tried and didn't pay**, with what would reopen it: [`../memory/rejected-ideas.md`](../memory/rejected-ideas.md).
- **The dated history of every finding and fix** â€” the DR-SPAAM cutout-normalization bug, the LFE-PPN anchor/recall bugs, the FROG odometry and session-splitting bugs, and everything else that happened along the way: [`../CHANGELOG.md`](../CHANGELOG.md).

---

## 9. Reference Quotes

A reference collection of direct quotes, for citing in an eventual writeup without re-fetching sources.

**On the cutout's limitations (motivating this research):**
> "DROW... obtaining state of the art detection results" but uses a fixed-size cutout with "zero cross-beam communication: the CNN processes each 48-beam window in complete isolation." â€” Section 3.1 above, summarizing Beyer et al. 2016/2018.

**On DR-SPAAM's design intent:**
> DR-SPAAM "addresses the sparsity problem of 2D LiDAR points by fusing multiple scans... using an alternative forward looking strategy that is more computationally efficient" than backward-looking fusion requiring explicit scan alignment. â€” paraphrased from Jia, Hermans, Leibe (IROS 2020), via secondary summary; verify exact wording against arXiv:2004.14079 before quoting in a final writeup.

**On the FROG dataset's motivation (also this project's training data):**
> "[We] propose a benchmark based on the FROG dataset, and analyze a collection of state-of-the-art people detectors." â€” Amodeo et al., arXiv:2306.08531 abstract.

**On LFE's full-scan, no-cutout design â€” the closest prior art:**
> LFE-Peaks and LFE-PPN use "a 1-D U-Net FCN... applied directly to the normalised raw scan vector rather than cutout windows" and are "single-scan only: no odometry, no temporal history." â€” Section 3.4 above.

**On PeTra's full-scan occupancy-grid design:**
> "the occupancy map is defined as a 256 Ã— 256 matrix, with a resolution of about 2 cm." â€” Guerrero-Higueras et al., *Frontiers in Neurorobotics* (2019), Section on input representation.
> "PeTra spends â‰ˆ0.3 s on calculating a location estimate... LD [Leg Detector baseline] spends â‰ˆ0.1 s" â€” same paper, on inference cost.

**On TCNs vs. recurrent models generally (motivating Architecture B):**
> Bai, Kolter, and Koltun's 2018 evaluation is the paper that established "TCN" as a name for causal dilated convolutional networks and showed they "convincingly outperform" canonical recurrent architectures (LSTM, GRU) "across a diverse range of tasks and datasets, while demonstrating longer effective memory." (Standard characterization of arXiv:1803.01271 â€” verify exact wording against the paper before quoting verbatim in a submitted paper.)

> The quote above is flagged "verify before quoting" because it was reconstructed from training-knowledge / search-engine summary rather than a directly fetched primary-source passage â€” re-fetch the primary source before using it as a verbatim quote in a submitted paper.
