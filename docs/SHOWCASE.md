# TAKHeLiPeD: a LiDAR person detector that remembers

**T**(emporal) **A**(daptive) **K**(nee-)**He**(ight) **Li**(dar) **Pe**(rson) **D**(etector) · *TA-KHé-Li-PeD*

Person detection from knee-height 2D LiDAR on mobile robots · FROG benchmark · **September 2026**

## State of the art

- **LFE-Peaks / LFE-PPN** (Amodeo et al., 2025): a whole-scan 1D U-Net, one scan at a time, no memory.
- **DROW3** (Beyer et al., 2018): a per-beam cutout through a small CNN and a vote grid; one scan on FROG.
- **DR-SPAAM** (Jia et al., 2020): DROW's cutouts plus spatial attention and an auto-regressive template — the strongest published detector here, over 5 consecutive scans, about 0.15 s.
- **Li2Former** (Yang et al., 2024): cutouts with a Transformer attending across each beam's last frames. Its authors never evaluated it on FROG, and it never trained to completion here: 2 GB of training memory per frame, and crashes in the environment available at the time. So it appears below only as a cost comparison.

The shared failure is **static false positives**: 1.4-3.4 per frame, 78-88% of them still present one frame later, sitting on furniture. Published scores are in the results table below.

## Theory: the evidence lives at three temporal horizons

- **Calibration** (under ~1 s): repeated scans of one surface average into a clean shape. The only horizon published detectors use; DR-SPAAM gains +1.9 AP from it.
- **Short** (1-10 s): object permanence through missed scans and occlusion.
- **Long** (10-60 s): people eventually move; furniture never does.

Beyond ~1 s memory must be addressed by **object**, not by beam or by place, since the robot moves ~0.5 m and turns up to ~10° per second; and association across frames must be **learned**, not a fixed distance gate. Replaying the annotations, people missed in one frame but detected shortly before or after are worth +9.6 points of recall, 90% of it within 10 s — and post-hoc memory cannot reach it: SORT, persistence maps and hindsight trail filters all remove fewer false positives per lost person than simply widening the detector's merging radius.

## The model

`detections, state = model.step(scan, odometry, state)`: streaming, constant cost per frame, ~1.8 MB of state.

1. **Input**: 10 pose-invariant features per beam, with one range rule (non-finite, non-positive or beyond the model range all read as the model range) standing in for any dataset-specific preprocessing.
2. **Calibration network** (1.18 M parameters): a 1D ConvNeXt U-Net over 720 beams, optionally with a causal temporal convolution per sector at the bottleneck (5 frames, ~1.2 s) whose cache is re-aligned by odometry. A DROW vote head yields 64 candidates, sub-threshold ones included. The result below uses the single-frame variant; the temporal one ties it.
3. **Object memory** (0.31 M parameters): 256 slots in room coordinates; cross-attention association, slot self-attention, and a selective diagonal recurrence (Mamba-style, Δt-aware, decay times 1-60 s). It rescores candidates, coasts briefly missed people, and can render predicted slot positions back into the calibration network.

## Results: FROG test set

| model | history used | AP @ 0.5 m | precision | recall | FP / frame |
| --- | --- | --- | --- | --- | --- |
| LFE-Peaks / LFE-PPN *(published)* | none | 65.6% / 69.2% | — | — | — |
| DROW3 *(published)* | one scan | 73.9% | — | — | — |
| DR-SPAAM `T=5` *(published)* | ~0.15 s | 75.6% | — | — | — |
| our calibration network alone | ~1 s | 77.7% | 50.9% | 88.8% | 2.63 |
| **+ object memory** | **seconds to a minute** | **83.3%** | **70.2%** | 83.9% | **1.09** |

Published rows are the papers' own figures; ours are measured here over every annotated frame of the test recording.
AP is a three-seed mean (82.96 / 83.17 / 83.88%, sd 0.47); the precision, recall and false-positive columns are the best of those seeds, at the shared 0.3 operating point.

- **The memory adds 5.7 points** over its own input and 7.7 over DR-SPAAM's published figure — ours measured here, three seeds, against a published number.
- **The gain is the learned memory, not temporal filtering:** a SORT tracker on the same candidates *lowers* AP at every setting, trading recall for precision about one for one (measured on an earlier chain's candidates: 76.7% against 76.9%).
- **And it is not the window.** The backbone this result sits on is single-frame; giving it a coarse temporal convolution over ~1.2 s is a tie within the run-to-run spread, and shuffling that window's order costs only ~0.6 points. The horizon that pays is the memory's, not the convolution's.
- **Capacity is not the limit:** the memory uses a median of 24 of its 256 slots, 107 at most.
- **Reporting is:** of the 26,996 people it does not report at 0.3, a live slot already sits on 95.5% of them.
- **Still open:** nearly a third of reported detections are false; and training both stages jointly, with the memory feeding back into the detector, has not yet beaten training the memory on a frozen backbone.

## Cost per frame

| model | multiply-adds | parameters | state between frames |
| --- | --- | --- | --- |
| LFE-Peaks | 24 M | 53 K | none |
| **ours (all three stages)** | **303 M** | **1.49 M** | **~1.8 MB** |
| DR-SPAAM, official streaming | 18.4 G | 1.98 M | ~10 MB |
| DR-SPAAM, 5-frame window | 48.0 G | 1.98 M | none |
| Li2Former, 5 frames | 88.2 G | 4.70 M | none |

Counted per frame at 720 beams, batch 1. **DR-SPAAM is the reference throughout** because it is the only temporal detector we could replicate: it has published FROG numbers and official weights that load and run here, while Li2Former has no published FROG result and never trained to completion on this hardware. **Against DR-SPAAM our detector performs ~60x fewer operations at a similar parameter count** — and ~158x fewer than the 5-frame windowed form — because the cutout it replaces re-encodes a window around every beam, while a full-scan network shares that work and its memory is carried as state instead of recomputed. Cost is flat in how far back memory reaches: reaching 60 s costs the same per frame as reaching 1 s.

Measured while streaming scans one at a time on one consumer GPU (Radeon RX 9060 XT, ROCm), the calibration network plus candidate decoding runs at **~10.4 ms per frame** and the memory adds **~4.4 ms** — against a scan every ~25 ms from this sensor. Those figures include a Python-side decoding loop; a proper single-frame latency benchmark of the whole detector is still to be run.
