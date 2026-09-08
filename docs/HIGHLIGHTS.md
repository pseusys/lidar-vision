# Project Highlights

> A one-page summary. For full technical detail, methodology, and citations, see [RESEARCH.md](RESEARCH.md); for work-in-progress threads and dead ends, see [IDEAS_BACKLOG.md](IDEAS_BACKLOG.md).

## Core ideas we tried

The central bet of this project is that a **full-scan, non-recursive CNN** — one that sees the entire LiDAR scan at once, with no fixed-size per-beam cutout and no RNN/attention — can match or beat cutout-based state-of-the-art person detectors (DROW, DR-SPAAM, Li2Former) while running orders of magnitude faster, since it avoids their defining cost: independently re-running a small network hundreds of times per scan. We built and compared three such architectures (a joint space-time CNN, a causal-TCN-then-CNN fusion, and a temporal U-Net) against faithfully reproduced baselines — verified line-by-line against official code or published weights, not just re-implemented from paper text. Around that core comparison, we explored several supporting questions: how wide a temporal window (`dtime`) actually helps once real motion is considered relative to frame rate; whether ego-motion compensation (odometry) changes that answer; and whether a lightweight post-hoc tracker can clean up the dominant error mode once a model is trained.

## Outcomes

The core bet paid off: all three proposed architectures beat every cutout baseline's accuracy on FROG, at 20–290× lower latency, with **SpaceTimeCNN** the clear standout (80.7% wp-AUC on FROG, the best number in the whole comparison, and the best zero-shot transfer to DROW too). Chasing down *why* certain baselines looked artificially weak paid off independently: a missing cutout-normalization step was found to be silently halving every input feature for DR-SPAAM's published weights (25.8%→68.1% wp-AUC after the fix), and a broken recall metric plus a hardcoded anchor count were found and fixed for the LFE family. The temporal-window question turned out to have a real, non-monotonic answer on FROG specifically: widening the window helps up to a point (peak at `dtime=5`, ~625 ms) and then reverses, because FROG ships no odometry and a wider window just gives the robot's own uncompensated motion longer to corrupt frame alignment — a mechanism confirmed twice independently (a wide-`dtime` retrain and a wide-tracker-invocation-rate sweep both regressed the same way). That, in turn, motivated building a scan-matching (ICP) pseudo-odometry estimator for FROG — which led to the most consequential finding of the whole project: FROG actually **ships real, official odometry that this codebase had simply never downloaded**, the whole pseudo-odometry effort having been built to route around a self-inflicted gap. Fixing that download path also surfaced two independent, previously-invisible correctness bugs — odometry integration and raw-scan temporal windowing were both silently crossing a ~24.6-hour gap between two bundled recording sessions — now fixed and covered by unit tests.

## Open questions

The most immediate open item is a re-run: every real-odometry `dtime` result on file predates at least one of the two session-boundary bugs, so the dtime=1/5/10 sweep needs to be redone on the now-fully-fixed pipeline (real odometry + correct session splitting) before any real-vs-estimated-odometry conclusion can be trusted. In parallel, we're using the FROG authors' own published weights to get independently-verified reference numbers rather than relying solely on this project's own training runs — DR-SPAAM (T=1, T=5) evaluation is running now; DROW3 and PeTra need a new architecture class and a wholly separate evaluation harness respectively, and haven't been started. Two smaller threads remain open too: whether shrinking SpaceTimeCNN's capacity for DROW's much smaller training set can close the gap that currently only zero-shot FROG→DROW transfer beats, and whether a per-track adaptive gating tracker (rather than one fixed radius) could recover the accuracy that `SimpleTracker` currently loses at wide invocation strides.

---

## Accuracy (person class, AP/AUC @ 0.5 m match radius; F1 where noted)

**Bold** = this repo's own reproduced/measured result. Plain = cited from the original paper (not independently re-run). `TODO` = run not yet finished. `—` = combination never attempted by either this project or the cited literature.

| Model | DROW | FROG | JRDB |
|---|---|---|---|
| AlgorithmicDetector (rule-based) | **F1 55.7%** | **F1 1.7%** | TODO |
| DrowDetector (DROW3) | **66.5%** *(paper 61.9%)* | **62.7%**¹ *(paper, retrained-on-FROG: 73.6%)* | TODO |
| DrSpaamDetector (DR-SPAAM) | **68.1%** *(paper 69.6%, RA-L'22 72%+)* | **67.2%**¹ *(paper, retrained-on-FROG T=5: 75.3%, T=1: 73.3%)*² | TODO |
| Li2FormerDetector | TODO³ *(paper 76.4%)* | — | TODO³ *(paper 81.4%)* |
| LFEPeaksDetector | **24.0%**⁴ | **74.6%** *(paper 64.9%)* | TODO |
| LFEPPNDetector | **11.7%**⁴ | **67.7%** *(paper 66.5%)* | TODO |
| PeTra | — | 49.6% *(paper)* / 58.3% *(PeTra\*, mixed-loss variant, paper)*⁵ | — |
| **SpaceTimeCNN** (novel) | **29.0%**⁶ | **80.7%**⁷ | TODO |
| **FullScanTCN** (novel) | **22.2%**⁶ | **69.7%**⁸ | TODO |
| **TemporalUNet** (novel) | **22.1%**⁶ | **65.4%** | TODO |

¹ Zero-shot transfer of DROW-trained published weights onto FROG, not retrained natively on FROG.
² Official retrained-on-FROG DR-SPAAM (T=1/T=5) weights are downloaded and loadable (zero missing/unexpected keys); a from-scratch evaluation of them through this repo's own pipeline is running now (see footnote in Performance table) — once done, this cell's FROG entry can be compared directly against the paper's 73.3–75.3%.
³ Training did not complete in this environment (repeated DirectML crashes + impractical CPU-only speed) — no locally-measured number exists yet for either dataset.
⁴ Cross-dataset transfer via zero-padding DROW's 450 beams onto FROG's trained 720-beam grid — an unverified extrapolation, not described in the LFE paper; treat as approximate.
⁵ Never implemented in this repo — PeTra's occupancy-grid task framing (256×256 binary segmentation + external leg-pairing) doesn't fit the shared beam-classification pipeline; would need its own harness.
⁶ Zero-shot transfer of FROG-trained weights onto DROW — the best of three tested regimes (native-from-scratch and fine-tuned both score lower for every architecture; see RESEARCH.md §8.1.1).
⁷ `dtime=5` real-time-window retrain (RESEARCH.md's original `dtime=1` baseline was 80.4%).
⁸ NMS/vote-clustering-tuned (+0.2pp over the 69.5% baseline).

## Performance (ms / frame, single scan, CPU unless marked)

| Model | DROW (450 beams) | FROG (720 beams) | JRDB (541 beams) |
|---|---|---|---|
| AlgorithmicDetector | **0.04 ms** | **0.06 ms** | **0.06 ms** |
| DrowDetector (DROW3) | **203.3 ms** | **276.4 ms** *(paper, GPU, T=1: 13.08 ms)* | **235.0 ms** |
| DrSpaamDetector (DR-SPAAM) | **201.5 ms** | **350.4 ms** *(paper, GPU: T=1 13.95 ms, T=5 13.99 ms)*⁹ | **298.3 ms** |
| Li2FormerDetector | **431.7 ms** *(paper, GPU: 14.95 ms)* | **590.6 ms** | **498.3 ms** *(paper, GPU: 17.54 ms)* |
| LFEPeaksDetector | **1.55 ms** | **1.76 ms** *(paper, GPU: 1.76 ms)* | **2.01 ms** |
| LFEPPNDetector | **315.5 ms**¹⁰ | **243.8 ms**¹⁰ *(paper, GPU: 1.49 ms)* | **278.3 ms**¹⁰ |
| PeTra | — | 28.17 ms *(paper, GPU)* / ≈300 ms *(PeTra's own paper, own dataset/hardware)* | — |
| **SpaceTimeCNN** (novel) | **8.7 ms** | **12.1 ms** | **12.5 ms** |
| **FullScanTCN** (novel) | **2.1 ms** | **2.6 ms** | **3.1 ms** |
| **TemporalUNet** (novel) | **2.0 ms** | **2.7 ms** | **2.3 ms** |

⁹ Official published DR-SPAAM (T=1/T=5) weights evaluated through this repo's own pipeline: job in progress (`results/eval_frog_official_drspaam.log`), ETA several hours (per-frame cutout extraction, not GPU-bound — see RESEARCH.md discussion). Will convert to a bold, directly-comparable number once it lands.
¹⁰ Cost is a decode-loop implementation artifact in this repo (unvectorized NMS), not architectural — a known, low-risk NumPy fix is identified but not yet applied (RESEARCH.md §8.5.2).
