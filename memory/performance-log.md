# Performance log

*keywords:* wp-AUC, ms/frame, dtime sweep, NMS sweep, threshold selection, tracker sweep, PeTra, Li2Former, comparison table

Measured numbers, compared row to row.
Read [`interpreting-evaluation.md`](interpreting-evaluation.md) first — several rows below are not directly comparable to each other without its caveats.

## Accuracy (person class, AP/AUC @ 0.5 m match radius; F1 where noted)

**Bold** = this repo's own reproduced/measured result.
Plain = cited from the original paper, not independently re-run.
`TODO` = run not yet finished.
`—` = combination never attempted by either this project or the cited literature.

| Model | DROW | FROG | JRDB |
| --- | --- | --- | --- |
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
² Official retrained-on-FROG DR-SPAAM (T=1/T=5) weights are downloaded and loadable (zero missing/unexpected keys) — a real, if partial, fidelity check already passed.
A from-scratch evaluation through this repo's own pipeline was attempted for both checkpoints and abandoned for both (T=5 by decision, T=1 after it stalled at >99.9% complete with an unexplained duplicate process running alongside it — `TODO.md` A2, settled); the paper's own numbers are cited here instead, deliberately, rather than spending further effort reproducing a number the paper already reports.
³ Training did not complete in this environment (repeated DirectML crashes + impractical CPU-only speed, `gotchas.md` I4) — no locally-measured number exists yet for either dataset.
⁴ Cross-dataset transfer via zero-padding DROW's 450 beams onto FROG's trained 720-beam grid — an unverified extrapolation, not described in the LFE paper; treat as approximate (`interpreting-evaluation.md`).
⁵ Never implemented in this repo — PeTra's occupancy-grid task framing (256x256 binary segmentation + external leg-pairing) doesn't fit the shared beam-classification pipeline; would need its own harness.
⁶ Zero-shot transfer of FROG-trained weights onto DROW — the best of three tested regimes (native-from-scratch and fine-tuned both score lower for every architecture; see `interpreting-evaluation.md`).
⁷ `dtime=5` real-time-window retrain (this project's `dtime=1` baseline was 80.4%) — see the dtime sweep below.
Predates the real-odometry/session-splitting fixes in `data-model.md`; the sweep has not yet been redone on the fixed pipeline (`TODO.md`).
⁸ NMS/vote-clustering-tuned (+0.2pp over the 69.5% baseline; see the NMS sweep below).

## Performance (ms / frame, single scan, CPU unless marked)

| Model | DROW (450 beams) | FROG (720 beams) | JRDB (541 beams) |
| --- | --- | --- | --- |
| AlgorithmicDetector | **0.04 ms** | **0.06 ms** | **0.06 ms** |
| DrowDetector (DROW3) | **203.3 ms** | **276.4 ms** *(paper, GPU, T=1: 13.08 ms)* | **235.0 ms** |
| DrSpaamDetector (DR-SPAAM) | **201.5 ms** | **350.4 ms** *(paper, GPU: T=1 13.95 ms, T=5 13.99 ms)*⁹ | **298.3 ms** |
| Li2FormerDetector | **431.7 ms** *(paper, GPU: 14.95 ms)* | **590.6 ms** | **498.3 ms** *(paper, GPU: 17.54 ms)* |
| LFEPeaksDetector | **1.55 ms** | **1.76 ms** *(paper, GPU: 1.76 ms)* | **2.01 ms** |
| LFEPPNDetector | **315.5 ms**¹⁰ | **243.8 ms**¹⁰ *(paper, GPU: 1.49 ms)* | **278.3 ms**¹⁰ |
| PeTra | — | 28.17 ms *(paper, GPU)* / ~300 ms *(PeTra's own paper, own dataset/hardware)* | — |
| **SpaceTimeCNN** (novel) | **8.7 ms** | **12.1 ms** | **12.5 ms** |
| **FullScanTCN** (novel) | **2.1 ms** | **2.6 ms** | **3.1 ms** |
| **TemporalUNet** (novel) | **2.0 ms** | **2.7 ms** | **2.3 ms** |

⁹ Official published DR-SPAAM weights evaluation through this repo's own pipeline was abandoned (`TODO.md` A2) — the T=1 attempt stalled at >99.9% complete after ~4h40m at ~7 fr/s (per-frame cutout extraction dominates, not GPU-bound) and never printed a result; T=5 was never attempted, on the reasoning that the paper's own number is enough (see the accuracy table's footnote 2).
¹⁰ Cost is a decode-loop implementation artifact (`interpreting-evaluation.md`), not architectural — a NumPy vectorization fix is identified but not yet applied (`TODO.md`).

## dtime sweep (SpaceTimeCNN unless noted, FROG, wp-AUC)

| dtime | window span | train | test |
| --- | --- | --- | --- |
| 0 (no temporal diversity) | none | 77.1% | 77.1% |
| 1 (original default) | ~125 ms | 76.7% | 76.7% |
| **5** | **~625 ms** | **80.7%** | **80.7%** |
| 10 | ~1.25 s | 78.1% | 78.0% |
| 20 (FullScanTCN, `T=10`) | ~4.5 s | 54.8%¹¹ | — |

Train ~ test at every point except the last, ruling out overfitting as the explanation for the dtime=5 peak.
¹¹ FullScanTCN, not SpaceTimeCNN, and val rather than test — see `rejected-ideas.md` for why this point regressed instead of continuing the trend.
This whole sweep predates the real-odometry and session-splitting fixes (`data-model.md`) and needs rerunning before its numbers are trusted further (`TODO.md`).

## Per-frame motion sanity check (real odometry, FROG train split, native 26.2 Hz / ~38.15 ms)

Run once `FROG_Dataset.scan_time`'s float32-precision bug (`CHANGELOG.md`, 2026-09-09) was fixed, to ground the `dtime` sweep in a measurement rather than an assumption about how much moves per frame.

**Robot's own ego-motion** (median per `dtime`-step, real odometry):

| `dtime` | span | median translation | median rotation |
| --- | --- | --- | --- |
| 1 | 38 ms | ~1.15 cm | ~0 deg |
| 5 | 191 ms | ~6.2 cm | ~0 deg |
| 10 | 382 ms | ~12.4 cm | ~0 deg |
| 20 | 763 ms | ~24.9 cm | ~0-0.07 deg |
| 40 | 1.5 s | ~49.8 cm | ~0-2.2 deg |

Small throughout — little for `aligned_raw_scan()` to have to correct regardless of `dtime`.

**Real person displacement**, matched directly against annotated positions (`det_wp`), not assumed from a walking-speed constant — greedy nearest-neighbour match per `dtime`-step, 2.5 m gate, ~13,000-15,000 matched pairs per row:

| `dtime` | span | median | mean | p90 |
| --- | --- | --- | --- | --- |
| 1 | 38 ms | 2.12 cm | 3.52 cm | 5.13 cm |
| 5 | 191 ms | 8.78 cm | 14.28 cm | 22.50 cm |
| 10 | 382 ms | 17.16 cm | 26.44 cm | 46.28 cm |
| 20 | 763 ms | 34.78 cm | 47.43 cm | 93.05 cm |
| 40 | 1.5 s | 67.14 cm | 81.56 cm | 160.70 cm |

Scales almost exactly linearly at ~1.7 cm per unit of `dtime` across this whole range.
At `dtime=1`, a person really does move only ~2 cm between sampled frames (well under a body width) — the "5 near-identical frames" intuition that prompted this check.
By `dtime=20`, ~35 cm — close to half a walking stride.

## NMS / vote-clustering sweep

| Model | Parameter | Sweep | Best | vs. default |
| --- | --- | --- | --- | --- |
| FullScanTCN | `vote_collect_radius` | 0.3/0.5/0.7/0.9/1.1/1.3 | **0.7** -> 69.7% | +0.2pp over 0.5 |
| LFE-PPN | `nms_radius` | 0.7/0.8/0.9/1.0 | **0.9** -> 61.2% | +0.5pp over 0.8 |

`min_thresh`/`score_thresh` are confirmed non-binding for both models across this whole range — the underlying score distributions don't fall where either sweep looked, so there is no useful middle ground on that axis for either model.

## Operating-threshold selection (best-F1 point vs. a 75% precision target, no retraining)

| Model | FP saved | FN added | Trade |
| --- | --- | --- | --- |
| SpaceTimeCNN | 968 | 718 | favorable (1.35:1) |
| TemporalUNet | 4179 | 3567 | favorable (1.17:1) |
| LFE-Peaks | 1278 | 936 | favorable (1.37:1) |
| FullScanTCN | 3944 | 4382 | unfavorable (0.90:1) |
| LFE-PPN | 3966 | 3951 | roughly break-even |

Three of five models have a real, free improvement sitting unused simply by moving the deployment threshold; FullScanTCN and LFE-PPN need their curve itself changed (the NMS sweep above), not just a different point on it.

## Tracker invocation-rate sweep (SpaceTimeCNN, FROG, wp-AUC)

| Stride | Real-time gap | No tracker | `SimpleTracker` (`radius=0.5, min_hits=3`) | Delta |
| --- | --- | --- | --- | --- |
| 5 | ~125 ms | 77.6% | **78.6%** | **+1.0pp** |
| 40 | ~1 s | 80.4% | 75.1% | **-5.3pp** |

The stride=40 regression is explained, not just observed — see `rejected-ideas.md`'s tracker entry for the radius/min_hits grid that isolated the cause.
