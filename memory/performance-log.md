# Performance log

*keywords:* wp-AUC, AP, ms/frame, SOTA, comparison table, published baseline, PeTra, Li2Former, LFE, DR-SPAAM, DROW3, state of the art, Aycard, AlgorithmicDetector, attribution, credit

The standing comparison: published state of the art against this project's own architectures, **one row per model, latest measurement only**.

**History does not live here.** How a number moved, which run produced it, and every sweep that led to a setting are in [`../CHANGELOG.md`](../CHANGELOG.md) — grep its `*keywords:*` lines.
Read [`interpreting-evaluation.md`](interpreting-evaluation.md) before quoting any row; several are not comparable to each other without its caveats.

## How a cell gets filled

**A cell holds the original authors' published number wherever they published one** (owner's call, 2026-09-10).
This project evaluates a third-party model only to **fill a gap** — a model/dataset pair nobody published — never to re-derive a number that already exists.
Re-deriving a published accuracy figure through this pipeline adds risk without adding information; this session's own audit is the case in point.

**Speed is the exception.** `ms/frame` is only comparable when every row was measured on one machine, so every timing number here is ours by necessity.

| notation | meaning |
| --- | --- |
| plain `61.9%` | published by that model's own authors — the yardstick, never re-run here |
| **bold** | measured by this project: our own architectures, or a gap the literature does not cover |
| `gap` | nobody published it and we have not measured it |
| `check paper` | the authors did publish one; it is simply not transcribed here yet. **Look it up before spending a run.** |
| `n/a` | not measurable here — JRDB needs manual registration and is not downloaded (`gotchas.md`) |

## Accuracy — person class, AP/AUC @ 0.5 m match radius (F1 where noted)

| Model | Kind | DROW | FROG | JRDB |
| --- | --- | --- | --- | --- |
| ROS `leg_detector` | classical | check paper | 20.2% | check paper |
| AlgorithmicDetector | rule-based, **O. Aycard**⁰ | **F1 55.7%** | **F1 1.7%** | n/a |
| PeTra | published | gap | 50.1% *(PeTra\* 59.0%)* | gap |
| LFE-Peaks | published | gap¹ | 65.6% | gap¹ |
| LFE-PPN | published | gap¹ | 69.2% *(ours 69.6%)* | gap¹ |
| DROW3 | published | 61.9% | 73.9% | 82.9% *(official repo README, merged 1091-point 360° scans)* |
| DR-SPAAM | published | 69.6% *(RA-L'22: 72%+)* | **75.6%** *(`T=5`)* / 73.7% *(`T=1`)* | 84.9% *(official repo README, merged 1091-point 360° scans)* |
| Li2Former | published | 76.4% | gap² | 81.4% |
| **SpaceTimeCNN** | ours, novel | re-measuring | re-measuring³ | n/a |
| **FullScanTCN** | ours, novel | re-measuring | re-measuring | n/a |
| **TemporalUNet** | ours, novel | re-measuring | re-measuring | n/a |
| **TAKHeLiPeD, stages 1-2** (calibration network) | ours, novel | gap | **82.2%**⁹ *(old recipe 77.7%⁴)* | n/a |
| **TAKHeLiPeD, stages 1-3** (+ object memory) | ours, novel | gap | **85.9%**⁹ *(old recipe 83.3%⁵)* | n/a |

**DR-SPAAM `T=5` at 75.6% on FROG was the number to beat, and the object memory clears it by 7.7 pp.**
That is ours measured here, three seeds, against theirs published, one paper figure — read footnote 5 and [`interpreting-evaluation.md`](interpreting-evaluation.md) before quoting it.

**The comparison a tracking reviewer will ask for is within one model, on identical candidates** (`TODO.md` A61, 2026-10-09, on the adopted recipe; one replay per stage-2 seed; FROG test, every 5th annotated frame, 95% block-bootstrap intervals per seed; tables in [`experiments/2026-09-tracking-claims.md`](experiments/2026-09-tracking-claims.md)):

| what reads TAKHeLiPeD's stage-2 candidates | AP, 3 seeds | vs the candidates |
| --- | --- | --- |
| nothing | 82.19% | -- |
| SORT (`min_hits` 1) / ByteTrack, as published | 82.31% / 81.17% | +0.13 / -1.02 |
| ByteTrack + CBMOT track confidence, tuned on val | 83.51% | +1.32, every seed's interval above zero |
| **object memory, no attention (E13)** | **85.87%** | **+3.68; +2.36 over the tuned tracker (+2.84 / +2.31 / +1.94), every seed's interval above zero** |

At one matched false-positive rate the two now find the same people (95.47 against 95.56%); the memory's lead is ranking, not recall at a threshold. On the old recipe (A56, 2026-10-03, vote-grid candidates) the rows read 77.66 / 77.40 / 76.49 / 80.96 / 83.86%, the memory +2.90 over the tuned tracker. All on FROG's official test labels, which include 2,254 duplicate circles that only a second detection can match; on the old recipe, without them every row rose 0.7-1.05 pp and the memory's margin over the tuned tracker was +2.71, still above zero at every seed ([`interpreting-evaluation.md`](interpreting-evaluation.md), 2026-10-05).

**TAKHeLiPeD** (`T`emporal `A`daptive `K`nee-`He`ight `Li`dar `Pe`rson `D`etector, named 2026-09-23) is what earlier entries in this file, `CHANGELOG.md` and `TODO.md` call *the three-horizon detector*; the code still spells it `three_horizon`.

### The FROG column is AP @ d = 0.5 m, and it did not used to be

**Corrected 2026-09-10, after reading the paper's Table 4 column headers rather than its first numeric column.**
FROG's Table 4 reports **nine** numbers per model, in three groups:

| | mAP | mPeakF1 | mEER | AP | PeakF1 | EER | AP | PeakF1 | EER | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | *averaged over d = [0.3 : 0.05 : 0.5]* | | | *d = 0.5 m* | | | *d = 0.3 m* | | | ms |
| ROS leg detector | 15.8 | 30.8 | 30.5 | **20.2** | 35.2 | 34.9 | 10.0 | 24.3 | 24.1 | 1.77 |
| PeTra | 49.6 | 66.4 | 66.2 | **50.1** | 66.6 | 66.4 | 49.1 | 66.1 | 65.9 | 28.17 |
| PeTra\* | 58.3 | 67.7 | 67.1 | **59.0** | 67.9 | 67.3 | 57.9 | 67.4 | 66.8 | — |
| LFE-Peaks | 64.9 | 70.2 | 70.2 | **65.6** | 70.7 | 70.7 | 63.2 | 69.0 | 69.0 | 1.76 |
| LFE-PPN | 66.5 | 68.7 | 68.6 | **69.2** | 69.5 | 69.5 | 62.5 | 67.3 | 67.2 | 1.49 |
| DROW3 (T=1) | 73.6 | 71.9 | 71.6 | **73.9** | 72.0 | 71.8 | 73.0 | 71.6 | 71.4 | 13.08 |
| DR-SPAAM (T=1) | 73.3 | 71.9 | 71.6 | **73.7** | 72.1 | 71.8 | 72.7 | 71.6 | 71.3 | 13.95 |
| DR-SPAAM (T=5) | 75.3 | 73.4 | 73.3 | **75.6** | 73.6 | 73.4 | 74.7 | 73.2 | 73.0 | 13.99 |

The leftmost column is **mAP over five association distances**, MS-COCO style — not AP at any single one.
Every FROG figure this project quoted before 2026-09-10 (15.8, 49.6, 64.9, 66.5, 73.6, 75.3) came from that column, while every number *we* produce is AP at `--eval-r 0.5`.
**They were never the same quantity**, and the gap is not uniform: it is +0.5pp for DROW3 and **+2.7pp for LFE-PPN**, because a detector's sensitivity to the association distance is itself a property that varies between detectors.

The table above now carries the **d = 0.5 m** column, which is what our numbers are comparable to.
When quoting FROG, always say which `d`.

⁰ **The algorithm is not this project's**: `AlgorithmicDetector` is the two-tier leg+chest clustering-and-tracking detector written by **prof. O. Aycard**, ported here from his C++ (`library/cpp_core/sources/detector.cpp`, "Written by O. Aycard").
Credit it as his in anything published from this repo; only the pybind11 binding, the Python wrapper and the evaluation harness around it are ours.
It is a classical reference point, not a baseline taken from any of the papers compared here (`../docs/RESEARCH.md` §2.1), and the formal reference to cite is `../TODO.md` §A.

¹ The LFE paper only ever evaluates on FROG's 720-beam scans.
This project *has* run LFE cross-dataset by zero-padding shorter scans to 720 (DROW: 24.0% / 11.7%), but that padding is our own invention, is not described in the paper, and does not resample to FROG's 0.25°/beam angular resolution — beam index stops meaning the same physical angle.
Degraded by construction, so it is recorded here as a footnote rather than reported as a result.
² Li2Former was never evaluated on FROG by its authors, and training it here has never completed (DirectML crashes, impractical CPU-only speed — `gotchas.md` I4).
³ In progress: `T=5, dtime=10` on the fixed pipeline (`TODO.md` A1).
⁴ `TODO.md` A51's backbone, measured 2026-09-19 over the whole FROG `official` test split: **77.66%**, the no-temporal stage 2 (seed 0) whose decoded candidates the object memory in the row below rescores, at 2.620 false positives per frame and 88.8% recall at the 0.3 operating point.
That backbone's own three-seed mean is **77.23%**; the default architecture, which kept the coarse temporal convolution (removed 2026-10-08, `TODO.md` A60), means **77.92%** over three seeds. The pooled per-run spread across five three-seed arms is ~0.95 pp (`interpreting-evaluation.md`), so those two are a tie, not a ranking.
Both are scored under the limit-beam decoding rule and the running time step adopted 2026-09-17.
**The vote-grid decoder understates the network** (`TODO.md` A59): the same no-temporal weights through `nms-mean`, NMS chosen on val on all six checkpoints, score **80.07%** (three seeds, every 5th annotated frame; default architecture 79.61%). Mirror augmentation lifts it further, to **81.49%**, and the whole adopted recipe with dropout 0.1 and weight averaging to **82.20%** (three seeds each, every 5th annotated frame). Neither is this row's figure yet, because every stage-3 result rests on vote-grid candidates and moves only once stage 3 is retrained on the new ones.
**The 76.9% this row used to carry is retired, not corrected**: a different backbone, measured before both of those changes (`CHANGELOG.md`, 2026-09-15 to 2026-09-17).
⁵ `TODO.md` A51's B0 baseline, measured 2026-09-19: stage 3 (object memory, 256 slots) trained by `step3a --chunk-s 1 10` on footnote 4's cached candidates with stage 2 frozen, **three seeds**; every test recording played from empty memory, every annotated frame.
Test AP **82.96 / 83.17 / 83.88%** (mean **83.34**, sd **0.47**), AP at 0.3 m 81.36 / 81.52 / 82.17%, and recall at the candidates' own false-positive rate 92.2 / 92.5 / 92.7% against 88.8% -- **+5.68 pp over its own input**.
On the best seed at the shared 0.3 threshold it covers 128,895 of 153,655 annotated people at **1.092 false positives per frame**, against the candidates' 136,417 at 2.628 (`a51_b0_memory_diagnosis.json`, 2026-09-21); precision there is 70.2%.
A51's screening threshold is set from this spread at **~1.5 pp**, and no design arm has cleared it in either direction yet (T1 +1.05, D2 +0.60, D3 -0.90, D1 -1.32, one seed each).
**What the gain is not.** SORT replayed on the same candidates loses AP at every setting, so it is the learned memory rather than temporal filtering as such; and capacity is not the lever -- the memory uses a median of 24 of its 256 slots (107 at most), and in the recovery diagnosis of 2026-09-22, of the 26,996 people it does not report at 0.3, **95.5% already have a live slot on them** at a median value of 0.986. The remaining loss is in the reporting head.
**The 80.10% this row used to carry is retired, not corrected**: a different backbone, and it predates the running time step and the limit-beam decoding. Joint training with feedback (step 3b) reached 78.84% on that older chain and has not been repeated on this one, so training the memory on a frozen stage 2 remains the best configuration.

### Reading our FROG row against a published one

Two things still keep the FROG comparison from being perfectly clean once our number lands, and the second does not flatter us:

- ~~Training-set size.~~ **Resolved**: we train on the paper's own `split == 0` frames, **108,356**, exactly theirs (`data-model.md`).
- Our val is three held-out recordings; theirs is an interleaved per-frame holdout inside the training data.
  A different — and stricter — model-selection signal, on the same test set.
  This does not flatter us either way; it just means our early stopping optimises for cross-recording generalization rather than for interpolation.
- Training budget: the FROG authors trained DROW3/DR-SPAAM for exactly 5 epochs at batch 8 (~67,700 minibatches, 541k scans seen), deliberately matched to DROW's scale, and LFE for up to 100 epochs.
  **TAKHeLiPeD's stage 2 sees fewer scans than any of them, not more**: its selected checkpoint sits at epoch ~1 at batch 32, ~3,400 minibatches and 108k scans, and its test AP *falls* with longer training (77.60% at epoch 1 to 72.36% at epoch 4, `TODO.md` A59).
  The "~38x the compute, in our favour" this bullet used to carry described the retired full-scan architectures' 100-epoch runs and does not apply to the TAKHeLiPeD rows.

The metric is not a caveat: `wp-AUC` and the papers' `AP` are the same trapezoidal-AUC-over-a-Hungarian-matched-PR-curve quantity, verified by direct source comparison against the official DR-SPAAM repo (`interpreting-evaluation.md`).
Both sides now score the same `frog_16-41` test recording, which was not true before 2026-09-10.

### What this project still has to measure

Much less than it looked.
Under the policy above, the standing work is:

- **our three architectures**, on DROW and FROG — nobody else can publish these;
- **the speed table**, all of it, because one machine is the only way those rows compare;
- **genuine gaps**, and only if they are worth the GPU time: Li2Former on FROG, PeTra anywhere outside FROG, LFE outside FROG (and that one is degraded by construction, so probably not worth it).

Everything marked `check paper` should be transcribed from the paper, not measured.

### Our reproductions are a harness self-test, not a result

This project *has* independently reproduced several published numbers, and those reproductions stay useful for one specific purpose: **checking the evaluation harness against a known answer.**
On DROW the pipeline lands close — DROW3 66.5% against a published 61.9%, DR-SPAAM 68.1% against 69.6%.

**DR-SPAAM's 1.5 pp undershoot was mostly our own wrapper, found 2026-10-10** (`TODO.md` A47). The official detector is
*stateful*: it keeps its spatial-attention template between calls and clears it only at a sequence boundary. Ours re-encoded a
`T`-scan window from scratch per frame and seeded the template from that window's oldest scan, which is a different detector.
Scored both ways on DROW test with the published weights, through one decoder and one metric: **windowed 67.84%, streaming
69.08%** — so the published algorithm is **+1.24 pp** better than what we were calling DR-SPAAM, and our reproduction of it
moves from −1.5 pp to **−0.52 pp** of the authors' 69.6%. **Every DR-SPAAM accuracy figure in this file is the windowed form and
understates the baseline by about that much**, on all three datasets.

**Its cost is misreported in the opposite direction.** The windowed form runs the first two conv blocks five times per frame
where the published one runs them once and attends to its carried template, so the 350.4 ms FROG row below *overstates* the
published detector — we have been reporting DR-SPAAM as both less accurate and slower than it is. The corrected figure is part of
the one-protocol timing sweep (`TODO.md` §A item 29), which now has `DrSpaamDetector.stream()` to time.

On FROG it did not, and that was the visible symptom of the contamination all along: **LFE-Peaks measured 74.6% here against its authors' own 64.9%** — a +9.7pp overshoot for the same published weights on the same dataset, which is not a thing that can legitimately happen.

**Standing rule: if our reproduction of a published number overshoots it, the harness is wrong.**
An undershoot has innocent explanations: a different frame sample, a wrapper detail, a merge radius we had to pick ourselves.
An overshoot on identical weights does not.

**Where that rule stands after 2026-09-10.** Mostly resolved, by finding that the comparison itself was wrong: we were reading the mAP column (above).
Against the correct `d = 0.5 m` column, on the genuine `frog_16-41` test recording with the two `LFEPPNDetector` bugs fixed:

| | ours (AP@0.5) | paper (AP@0.5) | delta |
| --- | --- | --- | --- |
| LFE-Peaks | **65.1%** | 65.6% | **-0.5** |
| LFE-PPN | **69.6%** | 69.2% | **+0.4** |

**Both rows updated 2026-09-11**, and both moved for the same reason — each detector's merge radius was recalibrated and the table was not re-run. They previously read 67.7% (+2.1) and 68.6% (-0.6).

**LFE-Peaks' +2.1pp overshoot is resolved**: it *was* `_merge_nearby`'s radius. The paper specifies "the most common ground truth circle **diameter**" (0.8 m in FROG's annotations) and the code used the *radius* (0.4 m); at **0.30 m** — neither of those — LFE-Peaks reproduces to -0.5pp at `d = 0.5` and +0.2pp at `d = 0.3`.
**LFE-PPN now overshoots by 0.4pp**, having previously undershot by 0.6. That is inside the harness's own sampling noise and far short of the +9.7pp that made the overshoot rule necessary, but it is on the wrong side of it and is logged rather than waved through (`TODO.md` A32d).

**A second defect surfaced from the same comparison, in the opposite direction.** Our LFE-PPN matches at `d = 0.5 m` and undershoots by **9.4pp at `d = 0.3 m`** (53.1% against 62.5%), against LFE-Peaks' flat +2.1/+2.5.
That signature is **mislocalisation**: the detections land near enough to pass a 0.5 m gate and too far to pass a 0.3 m one.
It points at the anchor-decode geometry calibrated on 2026-09-10, whose angular convention was explicitly recorded as weakly identified *because AP@0.5 could not distinguish it*. AP@0.3 can, and does.
**Use `--eval-r 0.3` as the discriminator when calibrating anything positional.**

## TAKHeLiPeD's cost, end to end, one scan at a time

**Measured 2026-10-10 on the adopted recipe** (`tracking_baselines.py latency`, E22; 2,000 consecutive test scans, batch 1, one stream, as on the robot; `checkpoints_three_horizon/a61/latency_*.json`).
The budget is **25 ms**, FROG's real frame period — not the 38.2 ms that every earlier cost claim was set against (`interpreting-evaluation.md`).

| | stage 1-2 | candidate decoding | object memory | **total** | p99 | share of the 25 ms budget |
| --- | --- | --- | --- | --- | --- | --- |
| **GPU** (RX 9060 XT) | 5.41 | 3.23 | 4.58 | **13.32 ms** | 15.32 | **53%**, 61% at p99 |
| **CPU, 4 threads** | 8.71 | 2.83 | 9.27 | **20.98 ms** | 25.32 | **84%**, **101% at p99** |

**The GPU has headroom; the CPU's tail does not.** At the 99th percentile four CPU threads take 25.32 ms against a 25 ms budget, so a CPU-only deployment would drop frames occasionally — which the old 38.2 ms figure concealed, since the same measurement read as 55% of budget.
**The memory is the larger half of the CPU cost** (9.27 ms against stage 2's 8.71) and the smaller half of the GPU's (4.58 against 5.41): it is latency-bound on a CPU and bandwidth-cheap on a GPU.
Decoding is 2.8-3.2 ms either way and is pure Python plus numpy, so it is the obvious target if a deployment needs margin.

**Slot count barely matters**: 4.86 / 4.81 / 4.74 / 4.92 / 5.21 ms per memory step at 64 / 128 / 256 / 512 / 720 slots (A51 E4), so the O(K²) exchange is not what the cost is made of — and E6 removed that exchange altogether for nothing.

## Empty-scene false positives — FROG `transferred`, person-free frames only

**A genuine gap: nobody has published this, because the FROG benchmark contains no person-free frames at all.**
All 72,533 of them, from the three raw recordings, **30.2 minutes of real time at 40.1 Hz**.
**Every `FP/s` in this file rose by 1.53x on 2026-10-10**, when the frame rate stopped being read as the median stamped interval: FROG's recorder stamps scans in bunches, so the median says 26.2 Hz where the sensor delivers one scan every 25 ms (`interpreting-evaluation.md`).
The `FP/frame` columns are unaffected, and they are the ones to compare between detectors; the per-second columns are the ones to quote at a robot.
wp-AUC is *undefined* here, not merely low — no true positives means precision 0 at every threshold and recall with no denominator (`interpreting-evaluation.md`).
Every detection is a false positive by construction, so the metric is a count.

| Model | FP/frame @0.3 | FP/s @0.3 | frames with ≥1 FP @0.3 | with SORT, best "free" setting |
| --- | --- | --- | --- | --- |
| LFE-Peaks | **1.39** | **55.7** | **51.0%** | **1.33** (−4%, −0.1pp AP) |
| LFE-PPN | **3.17** | **127.2** | **64.0%** | stale (1.87 was measured at `nms_radius = 0.8`) |
| **TAKHeLiPeD, stage 2 alone** | **2.62** (sd 0.20) | **105.1** | **67.2%** | — |
| **TAKHeLiPeD, whole detector** | **0.74** (sd 0.14) | **29.7** | **34.1%** | — (the memory is the filter) |
| SpaceTimeCNN | gap | gap | gap | gap |
| FullScanTCN | gap | gap | gap | gap |
| TemporalUNet | gap | gap | gap | gap |

0.3 is LFE-PPN's own published operating point.
The three `gap` rows are the pre-TAKHeLiPeD architectures and stay gaps: they belong to the superseded line of work (`TODO.md` A33-A36), and nothing in it is queued.

**LFE-PPN's row was stale here until 2026-10-09**, at the pre-recalibration **1.94 / 50.9**; the 2026-09-11 recalibration of `nms_radius` from 0.8 m to 0.30 m took it to **3.174 /frame** and `CHANGELOG.md` recorded that, but this table kept the old figure while its *frames-with-a-phantom* cell was updated to the new 64.0%.
The "with SORT" cell is stale for the same reason and is not reconstructed here: `tracker_sweep.py` was re-run (A37), and at `min_hits = 3` it leaves 34.7% of person-free frames with a phantom against the old 47.1%, but no corrected FP/frame was recorded.
**A row that is half-updated is worse than one that is wholly stale**, because nothing about it looks wrong.

### TAKHeLiPeD, measured 2026-10-09 (A50 phase 1 item 7)

Three seeds of the adopted recipe — C1's averaged stage 2 and the E13 memory trained on its candidates — over all 72,533 person-free frames, `utils/empty_scene.py`, `results/empty_scene_s{0,1,2}.json`:

| at threshold 0.3 | stage 2 alone | whole detector | change |
| --- | --- | --- | --- |
| FP/frame | 2.622 (2.834 / 2.587 / 2.446) | **0.740** (0.853 / 0.788 / 0.580) | **−72%** |
| frames with ≥1 | 67.2% | **34.1%** | −33 pp |
| recur within 0.3 m next frame | 84.9% | 92.2% | +7.3 pp |
| FP/frame, the 41,797 frames where nobody has been in view for 10 s | 2.460 | **0.590** | **−76%** |
| FP/frame at threshold 0.9 | 0.872 | **0.010** | −99% |
| frames with ≥1 at threshold 0.9 | 40.5% | **0.9%** | −39.6 pp |

**The memory is a phantom filter, and this is the clearest measurement of it in the project.** Stage 2 alone is *worse* than both published detectors here (2.62 against 1.39 and 3.17 — it is tuned for recall on populated frames); the memory cuts that by 72%, to **below LFE-Peaks**, which no published detector on this split reaches.
At 0.9 confidence the contrast is categorical: the published detectors still hallucinate in a third to a half of person-free frames, and the whole detector does so in **0.9%**.

**It is not just remembering people who have left.** 42.4% of person-free frames sit within 10 s of someone in view, which is inside the slots' fade time, so a detection there can be a true memory of a real person rather than a phantom.
Restricted to the 41,797 frames beyond that horizon the reduction is **larger**, not smaller (−76%), so the filtering is not an artefact of the stale-memory explanation.

**What the memory keeps is the furniture.** Its surviving phantoms recur next frame **92.2%** of the time against the candidates' 84.9% and real people's 99.8%: it removes the unstable ones and keeps the ones that look like a person in every frame, which is the failure mode A41 and A44 could not remove classically.
That is a creation-rule question rather than a scoring one (`TODO.md` §A, Tier 2 item 13).

**The FP/s column is on the real clock** since 2026-10-10: 24.96 ms per frame, 40.06 Hz, from the one shared `frame_period_s`. The whole detector's 29.7 FP/s against LFE-Peaks' 55.7 is the figure a deployment feels; the per-frame column is the one to compare between detectors.
**LFE-Peaks' figure moved from 1.28 to 1.39 on 2026-09-10** when its `merge_radius` was corrected from 0.40 m to 0.30 m — a tighter merge keeps double-reports the looser one was collapsing. The correction is what made its AP reproduce the paper (`CHANGELOG.md`).

**Both published static detectors hallucinate a person in half to two-thirds of frames containing nobody**, and still in a third to a half of them at 0.9 confidence.

**But empty scenes are not what triggers it.** The same detectors, at the same threshold, on the *populated* `official` test recording:

| Model | FP/frame official | FP/frame transferred | miss rate official |
| --- | --- | --- | --- |
| LFE-Peaks | **1.43** | **1.28** | **21.2%** |
| LFE-PPN | **1.98** | **1.94** | **21.1%** |

The rate is within 10% of identical either way.
**The benchmark hides a constant absolute error behind a ratio**: 1.43 phantoms per frame still reads as 63% precision when the frame also holds 3.06 people, and no FROG benchmark frame holds zero people.
Empty scenes do not create the failure, they only stop concealing it.

**AP and false-positive burden rank the two detectors oppositely.**
LFE-PPN wins on AP (69.6% vs 65.1%) while emitting **2.5x** as many phantoms per frame on the same data — AP rewards the recall it holds at high thresholds and never charges for absolute phantom count.
A ranking taken from FROG's published benchmark is not a ranking for deployment.

**The phantoms persist, so they are furniture and not noise**: 88.3% (Peaks) and 77.6% (PPN) recur within 0.3 m in the next frame, against 99.8% for real people.

**And once duplicates are separated out, the two splits agree almost exactly.**
On a populated split a second detection on an already-matched person counts as a false positive; where there are no people, no such duplicate can exist. Removing them:

| | true positives | duplicates | genuine phantoms | phantoms, person-free |
| --- | --- | --- | --- | --- |
| LFE-Peaks | **2.429** | **0.372** | **1.387** | **1.391** |
| LFE-PPN | **2.689** | **1.609** | **3.416** | **3.174** |

The phantom columns agree to **0.3%** (Peaks) and **7.6%** (PPN).

**Re-measured 2026-09-11, and the LFE-PPN row moved entirely.** Two independent causes, both found by chasing a 2x discrepancy between this table and `dataset-properties.md`:

1. **`nms_radius` was recalibrated 0.8 m -> 0.30 m and these tables were never re-run.** On person-free frames that alone takes LFE-PPN from 1.94 to 3.17 phantoms/frame. LFE-Peaks' own 0.40 -> 0.30 move changes its count by 3%, which is why only one row moved.
2. **`phantom_analysis.py` counted true positives as `min(detections-near-a-person, people)`**, which cannot tell one detection each on five people from five detections on one person. Harmless while LFE-PPN emitted 0.004 duplicates per frame; badly wrong at 1.609. It now assigns one-to-one by Hungarian, which makes it agree with `evaluate.py` exactly on LFE-Peaks (1.759 FP/frame against the paper table's 1.76).

**The genuine-phantom column was never affected by the second cause** — a detection near no person at all needs no assignment to identify — so the paper's headline comparison is a correction of magnitude, not of method.

**A standard tracker cannot fix this** (`docs/PAPER.md`). SORT over these same detectors trades phantoms for misses along a monotone curve with no operating point: ~13% phantom reduction costs 1.5pp of AP (LFE-Peaks) or 1.1pp (LFE-PPN) at `min_hits = 3`; at `min_hits = 50` the reductions reach 59% and 82% for **25.0pp and 47.2pp** of AP, and 32.6% / 34.7% of person-free frames still carry a phantom. *(LFE-PPN re-run 2026-09-11 at its recalibrated `nms_radius`; its curve is steeper than before, not shallower.)*

The `gap` rows are ours to fill and are **not** fillable by pointing our current checkpoints at this split: they were selected against a different training set, and A32b holds training data constant before any such comparison counts.

## Speed — ms/frame, single scan, CPU unless marked

All of these are ours by necessity: `ms/frame` only compares when every row came off one machine.
Unaffected by the 2026-09-10 accuracy fixes — timing does not depend on which frames were in which split.
The paper's own figures in parentheses are **GPU** numbers and are not comparable to this column; they are here to show the hardware gap, not to rank against.

| Model | DROW (450 beams) | FROG (720 beams) | JRDB (541 beams) |
| --- | --- | --- | --- |
| AlgorithmicDetector | **0.04 ms** | **0.06 ms** | **0.06 ms** |
| DrowDetector (DROW3) | **203.3 ms** | **276.4 ms** *(paper, GPU, T=1: 13.08 ms)* | **235.0 ms** |
| DrSpaamDetector (DR-SPAAM) ¹⁰ | **201.5 ms** | **350.4 ms** *(paper, GPU: 13.95/13.99 ms)* | **298.3 ms** |
| Li2FormerDetector | **431.7 ms** *(paper, GPU: 14.95 ms)* | **590.6 ms** | **498.3 ms** *(paper, GPU: 17.54 ms)* |
| LFEPeaksDetector | **1.07 ms** | **1.08 ms** *(paper, GPU: 1.76 ms)* | re-measure⁵ |
| LFEPPNDetector | **1.46 ms**⁴ | **1.56 ms**⁴ *(paper, GPU: 1.49 ms)* | re-measure⁵ |
| PeTra | — | 28.17 ms *(paper, GPU)* | — |
| **SpaceTimeCNN** | **8.7 ms** | **12.1 ms** | **12.5 ms** |
| **FullScanTCN** | **2.1 ms** | **2.6 ms** | **3.1 ms** |
| **TemporalUNet** | **2.0 ms** | **2.7 ms** | **2.3 ms** |
| **TAKHeLiPeD, stages 1-2 + decode** | n/a | **~10.4 ms**⁶ | n/a |
| **TAKHeLiPeD, stage 3 alone** (object memory) | n/a | **4.4-4.6 ms**⁷ | n/a |
| **TAKHeLiPeD, whole detector streaming** (A60 recipe and memory) | n/a | **20.8 ms** *(4 threads, p99 24.1)*⁹ *(old recipe 19.5 ms, GPU 12.9 ms⁸)* | n/a |

¹⁰ **The windowed form**, which encodes five scans per frame; the published streaming detector encodes one and attends to its
carried template, so this row overstates it. Re-timed under one protocol in `TODO.md` §A item 29.

⁴ Was **315.5 / 243.8 / 278.3 ms** here until 2026-09-10, and blamed on the Python decode loop.
It was not the loop: the decoder applied a second sigmoid to a channel `lfe_ppn.onnx` had already sigmoided, so every one of the 3,600 anchors cleared any threshold and entered an O(n²) greedy merge, 182 survivors per frame on scenes holding ~3 people.
With that fixed LFE-PPN lands within 0.1 ms of its paper's **GPU** figure, on CPU — and `TODO.md` A5 (vectorize the decode) is dissolved rather than done.

⁵ The old JRDB figures were measured with that same bug and are withdrawn rather than corrected; JRDB is not downloaded here (`gotchas.md`), so nothing can replace them yet.
Both LFE rows are otherwise real-data measurements (300 scans each) taken after the fix, so they stay methodologically comparable to each other.

⁶ Derived 2026-09-16 from step 3a's candidate caches, which stream the trained stages 1-2 one frame at a time (batch 1) and decode candidates with DROW's vote grid: 1,269 s for 120,396 train frames, 2,010 s for 195,058 val, 521 s for 50,088 test -- 10.3-10.5 ms per frame, consistently.
**Not comparable with the rows above**: those are CPU measurements on this machine, while this one runs on the RX 9060 XT under ROCm, carries a Python decode loop, and leaves out stage 3 (the object memory).
It is quoted only against the sensor: FROG's measured mean frame period is 24.99 ms (~40 Hz, corrected 2026-09-17 from a long-standing 26.2 Hz), so stages 1-2 plus the memory's 4.4 ms still stream in real time, with roughly a third of the budget spare. A proper single-frame latency benchmark of the whole detector, on one machine with the other rows, is `TODO.md` A50 phase 1.

⁷ Measured 2026-09-17 and confirmed across B0's three seeds (2026-09-19) by `memory_ms_per_frame`, timing one stream on the RX 9060 XT under ROCm: **4.42 ms** on the fixed end-of-run block, 4.5-4.6 ms across the seeds. Same caveat as footnote 6 -- not comparable with the CPU rows above.

⁹ `TODO.md` A60 / A59 phase 3a, 2026-10-08: E13's stage 3 (no attention) on the no-temporal stage 2 trained with the adopted recipe -- `nms-mean` decoding, mirror augmentation, nearest-person labels, dropout 0.1, weight averaging (stage 2 alone 82.20%) -- three seeds, every 5th annotated test frame: **86.35 / 85.77 / 85.49%, mean 85.87**, AP@0.3 84.68%. Against E13 on the old recipe, 83.86%, +2.01. Stage 2 alone is the no-temporal network averaged over its val-chosen epochs, 82.28 / 82.08 / 82.22% (mean 82.20). Streamed one scan at a time on this machine's CPU at 4 threads (`tracking_baselines.py latency`, 2,000 frames, 2026-10-08): stage 2 8.75 ms, decoding 2.77 ms (`nms-mean`'s per-voter loop, ~1 ms more than the vote grid), memory 9.22 ms, **20.79 ms median, p99 24.08** -- inside FROG's 25 ms period with under a millisecond to spare at p99. The A56 tracking comparison above is on the old recipe's candidates and has not been repeated.

⁸ `TODO.md` A56 E22, 2026-10-03: `tracking_baselines.py latency`, one raw scan at a time through stage 2, DROW vote decoding and the E13 memory, 2,000 frames of FROG test, median (p99). CPU on this machine with 4 threads: **8.60 + 1.76 + 9.12 = 19.53 ms (22.38)**, 8 threads 19.47 ms (22.77); RX 9060 XT under ROCm: **5.76 + 2.27 + 4.82 = 12.92 ms (15.11)**. The CPU figure is the one comparable with this column, and it fits FROG's 25 ms scan period at the 99th percentile. The memory's share here includes moving each frame's tensors from numpy, which footnote 7 leaves out.

**Exact parameter counts**, counted from the ONNX initializers now that `onnx` is installed, replacing the file-size estimates this table used to carry (~65K / ~180K): **LFE-Peaks 53,258**, **LFE-PPN 170,903**.
For scale, `SpaceTimeCNN` is 476,262 — **2.8x LFE-PPN, for +0.5pp on the benchmark and no use of temporal order at all.**

## Where the other tables went

- **The `(T, dtime)` informational-capacity grid** now lives with its own methodology, in [`informational-capacity-proxy.md`](informational-capacity-proxy.md).
- **Per-frame ego-motion and person-displacement measurements** are dataset properties, not model results — [`data-model.md`](data-model.md).
- **The `dtime`, NMS, operating-threshold and tracker sweeps** were all measured on the contaminated split and are retracted.
  Their numbers and the reasoning that produced them stay in [`../CHANGELOG.md`](../CHANGELOG.md); they are not settings to restore, they are runs to redo.
