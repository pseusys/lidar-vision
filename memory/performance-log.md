# Performance log

*keywords:* wp-AUC, AP, ms/frame, SOTA, comparison table, published baseline, PeTra, Li2Former, LFE, DR-SPAAM, DROW3, state of the art

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
| AlgorithmicDetector | ours, rule-based | **F1 55.7%** | **F1 1.7%** | n/a |
| PeTra | published | gap | 50.1% *(PeTra\* 59.0%)* | gap |
| LFE-Peaks | published | gap¹ | 65.6% | gap¹ |
| LFE-PPN | published | gap¹ | 69.2% *(ours 69.6%)* | gap¹ |
| DROW3 | published | 61.9% | 73.9% | 82.9% *(official repo README, merged 1091-point 360° scans)* |
| DR-SPAAM | published | 69.6% *(RA-L'22: 72%+)* | **75.6%** *(`T=5`)* / 73.7% *(`T=1`)* | 84.9% *(official repo README, merged 1091-point 360° scans)* |
| Li2Former | published | 76.4% | gap² | 81.4% |
| **SpaceTimeCNN** | ours, novel | re-measuring | re-measuring³ | n/a |
| **FullScanTCN** | ours, novel | re-measuring | re-measuring | n/a |
| **TemporalUNet** | ours, novel | re-measuring | re-measuring | n/a |
| **Three-horizon detector, stages 1-2** | ours, novel | gap | **76.9%**⁴ | n/a |
| **Three-horizon detector, stages 1-3** | ours, novel | gap | **80.1%**⁵ | n/a |

**DR-SPAAM `T=5` at 75.6% on FROG is the number to beat.**

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

¹ The LFE paper only ever evaluates on FROG's 720-beam scans.
This project *has* run LFE cross-dataset by zero-padding shorter scans to 720 (DROW: 24.0% / 11.7%), but that padding is our own invention, is not described in the paper, and does not resample to FROG's 0.25°/beam angular resolution — beam index stops meaning the same physical angle.
Degraded by construction, so it is recorded here as a footnote rather than reported as a result.
² Li2Former was never evaluated on FROG by its authors, and training it here has never completed (DirectML crashes, impractical CPU-only speed — `gotchas.md` I4).
³ In progress: `T=5, dtime=10` on the fixed pipeline (`TODO.md` A1).
⁴ `TODO.md` A43 step 2, measured 2026-09-15: one run, weights of epoch 1.0 selected on val AP, every annotated frame of the test recording; **75.6%** with each clip's history shuffled; 1.91 false positives and 0.39 missed people per frame at 0.3. No object memory yet (stage 3). 1.3 pp above DR-SPAAM's published `T=5` is ours measured against theirs published, on one seed -- inside the range where a harness difference or a second seed could reverse it (`interpreting-evaluation.md`). A continuation from these weights with a plateau-decayed learning rate did not improve val AP (73.0% best, then falling to 69.2% by epoch 3.25) and tested at 76.9% / shuffled 75.6% with 1.43 FP / 0.49 FN, so these remain the stage 1-2 weights. Scored before the vote-grid NaN fix of 2026-09-15; re-scored without the 2 non-finite detections among 800,856 it is 76.9218% instead of 76.9227%.
⁵ `TODO.md` A43 step 3a, measured 2026-09-15: stage 3 (object memory, 256 slots) trained on stage 2's cached candidates with stage 2 frozen, one run; checkpoint selected on val AP (30 s chunks, epoch 5, val 77.32% against 73.19% for its candidates). Every test recording played from empty memory, every annotated frame: **80.10%**, with 1.29 false positives and 0.47 missed people per frame at 0.3, against 76.92% / 1.91 / 0.39 for stage 2's candidates on the same frames (identical to step 2's own test result) -- an exchange rate of 7.6 false positives removed per person lost. Every 5th annotated frame: 80.15%. At the 0.3 operating point its precision is 66.8% and recall 84.7%, against 58.4% and 87.3% for the candidates alone -- the memory buys 8 points of precision for 2.6 of recall. **That trade is an artefact of scoring both stages at 0.3, corrected 2026-09-16** (`utils/memory_diagnosis.py`): the rescore head shifts stage 3's whole score distribution downward, so a threshold calibrated for stage 2's distribution understates stage 3. Scored at its own operating point, stage 3 **strictly dominates its input** — at 0.22 it covers 135,520 of 153,655 annotated people against stage 2's 134,196 *and* emits 1.767 false positives per frame against 1.910, better on both axes; at 0.20, where false positives match (1.925 against 1.910), it covers 136,643, or 2,447 more people. AP, being threshold-free, already reflected this; only the operating-point comparison was misleading. SORT replayed on the same candidates loses AP at every setting (76.7% / 75.7% / 74.9% at `min_hits` 1/3/5), so the gain is the learned memory, not temporal filtering as such; 64 slots score within 0.1 points of 256 (`CHANGELOG.md` 2026-09-16). Joint training with feedback (step 3b, 3 epochs, both models at 1e-4) did **not** improve on this: 78.84% test AP with 1.24 FP / 0.55 FN, and its stage 2 came out at 75.68%, below the 76.92% it started from -- so these frozen-stage-2 weights stay the best model. Against DR-SPAAM's published `T=5` it is ours measured against theirs published, on one seed.

### Reading our FROG row against a published one

Two things still keep the FROG comparison from being perfectly clean once our number lands, and the second does not flatter us:

- ~~Training-set size.~~ **Resolved**: we train on the paper's own `split == 0` frames, **108,356**, exactly theirs (`data-model.md`).
- Our val is three held-out recordings; theirs is an interleaved per-frame holdout inside the training data.
  A different — and stricter — model-selection signal, on the same test set.
  This does not flatter us either way; it just means our early stopping optimises for cross-recording generalization rather than for interpolation.
- Training budget differs hugely, **in our favour**: the FROG authors trained DROW3/DR-SPAAM for exactly 5 epochs at batch 8 (~67,700 minibatches), deliberately matched to DROW's scale.
  A 100-epoch run at batch 4 over 102,356 frames is up to ~2.56M.
  Beating them on ~38x the compute is a weaker claim than beating them at parity, and must be reported as such.

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

## Empty-scene false positives — FROG `transferred`, person-free frames only

**A genuine gap: nobody has published this, because the FROG benchmark contains no person-free frames at all.**
All 72,533 of them, from the three raw recordings, 46.1 minutes of real time at 26.2 Hz.
wp-AUC is *undefined* here, not merely low — no true positives means precision 0 at every threshold and recall with no denominator (`interpreting-evaluation.md`).
Every detection is a false positive by construction, so the metric is a count.

| Model | FP/frame @0.3 | FP/s @0.3 | frames with ≥1 FP @0.3 | with SORT, best "free" setting |
| --- | --- | --- | --- | --- |
| LFE-Peaks | **1.39** | **36.4** | **51.0%** | **1.33** (−4%, −0.1pp AP) |
| LFE-PPN | **1.94** | **50.9** | **64.0%** | **1.87** (−4%, −0.5pp AP) |
| SpaceTimeCNN | gap | gap | gap | gap |
| FullScanTCN | gap | gap | gap | gap |
| TemporalUNet | gap | gap | gap | gap |

0.3 is LFE-PPN's own published operating point.
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
| DrSpaamDetector (DR-SPAAM) | **201.5 ms** | **350.4 ms** *(paper, GPU: 13.95/13.99 ms)* | **298.3 ms** |
| Li2FormerDetector | **431.7 ms** *(paper, GPU: 14.95 ms)* | **590.6 ms** | **498.3 ms** *(paper, GPU: 17.54 ms)* |
| LFEPeaksDetector | **1.07 ms** | **1.08 ms** *(paper, GPU: 1.76 ms)* | re-measure⁵ |
| LFEPPNDetector | **1.46 ms**⁴ | **1.56 ms**⁴ *(paper, GPU: 1.49 ms)* | re-measure⁵ |
| PeTra | — | 28.17 ms *(paper, GPU)* | — |
| **SpaceTimeCNN** | **8.7 ms** | **12.1 ms** | **12.5 ms** |
| **FullScanTCN** | **2.1 ms** | **2.6 ms** | **3.1 ms** |
| **TemporalUNet** | **2.0 ms** | **2.7 ms** | **2.3 ms** |
| **Three-horizon detector, stages 1-2 + decode** | n/a | **~10.4 ms**⁶ | n/a |

⁴ Was **315.5 / 243.8 / 278.3 ms** here until 2026-09-10, and blamed on the Python decode loop.
It was not the loop: the decoder applied a second sigmoid to a channel `lfe_ppn.onnx` had already sigmoided, so every one of the 3,600 anchors cleared any threshold and entered an O(n²) greedy merge, 182 survivors per frame on scenes holding ~3 people.
With that fixed LFE-PPN lands within 0.1 ms of its paper's **GPU** figure, on CPU — and `TODO.md` A5 (vectorize the decode) is dissolved rather than done.

⁵ The old JRDB figures were measured with that same bug and are withdrawn rather than corrected; JRDB is not downloaded here (`gotchas.md`), so nothing can replace them yet.
Both LFE rows are otherwise real-data measurements (300 scans each) taken after the fix, so they stay methodologically comparable to each other.

⁶ Derived 2026-09-16 from step 3a's candidate caches, which stream the trained stages 1-2 one frame at a time (batch 1) and decode candidates with DROW's vote grid: 1,269 s for 120,396 train frames, 2,010 s for 195,058 val, 521 s for 50,088 test -- 10.3-10.5 ms per frame, consistently.
**Not comparable with the rows above**: those are CPU measurements on this machine, while this one runs on the RX 9060 XT under ROCm, carries a Python decode loop, and leaves out stage 3 (the object memory).
It is quoted only against the sensor: FROG delivers a scan every 38 ms at 26.2 Hz, so this sizing streams in real time with room to spare. A proper single-frame latency benchmark of the whole detector, on one machine with the other rows, is `TODO.md` A50 phase 1.

**Exact parameter counts**, counted from the ONNX initializers now that `onnx` is installed, replacing the file-size estimates this table used to carry (~65K / ~180K): **LFE-Peaks 53,258**, **LFE-PPN 170,903**.
For scale, `SpaceTimeCNN` is 476,262 — **2.8x LFE-PPN, for +0.5pp on the benchmark and no use of temporal order at all.**

## Where the other tables went

- **The `(T, dtime)` informational-capacity grid** now lives with its own methodology, in [`informational-capacity-proxy.md`](informational-capacity-proxy.md).
- **Per-frame ego-motion and person-displacement measurements** are dataset properties, not model results — [`data-model.md`](data-model.md).
- **The `dtime`, NMS, operating-threshold and tracker sweeps** were all measured on the contaminated split and are retracted.
  Their numbers and the reasoning that produced them stay in [`../CHANGELOG.md`](../CHANGELOG.md); they are not settings to restore, they are runs to redo.
