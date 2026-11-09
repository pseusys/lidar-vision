# Performance log

*keywords:* wp-AUC, AP, ms/frame, SOTA, comparison table, published baseline, PeTra, Li2Former, LFE, DR-SPAAM, DROW3, state of the art, Aycard, AlgorithmicDetector, attribution, credit, history ablation, shuffled frozen zeroed, five-frame averager, PeTra never run here

The standing comparison: published state of the art against this project's own architectures, **one row per model, latest measurement only**.

**History does not live here.** How a number moved, which run produced it, and every sweep that led to a setting are in [`../CHANGELOG.md`](../CHANGELOG.md) — grep its `*keywords:*` lines.
Read [`interpreting-evaluation.md`](interpreting-evaluation.md) before quoting any row; several are not comparable to each other without its caveats.

## How a cell gets filled

**A cell holds the original authors' published number wherever they published one** (owner's call, 2026-09-10).
This project evaluates a third-party model only to **fill a gap** — a model/dataset pair nobody published — never to re-derive a number that already exists.
Re-deriving a published accuracy figure through this pipeline adds risk without adding information; this session's own audit is the case in point.

**Speed is the exception.** `ms/frame` is only comparable when every row was measured on one machine **and under one protocol**, so every timing number here is ours by necessity. The second half of that was only true from 2026-10-11: before it, the table's rows came from two scripts at two thread counts on synthetic input.

| notation | meaning |
| --- | --- |
| plain `71.7%` | published by that model's own authors — the yardstick, never re-run here. **It carries the configuration it was published at**, because two cells of this table were found in 2026-11-04 to name a figure no primary source holds |
| **bold** | measured by this project: our own architectures, or a gap the literature does not cover |
| `gap` | nobody published it and we have not measured it |
| `check paper` | the authors did publish one; it is simply not transcribed here yet. **Look it up before spending a run.** |
| `n/a` | not measurable here — JRDB needs manual registration and is not downloaded (`gotchas.md`) |

## Accuracy — person class, AP/AUC @ 0.5 m match radius (F1 where noted)

| Model | Kind | DROW | FROG | JRDB |
| --- | --- | --- | --- | --- |
| ROS `leg_detector` | classical | check paper | 20.2% | check paper |
| AlgorithmicDetector | rule-based, **O. Aycard**⁰ | **F1 55.7%** | **F1 1.7%** | n/a |
| PeTra | published | gap | 50.1% *(PeTra\* 59.0%)*¹³ | gap |
| LFE-Peaks | published | gap¹ | 65.6% | gap¹ |
| LFE-PPN | published | gap¹ | 69.2% *(ours 69.6%)* | gap¹ |
| DROW3 | published | 71.7% *(the "DROW3x + validation" curve, which is the configuration our downloaded weights are; 68.1% train-only)* | 73.9% | 82.9% *(official repo README, merged 1091-point 360° scans)* |
| DR-SPAAM | published | 70.3% *(paper, Table I; the later repository reports 72.3% for a **retrained** model -- this cell is uncited until the provenance audit in `TODO.md` settles which)* | **75.6%** *(`T=5`)* / 73.7% *(`T=1`)* | 84.9% *(official repo README, merged 1091-point 360° scans)* |
| Li2Former | published | 76.4% *(Li2Former-A, `T=5`; 74.8% without the AFM; 72.5% at `d=0.3`)* | gap² | 81.4% |
| **SpaceTimeCNN** | ours, superseded | not measured | **63.0%**³ | n/a |
| **FullScanTCN**, **TemporalUNet** | ours, **retired 2026-10-31** | not measured | not measured | n/a |
| **TAKHeLiPeD, stages 1-2** (calibration network) | ours, novel | **29.5%**¹⁴ | **82.8%**¹² *(superseded recipe 82.2%⁹; old recipe 77.7%⁴)* | n/a |
| **TAKHeLiPeD, stages 1-3** (+ object memory) | ours, novel | **46.0%**¹⁴ | **86.3%**¹² *(superseded architecture 86.0%¹¹; with the coasting head 85.9%⁹; old recipe 83.3%⁵)* | n/a |

**DR-SPAAM `T=5` at 75.6% on FROG was the number to beat, and the frozen detector clears it by 10.7 pp.**
That margin was **7.7 pp** while this file still led with the old recipe's 83.3%, and the sentence outlived the number by three weeks -- which is the failure the stale-number audit existed to fix and the reason each cell here now carries the configuration it was measured on.
That is ours measured here, three seeds, against theirs published, one paper figure — read footnote 11 and [`interpreting-evaluation.md`](interpreting-evaluation.md) before quoting it.

**The comparison a tracking reviewer will ask for is within one model, on identical candidates** (2026-10-09, on the adopted recipe; one replay per stage-2 seed; FROG test, every 5th annotated frame, 95% block-bootstrap intervals per seed; tables in [`experiments/2026-09-tracking-claims.md`](experiments/2026-09-tracking-claims.md)):

| what reads TAKHeLiPeD's stage-2 candidates | AP, 3 seeds | vs the candidates |
| --- | --- | --- |
| nothing | 82.80% | -- |
| SORT (`min_hits` 1) / ByteTrack, as published | 82.85% / 81.46% | +0.05 / -1.34 |
| the memory's own fade/coast rules, applied after the threshold | 74.00% | -8.80, every seed far below zero |
| ByteTrack + CBMOT track confidence, tuned on val | 84.13% | +1.33, every seed's interval above zero |
| **object memory, no attention (E13), frozen** | **86.29%** | **+3.49; +2.16 over the tuned tracker (+1.98 / +2.34 / +2.16), every seed's interval above zero** |

**Re-measured 2026-10-31 on the frozen candidates; the rows above are no longer the superseded stage 2's.** The
previous line read 82.19 / 82.31 / 81.17 / 83.51 / 86.01% with the memory +2.36 over the tuned tracker. GOSPA moved with
it: the memory 0.435 against its candidates' 0.481, the tuned tracker 0.458. First-detection latency is new in this
table, because `report_focus` used to exclude any arm not labelled `E13:` -- the memory finds a person on their **first**
frame 92.78% of the time against the candidates' 90.48% and the tuned tracker's 80.56%.

At one matched false-positive rate the tuned tracker is marginally ahead (95.39 against the memory's 95.17%); the memory's lead is ranking, not recall at a threshold. On the old recipe (2026-10-03, vote-grid candidates) the rows read 77.66 / 77.40 / 76.49 / 80.96 / 83.86%, the memory +2.90 over the tuned tracker. All on FROG's official test labels, which include 2,254 duplicate circles that only a second detection can match; on the old recipe, without them every row rose 0.7-1.05 pp and the memory's margin over the tuned tracker was +2.71, still above zero at every seed ([`interpreting-evaluation.md`](interpreting-evaluation.md), 2026-10-05).

**TAKHeLiPeD** (`T`emporal `A`daptive `K`nee-`He`ight `Li`dar `Pe`rson `D`etector, named 2026-09-23) is what earlier entries in this file, `CHANGELOG.md` and `TODO.md` call *the three-horizon detector*. **The code followed on 2026-10-11**: the package, the C++ namespace, the ROS package and every script say `takheliped`. Only `checkpoints_three_horizon/` keeps the old name, deliberately -- each checkpoint stores the absolute path of its own stage 2.

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

¹⁴ **DROW-native, one seed, 2026-11-05** (`CHANGELOG.md`), trained and scored at DROW's own **29.99 m** sensor limit rather than FROG's 10 m, on DROW test's 2,428 annotated frames. **Read it as a floor, not a competitive number**: it is 24.3 pp below the weakest published DROW baseline here, which was the expectation recorded before the run. What it establishes is the memory's contribution -- **+22.56 pp over its own candidates against +3.49 pp on FROG, 6.5x** -- and at matched false positives (its own threshold 0.1358, 7.699 FP/frame) it reaches **93.88% recall against the candidates' 90.38%**, so the recall it appears to lose at the shared 0.3 threshold is decalibration. Two cautions: the 29.5% stage-2 cell is its **best** checkpoint through stage 2's **own** decoder, while the 46.0% row's internal baseline is the **averaged** checkpoint through the NMS candidate decoder (23.45%), so the two cells are not a 16 pp gain; and DROW annotates in bursts of 20 over 7.5 s with 24 s gaps that exceed the slots' 10 s fade, so no gradient crossed a gap and the figure measures this memory over 7.5 s windows (`dataset-properties.md`). **The 29.5% cell is the headline recipe's, not the best stage-2 AP on record for DROW**: a multi-class sigmoid head reached 32.10% on 2026-11-07, one seed and one confound away from being a figure, and a 10 m-trained arm reaches 36.36% on the labels inside 10 m. Both are below in "Four levers on DROW's 20.5 pp gap"; neither has a whole-detector number yet, which is why this row is still the chain's.

¹³ **PeTra is cited only, and that is not a failed attempt.** There is no PeTra implementation and no PeTra weights in this repository -- only its published figures, including the 28.17 ms GPU timing -- so it has never run here and cannot be timed under this project's CPU protocol. Filling either `gap` cell would mean reimplementing it from the paper, which is the thing that produced an untrustworthy number once already (footnote 2). Moved here from `TODO.md` on 2026-11-05, so the question stops returning as a queue item.
¹ The LFE paper only ever evaluates on FROG's 720-beam scans.
This project *has* run LFE cross-dataset by zero-padding shorter scans to 720 (DROW: 24.0% / 11.7%), but that padding is our own invention, is not described in the paper, and does not resample to FROG's 0.25°/beam angular resolution — beam index stops meaning the same physical angle.
Degraded by construction, so it is recorded here as a footnote rather than reported as a result.
² Li2Former was never evaluated on FROG by its authors. **The rest of this footnote was wrong and is corrected (2026-11-01).** Training does complete here: `AGENTS.md` I4's DirectML premise predates this host having ROCm, and on ROCm it trains at 2.09 GiB and 259 ms/frame. What blocks the cell is not the hardware but the model -- `Li2FormerDetector` is **this project's reading of the paper's prose, not the paper's architecture**, so a number from it would not be Li2Former's. The authors do publish code (`ai-winter/Li2Former-v1`, GPL-3.0, **no weights and no training recipe**), and `OfficialLi2FormerDetector` now wraps it; the cell stays `gap` until that port reproduces their DROW figure, which is the only check available without weights to match. **It does not reproduce it, and that is now measured (2026-11-03): 48.1% on DROW test at `d = 0.5` and 40.5% at `d = 0.3`, against their 76.4% and 72.5%** -- see the reproduction section below for the four-cell check that rules the harness out of the gap. So the FROG cell stays `gap`, now for a stated reason rather than a pending one: a FROG number from a port that misses their own DROW figure by 28 pp would not be Li2Former's either, and filling the cell would need a recipe that reproduces DROW first.
³ **63.0%** single-frame, the figure the acceptance criteria were written against and `docs/PAPER.md` argues from. **The ablation it comes from, in full** (2026-09-10, moved here from `TODO.md` on 2026-11-05 when that queue's closed sections were deleted): four history ablations of one epoch-1 checkpoint, identical weights, the whole `frog_16-41` test recording, 50,088 frames.

| history | wp-AUC | vs full | what the step removes |
| --- | --- | --- | --- |
| full | **67.1%** | -- | -- |
| shuffled | **66.6%** | **-0.5** | ordering and direction, i.e. **motion** |
| frozen | **63.0%** | -4.1 | the independent observations, i.e. **aggregation** |
| zeroed | 42.0% | -25.1 | nothing further -- frozen and zeroed carry the same information, so the extra 21 pp is pure out-of-distribution damage |

**The temporal window was worth 4.1 pp and essentially none of it was motion**: destroying the ordering cost 0.5 pp, removing the independent looks 3.6 pp. `SpaceTimeCNN` was a five-frame **averager**, and a 476K-parameter five-frame model beat a 180K single-frame one by 0.6 pp by averaging rather than by seeing motion. That is the measurement that redirected the project away from short temporal windows, and it is why TAKHeLiPeD's memory is scored over tens of seconds rather than five frames. These three are the project's own first line of work, superseded by TAKHeLiPeD and trained on nothing since 2026-09; `SpaceTimeCNN` is kept so the decisions made on it stay reproducible and removes the other two. The `T=5, dtime=10` re-measurement this footnote used to promise was never run and is not planned: a temporal window is not what the project proposes any more.
⁴ The slot study's backbone, measured 2026-09-19 over the whole FROG `official` test split: **77.66%**, the no-temporal stage 2 (seed 0) whose decoded candidates the object memory in the row below rescores, at 2.620 false positives per frame and 88.8% recall at the 0.3 operating point.
That backbone's own three-seed mean is **77.23%**; the default architecture, which kept the coarse temporal convolution (removed 2026-10-08), means **77.92%** over three seeds. The pooled per-run spread across five three-seed arms is ~0.95 pp (`interpreting-evaluation.md`), so those two are a tie, not a ranking.
Both are scored under the limit-beam decoding rule and the running time step adopted 2026-09-17.
**The vote-grid decoder understates the network**: the same no-temporal weights through `nms-mean`, NMS chosen on val on all six checkpoints, score **80.07%** (three seeds, every 5th annotated frame; default architecture 79.61%). Mirror augmentation lifts it further, to **81.49%**, and the whole adopted recipe with dropout 0.1 and weight averaging to **82.20%** (three seeds each, every 5th annotated frame). Neither is this row's figure yet, because every stage-3 result rests on vote-grid candidates and moves only once stage 3 is retrained on the new ones.
**The 76.9% this row used to carry is retired, not corrected**: a different backbone, measured before both of those changes (`CHANGELOG.md`, 2026-09-15 to 2026-09-17).
⁵ The slot study's B0 baseline, measured 2026-09-19: stage 3 (object memory, 256 slots) trained by `step3a --chunk-s 1 10` on footnote 4's cached candidates with stage 2 frozen, **three seeds**; every test recording played from empty memory, every annotated frame.
Test AP **82.96 / 83.17 / 83.88%** (mean **83.34**, sd **0.47**), AP at 0.3 m 81.36 / 81.52 / 82.17%, and recall at the candidates' own false-positive rate 92.2 / 92.5 / 92.7% against 88.8% -- **+5.68 pp over its own input**.
On the best seed at the shared 0.3 threshold it covers 128,895 of 153,655 annotated people at **1.092 false positives per frame**, against the candidates' 136,417 at 2.628 (`a51_b0_memory_diagnosis.json`, 2026-09-21); precision there is 70.2%.
The screening threshold is set from this spread at **~1.5 pp**, and no design arm has cleared it in either direction yet (T1 +1.05, D2 +0.60, D3 -0.90, D1 -1.32, one seed each).
**What the gain is not.** SORT replayed on the same candidates loses AP at every setting, so it is the learned memory rather than temporal filtering as such; and capacity is not the lever -- the memory uses a median of 24 of its 256 slots (107 at most), and in the recovery diagnosis of 2026-09-22, of the 26,996 people it does not report at 0.3, **95.5% already have a live slot on them** at a median value of 0.986. The remaining loss is in the reporting head.
**The 80.10% this row used to carry is retired, not corrected**: a different backbone, and it predates the running time step and the limit-beam decoding. Joint training with feedback (step 3b) reached 78.84% on that older chain and has not been repeated on this one, so training the memory on a frozen stage 2 remains the best configuration.

### Reading our FROG row against a published one

Two things still keep the FROG comparison from being perfectly clean once our number lands, and the second does not flatter us:

- ~~Training-set size.~~ **Resolved**: we train on the paper's own `split == 0` frames, **108,356**, exactly theirs (`data-model.md`).
- Our val is three held-out recordings; theirs is an interleaved per-frame holdout inside the training data.
  A different — and stricter — model-selection signal, on the same test set.
  This does not flatter us either way; it just means our early stopping optimises for cross-recording generalization rather than for interpolation.
- Training budget: the FROG authors trained DROW3/DR-SPAAM for exactly 5 epochs at batch 8 (~67,700 minibatches, 541k scans seen), deliberately matched to DROW's scale, and LFE for up to 100 epochs.
  **TAKHeLiPeD's stage 2 sees fewer scans than any of them, not more**: its selected checkpoint sits at epoch ~1 at batch 32, ~3,400 minibatches and 108k scans, and its test AP *falls* with longer training (77.60% at epoch 1 to 72.36% at epoch 4).
  The "~38x the compute, in our favour" this bullet used to carry described the retired full-scan architectures' 100-epoch runs and does not apply to the TAKHeLiPeD rows.

The metric is not a caveat: `wp-AUC` and the papers' `AP` are the same trapezoidal-AUC-over-a-Hungarian-matched-PR-curve quantity, verified by direct source comparison against the official DR-SPAAM repo (`interpreting-evaluation.md`).
Both sides now score the same `frog_16-41` test recording, which was not true before 2026-09-10.

### What this project still has to measure

Much less than it looked.
Under the policy above, the standing work is:

- **our three architectures**, on DROW and FROG — nobody else can publish these;
- **the speed table**, all of it, because one machine and one protocol are the only way those rows compare;
- **genuine gaps**, and only if they are worth the GPU time: Li2Former on FROG, PeTra anywhere outside FROG, LFE outside FROG (and that one is degraded by construction, so probably not worth it).

Everything marked `check paper` should be transcribed from the paper, not measured.

### Our reproductions are a harness self-test, not a result

This project *has* independently reproduced several published numbers, and those reproductions stay useful for one specific purpose: **checking the evaluation harness against a known answer.**
On DROW the pipeline **undershoots**, and that is the correction of 2026-11-04: DROW3 **66.5%** against the **71.7%**
its own weights were published at, DR-SPAAM **69.08%** (streaming) against the **70.3%** of the paper those weights
come from.

**The `61.9%` this file carried for DROW3 was never a published figure, and the defect was the citation, not the
harness.** `library/setup.py` downloads `final-WNet3xLF2p-T5-odom.rot-trainval-50ep.pth.tar`, and the filename states
the configuration: late fusion, five scans, **odometry rotation correction**, trained on **train + validation**. That
is exactly the "DROW3x + validation" curve of Beyer et al.'s Fig. 4 — person class, test split, 0.5 m radius, the
same quantity `_prec_rec_2d` computes — published at **71.7% AUC** (peak-F1 70.5, EER 69.5); the same paper's
train-only model is **68.1%**. **61.9% matches no row of any primary source.** The nearest values anywhere are 61.5
and 61.4, and those are the *peak-F1* and *EER* — not the AUC — of the `T=1, no time` configuration, which is not the
model we run.

**So the overshoot rule never fired on DROW.** What was logged for two months as +4.6 pp *over* a published figure is
**−5.2 pp under** one, and an undershoot has innocent explanations: the merge radius we had to choose, the decoding,
the frame sample. The direction is what matters for reading the DROW column: every detector in it goes through one
pipeline that appears to cost 1-5 pp, so **our own DROW numbers are depressed by the same amount and the comparison
between them stays fair**, while any absolute DROW figure taken from here is a floor rather than a match.

**DR-SPAAM's 1.5 pp undershoot was mostly our own wrapper, found 2026-10-10**. The official detector is
*stateful*: it keeps its spatial-attention template between calls and clears it only at a sequence boundary. Ours re-encoded a
`T`-scan window from scratch per frame and seeded the template from that window's oldest scan, which is a different detector.
Scored both ways on DROW test with the published weights, through one decoder and one metric: **windowed 67.84%, streaming
69.08%** — so the published algorithm is **+1.24 pp** better than what we were calling DR-SPAAM, and our reproduction of it
moves from −1.5 pp to **−0.52 pp** of the authors' 69.6%. **That reference is itself uncited, found 2026-11-04 beside
the DROW3 correction above**: 69.6% matches no row of the DR-SPAAM paper, whose Table I reports **70.3** AP at 0.5 m
on the test split, nor of the later `2D_lidar_person_detection` repository, which reports **72.3** for a retrained
model. Our weights are `dr_spaam_e40.pth` and our wrapper uses the later release's 56-point cutouts, so the
reproduction gap is **−1.2 pp** or **−3.2 pp** depending on which is the right row — not −0.52 pp. Until the
provenance audit queued in `TODO.md` settles it, treat the DR-SPAAM DROW cell as uncited. **Every DR-SPAAM accuracy figure in this file is the windowed form and
understates the baseline by about that much**, on all three datasets.

**Its cost is misreported in the opposite direction.** The windowed form runs the first two conv blocks five times per frame
where the published one runs them once and attends to its carried template, so the **1,195.88 ms** FROG row below *overstates* the
published detector — we have been reporting DR-SPAAM as both less accurate and slower than it is. The corrected figure is part of
the one-protocol timing sweep, which now has `DrSpaamDetector.stream()` to time.

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
**LFE-PPN now overshoots by 0.4pp**, having previously undershot by 0.6. That is inside the harness's own sampling noise and far short of the +9.7pp that made the overshoot rule necessary, but it is on the wrong side of it and is logged rather than waved through.

**A second defect surfaced from the same comparison, in the opposite direction.** Our LFE-PPN matches at `d = 0.5 m` and undershoots by **9.4pp at `d = 0.3 m`** (53.1% against 62.5%), against LFE-Peaks' flat +2.1/+2.5.
That signature is **mislocalisation**: the detections land near enough to pass a 0.5 m gate and too far to pass a 0.3 m one.
It points at the anchor-decode geometry calibrated on 2026-09-10, whose angular convention was explicitly recorded as weakly identified *because AP@0.5 could not distinguish it*. AP@0.3 can, and does.
**Use `--eval-r 0.3` as the discriminator when calibrating anything positional.**

### Four levers on DROW's 20.5 pp gap, one seed each (2026-11-07)

**Every number in this section is one seed, and a DROW seed is worth 3.19 pp.** Measured 2026-11-07, after
these arms: the headline recipe re-run at seed 1 and nothing else changed scores **26.26%** against seed 0's
**29.45%**, where FROG's spread on the same quantity is 0.16 pp over three seeds. So the table below ranks
arms whose differences are mostly smaller than their own noise, and only two conclusions in it survive: the
**range span** (+5.20 pp on matched labels, which has a second seed on its 29.99 m side -- 31.16% and 28.18%,
mean 29.67%, against the 10 m arm's 36.36%), and the fact that the **softmax head collapses** (-9.95 pp, three
times the spread). Everything between -5.3 and +2.7 pp is a guess, including the kernel arms -- whose
non-monotonicity *is* this spread -- and the sigmoid arm, whose +2.65 pp did not survive having its class
balance fixed (28.79% and 31.01% at two seeds, mean 29.90% against the headline's 27.86%). The stopping rule is
the likely mechanism: these runs took their best val AP anywhere between epoch 3.7 and epoch 8.0.

Seven stage-2 arms off one chain, all on DROW test's 2,428 annotated frames at 29.99 m, AP at 0.5 m, one seed
each, read against the headline recipe's **29.45%**. The chain's first job was a FROG regression arm that had
to reproduce `a71_freeze_s0` exactly and did -- candidates **82.7578%**, whole detector **86.1306%** -- and
`range_full` is the headline recipe itself re-run on the same code, which returned **29.4436%**. Both halves of
the determinism check passed, so the deltas below are the arms.

| arm | test AP | vs headline | FP/frame | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| the headline recipe | 29.45% | -- | 8.279 | 8.2% | 90.6% |
| `range_full` (the headline, re-run) | 29.44% | -0.01 | 8.278 | 8.2% | 90.6% |
| `heads_sigmoid` (3 class columns, independent) | **32.10%** | **+2.65** | 9.349 | 7.5% | 93.8% |
| `range10` (trained at 10 m) | **33.44%** | +3.99 | 5.665 | 10.3% | 80.2% |
| `kernel3` | 28.57% | -0.88 | 6.677 | 9.6% | 86.7% |
| `gather` (1.66 m metric window) | 24.64% | -4.80 | 6.753 | 8.9% | 81.3% |
| `kernel5` | 24.20% | -5.24 | 10.460 | 6.5% | 89.7% |
| `heads_softmax` (4 columns, background first) | **19.50%** | **-9.95** | 23.331 | 3.3% | 97.9% |

**`range10`'s unrestricted column is not a comparison** -- a 10 m model forfeits the 159 labels past 10 m, which
is the whole of its missing recall. Both range arms were re-scored on the same 1,817 labels inside 10 m, and
that is the chain's one clean isolation:

| | labels inside 10 m | labels and detections inside 10 m |
| --- | ---: | ---: |
| trained at 29.99 m | 30.66% | 31.16% |
| trained at 10 m | **36.36%** | **36.36%** |
| | **+5.70** | **+5.20** |

**Training at the range it is scored on is worth 5.2 pp of ranking**, which makes depth a real component of the
gap -- and the cheap explanation for it is already dead in the same chain, because the 1.66 m metric window
*lost* 4.80 pp here against the 1.42 pp it cost on FROG. So the answer is not an input crop. This architecture
has no distance normalisation of any kind, while every published DROW baseline resamples a fixed metric window
per beam, and what the two results together say is that the missing piece is in the **loss or the vote target**
rather than at the input. *(Premature, 2026-11-08: the metric window tested the input's angular axis only, its depth
axis is 95(c), and the section below finds the gap is transfer from train to test rather than anything stage 2
fails to fit.)*

**Read any single-seed DROW delta in this table against its own arc arms.** `kernel5` at 24.20% and `kernel3` at
28.57% are not monotone in `k`, and no mechanism makes a 5-beam arc worse than a 3-beam one -- so either the
response is genuinely non-monotone or **DROW single-seed spread is several points**. FROG's spread on the same
quantity is 0.16 pp over three seeds. Nothing in this table is a figure until a second seed exists, and the two
arms that beat the headline are the two worth spending it on.

**The sigmoid arm carries a second variable, and it is a defect rather than a property of the head.**
`compute_loss` promises that `pos_weight` is counted from the hard labels under a soft target; its sigmoid
branch sums the positives *after* the person column has been overwritten with the Gaussian, so that column's
weight is its Gaussian mass rather than its beam count. The mass is the smaller number, the weight the larger
one, and the arm fires harder -- which is the operating point it reported. Its +2.65 pp is therefore a head
change **and** a class-balance change. The binary numbers everywhere else in this file are unaffected: the
binary branch counts from hard labels, and FROG has one class.

**The softmax arm collapsed** to 19.50% at 23.331 FP/frame and 97.9% recall, early-stopping after 23
evaluations against the sigmoid arm's 59. Two unmeasured candidates: `cross_entropy` normalises a weighted mean
by the sum of the class weights for an index target and by the batch size for a probability target, so the
adopted recipe's soft row changes the loss's scale against the vote term in that branch alone; and softmax
makes the classes compete, which on a dataset where **56.4% of the objects are wheelchairs and walkers** at
knee height is a discrimination problem the person column pays for, where sigmoid lets one beam be a person and
a wheelchair at once.

**The whole detector followed stage 2 down.** The softmax arm's averaged stage 2 went on through stage 3 (38
epochs, best at 30) and scored **38.70%** at test against the binary chain's **46.01%**, one variable apart:

| | binary chain (`drow_stage3_s0`) | the softmax head | the sigmoid head, fixed |
| --- | ---: | ---: | ---: |
| candidates, test, stride 1 | 23.45% | 20.73% | 19.82% |
| **whole detector** | **46.01%** | **38.70%** | **37.51%** |
| the memory's contribution | **+22.56 pp** | +17.97 pp | +17.69 pp |
| best **val** AP | 47.43% @ 36 of 44 | 46.72% @ 30 of 38 | **51.27% @ 21 of 29** |
| stage 3 alone, one stream, GPU | 5.12 ms | 3.45 ms | 5.17 ms |

So **46.01% is still the best DROW whole detector on record**, and a multi-class head loses on both halves of
the pipeline: 9.95 pp at stage 2 and 7.31 pp after the memory, which means the memory amplified the candidate
deficit (2.72 pp) rather than absorbing it. Two readings worth keeping. At matched false positives the memory
reaches **96.8% recall against these candidates' 97.2%**, giving up 0.4 pp where the binary chain's memory
*gained* 3.5 pp at the same comparison -- on multi-class candidates its entire contribution is ranking. And
through 38 epochs of **val** this run was adding +23.20 pp against the binary chain's +19.70, the exact
opposite of the test ordering, which is why the memory's contribution is quoted from test and nowhere else
([`interpreting-evaluation.md`](interpreting-evaluation.md)).

**The sigmoid head's whole detector landed the same way, which makes it two arms rather than one opinion**
(2026-11-07, `drow_sigmoid_stage3_s0`, the `pos_weight`-fixed arm). **37.51%** at test stride 1, the worst of
the three on both halves. Its `--max-epochs 42` cap did **not** bind -- best val at epoch 21, patience ended it
at 29 -- so the figure is about the head.

Two readings come out of having three chains instead of two. **A multi-class head costs the DROW whole detector
about 8 pp whatever its normalisation**: 37.51% and 38.70% against 46.01%, agreeing with each other to 1.19 pp,
which is inside the 3.19 pp stage-2 seed spread and therefore not a ranking between them -- while the ~8 pp to
the binary chain is 2.7 times that spread and is held by both arms independently. And **the memory's lift falls
with the candidate footing rather than rising to compensate**, monotonically: candidates 23.45 / 20.73 / 19.82
give +22.56 / +17.97 / +17.69. That kills the standing hypothesis that a worse-ranked stage 2 leaves the memory
more to do, which all three runs' **val** readings supported.

**This run is also the sharpest instance of the val/test inversion in the file.** Its best val AP, 51.27% at
epoch 21, is the highest of the three and 3.84 pp above the binary chain's final best, held over seven
consecutive epochs, from the *worst* candidate footing (20.28% on val against 27.73%). Its val memory
contribution, +30.99 pp, is the largest recorded here. Both facts reverse on test. Nothing about DROW stage 3
can be read off val -- not the lift, not the ordering, not the sign of a difference
([`interpreting-evaluation.md`](interpreting-evaluation.md)).

**What this does to the 20.5 pp.** Nothing yet, and nothing here was expected to. What it does is close
suspects: the kernel arc, the metric window, capacity in both directions and the ignore region are now all
measured on DROW and all closed. What is left is **distance** -- real, +5.2 pp, mechanism open -- and
**temporal input**, which is untouched: DROW3's 71.7% is a five-scan odometry-aligned model, and this project's
memory cannot learn across DROW's 24 s annotation gaps (footnote 14).

### DROW's stage-2 headline at DROW's own label radii: 28.92% over five seeds (2026-11-09)

Stage 2 alone, DROW test, every annotated frame, the adopted recipe; DROW's default label radii since 2026-11-09 are its
own 0.35 / 0.6 / 0.4 m (`CHANGELOG.md`):

| label radii | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | sd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 m every class (the default until 2026-11-09) | 29.45% | 26.26% | 28.64% | 26.97% | 24.29% | 27.12% | 2.03 |
| 0.35 m every class | 28.26% | 32.21% | 29.63% | 30.14% | 26.55% | 29.36% | 2.11 |
| **0.35 / 0.6 / 0.4 m, DROW's own (the default)** | 28.62% | 29.43% | 29.63% | 30.40% | 26.55% | **28.92%** | 1.47 |

**The headline is 28.92%**, +1.80 pp over 0.6 m (paired by seed, SE 0.78, ahead on four seeds of five). The last two rows
are **one objective trained twice**: the binary head reads only the person's radius, and the other classes' radii draw
an ignore region only under `--ignore-other-classes`, which is off. Their difference is this machine's run-to-run spread
-- see `interpreting-evaluation.md` -- so every DROW comparison above that was read against 27.86% or 27.12% stands, and
reads about 1.8 pp further from the cutout's 63.20% against the new default.

### On a geometry-balanced DROW the gap to the cutout is 11.9 pp, not 34.2 (2026-11-09)

Each model trained and scored on one partition, person AP at 0.5 m (`CHANGELOG.md`; the partitions in `data-model.md`):

| | official DROW | `geometry-balanced` |
| --- | ---: | ---: |
| cutout (DR-SPAAM's recipe, one seed): test AP, train / test | 63.20%, 1.05 | 53.75%, 1.27 |
| our stage 2 (default radii, two seeds): test AP, train / test | 29.03%, 2.40 | 41.89%, 1.76 |
| ours over the cutout, test AP | 0.46 | 0.78 |

The test APs of the two columns are on different test splits and are read only against the same column's cutout.

### Item 104's two DROW arms leave the gap where it was (2026-11-09)

Official DROW test, stage 2 alone, two seeds each at DROW's default radii, beside the headline at the same seeds:

| arm | test AP (s0 / s1, mean) | train / test | pooled AP / within-scan hits |
| --- | --- | ---: | ---: |
| headline | 28.62 / 29.43%, **29.03%** | 2.48 / 2.33 | 0.50 / 0.51 |
| scan-normalised range (`--input-features scan-norm`) | 32.16 / 25.01%, **28.59%** | 2.04 / 2.67 | 0.57 / 0.50 |
| local level (`--local-level`) | 28.12 / 27.56%, **27.84%** | 2.36 / 2.35 | 0.49 / 0.50 |
| DROW3's cutout, retrained here | 63.20% | 1.05 | 0.86 |

Both inside DROW's seed spread. The local level fixes each scan's mean logit to the local branch's and still leaves cross-scan comparability unchanged, so the failure is not a per-scan additive offset (`CHANGELOG.md`). No CPU ms/frame measured: neither arm is a candidate.

### DROW's gap is transfer, not fitting, and not the harness (2026-11-08)

**Temporal input cannot be the gap at stage 2**: at one frame our stage 2 scores 80.89% on FROG against DROW3 `T=1`'s
73.9%, and 27.86% (two seeds) on DROW against DROW3 `T=1`'s 59.4% (Beyer et al.) and 66.6% (DR-SPAAM's retrain). Neither
side of that comparison reads a second scan. Three measurements since, all on DROW test at stride 1:

| measurement | result |
| --- | --- |
| DROW3's published weights, `evaluate.py` against `score_detections` on `FrameSplit`'s labels | **66.48% and 66.48%**: the harness and the labels are sound |
| our stage 2 on the published schedule, 40 epochs, last weights, two seeds | **16.98%** against the headline's 27.86%: **-10.9 pp** |
| our stage 2, train / test AP of the same weights | DROW **63.6% / 29.5%** (82.6% / 18.1% after 40 epochs); FROG **93.7% / 80.9%** |
| DROW3 at `T=1` **retrained here** on DR-SPAAM's recipe, 40 epochs, last weights | **63.20%** against their published 66.6% (-3.4 pp); val loss flat from epoch 7 to 40 |
| that cutout, train / test AP of the same weights | **66.5% / 63.2%**: train/test 1.05, against our stage 2's 2.16 at its best epoch |
| our stage 2 at 15x and 58x fewer parameters (62,867 / 16,667), and LFE-shaped (LFE's input and widths), two seeds | test **22.97% / 24.15% / 25.38%**, train/test **2.47 / 1.94 / 2.19**: capacity is not what memorises |

**So the network fits DROW's training recordings and does not carry what it learns to the test ones**, and the longer it
trains the worse that gets, while on FROG train and test stay within 13 pp. **And a cutout network does carry it**, on
the same frames through a training path now validated against a published number: 63.20% against our 27.86%. The stopping rule is not hiding a
better model; it is the only thing standing between DROW's stage 2 and its training set. Read every DROW lever as a
generalisation lever: what it takes away from what can be memorised, not what it adds. The composition that makes
memorising pay is in [`dataset-properties.md`](dataset-properties.md): a third of train from a stationary robot, val
essentially one recording, test all driving runs. Details and the per-seed table: `CHANGELOG.md`, 2026-11-08.

### Li2Former on DROW: the one reproduction that did not land, and what it is not caused by

**2026-11-03.** Their own network, trained here because no weights exist to score, against
their published pair:

| matching radius | their network, trained here | theirs, published | delta |
| --- | --- | --- | --- |
| `d = 0.5` | **48.1%** wp-AUC | 76.4% AP | **-28.3 pp** |
| `d = 0.3` | **40.5%** | 72.5% | **-32.0 pp** |

It is an undershoot, so the overshoot rule above does not fire -- but 28 pp is far outside anything a wrapper detail
explains, so the harness was tested instead of assumed innocent. **The upstream grid crossed their decoder and their metric with
ours, all four combinations over one inference pass** (`evaluate.py --upstream-grid`, whole DROW test split, 2,428 frames,
epoch 8's checkpoint; the baseline cell equals `evaluate_auc`'s own answer to 1e-9 by construction, and matched the
`Person (wp)` line of its own run exactly in both passes):

| decoder / metric | `d = 0.5` | `d = 0.3` |
| --- | --- | --- |
| ours / ours (the rows above) | **48.1%** | **40.5%** |
| theirs / ours | 48.2% | 41.0% |
| ours / theirs | 48.1% | 40.5% |
| theirs / theirs (their protocol end to end) | **48.2%** | **41.0%** |

**The whole decode-and-score path is worth +0.1 pp at `d = 0.5` and +0.5 pp at `d = 0.3`** -- 0.4% and 1.6% of the
respective gaps. Their metric is worth 0.0 pp at either radius; their decoder is worth all of the difference, and more at
the tighter one, which is what per-beam placement should buy over a blurred vote grid. **So the undershoot is in the
weights and the training, not in the measurement**, and adopting their decoder and metric wholesale -- proposed 2026-11-01,
~4 h, CPU-only -- is a protocol-comparability decision rather than an accuracy one.

**Why the training stopped where it did.** Twelve epochs at batch 1 and `lr 1e-4`; **epoch 8 is the best checkpoint, at
32.9% val wp-AUC**, and epochs 10-12 scored 0.0% because their regression objective has an **escape basin** past 0.70 m of
vote error, where its gradient reverses sign (the mechanism is in [`../docs/RESEARCH.md`](../docs/RESEARCH.md) section
3.3.1). So this row is the arm at its best rather than a collapsed head -- and it is **one arm of one recipe**, not a
demonstration that their figure is unreachable. Their code was left exactly as released.

Three by-products worth keeping:

- **Their AP and this project's `wp-AUC` agree to within 0.1 pp at split scale** on identical detections. That is the
  strongest check this file has that the two are the same quantity -- stronger than the DR-SPAAM source comparison above,
  because it is measured rather than read off the source. The quirks in their implementation are real and small: their AP
  has no anchor at recall 0, so `n` perfect detections score `1 - 1/n`, and it is blind to false positives arriving after
  recall saturates. Both matter on a handful of frames and not on a split.
- **The `d = 0.5` to `d = 0.3` drop is 7.6 pp, -15.8% relative, against their published pair's -5.1%.** Our localisation
  degrades about three times faster than theirs, which is the one shape in this comparison that is not explained by the
  training budget.
- **The val-test split gap is 15.2 pp** (32.9% val against 48.1% test), larger than any epoch-to-epoch move in the whole
  run -- DROW's val split is harder than its test split, so a val-side conclusion must not be quoted as a test-side one.

Their metric also reports **peak F1 52.5% and EER 52.3%** at `d = 0.5` (48.9% and 48.5% at `d = 0.3`), which are the first
such figures for any detector in this project and have no published counterpart: their paper quotes AP alone.

## FROG `balanced`: the published pair, and the two retracted numbers that turn out to have been right

**Measured 2026-11-03**, the first time anything has been run on this partition (`queue_balanced.py`'s
`published_baselines`, 48 min on the CPU over all 110,846 test frames; `evaluate.py --frog-mode balanced
--split test --lfe-peaks --lfe-ppn`). `balanced`'s test is 16-41 + 15-53, which holds out **both** recordings
LFE's authors trained on, so unlike its val this split is an honest place to score them.

| detector | `balanced` test | `official` test (ours) | published | difference |
| --- | ---: | ---: | ---: | ---: |
| LFE-Peaks | **59.8%** | 65.1% | 65.6% | **-5.3 pp** |
| LFE-PPN | **63.5%** | 69.6% | 69.2% | **-6.1 pp** |
| AlgorithmicDetector | F1 **1.4%** | F1 1.7% | -- | -0.3 pp |

**Do not read that difference as worse generalisation**, and this is the cross-regime trap `AGENTS.md` rule 3
exists for. `official`'s test is one recording, populated frames only, every 5th frame annotated.
`balanced`'s is two recordings with **19.1% person-free frames**, and a person-free frame can only contribute
false positives -- there is no true positive in it to find -- so it lowers precision at every threshold and
therefore the AUC, before any question of generalisation arises. Several of those 5-6 points are bought by
composition alone. The like-for-like comparison is the `official`-test subset of the same pass, which is
extractable (verified: 40,022 frames, 100% of `official`'s test) but has no entry point yet -- `TODO.md` item
69. **Expect this project's own number to drop on `balanced` for the same free reason**, so the one-seed
decision rule should be read against the subset, not against the whole split.

**And the two figures reproduce the ones retracted on 2026-10-10 exactly.** `docs/PAPER.md` carried LFE-Peaks
**59.8%** and LFE-PPN **63.5%** on this split; both were deleted because no `CHANGELOG.md` entry, no
`memory/` page and no `TODO.md` row recorded the measurement, and the owner's recollection was that nothing
had ever been run on `balanced`. Measured now, cold, they come back **59.8%** and **63.5%**. Two independent
AUCs agreeing to a tenth of a point is not coincidence, so a run did happen and nothing wrote it down.
**That does not make the retraction wrong**: a number nobody can trace is unpublishable whatever it later
turns out to equal, and the retraction is what made this check possible, because it recorded both values
instead of only deleting them. Keep doing that.

### This project's own detector on `balanced`, one seed (2026-11-04)

**The first model ever trained on this partition** (`queue_balanced.py`, 5.45 h: 46 min of stage 2 and the
rest stage 3, early-stopped at epoch 12 on patience with the best at epoch 4). Scored over all 110,846 test
frames, and again at stride 5, which **agree to 0.01 pp** -- so the number is not a sampling artefact.

| | `balanced` test | `official` test (frozen, 3 seeds) | difference |
| --- | ---: | ---: | ---: |
| whole detector, wp-AUC | **82.49%** | **86.29%** | **-3.80 pp** |
| the same at 0.3 m | 81.37% | 85.19% | -3.82 pp |
| stage 2 alone | 77.89% | 82.8% | -4.91 pp |
| **the memory's contribution** | **+4.60 pp** | +3.49 pp | **+1.11 pp** |
| FP/frame, whole detector | 1.327 | 0.849-1.239 | -- |
| recall / precision | 87.6% / 63.1% | -- | -- |
| memory cost | 5.29 ms/frame | 7.23 ms/frame | -- |

**Read the -3.80 pp against what the same split change costs everyone else, not against zero.** Measured the
same night on the same two splits: **LFE-Peaks -5.3 pp**, **LFE-PPN -6.1 pp**, this detector **-3.80 pp**.
`balanced`'s test is 19.1% person-free and a person-free frame offers no true positive, so it can only cost
precision -- part of every one of those drops is bought rather than earned. **This architecture loses the
least of the three**, and the memory's contribution *grows* from +3.49 to +4.60 pp, which is what the design
predicts: phantom suppression has more to do where there are more frames with nobody in them. The same
reading shows in the operating point -- the memory cuts false positives from 2.833 to **1.327** per frame, a
53% cut, against 72% on `official`'s person-free split.

**What this does not settle, and no amount of seeds would.** Whether 82.49% and 86.29% differ for any reason
beyond composition is answerable only on the same frames, and `balanced`'s test contains `official`'s whole
(40,022 frames, verified). Nothing can score a trained checkpoint on a partition it did not train on yet --
`TODO.md` item 69 -- so **the second and third seeds would reproduce a composition effect rather than test
it**, and the cheap answer comes first. Coasting fired **0.000** times per frame here too, as on `official`.

### The same model on the same frames: 2.44 of those 3.80 points were composition (2026-11-04)

**`train_takheliped.py score`** scores a trained detector on a partition it was not trained on, reusing the
same `evaluate` the training path uses so the number stays comparable with the frozen one. The
`balanced`-trained pair on **`official`'s own test split**, stride 5 -- the stride 86.29% was measured at --
over 50,088 annotated frames:

| trained on | scored on | stage 2 | whole detector | memory's gain |
| --- | --- | ---: | ---: | ---: |
| `official` (3 seeds, frozen) | `official` test | 82.8% | **86.29%** | +3.49 pp |
| `balanced` (1 seed) | `official` test | **82.38%** | **84.93%** | +2.55 pp |
| `balanced` (1 seed) | `balanced` test | 77.89% | 82.49% | +4.60 pp |

**So the recipe costs 1.36 pp, and composition accounted for the other 2.44.** On `balanced`'s own test the
gap read -3.80 pp; on identical frames it is **-1.36 pp** for the whole detector and **-0.42 pp** for stage 2,
the latter inside what three `official` seeds span twice over. That is the measurement the mode comparison
needed and could not take before: two thirds of the apparent penalty was the 19.1% person-free frames in
`balanced`'s test, which can only cost precision.

**What the remaining 1.36 pp is, stated as a trade rather than a loss.** Training on `balanced` means training
on 12-43 + 14-57 with a fifth of the frames empty, instead of `official`'s two populated-only recordings. It
costs about a point and a half on a populated-only test split, and it buys the shallower degradation measured
above -- **-3.80 pp** across the composition change against LFE-Peaks' **-5.3** and LFE-PPN's **-6.1**. The
memory's gain moves the same way for the same reason: **+4.60 pp** where empty frames exist, **+2.55 pp** on
a split with none, against the `official`-trained model's +3.49 pp.

**One seed, and what that does and does not support.** The direction and rough size are safe: 1.36 pp is six
times the 0.22 pp spread three `official` seeds show. What one seed cannot do is bound `balanced`'s *own*
spread, so a pre-print quoting this row beside a three-seed row should either say so plainly or spend the
~11 h on seeds 1 and 2. That is now a presentation decision rather than an open measurement.

### Person-free training frames cut stage 2's phantoms by 21%, and the pipeline's output not at all (2026-11-04)

**The one thing `balanced` exists to test.** `official`'s train split is 100% populated and `transferred`'s train and val
are *identical to it*, so no detector this project trained had ever seen a room with nobody in it -- the standing
explanation for stage 2 emitting ~2.1 phantoms per person-free frame. `balanced` shows it one in every split, and both
arms are counted here on the **same 21,149 person-free frames** of `balanced` test (19.07% of 110,846, matching the
composition measured from the real files). Those frames are all **15-53**'s, since 16-41 is 100% populated, and 15-53
is in *neither* arm's training set. `utils/queue_person_free.py`, `results/person_free_*.json`.

Detections per frame at threshold 0.3, both arms read at the same scale:

| at threshold 0.3 | `official`-trained | `balanced`-trained | change |
| --- | ---: | ---: | ---: |
| **stage 2 candidates** | 2.216 | **1.751** | **−21.0%** |
| frames holding at least one | 59.7% | 49.2% | −10.5 pp |
| **stage 2, the 9,917 frames beyond the 10 s fade** | 1.788 | **1.153** | **−35.5%** |
| whole detector | 0.710 | 0.684 | −3.8% |
| whole detector, beyond the fade | 0.414 | 0.295 | −28.8% |
| what the memory itself removes | −67.9% | −60.9% | — |

**The effect is in stage 2, it is large, and it is outside the noise.** The cut grows with the threshold --
**−23.0%** at 0.5 and **−33.5%** at 0.9 -- and roughly doubles on the frames where nobody has been in view for ten
seconds (**−57.5%** at 0.9), which is the subset where "a stale memory of someone who just left" is unavailable as an
explanation. Against the stage-2 column's three-seed spread of **sd 0.03** on this measurement, a 0.465 move is about
fifteen standard deviations.

**The whole detector's output does not improve, and that is the honest headline.** 0.710 against 0.684 is **−3.8%**,
and the whole-detector column spreads **sd 0.073 (11% relative)** across three seeds -- so this difference is inside
one seed's noise and is not a result. The reason is visible in the last row: **the memory's filtering is
substitutive**. It removes 67.9% of the worse arm's candidates and 60.9% of the better arm's, so a 21% improvement
upstream arrives at the output as nothing. The better stage 2 is reached with the memory doing *less* work, not with a
better final count. Beyond the fade horizon the output does move, −28.8%, but on a column with 11% seed noise and one
seed per arm that is suggestive rather than settled. At thresholds 0.2 and 0.1 the two arms' outputs cross, with the
`balanced` arm marginally **higher** (0.975 against 0.974, 1.606 against 1.572).

**Coasting is zero at every threshold in both arms.** Item 27's finding -- every person-free phantom is a stage-2
candidate the memory failed to suppress, never a memory invention -- now holds on a second partition and for a
detector trained on empty rooms. The surviving phantoms also recur slightly *less*: 83.8% against 85.1% within 0.3 m
in the next frame, and 77.1% against 82.5% beyond the fade, so what the person-free training removed was
disproportionately the persistent, furniture-like kind.

**The price, and the trade.** Item 69 measured it on identical populated frames: **−1.36 pp** wp-AUC. So person-free
training buys a 21% stage-2 phantom cut (35.5% beyond fade) for about a point and a half of accuracy, with no change
to what the assembled pipeline emits. For a robot reading the pipeline's output, this is not yet a reason to train on
`balanced`; for the paper's account of *where* the phantoms come from, it is direct evidence.

**Two caveats that belong beside the numbers.** The `official` arm *selected* on 15-53's populated frames, since
`official`'s val holds all three raw recordings -- venue exposure the `balanced` arm never had, which favours the arm
this result argues against, so the direction is safe. And the seed spreads quoted above were measured on `official`
seeds over `transferred`'s frames; carrying their *relative* size to this partition is an assumption, not a
measurement.

**Checked rather than assumed, 2026-11-04**: no recording either arm trained on appears in any split it was scored
on, read from each checkpoint's own recorded `frog_mode` and `_MODES`' assignment. The 15-53 overlap above is
selection only, and the frames do not even coincide -- `official`'s val takes that recording's **populated** frames
and this comparison counts its **person-free** ones. Six tests in `tests/test_frog_splits.py` now assert the property
off the partition table, so a future edit cannot reintroduce it quietly. The two arms also record **identical**
hyperparameters except the partition and `chunk_s`, where the `balanced` run used a `1 -> 10 s` curriculum against
the adopted flat 10 s -- a choice adopted on a tie, so it does not re-base these figures
(`docs/ARCHITECTURE.md` section 6.1).

**Both arms were read with `--no-calibration`, and skipping that would have reversed the sign.** The `official` runs
carry a fitted `(temperature, bias)` and the `balanced` run does not, and every row here is a fixed-threshold row. As
published, the `official` arm reports **0.582** per frame against the `balanced` arm's 0.684 -- which reads as the
person-free training making the detector **17.5% worse**, the opposite of what the matched-scale comparison shows.
The third job measured that as-published row deliberately, and it reproduces the calibration's known behaviour on new
frames: **−18.1%** at threshold 0.3 and **3.7x** at 0.9, against the −11.8% and 8x measured on `transferred`. The two
`official` arms' candidate columns are **bit-identical**, which is the control proving the override touches stage 3
and nothing else (`interpreting-evaluation.md`).

## FROG at 0.5 deg: halving the angular resolution costs about a point, and the memory carries none of the repair

One `official` chain retrained on every second beam -- 360 beams over 180 deg, 45 sectors, no padding,
`--beam-stride 2` on every stage -- against the frozen three-seed headline at FROG's native 0.25 deg. Both on
FROG `official` test, 50,088 frames, 153,655 people, every annotated frame; stride 5 agrees to 0.02 pp.

**The decimated chain was run twice**, because the first arm's averaging window had been ranked at the wrong
angular spacing ([`gotchas.md`](gotchas.md)). The **corrected** arm is the measurement: its window was ranked on
its own 0.5 deg val split, exactly as the headline's was. The mis-ranked arm is kept beside it because the pair
differ in nothing else, which is what prices the defect.

| | 0.25 deg, 3 seeds | **0.5 deg, correct window** | 0.5 deg, mis-ranked window |
| --- | ---: | ---: | ---: |
| **whole detector, wp-AUC** | **86.30%** (86.13 / 86.31 / 86.44, sd 0.16) | **85.28%** (**-1.02 pp**) | 85.42% (-0.88 pp) |
| the same at 0.3 m | 85.20% | 84.16% (-1.04 pp) | 84.35% (-0.85 pp) |
| candidates (stage 2, averaged weights) | 82.79% (sd 0.10) | **82.18%** (-0.61 pp) | 81.74% (-1.05 pp) |
| **the memory's contribution** | **+3.51 pp** (sd 0.12) | **+3.10 pp** | +3.69 pp |
| recall / precision, AP-optimal | 84.8% / 73.2% | 88.2% / 69.6% | 88.3% / 68.1% |
| recall at each arm's own threshold | 95.1% (94.9 / 95.3 / 95.2) | 94.2% | 95.3% |
| FP/frame, AP-optimal | 0.95 (0.87-1.02) | 1.183 | 1.271 |
| candidate FP/frame, AP-optimal | -- | 2.536 | 2.914 |

**The cost is about a point, and the memory does not repair it.** Stage 2 at half the beams is **0.61 pp** behind
and the whole detector **1.02 pp**, so the deficit *grows* through the memory rather than shrinking. The memory is
worth **+3.10 pp** here against **+3.51** at full resolution. That reverses what the mis-ranked arm appeared to
show, and the reversal has a mechanism: a weaker stage 2 leaves the memory more to remove. Fixing the window cut
candidate false positives from 2.914 to 2.536 per frame and lifted candidate precision from 49.4% to 52.8%,
after which the memory's own false positives fell 1.271 -> 1.183 at unchanged recall (88.3% -> 88.2%) -- less
work done, because less was left to do. **The memory's contribution is therefore not a stable statistic for
comparing two arms**: it moved 0.59 pp between two runs that differ only in their stage-2 weights, while the
whole detector moved 0.14. What survives is the weaker claim: at half the beams the memory still adds about
**3 pp**, within half a point of what it adds at full resolution, and one seed per arm cannot say which side.

**What the mis-ranked window cost, exactly.** The two 0.5 deg arms differ in one thing, so the difference is the
defect's price. Candidates rose **+0.44 pp** (81.74 -> 82.18) -- an exact figure with no seed noise in it, since
the candidate column is a deterministic function of the stage-2 weights. The whole detector moved **-0.14 pp**
(85.42 -> 85.28), inside stage-3 seed noise, the headline's three seeds spanning 0.31 pp. So the defect was worth
about half a point of stage 2 and nothing measurable after the memory.

**The -0.88 pp was reported as an upper bound, and that framing was wrong.** It assumed a better stage 2
propagates to a better whole detector. It did not: stage 2's deficit nearly halved while the whole detector's
grew by 0.14 pp. The resolution cost is now measured on the correctly ranked arm -- **-1.02 pp**, one seed
against three -- rather than bounded by the mis-ranked one, and "about a point" is the honest reading until a
second seed exists.

**The arms sit at different AP-optimal thresholds** (0.116 for the corrected arm, 0.136 for the mis-ranked one,
~0.073 for the headline), so the precision/recall and exchange-rate rows are not a like-for-like trade. The
own-threshold row is not like-for-like either, because each arm is held at *its own* candidate FP rate: the
corrected arm reaches 94.2% recall at 2.536 FP/frame where the mis-ranked one reaches 95.3% at 2.914, which is a
cleaner input rather than a worse memory. Its exchange rate reads 10.4 against the mis-ranked 12.1 and the
headline's 6.8 for the same reason -- a different point on its own curve.

**Decimation does not make inference cheaper -- it makes it dearer.** Measured 2026-11-06 on the CPU, which is
the device this project is judged on: four threads, 2,000 consecutive scans, batch 1, back to back on a host idle
but for the two jobs, both arms through the same `tracking_baselines.py latency`.

| phase | 0.25 deg | **0.5 deg** | difference |
| --- | ---: | ---: | ---: |
| stage 2 | 8.41 ms (p99 9.35) | **10.29 ms (p99 12.12)** | **+1.88 ms, +22%** |
| decode | 1.34 ms (p99 1.94) | 1.26 ms (p99 1.83) | -0.08 ms |
| memory | 5.99 ms (p99 6.43) | 5.89 ms (p99 6.43) | -0.10 ms |
| **total** | **15.76 ms (p99 17.24)** | **17.46 ms (p99 19.82)** | **+1.70 ms, +11%** |

**Halving the beams made stage 2 22% slower.** This was predicted to be where decimation paid, and it is not: the
prediction was that half the beams would be roughly half the convolution. The memory and the decode are flat, as
expected -- the memory runs over 256 candidate slots and never sees a beam -- so the whole move is stage 2's.

**The explanation is a hypothesis, not a measurement.** The likeliest is thread efficiency: at four threads a
720-beam tensor fills the threads where a 360-beam one does not, so the halved arm pays the same per-operation
overhead over less work, and its p99 degrading further than its median (+2.77 ms against +1.88) fits that. What
would settle it is the same pair at one thread and at eight; nothing here establishes the cause.

**So the resolution question now has both halves, and they agree.** 0.5 deg costs **1.02 pp** of whole-detector
accuracy *and* **11%** more CPU time per frame. Against FROG's 25 ms budget both arms still fit, so this is a
margin question rather than a feasibility one -- but there is no version of this trade in which decimating to meet
a common angular spacing is the cheap option. `step3a`'s own GPU figure says nothing either way, and the two
decimated arms are the proof: the **same** architecture at the same slot count reads **3.58 ms** in the morning
run and **5.22 ms** in the evening one, 46% apart, because the host was slower throughout the second (596 s
against 398 s for an identical epoch). It is a host-load figure.

**Training is the one place decimation pays**, and it is not a deployment cost: 37 s against 50 s for a stage-2
epoch, 397 s against 582 s for a stage-3 one, since both replay stage 2 over every beam. Read that pair off the
morning runs only -- the corrected arm's epochs cost 596 s on the slower evening host, which says nothing about
the beam count.

**What it means for `docs/PROPOSAL.md` §5.3.** The proposal resamples every scan to one fixed angular spacing so
that a kernel covers the same angle on any sensor, and §6.1 leaves the exact value "to confirm". This is the first
measurement of what that spacing is worth: at **half** FROG's, about a point of whole-detector AP (1.02 pp). So
the resampler is a real but small effect rather than the requirement the proposal assumed, and a sensor at half
FROG's resolution is usable without one. It does not settle the *cross-sensor* case the resampler exists for,
where beam index stops meaning the same angle -- that is DROW and JRDB, not a decimated FROG.

## Why stage 2 beats LFE and DR-SPAAM: what it is *not*

**What the lead is made of, measured on fixed weights** (FROG `official` test, 0.5 m match radius). Each row changes one thing and keeps everything above it, so the deltas add to the whole gap:

| | wp-AUC | step |
| --- | --- | --- |
| LFE-Peaks, as its authors published it | 65.11% | — |
| **our network, read through LFE's own peak decoder** | 71.26% | **+6.15** |
| its peaks reported at the regressed vote, not the beam's endpoint | 74.10% | +2.84 |
| DROW's vote grid instead of per-beam peaks | 77.27% | +3.17 |
| DR-SPAAM's NMS at 0.5 m | 79.88% | +2.61 |
| The adopted training recipe (`nms-mean`, mirroring, nearest-person labels, dropout, weight averaging) | **82.20%** | +2.32 |

**Half the lead is the output head.** The three decoding steps are worth **+8.62 pp** together — 50% of the 17.09 pp gap — against **+6.15** for swapping in our network while keeping LFE's decoder, and **+2.32** for the training recipe on top.
So *how a detection is reported* matters more here than *what network produces it*: given identical per-beam outputs, reporting one peak per probability maximum and merging at 0.30 m leaves about 8.6 points on the table.

**Read the +6.15 honestly.** It is our *trained* network through their decoder, so it bundles the network's form with our training, and the ladder below is what narrows that: LFE's input features are worth nothing to us (+0.14), its widths cost 1.50 pp, both together 1.58, and DR-SPAAM's whole training protocol 1.84 — so the +6.15 is neither the input, nor the capacity, nor the optimiser.
What it is instead — the network's form, a ConvNeXt U-Net with a whole-scan aggregator, against the training that produced LFE's own weights — is the one split this study does not make, and it was dropped.

**The attribution ladder, completed 2026-10-11.** Each arm takes one thing from a published baseline and leaves everything else in the adopted recipe, three seeds, weights averaged, `nms-mean` decoding, every 5th annotated FROG test frame.

| arm | what it borrows | wp-AUC | against C1 |
| --- | --- | --- | --- |
| **C1** | nothing: the adopted recipe | **82.20%** | — |
| F1 | LFE's input: `1 - r/10` clipped to 0.2-10 m, as the only channel | 82.34% | **+0.14** |
| W | LFE's widths, 8/16/32 (63,891 parameters) | 80.70% | **−1.50** |
| **F1 + W** | both at once | **80.62%** | **−1.58** |
| R | DR-SPAAM's training protocol, last weights | 80.11% | −1.84 (of which −1.24 is taking the last weights, not the best on val) |

**The two costs do not add, because only one of them is real.** F1 + W lands within **0.08 pp** of W alone, against an additive prediction of 80.84: LFE's input contributes nothing on top of its width, exactly as it contributed nothing alone.
**So the width is the whole cost — and it is small.** At LFE's own 64k parameters, with LFE's own input, our network still scores **80.62% against LFE-Peaks' 65.1%**: a 15.5 pp gap that no ingredient in this table explains.
Width's 1.5 pp is also entirely on far people, which is what makes range anchors the best-motivated experiment left.

**CPU latency, `dim 128` against `dim 64`, matched and interleaved (2026-10-27).** The matched CPU protocol's
protocol: `tracking_baselines.py latency --device cpu --threads 4`, 2,000 consecutive scans, batch 1, three rounds
alternating the two arms **in one session on an idle host**. Both load the same stage 2
(`a69_no_aggregator_s0`), so the `stage2` phase is byte-identical work.

| phase | `dim 128` | `dim 64` | delta |
| --- | ---: | ---: | ---: |
| `stage2` (the noise floor) | 9.17 / 8.96 / 9.00 | 9.49 / 9.24 / 9.32 | +0.31, an ordering bias **against** `dim 64` |
| `decode` | 1.65 / 1.67 / 1.67 | 1.70 / 1.65 / 1.68 | ~0 |
| **`memory`** | 9.32 / 9.31 / 9.26 | **6.69 / 6.68 / 6.73** | **-2.60 ms, -28%** |
| **total** | 20.18 / 19.98 / 20.01 | **17.88 / 17.58 / 17.74** | **-2.33 ms, -11.6%** |
| **total p99** | 22.22 / 22.74 / 22.73 | **20.67 / 20.18 / 20.49** | **-2.25 ms** |

**How to read it, in the order the rule demands.** The `stage2` spread across all six runs is **0.53 ms** and that is
the session's noise floor; the memory phase's own within-arm spread is **0.05-0.06 ms**. A 2.60 ms drop is therefore 5x
the floor and ~50x its own measurement spread. The `stage2` phase also exposes a **systematic bias against `dim 64`**
-- 0.31 ms slower on identical work, because it always ran second in each round -- so the true saving is marginally
larger than measured. Against FROG's **25 ms** period, median headroom goes 4.94 -> **7.27 ms** and p99 margin goes
2.3-2.8 -> **4.3-4.8 ms**. Accuracy ties: 86.12% against 86.20%, recall at matched false positives 96.5% against
96.43%.

**Odometry drift on the adopted architecture (2026-10-28, `results/odometry_adopted.json`).** The room frame is this
design's central claim and it is bought from odometry, so its price matters. Replayed on `a74_fixed256_s0`:

| drift (m/s) | 0 | 0.005 | 0.020 | 0.050 | 0.100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adopted, room frame AP | **85.73%** | 85.09% | 84.57% | 83.13% | 81.82% |
| recall | 95.85% | 95.53% | 95.45% | 94.74% | 94.03% |
| The room-frame arm, old architecture | 83.88% | 82.96% | 81.25% | 78.27% | 73.10% |

**At 0.1 m/s the adopted model loses 3.91 pp where the old one lost 10.78.** So **E2's conclusion does not transfer**:
"the room frame's advantage survives only below roughly 0.01-0.02 m/s and inverts above it" described the superseded
architecture. No sensor-frame arm exists on the adopted recipe, so the crossover cannot be re-measured; carrying the old
relative -1.31 pp gap onto this baseline puts it near **0.04-0.05 m/s** against FROG's 0.005-0.025 m/s of real drift.
**That last figure is an extrapolation, not a measurement.**

**What the ladder leaves.** With phase 1's decomposition — about half the lead over LFE-Peaks is the output head: centre votes, aggregation, NMS — what remains is **the network's form and the training recipe**, and those two have not been separated. **Nor has any part of the form been shown to carry it** (2026-10-25): block and widths (19b), depth, kernel and expansion and the whole-scan aggregator all tie, so this half of the lead currently has no identified owner. Doing so needs LFE's architecture trained the way ours is, which was attempted and dropped: 29 convolutions reproduce in order from the released graph, but two blocks' dataflow did not resolve without a shape-inference pass. **The pre-print states this as a limitation rather than a result.**

## TAKHeLiPeD's cost, end to end, one scan at a time

**Re-measured 2026-10-11 on the adopted recipe** (`tracking_baselines.py latency`, E22; 2,000 consecutive test
scans, batch 1, one stream, as on the robot; `checkpoints_three_horizon/a61/latency_*.json`, with the pre-item-30 files kept
beside them as `*.pre-item30.json`).
The budget is **25 ms**, FROG's real frame period — not the 38.2 ms that every earlier cost claim was set against (`interpreting-evaluation.md`).

| | stage 1-2 | candidate decoding | object memory | **total** | p99 | share of the 25 ms budget |
| --- | --- | --- | --- | --- | --- | --- |
| **GPU** (RX 9060 XT) | 5.58 | 2.01 | 4.65 | **12.32 ms** | 13.63 | **49%**, 55% at p99 |
| **CPU, 4 threads** | 8.44 | 1.51 | 9.14 | **19.15 ms** | 22.07 | **77%**, **88% at p99** |

**The CPU row reproduces to 0.03 ms across sessions a week apart, and drifts about 3% across others.** The frozen model
re-measured 19.15 ms total on 2026-10-19, matching this row exactly; on 2026-10-20 the same checkpoint read 19.75 ms. The
budget share is therefore **77-80%**, and the number to quote for this row is the one above. What this forbids is comparing
*arms* across sessions: see the session-offset rule and the stage-2 control in `interpreting-evaluation.md`.

**Both now have headroom, and the CPU only got it when the budget was restated.** The same CPU row read **20.98 ms, p99 25.32 — 101% of the
budget** until the candidate decode was restructured: it ran one Python iteration per voting beam, 314 per frame to keep 22
detections, and built a 314x314 distance matrix to do it. Iterating only the beams still alive took decoding from **2.83 ms to
1.51** (p99 5.47 to 2.44) and the whole detector from 101% to **88% at p99**. The output is **bit-identical**, which is what
makes it free: no checkpoint, no candidate cache and no accuracy number moves, and a randomised test pins the new loop against
a transcription of the old one in both decoding modes (`tests/test_stage2_probe.py`).

**Thread scaling saturates at two, and the memory does not scale at all** (same 2,000 scans and checkpoints):

| CPU threads | stage 1-2 | decoding | memory | **total** | p99 | p99 vs budget |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20.56 | 1.47 | 8.92 | **31.03 ms** | 37.82 | 151% |
| 2 | 9.16 | 1.48 | 9.10 | **19.75 ms** | 22.68 | **91%** |
| 4 | 8.44 | 1.51 | 9.14 | **19.15 ms** | 22.07 | **88%** |
| 8 | 8.53 | 1.53 | 9.14 | **19.26 ms** | 22.17 | 89% |

**Two cores are enough and one is not.** The step from one thread to two halves stage 2 (20.56 -> 9.16) and takes the detector
from 151% of budget to 91%; everything after that is worth 0.6 ms, and eight threads is marginally *worse* than four. For a
robot that is the useful sentence: this detector needs **two CPU cores**, not eight.
**The memory is flat across every thread count** (8.92 / 9.10 / 9.14 / 9.14) because it is latency-bound rather than
compute-bound — pure PyTorch over 256 slots of which about 24 are in use — so threads cannot touch it, and at **48% of the CPU
total** it is now the largest term and the only remaining lever worth pulling (slot compaction). Decoding is
likewise flat (1.47-1.53), being numpy and Python, but it is now small enough not to matter.

**Slot count barely matters**: 4.86 / 4.81 / 4.74 / 4.92 / 5.21 ms per memory step at 64 / 128 / 256 / 512 / 720 slots, so the O(K²) exchange is not what the cost is made of — and E6 removed that exchange altogether for nothing.

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
| **TAKHeLiPeD, stage 2 alone** (frozen) | **2.13** (sd 0.03) | **85.6** | **61.4%** | — |
| **TAKHeLiPeD, whole detector** (frozen, **calibrated**) | **0.596** (sd 0.045) | **23.9** | **29.4%** | — (the memory is the filter) |
| *the same, uncalibrated* | *0.675 (sd 0.073)* | *27.0* | *31.9%* | — |
| *the same two, superseded architecture, 2026-10-09* | *2.62 (sd 0.20) / 0.74 (sd 0.14)* | *105.1 / 29.7* | *67.2% / 34.1%* | — |
| SpaceTimeCNN | gap | gap | gap | gap |
| FullScanTCN *(retired)* | gap | gap | gap | gap |
| TemporalUNet *(retired)* | gap | gap | gap | gap |

0.3 is LFE-PPN's own published operating point.

**The whole-detector row is the calibrated one since 2026-10-31**, and it is a fixed-threshold row, so the calibration's
`(temperature, bias)` moves it where it cannot move an AP.
The stage-2 row is **bit-identical** either way -- the key is in stage 3 and stage 2's candidates come from an untouched
network -- which is the control that makes the whole-detector change the memory's and nothing else's.
**At threshold 0.9 the same calibration goes the other way**, and badly: 0.0015 FP/frame becomes **0.0122**, 106 detections
across the three seeds becoming 883, because dividing by a temperature below 1 lifts the confident tail as surely as it
lowers the bulk.
Seed 1 emitted **zero** detections above 0.9 on all 72,533 person-free frames uncalibrated and emits **767** calibrated.
Against stage 2's own 0.666 at that threshold the qualitative claim survives -- -98.2% rather than -99.8% -- but the number
is 8x its old value, and nothing but measuring it would have said so: at 0.3 the prediction was right and at 0.9 it was
wrong (`CHANGELOG.md`).
The three `gap` rows are the pre-TAKHeLiPeD architectures and stay gaps: they belong to the superseded line of work, and nothing in it is queued.

**`FullScanTCN` and `TemporalUNet` are marked *retired*, not deleted, throughout this file.**
Their code was removed on 2026-10-31 -- nothing had trained on either since 2026-09 and neither influenced a
project decision -- and it is marked rather than deleted because **a measurement stays true after its code goes**.
What is lost is reproducibility, not validity: these rows can no longer be re-run from this tree, only from commit
**`98b0716`**, the last one that carries the classes. `SpaceTimeCNN`'s rows stay reproducible, which is why that one was
kept.

**LFE-PPN's row was stale here until 2026-10-09**, at the pre-recalibration **1.94 / 50.9**; the 2026-09-11 recalibration of `nms_radius` from 0.8 m to 0.30 m took it to **3.174 /frame** and `CHANGELOG.md` recorded that, but this table kept the old figure while its *frames-with-a-phantom* cell was updated to the new 64.0%.
The "with SORT" cell is stale for the same reason and is not reconstructed here: `tracker_sweep.py` was re-run, and at `min_hits = 3` it leaves 34.7% of person-free frames with a phantom against the old 47.1%, but no corrected FP/frame was recorded.
**A row that is half-updated is worse than one that is wholly stale**, because nothing about it looks wrong.

### TAKHeLiPeD, the frozen architecture, measured 2026-10-30

Three seeds of the **frozen** architecture — the aggregator-free stage 2 on the soft Gaussian target paired with its own `dim 64` memory — over all 72,533 person-free frames, `utils/empty_scene.py`, `results/empty_scene_frz_s{0,1,2}.json` (archived as `weights_archive/frozen-frog-2026-10-30/results/empty_scene_s{0,1,2}.json`):

| at threshold 0.3 | stage 2 alone | whole detector | change |
| --- | --- | --- | --- |
| FP/frame | 2.134 (2.106 / 2.156 / 2.139) | **0.675** (0.706 / 0.591 / 0.727) | **−68.4%** |
| frames with ≥1 | 61.4% | **31.9%** | −29.5 pp |
| recur within 0.3 m next frame | 85.8% | 93.6% | +7.8 pp |
| FP/frame, the 41,797 frames where nobody has been in view for 10 s | 1.984 | **0.524** | **−73.5%** |
| FP/frame at threshold 0.9 | 0.666 | **0.0015** | −99.8% |
| frames with ≥1 at threshold 0.9 | 32.7% | **0.15%** | −32.5 pp |

**Every qualitative finding of the 2026-10-09 measurement below survives the architecture change**, which is the point of repeating it: the memory is a phantom filter, the reduction is *larger* beyond the fade horizon so it is not stale-memory bookkeeping, what survives recurs next frame and is therefore furniture, and the high-confidence count is categorically better than any published detector's.

**The absolute count improved and the relative cut did not, and the absolute one is what matters.** 0.675 against the superseded model's 0.740 is fewer phantoms, but the cut reads 68.4% against 72% because the frozen stage 2 hands the memory **fewer candidates to begin with** (2.134 against 2.622). A ratio against a different denominator is not a like-for-like comparison; a robot sees the numerator. The same applies to the 0.9 row, where the frozen detector's **0.0015 FP/frame** is about 7x better than the superseded 0.010 — 298, **0** and 20 detections in 72,533 frames each, and one seed emitting none at all.

**Coasting contributes exactly zero, by construction rather than by measurement.** The frozen architecture has no coasting head, so `from_coasting` is 0 at every threshold in every seed. That makes the phantom audit's reading exact instead of merely small: **every** person-free phantom is a candidate stage 2 emitted and the memory failed to suppress. The 2026-10-09 figures below measured the same thing on a model that *had* the head, and found 2, 0 and 0 reports at the 0.3 threshold — which is what justified removing it.

**This column is far noisier across seeds than AP is**, and a single seed should not be quoted from it. The three phantom counts spread **0.136 FP/frame** around a 0.675 mean (sd 0.073, about 11% relative), while the same three seeds' AP spread 0.32 pp around 86.29% (0.2% relative) — fifty times tighter. Seed 1 is the best on phantoms (0.591) and middling on AP; seed 2 is the best on AP (86.43%) and the worst on phantoms (0.727), consistent with seeds placing their AP-optimal threshold differently rather than with any seed being better. Report the mean.

### TAKHeLiPeD, measured 2026-10-09

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

**Where those phantoms come from: the memory rescores them, it does not invent them** (2026-10-12, all three
seeds, `coasting_share` in `empty_scene.py`). At the 0.3 operating threshold the coasting head contributes **2, 0 and 0**
reports against **61,870 / 57,188 / 42,043** rescored candidates, and exactly **zero** at 0.5 and 0.9; coasting appears at all
only below 0.2 (0.3% of detections) and reaches 5.0% at 0.1. So every person-free phantom worth discussing is a candidate
stage 2 emitted and the memory failed to suppress -- which is the reading the person-free work rests on, now measured rather
than inferred from the populated split's ~0.001 coasting reports per frame.

**And a creation or confirmation rule cannot reach them** (2026-10-12, seed 0, `utils/slot_confirmation.py`,
`results/slot_confirmation_s0.json`). The creation-rule work proposed confirming a slot over several frames before it may report. The slots
behind the surviving phantoms are **old**: median **180 frames (4.5 s)**, quartiles 75/353, p95 728 -- and barely younger than
the slots behind real people, median **221 frames**. A confirmation rule filters *young* slots, so it is aimed at a population
that is not there:

| a slot must be k frames old to report | delay | phantoms cut | person frames lost | selectivity |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 25 ms | 0.2% | 0.1% | 1.39x |
| 5 | 100 ms | 1.0% | 0.8% | 1.39x |
| 13 | 300 ms | 3.2% | 2.4% | 1.34x |
| 21 | 499 ms | 5.8% | 3.8% | 1.52x |
| 40 | 973 ms | **12.6%** | **8.0%** | 1.57x |

**It is barely selective at any delay.** A full second of confirmation removes 12.6% of phantoms -- 0.740 per frame to about
0.65 -- while suppressing 8.0% of the frames on which a real person is reported, and at a deployable 100 ms it removes 1.0%.
**The frame-to-frame recurrence rate badly overpredicts this**: 92.2% recurrence suggests 0.922^4 = 28% of phantoms would fail
a 5-frame test, against a measured **1.0%**, because furniture's recurrence is strongly correlated across frames rather than
independent. That gap is the finding -- persistence is not a lever on this population, because the phantoms are the *most*
persistent thing the detector sees.

**What the memory keeps is the furniture.** Its surviving phantoms recur next frame **92.2%** of the time against the candidates' 84.9% and real people's 99.8%: it removes the unstable ones and keeps the ones that look like a person in every frame, which is the failure mode neither classical map could remove.
That is a creation-rule question rather than a scoring one, and it is queued.

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

The `gap` rows are ours to fill and are **not** fillable by pointing our current checkpoints at this split: they were selected against a different training set, and the mode control holds training data constant before any such comparison counts.

## Speed — ms/frame on this host, one protocol

**Re-measured 2026-10-11 under one protocol** (`utils/cpu_latency.py`). That is the point of this table
rather than a detail of it: the rows below it were produced by **two different scripts**, on synthetic input, at an unstated
thread count and with no percentile, so they could not be compared with each other — the one thing a speed table exists to
allow. `AGENTS.md` rule 2b makes speed half of every accuracy answer, and half an answer cannot come from rows measured under
conditions nobody wrote down.

**The protocol.** Every detector sees the same **2,000 consecutive FROG test scans, in order, one at a time, batch 1, on four
CPU threads**, after an untimed warm-up of five. What is timed is **scan in, detections out**, split into the phases a
deployment pays for separately — building the input (cutouts, or the aligned raw scan), the forward pass, and decoding scores
and votes into detections with `evaluate.py`'s own vote grid or peak rule, whichever that head uses. A detector whose released
form does all three inside one call — the two LFE ONNX graphs, the C++ `AlgorithmicDetector` — is timed as one phase and its
row leaves the rest blank. **Each total is that frame's own phases added, percentiled afterwards**, never the phases'
percentiles added: the frame that is slow to preprocess is rarely the frame that is slow to decode, and adding their separate
tails describes a frame that never happened (it read our own tail as 27.90 ms against a measured 25.32). The budget column is
the **tail** against FROG's real **25 ms** period, which is the measured one — the stamped median said 26.2 Hz and made
every budget 1.53x too generous.

**CPU, 4 threads, batch 1, 2,000 consecutive FROG test scans, against FROG's 25 ms frame period.**

| detector | pre | network | decode | memory | **total** | p99 | p99 vs budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AlgorithmicDetector *(rule-based, C++)* | -- | 0.05 | -- | -- | **0.05 ms** | 0.07 | **0%** |
| LFE-Peaks | -- | 1.14 | -- | -- | **1.14 ms** | 1.82 | **7%** |
| LFE-PPN | -- | 2.06 | -- | -- | **2.06 ms** | 3.65 | **15%** |
| TemporalUNet *(retired)* | 0.13 | 5.27 | 0.07 | -- | **5.48 ms** | 7.21 | **29%** |
| FullScanTCN *(retired)* | 0.13 | 3.71 | 1.66 | -- | **5.52 ms** | 7.91 | **32%** |
| SpaceTimeCNN | 0.13 | 14.36 | 1.75 | -- | **16.29 ms** | 25.93 | **104%** |
| **TAKHeLiPeD**, whole detector | -- | 8.44 | 1.51 | 9.14 | **19.15 ms** | 22.07 | **88%** |
| DrowDetector (DROW3) | 2.25 | 317.33 | 1.34 | -- | **320.93 ms** | 339.86 | **1359%** |
| DrSpaamDetector (DR-SPAAM) | 2.52 | 1191.46 | 1.89 | -- | **1195.88 ms** | 1316.46 | **5266%** |
| Li2FormerDetector | 2.86 | 1670.19 | 1.98 | -- | **1675.24 ms** | 1758.21 | **7033%** |

**Six detectors fit the sensor and four do not, and TAKHeLiPeD is on the right side of the line.** At the 99th
percentile `AlgorithmicDetector`, both LFE graphs, `TemporalUNet`, `FullScanTCN` and **the proposed detector at 88%** all
finish inside 25 ms. It got there by being made to fit rather than by being measured generously — the same row read **101%** until the
candidate decode was restructured — and the comparison worth stating is with `SpaceTimeCNN`, the architecture it replaced,
which **does not fit, at 104%**: ours additionally carries an object memory costing **9.14 ms** and a candidate decode, and
still has both the better median and the better tail.

**The cutout detectors are 13x to 70x over budget, and that is structural rather than an implementation failing.** They run one
forward pass **per beam**: at FROG's 720 beams that is **3,600 cutouts per frame** for DROW3 at its `T=5`, **7,200** for
DR-SPAAM at the `T=10` its released config specifies, and 3,600 for `Li2FormerDetector`. The formulation is what costs, not the code —
and comparing DROW3 with it isolates that, since both encode the same 3,600 cutouts and the transformer behind them costs
**5.2x**. **Read that row as this project's class and not as Li2Former** (2026-11-01): `Li2FormerDetector` has one
temporal encoder layer and no cross-beam path, where the paper cascades omni-dimension aggregation blocks, so the published
model is **slower than this row, not equal to it**. The authors' own network is `OfficialLi2FormerDetector` and **will not be
timed here**: their code builds 16.85 M parameters in the mode trained here and 29.47 M as their `__init__` builds it, so there
is no single "their time" to measure (dropped 2026-11-03, `rejected-ideas.md`). At batch 1 in training it is ~3.5x our class's
CPU cost per frame, which fixes the ordering without a number. **Building the cutouts is itself 17-22x the cost of building a raw scan** (2.25-2.86 ms
against 0.13), before a single multiply of the network. This is the table's most important lesson and the reason this project's
detector reads whole scans.

**Three caveats that the rows cannot carry themselves.**

- **DR-SPAAM's row is its windowed form, and overstates what the published detector costs.** It encodes ten scans per frame
  here; the released streaming detector encodes **one** and attends to a carried template, so a streaming row would be a
  fraction of this. The same distinction runs the other way on accuracy — streaming is worth **+1.24 pp** to it — so **our tables understate its accuracy and overstate its cost at the same time**. `DrSpaamDetector.stream()` exists;
  timing it was not in that sweep's scope and is the obvious follow-up.
- **`AlgorithmicDetector`'s row includes its own console output.** The C++ detector prints two lines per call ("N legs have
  been detected", "N persons have been detected") from inside the timed window, and `verbose=False` does not suppress them —
  3,933 lines over 2,000 frames. Against a redirected file that is cheap and the row reproduces exactly (0.05 ms across two
  independent 2,000-frame runs); against a live console it would be far worse, which is a real fact about deploying it.
- **Li2Former's row is inference only.** It is the one detector never trained here, so it has a speed row and no accuracy row
  of ours. Its **1,675 ms** is also 2.8x the 590.6 ms the superseded table carried, most plausibly the
  thread count — that harness ran at torch's default 8 — which is one more reason those rows are not comparable with these.

**The paper GPU figures below are not a ranking column.** They are kept to show the hardware gap, and no row here should be
read against them.

<details>
<summary><strong>The superseded rows, kept: they are the only numbers for the DROW and JRDB beam counts, and their provenance
is now known</strong></summary>

**Why they are superseded rather than reconciled.** Traced 2026-10-11 to their two source scripts, they differ from the
protocol above on **five axes at once**, so no per-detector difference can be attributed to any one of them:

| axis | these rows | the one protocol |
| --- | --- | --- |
| input | synthetic random (`np.random.uniform`, `torch.randn`), one scan repeated 100x | 2,000 consecutive real FROG test scans, in order |
| scope | the forward pass only, for the six torch rows | scan in -> detections out, phases separated |
| threads | torch's default, unstated — **8** on this host | **4**, stated |
| statistic | a mean and a standard deviation, no percentile | median, p99, and the p99's share of the 25 ms period |
| window | one `--time-frame`, default **5**, for every cutout detector | each detector's own (DROW 5, DR-SPAAM **10**, Li2Former 5) |

The three non-torch rows came from **`utils/_bench_manual.py`**, which times 100 iterations of a single uniform-random scan —
and those are the rows whose cost is most data-dependent, since the C++ detector's work scales with how many leg clusters a
scan holds (real FROG frames hold 10-24) and LFE-PPN's with how many anchors survive to its greedy merge. The six torch rows
came from **`utils/evaluate.py --bench`** (`_build_bench_models`, `bench_model`), which never calls `torch.set_num_threads`
and passes one `T` to every detector — which is why DR-SPAAM's old row ran at **half the window its own configuration
specifies**, and why the old footnote 10 below describes the measurement accurately and the detector wrongly.
**Note the direction:** these rows had *twice* the threads and did *less* work per frame, so they were bound to read lower.
A higher number above is not a regression.


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
| **FullScanTCN** *(retired)* | **2.1 ms** | **2.6 ms** | **3.1 ms** |
| **TemporalUNet** *(retired)* | **2.0 ms** | **2.7 ms** | **2.3 ms** |
| **TAKHeLiPeD, stages 1-2 + decode** | n/a | **~10.4 ms**⁶ | n/a |
| **TAKHeLiPeD, stage 3 alone** (object memory) | n/a | **4.4-4.6 ms**⁷ | n/a |
| **TAKHeLiPeD, whole detector streaming** (the frozen architecture) | n/a | **19.3 ms** *(4 threads, p99 21.7, cold host)*¹² *(the same on a warm host 19.4, p99 21.8; superseded architecture 20.8 ms, p99 24.1⁹; old recipe 19.5 ms, GPU 12.9 ms⁸)* | n/a |

¹⁰ **The windowed form**, which encodes five scans per frame; the published streaming detector encodes one and attends to its
carried template, so this row overstates it. Re-timed under one protocol.

⁴ Was **315.5 / 243.8 / 278.3 ms** here until 2026-09-10, and blamed on the Python decode loop.
It was not the loop: the decoder applied a second sigmoid to a channel `lfe_ppn.onnx` had already sigmoided, so every one of the 3,600 anchors cleared any threshold and entered an O(n²) greedy merge, 182 survivors per frame on scenes holding ~3 people.
With that fixed LFE-PPN lands within 0.1 ms of its paper's **GPU** figure, on CPU — and the Python decode loop is dissolved rather than done.

⁵ The old JRDB figures were measured with that same bug and are withdrawn rather than corrected; JRDB is not downloaded here (`gotchas.md`), so nothing can replace them yet.
Both LFE rows are otherwise real-data measurements (300 scans each) taken after the fix, so they stay methodologically comparable to each other.

⁶ Derived 2026-09-16 from step 3a's candidate caches, which stream the trained stages 1-2 one frame at a time (batch 1) and decode candidates with DROW's vote grid: 1,269 s for 120,396 train frames, 2,010 s for 195,058 val, 521 s for 50,088 test -- 10.3-10.5 ms per frame, consistently.
**Not comparable with the rows above**: those are CPU measurements on this machine, while this one runs on the RX 9060 XT under ROCm, carries a Python decode loop, and leaves out stage 3 (the object memory).
It is quoted only against the sensor: FROG's measured mean frame period is 24.99 ms (~40 Hz, corrected 2026-09-17 from a long-standing 26.2 Hz), so stages 1-2 plus the memory's 4.4 ms still stream in real time, with roughly a third of the budget spare. A proper single-frame latency benchmark of the whole detector, on one machine with the other rows, is the post-FROG programme's.

⁷ Measured 2026-09-17 and confirmed across B0's three seeds (2026-09-19) by `memory_ms_per_frame`, timing one stream on the RX 9060 XT under ROCm: **4.42 ms** on the fixed end-of-run block, 4.5-4.6 ms across the seeds. Same caveat as footnote 6 -- not comparable with the CPU rows above.

¹² **The freeze, three seeds, 2026-10-30** (`a71_freeze_s0`, `a71_freeze_s1`, `a71_freeze_s2`; FROG `official` test, every 5th annotated frame; full entry in `CHANGELOG.md`): the whole detector **86.29%** (86.11 / 86.33 / 86.43), sample sd **0.17 pp**, over candidates of **82.80%** — the memory worth **+3.49 pp**. AP@0.3 **85.19%**. At the AP-optimal threshold P 73.2% / R 84.8% / 0.953 false positives per frame; at each seed's own threshold recall is **95.2%** (95.0 / 95.3 / 95.2), which is the deployment-relevant operating point and the figure to quote when a threshold is being chosen rather than swept. **This is the architecture the project now proposes** and it is *not* footnote 11's: aggregator-free stage 2 trained against the **soft Gaussian beam target** (σ = 0.3 m), stage 3 at 256 slots, **`dim 64`**, 4 heads, **minGRU** recurrence, no slot exchange, no learned matching, no coasting head, no chunk curriculum. Against footnote 11's 86.01% that is **+0.28 pp**, inside the 1.32 pp bar — **this row replaces it because it measures the proposed model, not because it is better**, and the honest summary of the freeze is "confirmed". CPU **19.32 ms median / 21.67 p99** at 4 threads, batch 1, under the matched interleaved protocol on a host idle for 63 minutes; the warm sweep that first produced this row read **19.39 / 21.78**, so the figure is **measured twice in two independent sessions and reproduces to 0.4%**. The `dim 128` control, interleaved in both, costs 22.34 ms median and 24.65 ms p99 — the memory phase falls 10.08 → 7.23 ms (−28.3%) and the total −13.5%, matching the width halving's −28% and −11.6% on the same two shapes. Cold, the `stage2` phase spreads only **0.084 ms** across arms, so the floor is four times tighter than the warm sweep's 0.36 ms. **Still not comparable with footnote 9's 20.79 ms**, and the discrepancy with the older `stage2` levels (9.35 ms on 2026-10-27, 8.44 ms on 2026-10-11, the latter on the *aggregator-ful* network, which does more work) is **narrowed but unexplained**: it is not thermal, since the cold host reproduces the level, and not the stage-2 checkpoint, since `a69_no_aggregator_s0` and `a71_soft_label_s0` are both 960,323 parameters and both aggregator-free and therefore identical at inference. Something in the environment changed; every conclusion drawn rests on within-session comparisons, which that cannot touch.

¹¹ **The frozen architecture, three seeds, 2026-10-19** (`a64_no_coast_s0`, `a65_frozen_s1`, `a65_frozen_s2`; FROG `official` test, every 5th annotated frame): **86.01%** (86.08 / 86.08 / 85.88), 0.958 false positives per frame, recall 84.03%, over candidates of 82.19% -- the memory worth **+3.82 pp**. This is the model `memory/detector-architectures.md` freezes: the C1 stage-2 recipe over a fixed-beam U-Net, E13's memory, and **no coasting head**. The 85.9% of footnote 9 is the same model *with* that head -- 0.14 pp apart, inside the 1.32 pp three-sigma bar. The seed spread fell from 0.86 to 0.21 pp, which is suggestive and **not** a claim three seeds can support.

⁹ 2026-10-08: E13's stage 3 (no attention) on the no-temporal stage 2 trained with the adopted recipe -- `nms-mean` decoding, mirror augmentation, nearest-person labels, dropout 0.1, weight averaging (stage 2 alone 82.20%) -- three seeds, every 5th annotated test frame: **86.35 / 85.77 / 85.49%, mean 85.87**, AP@0.3 84.68%. Against E13 on the old recipe, 83.86%, +2.01. Stage 2 alone is the no-temporal network averaged over its val-chosen epochs, 82.28 / 82.08 / 82.22% (mean 82.20). Streamed one scan at a time on this machine's CPU at 4 threads (`tracking_baselines.py latency`, 2,000 frames, 2026-10-08): stage 2 8.75 ms, decoding 2.77 ms (`nms-mean`'s per-voter loop, ~1 ms more than the vote grid), memory 9.22 ms, **20.79 ms median, p99 24.08** -- inside FROG's 25 ms period with under a millisecond to spare at p99. The tracking comparison above is on the old recipe's candidates and has not been repeated.

⁸ E22, 2026-10-03: `tracking_baselines.py latency`, one raw scan at a time through stage 2, DROW vote decoding and the E13 memory, 2,000 frames of FROG test, median (p99). CPU on this machine with 4 threads: **8.60 + 1.76 + 9.12 = 19.53 ms (22.38)**, 8 threads 19.47 ms (22.77); RX 9060 XT under ROCm: **5.76 + 2.27 + 4.82 = 12.92 ms (15.11)**. The CPU figure is the one comparable with this column, and it fits FROG's 25 ms scan period at the 99th percentile. The memory's share here includes moving each frame's tensors from numpy, which footnote 7 leaves out.

</details>

**Exact parameter counts**, counted from the ONNX initializers now that `onnx` is installed, replacing the file-size estimates this table used to carry (~65K / ~180K): **LFE-Peaks 53,258**, **LFE-PPN 170,903**.
For scale, `SpaceTimeCNN` is 476,262 — **2.8x LFE-PPN, for +0.5pp on the benchmark and no use of temporal order at all.**

## Where the other tables went

- **The `(T, dtime)` informational-capacity grid** now lives with its own methodology, in [`informational-capacity-proxy.md`](informational-capacity-proxy.md).
- **Per-frame ego-motion and person-displacement measurements** are dataset properties, not model results — [`data-model.md`](data-model.md).
- **The `dtime`, NMS, operating-threshold and tracker sweeps** were all measured on the contaminated split and are retracted.
  Their numbers and the reasoning that produced them stay in [`../CHANGELOG.md`](../CHANGELOG.md); they are not settings to restore, they are runs to redo.
