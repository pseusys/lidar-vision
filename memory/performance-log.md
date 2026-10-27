# Performance log

*keywords:* wp-AUC, AP, ms/frame, SOTA, comparison table, published baseline, PeTra, Li2Former, LFE, DR-SPAAM, DROW3, state of the art, Aycard, AlgorithmicDetector, attribution, credit

The standing comparison: published state of the art against this project's own architectures, **one row per model, latest measurement only**.

**History does not live here.** How a number moved, which run produced it, and every sweep that led to a setting are in [`../CHANGELOG.md`](../CHANGELOG.md) — grep its `*keywords:*` lines.
Read [`interpreting-evaluation.md`](interpreting-evaluation.md) before quoting any row; several are not comparable to each other without its caveats.

## How a cell gets filled

**A cell holds the original authors' published number wherever they published one** (owner's call, 2026-09-10).
This project evaluates a third-party model only to **fill a gap** — a model/dataset pair nobody published — never to re-derive a number that already exists.
Re-deriving a published accuracy figure through this pipeline adds risk without adding information; this session's own audit is the case in point.

**Speed is the exception.** `ms/frame` is only comparable when every row was measured on one machine **and under one protocol**, so every timing number here is ours by necessity. The second half of that was only true from 2026-10-11 (item 29): before it, the table's rows came from two scripts at two thread counts on synthetic input.

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
| **SpaceTimeCNN** | ours, superseded | not measured | **63.0%**³ | n/a |
| **FullScanTCN**, **TemporalUNet** | ours, superseded | not measured | not measured | n/a |
| **TAKHeLiPeD, stages 1-2** (calibration network) | ours, novel | gap | **82.2%**⁹ *(old recipe 77.7%⁴)* | n/a |
| **TAKHeLiPeD, stages 1-3** (+ object memory) | ours, novel | gap | **86.0%**¹¹ *(with the coasting head 85.9%⁹; old recipe 83.3%⁵)* | n/a |

**DR-SPAAM `T=5` at 75.6% on FROG was the number to beat, and the frozen detector clears it by 10.4 pp.**
That margin was **7.7 pp** while this file still led with the old recipe's 83.3%, and the sentence outlived the number by three weeks -- which is the failure `TODO.md` item 4 existed to fix and the reason each cell here now carries the configuration it was measured on.
That is ours measured here, three seeds, against theirs published, one paper figure — read footnote 11 and [`interpreting-evaluation.md`](interpreting-evaluation.md) before quoting it.

**The comparison a tracking reviewer will ask for is within one model, on identical candidates** (`TODO.md` A61, 2026-10-09, on the adopted recipe; one replay per stage-2 seed; FROG test, every 5th annotated frame, 95% block-bootstrap intervals per seed; tables in [`experiments/2026-09-tracking-claims.md`](experiments/2026-09-tracking-claims.md)):

| what reads TAKHeLiPeD's stage-2 candidates | AP, 3 seeds | vs the candidates |
| --- | --- | --- |
| nothing | 82.19% | -- |
| SORT (`min_hits` 1) / ByteTrack, as published | 82.31% / 81.17% | +0.13 / -1.02 |
| ByteTrack + CBMOT track confidence, tuned on val | 83.51% | +1.32, every seed's interval above zero |
| **object memory, no attention (E13), frozen** | **86.01%** | **+3.82; +2.36 over the tuned tracker (+2.84 / +2.31 / +1.94), every seed's interval above zero** |

At one matched false-positive rate the two now find the same people (95.47 against 95.56%); the memory's lead is ranking, not recall at a threshold. On the old recipe (A56, 2026-10-03, vote-grid candidates) the rows read 77.66 / 77.40 / 76.49 / 80.96 / 83.86%, the memory +2.90 over the tuned tracker. All on FROG's official test labels, which include 2,254 duplicate circles that only a second detection can match; on the old recipe, without them every row rose 0.7-1.05 pp and the memory's margin over the tuned tracker was +2.71, still above zero at every seed ([`interpreting-evaluation.md`](interpreting-evaluation.md), 2026-10-05).

**TAKHeLiPeD** (`T`emporal `A`daptive `K`nee-`He`ight `Li`dar `Pe`rson `D`etector, named 2026-09-23) is what earlier entries in this file, `CHANGELOG.md` and `TODO.md` call *the three-horizon detector*. **The code followed on 2026-10-11** (item 9 / A52): the package, the C++ namespace, the ROS package and every script say `takheliped`. Only `checkpoints_three_horizon/` keeps the old name, deliberately -- each checkpoint stores the absolute path of its own stage 2.

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
³ **63.0%** single-frame, the figure A35's acceptance criteria were written against and `docs/PAPER.md` argues from (2026-09-10's history ablation). These three are the project's own first line of work, superseded by TAKHeLiPeD and trained on nothing since 2026-09; `TODO.md` item 32 keeps `SpaceTimeCNN` so the decisions made on it stay reproducible and removes the other two. The `T=5, dtime=10` re-measurement this footnote used to promise was never run and is not planned: a temporal window is not what the project proposes any more.
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
- **the speed table**, all of it, because one machine and one protocol are the only way those rows compare;
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
where the published one runs them once and attends to its carried template, so the **1,195.88 ms** FROG row below *overstates* the
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

## Why stage 2 beats LFE and DR-SPAAM: what it is *not*

**What the lead is made of, measured on fixed weights** (A59 phase 1; FROG `official` test, 0.5 m match radius). Each row changes one thing and keeps everything above it, so the deltas add to the whole gap:

| | wp-AUC | step |
| --- | --- | --- |
| LFE-Peaks, as its authors published it | 65.11% | — |
| **our network, read through LFE's own peak decoder** | 71.26% | **+6.15** |
| its peaks reported at the regressed vote, not the beam's endpoint | 74.10% | +2.84 |
| DROW's vote grid instead of per-beam peaks | 77.27% | +3.17 |
| DR-SPAAM's NMS at 0.5 m | 79.88% | +2.61 |
| A59's adopted training recipe (`nms-mean`, mirroring, nearest-person labels, dropout, weight averaging) | **82.20%** | +2.32 |

**Half the lead is the output head.** The three decoding steps are worth **+8.62 pp** together — 50% of the 17.09 pp gap — against **+6.15** for swapping in our network while keeping LFE's decoder, and **+2.32** for the training recipe on top.
So *how a detection is reported* matters more here than *what network produces it*: given identical per-beam outputs, reporting one peak per probability maximum and merging at 0.30 m leaves about 8.6 points on the table.

**Read the +6.15 honestly.** It is our *trained* network through their decoder, so it bundles the network's form with our training, and the ladder below is what narrows that: LFE's input features are worth nothing to us (+0.14), its widths cost 1.50 pp, both together 1.58, and DR-SPAAM's whole training protocol 1.84 — so the +6.15 is neither the input, nor the capacity, nor the optimiser.
What it is instead — the network's form, a ConvNeXt U-Net with a whole-scan aggregator, against the training that produced LFE's own weights — is the one split this study does not make (`TODO.md` §A item 7, dropped).

**A59's attribution ladder, completed 2026-10-11.** Each arm takes one thing from a published baseline and leaves everything else in the adopted recipe, three seeds, weights averaged, `nms-mean` decoding, every 5th annotated FROG test frame.

| arm | what it borrows | wp-AUC | against C1 |
| --- | --- | --- | --- |
| **C1** | nothing: the adopted recipe | **82.20%** | — |
| F1 | LFE's input: `1 - r/10` clipped to 0.2-10 m, as the only channel | 82.34% | **+0.14** |
| W | LFE's widths, 8/16/32 (63,891 parameters) | 80.70% | **−1.50** |
| **F1 + W** | both at once | **80.62%** | **−1.58** |
| R | DR-SPAAM's training protocol, last weights | 80.11% | −1.84 (of which −1.24 is taking the last weights, not the best on val) |

**The two costs do not add, because only one of them is real.** F1 + W lands within **0.08 pp** of W alone, against an additive prediction of 80.84: LFE's input contributes nothing on top of its width, exactly as it contributed nothing alone.
**So the width is the whole cost — and it is small.** At LFE's own 64k parameters, with LFE's own input, our network still scores **80.62% against LFE-Peaks' 65.1%**: a 15.5 pp gap that no ingredient in this table explains.
Width's 1.5 pp is also entirely on far people, which is what makes range anchors the best-motivated experiment left (`TODO.md` §A item 13).

**CPU latency, `dim 128` against `dim 64`, matched and interleaved (2026-10-27, `TODO.md` item 49).** Item 29's
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

**What the ladder leaves.** With phase 1's decomposition — about half the lead over LFE-Peaks is the output head: centre votes, aggregation, NMS — what remains is **the network's form and the training recipe**, and those two have not been separated. **Nor has any part of the form been shown to carry it** (2026-10-25): block and widths (items 19a, 19b), depth, kernel and expansion (item 45) and the whole-scan aggregator (item 40) all tie, so this half of the lead currently has no identified owner. Doing so needs LFE's architecture trained the way ours is, which was attempted and dropped (`TODO.md` §A item 7): 29 convolutions reproduce in order from the released graph, but two blocks' dataflow did not resolve without a shape-inference pass. **The pre-print states this as a limitation rather than a result.**

## TAKHeLiPeD's cost, end to end, one scan at a time

**Re-measured 2026-10-11 on the adopted recipe** (`tracking_baselines.py latency`, E22 / item 30; 2,000 consecutive test
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

**Both now have headroom, and the CPU only got it in item 30.** The same CPU row read **20.98 ms, p99 25.32 — 101% of the
budget** until the candidate decode was restructured: it ran one Python iteration per voting beam, 314 per frame to keep 22
detections, and built a 314x314 distance matrix to do it. Iterating only the beams still alive took decoding from **2.83 ms to
1.51** (p99 5.47 to 2.44) and the whole detector from 101% to **88% at p99**. The output is **bit-identical**, which is what
makes it free: no checkpoint, no candidate cache and no accuracy number moves, and a randomised test pins the new loop against
a transcription of the old one in both decoding modes (`tests/test_stage2_probe.py`).

**Thread scaling saturates at two, and the memory does not scale at all** (item 30, same 2,000 scans and checkpoints):

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
total** it is now the largest term and the only remaining lever worth pulling (slot compaction, `TODO.md` item 30). Decoding is
likewise flat (1.47-1.53), being numpy and Python, but it is now small enough not to matter.

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

**Where those phantoms come from: the memory rescores them, it does not invent them** (item 27, 2026-10-12, all three
seeds, `coasting_share` in `empty_scene.py`). At the 0.3 operating threshold the coasting head contributes **2, 0 and 0**
reports against **61,870 / 57,188 / 42,043** rescored candidates, and exactly **zero** at 0.5 and 0.9; coasting appears at all
only below 0.2 (0.3% of detections) and reaches 5.0% at 0.1. So every person-free phantom worth discussing is a candidate
stage 2 emitted and the memory failed to suppress -- which is the reading items 24, 25 and 26 rest on, now measured rather
than inferred from the populated split's ~0.001 coasting reports per frame.

**And a creation or confirmation rule cannot reach them** (item 14, 2026-10-12, seed 0, `utils/slot_confirmation.py`,
`results/slot_confirmation_s0.json`). Item 14 proposed confirming a slot over several frames before it may report. The slots
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

## Speed — ms/frame on this host, one protocol

**Re-measured 2026-10-11 under one protocol** (`utils/cpu_latency.py`, `TODO.md` §A item 29). That is the point of this table
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
the **tail** against FROG's real **25 ms** period, which is the one item 1 settled — the stamped median said 26.2 Hz and made
every budget 1.53x too generous.

**CPU, 4 threads, batch 1, 2,000 consecutive FROG test scans, against FROG's 25 ms frame period.**

| detector | pre | network | decode | memory | **total** | p99 | p99 vs budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AlgorithmicDetector *(rule-based, C++)* | -- | 0.05 | -- | -- | **0.05 ms** | 0.07 | **0%** |
| LFE-Peaks | -- | 1.14 | -- | -- | **1.14 ms** | 1.82 | **7%** |
| LFE-PPN | -- | 2.06 | -- | -- | **2.06 ms** | 3.65 | **15%** |
| TemporalUNet | 0.13 | 5.27 | 0.07 | -- | **5.48 ms** | 7.21 | **29%** |
| FullScanTCN | 0.13 | 3.71 | 1.66 | -- | **5.52 ms** | 7.91 | **32%** |
| SpaceTimeCNN | 0.13 | 14.36 | 1.75 | -- | **16.29 ms** | 25.93 | **104%** |
| **TAKHeLiPeD**, whole detector | -- | 8.44 | 1.51 | 9.14 | **19.15 ms** | 22.07 | **88%** |
| DrowDetector (DROW3) | 2.25 | 317.33 | 1.34 | -- | **320.93 ms** | 339.86 | **1359%** |
| DrSpaamDetector (DR-SPAAM) | 2.52 | 1191.46 | 1.89 | -- | **1195.88 ms** | 1316.46 | **5266%** |
| Li2FormerDetector | 2.86 | 1670.19 | 1.98 | -- | **1675.24 ms** | 1758.21 | **7033%** |

**Six detectors fit the sensor and four do not, and TAKHeLiPeD is on the right side of the line.** At the 99th
percentile `AlgorithmicDetector`, both LFE graphs, `TemporalUNet`, `FullScanTCN` and **the proposed detector at 88%** all
finish inside 25 ms. It got there in item 30 rather than by being measured generously — the same row read **101%** until the
candidate decode was restructured — and the comparison worth stating is with `SpaceTimeCNN`, the architecture it replaced,
which **does not fit, at 104%**: ours additionally carries an object memory costing **9.14 ms** and a candidate decode, and
still has both the better median and the better tail.

**The cutout detectors are 13x to 70x over budget, and that is structural rather than an implementation failing.** They run one
forward pass **per beam**: at FROG's 720 beams that is **3,600 cutouts per frame** for DROW3 at its `T=5`, **7,200** for
DR-SPAAM at the `T=10` its released config specifies, and 3,600 for Li2Former. The formulation is what costs, not the code —
and comparing DROW3 with Li2Former isolates it, since both encode the same 3,600 cutouts and Li2Former is **5.2x** slower for
putting a transformer behind them. **Building the cutouts is itself 17-22x the cost of building a raw scan** (2.25-2.86 ms
against 0.13), before a single multiply of the network. This is the table's most important lesson and the reason this project's
detector reads whole scans.

**Three caveats that the rows cannot carry themselves.**

- **DR-SPAAM's row is its windowed form, and overstates what the published detector costs.** It encodes ten scans per frame
  here; the released streaming detector encodes **one** and attends to a carried template, so a streaming row would be a
  fraction of this. The same distinction runs the other way on accuracy — streaming is worth **+1.24 pp** to it (item 6 /
  A47) — so **our tables understate its accuracy and overstate its cost at the same time**. `DrSpaamDetector.stream()` exists;
  timing it was not in item 29's scope and is the obvious follow-up.
- **`AlgorithmicDetector`'s row includes its own console output.** The C++ detector prints two lines per call ("N legs have
  been detected", "N persons have been detected") from inside the timed window, and `verbose=False` does not suppress them —
  3,933 lines over 2,000 frames. Against a redirected file that is cheap and the row reproduces exactly (0.05 ms across two
  independent 2,000-frame runs); against a live console it would be far worse, which is a real fact about deploying it.
- **Li2Former's row is inference only.** It is the one detector never trained here, so it has a speed row and no accuracy row
  of ours (`TODO.md` item 31). Its **1,675 ms** is also 2.8x the 590.6 ms the superseded table carried, most plausibly the
  thread count — that harness ran at torch's default 8 — which is one more reason those rows are not comparable with these.

**The paper GPU figures below are not a ranking column.** They are kept to show the hardware gap, and no row here should be
read against them.

<details>
<summary><strong>The superseded rows, kept: they are the only numbers for the DROW and JRDB beam counts, and their provenance
is now known</strong></summary>

**Why they are superseded rather than reconciled.** Traced 2026-10-11 to their two source scripts, they differ from the
protocol above on **five axes at once**, so no per-detector difference can be attributed to any one of them:

| axis | these rows | item 29 |
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

¹¹ **The frozen architecture, three seeds, 2026-10-19** (`a64_no_coast_s0`, `a65_frozen_s1`, `a65_frozen_s2`; FROG `official` test, every 5th annotated frame): **86.01%** (86.08 / 86.08 / 85.88), 0.958 false positives per frame, recall 84.03%, over candidates of 82.19% -- the memory worth **+3.82 pp**. This is the model `memory/detector-architectures.md` freezes: the C1 stage-2 recipe over a fixed-beam U-Net, E13's memory, and **no coasting head** (item 24). The 85.9% of footnote 9 is the same model *with* that head -- 0.14 pp apart, inside the 1.32 pp three-sigma bar. The seed spread fell from 0.86 to 0.21 pp, which is suggestive and **not** a claim three seeds can support.

⁹ `TODO.md` A60 / A59 phase 3a, 2026-10-08: E13's stage 3 (no attention) on the no-temporal stage 2 trained with the adopted recipe -- `nms-mean` decoding, mirror augmentation, nearest-person labels, dropout 0.1, weight averaging (stage 2 alone 82.20%) -- three seeds, every 5th annotated test frame: **86.35 / 85.77 / 85.49%, mean 85.87**, AP@0.3 84.68%. Against E13 on the old recipe, 83.86%, +2.01. Stage 2 alone is the no-temporal network averaged over its val-chosen epochs, 82.28 / 82.08 / 82.22% (mean 82.20). Streamed one scan at a time on this machine's CPU at 4 threads (`tracking_baselines.py latency`, 2,000 frames, 2026-10-08): stage 2 8.75 ms, decoding 2.77 ms (`nms-mean`'s per-voter loop, ~1 ms more than the vote grid), memory 9.22 ms, **20.79 ms median, p99 24.08** -- inside FROG's 25 ms period with under a millisecond to spare at p99. The A56 tracking comparison above is on the old recipe's candidates and has not been repeated.

⁸ `TODO.md` A56 E22, 2026-10-03: `tracking_baselines.py latency`, one raw scan at a time through stage 2, DROW vote decoding and the E13 memory, 2,000 frames of FROG test, median (p99). CPU on this machine with 4 threads: **8.60 + 1.76 + 9.12 = 19.53 ms (22.38)**, 8 threads 19.47 ms (22.77); RX 9060 XT under ROCm: **5.76 + 2.27 + 4.82 = 12.92 ms (15.11)**. The CPU figure is the one comparable with this column, and it fits FROG's 25 ms scan period at the 99th percentile. The memory's share here includes moving each frame's tensors from numpy, which footnote 7 leaves out.

</details>

**Exact parameter counts**, counted from the ONNX initializers now that `onnx` is installed, replacing the file-size estimates this table used to carry (~65K / ~180K): **LFE-Peaks 53,258**, **LFE-PPN 170,903**.
For scale, `SpaceTimeCNN` is 476,262 — **2.8x LFE-PPN, for +0.5pp on the benchmark and no use of temporal order at all.**

## Where the other tables went

- **The `(T, dtime)` informational-capacity grid** now lives with its own methodology, in [`informational-capacity-proxy.md`](informational-capacity-proxy.md).
- **Per-frame ego-motion and person-displacement measurements** are dataset properties, not model results — [`data-model.md`](data-model.md).
- **The `dtime`, NMS, operating-threshold and tracker sweeps** were all measured on the contaminated split and are retracted.
  Their numbers and the reasoning that produced them stay in [`../CHANGELOG.md`](../CHANGELOG.md); they are not settings to restore, they are runs to redo.
