# Interpreting evaluation numbers

*keywords:* wp-AUC, AP, zero-shot transfer, DROW-native, fine-tuned, precision, recall, false positive, eval-stride, headline metric

By how much should a given accuracy or speed number be discounted, and why.
The standing comparison against published SOTA is in [`performance-log.md`](performance-log.md), and individual runs and sweeps in [`../CHANGELOG.md`](../CHANGELOG.md); this file is about how to read either correctly.

## Before anything else: every pre-2026-09-10 FROG number is retracted

Four measurement bugs, all fixed (`TODO.md` A11-A19, `CHANGELOG.md`), none of them re-measured yet.
Read `performance-log.md`'s status table before quoting any row — it says precisely which rows the fixes invalidated and which they did not.
The two that generalize past this incident, and are worth carrying as habits:

**A number is only comparable to another number measured the same way.**
The `dtime=10` "regression" decomposed exactly into four protocol differences — evaluation `dtime`, which epoch's weights, which split, which batch size — with nothing left over for the variable actually under study.
Before comparing two runs, check all four.

**An evaluation implementation detail can move the number more than the model does.**
See the batch-size section below.

## `wp-AUC` is this project's headline metric, and is the same thing as the papers' "AP"

Person-class AUC at a 0.5 m matching radius, computed by `_prec_rec_2d`/`_safe_auc` — a direct port of the original DROW-v2 evaluation code, and verified by direct source comparison to compute the identical trapezoidal-AUC-over-a-Hungarian-matched-PR-curve formula the official DR-SPAAM repo calls "AP".
A gap between this project's number and a paper's own number is a real accuracy difference, not a metric-definition artifact.
Never substitute the agnostic (any-class) AUC column for it — see `AGENTS.md` rule 2b for the incident that makes this an explicit rule rather than a preference.

## A cross-dataset number is not one measurement, it's three, and they disagree

Every architecture in `detector-architectures.md` has been evaluated on DROW in up to three different regimes, and they are not interchangeable:

- **Zero-shot transfer** — weights trained on FROG (or, for `drow`/`drspaam`, published weights trained on DROW), evaluated on DROW without any DROW-specific retraining.
- **DROW-native** — trained from scratch on DROW only.
- **Fine-tuned** — DROW-native training initialized from the FROG checkpoint's weights at a lower learning rate (`--init-weights`, not `--resume` — the latter also restores the wrong optimiser state).

The counterintuitive, load-bearing finding: **zero-shot transfer wins outright over both other regimes, for every one of the three proposed architectures.**
This is a genuine result, not a bug — DROW's much smaller effective training set (next section) means "more DROW-specific training" does not currently mean "better on DROW."
Quoting a DROW number without saying which of the three regimes produced it is close to meaningless; the spread between them is 2x or more for every architecture tested.

## DROW's numbers are structurally handicapped by annotation coverage, not by being a harder dataset

DROW's train split has roughly **17.8x fewer total person-annotation instances** than FROG's: 6.1x fewer annotated frames (17,665 vs. 108,356) and 2.9x fewer people per annotated frame (1.35 vs. 3.93).
A DROW-native `spacetime_cnn` run shows textbook overfitting (val loss rising from epoch 2 onward while train loss keeps falling) on the same regularization that works fine on FROG — this is a data-scarcity artifact relative to model capacity, not evidence that the architecture transfers badly to DROW's sensor geometry.
The architectures were sized for FROG's data volume; nobody has yet tried shrinking one for DROW's scale specifically (`TODO.md`).

## LFE numbers on DROW/JRDB are an approximate cross-dataset transfer, not a faithful replication

`LFEPeaksDetector`/`LFEPPNDetector` were trained on FROG's 720-beam, 0.25 deg/beam scans.
Running them on DROW (450 beams, 0.5 deg/beam) or JRDB (541 beams) requires zero-padding shorter scans to 720 — this project's own extrapolation, never described in the LFE paper, and one that does not correct for the different angular resolution.
Treat any LFE DROW/JRDB number as approximate, degraded-by-construction, not comparable one-for-one to a same-dataset number.

## A tracker must be fed every frame, not the frames you happen to score

`SimpleTracker` assumes `dt = 1.0` — one `step()` per sensor frame. Feeding it only the *scored* frames breaks it in two ways that both look like "tracking does not help":

- under `--eval-stride N` its time step silently becomes N frames long, so its velocity estimates and its `match_radius` gate are N times too tight;
- on the `transferred` split the scored frames are the person-free ones, **interleaved in the raw recording with populated frames the split drops** — so consecutive scored frames can be minutes apart, and joining them into a track is meaningless.

The LFE evaluation paths therefore walk the raw sequence at full rate whenever a tracker is active, and score only the subset (`_lfe_frames`). Without a tracker the walk *is* the scored set, so no number measured before this existed moved.

**Pass real ego-motion.** FROG's odometry is not zero — 1.88 cm and up to 0.4° per frame on the test recording. The belief that it was zero dates from `_load_or_fake_odom()` substituting zeros on failure (`TODO.md` A14) and survived in two code comments and a docstring; 0.4°/frame over a track's ~6-frame lifetime is 2.4°, which at 8 m is 33 cm — most of the tracker's 0.5 m gate.

## A FROG number means nothing without its association distance

FROG's Table 4 reports **nine** figures per model: mAP / mPeakF1 / mEER averaged over `d = [0.3 : 0.05 : 0.5]` m, then AP / PeakF1 / EER at `d = 0.5 m`, then the same at `d = 0.3 m`.
**The leading column is the averaged mAP, not AP at any single distance**, and every FROG figure this project quoted before 2026-09-10 was taken from it while every number we produce is AP at `--eval-r 0.5`.

The offset is not a constant you can subtract: +0.3pp for DR-SPAAM, **+2.7pp for LFE-PPN**, because how fast a detector's AP falls as the gate tightens is itself a property of the detector.
`performance-log.md` now carries the `d = 0.5 m` column. **Always say which `d`.**

## Calibrate anything positional at `--eval-r 0.3`, not 0.5

A 0.5 m association gate is wider than most localisation errors worth finding, so it will happily accept a decoder that is systematically off by 10-20 cm.
This is not hypothetical: the LFE-PPN anchor-decode calibration reported its angular convention as "weakly identified" because the candidates spanned 66.6-68.5% at `d = 0.5`. At `d = 0.3` the same candidates span **28.7-56.4%**, and the chosen one undershoots the published figure by 9.4pp.

And do **not** try to recover a decode convention by fitting localisation residuals against the raw offset channel. It was tried, it looked conclusive, and it is confounded: ground-truth association uses a *metric* radius, so angular residuals are capped at ~`r/d` by construction and any convention produces a `scale x depth = const` signature.

## We cite published accuracy numbers; we do not re-derive them

Owner's call, 2026-09-10, now the standing policy (`performance-log.md` holds the notation):
**a table cell takes the original authors' published number wherever they published one.**
This project evaluates a third-party model only to fill a genuine gap — a model/dataset pair nobody published — or to measure `ms/frame`, which is comparable only when every row came off one machine.

Re-deriving a published accuracy figure through this pipeline costs a real run and adds a reproduction risk, for a number that was already authoritative.
This session's audit is the case in point: four measurement bugs silently reshaped every locally-derived FROG figure, while the published ones were never in doubt.

**Standing corollary — if our reproduction of a published number overshoots it, the harness is wrong.**
An undershoot has innocent explanations: a different frame sample, a wrapper detail, a merge radius we had to choose ourselves.
An overshoot on identical published weights does not.
The live example: LFE-Peaks measured **74.6%** here against its authors' own **64.9%** on FROG — a +9.7pp overshoot that was the visible symptom of the train/test contamination for weeks before anyone read it that way.
Check this the moment a reproduction lands, not later.

## Published-weights vs. this project's own training

Every detector falls into exactly one of: **replicated with official weights** (`drow`, `drspaam`, `lfe_peaks`, `lfe_ppn` — genuine published-model inference, no training performed here), **reimplemented with no official weights** (`li2former` — trained from scratch from the paper's description alone, since training never completed, no locally-measured number exists at all), or **novel** (`spacetime_cnn`, `fullscan_tcn`, `temporal_unet` — no prior paper describes them, so no official weights could exist).
A "replicated" score is a reproduction of a published result; a "novel" score is this project's own claim, with nothing to check it against except internal consistency (train ~ test, ablation sanity).
An official-weights evaluation of DR-SPAAM (T=1, T=5) through this project's own pipeline was attempted and abandoned (`TODO.md` A2) — the paper's own numbers are cited instead, deliberately, since the weights-compatibility check (zero missing/unexpected keys) already passed independently.

## Epoch counts across papers are not comparable without converting to minibatches

FROG's own authors trained DROW3/DR-SPAAM for exactly 5 epochs (batch 8) on FROG's full 108,356-frame train split — about 67,700 minibatches — a number chosen deliberately "to match the magnitude of minibatches processed when training on the original DROW dataset," not because 5 epochs is enough in general.
LFE-Peaks/LFE-PPN targeted 100/150 epochs (patience 20) at batch 32/4 instead.
Comparing raw epoch counts across these (or against this project's own runs) is meaningless without converting to `epochs x frames_seen / batch_size`; a screening-scale run here (8 epochs, 20% subsample, batch 4, ~43,300 minibatches) lands in the same order of magnitude as DROW3/DR-SPAAM's own training, while this project's full runs (30 epochs, full data, batch 4, up to ~812,700 minibatches) run roughly 12x longer than that.
A model still improving (val wp-AUC rising, patience not triggered) at a cheap screening run's cutoff is undersold by that run's own number — a real risk specific to short/subsampled runs, not evidence the configuration is actually worse (`CHANGELOG.md`'s `dtime` screening table has a live example).

## What temporal information is worth on FROG, according to the literature

**+1.9pp.** DR-SPAAM's own published FROG numbers at d = 0.5 m are `T=1` **73.7%** and `T=5` **75.6%** (73.3% / 75.3% in the paper's mAP column, `performance-log.md`) — the best temporal model in the literature gains about two points from five frames on this benchmark.
LFE-Peaks reaches 65.6% from a **single frame** with ~65K parameters at 1.76 ms; neither LFE model carries any memory between scans.

**Those five frames span about 0.15 s.** The FROG paper gives only "a temporal window size of 5 scans" and names no stride. It trained DR-SPAAM with the official repository (`VisualComputingInstitute/2D_lidar_person_detection`), whose DR-SPAAM config sets `scan_stride: 1` and whose loader takes scans `scan_idx - i * scan_stride`, i.e. consecutive; FROG's own export (`robotics-upo/2DLaserPeopleBenchmark`, `convert_frog_bags.py`) keeps every scan. Five consecutive scans at FROG's measured 26.2 Hz (`dataset-properties.md`) are 4/26.2 = 0.15 s apart end to end -- inferred from the code, since FROG's own DR-SPAAM config is not published. Even run as a stream, DR-SPAAM's template update (α = 0.5) halves an older scan's weight per scan. So the published temporal gain comes from the span of stage 2's fine temporal convolution in `docs/PROPOSAL.md` (~0.08 s), well inside its calibration horizon (coarse convolution ~1.3 s); the short and long horizons have no published counterpart on this sensor.

Keep that ceiling in view before reading any temporal result here as a large effect.
It also frames the risk this project actually runs: every one of its three novel architectures exists *for* the temporal channel, so if the trained model turns out to be a static shape detector, LFE already does that job far more efficiently and the architectures have no rationale (`TODO.md` A21).

The corollary is a reframing worth holding onto: on a benchmark where every frame contains 3-4 people, shape alone gets most of the way, so temporal information has little room to show up as raw AP.
Where it *should* show up is in rejecting **static** false positives — a chair leg and a standing person are near-identical in a 2D knee-height slice, and motion is the only cue separating them.
That is what `TODO.md` A20's person-free false-positive rate measures, and nobody has published it.

## The informational-capacity proxy's peak is a prior for where to train, not a prediction of the trained result

`informational-capacity-proxy.md`'s AUC/MI grid found a clean peak around a ~1.2-1.8s total window span, stable across `T` from 2 to 25 — a real, useful finding about the *raw input*, but it is not the same claim as "this is where the trained `wp-AUC` will peak too."
Expect a real, positive correlation, not an exact match, for reasons that cut in both directions:

**Why a correlation is expected at all.** The proxy is computed on the exact same `aligned_raw_scan()` window a real detector trains on — if the raw per-beam signal has low separability at some `(T, dtime)`, the network has strictly less to work with there, and no amount of training invents information that was never in the input.
This is the same logic behind the "zero-cost NAS proxy" literature (Mellor, Turner, Storkey & Crowley, "Neural Architecture Search without Training", *ICML* 2021; Abdelfattah, Mehrotra, Dudziak & Lane, "Zero-Cost Proxies for Lightweight NAS", *ICLR* 2021) — those proxies are computed from network gradients/Jacobians at initialization rather than raw input statistics, a different mechanism from this one, but published for the same reason: to narrow a search space before paying for full training, and reported there with a *moderate*, not perfect, rank correlation to final trained performance (typically informative enough to discard clearly-bad candidates, not precise enough to pick the exact winner without confirming it).

**Why it won't be exact.** This proxy is univariate — one beam's own feature, in isolation.
A real detector combines many beams nonlinearly (a person spans several beams; correlated motion across adjacent beams is a much stronger cue than any one beam's variance) and applies a fixed architecture whose own temporal kernels (`SpaceTimeCNNDetector`'s 3/5/7-tap time-axis convolutions, sized around `T=5`, `detector-architectures.md`) may not extract a given `(T, dtime)`'s available information equally well regardless of how much is nominally there.
Training-dynamics effects already found in this project (different `dtime` values converging at different *rates*, not just different ceilings — `CHANGELOG.md`'s dtime screening table) are a separate axis entirely, orthogonal to how much raw information exists.

**The honest position, and the concrete check that resolves it**: treat the proxy's peak as a *prior* — worth concentrating training compute near it, worth being suspicious of a config it says is well past its peak — but not as a substitute for the phase-2 training runs themselves.
Once several real `(T, dtime)` points have both a proxy value and a trained `wp-AUC`, compute the actual (Spearman) rank correlation between them directly rather than assume it — this project's own convention throughout (`dos-and-donts.md`'s evidence section) — and record whatever that correlation turns out to be here, since it determines how much weight this proxy should carry in future sweep decisions.

## False positives, not false negatives, are the dominant error on FROG — for every model tested

At each model's own best-F1 operating point, the FN/FP ratio never exceeds 0.8x, for `spacetime_cnn`, `fullscan_tcn`, `temporal_unet`, `lfe_peaks`, and `lfe_ppn` alike.
A proposal to "improve recall" is targeting the wrong side of the error for this dataset; a proposal targeting false positives (threshold selection, NMS tuning, a confirmed-track requirement) is aimed at the actual dominant error.
This also means the best-F1 point on a model's PR curve is not necessarily the right *deployment* threshold: for three of the five models above, moving to a ~75-80% precision operating point trades favorably (saves more false positives than it costs in false negatives); past ~80% the trade reverses.

## wp-AUC is *undefined* on a person-free split, not merely low

With no true positives, precision is 0 at every threshold and recall has no denominator.
`evaluate.py` therefore switches metric with the data rather than with a flag: a split carrying no annotated people is scored by `false_positive_rate()` — FP/frame, FP/second, and the share of frames carrying at least one, at a sweep of operating points — because there every detection *is* a false positive by construction.
The sweep is deliberate: a single operating point would be a choice presented as a measurement.

`duration_s` is `n_frames * median_frame_period`, never the span from the first timestamp to the last: `transferred`'s person-free frames are interleaved with populated ones the split drops, so a span would bill the detector for time it was never asked about and understate its rate.

## `LFEPPNDetector` was never architecturally slow — it was decoding a bug

Recorded here because the opposite claim stood in this file until 2026-09-10, and was quoted.
Its ~250-300 ms/frame was blamed on a nested pure-Python decode loop. That was wrong.
The decoder applied a **second sigmoid** to an objectness channel `lfe_ppn.onnx` had already sigmoided, squashing every score into [0.5, 0.731]; `score_thresh` went inert, and all 3,600 anchors entered an O(n²) greedy merge every frame, yielding 182 "detections" per frame on scenes holding ~3 people.
Fixed at the source, it runs at **1.56 ms/frame** — within 0.1 ms of the paper's own *GPU* figure, on CPU — with the same Python loop untouched.

**The general lesson, which is why this entry stays**: a performance anomaly of two orders of magnitude was explained by the first plausible implementation story and the story was written down as fact. The measurement that would have refuted it — how many detections per frame? — cost nothing and was never taken.

## Run-to-run spread is ~0.9 points of test AP — two single runs must differ by ~2.5 points to mean anything

Measured 2026-09-17 by training two step-2 arms at three seeds each, all else equal:

| arm | seed 0 | seed 1 | seed 2 | mean | sd |
| --- | --- | --- | --- | --- | --- |
| no-fine (1 fine tap) | 78.03% | 77.89% | 76.60% | 77.51% | 0.79 |
| 5 fine taps | 78.56% | 77.32% | 76.63% | 77.50% | 0.98 |

With a per-run sd of ~0.9 points, the difference between two single runs has an sd of ~1.3, so **anything under ~2.5 points between two single runs is noise**. The 0.30-point spread used to screen every architecture arm on 2026-09-16 came from one pair of runs that also differed in code (the plateau scheduler), so it was never a variance estimate. It understated the spread about threefold.

That retracts every single-seed ranking of 2026-09-16 smaller than ~2.5 points: dropout over the control (+0.70), each temporal ablation, 5 taps over 3 (+1.94), 7 taps under 5 (−1.42), and the mechanisms argued from them. **Still standing:** effects far outside the spread (the median denoiser, −9.16), and findings about one fixed model, which carry no seed variance (stage-3 threshold calibration, the memory diagnosis, the `--clip-frames` bug).

The rule going forward: screen at several seeds from the start, or let single-run screening rank only effects above ~2.5 points.

**Seeds do not pair runs; compare means, not per-seed differences.** Measured 2026-09-17. The default architecture was retrained at seeds 0-2 on current code and scored 78.19 / 76.64 / 75.13. The same seeds' earlier runs, rescored with the same decoding, gave 78.40 / 78.28 / 77.09. The code differed only in how targets anchor beyond 10 m. A same-seed gap of up to 1.96 points means a seed does not reproduce a run here: GPU kernels are nondeterministic, and seeding fixes data order but not the arithmetic. So a paired-by-seed t-test is not valid on this machine. Two paired "results" are retracted: static vs the default architecture ("t = 3.4") and limit-beam exclusion ("−0.93, lower on every seed"). The latter reversed to +0.34 against fresh controls. Five three-seed arms now pool to a per-run sd of **~0.95 points** (0.46-1.53 per arm), so the ~2.5-point single-run rule stands. At three seeds a mean difference needs roughly 1.5-2 points to count.

## FROG `official`'s validation split does not track its test split — do not read a val difference as a test result

Three independent measurements on 2026-09-16, all in the same direction, two of them pointing the *wrong* way:

| change | val AP | test AP |
| --- | --- | --- |
| dropout 0.1 against dropout 0.0 | 73.88% -> 73.93% (+0.05) | 76.62% -> 77.32% (**+0.70**) |
| removing the fine temporal convolution | 73.88% -> 73.76% (-0.12) on the sample, 73.79% -> 73.88% (+0.09) on every val frame | 76.62% -> **78.03%** (**+1.41**) |
| the dropout run's own checkpoint choice | picked epoch 0.75 | epoch 1.00 scores 0.84 higher |

Val also sits about 3 points *below* test for identical weights (73.88% against 76.62%), and swings more than 1.5 points between adjacent evaluations of one run, so a val difference under ~1.5 points carries no information at all.

**Scoring every val frame rather than the 2,000-frame sample narrows the charge** (2026-09-16, `duration_curve.py --split val`, five arms at their selected weights): control 73.79%, no-fine 73.88%, no-coarse 72.95%, static 72.78%, dropout 73.89%, against test APs of 76.62 / 78.03 / 77.01 / 77.22 / 77.32. The sample was accurate to about 0.12 points, so its ordering of control against no-fine flipped only because those two are **tied on val either way** — val cannot resolve the 1.4-point difference test sees between them. Full val does agree with test that no-fine and dropout are the best two. But it disagrees outright elsewhere: **static is last on val and third of five on test, control is third on val and last on test**. So the rule is not "val is reversed" — it is that val cannot resolve differences of this size and is actively misleading on some arms.

Two consequences. **Screen and rank on test AP of the val-selected checkpoint, never on val AP itself** — and say which checkpoint a number came from. **Never select on test**, which is why the mis-selection above is recorded rather than corrected: 77.32% stays the reported dropout figure even though 78.16% was available at another epoch.

The cause is most likely the split itself: `official`'s val is the populated-only frames of three recordings, sampled 2,000 frames at a time, while its test is one recording scored on every annotated frame. `balanced`'s val is 121,329 frames including 19.8% empty and is the obvious candidate for a better selector — but only for a model trained on `balanced`, since `balanced`'s val contains 11-36, which is half of `official`'s **train** (`memory/dataset-properties.md`, `TODO.md` A50).

## `--eval-stride` changes which frames were measured, not just how many

FROG's every-frame-annotated-at-40Hz protocol means adjacent frames are highly correlated near-duplicates, so a `--eval-stride 40` number is usually close to the `stride=1` number on FROG specifically — but it is still a different sample, and DROW's already-sparse ~5%-of-scans annotation means this correlation argument does not carry over there.
Never compare a strided FROG number against an unstrided DROW number and call the gap architectural.

## `--eval-batch-size` used to change the answer, and the habit it leaves behind

Until 2026-09-10 the full-scan models folded the minibatch into the **beam** axis, so a batch of B scans became one `B*720`-beam "scan" and normalization statistics, the SE gate and the beam-axis convolutions all leaked between frames sharing a batch.
Evaluation batches are consecutive frames in time, so a larger batch leaked more temporal context and scored higher — measured on the real val split at `dtime`=10, beam-level person AUC ran **0.9644 (batch 1), 0.9696 (4), 0.9765 (16), 0.9810 (64)**.
`evaluate.py` defaults to 16, `train.py` used 4, the robot runs `forward_one()` at batch 1: the deployed model was the 0.9644 one and every published figure was inflated relative to it.

Fixed (`TODO.md` A18), and a regression test now asserts a frame's output is identical alone or inside a batch of 16.
The habit worth keeping: **when a metric moves with a knob that should not affect the model, that is a bug in the harness, not a property of the model.**
Batch size, evaluation `dtime`, split membership and checkpoint selection are all knobs of exactly that kind.

## `dtime` at evaluation must match `dtime` at training

Scoring a `dtime=10` model at `dtime=1` cost **2.5pp wp-AUC** (76.3% vs 78.8%) and was silent — the parameter simply defaulted to 1 and nothing passed it.
`dtime` now lives in the checkpoint and `evaluate.py --eval-dtime` defaults to it, so a hand-passed flag can no longer disagree with how the weights were trained.
Pass the flag only to deliberately evaluate off-distribution, and say so when you report the number.
Checkpoints saved before 2026-09-10 carry no `dtime` and fall back to 1 — for those, pass it by hand and check the training log for what it should be.
