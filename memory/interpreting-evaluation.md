# Interpreting evaluation numbers

*keywords:* wp-AUC, AP, zero-shot transfer, DROW-native, fine-tuned, precision, recall, false positive, eval-stride, headline metric

By how much should a given accuracy or speed number be discounted, and why.
The raw numbers and sweep tables themselves are in [`performance-log.md`](performance-log.md); this file is about how to read them correctly.

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

## Published-weights vs. this project's own training

Every detector falls into exactly one of: **replicated with official weights** (`drow`, `drspaam`, `lfe_peaks`, `lfe_ppn` — genuine published-model inference, no training performed here), **reimplemented with no official weights** (`li2former` — trained from scratch from the paper's description alone, since training never completed, no locally-measured number exists at all), or **novel** (`spacetime_cnn`, `fullscan_tcn`, `temporal_unet` — no prior paper describes them, so no official weights could exist).
A "replicated" score is a reproduction of a published result; a "novel" score is this project's own claim, with nothing to check it against except internal consistency (train ~ test, ablation sanity).
An official-weights evaluation of DR-SPAAM (T=1, T=5) run through this project's own pipeline is in progress as of this writing, specifically to get a number independently verified against this project's own zero-shot-transfer figure rather than only against the paper's reported number (`TODO.md`).

## False positives, not false negatives, are the dominant error on FROG — for every model tested

At each model's own best-F1 operating point, the FN/FP ratio never exceeds 0.8x, for `spacetime_cnn`, `fullscan_tcn`, `temporal_unet`, `lfe_peaks`, and `lfe_ppn` alike.
A proposal to "improve recall" is targeting the wrong side of the error for this dataset; a proposal targeting false positives (threshold selection, NMS tuning, a confirmed-track requirement) is aimed at the actual dominant error.
This also means the best-F1 point on a model's PR curve is not necessarily the right *deployment* threshold: for three of the five models above, moving to a ~75-80% precision operating point trades favorably (saves more false positives than it costs in false negatives); past ~80% the trade reverses.

## `LFEPPNDetector`'s ~250-300 ms/frame cost is an implementation artifact, not an architectural one

Its accuracy is close to its own paper's figure and comparable to `LFEPeaksDetector`'s, but its decoder is a nested pure-Python loop over every (sector x anchor) score, while `LFEPeaksDetector` decodes with one vectorized call.
Quoting LFE-PPN's speed as "the cost of its anchor-grid design" is wrong; quoting it as "this repo's current decoder, fixable with a NumPy rewrite" is right (`TODO.md`).

## `--eval-stride` changes which frames were measured, not just how many

FROG's every-frame-annotated-at-40Hz protocol means adjacent frames are highly correlated near-duplicates, so a `--eval-stride 40` number is usually close to the `stride=1` number on FROG specifically — but it is still a different sample, and DROW's already-sparse ~5%-of-scans annotation means this correlation argument does not carry over there.
Never compare a strided FROG number against an unstrided DROW number and call the gap architectural.
