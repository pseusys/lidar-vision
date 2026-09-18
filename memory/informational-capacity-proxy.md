# Informational-capacity proxy — measuring temporal signal without training

*keywords:* informational capacity, mutual information, AUC, Mann-Whitney, filter method, zero-cost proxy, dtime, time_frame, informational_capacity.py

A no-training method for asking whether a `(time_frame, dtime)` configuration carries decodable person-vs-background signal at all, before spending epochs finding out the slow way.

## Why this exists

A cheap `SpaceTimeCNN` screening run (`--subsample 0.2 --epochs 8`) ranked `dtime=15` below `dtime=1` — but the per-epoch curves showed `dtime=15` still improving faster early on before overfitting the small screening subsample within budget, while `dtime=1` never converged in the same budget at all (`CHANGELOG.md`, `TODO.md` A1).
Training-based screening confounds "this configuration carries less information" with "this run hadn't converged yet," and widening a sweep by adding more `(T, dtime)` points multiplies that confound by however many points are added.
This method sidesteps training entirely: it asks the input-side question directly, using the exact same `aligned_raw_scan()` window a real detector would train on.

## The method

For each real annotated FROG frame, build the `T`-frame `aligned_raw_scan()` window at a given `dtime`, and compute one scalar feature per beam: the per-beam standard deviation across the `T` aligned frames.
At `T=2` this is a monotone transform of the simple two-frame `|diff|` (so results are consistent with an earlier `dtime`-only pass at fixed `T=2`); at larger `T` it generalizes to "how much does this beam's aligned value move across the whole window."
Each beam is labelled `person` if it falls within a few beams of a real annotated person's angular position in the window's current (last) frame, else `background`.

Two measures of how informative that one feature is for the label, chosen to be complementary:

**1. AUC of the feature alone as a person-vs-background classifier.**
Equivalent to the probability a randomly drawn person-beam's feature value exceeds a randomly drawn background-beam's — the Mann-Whitney U / Wilcoxon rank-sum statistic (Hanley & McNeil, "The Meaning and Use of the Area Under a Receiver Operating Characteristic (ROC) Curve", *Radiology* 143(1), 1982).
This is the same quantity as this project's own headline `wp-AUC` metric (both are trapezoidal ROC-AUC), just computed on one raw feature instead of a trained model's output — the same idea used throughout applied statistics and bioinformatics for cheap univariate marker screening before fitting anything (Pepe et al., "Combining diagnostic test results to increase accuracy", *Biostatistics* 4(3), 2003).
It is a textbook **filter method** in the feature-selection sense: score each feature's own discriminative power before any model is trained, as opposed to a **wrapper method** that needs a trained model to score it at all (Guyon & Elisseeff, "An Introduction to Variable and Feature Selection", *JMLR* 3, 2003).

**2. Mutual information `I(feature; label)`, in bits.**
An actual Shannon information quantity (Shannon, "A Mathematical Theory of Communication", *Bell System Technical Journal* 27, 1948), not an analogy.
Estimated via `sklearn.feature_selection.mutual_info_classif`, a k-nearest-neighbour estimator built for exactly this continuous-feature/discrete-label case (Ross, "Mutual Information between Discrete and Continuous Data Sets", *PLoS ONE* 9(2), 2014).

## What this does and does not prove

**Absence of a trend here is strong evidence against a configuration.**
If the raw temporal channel carries no rank-order signal at all, no amount of training can invent it from nothing.

**Presence of a trend is necessary but not sufficient evidence for it.**
Both measures are univariate: they see one beam's own feature in isolation and cannot capture what a real network can — nonlinear combinations across many beams, spatial context, learned thresholds.
A real detector could still outperform (or, in principle, underperform, if the learning problem itself is harder at that configuration) what this proxy predicts.
Treat the *direction and shape* of the trend across the `(T, dtime)` grid as the signal, not the absolute AUC/MI values as a prediction of final `wp-AUC`.

## Two things that keep this honest

**It reads the train split only.** Choosing a `(T, dtime)` by peeking at val or test would leak the very thing the sweep it guides then measures.

**It must use FROG's own beam spacing.**
`aligned_raw_scan()` defaults `beam_spacing` to `laser_increment = radians(0.5)` — DROW's resolution.
Calling it without the argument on FROG's 0.25 deg/beam data applied **half** the rotation correction training applies through `cfg.laser_inc`, a train/proxy inconsistency on the exact axis the proxy is used to choose (`TODO.md` A19, fixed 2026-09-10).

## The grid

`utils/informational_capacity.py`, run standalone from `utils/` (no training, no GPU, a few minutes).
Current results: `results/informational_capacity_grid3.log` — 83 points, re-run 2026-09-10 on the fixed pipeline.
`grid2`/`grid` are superseded (wrong beam spacing, pre-split-fix `train` split).

Per-`T` peak:

| `T` | peak `dtime` | span | peak AUC | MI (bits) |
| --- | --- | --- | --- | --- |
| 2 | 26 | 0.99s | 0.6545 | 0.0120 |
| 3 | 13 | 0.99s | 0.6719 | 0.0189 |
| **5** | **9** | **1.37s** | **0.6804** | 0.0228 |
| **10** | **4** | **1.37s** | **0.6816** | 0.0197 |
| 15 | 3 | 1.60s | 0.6800 | 0.0262 |
| 20 | 2 | 1.45s | 0.6796 | 0.0256 |
| 25 | 2 | 1.83s | 0.6769 | 0.0206 |

**A genuine interior maximum at every `T`, governed by total window span** rather than by `T` or `dtime` individually.
Every row rises from `dtime=1`, peaks, then declines monotonically to its widest tested `dtime`.
Peak spans cluster at **0.99-1.83s** across a more-than-10x range of `T`, matching Winter (1990)'s ~400-600ms single-step timing already cited in this project's `T=5` design rationale (`detector-architectures.md`), with a full gait cycle (~0.8-1.2s) inside the band.

**Peak AUC saturates once `T>=5`** — 0.6545, 0.6719, 0.6804, 0.6816, 0.6800, 0.6796, 0.6769 for `T`=2,3,5,10,15,20,25, flat within noise from `T=5` on.
Stacking more than ~5 frames buys this proxy nothing once `dtime` hits the right span.

Top five points overall: `(T=10, dtime=4)` 0.6816, `(T=5, dtime=9)` 0.6804, `(T=10, dtime=5)` 0.6804, `(T=5, dtime=7)` 0.6802, `(T=5, dtime=8)` 0.6801.
Both findings and both winning configurations survived the re-run unchanged, which is itself mild evidence the proxy measures the input rather than the harness.

**What it changes about the training sweep** (`TODO.md` A1): concentrate full runs on `T=5, dtime~=9-10`, with `T=10, dtime~=4-5` as the confirmation point.
`T=15/20/25` stay low priority — a real gain there would be an architecture-capacity finding, not an information-budget one.

Model accuracy numbers, published and measured, are in [`performance-log.md`](performance-log.md); this doc owns the proxy and nothing else.
