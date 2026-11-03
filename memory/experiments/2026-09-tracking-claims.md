# Tracking-literature experiments: the claims they decide, predicted before running

*keywords:* A56, pre-registration, paper claims, claim ledger, tracking ladder, ByteTrack, CBMOT, rule scorer, E14, E15, E16, E20, E21, E22, GOSPA, block bootstrap, stratified gain, occlusion, first-detection latency, val-tuned baseline

**Status: done 2026-10-03.** The predictions below were written 2026-09-30 and were not edited after the runs started; the Outcome section at the end scores each one.
**Updated before any run, 2026-10-02:** E10 now has two seeds (81.84 / 82.04, mean 81.94; its AP loss of 1.40 sits inside the 1.45 bar, while recall at matched FP and coasting precision still clear), and E19 ran as S2 (a free-space input for the coasting head: better coasting ranking, flat end to end), so it is dropped from this batch.
The literature behind it is [`../../docs/RESEARCH.md`](../../docs/RESEARCH.md) §7.4; the work items are [`../../TODO.md`](../../TODO.md); the slot evidence it builds on is [`../slot-design-evidence.md`](../slot-design-evidence.md).
This file records what is predicted and what each outcome changes in the paper, so that a result cannot be reinterpreted after it arrives.

## Where the evidence stands

All on FROG `official` test, AP at 0.5 m, every 5th annotated frame, on the same no-temporal stage-2 candidates.

| rung | what reads the candidates | AP | seeds | memory ms/frame |
| --- | --- | --- | --- | --- |
| 0 | nothing (the candidates) | 77.66 | fixed | -- |
| 1 | SORT, fed every candidate to 0.01, reporting each with its own score | 76.7 / 75.7 / 74.9 at `min_hits` 1 / 3 / 5 | earlier chain's candidates | -- |
| 2 | **a rule that raises scores from track confidence (CBMOT)** | **not measured** | -- | -- |
| 3 | a learned scorer on the slot's bookkeeping statistics, no state (E10) | 81.94 (81.84 / 82.04) | 2 | 2.70 |
| 4 | learned per-slot state, no attention (E13, the architecture to propose) | 83.86 (sd 0.32) | 3 | 3.72 |
| 5 | learned state plus both attention mechanisms (B0) | 83.34 (sd 0.48) | 3 | 4.53 |

Rung 2 is the gap.
SORT here sees every sub-threshold candidate but can only filter; the tracking literature's way of *lifting* a weak detection from its track is CBMOT's score refinement (IROS 2021), and its strongest association is ByteTrack's two-stage pass (ECCV 2022).
Until rung 2 is measured, "a memory inside the detector beats post-hoc tracking" is tested only against a tracker that cannot rescore.

## The claim ledger

| # | claim, as the paper would make it | evidence now | still has to hold | decided by | if it fails, the paper says instead |
| --- | --- | --- | --- | --- | --- |
| C1 | AP conceals a constant phantom rate in knee-height 2D-LiDAR detection | LFE's genuine phantoms 1.387 per populated frame against 1.391 per empty one (`../../docs/PAPER.md`) | an established metric shows the same | E16 (GOSPA) | the hand-built FP/FN accounting stands alone, with GOSPA cited as its formal relative |
| C2 | Object-level memory inside the detector raises its detection AP by **+6.2 pp** over its own candidates (83.86 against 77.66, 3 seeds): the first detection gain from tracking-like memory reported on a 2D LiDAR (MAD and MoDAR are 3D) | E13, B0, E1 | the gain survives a paired block bootstrap, and no val-tuned classical pipeline on identical candidates comes within the bar | E14, E15, bootstrap | see the decision table below |
| C3 | The gain is the learned state, not tracking in general: rule scorer < learned scorer < learned state, attention adding nothing | E10 (-1.50 without state), E13 (ties without attention) | rung 2 sits below rung 3; E10's second seed agrees | E15, E14, E10 s1 | "a tracker with a rule scorer delivers X of the gain; the learned state adds the rest" |
| C4 | The memory uses about 10 s of history and pays where the detector is weakest | E1 (95% of the gain by 10 s), E3 (flat across crowd size) | the gain concentrates on occluded, far or stationary people, as MoDAR reports for 3D | E20 | "memory helps everywhere", which E3 already supports |
| C5 | The memory adds no confirmation latency, unlike `min_hits` tracking | none | first detection no later than the candidates' | E21 | the latency is stated as a cost |
| C6 | Constant per-frame cost, flat in slot count, well inside a 25 ms frame | E4 (4.3-4.5 ms, 64 to 720 slots); stage 2 + decode ~10.4 ms | one end-to-end streaming benchmark of the proposed architecture | E22 | components reported separately, as now |
| C7 | Room coordinates help only with good odometry | D1 (+1.14, 3 seeds), E2 (crossover ~0.01-0.02 m/s) | training with jittered poses closes the crossover | E11 (agreed) | the caveat is stated |
| C8 | Honest negatives: attention is unnecessary; the value rule is CBMOT's and earns nothing here; feedback into stage 2 does nothing | E6, E7, E13, D2, E12 | nothing | -- | -- |
| C9 | The slots are also a tracker, and the detector transfers zero-shot | none | identities (JRDB) or mocap (Plozza et al.) | E17 | out of scope for this paper |

## Pre-registered predictions and decisions

**C2 and C3, from the best val-tuned classical pipeline scored once on test** (the higher of E14 and E15; the bar is B0's three-sigma 1.45 pp):

| classical AP on test | reading | what the paper says |
| --- | --- | --- |
| at most 79.11 (candidates + 1.45) | rule-based rescoring barely moves AP | C2 and C3 as stated |
| 79.11 to 82.41 | rule-based rescoring recovers part of the gain | C2 holds; C3 states the split between the rule and the learned state |
| at least 82.41 (E13 − 1.45) | a tuned classical pipeline matches the memory | C2 is rewritten as "tracking-based rescoring raises 2D-LiDAR detection AP by ~6 pp", novel on this sensor; the memory's case moves to coasting precision and cost; C3 is withdrawn |

**Prediction: 79-81%, the middle row.** A rule cannot tell a remembered person from a remembered chair, which is exactly where E10 shows the state earning its keep (coasting precision 11.4% without it, 29.8% with it).

**Other predictions:**

- E14 with detection-score output (no rescoring) lands within ±0.5 pp of the candidates: a filter cannot raise AP much.
- E15 (the memory's own bookkeeping with CBMOT's rule) lands within 1 pp of E14's best: association is not the difference (S1: 97% of people already have a matched slot).
- E16: the memory's GOSPA false-target term is about half the candidates' at each method's val-best threshold, with the missed term within 10%; LFE-PPN has the largest false term.
- E20: the gain is larger for occluded than for visible people, larger beyond 6 m than within 4 m, larger for stationary than for moving people, and near zero in a trajectory's first 0.5 s.
- E21: the memory's median first detection is within one frame of the candidates'; SORT at `min_hits` 3 is at least three frames later.

## Experiments

| id | what | how | machine | decides |
| --- | --- | --- | --- | --- |
| E14 | the strongest classical pipeline on identical candidates | a `ByteTracker` for points beside `SortTracker` in `tracking.py`: SORT's Kalman point, ego-motion, ByteTrack's two-stage association (tracks born only from high-score candidates), CBMOT's track score (eq. 6, decay, deletion on score), optional OC-SORT/ImmortalTracker-style coasting at the decayed score. Output modes: detection score (pure ByteTrack), CBMOT score, CBMOT plus coasting. Grid over `tau_high`, `tau_low`, gate, decay and max age, tuned on **val**, scored on test once; SORT re-measured on this cache | CPU, ~2-3 h | C2, C3 |
| E15 | the memory's own bookkeeping with a rule instead of a learned head | `manage_slots` replayed on the GPU; each matched candidate rescored with CBMOT eq. 6 from its slot's value, unmatched ones keep their score, alive unmatched slots coast at their decayed value; `gain` and `fade_time_s` tuned on val | GPU, ~30 min | C3: the rung directly below E10 on identical association |
| E16 | GOSPA beside AP | per frame, c = 0.5 m, p = 1, alpha = 2, with its localisation / missed / false split; each method at its val-best threshold and at 0.3; LFE-Peaks, LFE-PPN, candidates, E14, E15, E10, E13, B0 | minutes, from saved detections | C1 |
| E20 | where the gain is | per annotated person: range bin, occluded (a return nearer than the person inside the angular extent of the person at their range, in that frame's raw scan), stationary or moving and trajectory age (from `takheliped_oracle.py`'s linked trajectories); recall at matched FP per stratum for candidates, E15 and E13 | ~1 h (reads the raw test scans) | C4; gates E19 |
| E21 | first-detection latency | per linked trajectory, frames from first annotation to first detection at each method's own threshold; candidates, SORT 1/3/5, E14, E13 | minutes | C5 |
| E22 | end-to-end streaming latency of the proposed architecture | stages 1-3 (E13), one stream, 2,000 frames, median and p99, GPU and CPU | ~20 min | C6 |
| -- | uncertainty for every evaluation-only comparison | paired block bootstrap: 30 s blocks within the test recording, 1,000 resamples, 95% interval on the AP difference, per seed of the learned arm | ~10 min per pair | C2 |

Every classical hyperparameter is tuned on val and scored on test once; tuning on test is what inflated the radius sweep's first result (`../rejected-ideas.md`).
Not in this batch: E17 and E18 wait for JRDB; E19 already ran as S2; E11 ran as S3.

## Tests, written before the code

- **`ByteTracker`** (`tests/test_tracker.py`): a low-score candidate extends an existing track in the second pass; a low-score candidate never starts a track and a high-score one does; with `tau_low = tau_high`, birth from every candidate and detection-score output it reproduces `SortTracker` frame for frame on a synthetic stream; a world-static point stays matched under robot rotation; CBMOT's four update functions give their published values; an unmatched track decays and is deleted on score; coasting reports the decayed score and stops at max age.
- **The rule scorer** (`tests/test_tracking_baselines.py`): with `gain = 1`, `manage_slots`' matched value equals CBMOT eq. 6 on a hand-built two-frame case.
- **GOSPA**: hand-computed cases (all matched; all missed gives `c^p / 2` per target; mixed), symmetry, the triangle inequality on random sets, components summing to the total.
- **Block bootstrap**: identical inputs give a zero-width interval, the same seed the same interval, and no block crosses a recording.
- **Strata**: the occlusion flag and trajectory age on synthetic scans and tracks.
- **Real-data checks** (`../dos-and-donts.md`, evidence rules): E14 in its SORT-equivalent setting reproduces `replay_sort`'s AP on the test cache to four decimals; E15 with the default rules and no rescoring reproduces S1's 99.2% / 97.0% coverage; GOSPA's false and missed terms for LFE-Peaks at 0.3 agree with its documented 1.76 FP and 0.63 FN per frame up to the matching rule.

## Run order on a free machine

1. `python memory/scripts/machine_is_free.py`, then the new tests, the full suite and ruff.
2. E15 (fastest, and it decides the bottom rung), then E14, then E16 and E21 from the saved detections, then E20, E22 and the bootstrap.
3. Write the outcome into this file, `../slot-design-evidence.md` (the ladder), `../performance-log.md` (C2's row with its interval) and `../../CHANGELOG.md`.

About 6 h of machine time, no training.

## Outcome

Run 2026-10-02/03 with `utils/tracking_baselines.py` (results in `checkpoints_three_horizon/a56/`, `report.json`).
**On the old stage-2 recipe** (vote-grid candidates, 77.66%; E13 83.86%). The re-run on the adopted recipe (C1's `nms-mean` candidates, the memories at 85.87%), one replay per stage-2 seed, was done 2026-10-09 (`checkpoints_three_horizon/a61/summary.json`); the numbers further below are the old recipe's.

### On the adopted recipe (2026-10-09)

| what reads the stage-2 candidates | AP, 3 seeds | vs candidates, per seed | recall at matched FP | GOSPA |
| --- | --- | --- | --- | --- |
| nothing | 82.19% | -- | 94.13% | 0.494 |
| SORT `min_hits` 1 | 82.31% | +0.14 / +0.12 / +0.12, all above zero | 94.19% | 0.494 |
| ByteTrack as published | 81.17% | -0.94 / -1.11 / -1.00 | 94.29% | 0.494 |
| the memory's bookkeeping with CBMOT's score (E15) | 72.02% | -10.55 / -9.80 / -10.14 | 94.58% | 0.619 |
| ByteTrack + CBMOT, tuned on val (E14) | 83.51% | +1.30 / +1.35 / +1.32, all above zero | **95.56%** | 0.493 |
| **learned state, no attention (E13)** | **85.87%** | **+4.13 / +3.66 / +3.26; over E14 +2.84 [+1.99, +3.73] / +2.31 [+1.28, +3.36] / +1.94 [+0.94, +3.01]** | 95.47% | **0.454** |

- **The claim survives**: E13 over the tuned tracker +2.36 pp (old recipe +2.90), every seed's interval above zero. The tuned tracker's own gain shrank more, +3.30 -> +1.32: a stronger stage 2 leaves a rule less to fix.
- **Predictions**: E14's gain missed low (+1.3 against +1.5 to +2.5); E13 - E14 held (+1.0 to +2.5, all three above zero); E15 below the candidates held; the lead on occluded people (+3.4 recall at matched FP over E14), a trajectory's first 0.5 s (+3.4) and beyond 8 m (+3.2) held.
- **Changed: recall at one matched false-positive rate is now a tie with the tuned tracker** (95.47 against 95.56%; old recipe +1.0). The memory trades visible, standing and near people (-0.5, -1.8, -1.5) for occluded, new and far ones; its lead is ranking over the whole curve (AP, GOSPA 0.454 against 0.493), not more people at one threshold. The paper must not claim higher recall at matched FP on this recipe.
- First-frame detection: E13 93.0%, the candidates 91.2%, E14 83.0% -- still no confirmation delay.
Test split, AP at 0.5 m on every 5th annotated frame; every classical setting tuned on val and scored on test once; 95% intervals from a paired block bootstrap (30 s blocks, 1,000 resamples), one per learned seed.

**Four real-data checks passed before any number was read:** the candidates reproduce their recorded 77.66% AP and 88.8% recall at 2.62 FP per frame; re-scoring the stored memory checkpoints reproduces their recorded test APs (B0 82.96 / 83.18 / 83.88, E10 81.84 / 82.04); E14's SORT row is frame-for-frame identical to `replay_sort` (AP 0.774032 both); and GOSPA's false and missed terms for LFE-Peaks at 0.3 are 1.760 and 0.635 per frame, against its documented 1.76 FP and 0.63 FN (LFE-PPN 5.028 / 0.377 against 5.03 / 0.38).

### The ladder, measured

| rung | AP | vs candidates | recall at matched FP | per-frame cost |
| --- | --- | --- | --- | --- |
| candidates | 77.66 | -- | 88.80 | -- |
| SORT, `min_hits` 1 / 3 | 77.40 / 76.67 | -0.25 [-0.42, -0.08] / -0.98 | 88.52 / 87.68 | ~0.3 ms (CPU, Python) |
| ByteTrack as published (detection scores) | 76.49 | -1.17 [-1.59, -0.78] | 88.83 | ~0.3 ms |
| E15: the memory's bookkeeping, its value used as the score | 69.51 | -8.14 [-12.44, -4.20] | 89.77 | -- |
| **E14: ByteTrack + CBMOT, tuned on val** | **80.96** | **+3.30 [+2.83, +3.77]** | 92.17 | ~0.3-0.5 ms |
| E10: learned scorer on the bookkeeping, no state (2 seeds) | 81.94 | +4.28 | 90.88 | 2.70 ms |
| **E13: learned state, no attention (3 seeds, the proposal)** | **83.86** | **+6.20** | **93.16** | 3.72 ms |
| B0: learned state with attention (3 seeds) | 83.34 | +5.68 | 92.47 | 4.53 ms |

**E13 against the tuned classical pipeline, per seed: +2.95 [+2.06, +3.89], +3.20 [+2.17, +4.31], +2.56 [+1.08, +3.88]** -- mean +2.90, every interval above zero, the mean twice the 1.45 pp seed bar.
B0 against it: +2.00 / +2.22 / +2.92, all intervals above zero.
E10 against it: +0.88 [-0.08, +1.81] and +1.09 [+0.10, +2.07] -- a learned scorer without state barely clears a tuned rule.

E14's tuned configuration: two-stage association, `high_thresh` 0.1, `low_thresh` 0.01, gate 0.3 m, tracks kept 1 s, CBMOT's parallel-addition update, confidence decaying 0.1 per second, coasted tracks reported for 2 s, unmatched candidates at half their score.
**The decay is what makes the rule work**: without it every long-lived track, phantom or person, saturates near 1 and val AP falls to 52.9%; at 0.1 per second it is 73.4%, and coasting and the update rule take it to 76.5% (candidates 72.95% on val).
The same saturation explains E15: the memory's value rule, used directly as a score, either saturates (gain 1, CBMOT's multiplication: 47-49% val) or, below gain 1, pulls a confident detection down to a young slot's low value (best 66.1% val).

### Where the gain sits (E20), recall at matched FP

| stratum | people | candidates | tuned classical | E13 (3-seed mean) | E13 over classical |
| --- | --- | --- | --- | --- | --- |
| visible | 134,585 | 91.2 | 93.9 | 94.7 | +0.8 |
| partly occluded | 17,589 | 72.9 | 81.0 | 83.2 | +2.1 |
| mostly occluded | 1,481 | 53.5 | 65.4 | 68.2 | +2.9 |
| first 0.5 s of a trajectory | 13,080 | 86.0 | 85.1 | 88.1 | +3.0 |
| 0.5-2 s | 31,120 | 90.4 | 94.4 | 93.9 | -0.5 |
| 2-10 s | 79,588 | 91.1 | 95.0 | 95.7 | +0.7 |
| over 10 s | 27,613 | 88.0 | 91.1 | 91.7 | +0.6 |
| moving / slow / still | 83,158 / 26,055 / 42,180 | 92.8 / 88.4 / 85.3 | 95.9 / 92.2 / 89.0 | 97.0 / 92.8 / 88.7 | +1.0 / +0.6 / -0.2 |
| 0-2 / 2-4 / 4-6 / 6-8 / 8+ m | | 89.2 / 90.1 / 89.3 / 86.2 / 88.7 | 92.1 / 93.6 / 93.4 / 89.8 / 91.0 | 93.9 / 93.9 / 94.3 / 90.8 / 92.2 | +1.8 / +0.4 / +0.9 / +1.0 / +1.2 |

Occlusion is "a return at least 0.5 m in front of the person within their angular width" in that frame's raw scan; partly is up to half the width, mostly beyond.
Trajectories are annotations linked within 0.6 m; 2,254 annotations link to nothing and are left out of the age rows (2,262 of the motion rows, adding 8 single-point trajectories with no speed).
**Those 2,254 are FROG's duplicate labels, not people** (checked 2026-10-05): each repeats another circle of its frame in all six columns, the reverse lookup from trajectory to annotation takes the first of the two, and the copy is left without one.
The "recall" on them (candidates 9.7%, classical 15.7%, E13 39.0%) measures how often a method puts a second detection beside a person, which is a false positive everywhere else (see below).

### With the duplicate labels removed (2026-10-05)

| | AP, official labels | AP, de-duplicated | a second detection within 0.5 m of a person, at matched FP |
| --- | --- | --- | --- |
| candidates | 77.66 | 78.70 | 1.7% |
| E14: ByteTrack + CBMOT | 80.96 | 81.90 | 6.5% |
| E10 (2 seeds) | 81.94 | 82.86 | 4.9-5.4% |
| E13 (3 seeds) | 83.86 | 84.60 | 12.5-14.8% |
| B0 (3 seeds) | 83.34 | 84.14 | 8.4-11.5% |
| LFE-Peaks / LFE-PPN | 65.14 / 69.56 | 65.23 / 69.99 | 17.5% / 35.8% |

Every claim survives: E13 over E14 per seed is **+2.74 [+1.85, +3.68], +3.02 [+2.01, +4.14], +2.36 [+0.91, +3.66]** (official +2.95 / +3.20 / +2.56), E14 over the candidates +3.19 [+2.75, +3.64], E10 over E14 +0.86 [-0.11, +1.79] and +1.06 [+0.06, +2.06].
Removing the copies helps every method, and helps least those that double-report most, since a double report was the only way to score a copy.
Scripts and outputs: `checkpoints_three_horizon/a56/duplicates/`.

**Nor does the double reporting move the comparison** (same day): a merge radius chosen on val buys no output anything (E13 -0.04 to -0.05 pp, E14 chooses none), and with both merged E13 over E14 is +2.90 [+2.00, +3.81], +3.16 [+2.14, +4.24], +2.51 [+1.09, +3.82].
Even an oracle that removes every redundant detection near a person lifts E13 and E14 alike, by about 0.5 pp (`rejected-ideas.md`).

### First detection (E21), 672 linked test trajectories

| | detected on its first annotated frame | within 0.5 s | never |
| --- | --- | --- | --- |
| candidates | 84.1% | 97.8% | 26 |
| tuned classical | **74.1%** | 95.7% | 42 |
| E13 (seeds 0 / 1 / 2) | 83.7 / 88.5 / 85.3% | 97.0 / 97.7 / 96.6% | 33 / 26 / 32 |

### GOSPA (E16), mean per frame at each method's GOSPA-best threshold on val

| | GOSPA | localisation | missed | false |
| --- | --- | --- | --- | --- |
| LFE-Peaks / LFE-PPN | 0.679 / 0.740 | 0.146 / 0.243 | 1.816 / 0.972 | 0.315 / 1.015 |
| candidates | 0.530 | 0.106 | 1.368 | 0.328 |
| tuned classical | 0.516 | 0.126 | 1.092 | 0.468 |
| E13 (3-seed mean) | **0.502** | 0.119 | 1.247 | 0.285 |

c = 0.5 m, p = 1, alpha = 2.
GOSPA charges a missed person and a phantom alike, so its best thresholds sit high (0.97-0.99 for the single-frame detectors, 0.55-0.75 for the memory).

### End-to-end latency (E22), the proposed detector streaming one scan at a time

| device | stage 2 | decoding | memory | total, median (p99) |
| --- | --- | --- | --- | --- |
| RX 9060 XT (ROCm) | 5.76 | 2.27 | 4.82 | **12.92 (15.11) ms** |
| CPU, 4 threads | 8.60 | 1.76 | 9.12 | **19.53 (22.38) ms** |
| CPU, 8 threads | 8.60 | 1.72 | 9.10 | 19.47 (22.77) ms |

FROG scans arrive every 25 ms.
The memory's 4.82 ms here includes moving each frame's tensors from numpy, which `memory_ms_per_frame`'s 3.72 ms does not.

### The predictions, scored

| prediction | outcome |
| --- | --- |
| C2/C3: the best classical pipeline lands at 79-81%, the middle row | **held**: 80.96%. C2 stands (E13 +2.90 over it, intervals above zero); C3 is stated as a split -- a tuned rule takes 3.30 of the memory's 6.20 pp, the learned state the remaining 2.90 |
| pure ByteTrack within +-0.5 pp of the candidates | **missed**: -1.17; its high threshold for starting tracks filters real people |
| E15 within 1 pp of E14's best | **missed badly**: -11.4 pp. Association is not the difference, the scoring function is: the value rule saturates or demotes, and only CBMOT's own update functions with a decay work |
| the memory halves GOSPA's false term | **missed**: -13% (0.328 to 0.285); the gain is spread over missed people (-9%) as much as phantoms |
| LFE-PPN has the largest false term | **held** (1.015) |
| larger gain for occluded people | **held, strongly**: +14.7 pp mostly occluded against +3.5 visible, and the learned state's edge over the rule is largest exactly there (+2.9 against +0.8) |
| larger gain beyond 6 m than within 4 m | **missed**: flat, +3.4 to +5.0 at every range |
| larger gain for stationary than moving people | **missed**: +3.5 still against +4.1 moving |
| gain near zero in a trajectory's first 0.5 s | **partly**: the smallest gain (+2.1), not zero; the classical pipeline loses there (-0.9) |
| the memory's first detection within a frame of the candidates' | **held**: same median (0 frames), first-frame share 85.8% against 84.1%; the tuned classical pipeline confirms late, 74.1% |

### What changes in the paper

- **C2 is stated against the strongest classical pipeline, not SORT**: +2.90 pp AP (intervals above zero at every seed) and +1.0 pp recall at matched false positives over ByteTrack + CBMOT tuned on val.
- **C3 becomes a split**: a tuned track-confidence rule recovers about half the memory's gain on its own -- itself a new result for 2D LiDAR -- and the learned state the rest, concentrated on occluded people and new trajectories, where a rule cannot tell a remembered person from a stale track.
- **C5 holds as stated, and sharpens**: the memory adds no confirmation latency, while the classical pipeline that comes closest to it does (first-frame detection 74% against 86%).
- **C6 holds**: 12.9 ms per frame on the GPU and 19.5 ms on four CPU threads for the whole streaming detector, inside a 25 ms scan.
- **C1 gains a formal metric but a smaller effect**: GOSPA ranks the memory best, 5% below its candidates, not by halving the phantom term.
- **C4's mechanism claim is narrowed**: the gain is not range- or motion-specific; it is occlusion-specific.
- **Nothing rests on the unlinked annotations**, since they are duplicate labels; proposed, awaiting the owner: state every number on the official labels, for comparability with every published FROG result, and give the de-duplicated margin (+2.71 mean over E14) once, in a footnote.
