# Tracking-literature experiments: the claims they decide, predicted before running

*keywords:* A56, pre-registration, paper claims, claim ledger, tracking ladder, ByteTrack, CBMOT, rule scorer, E14, E15, E16, E20, E21, E22, GOSPA, block bootstrap, stratified gain, occlusion, first-detection latency, val-tuned baseline

**Status: plan, 2026-09-30, awaiting the owner.**
**Nothing below has run.**
The literature behind it is [`../../docs/RESEARCH.md`](../../docs/RESEARCH.md) §7.4; the work items are [`../../TODO.md`](../../TODO.md) A56; the slot evidence it builds on is [`../slot-design-evidence.md`](../slot-design-evidence.md).
This file records what is predicted and what each outcome changes in the paper, so that a result cannot be reinterpreted after it arrives.

## Where the evidence stands

All on FROG `official` test, AP at 0.5 m, every 5th annotated frame, on the same no-temporal stage-2 candidates.

| rung | what reads the candidates | AP | seeds | memory ms/frame |
| --- | --- | --- | --- | --- |
| 0 | nothing (the candidates) | 77.66 | fixed | -- |
| 1 | SORT, fed every candidate to 0.01, reporting each with its own score | 76.7 / 75.7 / 74.9 at `min_hits` 1 / 3 / 5 | earlier chain's candidates | -- |
| 2 | **a rule that raises scores from track confidence (CBMOT)** | **not measured** | -- | -- |
| 3 | a learned scorer on the slot's bookkeeping statistics, no state (E10) | 81.84 | 1 (second running) | 2.70 |
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
| C7 | Room coordinates help only with good odometry | D1 (+1.14, 3 seeds), E2 (crossover ~0.01-0.02 m/s) | training with jittered poses closes the crossover | E11 (A51, agreed) | the caveat is stated |
| C8 | Honest negatives: attention is unnecessary; the value rule is CBMOT's and earns nothing here; feedback into stage 2 does nothing | E6, E7, E13, D2, E12 | nothing | -- | -- |
| C9 | The slots are also a tracker, and the detector transfers zero-shot | none | identities (JRDB) or mocap (Plozza et al.) | E17 (A49) | out of scope for this paper |

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
| E20 | where the gain is | per annotated person: range bin, occluded (a return nearer than the person inside the angular extent of the person at their range, in that frame's raw scan), stationary or moving and trajectory age (from `three_horizon_oracle.py`'s linked trajectories); recall at matched FP per stratum for candidates, E15 and E13 | ~1 h (reads the raw test scans) | C4; gates E19 |
| E21 | first-detection latency | per linked trajectory, frames from first annotation to first detection at each method's own threshold; candidates, SORT 1/3/5, E14, E13 | minutes | C5 |
| E22 | end-to-end streaming latency of the proposed architecture | stages 1-3 (E13), one stream, 2,000 frames, median and p99, GPU and CPU | ~20 min | C6 |
| -- | uncertainty for every evaluation-only comparison | paired block bootstrap: 30 s blocks within the test recording, 1,000 resamples, 95% interval on the AP difference, per seed of the learned arm | ~10 min per pair | C2 |

Every classical hyperparameter is tuned on val and scored on test once; tuning on test is what inflated A38's first result (`../rejected-ideas.md`).
Not in this batch: E17 and E18 wait for A49; E19 (an occlusion input for coasting) runs only if E20 shows that coasting's failures sit on occluded people; E11 stays in A51.

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

*Empty until the runs report.*
