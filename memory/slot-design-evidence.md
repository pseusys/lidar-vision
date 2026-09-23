# Slot design: every decision, and what measured it

*keywords:* slots, object memory, stage 3, design decisions, ablation, D1, D2, D3, D4, room coordinates, value rule, decay, association gate, fade time, rescoring, coasting, defensible, novelty

Stage 3 (`ObjectMemory`) is the part of this project with no direct precedent, so each of its choices has to be stated with the
measurement that supports it -- or stated as unsupported. This is that list. Build status and open work are in `TODO.md` A51;
history and full numbers in `CHANGELOG.md`; the design's own reasoning in `docs/PROPOSAL.md` §5.5.

## How to read the numbers here

- Every stage-3 figure below is FROG `official` test, whole split, every 5th annotated frame, on the **no-temporal** stage 2
  (seed 0, 77.66% AP candidates). Coarse convolutions were deliberately excluded while the slots were being studied.
- **The screening threshold is ~1.5 pp of test AP**, from B0's three seeds (82.96 / 83.17 / 83.88, sd 0.47).
- **No threshold was ever pre-registered for the calibration metrics** (recall at the candidates' false-positive rate, coasting
  precision). Two unrelated arms moved them by nearly the same amount, so a difference there is currently suggestive only.
- One seed unless a row says otherwise. An ablation "supports" a decision when it *loses*.

## What the memory is worth at all

| claim | evidence |
| --- | --- |
| The memory earns its place | **+5.68 pp of AP over its own candidates** (83.34% mean against 77.66%), and **+3.7 pp of recall at the candidates' own false-positive rate** (92.5% against 88.8%), 3 seeds |
| It is not just temporal filtering | SORT replayed on the same candidates **loses** at every setting (76.7 / 75.7 / 74.9% at `min_hits` 1/3/5 against 76.92%) -- `CHANGELOG.md` 2026-09-16 |
| Memory belongs inside the model, not behind it | every place-keyed variant failed first: persistence map 0.57 false positives removed per false negative added, background subtraction 1.90, acausal trail oracles 2.54 and 4.20, against the merge radius's 21.9 (`rejected-ideas.md`) |
| It costs little | **4.4-4.6 ms per frame**, one stream, beside stage 2's 6.5 ms, inside FROG's 25 ms budget |

## The decisions inside it

| decision | what measured it | verdict |
| --- | --- | --- |
| **Room coordinates** for slots, so a static object keeps zero velocity | D1, sensor frame instead: **82.02%, -1.32 pp**; recall at matched FP 92.2%, B0's lowest | **supported, not settled**: one seed, inside the threshold, and the number bundles a second effect (in the sensor frame the robot's own motion pushes objects out of the 0.6 m association gate) |
| **256 slots** | 64 / 128 / 256 score 80.12 / 80.20 / 80.10%, and the memory never fills | **settled**: capacity is not a limit, and no slot is ever evicted for it |
| **Mutual-nearest association within a 0.6 m gate**, and a **10 s fade** | S1, a rule-only replay over 3 gates x 3 fades: people with a live slot within 0.5 m move 99.0-99.3%, and with a *matched* slot 96.9-97.0%, across the whole grid | **settled, and the rules are not the limit**: every remaining loss is downstream, in the learned head. This is why a learned association (S4) was dropped without a training run |
| **Accumulated value** holds a slot, not the latest score | D2, latest score instead: **83.94%, +0.60 pp**; recall 92.7%, B0's best | **not supported**. The ablation does not lose. Either the learned slot state already covers a briefly hidden person, or FROG does not test the case. Do not claim this rule without more seeds |
| **Learned, input-dependent decay** per slot (1-60 s) | D3, one fixed 8 s decay: **82.44%, -0.90 pp**; recall 92.1%, below every B0 seed; 17,536 parameters cheaper | **weakly supported**: right direction, inside the threshold, one seed |
| **A slot is created by any candidate above 0.01** | replay: 0.11-0.25 spawns per frame at any setting, and capacity is never reached | **settled as safe**, though confirmation-before-creation has not been tried (A51 S3) |
| **Rescoring starts silent** | T1, zero-initialised head instead of a scalar gate: AP +1.05 (inside the threshold), recall at matched FP **93.4%** and coasting precision **32.2%**, both above every B0 seed | **the mechanism is confirmed, the parameterisation is not**: with a scalar gate at zero the head's own gradient is scaled by zero, which a test pins directly |
| **Slots report people no candidate covers** (coasting) | trained: 0.08-0.29 reports per frame at **22-32% precision**; at one epoch it is 0.002 at 0% | **works, and is undertuned**: it needs training time before it does anything, and its precision is the open item (A51 O2) |
| **The bookkeeping is a fixed rule, not learned** | S1 (the rules do not bind) and D2 (the value rule earns nothing) both say there is little to learn; the spawn and eviction decisions are `argmin`/`argsort` and have no gradient without a relaxation | **defensible, and D4 tests the cheap part** -- learning the value rule's `gain` and `fade_time_s` -- but D2's null makes a null there likely |

## Two findings that change how stage 3 is read

**The memory was never "buying precision for recall".** Its rescoring shifts the *whole* score distribution down (people by
-2.8 to -3.5 logits), so a threshold calibrated for stage 2 understates it. Scored at its own operating point the memory
**strictly dominates its input**: at 0.22 it covered 135,520 people against 134,196 at **1.767 false positives per frame against
1.910**. Every stage-3 number is now reported at its own threshold as well as at 0.3.

**The time step is part of the design, not a detail.** FROG stamps its 40 Hz scans in bunches (38 ms, 38 ms, under 0.1 ms), and
dividing a slot's displacement by one stamped interval gave matched slots a median speed of **3.97 m/s against a measured 0.56
m/s**, with unmatched slots coasting away at ~11 m/s. Stage 3 now uses a running average of the intervals (`frame_periods`).
Any dataset with irregular stamps needs the same treatment -- see `gotchas.md`.

## Where the remaining misses are, on the current model (2026-09-21)

Measured on B0's best seed with `memory_diagnosis.py` and `three_horizon_oracle.py`. Both had last been run on the old chain,
whose stage 2 no longer loads, so the figures they replace are retired.

**At the shared 0.3 threshold**, stage 3 covers 128,895 of 153,655 people against stage 2's 136,417 -- but at **1.092 false
positives per frame against 2.628**. Of the 10,197 people it "loses", **99.1% were pushed under the threshold by rescoring**, and
**a live slot sat within the gate for 100% of them**, median value 0.966, matched 0.0 s earlier. Nothing was forgotten: the memory
knew about every one of them. This is the same decalibration as before, now measured on the current model, and it is why stage 3
is reported at its own operating point.

**The rescoring head is not selective.** Candidates covering a person are pushed down by a median **-3.526 logits** (97.3% of
them), candidates covering nobody by **-0.729** (69.6%). It suppresses the right things more weakly than the wrong ones, which is
what A51's O1 targets.

**The headroom above stage 3 is large and mostly untaken.** Counting only stage-2 misses whose own trajectory was detected before
and after the gap:

| bridge | recoverable people | ceiling | stage 3 takes | share |
| --- | --- | --- | --- | --- |
| 0.5 s | 6,371 | 91.7% | 1,806 | 28.3% |
| 2.0 s | 10,171 | 94.2% | 2,209 | 21.7% |
| 10.0 s | 13,067 | 96.1% | 2,301 | 17.6% |

Per trajectory, stage 2 mostly tracks 71.1% of people, partially tracks 21.4% and mostly loses 7.4%; only **0.7%** of people sit
on trajectories it never detects at all, so almost nothing is irreducible for interpolation.

## Capacity is not the limit; reporting is (2026-09-22)

`memory_diagnosis.py --recovery` on B0's best seed, asking the question that decides where the remaining work goes.

**The memory already holds almost everyone it fails to report.** Of the 26,996 people stage 3 does not cover at 0.3, a live slot
sits on **25,778 (95.5%)**, carrying a median value of **0.986**, and that slot is coasting for 12,843 of them. Meanwhile the
memory uses a median of **24 slots of 256** (p99 84, max 107). **So spare capacity and more candidates per frame are not the
lever** -- the slots exist, hold the right positions, and are confident.

**What reporting more of them would cost**, sweeping the coasting threshold:

| coast threshold | people recovered | false positives added per frame | FP per person |
| --- | --- | --- | --- |
| 0.30 (current) | 1,926 | 0.114 | 3.0 |
| 0.20 | 4,602 | 0.426 | 4.6 |
| 0.10 | 8,347 | 1.848 | 11.1 |

**The learned head is better than the rule it sits on**, so it is not ignoring the memory: ranking the same reports by the slot's
accumulated value instead recovers 2,946 people at 0.299 false positives per frame (5.1 each) where the head reaches ~3,000 at
about 0.2 (roughly 3 each). Both are expensive, so **coasting cannot be fixed by moving its threshold** -- the head needs to
separate a remembered person from a remembered phantom better than either signal does today.

**Order this implies:** the 26,996 figure is measured at the shared 0.3 threshold, which stage 3 is known to be miscalibrated
against, so calibration (O1) comes first and the recovery pool is re-measured on its result before any coasting work (O2).

## What is not established

- **D1, D2 and D3 each rest on one seed**, and none crosses the screening threshold. The memory as a whole is well supported; its
  internal choices are not, individually.
- **The calibration metrics have no noise estimate**, so "T1 beats every B0 seed" is weaker than it sounds.
- **Coarse convolutions were excluded throughout**, so nothing here says how the slots interact with a temporal stage 2; that is
  `TODO.md` §D's question and A51 phase 4's job.
- **Only FROG.** DROW and JRDB are A49.
- **The oracle bound is an interpolation bound, not an achievable target**: it assumes a perfect decision about when to bridge a gap, which is exactly what stage 3 has to learn.
