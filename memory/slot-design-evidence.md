# Slot design: every decision, and what measured it

*keywords:* slots, object memory, stage 3, design decisions, ablation, D1, D2, D3, D4, room coordinates, value rule, decay, association gate, fade time, rescoring, coasting, defensible, novelty

Stage 3 (`ObjectMemory`) is the part of this project with no direct precedent, so each of its choices has to be stated with the
measurement that supports it -- or stated as unsupported. This is that list. Build status and open work are in `TODO.md` A51;
history and full numbers in `CHANGELOG.md`; the design's own reasoning in `docs/PROPOSAL.md` §5.5.

## How to read the numbers here

- Every stage-3 figure below is FROG `official` test, whole split, every 5th annotated frame, on the **no-temporal** stage 2
  (seed 0, 77.66% AP candidates). Coarse convolutions were deliberately excluded while the slots were being studied.
- **Every metric's spread, from B0's three seeds** (2026-09-25). An arm is judged against the last column, three standard
  deviations, which is where the ~1.5 pp AP threshold came from:

  | metric | seeds | mean | sd | 3 sd |
  | --- | --- | --- | --- | --- |
  | test AP | 82.96 / 83.17 / 83.88 | 83.34 | 0.48 | **1.45 pp** |
  | AP at 0.3 m | 81.36 / 81.52 / 82.17 | 81.68 | 0.43 | 1.29 pp |
  | recall at matched FP | 92.16 / 92.53 / 92.72 | 92.47 | 0.29 | **0.86 pp** |
  | coasting precision | 27.37 / 22.31 / 26.61 | 25.43 | 2.73 | **8.19 pp** |

  **A three-sample standard deviation is itself uncertain** -- its 95% interval spans roughly half to six times the estimate --
  so read these as order-of-magnitude bars, not decision rules. Coasting precision is the noisiest metric by far, which is why
  the coasting gains reported for T1 and T2 do not survive it.
- One seed unless a row says otherwise. An ablation "supports" a decision when it *loses*.

## What the memory is worth at all

| claim | evidence |
| --- | --- |
| The memory earns its place | **+5.68 pp of AP over its own candidates** (83.34% mean against 77.66%), and **+3.7 pp of recall at the candidates' own false-positive rate** (92.5% against 88.8%), 3 seeds |
| It is not just temporal filtering | SORT replayed on the same candidates **loses** at every setting (76.7 / 75.7 / 74.9% at `min_hits` 1/3/5 against 76.92%) -- `CHANGELOG.md` 2026-09-16. **Only against SORT**, which saw every candidate down to 0.01 but can only filter: a rule that *raises* scores from track confidence (CBMOT) has not been run (`../TODO.md` A56 E14, E15) |
| Memory belongs inside the model, not behind it | every place-keyed variant failed first: persistence map 0.57 false positives removed per false negative added, background subtraction 1.90, acausal trail oracles 2.54 and 4.20, against the merge radius's 21.9 (`rejected-ideas.md`) |
| It costs little | **4.4-4.6 ms per frame**, one stream, beside stage 2's 6.5 ms, inside FROG's 25 ms budget |
| Its cost does not grow with the horizon | the memory is carried as state, so per-frame work is **constant in elapsed time**; window methods (DR-SPAAM, Li2Former) re-encode `T` scans every frame, so their cost scales with the horizon they want. This is the efficiency claim the design is built on (owner, 2026-09-25): linear-time memory for an embedded platform |

## The decisions inside it

| decision | what measured it | verdict |
| --- | --- | --- |
| **Room coordinates** for slots, so a static object keeps zero velocity | D1 at **three seeds**, sensor frame instead: 82.02 / 82.57 / 82.01 (mean **82.20**, sd 0.32) against B0's 83.34, **-1.14 pp, Welch t = 3.40, p = 0.034** | **supported**, the only internal choice that is. Two caveats stay attached: the ablation also disturbs association (the robot's own motion pushes a static object out of the 0.6 m gate, so -1.14 pp is a lower bound on the frame's own contribution), and recall at matched false positives barely moves (-0.17 pp, p = 0.41), so the gain is in ranking rather than at the operating point |
| **256 slots** | 64 / 128 / 256 score 80.12 / 80.20 / 80.10%, and the memory never fills | **settled**: capacity is not a limit, and no slot is ever evicted for it |
| **Mutual-nearest association within a 0.6 m gate**, and a **10 s fade** | S1, a rule-only replay over 3 gates x 3 fades: people with a live slot within 0.5 m move 99.0-99.3%, and with a *matched* slot 96.9-97.0%, across the whole grid | **settled, and the rules are not the limit**: every remaining loss is downstream, in the learned head. This is why a learned association (S4) was dropped without a training run |
| **Accumulated value** holds a slot, not the latest score | D2, latest score instead: **83.94%, +0.60 pp**; recall 92.7%, B0's best | **not supported**. The ablation does not lose. Either the learned slot state already covers a briefly hidden person, or FROG does not test the case. Do not claim this rule without more seeds. **And it is not ours:** the update is CBMOT's score update `1 - (1-c)(1-s)` with a gain (Benbarka et al., IROS 2021), and D2 is its count-based ablation, which lost there by up to 1.83 AMOTA on nuScenes (`../docs/RESEARCH.md` §7.4.1) |
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

## How to state the efficiency claim (2026-09-25)

The owner's framing is **linear-time memory, not quadratic-time attention over a window**. Three things have to be said precisely
or a reader will object:

- **A recurrent update is not the quadratic one.** It is O(1) per step and O(T) over a sequence -- sequential, not quadratic. What
  is quadratic is self-attention over a `T`-frame window, or attention across beams as in Li2Former. Stage 3's selective diagonal
  recurrence is therefore **the mechanism that delivers the claim**, not an exception to it, and calling the design
  "memory without recurrence" would be wrong.
- **What the design avoids** is re-encoding a window every frame. Per-frame work is constant in elapsed time: attention runs over
  `K` slots and `P` candidates, never over history.
- **The one super-linear term is the slot-to-slot exchange**, O(K^2) per frame -- about 65k pairs at 256 slots. It is constant in
  time but quadratic in slot count. A51 E4 measures the cost curve against slot count and E6 ablates the exchange; if E6 is a null,
  the design loses its only quadratic term and the claim becomes cost **linear in slot count and constant per frame**.

## Phase 5: the mechanism, not the choices inside it (2026-09-27)

Four evaluation-only experiments and the first ablation of a learned mechanism. `utils/slot_experiments.py` runs the four;
`utils/score_memory.py` scores a checkpoint from a run that was interrupted.

**How long a memory the model uses (E1).** The memory is emptied every N seconds at evaluation; candidates alone score 77.66%.

| emptied every | 0.5 s | 1 s | 2 s | 5 s | 10 s | 30 s | never |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AP | 79.41 | 80.46 | 81.51 | 82.65 | 83.53 | 83.73 | 83.88 |
| gain over candidates | +1.75 | +2.80 | +3.85 | +4.99 | **+5.87** | +6.07 | +6.22 |

**About 95% of the memory's value is realised by 10 seconds**, and the curve climbs steeply through 2-10 s. This is the
measurement the project's thesis needed: the useful horizon is tens of seconds, far past a 5-frame window (0.15 s) or the coarse
convolution (0.8 s). It also validates `fade_time_s = 10 s`, which is where the curve flattens.

**What the room frame costs when odometry drifts (E2).** Drift is applied to the poses at evaluation; the sensor-frame arm (D1)
never reads them, so it is flat by construction.

| drift | 0 | 0.005 m/s | 0.02 m/s | 0.05 m/s | 0.1 m/s |
| --- | --- | --- | --- | --- | --- |
| room frame AP | 83.88 | 82.96 | 81.25 | 78.27 | 73.10 |
| sensor frame AP | 82.57 | 82.57 | 82.57 | 82.57 | 82.57 |

**The room frame's advantage survives only below roughly 0.01-0.02 m/s of drift**, and inverts above it. FROG's robot moves about
0.5 m/s and wheel odometry typically drifts 1-5% of distance travelled, i.e. 0.005-0.025 m/s, so the crossover sits inside the
realistic range. D1's +1.14 pp is therefore conditional on good odometry. Caveats: the drift here is a smooth wander rather than a
random walk, the two curves are different trained models, and the room-frame model never saw drift in training -- training with
jittered poses is the obvious repair and is untried.

**Where the gain sits by crowd size (E3).**

| people per frame | 0-1 | 2-3 | 4-5 | 6+ |
| --- | --- | --- | --- | --- |
| frames | 1,823 | 4,748 | 2,370 | 1,093 |
| candidates AP | 65.21 | 80.83 | 80.12 | 80.71 |
| memory AP | 71.44 | 86.70 | 86.25 | 87.34 |
| gain | +6.22 | +5.87 | +6.13 | **+6.62** |

The gain is largest in the most crowded frames and nearly as large when almost nobody is present. **Memory helps roughly
everywhere**, which is a weaker claim than "memory pays in crowds" and the one the data supports.

**Cost against slot count (E4).** 4.37 / 4.37 / 4.32 / 4.37 / 4.46 ms per frame at 64 / 128 / 256 / 512 / 720 slots: **11x the
slots for 2% of the time**. The O(K^2) exchange is invisible at these sizes on this GPU; on a weaker embedded part it may not be.

**The slot-to-slot exchange earns nothing (E6).** Removing it: AP 83.58 against B0's 83.34, recall at matched false positives 92.63
against 92.47 -- both inside the bars -- while **coasting precision rises to 37.35% (+11.9, its bar is 8.19)**, 66,560 parameters
disappear, per-frame time falls 9% (3.93 ms) and GPU memory 40% (3.2 against 5.3 GiB). One seed, and the checkpoint comes from the
epoch where the run was interrupted, so the arm is slightly under-trained -- which makes the null more convincing, not less.
**Dropping the exchange removes the design's only super-linear term at no measurable accuracy cost.**

**The learned slot state is the mechanism, and it is the only ablation that has ever cost anything (E10, two seeds,
2026-09-29/10-01).** Removing the state entirely -- no recurrence, no `spawn`/`update`/`recall`, the rescoring head reading a
candidate and the bookkeeping statistics of the slot matched to it -- costs on every metric, and clears its bar on three of four.

| | B0 mean (3 seeds) | E10 (2 seeds) | per seed | delta | 3 sigma bar | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| test AP | 83.34 | 81.94 | 81.84 / 82.04 | **-1.40** | 1.45 | **inside the bar** |
| AP at 0.3 m | 81.68 | 80.38 | 80.26 / 80.51 | -1.30 | 1.29 | clears, just |
| recall at matched FP | 92.47 | **90.87** | 90.80 / 90.94 | **-1.60** | 0.86 | clears, 1.9x |
| coasting precision | 25.43 | **11.63** | 11.35 / 11.90 | **-13.80** | 8.19 | clears, 1.7x |
| parameters | 306,443 | **55,691** | -- | -82% | -- | |
| ms per frame | ~4.5 | **2.69** | 2.70 / 2.67 | -40% | -- | |

**The second seed moved the headline metric inside its bar, and that is recorded rather than worked around.** At one seed AP read
-1.50 against a 1.45 bar; the second seed's 82.04 pulled the mean to -1.40. AGENTS.md makes wp-AUC at 0.5 m lead any accuracy
answer, so the honest headline is now **three of four metrics clear, and AP does not**.

**Where the uncertainty actually sits.** E10's two seeds agree to 0.20 pp on AP, far tighter than B0's own three (sd 0.48). The
bar is three times *the control's* standard deviation, so what limits this comparison is the control's spread, not the arm's.
**The pre-registered screening rule does not say what to do with a multi-seed arm** -- it was written for one-seed screening
against a three-seed control -- and choosing a two-sample statistic now, after seeing the result, would be picking the test that
gives the answer we want. So the recorded verdict stays the pre-registered one. If a two-sample reading is wanted, it should be
pre-registered before any further seed is run.

**What it settles.** `docs/RESEARCH.md` §3.5 lists three things separating this design from a tracker with a learned scorer, and
"the slot state is learned, not hand-coded" was the weak one, resting only on D3's 0.90 pp, which sits inside its bar. It is now
much the strongest: without the state the arm loses more than any other arm has ever lost, on all four metrics at once, and past
its bar on three. Recall at matched false positives, at 1.9x its bar, and coasting precision, at 1.7x, carry the claim -- not AP.

**Where the state earns it.** Coasting precision collapsing by 14 pp -- to less than half the control -- says where. Coasting is
exactly the case with no current measurement, so nothing but the carried state can distinguish a person who is genuinely occluded
from a slot that is merely stale. Bookkeeping alone cannot, and the AP and recall losses follow from that.

**What it costs to keep.** 250,752 parameters and about 1.8 ms per frame, for +1.50 pp AP, +1.67 pp recall at matched false
positives and +14.08 pp coasting precision. Contrast E6 and E7, which remove attention for nothing: the state is not where this
architecture is over-built. **E13 should therefore propose E6+E7, not E10.**

One seed, run uninterrupted, against a three-seed control. The effect is large enough relative to the bars that a second seed
would sharpen rather than decide it.

**Both attention mechanisms are free to remove, at three seeds (E13 / E6+E7, 2026-09-30).** `--no-exchange --no-learned-match`
leaves a rule-based association, a learned per-slot recurrence and the rescoring head: 173,579 parameters against 306,443.

| metric | B0 mean (sd) | E6+E7 mean (sd) | delta | 3 sigma bar | verdict |
| --- | --- | --- | --- | --- | --- |
| test AP | 83.34 (0.48) | 83.86 (0.32) | +0.53 | 1.45 | ties |
| AP at 0.3 m | 81.68 (0.43) | 82.14 (0.35) | +0.46 | 1.29 | ties |
| recall at matched FP | 92.47 (0.29) | 93.16 (0.70) | +0.68 | 0.86 | ties |
| coasting precision | 25.43 (2.73) | 29.81 (0.56) | +4.38 | 8.19 | ties |
| parameters | 306,443 | **173,579** | **-43%** | -- | |
| ms per frame | 4.53 | **3.72** | **-18%** | -- | |

Per-seed AP 83.91 / 84.15 / 83.52. **Nothing clears in either direction**, which is the result E13 wanted: the simplification
costs nothing measurable and every point estimate is slightly *above* the control. Note that the third seed matters -- at two
seeds recall at matched false positives read +1.08 and cleared its 0.86 bar; seed 2's 92.36 pulled the mean to +0.68 and inside
it. A two-seed reading of the tightest-barred metric was wrong, exactly the failure the three-seed protocol exists to prevent.

**Read E10 and E13 together.** They are the same experiment run in opposite directions, and they separate the architecture
cleanly: the memory's **state** is load-bearing (removing it clears every bar downward), its **attention** is not (removing both
mechanisms clears nothing in either direction). The architecture to propose is E6+E7 -- 43% fewer parameters, 18% faster, no
measurable cost -- with the full model as the ablation rather than the reverse.

**How much headroom feedback into stage 2 has at all (2026-09-30, no training).** `memory_diagnosis.py --recovery` now splits
the memory's misses by whether stage 2 produced **any** candidate near the person, at any score. Where one exists, rescoring can
raise it and feedback is not needed; where none exists, no rescoring invents one, and only feedback -- or a coasting report --
can reach that person. On B0 seed 0, FROG test, threshold 0.3:

| | people | share |
| --- | --- | --- |
| annotated people | 153,655 | |
| covered by stage 3 | 129,755 | 84.4% |
| missed **with** a candidate rescoring could raise | 19,832 | 83.0% of misses |
| missed with **no candidate at all** | 4,068 | 17.0% of misses |
| ... of those, a live slot holds the person | **2,930** | **1.91% of all people** |
| ... of those, the slot is already coasting | 2,930 | all of them |

**So the ceiling on what feedback can do that nothing else can is +1.91 pp of recall**, and it is reached only by converting
every one of those 2,930 into a detection with no false positives. Every one of them is already held by a coasting slot, so
feedback finds nobody the memory does not already have -- its unique value would be turning a coasted slot position into a
scan-localised detection, and possibly at better precision than the coasting head manages.

**Beyond that pool, feedback can only duplicate rescoring.** It could improve the 19,832 by giving stage 2 better features, but
that is the job rescoring already does, one operation before the loss instead of five. Which is also why the E12 runs found
nothing to learn: for 83% of the misses the gradient has a shorter path to the same improvement.

**Read with E12's capacity measurement**, this says the feedback path was both under-built (0.02-0.045% of the bottleneck) *and*
aimed at a small target (1.91 pp). Fixing the first is cheap; the second bounds the reward.

**Free-space evidence does not fix coasting precision (S2, one seed, 2026-10-01).** Each slot was told how much further the laser
reached than the slot claims to be -- negative where something is in front of it, positive where the beam passed through its own
position and found nothing, which is direct evidence against it. The feature is real and verified; the result is a null.

| | B0 mean (3 seeds) | E6+E7 (3 seeds) | S2, visibility (1 seed) | vs B0 | 3 sigma bar | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| test AP | 83.34 | 83.86 | 83.58 | +0.24 | 1.45 | ns |
| AP at 0.3 m | 81.68 | 82.14 | 81.92 | +0.24 | 1.29 | ns |
| recall at matched FP | 92.47 | 93.16 | 92.58 | +0.10 | 0.86 | ns |
| **coasting precision** | 25.43 | 29.81 | **27.38** | **+1.95** | 8.19 | **ns** |
| coasting reports per frame | 0.176 | 0.237 | **0.074** | **-58%** | -- | |

**At the shared threshold the head looks merely more conservative** -- the matched-rate sweep below shows it is also more
discriminating, so read both. It reports 58% fewer coasting
detections for 1.95 pp more precision. A genuinely better ranking would buy far more precision than that at less than half the
volume -- keeping only the best half of a well-ordered list raises precision sharply. Instead the whole model shifted down:
precision at the shared 0.3 threshold rose to 70.3% with false positives at 1.087 against B0's ~1.6, and false negatives rose to
0.495. That is a calibration shift, not a discrimination gain, and `recall at matched FP` -- which removes calibration from the
comparison -- is flat at +0.10.

**The matched-rate comparison, done 2026-10-02, corrects the paragraph above: the head *is* more discriminating.** Comparing at
a fixed 0.3 threshold was the wrong comparison. Sweeping the coasting threshold and reading people recovered against the false
positives that recovery costs:

| false positives added per frame | 0.05 | 0.26 | 1.10 | 3.35 | 25.1 |
| --- | --- | --- | --- | --- | --- |
| S2 recovers | 978 | 3,288 | 6,904 | 9,250 | 12,379 |
| B0 recovers, interpolated to the same cost | 1,040 | 2,975 | 6,113 | 8,525 | 11,658 |
| S2 / B0 | 0.94 | **1.11** | **1.13** | **1.08** | **1.06** |

**At matched cost the free-space feature recovers 6-13% more people**, everywhere except the cheapest point. So "conservative,
not discriminating" was half right and half wrong: it is both. The ranking improved.

**It still buys nothing end to end, and that is the result that governs.** S2's candidates-plus-rescoring stage covers **2,240
fewer** people than B0 before coasting begins, so total coverage at every matched cost is 1,400-2,300 people *worse*. Part of
that is the shared 0.3 threshold being miscalibrated for both models, which is exactly why the operating-point metric exists --
and recall at each model's own matched-false-positive point is **92.58 against 92.47, flat**. **A better coasting ranking does
not reach the end of the pipeline, because coasting contributes a small share of total coverage.**

**Timing cleared, and the first measurement was wrong.** 7.76 ms per frame was measured while the host was loaded enough to
stretch training epochs from 677 s to 970 s. Re-timed on a quiet machine: **4.74 ms at 256 slots** (4.86 / 4.81 / 4.74 / 4.92 /
5.21 at 64 / 128 / 256 / 512 / 720), against B0's 4.37. **The feature costs about 0.4 ms, not 3.2**, and the curve stays flat in
slot count. Never quote a timing taken beside other work.

**Why it may have failed, for whatever comes next.** The information may already be reachable without it -- a slot's learned
state sees the pattern of candidate matches over time, from which "currently occluded" is partly inferable, which would make the
explicit feature redundant exactly as the feedback prior was. Or the question coasting actually fails is not "is this position
occluded" but "was this remembered object ever a person", which free space cannot answer.

**Training with jittered odometry flattens the room frame's collapse, but does not remove D1's caveat (S3 / E11, one seed,
2026-10-02).** Drift was applied during training at a rate drawn **per stream and redrawn each chunk**, uniform over
`[0, 0.05] m/s` plus `2 deg/s` of heading -- domain randomisation rather than one fixed rate. Then E2's sweep was re-run on it.

| room-frame AP at drift | 0 | 0.005 | 0.020 | 0.050 | 0.100 |
| --- | --- | --- | --- | --- | --- |
| B0, trained on clean odometry (E2) | 83.88 | 82.96 | 81.25 | 78.27 | 73.10 |
| **S3, trained with jitter** | 83.50 | 82.77 | 81.59 | **80.02** | **77.44** |
| delta | -0.38 | -0.19 | +0.34 | **+1.75** | **+4.34** |

**It works where it was worst and costs nothing where it was best.** The penalty at 0.1 m/s falls from 10.78 pp to 6.06 pp, and
zero-drift AP gives up 0.38 pp, well inside its 1.45 bar. On clean test odometry S3 is indistinguishable from the control on
every barred metric: AP 83.50 (+0.16), AP at 0.3 m 81.76 (+0.07), recall at matched FP 93.09 (+0.61), coasting precision 29.04
(+3.61). **So the robustness is free.**

**What it does not do is move the crossover.** Within this sweep the sensor frame is flat at 82.03, so S3's room frame wins
below roughly **0.015 m/s** and loses above it -- essentially where B0's crossover already sat. Domain randomisation reduced the
*size* of the penalty beyond the crossover, not the existence of one. **D1's conditional claim therefore stands**, and the
write-up must keep it: room coordinates win on good odometry, and a deployment with drift past ~0.02 m/s should prefer the
sensor frame.

**Two provenance notes, because they limit what may be compared.** The sensor-frame baseline differs between the two sweeps
(82.03 here against E2's 82.57), so the sensor arm must have used a different D1 checkpoint; **compare room against room, and
take any crossover from within a single sweep**. And `slot_experiments.py odometry` writes a fixed path, so this run
**overwrote E2's `a51_odometry.json`** -- E2's numbers survive only in `results/a51_odometry.log`, and the script should take an
output path.

## What is not established

- **D2, D3, O1, O2, T1 and T2 each rest on one seed**, and none crosses the screening threshold; D1 has three. The memory as a
  whole is well supported, and so now are the two decisions that matter most -- the learned state (E10) and the removal of both
  attention mechanisms (E13, three seeds) -- but the smaller internal choices are single-seed screening and must be reported as
  that, not as results. **D3 matters most of the six**: a fixed 8 s decay costs 0.90 pp, inside its bar, so "the per-slot decay
  is learned" is currently unsupported even though E10 proves the state as a whole is load-bearing.
- **Nothing says how the slots interact with a coasting head that knows about occlusion**, which is where the remaining recall
  is (`TODO.md` A56 E19/E20).
- ~~**The calibration metrics have no noise estimate**~~ -- **superseded by E9, 2026-09-25**: recall at matched false positives
  has sd 0.29 (bar 0.86) and coasting precision sd 2.73 (bar 8.19) over B0's three seeds. Re-judged against those, T1's recall
  gain (+0.88) clears and its coasting gain (+6.72) does not.
- **Coarse convolutions were excluded throughout**, so nothing here says how the slots interact with a temporal stage 2; that is
  `TODO.md` §D's question and A51 phase 4's job.
- **Only FROG.** DROW and JRDB are A49.
- **The oracle bound is an interpolation bound, not an achievable target**: it assumes a perfect decision about when to bridge a gap, which is exactly what stage 3 has to learn.
