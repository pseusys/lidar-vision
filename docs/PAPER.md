# Long-horizon temporal evidence for person detection in 2D range data

**Working draft.** Started 2026-09-10.
Every number here is measured in this repository and traceable to a `CHANGELOG.md` entry; published numbers are cited, never re-derived (`memory/performance-log.md`).
This document holds the argument. The evidence behind each claim lives in `memory/`, the history in `CHANGELOG.md`, the open work in `TODO.md`.

---

## Theory

A 2D LiDAR mounted at knee height returns a single 720-point range slice in which a person's legs and a chair's legs are very nearly the same object: two narrow, roughly parallel returns about 20 cm apart. A detector given one such slice must therefore either miss people or fire on furniture, and the published state of the art does the latter at a rate of **1.4 to 3.4 false positives per frame** — from one phantom person for every two real ones to more phantoms than people. This is invisible on the field's own benchmark because average precision is a ratio: with three to four annotated people in every scored frame, 1.4 phantoms still leaves 63% precision, and the FROG benchmark contains no frame in which those phantoms would be the *entire* output. Our first contribution is to show that the phantom rate does not depend on whether anyone is present at all: separating duplicate detections on already-detected people from genuine phantoms, LFE-Peaks produces **1.387** phantoms per populated frame against **1.391** per person-free frame, and LFE-PPN **3.416** against **3.174** — agreement to within 0.3% and 7.6% respectively, on scenes that share nothing but the sensor. The benchmark is therefore not merely incomplete but structurally unable to charge for the dominant error mode, because a ratio conceals a constant. Our second contribution is to characterise what those phantoms are, since that determines what could remove them: they are **persistent**, not transient — 88.3% of LFE-Peaks' and 77.6% of LFE-PPN's recur within 0.3 m in the very next frame, against 99.8% for genuine people — so they are static scene structure correctly identified as leg-like, and no confirm-then-report filter can suppress them without suppressing people too. This is the crux of the argument. Over a window of a second or two a standing person and a chair are *both* stationary and *both* leg-shaped; there is no evidence in that window that separates them, which is precisely why destroying the temporal ordering of a five-frame window costs a temporal detector only **0.5 percentage points** of AP, and why — as we show below — a standard tracker buys a 13% phantom reduction for 1.5pp of AP and nothing better at any price. The evidence that does separate them is that a person eventually displaces and a chair never does, a fact that only becomes available over tens of seconds. We therefore argue that the useful temporal signal for knee-height person detection lives at a horizon one to two orders of magnitude longer than the 0.2–2 s windows the literature uses, that exploiting it requires an architecture whose receptive field in time is measured in tens of seconds rather than in frames, and that demonstrating it requires an evaluation protocol charging for absolute false positives rather than diluting them in a ratio.

## Problem statement

**Reduce the number of false-positive detections in knee-height 2D LiDAR person detection by introducing temporal information, without giving up single-frame accuracy.**

Stated this way the problem is measurable on a quantity the field's benchmark does not report — false positives per frame, at a fixed operating point — and it admits a clean negative result: any method that lowers the phantom rate only by detecting fewer people has not solved it. Every table below therefore reports false negatives per frame and the FP:FN ratio alongside, so that a method paying for precision with recall cannot hide.

## Dataset restructuring and experimental design

The FROG dataset comprises six ~30-minute recordings from one venue, of which the published benchmark uses only three: two chosen, in the authors' own words, "around the time of greatest attendance (around noon, maximizing the number of person annotations)" for training and validation, and one for test. The benchmark files are additionally pre-filtered — "scans with empty lists of person annotations are excluded" — so every scored frame contains at least one person, while the three unused recordings are the raw, unfiltered originals and are 34.8–39.2% person-free. That filtering is a defensible choice for a detection benchmark and an explicit trade of environment diversity for annotation density, but it is exactly what makes the dominant error invisible. We therefore reorganised the same six recordings into two partitions plus one additional evaluation split, selectable at load time. **`official`** reproduces the published protocol frame-for-frame (the paper's own `split == 0` half for training, `frog_16-41` for test) and exists so our numbers stay comparable to theirs. **`balanced`** is our own partition, pairing one populated-only recording with one raw recording in each of train, val and test so that all three see crowded, sparse and empty scenes in near-identical proportion (20.6 / 19.8 / 19.1% person-free); it assigns whole recordings rather than interleaved blocks, so no temporal window can bridge a split, and it places `frog_16-41` and `frog_15-53` in test specifically because neither appears in any published baseline's training data. **`transferred`** is not a third dataset but an extra *evaluation* split on the `official`-trained model — its train and validation halves are bit-identical to `official`'s by construction, so one checkpoint serves both — consisting of the 72,533 person-free frames of the three unused recordings. On it, average precision is not merely low but *undefined*: with no true positives, precision is zero at every threshold and recall has no denominator, so the metric there is a false-positive count, which is well-defined precisely because every detection is a false positive. The experiments below evaluate the two published single-frame FROG detectors, LFE-Peaks and LFE-PPN, using their authors' own released ONNX weights and their own operating threshold of 0.3, reporting average precision, false positives per frame, false negatives per frame and their ratio — the same quantities on both splits, so that the only thing differing between the columns is whether the scene contains people.

## Results

Published FROG detectors, authors' own ONNX weights, objectness threshold 0.3, AP at an association distance of 0.5 m.

| | AP<br>populated | FP/frame<br>populated | FN/frame<br>populated | FP:FN<br>populated | AP<br>person-free | FP/frame<br>person-free | FN/frame<br>person-free | frames with<br>a phantom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **LFE-Peaks** | 65.1% *(pub. 65.6%)* | 1.76 | 0.63 | 2.77 | undefined ¹ | 1.39 | 0 by construction ² | **51.0%** |
| **LFE-PPN** | 69.6% *(pub. 69.2%)* | 5.03 | 0.38 | 13.4 | undefined ¹ | 3.17 | 0 by construction ² | **64.0%** |

¹ Average precision is undefined on a person-free split, not low: no true positives means precision is 0 at every threshold and recall has no denominator. A number here would name something the quantity is not.
² A false negative needs something to miss. With no annotated people, `FN / (TP + FN)` has a zero denominator — which is itself informative: the person-free split cannot see the *cost* of any false-positive reduction, so the populated columns must be read beside it.

*Populated: `frog_16-41`, every 5th frame (10,034 frames, 30,737 annotated people). Person-free: all 72,533 such frames of `frog_10-31`, `frog_14-57`, `frog_15-53` — 46.1 minutes at 26.2 Hz, so the FP columns are 36 and 83 phantoms per second respectively. Published figures are the FROG benchmark's `d = 0.5 m` column, not its leading mAP-over-[0.3:0.05:0.5] column. Detections are matched to people one-to-one by Hungarian assignment gated at 0.5 m, so a second detection on an already-matched person is a false positive rather than a second true positive.*

**The LFE-PPN row was re-measured on 2026-09-11 and every figure in it moved.** Its `nms_radius` was recalibrated from the paper's implied 0.8 m to 0.30 m — the value that actually reproduces its published AP at *both* association distances the benchmark reports — and the false-positive tables were never re-run against it. A smaller suppression radius keeps more detections, so LFE-PPN now resolves adjacent people its old radius merged (its miss rate falls) while emitting far more phantoms and duplicates. LFE-Peaks is unaffected: its own recalibration, 0.40 m to 0.30 m, changes its detection count by 3%. `CHANGELOG.md` carries the full correction.

**The two false-positive columns are not quite measuring the same thing, and separating them sharpens the result.** On a populated split a *second* detection on an already-matched person counts as a false positive; on a person-free split no such duplicate can exist. Decomposing the populated column accordingly:

| | true positives | duplicates | **genuine phantoms** | **phantoms, person-free** | agreement |
| --- | --- | --- | --- | --- | --- |
| LFE-Peaks | 2.429 | 0.372 | **1.387** | **1.391** | 0.3% |
| LFE-PPN | 2.689 | 1.609 | **3.416** | **3.174** | 7.6% |

**The genuine phantom rate barely depends on whether anyone is in the room** — to within 0.3% for LFE-Peaks and 7.6% for LFE-PPN, on scenes that share nothing but the sensor. Empty scenes do not provoke the failure; they merely stop concealing it. Both detectors now merge at 0.30 m, the right scale for joining one person's two legs, and neither is at the 0.8 m person-diameter the paper's own sentence implies — which is why both now emit duplicates, LFE-PPN heavily.

**Three further things this table shows.**

1. **AP and false-positive burden rank the two detectors oppositely.** LFE-PPN wins on AP — by 3.6pp in the authors' own figures, 69.2% against 65.6% — while emitting **42% more phantoms per frame** and firing in 64.0% of empty frames against 51.0%. AP rewards the recall it holds at high thresholds and never charges for absolute phantom count, so a ranking taken from the published benchmark is not a ranking for deployment.
2. **Both miss more than a fifth of the people present** at their own operating threshold, so the phantoms are not the price of exhaustive recall — both errors are large at once.
3. **Half to two-thirds of person-free frames contain a phantom**, which at 26.2 Hz is a detector reporting a person who is not there dozens of times per second of empty corridor.

## Model extensions: temporal reasoning on top of a frozen detector

Before proposing an architecture, the cheapest possible use of temporal information: leave the published detectors and their weights untouched, and add a stage *after* them. We built three such stages — a standard tracker, a world-location persistence map, and a background model from raw scan returns — together with one non-temporal control, the detector's own merge radius, and two oracles that bound what any stage of this kind could reach.

**All of them are priced in one currency.** Every stage below trades false positives for false negatives along a monotone curve with no interior optimum, so no single operating point summarises it and average precision alone hides which side paid. We therefore report, beside AP, the **exchange rate**: false positives removed per false negative added, measured on populated frames, where a filter can destroy a person as well as a phantom.

### A standard tracker

We use **SORT** (Bewley et al., *ICIP* 2016) with its track management followed exactly — confirmation requires `min_hits` *consecutive* matches, `hit_streak` resets on any missed frame, coasted tracks are not reported, and the early-frame grace period is preserved — over a constant-velocity Kalman filter. Two deviations, both forced by the setting rather than chosen: association uses Euclidean distance rather than IoU, because these detections are points on the ground plane and have no bounding boxes (circle IoU is a monotone function of centre distance, so it is the same gate relabelled); and track state is compensated for the robot's own motion each frame from odometry, because SORT assumes a static camera while this robot translates 1.88 cm and rotates up to 0.4° per frame. The tracker is fed **every raw frame at the sensor's full rate**; only the scored subset reaches the metric.

| | min_hits | AP | FP/frame | FN/frame | FP:FN | FP/frame<br>person-free | frames with<br>a phantom |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **LFE-Peaks** | *none* | 65.1% | 1.76 | 0.63 | 2.77 | 1.39 | 51.0% |
| | 1 | 65.1% | 1.64 | 0.65 | 2.53 | 1.33 | 50.5% |
| | 3 | 63.6% | 1.43 | 0.72 | 1.98 | 1.21 | 49.1% |
| | 5 | 62.3% | 1.30 | 0.78 | 1.65 | 1.13 | 48.1% |
| | 10 | 59.5% | 1.07 | 0.92 | 1.16 | 1.00 | 45.9% |
| | 20 | 53.7% | 0.83 | 1.16 | 0.71 | 0.84 | 42.1% |
| | 50 | 40.1% | 0.55 | 1.66 | 0.33 | 0.57 | 32.6% |
| **LFE-PPN** | *none* | 69.6% | 5.03 | 0.38 | 13.42 | 3.17 | 64.0% |
| | 1 | 69.5% | 4.92 | 0.38 | 12.98 | 3.09 | 63.6% |
| | 3 | 68.5% | 4.40 | 0.44 | 10.03 | 2.77 | 62.1% |
| | 5 | 67.3% | 3.95 | 0.52 | 7.64 | 2.52 | 60.8% |
| | 10 | 62.6% | 3.03 | 0.75 | 4.03 | 2.05 | 57.8% |
| | 20 | 50.6% | 1.88 | 1.27 | 1.48 | 1.41 | 51.9% |
| | 50 | 22.4% | 0.72 | 2.29 | 0.32 | 0.57 | 34.7% |

*Both detectors at their recalibrated merge radius of 0.30 m (`CHANGELOG.md`, 2026-09-11).*

**There is no operating point to find.** Both curves are monotone: false positives fall and false negatives rise without turning, so `min_hits` is not a parameter with an optimum but an exchange rate — and the rate *worsens* the harder it is pushed. For LFE-Peaks it buys 3.85 false positives per false negative added at `min_hits = 3`, 2.39 at 10, and 1.18 at 50.

**What tracking buys, priced in AP** (phantoms removed is the reduction in person-free false positives per frame, from the rows above):

| min_hits | LFE-Peaks<br>phantoms removed | LFE-Peaks<br>AP cost | LFE-PPN<br>phantoms removed | LFE-PPN<br>AP cost |
| --- | --- | --- | --- | --- |
| 1 | 4.3% | <0.1 pp | 2.5% | 0.1 pp |
| 3 | 12.9% | 1.5 pp | 12.6% | 1.1 pp |
| 5 | 18.7% | 2.8 pp | 20.5% | 2.3 pp |
| 10 | 28.1% | 5.6 pp | 35.3% | 7.0 pp |
| 20 | 39.6% | 11.4 pp | 55.5% | 19.0 pp |
| 50 | 59.0% | 25.0 pp | 82.0% | 47.2 pp |

**Halving the phantom rate costs 19 to 25 points of AP.** And even at `min_hits = 50` — 1.9 seconds of unbroken confirmation, which is also 1.9 seconds of latency before a real person is first reported — **32.6% and 34.7% of person-free frames still contain a phantom**, at a cost of 25.0 and 47.2 points of AP. The phantom rate cannot be driven toward zero by temporal association at any price; it can only be exchanged for detections.

This is the quantitative form of the persistence result. A confirm-then-report rule can only remove a phantom that fails to reappear, and 88.3% of LFE-Peaks' phantoms reappear in the very next frame — so the reduction available cheaply (13% for 1.5 pp) is close to the transient fraction, and everything beyond it is bought by discarding real detections rather than by discriminating between the two classes.

**Two expectations we got wrong, recorded because they constrain the next step.** First, we predicted LFE-PPN would benefit *more* from tracking, having the larger transient fraction (22% against 12%). It does not: up to a third of phantoms removed, the two detectors pay almost exactly the same AP per point of reduction (5.6 pp for 28.1%, 7.0 pp for 35.3% — 0.20 pp per point on both), and LFE-PPN is cheaper only beyond that. **The transient fraction does not predict how much a tracker helps.** *(An earlier draft reported LFE-PPN benefiting less, and proposed phantom density as the mechanism; that was measured at LFE-PPN's pre-recalibration merge radius of 0.8 m and does not survive re-measurement.)* Second, we repeated the whole sweep with a deployment-oriented tracker variant that confirms on total rather than consecutive hits and reports coasted tracks; at matched false-positive rates the two trace the same curve to within about a point of AP. **The result is not an artifact of the tracker's design.**

### A non-temporal control: the merge radius

The detector's own merge radius — the distance below which two centroids are read as one person — trades the same two quantities along the same monotone curve, and it trades them far better. At **matched false-positive reduction** for LFE-Peaks, roughly 0.33 phantoms per frame, SORT at `min_hits = 3` adds 0.087 false negatives per frame while widening the merge radius from 0.30 m to 0.40 m adds **0.015** — an exchange rate of **21.9** against SORT's **3.8**, and **5.8x cheaper** in false negatives. LFE-PPN agrees: the radius removes 0.941 FP/frame for 0.015 FN/frame (~63 per false negative) where SORT removes 0.626 for 0.064 (9.8). Across radii from 0.20 m to 0.50 m recall moves only 1.8 percentage points while duplicate detections fall eighteen-fold.

This does not rescue the single-frame setting — the radius saturates exactly as the tracker does, leaving well over a phantom per frame at any setting, and it has no optimum either — but it does establish that association is not merely insufficient, it is *dominated* at its own job by a post-processing constant. **Whatever the case for temporal reasoning is, it cannot be that a stage after the detector is the cheapest available way to remove a false positive**, and every stage below is held to this price rather than to doing nothing.

### How far any such stage could reach: two oracles

The tracking result bounds what *association* can achieve; it does not bound what temporal reasoning could achieve, and conflating the two would let a weak method stand in for a class of methods. We therefore measure the second directly, with an oracle that is agnostic to mechanism: treating every annotated person as a trajectory, a miss counts as recoverable if that same trajectory was detected both before and after it within a horizon `h`; treating every prediction on a person-free frame as a trajectory, a phantom counts as removable if its longest consecutive trail is shorter than `h`. Sweeping `h` separates two horizons the literature has always reported as one. **Gap-bridging saturates early** — 90% of the available recall gain is reached by 10 s and all of it by 20-30 s, lifting LFE-Peaks from 75.9% to 85.5% recall and LFE-PPN from 83.8% to 90.5% — whereas **trail-rejection is still climbing where the other has finished**, reaching 90% of its available phantom reduction only at 30 s (89.7% of LFE-Peaks' phantoms, 93.9% of LFE-PPN's) and completing at 60 s. The recall half stops where it does for a reason that is a property of the venue rather than of any model: ground-truth person trajectories have a median duration of **4.9 s** and only **3.6%** survive 30 s, so beyond that horizon there is nothing left to bridge. The phantom side inverts under the correct weighting — 4.2% of LFE-Peaks' phantom trajectories reach one second, but those carry **67.7%** of its phantom detections, and the 0.1% reaching thirty seconds still carry 10.3% — so almost every phantom *trajectory* is a single-frame flicker while almost every phantom *detection* belongs to a handful of very long-lived ones, and **the object that must be remembered longest is never a person**. This is not an artifact of the observation window: the person-free run is 305 s long and the longest trail in it is 44.1 s.

**The precision half of that oracle bounds removal, not discrimination, and the difference turned out to be most of the story.** It was measured on person-free frames, where every detection is a phantom and a filter cannot destroy a true positive. We therefore re-measured it where it must discriminate: on all 50,088 populated frames of the test recording, with trails built by nearest-neighbour association over every detection and the filter charged for every person it removes.

| acausal filter, LFE-Peaks | best exchange rate | at |
| --- | --- | --- |
| trail length, no gap tolerated | 2.43 FP per FN | 0.10 s |
| trail length, 1 s gap tolerated | 2.54 | 0.25 s |
| **world-frame displacement** | **4.20** | 0.30 m |

Displacement is the right feature — the separating claim was always that *a person eventually moves and a chair never does*, which is displacement rather than persistence — and at 0.80 m it removes 36.1% of phantoms for 6.9% of true positives. But even with perfect hindsight on the right feature, **the best filter only just clears the tracker and is five times worse than the merge radius.** And the ceiling is **association-limited**: in FROG's crowds (~3.9 people per frame) nearest-neighbour association merges a person's detections into a nearby phantom's trail, which then carries the phantom's low displacement, so a filter aimed at the chair destroys the person. A trail cannot be filtered better than it can be formed.

The recall half is untouched by this correction: it follows annotated identities, so its association is perfect by construction, and its +9.6 and +6.7 points of recall remain the largest measured headroom in this study.

### Remembering where the furniture is

If the phantoms are furniture, the obvious remedy is to remember where furniture is: accumulate evidence per world location over tens of seconds and suppress detections standing where something has persisted. The oracle above had already lowered the ceiling on this; we built it anyway, twice, because a location-keyed map needs no trails and no association and so is not bound by the association limit.

| mechanism | keyed on | best exchange rate,<br>LFE-Peaks | LFE-PPN |
| --- | --- | --- | --- |
| **persistence map** — exponentially forgetting count of detections per ego-compensated world cell | world cell | **0.57** | — |
| **background model** — raw scan returns per world cell, background if hit *rate* and observed *span* both exceed thresholds | world cell | **1.90** | **3.32** |

The persistence map fails because **people revisit locations**. On person-free frames, at 0.50 m cells and a 30 s horizon, it removes 77.5% of phantoms; on populated frames the same setting removes 73.8% of *all* detections and takes recall from 79.3% to 23.2%. It is not discriminating; it is suppressing whatever it has seen before, and museum visitors queue at the same exhibits and walk through the same doorways.

The background model was designed against exactly that failure. It accumulates **returns** rather than detections — a chair reflects a beam whether or not anything detects it — and requires both a high hit *rate* and a long observed *span*, so that each of the three confusable cases is excluded by a different condition: a busy corridor cell has a long span but a low rate, someone standing still for five seconds has a high rate but a short span, and only a chair has both. It still fails:

| min span | min rate | phantoms removed | true positives lost | exchange rate |
| --- | --- | --- | --- | --- |
| 1 s | 0.95 | 45.4% | 45.5% | 0.72 |
| 10 s | 0.95 | 5.3% | 2.0% | **1.90** |
| 30 s | 0.30 | 11.3% | 6.4% | 1.29 |
| 100 s | any | **0.0%** | 0.0% | — |

*LFE-Peaks, all 50,088 populated frames of the test recording. LFE-PPN's best is 3.32, removing 3.5% of phantoms for 2.0% of true positives, and 2.53-2.85 wherever removal is substantial.*

The reason is structural, and the table shows it directly: requiring a location to have been under observation for 30 s admits only **11.3%** of phantoms, and requiring 100 s admits none. **On a robot that drives past its environment rather than watching it, how long something has been at a location measures how long the robot looked, not whether the object is furniture** — and "briefly observed" is precisely what a person is too. This also explains the 89.7% that motivated the attempt: removing everything short-lived is safe only on frames containing nobody to destroy. The failure is one of mobility rather than of physics, and a stationary or slowly patrolling platform, whose observation span is set by the scene instead of by its own path, would change the premise.

### What the extensions establish

| stage after the detector | causal | LFE-Peaks | LFE-PPN |
| --- | --- | --- | --- |
| persistence map over detections | yes | 0.57 | — |
| background model over scan returns | yes | 1.90 | 3.32 |
| trail-length filter | **no** — oracle | 2.54 | — |
| SORT, `min_hits = 3` | yes | 3.8 | 9.8 |
| displacement filter | **no** — oracle | 4.20 | — |
| **merge radius, 0.30 → 0.40 m** | yes | **21.9** | **~63** |

*False positives removed per false negative added, populated frames, best setting of each.*

**Nothing built on top of a frozen detector beats a post-processing constant, and only an acausal oracle beats the tracker.** The ordering is the same on both detectors wherever both were measured.

The failures share two causes, and neither is the horizon. **Association:** the tracker's single match radius cannot be tight enough for FROG's crowds and loose enough for one person's own motion at once, and the best displacement filter is capped by exactly that association. **Keying:** memory indexed by world location, on a platform that moves through its world, records the robot's path rather than the scene's structure. The horizon argument survives both — the separating evidence still lives at tens of seconds, and the recall half of the oracle, which assumes neither a radius nor a location, is still worth +9.6 points — but the negative results constrain what can reach it. **The memory must be keyed by object rather than by place, its association must be learned rather than set by a radius, and its evidence must reach the detector before the threshold rather than after it**: a stage after the threshold can only delete detections or coast tracks, while a fifth of LFE-Peaks' misses (4.4 points of recall) sit in the 0.1-0.3 band of its own probability map, where evidence from memory could lift them. Thirty seconds at FROG's 26.2 Hz is **786 frames**, which is infeasible as a feed-forward window over 720 beams and unremarkable as a recurrent state over a handful of objects — and that arithmetic, rather than any appeal to expressiveness, is our argument for a recurrent architecture. The design that follows from these constraints is [`PROPOSAL.md`](PROPOSAL.md).

## Reproducing every number in this document

Setup, once:

```bash
pip install ./library                     # library + FROG/DROW data + bundled LFE ONNX weights
pip install -r utils/requirements.txt
```

Use `.venv/Scripts/python.exe` directly on Windows; `python3` resolves to a Store stub here (`memory/gotchas.md`). All commands run from `utils/`.

| What it produces | Command |
| --- | --- |
| **Both result tables** in full — AP, FP/frame, FN/frame, FP:FN on the populated split, FP/frame and phantom-frame rate on the person-free one, for every tracker setting | `python tracker_sweep.py --tracker sort --max-age 1 --min-hits 1 2 3 5 10 15 20 30 50` |
| The same sweep with the coasting variant — the tracker-design robustness check | `python tracker_sweep.py --tracker simple --max-age 5` |
| **The phantom / duplicate decomposition** and the **persistence** figures | `python phantom_analysis.py` |
| **Both horizon curves**, the trajectory-duration tables and the saturation points | `python horizon_sweep.py --detector lfe-peaks` (and `--detector lfe-ppn`) |
| **Recall stratified by range**, the sub-threshold miss budget and the close-pair residual | `python range_probe.py` |
| **The scale-vs-capacity answer** and the Cover & Hart bracket | `python repr_probe.py` |
| **The merge-radius exchange rates** against SORT | `python nms_sweep.py --skip fullscan_tcn lfe_ppn --score-by operating-point` |
| **The displacement and trail-length oracles** on populated frames | `python horizon_sweep.py --discrimination` (add `--max-gap 26` for the gap-tolerant row) |
| **The persistence map** | `python persistence_map.py` |
| **The background model** | `python persistence_map.py --background` |
| **AP against the published figures** on the benchmark split | `python evaluate.py --dataset frog --frog-mode official --split test --eval-stride 5 --lfe-peaks --lfe-ppn --no-bench` |
| **False positives on person-free frames**, swept over operating thresholds | `python evaluate.py --dataset frog --frog-mode transferred --split test --lfe-peaks --lfe-ppn --no-bench` |
| **AP at the tighter association distance** the benchmark also reports | add `--eval-r 0.3` to either `evaluate.py` command |
| **The shuffled-history ablation** (temporal ordering is worth 0.5pp) | `python evaluate.py --dataset frog --split test --spacetime-cnn <ckpt> --eval-history-mode shuffle --no-bench` |
| **Dataset composition** for every partition | `python -c "from follow_the_drow.datasets import FROG_Dataset; [FROG_Dataset(split=s, mode=m) for m in ('official','transferred','balanced') for s in ('train','val','test')]"` |

Three things worth knowing before running these.

- **`tracker_sweep.py` caches detections to disk** (`utils/.tracker_cache/`) on first run and replays them through every tracker setting, so the first invocation takes several minutes and later ones are fast. `--no-cache` forces re-detection.
- **`horizon_sweep.py` also caches** (`utils/.horizon_cache/`): it runs the detector once, then sweeps both horizon axes analytically over the cached tracks, so only the first invocation per detector is slow.
- **`range_probe.py`, `repr_probe.py` and `persistence_map.py` cache too** (`utils/.range_cache/`, `utils/.repr_cache/`, `utils/.persist_cache/`). `range_probe.py` reports recall at a **single operating point** (`--thresh`, default 0.3), which is not AP and must never share a column with one.
- **`--eval-stride 5` thins what is *scored*, never what the tracker sees.** A tracker is a deployment-time filter and is always fed every raw frame, in order, with ego-motion compensation.
- **State the association distance with any FROG AP.** The benchmark reports AP at 0.5 m and at 0.3 m *and* an mAP averaged over [0.3 : 0.05 : 0.5]; its leading column is that mAP, not AP at either distance. Everything here is 0.5 m unless stated.

## Status of this draft

Written and measured: the theory, the dataset design, the baseline table, and every model extension — tracker, merge radius, both oracles, persistence map and background model — with a negative result for each.

Not yet written, and not yet measured:

- `balanced` results for a model of ours, which need their own training run to mean anything and are the expensive remaining item. The published detectors have now been measured there — LFE-Peaks **59.8%**, LFE-PPN **63.5%**, against 65.1% and 69.6% on `official` — so the target any temporal architecture of ours must beat is fixed.
- Any result for a temporal *architecture*. The theory predicts short-window models will not fix this and long-horizon ones will; the tracking result supports the first half, and the second half is untested.
- The architecture itself. [`PROPOSAL.md`](PROPOSAL.md) now carries a three-stage design derived from the extensions' failures — Cartesian input, a short normalising window, an object-keyed recurrent memory — settled with the owner component by component and awaiting review, with nothing built or measured (`TODO.md` A43).
- A residual +2.1pp on our LFE-Peaks reproduction was traced to our own merge radius and closed; a residual mislocalisation in our LFE-PPN decoder, visible only at the 0.3 m association distance, remains open (`TODO.md` A32d).
