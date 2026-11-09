# Dataset properties — what the data itself constrains

*keywords:* time span, temporal horizon, receptive field, how long, how fast, people move, stationary people, displacement, frame rate, Hz, annotation density, person density, phantom persistence, architecture constraints, cross-dataset, DROW vs FROG, ceiling, DROW splits, corrupt labels, rated range, dt window, 29.99 m, annotation bursts, every 5th scan, burst period, 31.5 s, supervision per chunk, class mix, wheelchair, walker, 56% not pedestrian, trained as background

**Measured behaviour of the recordings, not their file formats** — the formats are in [`data-model.md`](data-model.md), model scores in [`performance-log.md`](performance-log.md).

This file exists to answer one question before an architecture is designed: **what does the data support?**
Everything here is measured with `utils/motion_analysis.py` and `utils/phantom_analysis.py`, model-free except where a detector is named.
Re-derive with the commands in [`commands.md`](commands.md); the runs are minutes, not hours.

## Bottom line: the headroom, both datasets

**How much accuracy is left on the table for a perfect temporal model, given a
static detector's output?** Everything below is an *upper bound* from
`utils/temporal_oracle.py`; the sections that follow derive it.

| | FROG / LFE-Peaks | FROG / LFE-PPN | DROW / DROW3 |
| --- | --- | --- | --- |
| static detector, as published | 65.1% AP | 69.3% AP | 66.5% AP ¹ |
| **recall now** | 75.9% | 83.8% | 63.4% |
| **recall ceiling** (interpolation bound) | **82.8%** | **88.6%** | **69.2%** |
| **headroom** | **+6.9 pp** | **+4.8 pp** | **+5.8 pp** |
| unreachable — never detected at all | 3.8% | 3.0% | 3.5% |
| false positives now, per frame ² | 1.53 | 4.15 | 0.21 |
| after a 1 s consecutive-trail filter | 1.05 (**-32%**) | 2.20 (**-47%**) | 0.11 (**-48%**) |

¹ DROW3 measures 66.5% here against the **71.7%** its own weights were published
at — an undershoot of the ordinary, innocent kind (`performance-log.md`). The
"+4.6 pp harness overshoot" this footnote described until 2026-11-04 was a
mis-citation: 61.9% matches no row of any primary source.

² On the **longest unbroken person-free run**, the subset the oracle needs to build
trajectories at all. Across *all* 72,533 person-free frames the same detectors measure
**1.39** and **3.17** — the oracle's runs are 10% (Peaks) and 31% (PPN) denser in phantoms,
because a long unbroken stretch with nobody in it is a stretch where the robot is parked
somewhere cluttered. Both figures are correct; they are not interchangeable, and an earlier
draft of this file quoted them as if they were.

**The recall headroom is ~5-7 pp, and it is remarkably stable**: three
detector/dataset pairs spanning two venues, two sensors and two frame rates all
land in the same narrow band, and all three leave 3.0-3.8% of people
categorically unreachable (never detected on any frame of their trajectory, so
no trajectory exists to reason over). That consistency is the strongest single
result in this file: it suggests ~5-7 pp is a property of *trajectory reasoning
applied to a leg-detector's output*, not of any one venue or model.

**The precision headroom is not stable and does not transfer.** LFE-PPN's
phantoms are far more removable than LFE-Peaks' (-47% against -32% at the same
filter), consistent with their different frame-to-frame persistence, and DROW3
produces 7x fewer phantoms than LFE-Peaks to begin with. Any claim of the form
"temporal filtering removes N% of false positives" is a claim about one
detector in one venue.

**Three things this bound is not.**

- **Not a promise.** It is what a *perfect* trajectory reasoner could reach given
  the static detector's existing output. A real model gets some fraction of it,
  and the only measured attempt so far — SORT — got a *negative* share of the
  recall half (it trades recall away, see `docs/PAPER.md`).
- **Not a bound on accuracy in general.** It bounds what *temporal* reasoning
  adds on top of a fixed static detector. A better static detector moves every
  row, and given that stationary people are the ones being missed (below), that
  may be the larger prize.
- **Not comparable across the columns.** The three detectors sit at different
  operating points — DROW3 is far more conservative (36.6% miss, 0.21 FP/frame)
  than LFE-PPN (16.2% miss, 4.15 FP/frame) — so the venue, the model and the
  threshold are all varying at once.

**One caveat specific to DROW, and it matters for the precision row.** DROW
annotates every 5th scan, and its longest unbroken run of person-free *annotated*
frames is 16-20 frames, i.e. 8-10 s. **No phantom trail longer than the run can
be observed**, so DROW's "100% removable at a 9.5 s filter" is an artifact of the
observation window, not a finding. FROG, which annotates every scan across
unbroken 60k-frame recordings, is the only one of the two that can measure a
long trail at all — and there 47.2% of LFE-Peaks' phantoms sit on trails of 5 s
or more.

## The headline constraint: how long is the useful temporal horizon?

How far an annotated person has actually moved, in the **world** frame (odometry-compensated, so the robot's own motion is not counted as theirs), after a given elapsed time.
**A person who has moved less than ~0.5 m — less than their own width — is indistinguishable from furniture to any motion-based method**, whatever its architecture.

| elapsed | FROG: moved < 0.5 m | FROG, on the 26.2 Hz clock | DROW: moved < 0.5 m |
| --- | --- | --- | --- |
| 1 s | **52.4%** | 67.3% | 62.8% |
| 2 s | **41.4%** | 46.6% | 37.1% |
| 5 s | **32.7%** | 36.0% | 28.8% |
| 10 s | **24.0%** | 29.5% | 19.5% |
| 30 s | **13.3%** | 16.3% | *(unreliable, see below)* |

**The FROG column was restated on 2026-10-10** and the middle column is what this file carried before.
Every row was measured over too few frames: the horizon is specified in seconds and converted with the frame rate, and the rate was the median stamped interval, 26.2 Hz, where FROG's sensor delivers a scan every 25 ms — so a "1 s horizon" spanned 0.65 s of real motion (`interpreting-evaluation.md`).
The restated figures are *lower* at every horizon, because people are now given the time the row claims.
**DROW's column is not restated here**: its rate moved too, from the median's 10.0 Hz to 12.70, so its figures span 1.27x too *long* and will move upward; it is re-measured with the rest of the DROW work, because nothing on FROG depends on it.

**Three things this settles.**

**FROG's venue is 120-250 m across, and "room coordinates" means a museum floor, not a room** (measured 2026-10-27
from the odometry sidecars, `library/takheliped/include/FROG-data/*_odom.npz`). Per recording the robot covers
**670-1,260 m of path**, with pose spans of 124-165 m in x and 112-250 m in y, and a 95th-percentile radius of ~102 m
from each recording's own centre.

| recording | frames | x span | y span | path | max radius |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10-31 | 15,963 | 164.7 | 143.8 | 699.4 | 110.3 |
| 11-36 | 17,746 | 147.0 | 112.1 | 742.9 | 98.9 |
| 12-43 | 18,980 | 141.7 | 124.4 | 746.2 | 98.7 |
| 14-57 | 17,428 | 145.1 | 130.3 | 787.7 | 91.4 |
| 15-53 | 15,091 | 149.6 | 134.4 | 669.2 | 87.5 |
| 16-41 | 17,623 | 123.9 | 134.7 | 803.1 | 80.5 |
| train+val (11-36 + 12-43) | 120,396 | 144.1 | 250.2 | 1,260.6 | 154.0 |

**Why it matters, beyond description.** It decides what a spatial memory can be: a dense
person-scale grid over this extent is **~46,000 cells**, 180x the recall attention's O(P*K) cost, while tiling it with
the slots' 256 cells gives **~11 x 17 m** cells against a 0.5 m match radius. Neither is the experiment, which is why
the binding arm is run as **sparse grid slots** instead. It also qualifies every "room frame" claim in `docs/RESEARCH.md`
section 5.3: the property being exploited is that a static object keeps its world coordinate over a 150 m traversal,
which is a stronger statement than it sounds.

1. **Most of the available motion appears in the first ~5 seconds**, in both venues. The curve falls steeply to 5 s and then flattens. Horizon past that keeps paying on FROG but with sharply diminishing returns. The restatement sharpened this rather than changing it: the fall from 1 s to 5 s is now 52.4 -> 32.7% where it read 67.3 -> 36.0%.
2. **There is a hard ceiling on every motion-based method**, and it is not small: at a 10-second horizon, **about a quarter** of people on FROG have not moved a body width (24.0%, against 19.5% on DROW's un-restated clock). No architecture crosses this — it is a property of people, not of models.
3. **It is not a museum artifact.** This was the main risk to the whole premise (FROG is a science museum, where people stand and look at things). DROW is a care facility, and people there were ~8-10 pp more mobile at every horizon on the clocks both columns used at the time. **That margin is now unquantified**, since FROG's column fell by 4-15 pp and DROW's will rise; the qualitative claim — the problem survives the change of venue — rests on both venues showing the same steep-then-flat shape, which neither restatement changes.

**What a temporal window of a given size buys**, at each dataset's own rate:

| window | FROG frames (26.2 Hz; really 40 Hz) | DROW frames (10 Hz; really 12.7 Hz) | people it could possibly help with |
| --- | --- | --- | --- |
| 2 s | 52 | 20 | ~53% (FROG) / ~63% (DROW) |
| 5 s | 131 | 50 | ~64% / ~71% |
| 10 s | 262 | 100 | ~70% / ~80% |
| 30 s | 786 | 300 | ~84% / — |

**This is the argument for and against a long horizon at once.** The ceiling genuinely rises with horizon — so the horizon matters — but a feed-forward window of 262 frames x 720 beams is not a window, it is a sequence, and reaching 10 s that way is impractical. That is the case for a *recurrent* state (O(1) per frame, unbounded horizon) rather than a wider convolution, and the reason `rejected-ideas.md`'s recurrence entry is due for the re-examination its own reopening clause describes.

**Caveats, because they are load-bearing:**

- **Gate sensitivity is checked and the conclusion holds.** Trajectory association needs a gate wide enough to absorb annotation jitter; too tight and tracks shatter, which biases the statistic *toward* stationarity. Across gates of 0.4 / 0.6 / 1.0 m the 10 s figure moved 30.2 / 29.5 / 28.5% — stable, and a looser gate moves it conservatively. Only the 30 s row is materially gate-sensitive (18.7 / 16.3 / 14.5%), so quote it as "roughly an eighth" and not more precisely. *(The gate sweep was run on the 26.2 Hz clock; the restatement shifts each row down without changing the spread between gates, since the gate is in metres.)*
- **DROW cannot support a long-horizon claim.** 123 trajectories and 1,976 annotations, against FROG's 1,267 and 250,751. Its 30 s row came out *non-monotone* (median 0.94 m at 30 s against 2.44 m at 10 s) on 214 pairs — a small-sample artifact, almost certainly a couple of stationary staff. It is omitted above rather than reported. DROW supports the 1-10 s comparison and nothing longer.
- **Both are one venue each.** The agreement between them is evidence the effect is not venue-specific; it is not evidence that it is universal.

## The sharper question: how much of the miss rate can motion actually reach?

The horizon table above says how many people *move*. It does not say how many of the people a static detector **fails on** move — and those are not the same set, because **misses and stationarity turn out to be correlated, in the unhelpful direction.**

Ground-truth trajectories cross-tabulated against whether the detector found that person (Hungarian assignment at 0.5 m, the AP metric's own gate), 202,955 person-observations on `frog_16-41`:

| horizon | miss \| **stationary** | miss \| **moving** | addressable margin | recall ceiling |
| --- | --- | --- | --- | --- |
| **LFE-Peaks** | | | | |
| 1 s | 22.5% | 16.2% | 5.3% | 84.9% |
| 2 s | 23.2% | 14.8% | 7.8% | 89.1% |
| 5 s | 23.9% | 13.8% | 8.8% | 91.4% |
| 10 s | 27.0% | 15.2% | 10.6% | 91.9% |
| **LFE-PPN** | | | | |
| 1 s | 15.4% | 8.3% | 2.7% | 89.6% |
| 2 s | 16.5% | 8.0% | 4.3% | 92.3% |
| 5 s | 17.3% | 7.6% | 4.9% | 93.8% |
| 10 s | 18.9% | 8.7% | 6.1% | 94.3% |

*addressable* = share of all person-observations that are both missed **and** moving — the margin a perfect motion-based method could buy.
*ceiling* = best recall any motion-based method could reach, i.e. everything except the missed-and-stationary.

**Stationary people are 1.6-2x more likely to be missed**, consistently, across two independently-built detectors. The mechanism is not mysterious: a standing person holds their legs together and presents one blob, while a walking person shows the separated two-leg signature these detectors are built around. **The failure mode of static detection is precisely the case motion cannot address.**

**So the recall margin is 3-11 percentage points, not the 12-21% miss rate.** It does grow with horizon — 5.3 -> 10.6pp for LFE-Peaks, 2.7 -> 6.1pp for LFE-PPN between 1 s and 10 s — which is another argument for a long horizon, but it starts from a much lower base than the raw miss rate suggests.

**The distinction that keeps this from being fatal: temporal integration is not motion.** Everything above bounds methods that discriminate *by displacement*. A stationary person still produces a weak but **consistent** return, and averaging across frames raises signal-to-noise with zero motion involved. This project has already measured that effect at **3.6pp** in an architecture that provably ignores temporal ordering (`CHANGELOG.md`, the history ablations: 4.1pp total temporal contribution, only 0.5pp of it ordering). A tracker cannot do this — it sees only the detector's output, never the returns, even when it is fed every candidate down to 0.01 as this repository's SORT replay is (`../docs/RESEARCH.md` §7.4.1) — so it is headroom that belongs specifically to a temporal *architecture*, and it is not bounded by the table above.

**Caveat:** the 30 s row inverts the pattern in both detectors (`miss | still` falls to 10.3% / 4.2%, `miss | moving` rises). Treat it as a selection artifact and do not quote it: a person tracked continuously for 30 s is by construction well-observed — close, unoccluded, never leaving the field of view — so the long-horizon subset is not a random sample of people.

## The saturation bound: what a *perfect* temporal model could buy

The measurement above bounds methods that discriminate by displacement. This one bounds
**anything that reasons over trajectories at all**, which is the broader and more useful
question, and it is a standard shape: an oracle/upper-bound analysis in the vocabulary of
MOT evaluation — Mostly-Tracked / Partially-Tracked / Mostly-Lost (Li, Huang & Nevatia,
*CVPR* 2009) and track-lifetime filtering.

**The framing** (owner's, 2026-09-10). Every annotated person is a trajectory; a perfect
temporal model follows each one, so every point on a trajectory the static detector *missed*
is potentially recoverable, and the only irrecoverable people are those never detected at
all. Symmetrically, every prediction is a trajectory; a perfect model is not fooled by
something that appears from nowhere and vanishes, so every *short* prediction trajectory is
potentially removable, and the only surviving phantoms are those producing long consecutive
trails.

Produced by `utils/temporal_oracle.py`.

### Recall side

| | current recall | **interpolation bound** | gain | never detected at all |
| --- | --- | --- | --- | --- |
| LFE-Peaks | 75.9% | **82.8%** | +6.9 pp | 3.8% |
| LFE-PPN | 83.8% | **88.6%** | +4.8 pp | 3.0% |

*Interpolation bound*: a miss counts as recoverable only if the same trajectory was detected
both **before and after** it, within 2 s. The looser *extrapolation* bound — any point on a
trajectory detected at least once, including beyond its last detection — reaches 96-97%, but
no real model extends a track into the unknown that reliably, so it is recorded and not
used.

Trajectory coverage, which is where the asymmetry lives:

| | Mostly Tracked | Partially Tracked | Mostly Lost | ML as share of observations |
| --- | --- | --- | --- | --- |
| LFE-Peaks | 54.2% | 25.1% | 20.7% | **8.0%** |
| LFE-PPN | 66.8% | 19.3% | 13.9% | **5.5%** |

**A fifth of trajectories are Mostly Lost but they carry only 8% of the observations** — the
people a static detector fails on wholesale are mostly people who were briefly visible.

### The same bound above our own detector

Produced by `utils/takheliped_oracle.py` (`checkpoints_three_horizon/oracle_ours.json`) on the FROG **`official`** test split, so it is **not comparable with the LFE rows above**, which `utils/temporal_oracle.py` measures on `balanced`.

| | current coverage | **interpolation bound** (2 s) | gain | never detected at all |
| --- | --- | --- | --- | --- |
| three-horizon stage 2 | 86.1% | **92.6%** | +6.5 pp | 0.8% |

The headroom `docs/PROPOSAL.md` §9 asks about is **+6.5 pp above our own candidates, not the +9.6 pp measured above LFE-Peaks**.
Our stage 2 misses far fewer people outright — 0.8% never detected against LFE-Peaks' 3.8% — so less is left for memory to recover.

**Stage 3 takes 17.5% of that ceiling, and pays more than it takes.**
It recovered 1,769 of the 10,080 recoverable misses, against 6,076 people stage 2 had found that it then dropped: net −3,963, coverage 86.1% → 83.5%.
The memory's +3.2 points of AP come from precision, not from recovered recall.

| bridge | recoverable | ceiling | stage 3 takes |
| --- | --- | --- | --- |
| 0.5 s | 6,310 | 90.2% | 24.2% |
| 2 s | 10,080 | 92.6% | 17.5% |
| 10 s | 13,721 | 95.0% | 13.2% |
| 60 s | 13,721 | 95.0% | 13.2% |

**Nothing is available beyond 10 s**: the 10 s and 60 s rows are identical, which matches the 90%-within-10 s finding measured above LFE-Peaks.

Two caveats, and a correction.
The figures above run ~1.2 pp below the `recall` column of the result tables (86.1 against 87.3 for stage 2, 83.5 against 84.7 for stage 3).
That was first recorded as a metric difference — the result tables sweeping a precision-recall curve — which is **wrong**.
`utils/memory_diagnosis.py` scores the same per-frame gated Hungarian coverage without linking trajectories and gets 134,196 / 153,655 = **87.3%** for stage 2 and 130,131 / 153,655 = **84.7%** for stage 3, reproducing the reported recall exactly.
The shortfall is the trajectory round-trip in this table: coverage is looked up by rounded world position, and 2,254 person-observations (1.5%) collided with another person in the same frame and were dropped or misattributed.
**Those 2,254 are not people** (found 2026-10-05): they are FROG's duplicate labels, circles that repeat another circle of the same frame in all six columns (`data-model.md`).
A lookup by position cannot tell a copy from its original, so the round-trip loses exactly them.
So prefer the diagnostic's counts for anything per-person; this table's ceilings and fractions are trajectory-level and unaffected in their ordering, and its lost/gained counts (6,076 / 2,113) agree closely with the diagnostic's independent 6,119 / 2,054.
The remaining caveat stands: this is a single threshold while AP integrates over all of them, so stage 3 losing recall at 0.3 does not contradict its higher AP.

### Precision side

Phantom trajectories on person-free frames, keyed by **longest consecutive run** (which is
what "a long consecutive trail" means, and the only reading commensurable with a
confirm-then-report rule):

| | current FP/frame | after a 1 s trail filter | after a 5 s trail filter | irreducible (trail >= 5 s) |
| --- | --- | --- | --- | --- |
| LFE-Peaks | 1.53 | 1.05 (-31.9%) | 0.72 (-52.8%) | **47.2%** |
| LFE-PPN | 4.15 | 2.20 (-47.1%) | 1.32 (-68.1%) | **31.9%** |

Both rows are on the longest unbroken run (footnote ² above), which is the only subset a
trail-length filter can be defined on.

LFE-PPN's phantoms are the more removable of the two, consistent with its lower
frame-to-frame persistence (77.6% against 88.3%) — but it starts from 2.7x as many, so it
ends worse in absolute terms whatever the filter.

### The bound, and why it is not what a tracker achieved

Putting both halves together, a perfect trajectory-reasoning system on FROG could reach
roughly **82.8% recall at 0.72 phantoms/frame** (LFE-Peaks), from 75.9% at 1.53.

Contrast that with the measured *causal* tracker (`docs/PAPER.md`): SORT at `min_hits = 10`
removed 28.1% of phantoms **and gave up 5.6 pp of AP**. **The oracle improves both at once;
the tracker can only trade one for the other.** That difference — not the raw numbers — is
the case for a temporal architecture over a tracking cascade.

**Three caveats, all load-bearing:**

- **The two halves are measured on different frames** (recall on populated `balanced` test,
  precision on person-free `transferred`), so the combined figure is a bound assembled from
  two measurements, not one observed operating point.
- **This is not a bound on SORT** and an earlier draft wrongly said so. SORT additionally
  suppresses the first `min_hits - 1` frames of every surviving track, so at the same
  threshold it removes *more* phantoms while delaying every real person. The filters are not
  nested.
- **`--max-gap` must stay 0.** At 0 a trajectory breaks on any miss, so its length is its
  longest consecutive run. At 5 the same phantoms merge into 3x fewer, 3x longer
  trajectories and the filter looks far weaker than it is — an earlier run of this analysis
  made exactly that mistake and reported 8.0% where the answer is 22.9%.

## Horizon, asked of the oracle instead of displacement

The horizon table above answers *"how far has a person moved after t seconds"*, which bounds
methods discriminating **by displacement** -- its own stated scope. A trajectory reasoner is not
such a method: a stationary person still emits a weak-but-consistent return that integrates. So
the horizon question is better put to the oracle, which is agnostic to the mechanism (owner's
framing, 2026-09-11).

Produced by `utils/horizon_sweep.py`: one detector pass, then both axes swept analytically over
the cached tracks.

The oracle splits horizon into **three** axes; the displacement table measured only something
adjacent to the first.

| axis | question | what a model must do |
| --- | --- | --- |
| gap-bridging | how long a detection dropout must be bridgeable | *hold onto* a person temporarily lost |
| trail-length | how long a phantom must be watched before it is rejectable | *accumulate evidence against* a persistent false positive |
| trajectory duration | how long the thing reasoned about exists at all | nothing -- it caps the other two |

### The two horizons, and the ~3x gap between them

Horizon at which each half reaches 90% of its own total available benefit:

| | recall (gap-bridging) | precision (trail-length) | ratio |
| --- | --- | --- | --- |
| LFE-Peaks | **10 s** (9.1 of 9.6 pp) | **30 s** (89.7% removed) | 3x |
| LFE-PPN | **10 s** (6.4 of 6.8 pp) | **30 s** (93.9% removed) | 3x |

Full saturation: recall at 20-30 s on both detectors, precision not until 60 s.

**It is a ~3x gap, not an order of magnitude.** An earlier reading quoted 5 s against 60 s, which
mixed criteria -- 89% of the recall benefit against 100% of the precision benefit. Held to one
criterion the two horizons sit much closer together.

**And that is a simplification, not a complication.** A single ~30 s memory captures **100% of the
recall benefit and ~90% of the precision benefit**, on both detectors. Two pathways at different
time constants are *not* required by this data; proposing them on these numbers would be
over-design.

Full curves, both detectors:

| horizon | Peaks recall ceiling | PPN recall ceiling | Peaks phantoms removed | PPN phantoms removed |
| --- | --- | --- | --- | --- |
| 1 s | 81.6% | 87.5% | 31.9% | 47.1% |
| 2 s | 82.8% | 88.6% | 40.0% | 56.1% |
| 5 s | 84.4% | 89.6% | 52.8% | 68.1% |
| 10 s | 85.0% | 90.1% | 69.8% | 79.2% |
| 30 s | 85.5% | 90.5% | 89.7% | 93.9% |
| 60 s | 85.5% | 90.5% | 100% | 100% |

### Trajectory duration: the cap that makes the recall horizon stop where it does

| | median | p90 | p95 | max |
| --- | --- | --- | --- | --- |
| person trajectories | **4.9 s** | 20.4 s | 26.1 s | 67.7 s |
| phantom trails, LFE-Peaks | 0.0 s | 0.3 s | 0.8 s | 36.7 s |
| phantom trails, LFE-PPN | 0.0 s | 0.3 s | 0.5 s | 44.1 s |

1,087 ground-truth trajectories over 221,802 person-observations. The person row is ground truth,
so it is identical across detectors -- which is also a useful determinism check on the pipeline.

**The recall horizon saturates near 30 s because only 3.6% of people are still present past it.**
There is nothing left to bridge. This is a hard cap of the same kind as the 0.5 m displacement
ceiling: a property of the venue, not of any model.

**The phantom rows look like the opposite story until they are weighted by detections rather than
trajectories**, and then they invert:

| | trails >= 1 s | >= 5 s | >= 10 s | >= 30 s |
| --- | --- | --- | --- | --- |
| share of phantom *trajectories* (Peaks) | 4.2% | 1.2% | 0.5% | 0.1% |
| share of phantom *detections* (Peaks) | **67.7%** | **47.2%** | **30.2%** | **10.3%** |
| share of phantom *detections* (PPN) | **52.2%** | **31.9%** | **20.8%** | **6.1%** |

Almost every phantom *trajectory* is a single-frame flicker; almost every phantom *detection*
comes from a handful of very long-lived ones. **The thing that must be remembered longest is not a
person.**

LFE-PPN produces 4x as many phantom trajectories as LFE-Peaks (12,919 against 3,121) but a
*smaller* share of its detections on long trails -- more phantoms, flickering harder, consistent
with its lower frame-to-frame persistence (77.6% against 88.3%).

### What this constrains architecturally

**30 s at 26.2 Hz is 786 frames.** That is infeasible as a feed-forward window over 720 beams and
unremarkable as a recurrent state -- which is the recurrence argument with a number finally
attached to it, and it supersedes the vaguer "tens of seconds" phrasing used before.

**The precision half may not need sequence modelling at all.** Its phantoms are static (88.3% /
77.6% next-frame persistence at the same location) and long-lived. A spatially-indexed persistence
map -- classical background subtraction -- addresses that directly, without a 30 s sequence state.
Worth testing before any recurrent design is justified *on the precision half*; the recall half is
per-object gap-bridging and a map does not help it.

**Caveats, all load-bearing:**

- **The censoring is not binding here**, unlike DROW's. The person-free observation run is 305 s
  and the longest observed trail is 44.1 s, so the maximum is real rather than a window artifact.
  DROW cannot support this measurement at all (8-10 s of consecutive person-free annotation).
- **`max_gap = 0`, so a trail breaks on any single missed frame** -- these are strictly consecutive
  runs. A location-indexed model would see *longer* persistence, so 60 s is an upper bound for a
  consecutive-run rule and the background-map route needs less.
- **Length-biased sampling on the recall side.** The median person trajectory is 4.9 s, but 89.9%
  of observations sit on trajectories of 5 s or more and the observation-weighted mean duration is
  18.4 s. Both numbers are needed: the median says how long a typical person stays, the weighted
  mean says how long the person currently being looked at will stay.
- **The two halves are still measured on different frames** (recall on populated `balanced` test,
  precision on person-free `transferred`), as everywhere else in this file.
  **And that limits the precision half more than it first appears: it is a bound on *removal*,
  not on *discrimination*.** The trail-length filter never had to separate a phantom from a
  person, because the split contains no people. The persistence map measured what happens when a real
  mechanism must: a world-cell persistence map reaches 77.5% phantom removal on person-free
  frames and removes 73.8% of *all* detections on populated ones, taking recall from 79.3% to
  23.2%. **Do not quote 89.7% as an achievable target.** Measured properly on populated frames
  (`horizon_sweep.py --discrimination`), the best acausal trail-length filter reaches an
  exchange rate of **2.5 false positives per false negative**, and keyed on *displacement* --
  the feature `docs/PAPER.md` actually argues for -- **4.20**, against SORT's 3.8 and the merge
  radius's 21.9. The removal figure had no cost column and the cost column is most of the
  story.

## Per-dataset shape, and why numbers do not carry across

| | FROG | DROW |
| --- | --- | --- |
| venue | science museum, one site | care facility, one site |
| beams / FoV | 720 / 180° / 0.25° | 450 / 225° / 0.5° |
| sensor height | knee (~35 cm) | knee |
| **rate** | **40.0 Hz** (corrected 2026-09-17: the old 26.2 Hz was the median of bunched timestamps; every FROG duration in seconds on this page derived from 26.2 Hz is 1.53x too long) | **12.7 Hz** (per-file mean; the median interval says 10 Hz only because timestamps are rounded to 0.05 s, `noise-structure.md`) |
| annotated scans | **every one** | **5.0%** -- in bursts of 20 at every 5th scan, 24 s apart (below) |
| **people per annotated frame** | **3.06** (test; 3.13-4.08 across recordings) | **0.81** |
| classes | pedestrian only | pedestrian / wheelchair / walker |
| odometry, per frame | 1.88 cm, up to 0.4° (p95) | 0.32 cm |
| session structure | test fragmented into 47 sessions, median ~26 s; the three raw extras are unbroken 60-70k-frame recordings | 5 sessions, ~18 min each |

**Four consequences for any architecture meant to run on both.**

1. **Parameterise the temporal window in seconds, never in frames.** A `T=5, dtime=10` window spans **1.25 s on FROG and 3.9 s on DROW** — the same hyperparameters are two different experiments. Every cross-dataset comparison in this repo predating 2026-09-10 has this confound.
2. **Person density differs 3.8x**, so AP is not comparable across the two: precision is a base-rate-sensitive quantity and FROG hands a detector nearly four times as many chances to be right per frame.
3. **Annotation density differs 20x.** DROW's 2,428 labelled frames are the binding limit on any supervised temporal model trained there — and the reason the motion table above is thin on the DROW side. Densifying it by *interpolating between real annotations* (never by pseudo-labelling with a detector, whose 1.4-1.9 phantoms/frame would be taught as positives) is tracked as a research item.
4. **Beam count and angular resolution differ**, so a beam-indexed architecture is not transferable without resampling; beam index does not mean the same physical angle in the two.

### More than half of what DROW annotates is not a pedestrian, and we train all of it as background (2026-11-05)

Counted from the raw `.wp` / `.wc` / `.wa` files. This page had the class *list* and never the class *mix*:

| | pedestrian | wheelchair | walker | **not pedestrian** |
| --- | --- | --- | --- | --- |
| train | 23,910 | 14,455 | 2,047 | **40.8%** |
| val | 3,102 | 5,678 | 219 | **65.5%** |
| test | **1,976** | **1,970** | **581** | **56.4%** |

**On test there are as many wheelchairs as pedestrians.** A person in a wheelchair is a person, and at knee
height the return is person-shaped; a walker is a person standing behind a metal frame. The trainer reads
`ds.det_wp` only, so each of those 2,551 test objects is a knee-height person-like return that the loss teaches
as **background**, and that the metric scores as a **false positive** if the network finds it anyway.

**Two separate costs, and only one of them is bounded.** The scoring cost is bounded: 2,551 objects over 2,428
test frames is **1.05 per frame**, so at most that much of a measured false-positive rate is a correct detection
counted as an error -- up to 14% of the candidates' 7.699 and **43% of the memory's 2.417**. The training cost is
not bounded by anything: 40.8% of the annotated objects in the train split carry a background target, which is
contradictory supervision on the exact shape the network exists to recognise.

**It is a choice nobody examined, not a bug.** `DatasetSpec` records `classes=("pedestrian", "wheelchair",
"walker")` and the parameter table prints all three, while the targets have only ever been the first. The
published detectors' own architecture carries **four** class logits (`architectures.py`: `n_cls = 1 if
pedestrian_only else 4`) and the weights this project downloads are the one-class inference build, so what their
*training* did with wheelchairs is a question for the provenance audit rather than something to assume.

### DROW annotates in bursts of 20, not uniformly, and the gaps are longer than the memory's fade (2026-11-05)

The cell above used to read *"5.0% -- every 5th scan, 0.39 s apart"*, and those two halves contradict each
other: every 5th scan is 20%, not 5%. Measured from the raw `.wp` annotation ids, both are describing something
real, and neither describes it correctly.

| | train | val | test |
| --- | --- | --- | --- |
| annotations | 17,665 | 3,919 | 2,428 |
| **bursts** | **890** | **199** | **122** |
| annotations per burst, median | **20.0** | 20.0 | 20.0 |
| gap *inside* a burst | **5 scans** (95.4% of all gaps) | 5 | 5 |
| burst span | **95 scans = 7.5 s** | 95 | 95 |
| **burst to burst** | **400 scans = 31.5 s** | 400 | 400 |

So the protocol is: **20 annotations at every 5th scan, spanning 7.5 s, then about 24 s of nothing, repeating** --
identical in all three splits, which makes it the annotators' procedure rather than an artefact. "Every 5th
scan" is right *inside* a burst; "5.0%" is right *globally*; and the global share is 5.18% because the 24 s gaps
take three quarters of every recording.

**Two consequences, and both change a decision.**

* **A 10 s chunk holds at most 20 annotated frames and about 6 on average.** 10 s is 127 frames, which is 0.32
  of a burst period, so a window placed at random carries `127 x 0.0518 ~ 6.6` supervised frames and one placed
  on a burst carries 20 -- against FROG's ~400, every frame of which is annotated. The deficit per rollout is
  **20x at best and ~60x typical**, not the 16x computed from a uniform every-5th assumption.
* **The 24 s unannotated gap is longer than the slots' 10 s fade time.** A slot spawned in one burst is retired
  before the next burst starts, so **no gradient ever crosses a gap**: on DROW every rollout is effectively one
  isolated 7.5 s observation, whatever the chunk length. That is a structural reason DROW cannot reward a
  long-horizon memory, separate from having less data, and it is measured rather than argued.

### DROW per split, measured through the trainer's own loader (2026-11-05)

Measured while making the trainer dataset-agnostic, which is the first time DROW was read through the path that
trains on it. Scan and annotation counts reproduce this page's table exactly; the rest is new.

| | train | val | test |
| --- | --- | --- | --- |
| scans | 341,147 | 74,745 | 48,131 |
| annotated | 17,665 | 3,919 | 2,428 |
| labels | 23,906 ¹ | 3,102 | 1,976 |
| **people per annotated frame** | **1.35** | 0.79 | **0.81** |
| furthest label | 19.09 m | 17.73 m | 14.41 m |
| labels at or past 10 m | 2,809 (11.8%) | 197 (6.4%) | 159 (8.0%) |

¹ After four corrupt annotations are dropped: 316, 352, 498 and 716 m, all in train, in a care facility with a
30 m SICK S300. They are readings of the annotation tool, not distant people, and the trainer removes anything
past the **sensor's** rated range -- not past the run's `--max-range-m`, which is a different question, since
the 2,809 real people past 10 m are precisely what the 29.99 m limit exists to keep.

**The 0.81 people per annotated frame on this page is a test figure**, as FROG's 3.06 is. DROW's train split is
**1.35**, so the density ratio against FROG is 3.8x on test and 2.3x on train. Quote the one that matches the
split being discussed.

**Two consequences of the owner's 29.99 m range limit, both measured.**

* **It makes DROW's missing-return marker self-cancelling.** DROW holds no non-finite and no non-positive
  reading in any split; a missing return is a finite value at the sensor limit, 29.96-29.99 m. At a 29.99 m
  limit `sanitize_ranges` maps the 29.99 marker onto "nothing within range" and `voting_beams` lets no such
  beam cast a vote, so no `missing_return_m` rewrite is owed the way FROG's `+inf`/61.0 needs one.
* **It puts three times FROG's scale into stage 2 at initialisation.** `beam_features` emits metres and
  `ConvNeXtBlock` carries its input forward on an un-normalised residual path. On 256 real annotated frames at
  the adopted widths and a fixed seed, the initial logit spread rises from sd 0.167 (FROG at 10 m) to 0.301,
  and the extreme beam from 1.07 to 3.78, against a saturation line near 4 where `sigmoid'` falls under 0.02.
  Bounded rather than fatal, and the first thing to look at if a DROW stage 2 plateaus early.

**The time-step averaging window stays in frames, and this is the measurement that says so.** Stage 3's
running mean of the stamped intervals is 32 intervals long because FROG stamps three scans to a timestamp.
DROW's defect is different -- stamps are granular to 0.05 s against a 0.0787 s period -- but the window that
serves it is the same: across twelve real files the error against each file's own mean period is **0.5-0.9% at
32 intervals**, 1.7% at 10, 2.1% at 8, where a *single* interval costs **36%**. Restating the window in
seconds is what would hurt, not help: 32 FROG frames are 0.80 s, and 0.80 s of DROW is 10 frames. What is
averaged out is a defect of the stamping mechanism, counted in frames, so frames is its unit -- stated per
dataset rather than shared, which is the part that was wrong.

### DROW train is not test's kind, and val is one recording (2026-11-07)

Measured read-only from DROW's `.wp` / `.wc` / `.wa` and `.odom2` files, by recording name and odometry extent.

| DROW train, by recording type | files | annotated | pedestrians | robot path |
| --- | ---: | ---: | ---: | ---: |
| `run_*`, driving | 30 | 10,578 | 15,475 | 2,313 m |
| `lunch_*`, one room | 1 | 2,623 | 4,998 | 17 m |
| staged `n-` / `r-` / `s-`, stationary | 36 | 3,181 | 2,571 | 0 m |
| staged `rot-`, turning in place | 11 | 1,283 | 866 | 86 m |

- **Concentrated.** Two recordings hold 46% of train's pedestrians (`run_2015-11-25-14-58-49-a` 6,000, `lunch` 4,998); five hold 72%.
- **Val is one recording.** `run_2015-11-26-14-07-15-a` holds 2,207 of val's 3,102 pedestrians (71%), so any stopping rule or checkpoint choice made on DROW val reads essentially that recording.
- **Test is 42% person-free.** 1,026 of test's 2,428 annotated scans hold no person -- annotated for a wheelchair or a walker, or for nothing -- where FROG's `official` test is populated scans only, so a FROG pooled AP never charges a detector for confidence on an empty scan and a DROW one does (measured 2026-11-09, `utils/drow_locality_property.py`; worth 2-3 pp of our stage 2's deficit to the cutout, `CHANGELOG.md`).
- **Val leaks a little, test does not.** 26.1% of val's pedestrians sit in a sister segment (`-a` / `-b`) of a train recording; none of test's do.
- **Train is not test's kind.** A third of train's annotated frames come from a stationary robot, and every val and test recording is a driving run; four of five test recordings range over 52-69 m of odometry extent, against at most 28 m for any train run.
- **What the protocol says** (Beyer et al. 2018, arXiv 1804.02463): persons were annotated only in frames already annotated for wheelchairs and walkers; a wheelchair user is not labelled a person; a walker user is labelled twice, person and walker; batches of 100 frames, every 4th batch, every 5th frame. DR-SPAAM and the FROG paper both call DROW's val split harder than its test split.

**The labels on the scans, measured 2026-11-08** (`utils/drow_label_geometry.py`, model-free, `results/drow_label_geometry.json`):

| | train | val | test |
| --- | ---: | ---: | ---: |
| people with a return within 0.2 m of their centre | 95.1% | 98.5% | 99.0% |
| best alignment over +-4 beams, mirrored or not | as loaded | as loaded | as loaded |
| beams per person, 0.6 m disk / 0.35 m disk | 14.6 / 11.3 | 17.7 / 12.5 | 21.5 / 14.0 |
| of a person's 0.6 m disk, claimed by another class (0.6 m for all, nearest) | 12.1% | 12.6% | 17.8% |
| of a person's 0.35 m disk, claimed by another class (0.6 m for all, nearest) | 1.7% | 0.9% | 3.3% |
| walkers with a person annotation within 0.3 m | 5.4% | 5.0% | 8.4% |
| wheelchairs with a person annotation within 0.3 m | 0.2% | 0.0% | 0.1% |

- **No loader defect.** Every split scores best exactly as loaded, and nearly every person sits on a return.
- **The 0.6 m disk reaches into the other classes.** An eighth to a sixth of the beams a 0.6 m person disk trains as person are nearer a wheelchair or walker annotation; at DROW's own 0.35 m it is 1-3%.
- **"A walker user is labelled twice" is not what the files show at 0.3 m.** If walker users carry a person label too, it sits more than 0.3 m from the walker's centre -- plausible, since the frame is held in front of the body -- and the share at a wider radius is not measured yet. The wheelchair rule holds as written.

**The scan geometry of each split, measured 2026-11-09** (`utils/drow_locality_property.py --readings geometry`,
model-free, every annotated scan, `results/drow_geometry.json`):

| p25 / p50 / p75 | DROW train | DROW val | DROW test | FROG train (every 25th) | FROG test (every 5th) |
| --- | --- | --- | --- | --- | --- |
| median range of the returns within 10 m | 3.25 / 4.33 / 5.21 m | 1.47 / 1.92 / 3.77 m | 1.46 / 1.74 / 2.12 m | 2.64 / 3.61 / 4.90 m | 2.66 / 3.72 / 5.15 m |
| share of beams returning within 10 m | 0.71 / 0.80 / 0.88 | 0.84 / 0.94 / 0.96 | 0.94 / 0.95 / 0.97 | 0.75 / 0.86 / 0.92 | 0.72 / 0.83 / 0.90 |

- **DROW's test is enclosed and its train is open.** 72% of test scans fall under train's 10th percentile of median
  return range (2.08 m), 98% under its 25th (3.25 m); FROG's test sits on FROG's train. Val spans both.
- **It is part of what our stage 2 fails on** (`CHANGELOG.md`, 2026-11-09): on DROW val its scores stay comparable
  between scans of typical geometry and break on enclosed ones, while a cutout's do not; but balancing the geometry
  does not make it transfer (next bullet), so geometry is one measured part of the shift, not all of it.
- **Scene novelty, as nearest-neighbour distance of coarse sector medians, does not show it**: FROG test is as far from
  its nearest training scan (median 0.765 m) as DROW test (0.541 m). A nearest neighbour can be close while the
  distribution is not, which is why the shift is read off the quartiles instead.
- **Balancing the geometry is not enough for our stage 2 to transfer** (`CHANGELOG.md`, 2026-11-09, `TODO.md` row 106):
  on a repartition with every split ~24% enclosed its train / test ratio is 1.76, against 2.63 on a control that keeps
  the official test. Under that control the official test's *open* scans break as badly as its enclosed ones, so the
  official test differs from training as a set of recordings, and geometry is one measured part of that difference.

## What the false positives are, and what that rules out

Detector-dependent, but stable across the two published models and therefore more a property of the scenes than of either model.
Measured on FROG's person-free frames at threshold 0.3.

| | phantoms / frame ¹ | recur within 0.3 m next frame |
| --- | --- | --- |
| LFE-Peaks | 1.38 | **88.3%** |
| LFE-PPN | 4.23 | **77.6%** |
| *annotated people, same measure* | 3.06 | *99.8%* |

¹ On the **longest unbroken person-free run** per recording, which is what a next-frame
recurrence measure needs. That subset is denser in phantoms than the person-free frames at
large — 1.53 against 1.39 for LFE-Peaks and 4.15 against 3.17 for LFE-PPN — so quote the
run figure only against other run figures. Re-measured 2026-09-11; LFE-PPN's row was
previously 2.36 / 69.9%, taken before its `nms_radius` was recalibrated 0.8 m -> 0.30 m
(`performance-log.md`).

**Phantoms are static scene structure, not noise.** They persist almost as reliably as real people do.
Two consequences, both negative and both worth knowing before building anything:

- **A confirm-then-report filter cannot remove them.** It can only remove a phantom that fails to reappear, and 88% reappear immediately. Measured directly: SORT over these detectors buys ~13% phantom reduction for 1.5pp of AP, ~50% for ~25pp, and still leaves a third to a half of person-free frames carrying one (`performance-log.md`, `docs/PAPER.md`).
- **Nor does scene occupancy explain them.** Separating duplicate detections on already-detected people from genuine phantoms, the genuine rate is **1.387 per populated frame against 1.391 per person-free frame** (LFE-Peaks) and **3.416 against 3.174** (LFE-PPN) — the same to within 0.3% and 7.6% respectively. Empty scenes do not provoke the failure; they stop concealing it.

**A denser phantom field is harder to filter temporally, not easier.** LFE-PPN has the larger *transient* fraction (22% against 12%) yet benefits *less* from tracking at every setting, because at 4.23 phantoms per frame a flickering phantom is usually close enough to a *different* phantom for a tracker to associate them into one continuing track. Phantom density, not phantom persistence alone, sets what association can do.

## What is not measured yet

- **Whether temporal *ordering* carries information about the label at all**, model-free. The 0.5pp shuffle result tested one architecture with a known-weak spatial backbone; it is not evidence about the data.
- **Whether missed people leave sub-threshold evidence.** If the 21% of people a detector misses mostly produce weak-but-persistent responses, a temporal model has recall headroom that a tracker which only filters cannot reach. A tracker that *rescores* from track confidence can (CBMOT, `../docs/RESEARCH.md` §7.4.1), so that part of the headroom is not exclusive to an architecture, and E14-E15 measure how much of it a rule reaches.
- **JRDB**, at all. Needs manual registration and is not downloaded (`gotchas.md`).
- **Any venue beyond these two.**
