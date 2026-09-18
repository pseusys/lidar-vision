# Dataset properties — what the data itself constrains

*keywords:* time span, temporal horizon, receptive field, how long, how fast, people move, stationary people, displacement, frame rate, Hz, annotation density, person density, phantom persistence, architecture constraints, cross-dataset, DROW vs FROG, ceiling

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

¹ DROW3 measures 66.5% here against its authors' published 61.9% — a known
harness overshoot on DROW that predates this analysis (`performance-log.md`), not
a property of the data.

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

| elapsed | FROG: moved < 0.5 m | DROW: moved < 0.5 m |
| --- | --- | --- |
| 1 s | 67.3% | 62.8% |
| 2 s | 46.6% | 37.1% |
| 5 s | 36.0% | 28.8% |
| 10 s | 29.5% | 19.5% |
| 30 s | 16.3% | *(unreliable, see below)* |

**Three things this settles.**

1. **Most of the available motion appears in the first ~5 seconds**, in both venues. The curve falls steeply to 5 s and then flattens. Horizon past that keeps paying on FROG but with sharply diminishing returns.
2. **There is a hard ceiling on every motion-based method**, and it is not small: at a 10-second horizon, 20-30% of people have not moved a body width. No architecture crosses this — it is a property of people, not of models.
3. **It is not a museum artifact.** This was the main risk to the whole premise (FROG is a science museum, where people stand and look at things). DROW is a care facility, and people there are only ~8-10pp more mobile at every horizon. The problem survives the change of venue.

**What a temporal window of a given size buys**, at each dataset's own rate:

| window | FROG frames (26.2 Hz; really 40 Hz, TODO A49) | DROW frames (10 Hz; really 12.7 Hz, TODO A49) | people it could possibly help with |
| --- | --- | --- | --- |
| 2 s | 52 | 20 | ~53% (FROG) / ~63% (DROW) |
| 5 s | 131 | 50 | ~64% / ~71% |
| 10 s | 262 | 100 | ~70% / ~80% |
| 30 s | 786 | 300 | ~84% / — |

**This is the argument for and against a long horizon at once.** The ceiling genuinely rises with horizon — so the horizon matters — but a feed-forward window of 262 frames x 720 beams is not a window, it is a sequence, and reaching 10 s that way is impractical. That is the case for a *recurrent* state (O(1) per frame, unbounded horizon) rather than a wider convolution, and the reason `rejected-ideas.md`'s recurrence entry is due for the re-examination its own reopening clause describes.

**Caveats, because they are load-bearing:**

- **Gate sensitivity is checked and the conclusion holds.** Trajectory association needs a gate wide enough to absorb annotation jitter; too tight and tracks shatter, which biases the statistic *toward* stationarity. Across gates of 0.4 / 0.6 / 1.0 m the 10 s figure moves 30.2 / 29.5 / 28.5% — stable, and a looser gate moves it conservatively. Only the 30 s row is materially gate-sensitive (18.7 / 16.3 / 14.5%), so quote it as "roughly a sixth" and not more precisely.
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

**The distinction that keeps this from being fatal: temporal integration is not motion.** Everything above bounds methods that discriminate *by displacement*. A stationary person still produces a weak but **consistent** return, and averaging across frames raises signal-to-noise with zero motion involved. This project has already measured that effect at **3.6pp** in an architecture that provably ignores temporal ordering (`CHANGELOG.md`, the history ablations: 4.1pp total temporal contribution, only 0.5pp of it ordering). A tracker cannot do this — it only ever sees post-threshold detections — so it is headroom that belongs specifically to a temporal *architecture*, and it is not bounded by the table above.

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

Produced by `utils/three_horizon_oracle.py` (`checkpoints_three_horizon/oracle_ours.json`) on the FROG **`official`** test split, so it is **not comparable with the LFE rows above**, which `utils/temporal_oracle.py` measures on `balanced`.

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
  person, because the split contains no people. A41 measured what happens when a real
  mechanism must: a world-cell persistence map reaches 77.5% phantom removal on person-free
  frames and removes 73.8% of *all* detections on populated ones, taking recall from 79.3% to
  23.2%. **Do not quote 89.7% as an achievable target.** Measured properly on populated frames
  (`horizon_sweep.py --discrimination`, A45), the best acausal trail-length filter reaches an
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
| annotated scans | **every one** | **5.0%** — every 5th scan, 0.39 s apart |
| **people per annotated frame** | **3.06** (test; 3.13-4.08 across recordings) | **0.81** |
| classes | pedestrian only | pedestrian / wheelchair / walker |
| odometry, per frame | 1.88 cm, up to 0.4° (p95) | 0.32 cm |
| session structure | test fragmented into 47 sessions, median ~26 s; the three raw extras are unbroken 60-70k-frame recordings | 5 sessions, ~18 min each |

**Four consequences for any architecture meant to run on both.**

1. **Parameterise the temporal window in seconds, never in frames.** A `T=5, dtime=10` window spans **1.25 s on FROG and 3.9 s on DROW** — the same hyperparameters are two different experiments. Every cross-dataset comparison in this repo predating 2026-09-10 has this confound.
2. **Person density differs 3.8x**, so AP is not comparable across the two: precision is a base-rate-sensitive quantity and FROG hands a detector nearly four times as many chances to be right per frame.
3. **Annotation density differs 20x.** DROW's 2,428 labelled frames are the binding limit on any supervised temporal model trained there — and the reason the motion table above is thin on the DROW side. Densifying it by *interpolating between real annotations* (never by pseudo-labelling with a detector, whose 1.4-1.9 phantoms/frame would be taught as positives) is tracked as a research item.
4. **Beam count and angular resolution differ**, so a beam-indexed architecture is not transferable without resampling; beam index does not mean the same physical angle in the two.

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
- **Whether missed people leave sub-threshold evidence.** If the 21% of people a detector misses mostly produce weak-but-persistent responses, a temporal model has recall headroom that a tracker structurally cannot reach — it only ever sees post-threshold detections.
- **JRDB**, at all. Needs manual registration and is not downloaded (`gotchas.md`).
- **Any venue beyond these two.**
