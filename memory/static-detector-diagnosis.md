# Static-detector diagnosis — where LFE loses, and to what

*keywords:* range-stratified, scale vs capacity, distance normalisation, cutout, depth tunnel, sub-threshold, miss budget, Cover and Hart, 1-NN, Bayes error, range_probe.py, repr_probe.py, Tier 0

Why a single-frame detector misses what it misses, measured rather than assumed.
Produced by `utils/range_probe.py` (one detector pass, per-person records) and `utils/repr_probe.py` (no detector at all, 1-NN on four window parameterisations).
Both cache, both run in minutes, neither trains anything.
This is `docs/PROPOSAL.md`'s Tier 0, run 2026-09-11 on the FROG `official` test recording.

## The question these answer

`docs/PROPOSAL.md` states a **governing hypothesis**: *DROW's 4.7 pp advantage over LFE-PPN comes from the cutout's distance normalisation, not from its 8.8x parameter count.*
Scale or capacity — the answer decides most of that document's Tier 2.

**The obvious test is unavailable and it is worth knowing why.**
Comparing per-range recall against DROW3's needs DROW3 on FROG, and its 73.9% there is *published*, not ours: the bundled DROW weights are DROW-trained, so a per-bin curve from them would measure a zero-shot transfer, not the published model.
So both probes below are built to need only detectors we can actually run.

## Answer: scale — but the dominant axis is depth, not angle

`repr_probe.py` removes the model.
It builds the same 48-point window around the same beams under four parameterisations and classifies each with **1-NN**, which has no capacity worth confounding: whatever separates the rows is the *representation*.
DROW's cutout normalises distance on two independent axes, so the probe is a 2x2.

| axis | DROW's cutout | LFE's full scan |
| --- | --- | --- |
| **angular** | half-width `atan(0.5 * 1.66 / z)` beams — always spans 1.66 m of arc, then resampled to 48 points | a fixed beam grid |
| **depth** | `clip(w, z +- 1) - z` — a tunnel centred on the beam's own range | one global affine map, `1 - clip(r, 0.2, 10) / 10` |

14,002 reference windows from `official` train, 10,226 query windows from the test recording, balanced 50/50 person-beam vs background-beam.

| angular | depth | 1-NN err | miss-p | 0-2 m | 2-3 | 3-4 | 4-5 | 5-6 | 6-8 | 8-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive | centred | **11.08%** | 16.3% | 8.7% | 8.3% | 10.4% | 10.7% | 12.0% | 12.1% | 15.2% |
| adaptive | lfe | 18.86% | 28.1% | 19.5% | 17.4% | 19.6% | 16.5% | 18.5% | 18.1% | 22.4% |
| fixed | centred | 13.56% | 17.0% | 12.7% | 8.2% | 11.3% | 12.6% | 14.5% | 15.1% | 20.1% |
| fixed | lfe | 19.60% | 27.8% | 15.8% | 14.9% | 18.5% | 17.7% | 20.7% | 20.7% | 28.7% |

**The depth tunnel is worth ~3x what the adaptive angular width is.**
Holding the angular axis fixed, `centred` beats `lfe` by **7.8 pp** (adaptive) and **6.0 pp** (fixed).
Holding depth fixed, `adaptive` beats `fixed` by only **2.5 pp** (centred) and **0.7 pp** (lfe).

This was not the expected shape, and it *lowers* the cost of acting on it: per-beam depth centring is an **input transform**, not an architecture change.

### The angular term is purely a long-range term

Its per-bin advantage (`fixed` error minus `adaptive` error) is the cleanest curve either probe produced:

| depth | 0-2 m | 2-3 | 3-4 | 4-5 | 5-6 | 6-8 | 8-10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| centred | +4.0pp | -0.1pp | +1.0pp | +2.0pp | +2.6pp | +2.9pp | **+4.9pp** |
| lfe | -3.6pp | -2.5pp | -1.1pp | +1.2pp | +2.2pp | +2.6pp | **+6.4pp** |

Zero at ~4 m, growing in both directions, and far more expensive on the far side.
That is what a scale-normalisation effect looks like, and 4 m is where the fixed window was calibrated — its half-width is set to the *median* of what the adaptive window uses (41 beams, 20.5 deg), so neither side is handed a wider window on average.
**A fixed-geometry window is correct at exactly one range and pays for it everywhere else.**

### Cover & Hart bracket, and what it does not bound

The best row gives `E_1NN = 11.08%`, so by Cover & Hart (*IEEE Trans. Inf. Theory* 13(1), 1967) the Bayes error of that representation is in **[5.9%, 11.1%]**.

Two limits, both real:

- **It bounds the 48-point window, not the task.** A model with a wider receptive field — LFE's U-Net, and any full-scan design — sees strictly more than the window does and may beat the window's Bayes error. It *is* the right bound for cutout-based detectors, which is to say DROW and DR-SPAAM.
- **It is asymptotic**, and 14,002 reference points in 48 dimensions is not asymptotic. The true `E_1NN` is lower, so the lower bound is optimistic.

### Cartesian input: can it replace DROW's preprocessing?

Owner's proposal (A42), and the answer is a qualified yes. Instead of a 1D range
tunnel, express each window's beams as **metric offsets from the window's own
centre point, rotated into that centre beam's frame** -- Cartesian coordinates
from the start. Same beams, same 1-NN, 24 points x 2 channels = **48 dims**,
matched to the cutout so this is not a dimensionality comparison. The radial
component is clipped to +-1 m, the exact analogue of DROW's tunnel, so the two
differ in coordinate system and nothing else.

| angular | depth | 1-NN err | 0-2 m | 2-3 | 3-4 | 4-5 | 5-6 | 6-8 | 8-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive | centred | **11.08%** | 8.7% | 8.3% | 10.4% | 10.7% | 12.0% | 12.1% | 15.2% |
| adaptive | **cartesian** | 12.38% | 11.5% | 10.2% | 11.2% | 12.3% | 12.1% | 12.4% | 17.0% |
| fixed | **cartesian** | **13.39%** | 12.2% | 9.2% | 9.8% | 12.6% | 13.5% | 15.9% | 19.5% |
| fixed | centred | 13.56% | 12.7% | 8.2% | 11.3% | 12.6% | 14.5% | 15.1% | 20.1% |
| adaptive | lfe | 18.86% | 19.5% | 17.4% | 19.6% | 16.5% | 18.5% | 18.1% | 22.4% |
| fixed | lfe | 19.60% | 15.8% | 14.9% | 18.5% | 17.7% | 20.7% | 20.7% | 28.7% |

**1. At matched window width, Cartesian ties the depth tunnel** -- 13.39% against
13.56%, inside noise. So the simplification is **free**: no range-dependent
half-width, no bilinear resample, no out-of-bounds branches, and the same
separability. It also keeps *both* displacement components where the tunnel
keeps only the radial one.

**2. Against what LFE actually does, Cartesian wins by 5.5-6.2 pp.** That is the
comparison that matters for designing a full-scan model of our own: `1 - r/10`
is a global affine map that leaves a person's appearance range-dependent, and
metric offsets do not.

**3. It does not replace the range-adaptive angular window.** `adaptive/centred`
stays best at 11.08%, and `adaptive/cartesian` is 1.30 pp behind it. But the
*interaction* is informative: going adaptive buys the tunnel **2.48 pp** and
Cartesian only **1.01 pp**, which is what you would expect if metric coordinates
already carry part of the scale information the adaptive window otherwise has to
supply.

**Verdict: take the coordinates, keep the range conditioning.** For an
architecture of ours that means Cartesian input *plus* a range-conditioned
receptive field (A39), not Cartesian instead of it.

**Prediction and outcome, recorded** (`dos-and-donts.md`): predicted that
`fixed/cartesian` would beat `fixed/centred` -- it does, 13.39% against 13.56%,
though by less than the prediction implied. Predicted the open question of
whether it reaches `adaptive/centred`'s 11.08% -- it does not.

**One methodological note, because it is the second instance of the same error.**
The first run of this comparison gave Cartesian **14.88%** and would have said
"clearly worse". It was unfair: the `fixed` cutout row uses a **41-beam**
half-width resampled to 48 points, while the Cartesian row was taking **+-12 raw
neighbouring beams** -- a third of the context. Extent and dimensionality were
the same knob. Separating them changed the verdict from "worse" to "ties". A38's
confound was an under-explored grid; this was an unmatched control. Both were
caught by noticing an asymmetry rather than by any check that runs on its own.

### What to raster, and what not to

Converting coordinates is not the same as **rasterising onto a grid**, and only
the second carries a sparsity cost: 720 points into a 20x20 m grid at 10 cm is
~40,000 cells at ~2% occupancy, which is why 2D-LiDAR person detection stayed
polar while 3D LiDAR went to bird's-eye view with ~100k points per sweep.
Coordinate conversion costs two multiplies per beam against a precomputed
sin/cos table and has no such problem. The rasterised form is a separate
question -- a substrate on which roles A, B and C could share one coordinate
system -- and is deferred to `TODO.md` A43.

## Where the recall actually goes

`range_probe.py`, 30,737 annotated person-observations, `score >= 0.3`.
Recall at that single operating point is **78.9%** (LFE-Peaks) and **80.7%** (LFE-PPN) — an operating point, not AP; do not put these in a table next to 65.1% / 69.6%.

| range | share of people | Peaks recall | PPN recall | median beams on target |
| --- | --- | --- | --- | --- |
| 0-1 m | 1.6% | 88.3% | 84.2% | 68 |
| 1-2 m | 12.2% | 89.2% | 87.7% | 42 |
| 2-3 m | 14.0% | 90.0% | 89.7% | 26 |
| 3-4 m | 12.9% | 88.1% | 89.1% | 19 |
| 4-5 m | 12.3% | 84.3% | 86.4% | 16 |
| 5-6 m | 12.8% | 79.9% | 81.9% | 13 |
| 6-8 m | 21.1% | 73.9% | 74.5% | 11 |
| 8-10 m | 13.1% | **48.8%** | **59.0%** | 9 |

**No annotated person in FROG's test recording is beyond 10 m** (0.0%), so LFE's `_SCAN_FAR = 10.0` input clip is *not* binding — a confound checked and dismissed, not assumed.

**A third of the people are past 6 m and that is where a third of the recall is lost.**
The 8-10 m bin alone holds 13.1% of annotations at under 60% recall.

### LFE-PPN vs LFE-Peaks is a natural experiment, and it points the same way

Same backbone *architecture*, different head, separately trained (two ONNX files — so this is a natural experiment, not a weight-controlled ablation).
LFE-Peaks' head is **beam-indexed**: `find_peaks` on the beam axis, then a metric merge.
LFE-PPN's head is **metric**: 30 range anchors per 6-beam sector, and an arc offset the decoder converts to an angle by *dividing by the decoded depth* (`lfe_detector.py`, the `phi_offset` line) — literally the `arc / d` conversion distance normalisation is about.

**PPN wins by 10.2 pp at 8-10 m and loses by 4.1 pp at 0-1 m.**
The more range-aware head is better exactly where range hurts, and worse where it does not.

## The miss budget — and a third of it is not the network's fault

`range_probe.py` reads LFE-Peaks' raw per-beam probability map *before* `find_peaks` and merging, at the beams the annotated person actually returns.
This is the long-standing "sub-threshold evidence" measurement.

| | n | p10 | p25 | median | p75 | p90 |
| --- | --- | --- | --- | --- | --- | --- |
| detected | 24,237 | 0.781 | 0.954 | 0.991 | 0.997 | 0.999 |
| **missed** | 6,483 | 0.001 | 0.005 | 0.036 | 0.162 | 0.545 |

The model is not poorly calibrated — it is near-binary, and confident both ways.

| what the map said | n | share of misses | median beams | median range | median nn | detection within 0.5 m |
| --- | --- | --- | --- | --- | --- | --- |
| ~0 (< 0.01) | 2,080 | 32.1% | 8 | 7.8 m | 3.08 m | 0.4% |
| 0.01-0.1 | 2,256 | 34.8% | 10 | 6.3 m | 1.64 m | 1.6% |
| 0.1-0.3 | 1,347 | 20.8% | 11 | 6.0 m | 0.16 m | **84.3%** |
| 0.3-0.5 | 135 | 2.1% | 11 | 6.5 m | 0.24 m | **63.0%** |
| > 0.5 | 665 | 10.3% | 15 | 4.2 m | 0.28 m | **60.0%** |

| miss class | n | cost |
| --- | --- | --- |
| **representation** (map < 0.1) | 4,336 | **14.11 pp** of recall |
| **borderline** (0.1-0.3) | 1,347 | **4.38 pp** |
| **decoder** (map >= 0.3, peak lost) | 800 | **2.60 pp** |

Two distinct failures, and they do not live at the same range:

- **Two thirds of misses are representation failures** — the map says ~0, the nearest detection is metres away, and the median case is a 10-beam return at 6-8 m. Nothing downstream can recover these.
- **The other third is crowding.** Above a map value of 0.1, **60-84% of missed people had a detection within 0.5 m, median 0.16-0.28 m** — closer than `merge_radius` (0.30 m). These sit at 4-6 m with 11-15 beams on target: *well-observed people, standing near another person, whose detection was merged into their neighbour's.*

  > **Corrected 2026-09-11 — this class is ~1.1% of people, not the ~5 pp an earlier draft of this file claimed.** The `nn` column above is the distance to the nearest detection **at any score**, because `range_probe.py` computes it before thresholding. Many of those neighbours are sub-threshold detections that would never be reported, so the figure bounds a larger set than the operating point contains. Measured at the operating point itself (`nms_sweep.py --score-by operating-point`, A40), close-pair misses are **0.033 per frame — 331 of 30,737 people, 1.1%** — and they move only between 0.028 and 0.062 per frame across merge radii from 0.20 m to 0.50 m. **The merge radius is not where 5 pp of recall is hiding.** What it does control is *duplicates*, which fall 18-fold over that same range.

That is a post-processing problem, and it is the concrete mechanism behind the merge-radius sensitivity already recorded in `LFEPeaksDetector`'s docstring (0.25 m costs 1.9 pp, 0.80 m costs 6.3 pp).
**Caveat:** "a detection within 0.5 m" is the crowding signature, not proof of error — in a genuine crowd the Hungarian assignment may be giving that detection to its rightful owner. What it establishes is that these misses are *resolution between adjacent people*, not blindness.

## The weak control, stated because it is easy to over-read

`range_probe.py` also reports recall at fixed **beams-on-target**, intended as an information-content control: if recall still falls with range at a fixed number of returning beams, that is a scale deficit.
It does fall — the best-populated clean comparison is 9-14 beams, **83.9% at 5-7 m against 73.9% at 7-10 m** (n ~ 3,000 each).

**But the control is weak, because the fill factor is nearly constant:**

| range | median beams | geometric subtense | fill |
| --- | --- | --- | --- |
| 0-1 m | 68 | 196.9 | 35% |
| 2-3 m | 26 | 73.1 | 36% |
| 4-5 m | 16 | 40.6 | 39% |
| 6-8 m | 11 | 26.5 | 42% |
| 8-10 m | 9 | 20.9 | 43% |

A 0.8 m person returns ~37-43% of the beams they subtend at *every* range, so beams-on-target is close to collinear with range and the off-diagonal cells of that table select for physically atypical configurations (partial occlusion at short range, groups at long range).
Read it as corroboration only.
**`repr_probe.py` is the load-bearing evidence**, because there the model is gone and the two representations see identical beams.

## What this changes

- **The governing hypothesis is supported, with its axes reordered.** The gap is representation, not capacity — but per-beam **depth** normalisation is the larger term, and `docs/PROPOSAL.md` had the angular term as its headline candidate.
- **A new top candidate, cheaper than the one it displaces:** give the full-scan FCN a locally range-centred input channel (`r_i` minus a windowed median, the high-pass equivalent of the cutout's `- z`) alongside `1 - r/10`. An input transform plus one training run, against a build plus a run for range-conditioned dilation.
- **Range-conditioned dilation stays, demoted to second**, and now has a measured size: ~2.5 pp of 1-NN error overall, up to 6.4 pp at 8-10 m.
- **The head swap gets a sharper motive.** It is no longer "why does PPN beat Peaks by 3.6 pp" but "PPN's metric decode is worth +10.2 pp at 8-10 m and -4.1 pp at 0-1 m — what happens if the Peaks head decodes metrically too?"
- ~~**A Tier 1 item that needs no network at all:** ~5 pp of recall sits in close-pair over-merging.~~ **Withdrawn 2026-09-11**, by the measurement that was supposed to confirm it. Recall moves only **1.8 pp** across the entire merge-radius range 0.20-0.50 m (79.5% to 77.7%), while duplicates fall from 0.636 to 0.035 per frame. The radius governs duplicates, not close pairs, and there is no cheap recall here. See `rejected-ideas.md` and `TODO.md` A40.

## Caveats that apply to every number here

- **Single operating point.** `score >= 0.3` throughout; AP integrates the whole curve. The two are not interchangeable and the recall figures here must never share a column with an AP.
  They are, however, on exactly the same data: `--stride 5` yields 30,737 annotated people, the same count `docs/PAPER.md`'s results table scores its 65.1% / 69.6% AP on. Same frames, different quantity.
- **One recording.** FROG `official` test is `frog_16-41`, one venue. `dataset-properties.md`'s warnings about the museum carry over.
- **`--stride 5`** on 30,737 of ~154k person-observations, systematic not random. Consistent with the `--stride 50` smoke run to within 0.2 pp on every headline figure.
- **The 1-NN probe uses train as reference and test as query**, different recordings, so there is no leakage — but also no tuning, so 1-NN error is an honest-but-blunt instrument.
