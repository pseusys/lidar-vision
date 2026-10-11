# Changelog

Step-by-step history of this repo: what changed, when, what was tested, what the result was, and the decision that came out of it.
Source files carry a short pointer here instead of the full history inline.
Explanations of how things *currently* work live in [`memory/`](memory/README.md), not here.

Organized by date, **newest first**.
Entries before 2026-09-08 are a one-time backfill from `docs/RESEARCH.md`, written during the documentation-layout migration on that date — dates come from file modification times and the evaluation run's own stated date, not from separate per-day commits.

## How to navigate this file

Every entry carries a `*keywords:*` line right under its heading, holding the identifiers, flags and constants that distinguish that entry from the rest of the file.
That is the fast path:

```bash
grep -n "^\*keywords:.*<identifier>" CHANGELOG.md                        # entries touching it
grep -n "^##.* <YYYY-MM>" CHANGELOG.md                                   # everything from a month
```

Topic index — grep the phrase in the right column to land on the entries:

| Looking for | Grep |
| --- | --- |
| the DR-SPAAM cutout-normalization bug | `window_depth` |
| the FROG odometry / session-splitting bugs | `session_gap_s` |
| the LFE-PPN anchor/recall bugs | `LFEPPNDetector` |
| the dtime sweep | `dtime=5` |
| the tracker's invocation-rate finding | `SimpleTracker` |
| this documentation layout itself | `agentic-layout-template` |

Decisions to *not* do something are in [`memory/rejected-ideas.md`](memory/rejected-ideas.md).
Open work is in [`TODO.md`](TODO.md).

## Rotation

This file covers the current development period; this project has not yet had a release to rotate against.
When it does, move this file's entries to `memory/changelog-archive/CHANGELOG-v<X>.md`, copy this header to the top of the archived file, and leave this file with the header and no entries — see the template this layout was adapted from for the full procedure, preserved in `memory/changelog-archive/` once the first rotation happens.

---

## Items 9, 11 and 12: the rename lands, the linter finds two crashes, and the ladder closes (2026-10-11)

*keywords: A52, takheliped, follow_the_drow, rename, pybind11, cpp_binding, ruff.toml, Makefile, .venv, item 12, F1+W, a59_f1w_lfe_input_width, attribution ladder*

**Item 12 completes A59's attribution ladder, and the answer is that the two costs do not add.** F1 (LFE's input alone) was
worth nothing measurable and W (LFE's widths alone) cost 1.5 pp; taken together, three seeds give **80.68 / 80.66 / 80.51, mean
80.62** -- which is W alone (80.70) to within **0.08 pp**, against an additive prediction of 80.84. **LFE's input contributes
nothing on top of its width, exactly as it contributed nothing alone.** So the width is the whole cost, and the ladder's
conclusion strengthens: with LFE's input *and* its width at once, our network still scores 80.62% against LFE-Peaks' 65.1% -- a
15.5 pp gap that neither ingredient explains, nor their combination, nor DR-SPAAM's training protocol. What is left is the
network's form and the training recipe, and item 7 was dropped before it could separate those.

**The project has one name again (item 9, A52, widened by the owner mid-rename).** `follow_the_drow` was the last name from the
superseded line of work, and it is gone from the Python package, the C++ namespace and header tree, the CMake target and config,
the ROS package with its launch file and image name, every script and test, and the prose. The C++ half was **compiled** rather
than assumed: MSVC 2022 builds and links the extension clean against the renamed namespace. `pybind11` was missing from the
venv, which means nobody could have rebuilt that extension at all before today.

**Two paths deliberately keep the old word.** `checkpoints_three_horizon/` stays, because every checkpoint stores the absolute
path of the stage 2 it was trained on and `coarse_e2e.py` resolves it -- renaming it would dangle ~30 runs' provenance for a
name nobody outside this repository sees. And `CHANGELOG.md` is left alone, because its entries describe what the names were
when each thing happened.

**Item 11's three gaps were each bigger than recorded.** `--help` exited 1 in **eleven** scripts, not one: the `os._exit`
wrapper carrying the ROCm teardown workaround caught `BaseException`, which catches argparse's `SystemExit` too, so every one of
them printed its help correctly and then reported failure. A project-level `ruff.toml` now selects the **bug-shaped** rules --
the full set reports ~2,800 findings here, nearly all annotations and whitespace, and a config that fails on every run teaches
people to ignore the linter -- and it found **two latent crashes**: `render_video.py` called `_win2global` without importing it,
and `li2former.py` imported numpy inside a function so its return annotation had to be quoted to compensate. The `Makefile`
built `venv/` while the working interpreter is `.venv/`, so `make test` ran against an environment nobody else used; its venv
target is now a **file** target, which stops `make test` reinstalling pip, jupyter and the library before every run.

## Items 8 and 28: the value rule's constants defend themselves, and the horizon claim moves onto the adopted recipe (2026-10-11)

*keywords: D4, learn_value, value_gain, value_fade, gain 0.085, fade_time_s 8.94, E1, horizon sweep, a51_horizon_adopted.json, a51_d4_learn_value_s0, A35 criterion, item 5*

**D4, the last unmeasured choice in A51 phase 3b.** `manage_slots` runs under `@no_grad`, so its constants were never trained:
a matched slot's value climbs at `gain` = 0.1 and an unmatched one fades over `fade_time_s` = 10 s. `--learn-value` makes those
two numbers parameters, leaving every discrete decision where it was, and trains them from the hand-set values. One seed on the
adopted recipe, against E13's three-seed control: **AP 86.73 (+0.86, bar 1.32), AP at 0.3 m 85.65 (+0.97, bar 1.62), recall at
matched false positives 95.99 (+0.52, bar 2.37)** -- every delta positive and every one inside its bar, so a null with a
direction. 12.4 h over 51 epochs, 3.97 ms per frame.

**The constants moved by about a tenth: `gain` 0.100 -> 0.085, `fade_time_s` 10.0 -> 8.94 s.** That is the result. Asked why 0.1
and 10 seconds, the write-up can now answer that training them from those values moves them 15% and 11% and buys nothing
outside the noise -- the hand-setting is defended by measurement instead of by assertion. **D4 is also the only A51 arm positive
on all three bars at once**, so claiming the +0.86 would need two more seeds.

**Three independent measurements now agree on the horizon**: the learned fade time at 8.94 s, D3's fixed 8 s decay costing only
0.90 pp, and E1's sweep putting 94% of the gain inside 10 s. `SlotRules`' 10 s was never tuned, which is what makes the
agreement worth stating.

**E1's reset sweep on the adopted recipe (item 28)** closes the caveat item 5 was carrying. AP against how often the memory is
emptied: **83.16 / 83.79 / 84.63 / 85.64 / 86.11 / 86.25 / 86.35** at 0.5 / 1 / 2 / 5 / 10 / 30 s / never, against candidates at
82.22. So the memory is worth +4.13 pp and **+3.19 of it needs history older than half a second**, 77% of the gain; 83% is
reached by 5 s and 94% by 10 s. **A35's "strictly benefits from ordered time" criterion passes by about six times**
`SpaceTimeCNN`'s 0.5 pp shuffle penalty, where B0's backbone gave nine -- narrower with the better stage 2, and now measured on
the model being proposed.

## Item 6 / A47: our DR-SPAAM was a windowed approximation, and that was most of its reproduction gap (2026-10-10)

*keywords: A47, A48, DR-SPAAM, DrSpaamDetector, stream, _SpatialAttentionMemory, windowed, streaming, DROW test, 67.84, 69.08, 69.6, baseline fidelity, evaluate_auc stream*

The official DR-SPAAM keeps its spatial-attention template between calls and clears it at `reset()`; ours re-encoded a `T`-scan
window per frame and seeded the template from that window's oldest scan. `DrSpaamDetector.stream(scan, template)` adds the
published form -- state in, state out, as `ObjectMemory.step` already does here -- and six tests pin the equivalence exactly:
streaming frames 0..k reproduces the windowed forward over that prefix window, for every k, because both evaluate the same
recursion. `evaluate_auc(stream=True)` walks every scan of each sequence in order and scores the annotated ones, so both forms
share one decoder and one metric.

| DR-SPAAM, DROW test, published weights | wp-AUC | wall clock |
| --- | --- | --- |
| windowed `T=5`, every figure in this repository until today | 67.84% | 1,239 s |
| **streaming, the published algorithm** | **69.08%** | 8,719 s |
| the authors' own published figure | 69.6% | -- |

**The published algorithm is +1.24 pp better than what we were calling DR-SPAAM**, and our reproduction of it moves from -1.5 pp
to **-0.52 pp**. Under this project's standing rule -- an undershoot has innocent explanations, an overshoot means the harness is
wrong -- the innocent explanation turned out to be the wrapper, which also makes the result a harness self-test that now passes
more tightly. Every DR-SPAAM accuracy figure here is the windowed form and understates the baseline by about that much.

**Its cost is wrong in the other direction.** The windowed form runs the first two conv blocks five times per frame where the
published one runs them once, so our 350.4 ms FROG row *overstates* it: we have been reporting DR-SPAAM as both less accurate
and slower than it is. The corrected timing belongs to item 29's one-protocol sweep, which now has `stream()` to measure. The
8,719 s above is an evaluation artefact, not a deployment cost -- the streaming pass walks every scan of every sequence one at a
time where the windowed pass scores only annotated frames in batches of eight.

**A48 item 6 closes with it.** Today's windowed run is the post-heading-fix re-measurement that item asked for: **67.84% against
the 68.1% on record**, so the two heading-wrap fixes moved DR-SPAAM on DROW by at most ~0.3 pp. Since both fixes act through
`cutout()`, which every cutout detector shares, that bounds their effect on DROW3 and Li2Former too.

## Retracted: the draft's `balanced` figures, which no measurement produced (2026-10-10)

*keywords: balanced, retraction, 59.8, 63.5, PAPER.md, traceability, A32, A63, LFE-Peaks, LFE-PPN, unverified number*

`docs/PAPER.md` carried **LFE-Peaks 59.8%** and **LFE-PPN 63.5%** on FROG `balanced` test, supporting the claim that "the target
any temporal architecture of ours must beat is fixed". Found while refreshing the draft for the pre-print (item 4): no
`CHANGELOG.md` entry and no file under `memory/` holds either number, and `TODO.md` A32 still listed that evaluation as pending.
**The owner confirms nothing has ever been exercised on `balanced`** -- the partition is defined in the loader and has never
been run. Both figures are **deleted** from the draft, with a note in their place; `TODO.md` A32 and A63 step 0 now say plainly
that the step starts from nothing.

**The general lesson is recorded in `memory/interpreting-evaluation.md`**, because it outlives this retraction: nothing in this
repository checks that a number in `docs/` corresponds to a measurement. `verify_memory.py` checks links, indexes and markdown
style, and the convention that every figure traces to a `CHANGELOG.md` entry is enforced by review alone. A number that appears
only in `docs/` is unverified until the entry is found -- and this one got as far as supporting a claim about what our own
architecture must beat.

## Item 3: the heading-wrap fix did not move the displacement oracle, so 4.20 is superseded, not contradicted (2026-10-10)

*keywords: A45, A48, displacement oracle, role C, 4.20, 2.58, heading wrap, _load_odom, np.unwrap, heading_ab.py, exchange rate, association gate, merge radius*

A48 recorded that A45's role-C displacement bound "fell from 4.20 to 2.58 with the same command" after the heading-wrap fix,
and flagged it as far larger than the ~0.14% of affected frames should cause. `utils/heading_ab.py` answers it by re-projecting
the **same cached detections** through both pose series -- the loader's wrapped interpolation and today's unwrapped one -- with
the association gate, the one-to-one assignment, the bounding-box rule and every threshold held constant.

| arm | best exchange rate | at D = 0.30 m, where the 4.20 was read |
| --- | --- | --- |
| today's unwrapped heading | **2.58** | 2.44 |
| the old wrapped heading | **2.53** | 2.40 |
| unwrapped, association gate 1.0 m | 2.56 | 2.39 |
| unwrapped, association gate 1.2 m | 2.56 | 2.43 |
| *recorded before 2026-09-14* | *4.20* | *4.20* |

**The heading bug is worth 0.05 of the 1.62, and the association gate 0.02.** The bug's footprint is reproduced -- 49 frames
where the two headings disagree by more than 2 degrees, 0.10% of the annotated ones, against A48's count of 54 -- and its effect
on the oracle is proportional to that, which is what the original suspicion implied but the attribution denied.

**The reimplementation is faithful, which is what makes the exclusion mean anything.** Its unwrapped column reproduces
`horizon_sweep.py --discrimination` to the digit at every displacement (2.58 / 2.50 / 2.44 / 2.36 / 1.94 / 1.51 / 1.26 / 1.13 /
0.95) and reproduces A48's own post-fix cross-check: at 0.30 m, 36.0% of phantoms removed for **10.7%** of true positives,
exactly the figures recorded there.

**So 4.20 is a superseded measurement rather than a contradiction**, and what it rests on is detections we no longer use: the
remaining untested difference in that window is LFE-Peaks' `merge_radius`, recalibrated 0.40 -> 0.30 m on 2026-09-10, which took
its phantom rate from 1.28 to 1.39 per frame and therefore changed every trail. Testing that needs a detector pass at the old
radius and is **not scheduled**: the decision it would inform -- which number the paper quotes -- is already settled, since
4.20 was measured on a mis-calibrated detector and 2.58 is reproducible from the current one.

**`docs/PAPER.md` must therefore quote 2.58** (and 2.44 at 0.30 m) with its inputs stated, and must not present the move as a
consequence of the heading fix. The trail-length row of the same table moved for a different and now-understood reason -- the
frame rate -- to 2.37 at its best point.

## Items 1, 2 and 5: FROG is 40 Hz, the whole detector fits the real budget on the GPU but not on the CPU tail (2026-10-10)

*keywords: frame_period_s, 40 Hz, 26.2 Hz, FP/s, cost table, latency_cpu_t4.json, latency_gpu.json, 25 ms budget, horizon_sweep cache, A33-A36, A35 criteria, motion_analysis, displacement oracle*

**The rate.** One definition now serves every script: `frame_period_s`, the mean of the intervals inside each recording. FROG
reads **40.01 Hz** (`official` test) and **40.06 Hz** (`transferred`), matching the sensor's 25 ms per scan; DROW reads **12.70
Hz**, matching its paper's 12.5. The medians they replaced were 26.23 and 10.00 Hz, because FROG's recorder stamps scans in
bunches -- 20.2% of its intervals are under 1 ms. Eleven call sites and three hardcoded constants moved;
`motion_analysis.FRAME_HZ` was unused and went.

**A cache was hiding the rate, and it took the first restatement with it.** `horizon_sweep` stored `hz` inside its pickles
beside the detector pass, so the first re-run replayed 26.2 Hz into every seconds axis and *looked* restated: 30 s converted to
787 frames. The rate is now read from the dataset on every run, never cached, at the price of one dataset load on a cache hit.
**A derived quantity in a cache is a stale number waiting for a definition to change.** The two other caches
(`persistence_map`, `slot_budget`) hold detections only, so their restatements were real.

**What moved, on real data.** FROG's people-move-less-than-their-own-width figures are **52.4 / 41.4 / 32.7 / 24.0 / 13.3%** at
1 / 2 / 5 / 10 / 30 s, against the recorded 67.4 / 47.0 / 36.5 / 30.3 / 15.3%: "two thirds of people move less than their own
width in a second" becomes **just over half**, because a "1 s horizon" was really 0.65 s. The trail-length role-C bound moved
with it -- at a 10 s filter, 90.8% of phantoms removed for 67.1% of true positives, against 82.7% / 47.6% on the old clock. The
**displacement** oracle did not move at all (2.58 at its best point), since displacement has no seconds axis, which confirms
item 3's open question is independent of the rate: why the heading-wrap fix took it from 4.20 to 2.58 is still unexplained.
Every `FP/s` in `memory/performance-log.md` rose 1.53x; the 72,533 person-free frames are **30.2 minutes** of real time, not 46.1.

**The cost table (item 2), on the adopted recipe, one scan at a time, 2,000 frames** (`checkpoints_three_horizon/a61/latency_*.json`):

| | stage 2 | decoding | memory | **total** | p99 | against the **25 ms** budget |
| --- | --- | --- | --- | --- | --- | --- |
| GPU (RX 9060 XT) | 5.41 | 3.23 | 4.58 | **13.32 ms** | 15.32 | **53%** median, 61% at p99 |
| CPU, 4 threads | 8.71 | 2.83 | 9.27 | **20.98 ms** | 25.32 | **84%** median, **101% at p99** |

**Item 1 changed item 2's conclusion**, which is the reason it was done first: against the old 38.2 ms assumption the CPU looked
comfortable at 55% of budget, and against the real 25 ms its **99th percentile misses real time**. The GPU has headroom either
way. The memory is the larger half of the CPU cost (9.27 ms against stage 2's 8.71) and the smaller half on the GPU (4.58
against 5.41).

**A33-A36 closed** against A35's own pre-registered criteria, both met: single-frame **82.20%** against the 66.5% bar, and
**4.47 pp** lost when the memory is emptied every 0.5 s against `SpaceTimeCNN`'s 0.5 pp shuffle penalty, about nine times. The
caveat kept: the 4.47 is on B0's backbone, and E1's reset sweep on the adopted recipe is queued.

## The FROG work becomes a pre-print list; `balanced` becomes another dataset; SHOWCASE.md is dropped (2026-10-10)

*keywords: pre-print, FROG only, balanced, A62, A63, A32, SHOWCASE.md, docs, actionable list, frame rate convention, A33-A36*

Three owner calls, all bookkeeping rather than measurement, and all of them narrow what the next weeks are for.

**`docs/SHOWCASE.md` is deleted.** A standalone one-page summary of the result duplicates what the write-up says, and it had
already gone stale on two points -- it called the joint-training regression "under investigation" after A50 phase 1 item 2 closed
it, and it presented step 3b as a test of feedback when that run's prior never left zero. References in `README.md`,
`docs/RESEARCH.md` and `TODO.md` are gone with it; `CHANGELOG.md` keeps its mentions, being history.

**`balanced` is another dataset, not another FROG split.** Every model needs retraining on it, so it moved out of the FROG queue
into **A63 step 0**, where it is the *first* step because it is the cheapest: same sensor, same venue, same loader and the same
angular grid, so unlike DROW and JRDB it needs no part of A62 and can run whenever the machine is free. It takes A32's last open
row with it -- LFE-Peaks and LFE-PPN on `balanced` test, ~2 min each, never measured -- because our side of that comparison needs
retraining there to mean anything. Its partition rules are unchanged and inherited: `balanced`'s val holds half of `official`'s
train, so the two must never be mixed.

**The FROG list is now ordered by what a pre-print needs**, not by item number. Eight items block it: the frame-rate convention
behind every per-second figure (`evaluate` reads the stamped median, 26.2 Hz, where the sensor delivers 40 Hz, so every `FP/s` in
the repo is 1.53x low and the "38.2 ms budget" is really 25 ms); one end-to-end cost table on the adopted recipe; the
displacement oracle's unexplained 4.20 -> 2.58; refreshing `docs/PAPER.md` and `memory/performance-log.md` off the pre-A59
recipe; closing A33-A36 against A35's own acceptance criteria, which TAKHeLiPeD now meets (82.2% single-frame against the 66.5%
bar, +3.7 pp from ordered time); DR-SPAAM in its official streaming form; LFE retrained with its authors' recipe; and D4's one
unrun flag. Ten more experiments run **in parallel with the writing** rather than before it. The owner's one open decision is
whether A51's six single-seed arms are re-screened on the adopted recipe (~2 GPU-days) or reported labelled with the backbone
they ran on.

## A50 phase 1 item 7: the memory is a phantom filter, and it beats LFE-Peaks where nobody is (2026-10-09)

*keywords: A50, person-free, transferred, empty_scene.py, phantoms, furniture, 72533, LFE-Peaks 1.39, LFE-PPN 3.17, stale row, lidar_person_free, queue_a50_person_free*

`utils/queue_a50_person_free.py` ran the three seeds of the adopted recipe over all 72,533 person-free frames of FROG
`transferred` (the runner's first queue module; 2.2 h, three jobs, no retries). Every detection there is a false positive by
construction, so the measurement is a count.

| at threshold 0.3 | stage 2 alone | whole detector | published |
| --- | --- | --- | --- |
| FP/frame | 2.622 (2.834 / 2.587 / 2.446) | **0.740** (0.853 / 0.788 / 0.580) | LFE-Peaks 1.39, LFE-PPN 3.17 |
| frames with at least one | 67.2% | **34.1%** | 51.0%, 64.0% |
| recur within 0.3 m next frame | 84.9% | 92.2% | 88.3%, 77.6% (real people 99.8%) |
| FP/frame beyond the 10 s fade (41,797 frames) | 2.460 | **0.590** | -- |
| FP/frame at threshold 0.9 | 0.872 | **0.010** | "a third to a half of frames" |

**Stage 2 alone is worse than both published detectors here and the memory cuts it by 72%, to below LFE-Peaks.** No published
detector on this split reaches that. At 0.9 confidence the contrast is categorical: 0.9% of person-free frames hold a detection
against the published third to a half.

**It is not the stale-memory explanation.** 42.4% of person-free frames sit within 10 s of someone in view, inside the slots'
fade time, so a detection there can be a true memory rather than a phantom -- which is why `empty_scene.py` reports the
restricted count too. Beyond that horizon the reduction is **larger** (-76%), so the filtering is real.

**What survives is furniture**: the memory's remaining phantoms recur next frame 92.2% of the time against the candidates' 84.9%
-- it removes the unstable ones and keeps whatever looks like a person in every frame, the failure A41 and A44 could not remove
classically. That makes it a slot-creation question, now Tier 2 item 13.

**Found en route: a half-stale row is worse than a stale one.** `memory/performance-log.md` carried LFE-PPN at **1.94**
FP/frame, the pre-recalibration figure, while the *frames-with-a-phantom* cell beside it had been updated to the post-
recalibration 64.0%. The 2026-09-11 entry had recorded the correction to **3.174** and this table never took it. Fixed, and the
"with SORT" cell marked stale rather than reconstructed, since no corrected FP/frame was ever recorded for it.

**The three `gap` rows stay gaps.** They are `SpaceTimeCNN`, `FullScanTCN` and `TemporalUNet` -- the pre-TAKHeLiPeD line of work
-- so filling them with this detector's numbers would have mislabelled three models.

**The runner behaved as designed on the ROCm hang.** All three jobs wrote their results and then hung in teardown past
`os._exit`; each was waited out for its 600 s grace, killed, and recorded **DONE on its outputs rather than on its exit code**,
which is exactly the case `tests/test_run_queue.py` pins.

## A50 phase 1 closes, and the adopted recipe retires one of the four screening bars (2026-10-09)

*keywords:* A50, phase 1, A51, E13, a60_e13_static, screening bars, coasting precision, coast_per_frame, evaluation_segments, val windows, max_hours, EarlyStopping, slot_experiments --out, empty_scene.py, machine_is_free.py, triage

The owner asked for what was left of A50 and A51. Most of it was already answered elsewhere or made moot by the architecture
changing under it, which is now written into both items; three things were open and are closed here, and one measurement is left.

**The time budget is gone** (A50 phase 1 item 6, the owner's call of 2026-09-16, never executed). `--max-hours` is removed from
`step2`, `step3a` and `step3b`, and `EarlyStopping` no longer takes or checks hours. It only ever truncated a healthy run: A51's
first full-curriculum step 3a died inside its first chunk-length stage at the 6 h default, after which every A51 launcher passed
`--max-hours 24` by hand. A run is bounded by `--max-epochs` and `--patience`.

**Validation windows are placed where the annotations are** (the half of item 3 that was about validation, not feedback).
`evaluation_segments` now keeps only the window starts whose *scored* span holds an annotated frame, and spaces the budget over
those. Measured on the 2026-10-09 val cache (`a59_c1_recipe_dropout01_s0`, 64 windows of 30 s warm-up plus 60 s): **6 of 64
windows scored nothing**; afterwards 63 windows, **none empty**, and 10,091 scored frames against 9,924 -- more validation signal
from slightly less replay. It changes checkpoint selection for future runs and invalidates nothing already trained.

**`slot_experiments.py` takes `--out`** (A51's cheap open item): the hardcoded path destroyed E2's and E4's JSON on 2026-10-02.
The modes are now reached through an `EXPERIMENTS` table that the CLI's own choices come from.

**The adopted recipe has its own bars, and coasting has gone inert.** Read from A61's three E13 runs
(`a60_e13_static_s{0,1,2}/step3a_results.json`), the control every future stage-3 arm is judged against:

| metric | seed 0 / 1 / 2 | mean | three-sigma bar | B0's old bar |
| --- | --- | --- | --- | --- |
| AP | 86.35 / 85.77 / 85.49 | 85.87 | **1.32** | 1.45 |
| AP at 0.3 m | 85.24 / 84.65 / 84.16 | 84.68 | **1.62** | 1.29 |
| recall at matched false positives | 96.00 / 95.86 / 94.56 | 95.47 | **2.37** | 0.86 |
| coasting precision | 33.3 / none emitted / 10.0 | -- | **gone** | 8.19 |

The memory emits **0.0012 / 0.0000 / 0.0020 coasting reports per scored frame** against B0's 0.176 -- about a hundredth, with
seed 1 emitting none, which is why its precision is undefined. Nothing is broken: stage 2's recall at its own operating point is
95.47%, so there is almost nobody left for a slot to report alone. **Screening is now three bars, not four**, the tightest being
AP's 1.32; **O2 and S2, the arms that aimed at coasting, have nothing left to aim at**; and any sentence about the coasting head
must name the backbone it was measured on. The rescoring shift is -3.61 / -3.88 / -4.17 logits on candidates covering a person
against -0.54 / -0.38 / -1.43 on the rest (A50 phase 1b item 2), the same whole-distribution shift as before.

**What is left of A50 is one measurement**, item 7's person-free frames, and `utils/empty_scene.py` is the script for it: FROG
`transferred`'s 72,533 frames where nobody is present, counted for candidates and memory at five thresholds -- per frame, per
second, the share of frames holding any, and the share recurring within 0.3 m in the next frame, against LFE-Peaks' 1.39 and
LFE-PPN's 1.94. `load_split` takes a FROG mode (the `official` cache names are byte-identical, so no 1 GB cache is rebuilt).
**It also splits by time since anyone was in view**, because the three raw recordings interleave populated and person-free
frames and a 10 s memory can report someone who has just left -- an explanation a static detector cannot have.

**The free-machine guard was lying.** `machine_is_free.py` carried a hand-written list of six script names and reported a free
machine while `empty_scene.py` was replaying on the GPU. A stale list is indistinguishable from a free machine, which is the one
failure that script must not have, so the rule is derived from `utils/` instead: any python process running a `.py` that exists
there, plus `-m pytest`. Verified against the live job it had just missed.

## A57 done: one typed queue runner in place of eight copied PowerShell launchers (2026-10-09)

*keywords:* A57, run_queue.py, Job, Outcome, BLOCKED, queue module, progress.txt, QUEUE, training guard, test gate, spawn seam, run_queue_ladder_a61.ps1

`utils/run_queue.py` replaces the ~60-line launcher that had been hand-copied eight times, and every incident in `TODO.md` A57's
table traced to a copy rather than to the research. A queue is now a committed Python module that builds `Job`s and calls `main`,
so the exact queue behind a result outlives its session -- the gap that once left T1's and T2's settings unverifiable.

- **`Job`**: `name`, `cli`, `outputs`, `needs`, `cwd`. **`Outcome`**: `DONE` / `SKIP` / `FAILED` / `BLOCKED`.
- **Dependencies are explicit**, so a failed job's dependents are recorded `BLOCKED` and never start. The copies waited for an
  *idle machine* instead, and a chain that stops on failure makes the machine idle -- which is how S3 once trained ahead of a
  failed test gate.
- A job whose `outputs` all exist is skipped without spawning; a job with no outputs is judged by its exit code, which is how the
  test gate works; three attempts with `--resume`, but a **training guard in the log stops it after one**; every attempt keeps its
  own log, so a relaunch no longer overwrites the earlier stage's epoch history.
- The progress lines are byte-identical to the PowerShell format, so the watchers work unchanged; `QUEUE <job> BLOCKED, <dep> did
  not finish` is the one new line.
- **40 tests**, the process spawn behind a seam, so none needs a GPU -- including `spawn_process` itself against real processes.
  Three mutations (never block, retry past a guard, ignore the skip rule) were each caught by the suite.
- One behaviour deliberately improved rather than ported: the PowerShell loop slept a fixed 30 s between checks, so a job that
  finished in a second still cost 30; the runner waits on the process with a 30 s timeout.

## A61 done: the learned memory still beats a tuned tracker on the adopted recipe, by +2.36 pp (2026-10-09)

*keywords:* A61, A56, E13, E14, E15, ByteTrack, CBMOT, SORT, rule scorer, tracking_baselines summary, a61/summary.json, recall at matched FP, GOSPA, lidar_queue

The second half of `lidar_queue` finished at 19:02 (three replays of `rule`, `track`, `arms`, `report`, then `summary`; 12.3 h; each `arms` step hung in ROCm teardown after writing everything and was stopped by the chain 10 min later, as designed). The whole chain: 26 steps, 18.2 h, no retries; task deleted.

| what reads the stage-2 candidates | AP, 3 seeds | vs candidates, per seed | recall at matched FP | GOSPA |
| --- | --- | --- | --- | --- |
| nothing | 82.19% | -- | 94.13% | 0.494 |
| SORT `min_hits` 1 | 82.31% | +0.14 / +0.12 / +0.12, all above zero | 94.19% | 0.494 |
| ByteTrack as published | 81.17% | -0.94 / -1.11 / -1.00 | 94.29% | 0.494 |
| the memory's bookkeeping with CBMOT's score (E15) | 72.02% | -10.55 / -9.80 / -10.14 | 94.58% | 0.619 |
| ByteTrack + CBMOT, tuned on val (E14) | 83.51% | +1.30 / +1.35 / +1.32, all above zero | **95.56%** | 0.493 |
| **learned state, no attention (E13)** | **85.87%** | **+4.13 / +3.66 / +3.26; over E14 +2.84 [+1.99, +3.73] / +2.31 [+1.28, +3.36] / +1.94 [+0.94, +3.01]** | 95.47% | **0.454** |

- **Predictions** (`TODO.md` A61): E13 - E14 +1.0 to +2.5, at least two intervals above zero -- held (+2.36, all three); E15 below the candidates -- held; the lead on occluded people, first 0.5 s and far people -- held (+3.4 / +3.4 / +3.2 recall at matched FP); E14 +1.5 to +2.5 over the candidates -- missed low (+1.32).
- **Changed against A56: recall at the candidates' false-positive rate is now a tie** (E13 95.47%, E14 95.56%; old recipe +1.0). The memory trades visible, standing and near people for occluded, new and far ones; its lead is ranking over the whole curve, which GOSPA confirms (0.454 against 0.493).
- Docs: `memory/experiments/2026-09-tracking-claims.md` (new section), `memory/performance-log.md` and `memory/slot-design-evidence.md` (the new recipe leads, the old kept beside it), `docs/RESEARCH.md` §7.4.

## A59's attribution ladder scored: not the input, not the size, not the protocol (2026-10-09)

*keywords:* A59, F1, W, R, attribution ladder, stage2_probe_a59_ladder.json, a59_f1_lfe_input, a59_w_lfe_width, a59_r_drspaam_recipe, step2_calibration.last.pth, why do we beat LFE

The first half of `lidar_queue` finished at 06:45 (gate, nine trainings, averaging, the probe; 6 h, no retries). Three seeds per arm, `nms-mean`, every 5th annotated test frame, against C1 (81.95% as selected, 82.20% averaged):

| arm | parameters | selected | averaged |
| --- | --- | --- | --- |
| F1 LFE's input alone | 976,419 | 81.47% (-0.48) | **82.34% (+0.14)** |
| W LFE's widths | 63,891 | 80.66% (-1.29) | 80.70% (-1.50) |
| R DR-SPAAM's protocol, last weights | 976,707 | 80.11% (-1.84) | -- |
| R, best on val | 976,707 | 81.35% (-0.60) | -- |

- **Every prediction missed in the same direction**: each arm lost less than predicted (F1 -1 to -3, W -2 to -4, R 75-78%); only R's val-best (within 1.5) held.
- **W's loss is the far range** (8-10 m recall -4.8 pp at LFE-Peaks' FP rate) and it gains on close pairs (+2.4); **R's is selection**: -1.24 of its -1.84 is testing the last weights, which report 1.98 FP and 0.38 FN per frame at 0.3 against C1's 3.11 and 0.20.
- **For the paper**: input, size and DR-SPAAM's protocol each cost at most ~1.8 pp and every arm scores 80-82% against LFE-Peaks' 65.1%; what is left of the lead is the network's form and the training recipe (`memory/static-detector-diagnosis.md`, `docs/PAPER.md`). LFE retrained with its own code (A59 phase 3b) is the remaining converse test.
- **Correction, same session**: two progress messages quoted C1's operating point as ~2.4 FP and ~0.30 FN per frame; those were F1 seed 0's figures. C1's are 3.11 and 0.20 (three-seed means at 0.3), as above.
- **W's ms/frame, measured after A61 the same evening** (`results/a59_w_latency.json`, a scratch timing of network plus `nms-mean` decoding on 2,000 test scans): **no faster than C1** -- 11.14 against 11.46 ms on 4 CPU threads, 9.04 against 8.80 on the GPU. Per-scan time is overhead, not width.

## A59's attribution ladder and A61's tracker re-run: built and launched (2026-10-09)

*keywords:* A59, A61, F1, W, R, attribution ladder, input_features, lfe_features, LFE_NEAR_M, INPUT_CHANNELS, channels, network_shape, exponential_schedule, select_test_weights, --select, step2_calibration.last.pth, parse_arms, summarise, --arms, tracking_baselines summary, run_queue_ladder_a61.ps1, lidar_queue

**Owner's go-ahead** on both proposals of the A60 report. Plans, predictions and controls are in `TODO.md` A59 (the ladder) and A61 (new); nothing is measured yet.

- **Ladder flags, all opt-in** (`step2`): `--input-features lfe` (`CalibrationNetwork(input_features=)`, `lfe_features`: LFE's `_normalize_scan`, `1 - r/10` clipped at 0.2 m, as the only input channel; F1), `--channels 8 16 32` (LFE's widths, 63,891 parameters; W), `--schedule exponential` (`exponential_schedule`, DR-SPAAM's per-iteration decay) and `--select last` (`select_test_weights`: the test scores the last weights, saved as `step2_calibration.last.pth` beside the val-best; R). `calibration_network` now rebuilds the input and the widths from a checkpoint's args through `network_shape`, so the averaging, the probe and stage 3's caches read the arms without flags; checkpoints without those args rebuild at the defaults, as before.
- **Tracker re-run tooling** (`tracking_baselines.py`): `--arms ARM:RUN` on `arms`, `report` and `merge` replays memories other than A56's hardcoded A51 runs (`parse_arms`); `summary` lines up several replays' reports by arm (`summarise`), counting each bootstrap pair's intervals above zero. A61 needs one replay per seed, because each A60 memory was trained on its own stage-2 seed.
- **Control for the ladder: C1, not retrained.** Today's trainer draws the same permutation and mirror flips as the code C1 trained on, and loads C1 bit-identically (A60's fixture), so a fresh control would differ by GPU noise only.
- **Tested**: 19 new tests (5 network, 9 trainer, 5 baselines), red first; the full suite **681 passed**. A real-data smoke run with every ladder flag at once (63,819 parameters: W's widths with F1's one-channel stem) decayed the rate to exactly 1e-6, saved `last.pth`, and the probe's `nms-mean` AP on the rebuilt weights equalled the trainer's own (33.41%, every 500th frame). Nothing previously trained or cached is invalidated: the default network, its initialisation order and every existing checkpoint's rebuild are unchanged.
- **Launched 2026-10-09 00:48** as one resumable chain, `checkpoints_three_horizon/run_queue_ladder_a61.ps1` (Task Scheduler `lidar_queue`, hidden window; `progress.txt` lines start `QUEUE`): the test gate, F1/W/R at three seeds, averaging, the probe (`results/stage2_probe_a59_ladder.json`), then A61's `rule`, `track`, `arms`, `report` per seed into `checkpoints_three_horizon/a61/s{0,1,2}` and `summary`. ~4 h for the ladder, ~10 h for A61; a failed ladder step does not block A61.

## A60 done: the coarse convolution is removed; stage 2 reads one frame (2026-10-08)

*keywords:* A60, removal, CausalTemporalConv, TemporalCache, CalibrationState, align_sectors, POSE_CHANNELS, coarse_lags, clip_length, clip_frames, clip_batch, mirror_clips, mirror_scans, frame_batch, context, fold_legacy_calibration, LAST_TEMPORAL_COMMIT, e76aa94, no_temporal_stage2.pt

**Owner's call**, on A60's verdict (entry below). **The last commit with the coarse convolution is e76aa94**; `memory/rejected-ideas.md` lists every symbol and flag it held, the checkpoints that need it, and the restore (`git show e76aa94:<path> > <path>`).

- **Library** (`three_horizon.py`): `align_sectors`, `TemporalCache`, `CausalTemporalConv`, `CalibrationState`, `POSE_CHANNELS` and `CalibrationNetwork.step` are gone; `CalibrationNetwork(...)(ranges, prior=None)` takes one scan `[B, N]`, no pose, no state. The bottleneck's 3-tap convolution across sectors, which a one-lag `coarse.conv` always was, is now `context`.
- **Old checkpoints**: `fold_legacy_calibration` renames a no-temporal checkpoint's `coarse.conv.*` to `context.*` -- every stage-3 result rests on such checkpoints -- and refuses one whose convolution read past frames, naming e76aa94 (`LAST_TEMPORAL_COMMIT`).
- **Trainer and scripts**: `clip_length`, `clip_frames`, `clip_batch`, `mirror_clips` and `step2 --coarse-lags/--clip-frames` are gone; `frame_batch` and `mirror_scans` replace the clip and pose plumbing; `network_candidates`, `evaluate_calibration` and `joint_step` run single frames; `JointState` carries the slots alone; step 2's results drop the shuffled-history score, which meant nothing without history. `duration_curve.py`, `stage2_probe.py`, `average_checkpoints.py`, `measure_prior_share.py` and `tracking_baselines.py latency` follow.
- **Tests first**: `TestStage2WithoutTime` loads `tests/fixtures/no_temporal_stage2.pt` -- a small no-temporal network, its inputs and its outputs, written by e76aa94's code (`tests/fixtures/make_no_temporal_stage2.py`) -- and requires **bit-identical** outputs with and without a memory prior, and the refusal; obsolete tests of the alignment, the temporal convolution, streaming and clips are removed. **662 tests pass.**
- **Checked on real data**: a real averaged stage 2 (`a59_c1_recipe_dropout01_s0`) gives **bit-identical** outputs on 500 FROG test scans and identical stage-3 candidates over a whole 373-frame recording, old code against new, on the CPU. On the GPU the two graphs pick different kernels: logits differ by at most 4e-5 on 2,000 scans, and the three averaged stage 2s re-score at 82.284826 / 82.082105% (identical to the recorded values) and, for seed 2, 82.220811% in one process and 82.220823% in another, against the recorded 82.220805% (which e76aa94's own code reproduces). Within one process every pass repeats exactly; across processes the GPU's kernel choice moves the seventh digit, so stage-3 numbers re-run on the new code may differ there and nowhere meaningful. A smoke step 2 trains on the new defaults; the whole detector streams at 20.79 ms per frame (p99 24.08) on the CPU at 4 threads, with a 138 KB state.

## A60 verdict: drop the coarse convolution; the retrained detector scores 85.87% (2026-10-08)

*keywords:* A60, coarse convolution, CausalTemporalConv, verdict, drop, first-second recall, coarse_e2e.py, a60.json, phase 3a, 85.87

Six E13 stage-3 runs on A59's adopted recipe, the no-temporal and the default (coarse) stage 2 at three seeds each, scored by `utils/coarse_e2e.py` (tests 9/9 and the full suite 673/673 first; it reproduces every run's recorded test AP exactly), FROG test, every 5th annotated frame for AP:

| | AP | AP@0.3 | recall at 1.056 FP/frame | first-second recall |
| --- | --- | --- | --- | --- |
| no-temporal | **85.87%** | **84.68%** | **85.26%** | 75.11% |
| default (coarse) | 85.01% | 83.82% | 84.61% | 75.71% |

By the rule written before the runs -- keep only past +1.5 pp AP or +2 pp first-second recall -- **drop**: -0.87 and +0.59. Both predictions held. The coarse arm's stage-2 candidates are not worse (82.1-82.5% against 82.1-82.2%), but the memory adds less on top of them (+2.4 to +3.0 against +3.3 to +4.1). The no-temporal half is A59 phase 3a: **the full detector on the new recipe scores 85.87%, against 83.86% (E13) on the old one.** The removal itself awaits the owner.

## A60 stopped by input to its console, resumed hidden (2026-10-07)

*keywords:* A60, 0xC000013A, STATUS_CONTROL_C_EXIT, Kernel-Power 566, InputHid, -WindowStyle Hidden, watch_a59.sh, LAST_START

The A60 chain and its stage-3 run ended at 2026-10-06 21:49 with no traceback and no driver event, the task's result `0xC000013A`, in the second of a Kernel-Power 566 session transition with `Reason InputHid` -- the signature of 2026-10-05's 18:14 stop too. The chain ran in a visible console (`schtasks /it`), so a keystroke into it or a close of it is the likely mechanism (`memory/gotchas.md`, new entry; `memory/long-running-tasks.md`'s template now launches hidden). Relaunched 2026-10-07 05:17 with `-WindowStyle Hidden`: the four finished steps were skipped and `stage3_static_s0` resumed at its 10 s stage. The event watcher now reads a chain's finish only after its latest START line, since the earlier STOPPED line would otherwise have switched its hang check off.

## A60's scoring written, tests not yet run (2026-10-06)

*keywords:* A60, coarse_e2e.py, trajectory_ages, first_second_recall, verdict, first-second recall, build_tracks, copied labels

Owner's call: write it now and test it once A60 frees the machine. `utils/coarse_e2e.py` saves each A60 memory's test frames, links annotations into trajectories (no identities, no model; FROG's copied labels excluded), and applies A60's rule: drop the coarse convolution unless the default's mean AP beats the no-temporal mean by more than 1.5 pp or its first-second recall by more than 2 pp, every recall at the no-temporal arm's false-positive rate at 0.3. It reuses A56's `PrecisionRecall`, `subset`, `stride_frames`, `save`/`load` and coverage rule rather than repeating them, and replaces A60's earlier plan to define a first second by the memory's own slots, which would have measured the model by itself. 9 tests in `tests/test_coarse_e2e.py`, **written, not run**; ruff clean.

## Dropout 0.1 and weight averaging adopted; averaging moves into the repo; A60 launched as resumable steps (2026-10-06)

*keywords:* A59, A60, average_checkpoints.py, WINDOWS, window_members, average_window, step2_calibration.averaged.pth, --dropout default 0.1, run_a60.ps1, lidar_a60, resumable steps, watch_a59.sh

**Owner's call**: A60's stage 2 is C1 averaged, so dropout 0.1 and weight averaging join stage 2's recipe. **Tests first** (4 in the new `tests/test_average_checkpoints.py`, 1 extended in `TestStage2Recipe`): `step2 --dropout` defaults to 0.1, and `utils/average_checkpoints.py` averages a run's periodic checkpoints over each of the six windows the A59 numbers used, chooses on val's 2,000-frame sample, and saves `step2_calibration.averaged.pth` with `averaging.json`. **Checked on real runs before use**: on the three C1 seeds it chose the scratch analysis's windows (1.00-2.00, 0.25-2.25, 1.00-2.00) with identical val AP to six decimals. 664 tests pass.

**A60 launched as named steps** (owner: cancellable and resumable): `stage2_default_s{0,1,2}`, `average_default`, `stage3_{static,default}_s{0,1,2}`; each skipped when its output exists, training resuming from its state, every attempt logged under its own timestamp so a relaunch overwrites nothing (`checkpoints_three_horizon/run_a60.ps1`, task `lidar_a60`). The event watcher now takes a prefix, a chain, a stale limit and a log glob (`watch_a59.sh A60 '' 60 'a60_*'`), since a stage-3 cache build is silent for ~30 min.

## A59 combination run: the recipe plus dropout, averaged, is the best stage 2 at 82.2% (2026-10-06)

*keywords:* A59, combination, c0_recipe, c1_recipe_dropout01, weight averaging, a59_save_average.py, candidate_cache_path, run_a60.ps1

No-temporal stage 2 on the new defaults, three seeds per arm, `nms-mean`, every 5th annotated test frame (same frames as every A59 arm):

| | AP | averaged | AP@0.3 | close pairs | 8-10 m |
| --- | --- | --- | --- | --- | --- |
| control (old recipe) | 79.75% | -- | 77.46% | 83.6% | 78.2% |
| mirror alone | 81.49% | -- | 79.29% | 79.4% | 81.4% |
| C0: mirror + nearest labels | 81.60% | 81.90% | 79.52% | 81.6% | 80.2% |
| **C1: C0 + dropout 0.1** | **81.95%** | **82.20%** | **80.43%** | **86.0%** | **81.7%** |

All four written predictions held; by the rule written before the runs, C1 averaged is the stage 2 for `TODO.md` A60, the owner to confirm. Also: `candidate_cache_path` (2 tests) keys stage-3 caches by run directory, since every run writes identically named checkpoints; `checkpoints_three_horizon/a59_save_average.py` saves a run's val-chosen average as a loadable checkpoint and re-scores it; `run_a60.ps1` is prepared, not launched. 660 tests pass.

## A59 recipe made the default; F3's code removed (2026-10-06)

*keywords:* A59, decode_frame, nms_decode, beam_votes, NMS_DIST_M, VOTE_FLOOR, build_parser, --label-assign, --mirror, --no-mirror, -nms-mean cache key, median_windows_deg removed, n_beam_features removed

**Tests first** (3 new in `TestStage2Recipe`, red before the change): `step2` now defaults to `--label-assign nearest` and `--mirror`, both still switchable (`--label-assign last`, `--no-mirror`) for ablations; and stage 2 decodes through `decode_frame`, which is `nms_decode(average=True)` -- moved, with `beam_votes`, from `utils/stage2_probe.py` into `utils/train_three_horizon.py` -- in `decode_candidates` (stage-3 caches, the joint step, `tracking_baselines.py`) and in `evaluate_calibration` (stage 2's val selection and test). Candidate caches are keyed `-nms-mean`, so every one rebuilds. `main()`'s parser is now `build_parser()`, so the defaults are testable. 658 tests pass.

**What no longer reproduces from the command line:** every stage-2 and stage-3 number before this entry was decoded by the vote grid (`decode_votes`, still there and still the probe's `vote-grid` row) and trained without mirroring; re-scoring an old checkpoint now gives its `nms-mean` figure.

**F3 removed, never committed, so recorded here:** `beam_features(median_windows_deg=)` took a list of window widths, computed `r - median` over each, and appended every window after the first as an extra channel after the ten (so the default network was unchanged); `n_beam_features(windows)` gave the stem's width; `CalibrationNetwork(median_windows_deg=)` stored it and `calibration_network` rebuilt it from the checkpoint's `args`; `step2 --median-windows-deg` set it; four tests pinned it, including one where a 15-degree object vanishes from the 11-degree channel and shows in a 33-degree one. `library/follow_the_drow/detectors/three_horizon.py`, `tests/test_three_horizon.py` and `utils/measure_prior_share.py` are back to their committed state; the `a59_f3_median33_s*` checkpoints no longer load.

## A59 decided: stage 2 adopts the `nms-mean` decoder, mirror augmentation and nearest-person labels; F3 dropped (2026-10-06)

*keywords:* A59, adoption, nms-mean, --mirror, --label-assign nearest, F3 rejected, median_windows_deg, stage 2 recipe

**Owner's call**, on the evidence in the entry below: the three become stage 2's recipe, and the second median channel (F3, +0.48 pp) is rejected (`memory/rejected-ideas.md`). Dropout 0.1 and weight averaging stay leads for the combination run. The coarse-convolution end-to-end test (`TODO.md` A60) runs before stage 3 is retrained, so that the retrain need not carry a module that may be dropped. Making the defaults and removing F3's code follow in their own entry.

## A59: what stage 2's score is made of -- the decoder is a handicap, the gain is far and isolated people, close pairs are a deficit (2026-10-05)

*keywords:* A59, stage2_probe.py, decoder, vote grid, nms_predicted_center, nms-mean, peaks decoder, close pairs, make_targets, --label-assign, --label-radius, label overlap, duration_curve_test.json, weight averaging, SWA, 2DLaserPeopleBenchmark, SegDataGenerator shuffle, coarse convolution

The owner asked two questions: are the coarse convolutions worth keeping beside the slots, and why does stage 2 alone, with no temporal input, beat DR-SPAAM and LFE -- and is it below its potential? Everything below is evaluation only, on existing checkpoints, except where marked.

**Built, tests first:** `utils/stage2_probe.py` (12 tests in `tests/test_stage2_probe.py`): one network's per-beam outputs through seven decoders -- the vote grid and its probability-weighted variant, the official DR-SPAAM `nms_predicted_center` at 0.5 m and 0.30 m and a variant placing each detection at the weighted mean of the votes it suppresses, LFE-Peaks' `find_peaks` at the endpoint and at the vote -- and recall by range and crowding at LFE's own false positives per frame. `decode_votes(weighted_avg=)` passes the existing `votes_to_detections` option through, default off.

**Also built, for the training arms the owner approved** (all opt-in, defaults reproduce every earlier run and checkpoint): `step2 --label-assign {last,nearest} --label-radius` (`make_targets(nearest=)`, the official DR-SPAAM assignment; 4 tests), `step2 --median-windows-deg` (`beam_features(median_windows_deg=)`, `n_beam_features`, extra `r - median` channels appended after the ten; 3 tests), and `step2 --mirror` (`mirror_clips`, `mirror_people`, exact on FROG's asymmetric grid; 3 tests, including that mirrored targets are the mirror of the targets). Queued as three chains through the Task Scheduler: a three-seed no-temporal control with a checkpoint every 0.25 epoch, then L1/L2, then F3/M/D, each followed by weight averaging and the probe (`TODO.md` A59).

**A tooling lesson, written down in place of an apology:** `pytest` was run three times beside a live training run, against `memory/long-running-tasks.md`, once after the guard had said busy (`;` where `&&` belonged); the third hit an internal `MemoryError` at the end of a passing run. Nothing died, but `memory/gotchas.md` already says why it was wrong -- the limit is commit, not RAM, so nothing that imports torch runs beside a training job; the decoder probe, started two minutes before the control chain, broke the same rule until it finished. The later chains run strictly one after another, and the F3/M/D chain runs the full suite itself, as its gated first step, when nothing else is running.

**Results** (FROG `official` test, every 5th annotated frame, `results/stage2_probe.json`):

| | no-temporal, 3 seeds | default, 3 seeds |
| --- | --- | --- |
| vote grid (ours) | 77.27% | 76.75% |
| DR-SPAAM NMS, 0.5 m | **79.88%** | **79.42%** |
| LFE peaks / peaks at the vote | 71.26 / 74.10% | 71.14 / 73.09% |

- **The decoder is a handicap**: NMS wins on all six checkpoints, +2.33 to +3.03 pp, paired. Our LFE reproductions on the same frames: 65.11 / 69.44%, against the published 65.6 / 69.2.
- **Far and isolated people carry the gain over LFE; close pairs are a deficit.** At LFE-Peaks' 1.78 FP/frame, 8-10 m recall 50.3% -> 83.2%, people 1-2 m from the next 78.6 -> 92.8%, and people within 0.6 m of another 83.2 -> 59.9%; at 1-4 m stage 2 is 1-3 pp worse.
- **The default architecture's coarse convolution changes no row** by more than ~2 pp, and ties at the AP level under both decoders.
- **The gap to LFE-Peaks on fixed weights**: 65.11% -> our network through LFE's own peak decoder 71.26 (+6.15) -> peaks at the vote 74.10 (+2.84) -> vote grid 77.27 (+3.17) -> NMS 79.88 (+2.61); half the lead is the output head.
- **The depth-centring channel is blind up close**: over its 11 deg window a person's own beams read `r - median` ~0 for 98% of people at 0-1 m and 41% at 1-2 m; over 33 deg the signal survives (A59 arm F3). Its cost to recall is unmeasured: the 1-4 m deficit it was first matched to is the vote grid's.
- **The three-seed no-temporal control on current code** (`a59_static_ckpt_s*`, a checkpoint every 0.25 epoch): 77.29 / 77.18 / 76.60%, mean 77.02 (sd 0.37), every annotated test frame, against the earlier 77.23.
- **Label assignment, trained at three seeds** (`a59_l1_nearest_s*`, `a59_l2_nearest_s*`): L1 (nearest within 0.6 m) 78.64 / 78.66 / 77.68%, L2 (within 0.4 m) 77.37 / 78.55 / 78.56%, +1.31 and +1.14 pp over the control through the vote grid, every annotated frame. Under `nms-mean` (every 5th frame): control 79.75, **L1 80.89 (+1.14)**, L2 80.48 (+0.73); AP@0.3 +1.3 / +1.6. Every L1 seed beats every control seed under both NMS decoders, yet no arm clears the ~1.5 pp three-seed bar. Predicted close-pair recall >= 70% missed (64.7% through the vote grid). Recommended for adoption, the owner's call.
- **Weight averaging at three seeds, scored**: predicted +0.5 to +1.0 on every seed, missed -- mean +0.54 over nine runs, two of them negative (control +0.96 / -0.27 / +0.82, L1 +0.36 / -0.75 / +0.26, L2 +2.09 / +0.89 / +0.48). L2 averaged reaches 79.30% (79.39 / 79.38 / 79.12) through the vote grid, +1.69 over the averaged control, a comparison not registered in advance.
- **Phase 1 decoder choice, scored**: `nms-mean` (NMS placing each detection at the probability-weighted mean of the votes it suppresses) wins on val on all six checkpoints and gains +2.60 to +3.15 pp on test (no-temporal **80.07%**, AP@0.3 77.94). Predicted, and missed: that NMS would keep the close-pair deficit. It recovers it (59.9% -> 82.3%, LFE-Peaks 83.2%) and erases the 1-4 m deficit, so **both were the vote grid merging neighbours**; it gives some back at 8-10 m (83.2 -> 77.7%).
- **`make_targets`' label rule was first read as the close-pair cause** (the decoder result above says otherwise): the last-listed person within 0.6 m wins a beam, so on FROG train 3.3% of positive beams vote for someone other than their nearest (6.9% sit in two radii); at DR-SPAAM's nearest-within-0.35 m it is 0.4%.
- **Training duration**: the no-temporal seed 0 scores 75.19 / 77.60 / 76.54 / 74.45 / 72.36% at epochs 0.25 / 1 / 2 / 3 / 4 on every annotated test frame (`step2_static/duration_curve_test.json`), precision 43.0% -> 69.4%, recall 90.5% -> 76.5%.
- **Weight averaging**: the mean of epochs 0.5-1.5, chosen on val, scores 78.53% against the selected checkpoint's 77.58% (stride 5); every window tried gains +0.52 to +0.95 (`results/swa_step2_static.json`, a scratch analysis).
- **Neither capacity against the cutouts nor compute**: 976,707 parameters against our DROW3 port's 1,518,918 and DR-SPAAM's 1,978,054; the selected checkpoint has seen ~108k scans against DR-SPAAM's 541k.
- **Read from the FROG authors' code** (`robotics-upo/2DLaserPeopleBenchmark`): LFE's training code is published, contrary to `TODO.md` A32c (corrected); its segmentation trainer never shuffles (`SegDataGenerator.__call__` iterates the unshuffled list) and early-stops on loss over frames interleaved with its training frames; LFE-PPN trains only on scans with at least 3 people; DR-SPAAM's FROG numbers come from the official repository's NMS decoder.

**Further arms, and a pause** (vote grid, every annotated test frame, control mean 77.02): F3 77.20 / 78.15 / 76.82% (+0.37, noise); M 78.13 / 78.38 / 77.01% (+0.82, under the bar; selected later, epochs 2.25 / 2.25 / 2.75); D seeds 0-1 78.96 / 78.07%. **The owner paused the tests at ~18:14**; D seed 2 resumes from epoch 1.75 and F3/M/D's averaging and probe are still to run (`checkpoints_three_horizon/run_a59_arms_resume.ps1`, `TODO.md` A59).

**Resumed 2026-10-06 04:17 and finished 05:59.** D 78.96 / 78.07 / 77.91% (+1.29; seed 2 resumed, a different draw); averaged, D 79.50% (+1.89 over the averaged control, every seed gaining), M 78.39, F3 78.13. **Under `nms-mean` (every 5th frame): control 79.75, L1 80.89, L2 80.48, F3 80.23, M 81.49 (81.48 / 81.95 / 81.03), D 80.45.** Mirror augmentation is the one arm that clears the three-seed bar, +1.74, every seed above every control seed; it gains at 8-10 m (+3.2 pp recall) and gives some back on close pairs (-4.2). F3 missed its prediction (1-4 m recall +0.6 against +1 to +3). Twelve arm-decoder comparisons, so the rest are leads. Recommended: adopt `nms-mean`, mirror and nearest labels, drop F3, run the combination next (`TODO.md` A59).

**Coarse convolution:** owner's call -- test it end to end and drop it if it earns nothing, recording where it lived; planned with a written decision rule as `TODO.md` A60, which replaces the §D item.

**Docs:** `memory/static-detector-diagnosis.md` (new section), `memory/detector-architectures.md`, `memory/performance-log.md` (the "~38x the compute in our favour" caveat does not apply to TAKHeLiPeD: it sees fewer scans than any baseline), `memory/commands.md`, `memory/keywords.md`, `TODO.md` A59 (new), A32c and the §D coarse-horizon item (unblocked, planned).

## A58: a merge radius on the memory's output buys nothing -- negative (2026-10-05)

*keywords:* A58, merge radius, NMS, double report, merge_detections, drop_duplicate_annotations, double_report_anatomy, tracking_baselines.py merge, oracle bound, coasting slot beside a candidate

The owner approved testing the double reporting found while closing A56 (entry below) as a precision lever. Predictions were written into `TODO.md` A58 before the run.

**Built, tests first** (628 -> 636 tests): in `utils/tracking_baselines.py`, `merge_detections`, a vectorised copy of `lfe_detector._merge_nearby`'s greedy merge pinned to it by an equivalence test over random frames with tied scores; `drop_duplicate_annotations`; `double_report_anatomy`, which splits each double report into candidate + candidate, candidate + slot and slot + slot; and the `merge` subcommand (radius per output chosen on val from none / 0.10-0.60 m, test scored once, block bootstraps on the official and the de-duplicated labels, about 10 minutes on 6 torch-free workers).

**Result: no output gains** (FROG test, every 5th annotated frame):

| output | val chooses | test AP none -> chosen | at 0.30 m | doubled people at matched FP | oracle |
| --- | --- | --- | --- | --- | --- |
| candidates | 0.50 m | 77.66 -> 77.60 | 77.66 | 1.7% | +0.38 |
| E14 ByteTrack + CBMOT | none | 80.96 | 80.51 | 6.5% | +0.52 |
| E13, 3 seeds | 0.10-0.15 m | -0.04 to -0.05 | -0.56 to -0.63 | 12.4-14.7% | +0.43 to +0.55 |
| B0 3 seeds / E10 2 seeds | none | unchanged | -0.5 to -0.6 | 8.4-11.5% / 4.9-5.4% | -- |

- **Why**: a double report's two detections are a median 0.32 m apart, a neighbour's distance, so any radius that removes them removes close neighbours too. 85-87% of E13's are a candidate with a coasting slot beside it, and they fire at the matched-FP threshold (0.14-0.17), low in the ranking.
- **The oracle** (`checkpoints_three_horizon/a56/merge_oracle.py`, a scratch analysis, not in the repo) removes every detection that matches nobody but lies within 0.5 m of a person. It buys about +0.5 pp and the same for E14, so the most a fix inside the model could buy is +0.26-0.41 pp (slots only).
- **The headline is robust to merging**: E13 over E14 with both merged is +2.90 / +3.16 / +2.51, all intervals above zero; +2.75 / +3.02 / +2.37 on de-duplicated labels.
- **Predictions**: (1) missed, no 0.2-0.4 m radius and no gain; (2) held, vacuously; (3) held, candidate + slot dominates; (4) held, +2.51 at the weakest seed.

**Docs:** `memory/rejected-ideas.md` (new entry), `TODO.md` A58 closed, `memory/slot-design-evidence.md`, `memory/experiments/2026-09-tracking-claims.md`, `memory/commands.md`, `memory/keywords.md`.

## A56 close-out: the "unlinked annotations" are FROG duplicate labels, and the memory double-reports (2026-10-05)

*keywords:* A56, duplicate labels, duplicate annotations, circles, unlinked annotations, double report, de-duplicate, coast_targets, frog_16-41_test.h5, merge radius on memory output

A56's last unblocked item: what are the 2,262 test annotations that link to no trajectory, which the memory "finds" 39% of against the candidates' 9.7%?

**They are duplicate labels.** Counted in the raw `.h5` files, independent of every loader: `frog_16-41_test.h5` holds **2,254** circles that repeat another circle of the same frame in all six columns (1.47% of 153,655), in runs over consecutive frames (54 runs, median 16 annotated frames) across 19 of 47 recordings; `train_val` has 285 (0.06%), the extras 814-1,979 (0.6-1.55%). They are the report's 2,254 trajectory-less annotations one for one: its reverse lookup from trajectory point to annotation takes the first of two identical positions. The other 8 of the 2,262 are single-point trajectories with no speed. `dataset-properties.md`'s "2,254 collided with another person" round-trip shortfall was the same set, now explained.

**The memory's 39% was double reporting**: a copy can only be matched by a second detection beside the original. At matched false positives, the share of people with a second detection within 0.5 m:

| | AP official | AP de-duplicated | second detection |
| --- | --- | --- | --- |
| candidates | 77.66 | 78.70 | 1.7% |
| E14 ByteTrack + CBMOT | 80.96 | 81.90 | 6.5% |
| E10 (2 seeds) | 81.94 | 82.86 | 4.9-5.4% |
| E13 (3 seeds) | 83.86 | 84.60 | 12.5-14.8% |
| B0 (3 seeds) | 83.34 | 84.14 | 8.4-11.5% |

**Every A56 claim survives the clean labels**: E13 over E14 +2.74 [+1.85, +3.68] / +3.02 [+2.01, +4.14] / +2.36 [+0.91, +3.66] (official +2.95 / +3.20 / +2.56), E14 over the candidates +3.19 [+2.75, +3.64].

**Consequences:** the loaders keep the copies, since the published FROG and LFE numbers were scored on them, and every number stays on the official labels with the de-duplicated one as a sensitivity (proposed for the paper, awaiting the owner). In training a copy is an uncovered person to `coast_targets`, so it teaches a coasting slot to report beside a covered person: negligible on `train_val`, to be de-duplicated before anything trains on the extras. The double reporting itself was then tested as a precision lever (A58, the entry above). No code changed for this entry; scripts and outputs in `checkpoints_three_horizon/a56/duplicates/`.

**Docs:** `memory/data-model.md` (new section, counts per file), `memory/interpreting-evaluation.md` (new section), `memory/experiments/2026-09-tracking-claims.md` (de-duplicated table), `memory/dataset-properties.md`, `memory/performance-log.md`, `memory/slot-design-evidence.md`, `memory/keywords.md`, `TODO.md` (A56 open row closed, §D item added).

## A56 run: a tuned ByteTrack + CBMOT gains +3.30 pp on our candidates, and the memory beats it by +2.90 (2026-10-02/03)

*keywords:* A56, E14, E15, E16, E20, E21, E22, tracking_baselines.py, ByteTracker, cbmot_update, PrecisionRecall, GOSPA, block bootstrap, compute_metrics, frames_out, tracker_sweep --tracker byte, occlusion strata, first-detection latency, end-to-end latency, StepUntil, schtasks /st, tail -F lock

The owner approved A56 and asked for anything else tracking-related. Pre-registration, outcome and every prediction scored: `memory/experiments/2026-09-tracking-claims.md`.

**Built, tests first** (581 -> 628 tests, all passing): `ByteTracker` and `cbmot_update` in `follow_the_drow/utils/tracking.py`, whose single-stage setting reproduces `SortTracker` frame for frame (a test, and on the real test cache: AP 0.774032 both); `utils/tracking_baselines.py` with an exact fast AP (`PrecisionRecall`, equal to `score_detections` to the last bit, float32 curves included), GOSPA, a paired block bootstrap, the occlusion and trajectory strata, and a streaming latency benchmark; `evaluate(frames_out=..., compute_metrics=False)`; `tracker_sweep.py --tracker byte | byte-cbmot`. `ruff` installed in `.venv`.

**Results** (FROG test, every 5th annotated frame, 95% intervals over 30 s blocks):

| | AP | vs candidates |
| --- | --- | --- |
| candidates | 77.66 | -- |
| SORT `min_hits` 1 / ByteTrack as published | 77.40 / 76.49 | -0.25 / -1.17 |
| E15: the memory's value rule as the score | 69.51 | -8.14 |
| E14: ByteTrack + CBMOT, tuned on val | 80.96 | +3.30 [+2.83, +3.77] |
| E10: learned scorer, no state (2 seeds) | 81.94 | +4.28 |
| E13: learned state (3 seeds) | 83.86 | +6.20; **+2.95 / +3.20 / +2.56 over E14 per seed, all intervals above zero** |

- **The decay is the whole trick for the rule**: without it val AP is 52.9%, with 0.1 per second 73.4%, tuned 76.5% (candidates 72.95%).
- **E20**: the memory's gain is occlusion-specific (+14.7 pp recall at matched FP for mostly occluded people, +3.5 visible) and flat across range and motion; its edge over the tuned rule sits on occluded people (+2.9) and a trajectory's first 0.5 s (+3.0).
- **E21**: no confirmation delay (first-frame detection 85.8% against the candidates' 84.1%); the tuned rule confirms late (74.1%).
- **E16**: GOSPA 0.502 against the candidates' 0.530 and the rule's 0.516; the predicted halving of the phantom term did not happen (-13%).
- **E22**: the whole streaming detector at 12.92 ms (p99 15.11) per frame on the RX 9060 XT, 19.53 ms (p99 22.38) on four CPU threads, inside FROG's 25 ms scan.
- **On LFE** (`docs/RESEARCH.md` §7.4.7): ByteTrack trades 27.6 false positives per false negative on LFE-PPN against SORT's 10.4, and the merge radius still beats both.
- **Four real-data checks passed**: candidate AP and recall at matched FP, the stored memory APs, the SORT row against `replay_sort`, and GOSPA's terms against LFE's documented FP/FN.

**Operational incidents, each written into `memory/gotchas.md` / `memory/long-running-tasks.md`:** a Git-bash `tail -F` watcher locked the chains out of `progress.txt` and outlived `TaskStop`; a chain started early on a stale `STOPPED` line; the scheduler template's `/st 23:59` re-fired two finished chains at 23:59; the session's end killed chain B at 18:23; `evaluate()`'s scoring made the arms step 20x slower than its replay (fixed with `compute_metrics=False`); and a finished process hung at exit despite `os._exit` (chains now stop a step two minutes after its final log line). The two-torch-processes `MemoryError` repeated the known commit-limit gotcha. Nothing was lost but time; every output was checked against its log before being used.

Docs: `memory/slot-design-evidence.md` (the ladder), `memory/performance-log.md` (the within-model comparison and footnote 8), `memory/rejected-ideas.md` (a hand rule in place of the learned head), `docs/RESEARCH.md` §3.5, §7.4.5, §7.4.6, §7.4.7, `docs/PAPER.md` (ByteTrack in the tracker section), `TODO.md` A56 (done; E17-E18 wait for A49).

## The paper's claims against the tracking literature, pre-registered; our SORT baseline already saw every candidate (2026-09-30)

*keywords:* A56, pre-registration, memory/experiments/2026-09-tracking-claims.md, claim ledger, tracking ladder, replay_sort, CANDIDATE_FLOOR, CBMOT, ByteTrack, E14, E15, E16, E20, E21, E22, block bootstrap

The owner asked for a plan of paper claims, tests and experiments from the tracking survey, to run once the machine is free.
It is `memory/experiments/2026-09-tracking-claims.md`: nine claims, the evidence each has, the experiment that decides it, predictions, and what the paper says for each outcome; TODO A56 now carries the work.
No code written and nothing run: A56 waits for the owner's confirmation, then tests first.

**A correction found while planning.** `replay_sort` and `tracker_sweep.py` feed SORT every candidate down to 0.01, so the repo's SORT baseline always saw sub-threshold evidence; it loses because it only filters, reporting each detection with its own score.
The 2026-09-28 wording ("a post-hoc tracker only ever sees post-threshold detections") is corrected in `docs/RESEARCH.md` §3.5 and §7.4.1, `memory/dataset-properties.md` and `memory/slot-design-evidence.md`.
The comparison that matters is against a rule that *raises* scores from track confidence -- CBMOT's -- which is the one unmeasured rung between the candidates (77.66%) and the learned scorer without state (E10, 81.84%).

## Detection gains from tracking are claimed in 3D LiDAR and video, not in 2D LiDAR (2026-09-29)

*keywords:* RESEARCH §7.4.6, detection gain from tracking, MAD, MoDAR, Detect to Track, 3D Auto Labeling, novelty, within-model gain

The owner asked whether any tracking paper claims a boost in person *detection*.
Tracking-by-detection papers report tracking metrics only; the claims come from memory-augmented detection: MAD +2.5 to +7.6 AP (Waymo, AV2), MoDAR +1.5 to +3.4 pedestrian AP online from 9.1 s of tracked history, Detect to Track +5.6 mAP offline.
None is on 2D LiDAR, so TAKHeLiPeD's +5.68 pp over its own candidates appears to be the first such claim on this sensor.
The defensible figure is that within-model gain, not 83.3% against DR-SPAAM's published 75.6%.
Written up in `docs/RESEARCH.md` §7.4.6.

## S2's first launch: a flag that never reached the model, and the commit limit broken a third time (2026-10-01)

*keywords: S2, visibility, wiring bug, ObjectMemory, 306828, _ArrayMemoryError, commit limit, machine_is_free.py, pack, free_range*

Three faults in one launch, all mine, all found before any result was reported.

**`--visibility` never reached the model.** `train_memory` builds `ObjectMemory` directly and the new keyword was not passed, so
the run was a plain B0 under an S2 name. The header line gave it away: **306,443 parameters where visibility should give
306,828**. The header now names the flag when it is on, because that line is what caught this.

**`pack()` dropped the new cache field.** It rebuilds `SequenceCandidates` positionally, so `free_range` was lost and the run
crashed after spending 980 s building the train cache. The three `TestEvaluate` fixtures that construct through `pack` were
failing for a *different* reason -- a missing constructor argument -- and masked it. Both fixed; `build_candidates` also stopped
hardcoding 90 sectors, which cannot cut the 8-beam test scan. **581 tests pass.**

**The commit limit, broken a third time, by a one-line parameter count.** A `python -c` that imports torch for a second was run
beside the live job to check that count. The training process died 20 s later in `sklearn_auc` with `_ArrayMemoryError: Unable
to allocate 1.93 MiB`, 45 minutes in. The victim is whatever allocates next, so the traceback blames evaluation and looks like a
bug. `machine_is_free.py` takes a second and exists for exactly this. Recorded in `memory/gotchas.md`.

Nothing experimental was lost: the run had to be restarted anyway because the flag was not wired, and both rebuilt caches
survive on disk.

## E10's second seed pulls AP inside its bar: three of four metrics clear, not four (2026-10-01)

*keywords: E10, a51_e10_norecurrent_s1, second seed, correction, three sigma bar, wp-AUC, pre-registered rule, multi-seed arm*

`a51_e10_norecurrent_s1` finished clean in 296 min and scored AP **82.04**, AP at 0.3 m 80.51, recall at matched false positives
90.94, coasting precision 11.90. Against the three-seed control the two-seed means are now **-1.40 AP (bar 1.45, inside)**,
-1.30 at 0.3 m (bar 1.29, clears), **-1.60 recall (bar 0.86, 1.9x)** and **-13.80 coasting precision (bar 8.19, 1.7x)**.

**This is the correction the second seed was run to find.** At one seed AP read -1.50 against a 1.45 bar -- 1.03x, the thinnest
possible clearance on the metric `AGENTS.md` makes lead every accuracy answer. It is now inside. Every place that said E10
"clears its bar on every metric" is corrected: `memory/slot-design-evidence.md`, `TODO.md` and `docs/RESEARCH.md` §3.5.

**The claim survives, on different metrics.** Recall at matched false positives at 1.9x its bar and coasting precision at 1.7x
are not marginal, and the coasting collapse is still the mechanism's signature. But the headline sentence changes from "every
metric" to "three of four, and not AP".

**A methodological point worth stating, because it cuts against us.** E10's two seeds agree to 0.20 pp on AP where B0's three
have sd 0.48, so the comparison is limited by the *control's* spread, not the arm's -- a two-sample test would read this as
clearly significant. The pre-registered rule does not cover a multi-seed arm, and choosing a different statistic after seeing the
result would be selecting the test that gives the wanted answer. The recorded verdict stays the pre-registered one; any
two-sample reading must be pre-registered before further seeds.

## S3 / E11: training with odometry that drifts, at a rate drawn per stream (2026-09-30)

*keywords: S3, E11, train-drift-m-per-s, sample_drift, chunk_elapsed, perturb_poses, domain randomisation, room frame, odometry*

Room coordinates are the design's central claim and they are bought *from odometry*. E2 measured the price: the room frame
scores 83.88 / 82.96 / 81.25 / 78.27 / 73.10 at 0 / 0.005 / 0.02 / 0.05 / 0.1 m/s of drift, against a sensor frame flat at
82.57, so it **wins only below roughly 0.02 m/s** -- inside the band real wheel odometry reaches. The model had never seen a
drifting pose, because `perturb_poses` existed only in `evaluate`.

- **`sample_drift(rng, streams, max_drift, max_heading)`** draws one rate per stream, uniform over `[0, max]`, redrawn each
  chunk, so the model sees the whole band rather than one point in it.
- **`chunk_elapsed(dt, reset, carry)`** gives the seconds each stream has run since its memory was last emptied, carried across
  chunks. Drift accumulates from the frame the slots were created in, so the clock restarts wherever memory does.
- **`perturb_poses`** now takes either a scalar, as evaluation always did, or one rate per stream.
- **`--train-drift-m-per-s` and `--train-heading-deg-per-s`**, both 0 by default, so every existing configuration is unchanged.

Nine tests, written first. Queued as `run_s3.ps1` behind S2: `a51_s3_drift_s0` at 0.05 m/s and 2 deg/s, then **E2's own sweep
re-run on the jittered model**, which is the readout -- does the room frame still lose above 0.02 m/s?

## S2's follow-ups: the coasting ranking did improve, the pipeline did not, and the timing was host load (2026-10-02)

*keywords: S2, visibility, matched report rate, coasting sweep, a51_s2_recovery.json, ms per frame, 4.74, host load, a51_cost.json, correction*

Two evaluation-only follow-ups on `a51_s2_visibility_s0`, and both correct something reported yesterday.

**The coasting head is more discriminating, not just more conservative.** Yesterday's read came from coasting precision at a
fixed 0.3 threshold (+1.95 pp at 58% fewer reports), which compares two operating points. Sweeping the threshold and comparing
people recovered at matched false-positive cost: **S2 recovers 6-13% more people** than B0 at the same cost (ratios 1.11 / 1.13
/ 1.08 / 1.06 at 0.26 / 1.10 / 3.35 / 25.1 FP per frame), and 6% fewer only at the cheapest point. The ranking genuinely improved.

**It still buys nothing end to end.** S2's candidates-plus-rescoring stage covers 2,240 fewer people than B0 before coasting
starts, so total coverage at matched cost is 1,400-2,300 worse; and at each model's own matched-false-positive operating point
recall is **92.58 against 92.47, flat**. A better coasting ranking does not reach the end of the pipeline, because coasting is a
small share of total coverage. **S2 stays a null end to end**, for a better-understood reason.

**The 7.76 ms per frame was host load, as suspected.** Re-timed on a quiet machine: **4.74 ms at 256 slots** (4.86 / 4.81 / 4.74
/ 4.92 / 5.21 at 64 / 128 / 256 / 512 / 720) against B0's 4.37. The feature costs about **0.4 ms, not 3.2**, and the curve stays
flat in slot count.

**Same hardcoded-output bug as the odometry sweep:** `slot_experiments.py cost` overwrote E4's `a51_cost.json`. E4's numbers
survive in `CHANGELOG.md` and `memory/slot-design-evidence.md`; both scripts need an `--out`.

## S3 / E11: jittered odometry buys real robustness, free, but the crossover stays (2026-10-02)

*keywords: S3, E11, a51_s3_drift_s0, train-drift-m-per-s, domain randomisation, drift sweep, room frame, D1, crossover, a51_odometry.json*

`a51_s3_drift_s0` trained with pose drift drawn per stream and redrawn each chunk over `[0, 0.05] m/s` and `2 deg/s`, then
E2's sweep was re-run on it. Room-frame AP against B0's clean-trained curve: **83.50 / 82.77 / 81.59 / 80.02 / 77.44** against
83.88 / 82.96 / 81.25 / 78.27 / 73.10 at 0 / 0.005 / 0.02 / 0.05 / 0.1 m/s -- **+1.75 pp at 0.05 and +4.34 pp at 0.1**, for
-0.38 pp at zero drift. The penalty at 0.1 m/s falls from 10.78 pp to 6.06 pp.

**The robustness is free.** On clean test odometry every barred metric is inside its bar: AP 83.50 (+0.16), AP at 0.3 m 81.76
(+0.07), recall at matched false positives 93.09 (+0.61), coasting precision 29.04 (+3.61).

**It does not move the crossover.** The sensor frame is flat at 82.03 in this sweep, so the room frame still wins below about
0.015 m/s and loses above. Domain randomisation shrank the penalty past the crossover rather than removing the crossover, so
**D1 stays conditional** and the write-up keeps that caveat.

**Two provenance notes.** The sensor-frame baseline differs between the sweeps (82.03 against E2's 82.57), so the sensor arm used
a different D1 checkpoint -- compare room against room and read any crossover within one sweep. And
`slot_experiments.py odometry` writes a fixed path, so this run **overwrote E2's `a51_odometry.json`**; E2's numbers survive in
`results/a51_odometry.log` and the script should take an output path.

## S2's result: free-space evidence makes coasting conservative, not discriminating (2026-10-01)

*keywords: S2, a51_s2_visibility_s0, visibility, coasting precision, coast_per_frame, calibration shift, null result, matched report rate*

`a51_s2_visibility_s0` finished after a resume (10.4 h, best at epoch 22, 79.76% val AP). Against the three-seed control:
AP 83.58 (+0.24), AP at 0.3 m 81.92 (+0.24), recall at matched false positives 92.58 (+0.10), **coasting precision 27.38
(+1.95, bar 8.19)**. Every barred metric is inside its bar, so the feature costs nothing and buys nothing.

**The shape of the null is the interesting part.** Coasting reports fell from 0.176 to **0.074 per frame, 58% fewer**, for
1.95 pp more precision. A better ranking would pay far more than that at less than half the volume; what happened instead is
that the whole model grew conservative -- precision 70.3% at the shared threshold with false positives 1.087 against B0's ~1.6,
and false negatives up to 0.495. `recall at matched FP`, which removes calibration from the comparison, is flat at +0.10.

**Not yet like for like.** Coasting precision is read at a fixed 0.3 threshold, so +1.95 pp at a 58% lower report rate compares
two operating points. The sweep that answers it -- precision at a *matched* report rate, via `memory_diagnosis.py --recovery` on
this checkpoint -- needs no training and is queued.

**One number not to quote yet:** 7.76 ms per frame against B0's 4.53. The host was loaded enough that training epochs ran 970 s
where they had run 677 s, so this needs re-timing on a quiet machine before any of it is attributed to the feature.

## S2: the coasting head is told what the laser saw past each slot (2026-09-30)

*keywords: S2, visibility, free_range_profile, visibility_feature, coasting precision, SECTORS, cache key, free space, occlusion*

Coasting has to tell a genuinely occluded person from a slot that is simply wrong, and until now the head saw nothing of the
current scan -- only eight bookkeeping numbers and the slot's learned state. The decisive evidence was being thrown away: **if
the laser reaches past a slot's own position, nobody is standing there.**

- **`free_range_profile(ranges, sectors)`** -- the nearest return per angular sector, through `sanitize_ranges` first, so a
  missing return reads as empty space rather than as the closest obstacle in its sector.
- **`visibility_feature(slot_xy, free_range, angles, sectors)`** -- per slot, how much further the beam reached than the slot
  claims to be, scaled by `VISIBILITY_SCALE_M` and clamped to [-1, 1]. Negative means something is in front, so the person
  could still be hidden; positive means the beam passed through and found nothing; a slot outside the field of view reads 0.
- **`ObjectMemory(visibility=True)`** appends it to the slot statistics, so the rescoring head sees it too, and `step()` then
  requires `free_range` and `angles` rather than silently doing without.

**Two implementation decisions worth recording.** The flag is **off by default**, because the statistic widens every layer that
reads the statistics vector, and every existing checkpoint -- B0, E10, E13 -- must still load. And the candidate cache now
carries the profile, so its **key gained the sector count** and the next run rebuilds it (about an hour) rather than the code
gaining a migration path for old caches.

Tests first, twelve of them: the profile's treatment of missing returns, the feature's sign, saturation and field of view, and
that the default model's shape and numbers are untouched. They could not run beside S1, so `run_s2.ps1` runs the suite itself
when the machine frees and refuses to train if it fails.

## Arm provenance comes from the checkpoints, not the launchers; a second E10 seed is running (2026-09-30)

*keywords: provenance, judge_arms.py, UNVERIFIED_CLI, checkpoint args, T1, T2, a51_e10_norecurrent_s1, headline claim, seed count*

`judge_arms.py` flagged T1 and T2 as unverified because their launcher scripts no longer exist. That was the wrong place to
look: every stage-3 checkpoint stores its own CLI, and `torch.load(<run>/step3a_object_memory.best.pth)["args"]` shows both
arms match the control exactly -- `step3a` on `step2_static`, `chunk_s [1.0, 10.0]`, `test_strides [5]`, `max_hours 24`, 256
slots, seed 0, cosine -- differing only in their own intervention. The flag is removed and the script records where provenance
comes from.

**A second E10 seed is running.** The arm became the paper's headline after one seed, and its AP margin is 1.50 against a
1.45 bar. Earlier in this session a two-seed reading of the tightest-barred metric flipped when the third seed arrived
(E13's recall at matched FP, +1.08 clearing its bar, then +0.68 inside it), which is precisely the risk a single-seed headline
carries.

## The headroom feedback actually has: +1.91 pp of recall, all of it already held by coasting slots (2026-09-30)

*keywords: E12, feedback headroom, necessity, memory_diagnosis.py --recovery, no_candidate, feedback-exclusive pool, TOP_P*

Asked before committing to another feedback architecture: is there anything for it to win? `memory_diagnosis.py --recovery` now
splits misses by whether stage 2 produced **any** candidate near the person at any score -- the line between what rescoring can
fix and what only feedback can.

On B0 seed 0, FROG test, threshold 0.3: of 153,655 annotated people stage 3 covers 129,755 and misses 23,900. **83.0% of the
misses already have a candidate** that rescoring can raise. Only **4,068 (17.0%)** have no candidate at all, and of those
**2,930** are held by a live slot -- **1.91% of all annotated people**. Every one of those 2,930 slots is already coasting.

**So feedback's exclusive ceiling is +1.91 pp of recall**, attainable only by converting all of them with no false positives,
and it finds nobody the memory does not already hold; its unique value would be a scan-localised detection instead of a coasted
slot position. Outside that pool it can only duplicate rescoring, which reaches the same people by a far shorter gradient path
-- which is itself an explanation for why the three E12 arms had nothing to learn.

Read with E12's capacity measurement: the path was under-built **and** aimed at a small target. The first is cheap to fix; the
second bounds the reward.

## E12's real conclusion: the feedback path carries 0.02% of the bottleneck, so it could never have mattered (2026-09-30)

*keywords: E12, measure_prior_share.py, a51_e12ab_oracle_norescore, prior share, aggregate, render_prior, injection capacity, cancellation hypothesis*

The ceiling cell -- a perfect prior with **no** rescoring head to compete with it, the configuration neither earlier arm covered
-- returned **+0.02 pp**. With E12b's +0.0012 and E12a's -0.0022, all three cells are null.

**Both explanations offered along the way were wrong, and the real one is now measured.** *Competition* predicted the path would
wake once the rescoring head was removed: it did not. *Cancellation* -- large weights parked where they cannot act -- is false:
under an oracle prior the two channels are identical, so their gradients are identical and their difference stays exactly
`0.000000e+00`, putting the entire weight in the acting direction (sum rms 3.56e-2 against the body's 4.07e-2).

**`utils/measure_prior_share.py`** (new) measures the two terms of `aggregate` on real frames instead of comparing weight norms:
the prior carries **0.012-0.045% of the layer's input magnitude**. The arithmetic is not subtle once stated -- 2 prior channels
against 256 body channels, non-zero in 2% of sectors, at the most downsampled layer. A per-weight rms of 0.44 is a
per-contribution share of 0.045%.

**This changes what E12 licenses the write-up to say.** Not "memory cannot help the detector before the threshold", but "**this
injection** contributes nothing, and here is why". Testing the idea properly needs a wider path (project the prior to `c2`
channels), a denser prior (`render_prior` gives each person one sector with no extent), injection above the bottleneck, or
multiplicative gating. That is a design change rather than an ablation, so it is not queued.

## E12a settles it: the feedback path does nothing even with its competitor removed, and the oracle verifies (2026-09-30)

*keywords: E12a, a51_e12a_norescore, verify_oracle_prior.py, feedback contribution, stage 2 drift, rescoring head, channel degeneracy*

`a51_e12a_norescore` trained step 3b with the rescoring correction switched off, so the prior drawn into stage 2 was the memory's
only route to the loss. Contribution: **-0.0022 pp** (77.5909% against 77.5931% silenced). With E12b's +0.0012 pp under a perfect
prior, **both cells are null**, and the competition hypothesis is refuted on its own prediction -- removing the competitor did not
wake the path.

**The oracle verified first**, so E12b's null is no longer provisional: on 400 recorded FROG frames, 400 of 400 draw a non-zero
prior and **638 of 638** people's sectors, recomputed from `people_polar`'s bearing independently of `render_prior`'s xy path,
land on a marked sector.

**The trained weights are large, which is the surprise.** `aggregate.weight[:, 2*c2:]` reaches rms 1.78e-2 against the body's
4.07e-2 -- 44% -- with all 256 entries non-zero, yet silencing those channels moves nothing. Under an oracle prior the two
channels are identical, so only their sum can act and their difference is a free direction; `verify_oracle_prior.py` now reports
both, to be run on all three checkpoints. That degeneracy cannot explain E12a, whose prior has differing channels.

**Side finding.** Stage 2 finishes at 77.59% in E12a against 76.45% in E12b, with candidates alone at 77.66%: joint training's
~1.2 pp stage-2 drift (A50 phase 1 item 2) tracks the **rescoring head**, not the prior.

## E12b: a perfect prior buys stage 2 +0.0012 pp, pending one verification (2026-09-30)

*keywords: E12b, a51_e12b_oracle, oracle prior, feedback contribution, aggregate.weight, null result, verification pending*

`a51_e12b_oracle` trained step 3b for 3 epochs with stage 2 fed a ground-truth prior, then scored those weights twice on test:
**76.4517% with the prior against 76.4506% with it silenced, +0.0012 pp.** The whole detector moved +0.055 pp, to 81.69%. The
two runs are not bit-identical, so gradient reached the feedback channels and moved them -- to almost exactly nothing.

**Not yet a finding.** An empty oracle would produce an identical-looking null, and `oracle_prior` is so far covered only by
synthetic unit tests, which `AGENTS.md` rule 4 says is insufficient for anything reading recorded data. Queued for the gap after
E12a: confirm on real FROG frames that the rendered prior is non-zero where people are, and that `aggregate.weight[:, 2*c2:]` is
non-zero after training. The commit limit forbids running it beside a training job.

**A design gap, named rather than glossed.** E12b kept the rescoring head, so its null admits two explanations -- feedback is
useless here, or the head out-competed it again. E12a removes the head but feeds a real prior. The separating cell, oracle prior
*and* no rescoring, is neither run and is worth ~4 h if E12a also comes back null.

## E13 at three seeds: both attention mechanisms are free to remove, and a two-seed reading had been wrong (2026-09-30)

*keywords: E13, E6+E7, a51_e67_recurrent, --no-exchange, --no-learned-match, three seeds, recall at matched FP, two-seed error*

`a51_e67_recurrent_s1` and `_s2` finished the arm at three seeds (248 and 374 min, both clean on the first attempt). Against the
three-seed control: AP 83.86 (+0.53, bar 1.45), AP at 0.3 m 82.14 (+0.46, bar 1.29), recall at matched false positives 93.16
(+0.68, bar 0.86), coasting precision 29.81 (+4.38, bar 8.19). **Nothing clears in either direction**, at 173,579 parameters
against 306,443 (-43%) and 3.72 ms per frame against 4.53 (-18%). Per-seed AP 83.91 / 84.15 / 83.52.

**A two-seed reading of this arm was wrong and is corrected here.** At seeds 0 and 1, recall at matched false positives read
+1.08 and cleared its 0.86 bar, and was reported that way on 2026-09-29. Seed 2's 92.36 pulled the mean to +0.68, inside the bar.
The metric with the tightest bar is the one a short seed count misreads first; the three-seed protocol exists for exactly this.

**With E10, the architecture is now decided.** The two arms are the same experiment in opposite directions: remove the slot state
and every bar is cleared downward; remove both attention mechanisms and nothing is cleared at all. State is the mechanism,
attention is not. The proposal is E6+E7, with the full model as the ablation.

## E10: the learned slot state is the mechanism, and removing it is the only ablation that has ever cost anything (2026-09-29)

*keywords: E10, --no-recurrent, slot state, coasting precision, judge_arms.py, a51_e10_norecurrent_s0, tracker with a scorer*

`a51_e10_norecurrent_s0` rescored from the bookkeeping statistics alone -- value, age, time since match, displacement, hit rate --
with no slot state and no recurrence, 55,691 parameters against B0's 306,443. It finished clean in 300 min and is **the only arm
of the eleven that clears its three-sigma bar on every metric**, all four downward: test AP **81.84** (-1.50, bar 1.45), AP at
0.3 m 80.26 (-1.42, bar 1.29), recall at matched false positives **90.80** (-1.67, bar 0.86), coasting precision **11.35**
(-14.08, bar 8.19). It runs at **2.70 ms per frame** against ~4.5.

**This is the favourable answer.** The question E10 asked was whether the memory is a tracker with a learned scorer; the answer is
no. "The slot state is learned, not hand-coded" was the weakest of the three claims separating this design from tracking-by-
detection -- it rested on D3's 0.90 pp, inside its bar -- and is now the strongest. The coasting-precision collapse says where the
state earns its keep: with no current measurement, only carried state separates a genuinely occluded person from a stale slot.

**It also settles E13's shape.** The state costs 250,752 parameters and ~1.8 ms and pays for them; the attention does not (E6, E7).
The architecture to propose is E6+E7 -- whose validation AP, 80.67%, is the highest of any arm -- not the smaller E10.

`utils/judge_arms.py` now prints the whole comparison; `results/a51_arms.json` holds it as data.

## The feedback path becomes measurable: an oracle prior, rescoring that can be switched off, and a within-network ablation (2026-09-29)

*keywords: E12, E12a, E12b, oracle_prior, prior_mode, rescore=False, feedback contribution, test_prior_silenced, run_e12.ps1*

`docs/PROPOSAL.md`'s claim that memory changes what the detector *sees* has never been tested, because step 3b's prior never left
zero. Three pieces of code make it testable, written against tests first.

- **`ObjectMemory(rescore=False)`** removes the rescoring correction and its parameters, so a candidate keeps its own score and
  the prior drawn into stage 2 is the memory's only remaining route to the loss (E12a). Both call sites now share one
  `_rescored` helper instead of repeating the gate-and-limit arithmetic.
- **`oracle_prior()`** draws the prior a perfect memory would draw -- every annotated person marked in its sector at full value
  and probability (E12b) -- and **`joint_step(..., prior=...)`** feeds it in place of what the slots would have drawn.
  Unannotated frames draw nothing rather than carrying the last annotation forward: stale positions would stop it being an upper
  bound, and it costs nothing, because the oracle holds no state and every scored frame is annotated.
- **`evaluate_joint(..., prior_mode=)`** picks `slots`, `oracle` or `zero`, and raises on anything else rather than silently
  falling back.

**The measurement got cheaper than planned.** Each arm was going to need its own retrained control. Scoring one trained network
twice -- once with its prior, once with it silenced -- answers the same question in half the GPU time, and is the better
comparison: a retrained `prior_channels=0` control can compensate for the missing channels, while the same weights cannot. Every
step-3b run now ends with a `feedback contribution` line and stores `test_prior_silenced` in its results.

**Queued** as `run_e12.ps1` behind the E10 chain: `a51_e12b_oracle`, then `a51_e12a_norescore`. The chain runs the test suite
itself when the machine frees up and refuses to launch either run if it fails, because the commit limit made it impossible to run
`pytest` beside E10.

## Stage-3 training is bit-reproducible from its seed; `--resume` is what breaks it (2026-09-29)

*keywords: determinism, reproducibility, a51_b0f, a51_b0_short, resume divergence, judge_arms.py, fresh control, seed pairing*

**The fresh B0 control reproduced the old one exactly, which is the answer rather than a disappointment.** `a51_b0f_s1` retrained
stage 3 from scratch -- same seed, same flags, same code, a different output directory, eight days after `a51_b0_short_s1` -- and
matched it on **46 of 46** evaluations to six decimals of validation AP, and on every test metric: AP **83.17**, AP@0.3 81.52,
recall at matched FP 92.53, coasting precision 22.31. Only `ms/frame` differs (4.65 against 4.51), as a wall-clock number must.

**`--resume` is what does not reproduce.** `a51_b0f_s0` was interrupted twice by driver bugchecks and resumed. It tracks
`a51_b0_short_s0` for **36 of 42** evaluations and diverges at exactly the first epoch after the resume point -- 10 s stage epoch 4,
0.7885 against 0.7854 -- finishing 0.41 pp higher in AP (83.37 against 82.96) and 5.3 pp lower in coasting precision. A resumed run
is a different draw from the seed's distribution, not the same run continued. It is a valid sample of its arm and never a
reproduction of it, which matters for `a51_o1_bounded_s0c` (three attempts) and `a51_e6_noexchange_s0` (scored from an interrupted
checkpoint).

**Three consequences.**

- **The fresh control changes no verdict.** Its mean will land within ~0.15 pp of the short control's 83.34, so every arm judgement
  already recorded against the E9 bars stands, and the results table does not need rewriting. Seed 2 is a live prediction: if it
  runs uninterrupted it must score **83.88 / 82.17 / 92.72 / 26.61**.
- **`memory/interpreting-evaluation.md`'s "seeds do not reproduce runs on this machine" is corrected.** Its 2026-09-17 evidence
  compared runs whose *code differed*, so it could not separate seed irreproducibility from the code change; the stated mechanism
  ("GPU kernels are nondeterministic") is not what was measured. The practical rule -- compare means, not per-seed differences --
  survives for a different reason: an ablation changes the architecture, so the same seed's random stream is consumed differently
  and the two runs were never paired.
- **Re-running a configuration is now a legitimate check on a number that looks too good**, which it was not believed to be.

**`utils/judge_arms.py`** (new, standard library only, safe to run beside a training job) reads every arm's `step3a_results.json`,
computes a control's mean, sd and three-sigma bars, and prints each arm's deltas with a verdict against both those bars and E9's.

## Why step 3b's feedback prior is silent, and why no torch process can run beside a training job (2026-09-29)

*keywords: E12, E12a, E12b, feedback prior, aggregate.weight, rescoring head competition, WinError 1455, commit limit, pagefile, machine_is_free.py*

Written while the fresh B0 control chain was training, so both findings come from logs already on disk rather than from a new run.

**The feedback prior is silent because the rescoring head out-competes it, not because gradient fails to reach it.** Step 3b's log
reports stage 2 **with** feedback at 75.68% on test; A50 phase 1 item 2 had separately scored the same network with the prior
algebraically **silenced** at 75.68%. Equal to two decimals, so those channels carry nothing. The mechanism is competition, not a
bug: gradient does reach `aggregate.weight[:, 2*c2:]` from a zero start, because `dL/dW = dL/d(bottom) * prior` and the prior is
non-zero -- but the rescoring head reads the same slots one operation before the loss, while the prior must travel through the
bottleneck, decoder, vote grid and candidate decoder to matter. At 3 epochs and lr 1e-4 the optimiser takes the short path.
**Consequence for E12:** "initialise the prior's weights away from zero" (the design `TODO.md` recorded) would reproduce the null
more slowly. E12 must remove the competitor -- **E12a**, step 3b with the rescoring head frozen at identity, with the prior's
contribution logged per epoch -- optionally preceded by **E12b**, oracle priors, to tell a "feedback cannot help" null apart from a
"training will not find it" null. Awaiting the owner's approval; the options table is in `TODO.md` A51 phase 6.

**The machine's limit is commit, not RAM, and beside a live run there is none left.** A CPU-only inspection script
(`CUDA_VISIBLE_DEVICES=""`, one `torch.load`) could not even `import torch`: `OSError: [WinError 1455] The paging file is too small
... loading c10.dll`. At that moment: 15.9 GB physical with **5.4 GB free**, but a 30.6 GB commit limit with **1.9 GB free**, and
`S:\pagefile.sys` already at its configured 15000 MB maximum. So free physical memory does not predict whether a second process
starts, and `machine_is_free.py` covers more than `pytest`: **nothing that imports torch runs beside a training job**, including a
one-line checkpoint load -- queue it. Raising the page file's maximum would lift this, and is the owner's machine setting to make.
Recorded in `memory/gotchas.md` beside the older `_ArrayMemoryError` signature.

## Person-tracking literature surveyed: three claims narrowed, six experiments proposed (2026-09-28)

*keywords:* tracking literature, RESEARCH §7.4, ByteTrack, OC-SORT, CBMOT, IPDA, track-before-detect, Deep Tracking, GOSPA, OSPA, HOTA, JRDB tracking, Arreghini 2607.21309, E14, E15, E16, E17, E18, E19, A54

The owner asked for a review of every citation and experiment in the repo against the person-*tracking* literature, which had never been surveyed. Written up as `docs/RESEARCH.md` §7.4; no code changed, no run started.

- **SORT is not the strongest post-hoc baseline.** ByteTrack (ECCV 2022) associates low-score detections in a second pass, so "a post-hoc tracker only ever sees post-threshold detections" holds for SORT only. Corrected in `docs/RESEARCH.md` §3.5, `memory/dataset-properties.md` and `memory/slot-design-evidence.md`; E14 is the test.
- **The slot value rule is prior art.** `v + gain*(1-v)*s` is CBMOT's score update (IROS 2021) with a gain, and track existence probability is IPDA (1994); D2 is CBMOT's count-based ablation, which lost there and ties here.
- **"Acting before the threshold" is track-before-detect** (Salmond & Birch 2001), and learned recurrent memory over raw 2D laser exists as Deep Tracking (AAAI 2016, IJRR 2018), grid-keyed. The claim that survives: a learned, causal, object-keyed track-before-detect over sub-threshold candidates.
- **Two errors fixed in TODO A51's earlier literature note:** 99.1% is the share of people stage 3 *loses* that rescoring pushed under the threshold, not "of the people our memory recovers"; and JRDB's tracking benchmarks are image-plane and point-cloud, with none on its 2D laser.
- **A new FROG number exists:** Arreghini et al., arXiv 2607.21309 (July 2026), 65.9 / 66.4% AP with a 3 s space-time window, trained by cross-modal self-supervision; not added to `memory/performance-log.md` until its training regime is confirmed.
- **Proposed, not agreed:** TODO A56 (split out of A51 on 2026-09-29, owner's request) -- E14 ByteTrack on the same candidates, E15 CBMOT rescoring, E16 GOSPA beside AP, E17 slots scored as a tracker, E18 self-supervised memory, E19 an occlusion input for coasting. A54 gained two leads from prof. Aycard's own publications, neither of them the reference.

## `AlgorithmicDetector` is credited to prof. O. Aycard, wherever it is reported (2026-09-23)

*keywords:* Aycard, AlgorithmicDetector, attribution, credit, citation, provenance, A54

Owner's call. The rule-based leg+chest clustering-and-tracking detector this repository ports is **prof. O. Aycard's** algorithm and implementation; the only attribution that existed was the header comment in `library/cpp_core/sources/detector.cpp` and one parenthetical in `docs/RESEARCH.md` §2.1, which read as a provenance note rather than as a credit.

- **Credit added** to `README.md` (intro plus a new *Credits* section), `AGENTS.md`'s repo layout, `memory/performance-log.md` (the `Kind` column now reads *rule-based, O. Aycard*, with a new footnote 0), `memory/detector-architectures.md`, `memory/deployment.md`'s node table, `library/cpp_core/include/follow_the_drow/detector.hpp` and `AlgorithmicDetector`'s Python docstring.
- **`docs/RESEARCH.md` §2.1 rewritten** from "the C++ source attributes the algorithm to O. Aycard" to a statement that the algorithm is his, that it must be credited to him in any write-up, and that only the pybind11 binding, the wrapper and the harness are this project's. §2.2's mention now names him too.
- **What is deliberately absent: a bibliographic reference.** No paper, report or thesis for this detector exists anywhere in the repository, and inventing a plausible one would be worse than having none, so every place that credits him says the formal reference is still to be obtained. That is `TODO.md` A54, blocked on the owner.

No behaviour changed; 533 tests pass.

## The detector is named TAKHeLiPeD, and the docs are refreshed onto the current chain (2026-09-23)

*keywords:* TAKHeLiPeD, naming, three-horizon, README, RESEARCH.md, SHOWCASE.md, PROPOSAL.md, performance-log, A52, A53

**The name, owner's call.** **T**(emporal) **A**(daptive) **K**(nee-)**He**(ight) **Li**(dar) **Pe**(rson) **D**(etector), pronounced *TA-KHé-Li-PeD*.
It names what every earlier entry calls *the three-horizon detector*. **Documentation only for now**: every identifier still says `three_horizon`, so no checkpoint, cache or log path moved. The code half is `TODO.md` A52.

**The docs were further behind than the name.** `README.md` still advertised the three full-scan architectures as the contribution and quoted `SpaceTimeCNN` at 80.7% wp-AUC; `memory/performance-log.md`'s own rows for this detector were the 2026-09-15 chain, four backbone changes stale.

- **`memory/performance-log.md`**: the two rows are now **77.7%** (stages 1-2, the no-temporal seed-0 backbone the memory sits on) and **83.3%** (stages 1-3, B0's three-seed mean, sd 0.47), with footnotes 4 and 5 rewritten from the B0 and diagnosis entries of 2026-09-19 and 2026-09-21. The retired 76.9% / 80.10% are marked **retired, not corrected** -- different backbone, and both predate the running time step and the limit-beam decoding. Added a speed row for stage 3 alone (4.4-4.6 ms) and a seventh footnote for it.
- **`DR-SPAAM T=5 at 75.6% is the number to beat`** becomes *was*, cleared by 7.7 pp, with the "ours measured, theirs published" caveat kept beside it.
- **Two figures were stale in the same way in three places** and are fixed: the sensor is ~40 Hz (a scan every ~25 ms), not 26.2 Hz, corrected 2026-09-17; and the test suite is 533 tests in ~60 s, not 347 in ~30 s (`memory/commands.md`) or 63 in ~6 s (`README.md`).
- **`memory/detector-architectures.md` had no section on this detector at all** -- it described only the superseded trio. It now leads with TAKHeLiPeD's three stages, parameter counts, per-frame cost and the `no_grad` slot boundary, states that its absence from `DETECTOR_REGISTRY` is deliberate (it is stateful, so it does not fit the registry's contract), and retitles the old table *superseded*.
- **`README.md`** rewritten: the name and its pronunciation, the streaming `step()` interface, the FROG table above with false positives and ms/frame beside it, what the gain is *not* (SORT, capacity), and a documentation table of deep links into `docs/RESEARCH.md`. The three legacy architectures appear once, as provenance.
- **`docs/SHOWCASE.md`** retitled and re-measured onto the B0 chain; its "history is used: shuffling costs 1.3 points" bullet is replaced, because the backbone this result sits on is single-frame and the temporal one ties it -- the horizon that pays is the memory's, not the convolution's. Its feature count was also 11, which has been 10 since the validity channel was dropped on 2026-09-17.
- **`docs/PROPOSAL.md`** and **`docs/PAPER.md`** carry the name; `PROPOSAL.md`'s status line no longer says "nothing described here is built or trained yet".
- **`docs/RESEARCH.md`** gets a header note saying its title names the superseded line of work, which of its sections stay authoritative (§1, §3, §4, §7) and which are provenance (§2, §5, §6). Rewriting the body is `TODO.md` A53.
- **`AGENTS.md`** gains a paragraph on why this detector is the standing exception to its own six-stage pipeline list.

No code changed; 533 tests pass and `verify_memory.py` is clean.

## The four slot-design ablations, off by default: coordinates, value rule, decay, and a trainable value rule (2026-09-20)

*keywords:* A51, D1, D2, D3, D4, room_frame, latest_value, fixed_decay_s, learn_value, --no-room-frame, --latest-value, --fixed-decay-s, --learn-value

Owner-approved (2026-09-20) after asking whether the slot bookkeeping should be trainable at all. `manage_slots` runs under `no_grad`, so association, value accumulation, spawn, replacement and retirement are fixed rules and only what reads them learns. Each ablation is one flag, each default reproduces B0.

- **D1 `--no-room-frame`**: slots in the sensor frame. Its test shows the ablation costs more than the velocity feature: at 0.2 m of robot travel a static object gains velocity, and at 1 m per frame it leaves the 0.6 m association gate and the slot is lost outright.
- **D2 `--latest-value`**: a matched slot takes the candidate's score instead of accumulating. Both rules create a slot at the candidate's score; accumulation then climbs past it towards certainty.
- **D3 `--fixed-decay-s T`**: one decay time for every slot, removing the decay layer and its bias (1,312 parameters at dim 32).
- **D4 `--learn-value`**: the value rule's `gain` and `fade_time_s` become parameters (+2). The discrete decisions stay under `no_grad` and take the constants as detached floats, while the value the network reads is recomputed with gradient, so the rule moves as they train.
- **Interaction, found by D4's test:** with T1's zero-initialised rescoring head no gradient reaches anything the head reads at initialisation, value constants included, so D4 must be run on whichever rescoring parameterisation wins T1.
- `object_memory` carries all four through a checkpoint round trip. Tests: 5 in `TestObjectMemory`, 1 in `TestObjectMemoryFor`.

## The driver bugchecks are a pattern on one branch, and AMD requires 26.2.2 for ROCm 7.2.1: revert decided (2026-09-26)

*keywords:* A55, amdkmdag.sys, bugcheck, 0x7E, Adrenalin 26.2.2, Adrenalin 26.8.1, 32.0.23027.2005, 32.0.31041.1004, TDR, driver revert

- **The full crash history, read from the System log and WER:** five `0x7E`/`c0000005` bugchecks since 2026-05-13, **all on the `32.0.31xxx` driver branch** installed from 2026-05-08 (26.5.x, 26.6.x, then three on 26.8.1: 09-01, 09-22, 09-25). The three on 26.8.1 fault at the same code offset (`...4f50`), so one deterministic code path. TDR live dumps also exist (`0x141` 08-09 and 09-14, `0x117` 07-27 and 09-21), which corrects this file's 2026-09-23 entry ("no TDR events"). The earlier `22xxx`/`23xxx` drivers, Dec 2025 to May 2026 (26.2.2 among them), logged no bugcheck, but never ran ROCm.
- **AMD's *Install PyTorch on Windows* page says 26.2.2 "must be installed"** for the ROCm 7.2.1 wheels, and 26.2.2 is driver store `32.0.23027.2005`, the `23xxx` branch. 26.8.1 was never a validated pairing.
- **Decision:** revert to 26.2.2 rather than to 26.7.x, which would stay on the branch that crashed. The owner stopped the A51 queue at 04:51 to do it; O2's resume file is intact. Verification plan and fallback in TODO A55; `memory/gotchas.md` updated.
- **Then the dumps were read (WinDbg), and the revert is on hold behind one test.** 09-22 and 09-25 are one bug: a null-pointer read in `amdkmdag` on Windows' power-IRP worker, i.e. while the GPU's power state was being changed. The display turns off after 5 min, the trainer keeps only the system awake, and the driver has GPU power-down (BACO) enabled; 09-25's crash came 4 min 50 s after the last unlock. Hypothesis: display off -> chip power-down under a live ROCm context. Two side findings: the 09-21 "device-side corruption" was a TDR at 23:06:26 inside epoch 12's evaluation, and the 09-15 "sleep" was a Windows Update restart.
- **The display-off hypothesis failed its test.** Ten minutes of ROCm load with 0.2-30 s idle gaps, display off for ~9: first-kernel latency after a gap 0.4-1.3 ms (display on: 0.7-1.2 ms), D-state D0 throughout, data intact, no crash. The trigger of the power request stays unknown; the minidumps cannot show it (small dumps only). Revert back on, preceded by switching to kernel dumps.
- **Then the owner chose to find the cause before reverting.** Kernel dumps enabled (live since the 05:58 boot, 8 GB dedicated dump file on `C:`), and the queue relaunched O2 at 06:03 on 26.8.1. New suspect: a display **wake** under load -- both crashes came as the owner logged in to check progress, and the display-off test never woke the display with a GPU program running. The System log shows 401 wake/unlock transitions since 09-14 against 2 bugchecks, so a wake trigger would fire on well under 1% of wakes.

## D1 settled at three seeds: room coordinates are worth 1.14 pp; a second driver bugcheck, survived by resume (2026-09-26)

*keywords:* A51, D1, room_frame, a51_d1_sensor_s1, a51_d1_sensor_s2, amdkmdag.sys, bugcheck, resume, O2

- **D1, the design's central claim, is the one internal choice that is now established.** Slots in the sensor frame score 82.02 / 82.57 / 82.01 (mean **82.20**, sd 0.32) against B0's 83.34 (sd 0.48): **-1.14 pp, Welch t = 3.40, p = 0.034**. Recall at matched false positives moves only -0.17 pp (p = 0.41), so the effect is in ranking, not at the operating point, and the ablation still bundles the association loss the sensor frame causes -- the robot's own motion pushes a static object out of the 0.6 m gate.
- Seed 1 needed **five attempts and 835 min**; seed 2 took 321 min on its first.
- **The AMD driver bugchecked a second time** (2026-09-25 22:29, same `0x7E` / `amdkmdag.sys`), killing O2 at 3.9 h and the launcher with it. **Resume worked in a real crash**: O2 restarted from chunk stage 1, epoch 7, with 3.86 h credited, losing only the 16 minutes since its last evaluation. The crash also caught a checkpoint mid-write, leaving a 0-byte `10s.best.pth` -- exactly what the resume file avoids by writing beside itself and renaming.
- O2 is running again; the queue after it is E1-E4, then E6/E7, then O1.

## Metric spreads re-judge phase 1, four ideas closed, and the write-up's experiment list (2026-09-25)

*keywords:* A51, E1, E2, E3, E4, E5, E6, E7, E8, E9, noise floor, coast_precision spread, phase 5, rejected-ideas

- **E9, the spread of every reported metric** over B0's three seeds: AP sd 0.48, AP at 0.3 m 0.43, **recall at matched FP 0.29**, **coasting precision 2.73**; three-sigma bars 1.45 / 1.29 / 0.86 / 8.19 pp. **This re-judges phase 1**: T1's recall gain (+0.93 pp) is the only arm effect that clears its own bar, barely; T2's (+0.73) does not, and neither arm's coasting-precision gain (+6.8, +6.6) approaches 8.19. The two arms are therefore not the equivalent pair they looked like. A three-sample sd is itself uncertain, so these are order-of-magnitude bars.
- **Four slot ideas closed with reasons** in `memory/rejected-ideas.md`: S3 (capacity is not scarce -- 24 of 256 slots, 95.5% of unreported people already held), S4 (97% already have a matched slot across a 3x gate and 15x fade sweep), T3 (the offending input was the time step, since fixed), and D2 recorded as *unsupported rather than rejected*.
- **O1 unparked** (owner's call): the per-iteration `nan_diagnosis`, `--resume`, and a launcher that retries a crash but not a guard make a third attempt cheap to read and cheap to lose.
- **Phase 5 proposed and settled with the owner**: E1 memory-horizon curve, E2 odometry robustness, E3 crowding, E4 cost scaling accepted (none needs training); E6 and E7 accepted; **E5 dropped** -- and noted as already covered, since E6 and E7 together *are* the per-track recurrent baseline; **E8 (DROW) postponed** until the FROG architecture is settled, because the plan is to carry it across by parameters only.
- **Two corrections recorded for the write-up.** Li2Former's crashes were DirectML-era, and stage 3 is attention that trains fine on ROCm, so "this setup cannot train attention" no longer holds. And the memory **does** contain recurrence: `state = keep * state + (1 - keep) * update` with a learned per-slot decay is a diagonal RNN. The claim that survives is object-keyed memory carried as state at constant per-frame cost, rather than re-encoding a T-scan window.

## Resume verified end to end by killing a run mid-flight (2026-09-24)

*keywords:* A51, resume, --resume, save_resume, a51_resume_check

The owner asked for autosave-and-resume to be armed before any further long run. Unit tests covered the machinery; this covered the path. A `step3a` run was hard-killed at epoch 3 and relaunched with the identical command:

| | epoch 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| val AP | 89.85 | 91.41 | 91.58 | **91.87** | 91.98 | 91.61 |
| gate | +0.024 | +0.025 | +0.026 | **+0.031** | +0.031 | +0.032 |

The second leg logged `resuming step3a_resume.pth: chunk stage 0, 3 epochs done, 0.07 h already spent`, started at **epoch 4**, and carried the validation curve and the gate forward -- a restart would have begun near 89.85% with the gate at +0.024. It then finished the stage, scored the test split (80.17% wp-AUC against its candidates' 77.66%, 4.38 ms per frame) and wrote its results, so the whole path after a resume works, not only the restore.

Note the run itself is a 6-epoch toy at four validation windows, kept only as the resume fixture; its accuracy is not an A51 result.

## Every trainer can resume, after an AMD driver bugcheck took a healthy run (2026-09-23)

*keywords:* A51, resume, --resume, save_resume, load_resume, rng_snapshot, rng_restore, EarlyStopping.state, BestCheckpoint.state, amdkmdag.sys, bugcheck

- **The crash.** 2026-09-22 23:52, bugcheck `0x7E` with exception `0xc0000005`; event 1019 names **`amdkmdag.sys`**, the AMD kernel-mode graphics driver (32.0.31041.1004, 2026-08-17). It killed `a51_d1_sensor_s1` at 42 epochs and 3.5 h, with validation AP 77.91% and still improving. Only bugcheck in eight days, no TDR events. The minidump needs elevation and a debugger, neither available here, so no call stack.
- **`--resume` on every step** (`step1`, `step2`, `step3a`, `step3b`), off by default. After each evaluation a run writes `<step>_resume.pth` next to its checkpoints: weights, optimizer, scheduler, early-stopping counters, the best-checkpoint bar, the periodic-checkpoint schedule, history, elapsed hours and both random streams. The file is written beside the target and renamed, so a crash during the save cannot leave rubble in its place, and a file that will not load is ignored rather than fatal.
- A resumed run is **not** bit-identical to an uninterrupted one -- GPU nondeterminism already means a seed does not reproduce a run -- but it continues from the last evaluation instead of from nothing, and the stopper and best-checkpoint bars survive, so a worse epoch after a crash cannot overwrite a better model.
- Tests: `TestResume` (6), covering the random-state round trip, the disk round trip, both "nothing to resume" paths, a truncated file, and the two bookkeeping bars; 84 pass in the trainer file.

## Capacity is not the limit, reporting is; O1 failed twice with no established cause (2026-09-22)

*keywords:* A51, recovery, --recovery, label_coasting_reports, coast threshold, coast_weight, O1, O2, S3, a51_o1_bounded_s0, a51_o1_bounded_s0b, training_fault

- **`memory_diagnosis.py --recovery`** (new, tests first) settles where the remaining work goes. Of the 26,996 people stage 3 does not report at 0.3, **a live slot already sits on 25,778 (95.5%)** with a median value of 0.986, while the memory uses a median of **24 slots of 256** (max 107). **Spare capacity and more candidates are not the lever**, so A51 S3 is deprioritised.
- **Coasting cannot be fixed by thresholding**: 3.0 false positives per person recovered at 0.3, 4.6 at 0.2, 11.1 at 0.1. The learned head already beats ranking the same reports by the rule's accumulated value (5.1 each at 2,946 recovered), so it is not ignoring the memory -- it cannot separate a remembered person from a remembered phantom. `step3a --coast-weight` is O2's first lever.
- **O1 failed twice and the cause is not established.** Both attempts died of a NaN loss in the 10 s chunk stage (4.4 h with the scalar gate, 2.4 h without) after the 1 s stage ran cleanly at val AP 77-80%. Dropping the gate did **not** fix it, so the "two multiplicative scales" reading was wrong; the decay layer does not separate the arms either (norm 16.06 against B0's 14.52 at the same stage, near-identical implied decay times). O1 is 2 of 2 failures against 8 of 8 completions elsewhere, but the same window held two genuine hardware failures, so attribution stays open and the next known-good runs decide it.
- **The `training_fault` guard earned its place**: it stopped both runs at the first bad epoch, and its candidate-AP check showed the data was sound (72.85% throughout) where the 2026-09-21 fault had corrupted it.
- A relaunch was also lost when a tool call timed out and killed the launcher's process tree; two minutes, not hours.

## Diagnosis and oracle refreshed on the current model, and `memory/slot-design-evidence.md` (2026-09-21)

*keywords:* A51, slot-design-evidence, memory_diagnosis, three_horizon_oracle, object_memory, capacity override, a51_b0_memory_diagnosis.json, a51_b0_oracle.json

- **New doc `memory/slot-design-evidence.md`**: every stage-3 design decision with the measurement that supports it, or an explicit note that nothing does, plus how to read the numbers (screening threshold, missing calibration-metric threshold, one seed each). Indexed in `README.md` and `keywords.md`.
- **Bug found by running them:** `object_memory()` took `rules` positionally and also accepted it as an override, so every caller that replays a trained memory at another slot count (`memory_diagnosis.py`, `three_horizon_oracle.py`, `run_ablations`) raised `TypeError`. It now takes `capacity=` and keeps the checkpoint's own slot rules, which the naive fix would have dropped silently for a `latest_value` checkpoint. Regression test added; 75 pass.
- **Diagnosis on B0's best seed** (`a51_b0_memory_diagnosis.json`): at the shared 0.3 threshold stage 3 covers 128,895 of 153,655 people against stage 2's 136,417, at **1.092 false positives per frame against 2.628**. Of the 10,197 it loses, **99.1% were pushed under by rescoring** and **100% had a live slot inside the gate** (median value 0.966, matched 0.0 s earlier). The head pushes people down harder (-3.526 logits, 97.3%) than candidates covering nobody (-0.729, 69.6%).
- **Oracle on the same checkpoint** (`a51_b0_oracle.json`): at a 2 s bridge 10,171 people are recoverable and stage 3 takes **21.7%**; at 10 s, 13,067 and 17.6%. Stage 2 mostly tracks 71.1% of trajectories, mostly loses 7.4%, and only 0.7% of people sit on trajectories it never detects.
- Both scripts had last run on the old chain, so their previous figures are retired.

## A transient device fault, and a guard that catches it in one epoch (2026-09-21)

*keywords:* A51, training_fault, CANDIDATE_AP_TOLERANCE, HIP out of memory, bad allocation, a51_d1_sensor_s1

D1's confirmation seed failed twice, in two different ways, and neither was the arm's doing.

- **Host memory.** The first attempt died 3.5 h in with `RuntimeError: bad allocation` inside `loss.backward()`. `train_memory` held the train, val and test candidate caches (614 + 998 + 254 MB on disk) in RAM at once on a 16 GB machine, although test is only scored at the end. It now loads test after training and frees the other two first.
- **Device state.** The relaunch diverged at epoch 12: the *candidates'* val AP fell from 72.85% to 3.40% -- a number read from the cache that no model state can change -- with the loss at exactly 0.0000. Six epochs ran on garbage before a HIP OOM that claimed 14.87 GiB was free. A GPU sanity check (matmul against CPU, attention-shaped ops) passed afterwards, so the card is fine and the fault was transient.
- **Guard.** `training_fault` stops a run at the first evaluation whose candidate AP drifts past `CANDIDATE_AP_TOLERANCE` or whose loss is zero or non-finite. Tests: `TestTrainingFault` (4); 74 pass in the trainer file.
- **D1 still rests on seed 0 alone.** Reboot before retrying the seeds.

## Slot-design ablation D3: a fixed decay costs 0.90 pp and 17,536 parameters less (2026-09-21)

*keywords:* A51, D3, a51_d3_fixeddecay_s0, fixed_decay_s, selective recurrence, decay

`step3a --fixed-decay-s 8`, one seed, 352 min, 288,907 parameters against B0's 306,443 -- exactly the decay layer and its bias.

- Test AP **82.44% against B0's 83.34% mean (-0.90 pp)**; recall at the candidates' false-positive rate **92.1%**, below every B0 seed (92.2 / 92.5 / 92.7); coasting 0.084 per frame at 25.4% precision.
- The direction supports the learned, input-dependent decay, but the gap is inside the ~1.5 pp threshold on one seed.
- **Phase 3b at one seed each: D1 -1.32, D2 +0.60, D3 -0.90.** The memory as a whole is worth +5.68 pp over its own candidates, yet no single design choice inside it clears the threshold. D1, the central claim, gets the confirmation seeds first.

## Slot-design ablations D1 and D2: room coordinates earn 1.3 pp, accumulated value earns nothing (2026-09-21)

*keywords:* A51, D1, D2, a51_d1_sensor_s0, a51_d2_latest_s0, room_frame, latest_value, value rule

One seed each on the no-temporal stage 2, short curriculum, against B0's 83.34% mean and a ~1.5 pp threshold.

- **D1, slots in sensor coordinates: 82.02% (-1.32 pp)**, recall at the candidates' false-positive rate 92.2%, B0's lowest seed; 431 min. The design's central claim is supported, though by less than the threshold on one seed, and the number bundles the lost association that the sensor frame also causes when the robot moves further than the gate in a frame.
- **D2, a slot held by the latest score instead of accumulated value: 83.94% (+0.60 pp)**, recall 92.7% (B0's best), coasting precision 31.7%; 700 min. **The ablation does not lose.** Accumulated value, argued in `docs/PROPOSAL.md` §5.5, earns nothing measurable on FROG -- consistent with S1's finding that the bookkeeping rules do not bind, and suggesting the learned slot state already covers a briefly hidden person. It also weakens D4's prospects.
- D2's mid-run slowdown (1,383 s epochs against ~640 s) was transient and returned to 629 s; inference is unchanged at 4.44 ms per frame.

## T2 screened: the plateau schedule matches T1's calibration gain, which makes both readings ambiguous (2026-09-20)

*keywords:* A51, T2, a51_t2_s0, plateau, schedule, own_threshold, coast_precision, calibration threshold

`step3a --schedule plateau`, no-temporal stage 2, seed 0, short curriculum, 500 min.

- Test AP **83.89%** against B0's 83.34% mean, inside the ~1.5 pp threshold and inside B0's own 82.96-83.88 range.
- Recall at the candidates' false-positive rate **93.2%** and coasting precision **32.0%**, both above every B0 seed -- almost exactly T1's 93.4% and 32.2%.
- **Two unrelated changes producing the same calibration lift is ambiguous**: either anything that lets the rescoring head train harder does this, or the calibration metrics vary more across runs than B0's three seeds show. No threshold was pre-registered for them, so neither arm is adopted; T1's confirmation seeds move to phase 4.
- 516 tests pass (full suite, run with the GPU idle).

## T1 screened: dropping the scalar rescore gate buys calibration, not AP (2026-09-20)

*keywords:* A51, T1, a51_t1_s0, rescore_gate, own_threshold, coast_precision

`step3a --no-rescore-gate`, no-temporal stage 2, seed 0, short curriculum, 367 min, 306,442 parameters (B0: 306,443 -- exactly the dropped scalar).

- Test AP **84.39%** against B0's 83.34% mean (82.96 / 83.17 / 83.88). **+1.05 pp is inside the ~1.5 pp screening threshold**, so by the pre-registered rule this is not a win on AP.
- **Recall at the candidates' own false-positive rate: 93.4%**, above every B0 seed (92.2 / 92.5 / 92.7), at a lower own threshold (0.140).
- **Coasting precision 32.2%** against 22.3-27.4% for B0, also above every seed, at 0.225 reports per frame.
- The rescoring shift on non-people moved from about +0.2 to **-0.87 logits**, i.e. the head now pushes phantoms down rather than nudging everything, which is what removing the zero-scaled gradient was meant to allow.
- One seed. The arms that follow are judged the same way, and the calibration metrics -- not AP -- are the ones T1 and O1 aim at.

## Stage 3 gets a plateau schedule and single-pass test scoring; T1 and T2 screening (2026-09-20)

*keywords:* A51, T1, T2, T4, memory_schedule, --schedule plateau, --test-strides, a51_t1_s0, a51_t2_s0

- `memory_schedule` gives `step3a` the choice between its cosine schedule (the default, which spans `--max-epochs` per chunk length and so hardly decays in the epochs a stage runs) and step 2's plateau schedule on val AP, with `--lr-factor`, `--lr-patience` and `--min-lr`.
- `--test-strides` scores the test split once rather than twice, saving ~25 min per screening run with no effect on training.
- Shrinking validation was considered and deferred: it changes checkpoint selection, so it would cost a 3-seed re-baseline of B0 before any arm could be compared with it.
- T1 (`--no-rescore-gate`) started screening at 03:22, verified by its parameter count, 306,442 against B0's 306,443; T2 (`--schedule plateau`) follows. Both: no-temporal stage 2, seed 0, short curriculum, judged against B0's 83.34% mean with a ~1.5 pp threshold.
- Tests: `TestMemorySchedule` (2); 122 pass in the two three-horizon files.

## B0: the stage-3 baseline on the no-temporal backbone, +5.68 pp over its candidates, seed spread 0.47 pp (2026-09-19)

*keywords:* A51, B0, a51_b0_short_s0, a51_b0_short_s1, a51_b0_short_s2, noise floor, screening threshold, own_threshold, coast_precision

Three seeds of `step3a --chunk-s 1 10` on the no-temporal stage 2 (seed 0), scored on the whole FROG `official` test split against identical candidates (77.66% AP, 88.8% recall at 2.620 FP/frame).

- Memory AP **82.96 / 83.17 / 83.88%** (mean **83.34**, sd 0.47); AP at 0.3 m 81.36 / 81.52 / 82.17%; recall at the candidates' own false-positive rate 92.2 / 92.5 / 92.7% against 88.8%.
- **The A51 screening threshold is ~1.5 pp of test AP**, half of stage 2's ~0.95 pp per-run spread.
- Stage 3 alone runs at 4.5-4.6 ms per frame, one stream.
- The rescoring shift persists at every seed (-2.8 to -3.5 logits on people), and coasting reports 0.08-0.29 per frame at 22-27% precision, where the 1-epoch smoke had 0.002 at 0%.
- Runs cost 4.1-9.7 h through early stopping alone; validation is ~70% of a 1 s epoch, which moves the validation-cost item to the front of phase 1.
- The full-curriculum run (with 30 s chunks) was stopped 1.9 h in, inside its first stage, by the owner's call; whether the long chunks add anything is still open.
- Not comparable with the old 80.10%: different backbone, and that number predates the running time step and the limit-beam decoding.

## Two stage-3 rescoring arms, off by default: no scalar gate (T1) and a bounded correction (O1) (2026-09-18)

*keywords:* A51, T1, O1, rescore_gate, rescore_limit, --no-rescore-gate, --rescore-limit, object_memory

`ObjectMemory` takes `rescore_gate` and `rescore_limit`; `step3a` exposes them as `--no-rescore-gate` and `--rescore-limit`. Both defaults reproduce the current model exactly, so B0 is unaffected and each arm is screened separately.

- **T1.** `candidate_logit = logit(score) + gate * mlp(...)` with the gate starting at zero scales the head's own gradient by zero: a test asserts the rescoring head's gradient is 0 over the first steps with the gate and non-zero without it. `--no-rescore-gate` drops the scalar and zero-initialises the head's last layer instead, which starts equally silent.
- **O1.** `--rescore-limit L` bounds the correction to +-L logits through a tanh, against the signed correction that shifted the whole distribution down.
- **Loading.** A T1 checkpoint has no `rescore_gate` parameter, so `object_memory(checkpoint, feature_dim, **overrides)` now rebuilds a stage-3 memory the way it was trained, as `calibration_network` does for stage 2. `train_joint`, `run_ablations`, `memory_diagnosis.py` and `three_horizon_oracle.py` all go through it.
- Tests: 3 in `TestObjectMemory`, 2 in `TestObjectMemoryFor`; 508 pass.

## Slot bookkeeping is not the bottleneck: a matched slot sits on 97% of annotated people at every gate and fade (2026-09-18)

*keywords:* A51, slot-rules, replay_slot_rules, covered_people, retirement_gap, gate_m, fade_time_s, GATES_M, FADE_TIMES_S, slot_rules.json

`memory_diagnosis.py --slot-rules` (new, tests first: `TestCoveredPeople`, `TestRetirementGap`) replays `manage_slots` with no trained model over the whole FROG `official` test split on the no-temporal seed-0 candidates, sweeping 3 association gates x 3 fade times.

- **Nothing binds.** People with a live slot within 0.5 m: 99.0-99.3% across gates 0.3-1.0 m and fades 2-30 s; people whose slot also matched a candidate that frame: 96.9-97.0%. Matches per annotated frame 9.33-9.47.
- **So the remaining loss is downstream of the rules**, in the learned head, which extends A50 phase 1b item 1 from the lost people to all of them. S4 (ablating the hand-coded association) is dropped as not worth a training run, and A51's priority moves to calibration and coasting.
- **Returns are rare and short.** At the default 0.6 m / 10 s a slot is lost 0.10 times per frame and something reappears within 0.5 m of it 0.02 times per frame, median gap 3.0 s, 17.8% beyond the fade. Carrying state across retirement would address a thin slice, so S2 stays parked.
- **Capacity note:** 0.3 m / 30 s reaches 242 live slots of 256 at p99, the first setting where capacity would bind.
- Coverage counts any live slot within the radius, so it is an upper bound (`docs/PROPOSAL.md` §5.5), and FROG has no identities, so a return is evidence rather than proof that the same person came back.

## Stage-3 time step: stamped intervals corrupted slot velocity, now a running average; stage-3 evaluation reports calibration, coasting and 0.3 m AP (2026-09-17)

*keywords:* A51, slot-speed, windowed_speed, mean_frame_period, frame_periods, DT_AVERAGE_FRAMES, Split.dt, manage_slots, MIN_DT_S, FRAME_PERIOD_S, own_threshold, rescore_shift_person, coast_precision, coasting_hits, memory_ms_per_frame, extra_radii, ap_0.3m

The owner confirmed a stage-3 slot plan, now `TODO.md` A51, on the no-temporal stage 2.

- **0a, measured.** `memory_diagnosis.py --slot-speed` replays the rule-based `manage_slots` on the old test candidate cache (50,088 frames, no model). Mean frame period is 24.99 ms; 20.2% of stamped intervals are clipped to 1 ms. With stamped `dt`, matched slots near people carry a velocity of p50 3.97 / p90 8.63 m/s (63.9% above 3 m/s) against a displacement-over-1-s reference of 0.66 / 1.67 m/s, and slots elsewhere drift at p50 10.83 m/s. With the mean period the state reads 0.58 / 1.16 m/s against a reference of 0.56 / 1.10 m/s, and matches per annotated frame rise 8.51 -> 8.84. Every stage-3 result so far, 80.10% included, was trained and scored with this corruption. Tests: `TestWindowedSpeed` (4).
- **0c, implemented.** `evaluate` adds stage 3's own operating point at the candidates' FP per frame, the median rescoring shift on person and other candidates, coasting reports at 0.3 per frame and their precision (`coasting_hits`), and AP at extra match radii; `memory_ms_per_frame` times one stream. Step 3a's test report prints them. Validation selection is unchanged, so no checkpoint or cache is invalidated. Tests: 2 in `TestScoreDetections`, `TestCoastingHits` (3), 2 in `TestEvaluate`, `TestMemoryMsPerFrame`.
- **The first B0 launch failed and cost a night.** `memory_ms_per_frame`'s `torch.cuda.synchronize()` guard read `device.type`, but `detect_device()` returns the string `"cuda"`; the unit test passed a real `torch.device` and so missed it. It crashed after the smoke run's training, the chain stopped by design, and the GPU idled for 19 hours. Fixed to `str(device).startswith("cuda")`, the test now covers both forms, and the whole end-of-run block was re-run against `detect_device()`'s own value on the GPU before relaunching (stage 3 alone: 4.42 ms per frame). Recorded in `memory/gotchas.md`.
- **0b, fixed.** The planned rule (time since last match) would not have worked, since for a slot matched every frame it equals the stamped step. The owner chose a running average instead: `frame_periods` is the plain mean of the intervals for the first `DT_AVERAGE_FRAMES` = 32 frames of a recording and an exponential average with rate 1/32 after, stored as `Split.dt` and used in the joint path too. `FRAME_PERIOD_S = 1/26.2` is gone; chunk and validation lengths use `mean_frame_period` of their split, so they are 1.53x longer in frames than before. Replayed on the test split, velocity reads p50 / p90 0.58 / 1.17 m/s near people against a 0.56 / 1.10 reference, matches per frame 8.84, and `dt` stays in 24.23-25.82 ms (p0.1-p99) after warm-up. Tests: `TestFramePeriods` (7). Nothing trained is invalidated that was still loadable: every stage-3 checkpoint already predated the current stage 2.

## Fresh controls turn the limit-beam "loss" into a tie, and show seeds do not pair runs (2026-09-17)

The default architecture was retrained on current code at seeds 0-2 (`checkpoints_three_horizon/default_s*`, 1,177,923 parameters). Test wp-AUC was **78.19 / 76.64 / 75.13%** (mean 76.65, sd 1.53; shuffled history 77.52 / 76.05 / 74.48).

- **Limit-beam exclusion is a tie, not a loss.** Its mean of 76.99 is +0.34 above the fresh controls, inside the spread. Training stays `--limit-beams-in-loss train` under the pre-registered rule, and the flag, `training_beams` and `calibration_loss` were removed (owner's call; 476 tests pass).
- **Same seed, different run.** The old controls, rescored with the same decoding, were 0.21 / 1.64 / 1.96 above the fresh ones at the same seeds. GPU nondeterminism means a seed does not reproduce a run, so paired-by-seed comparisons are invalid. That retracts the "−0.93 on every seed" reading below and the "t = 3.4" static comparison. Pooled per-run sd across five three-seed arms is ~0.95 pp (`memory/interpreting-evaluation.md`).
- **Stage 2 for the slots session:** `default_s0` (78.19%).

## Excluding limit beams from the training loss: first read as a 0.9 pp loss, superseded above (2026-09-17)

`step2 --limit-beams-in-loss exclude` (new; tests in `TestLimitBeamsInLoss`) drops beams at the range limit from stage 2's classification and vote loss, with class balance over the kept beams. Three seeds on the default architecture (1,177,923 parameters) scored **77.68 / 76.72 / 76.58%** test wp-AUC (mean 76.99; shuffled history 76.96 / 75.94 / 75.69). The same-seed controls score 78.40 / 78.28 / 77.09 (mean 77.92) under the same decoding. That is lower on every seed (−0.72 / −1.56 / −0.51, mean −0.93), so by the pre-registered rule the default stays `train`. Limit beams appear to matter as negatives even though they cast no vote at decoding. Minor confound: the controls predate `anchor_ranges` in target construction. The flag was removed afterwards (entry above).

## Beams at the range limit cast no vote, now the default: +0.34 pp on all six checkpoints (2026-09-17)

Evaluation-only experiment (TODO A50), `evaluate_calibration_rules` / `duration_curve --silence-limit-beams`: both rules from one forward pass, on the default-architecture (formerly "no-fine") and no-temporal ("static") best checkpoints at seeds 0-2, FROG `official` test.
Plain → silenced wp-AUC:

- no-fine: 78.11 → 78.40, 77.92 → 78.28, 76.70 → 77.09;
- static: 77.22 → 77.60, 77.04 → 77.40, 76.40 → 76.68.

Every pair improves (+0.28 to +0.39, mean +0.34). The gain comes from recall (FN −0.007 to −0.012 per frame) with FP almost unchanged, so the silenced votes were displacing real detections rather than creating phantoms.
**Adopted as the default decoding rule** (owner's call): `voting_beams` in `train_three_horizon.py` is applied in candidate decoding and in evaluation; the one-off comparison option and flag were removed, and network candidate caches are keyed `-limit-silenced` so stale ones rebuild. Three-seed test means under the default decoding: **default architecture 77.92, no-temporal 77.23**. Naming: "no-fine" runs are the default architecture now that `--fine-lags` is gone; "static" is no-temporal. 476 tests pass.

## Fine convolution removed; missing returns handled by one configurable range rule (2026-09-17)

Owner's calls: the fine temporal convolution is removed from the three-horizon detector as a tested hypothesis that turned out not to be needed (`memory/rejected-ideas.md`), and missing returns are handled by a simple normalisation with a configurable limit instead of loader preprocessing (TODO A49/A50).

- **The rule:** `sanitize_ranges`. A reading that is non-finite, non-positive, or at or beyond `max_range_m` (`--max-range-m`, default 10 m) reads as `max_range_m`. It handles FROG's `+inf` and 61.0 m and DROW's 29.96 m without knowing the dataset. The validity input channel is gone (11 -> 10 features).
- **Removed:** `align_beams`, beam alignment and ranges in `CausalTemporalConv`/`TemporalCache`, `CalibrationState.fine`, `--fine-lags`. `clip_length` is `max(coarse_lags) + 1`.
- **Loader:** `FROG_Dataset(missing_return_m=...)` keeps the legacy 10 m clamp as its default for the baselines. The three-horizon trainer passes `None` and reads raw scans.
- **Old checkpoints:** a one-tap checkpoint (no-fine, static) folds exactly into the new stem (`fold_legacy_calibration`): 1x1 stem and 1x1 fine convolution compose, and the always-1 validity column goes into the bias. On 2,000 real test scans the folded stem matches the old computation to 3e-6. Multi-tap checkpoints are refused with a clear error. `calibration_network` rebuilds any checkpoint from its stored `coarse_lags` and `max_range_m`; `duration_curve` and `load_split` use it, fixing `load_split`'s old habit of building a default-shaped network.
- **A second consumer of raw ranges was found by re-scoring.** Vote decoding and training targets anchor on each beam's range, and they would have seen raw `inf`. The first re-score gave 78.25 / 77.47% where 78.03 / 77.22% was recorded, because `evaluate_calibration` zeroes non-finite beams. They now use `anchor_ranges`, the network's own sanitised ranges (2 new tests, both failing against the old anchoring).
- **One convention change remains, and it is deliberate.** The old decoding anchored finite readings beyond 10 m at their true range, up to 60 m, although the network saw 10 m. Patching that back reproduces static seed 0's recorded test result exactly (AP 77.218%, P 49.484%, FP 2.7765, FN 0.348). Under the new convention the same weights score **78.11%** (no-fine seed 0, recorded 78.03) and **77.22%** (static, P 50.5%, FP 2.658). The shifts are ≤0.1 pp of AP, inside the per-run spread. Recorded numbers stay as they are; any new comparison re-scores its control with the current code.
- 474 tests pass; markdown lint clean; ruff is not installed (TODO).

## Scan anomaly checker: FROG is 40 Hz, writes no return two ways, and stage 3 divides by sub-millisecond steps (2026-09-17)

`utils/scan_anomalies.py` (TODO A49; 22 tests in `tests/test_scan_anomalies.py`) reads raw scans and reports, grouped by denoising mechanism: missing-return encodings, how long and wide invalid readings stay, spikes against steps, jitter by range, annotations, and timestamps.
Its first pass found three things that were not the noise it was built for:

- **FROG records at 40 Hz, not 26.2 Hz.** Its timestamps come in bunches (about 38 ms, 38 ms, then under 0.1 ms). The mean over linked intervals is 25.0 ms in all five files, no two consecutive scans are identical, and the median gives 26.2. Every FROG duration in seconds derived from 26.2 Hz is 1.53x too long, including the coarse lags (0.80 s, not 1.22 s) and `fp_per_s`. The blast radius is listed in TODO A49; nothing has been changed in code yet.
- **FROG encodes no return two ways.** The test file and the three extras use `+inf` (17.6M readings in total). `frog_11-36_12-43_train_val.h5` has no inf at all and writes exactly **61.0 m** instead (7.46M readings), past the sensor's 60 m maximum. The loader clamps only non-finite values to 10 m, and `beam_features` clips 61.0 to 10 m, so today both reach the network as a valid 10 m. Removing the clamp alone would make train and test disagree.
- **Stage 3 divides by stamped intervals.** They are clipped to 1 ms, and `manage_slots` computes velocity as displacement over the step. This is a new unmeasured suspect for stage 3's losses (TODO A50 phase 1b).

The checker's first sentinel search was fixed before any result was used. It treated every centimetre-rounded DROW value as a peak, and it could not see 61.0 above its 60 m histogram.

## A50: the memoryless static arm ties no-fine at three seeds (2026-09-17)

Static (no fine, no coarse, 977,795 parameters) scored **77.22 / 76.92 / 76.31% wp-AUC** at seeds 0/1/2, mean **76.82** (sd 0.46), identical with shuffled history each time.
No-fine, which keeps the 1.2 s coarse horizon, scored 78.03 / 77.89 / 76.60, mean **77.51** (sd 0.79).
The seed ranges overlap, so the pre-registered rule calls it a tie.
**Retracted pairing (seeds do not reproduce runs, entry above):** per-seed leads of +0.81, +0.97, +0.29 (mean +0.69) are not a paired result; only the means compare. The coarse horizon may be worth well under 1 pp on FROG `official`, and three seeds cannot confirm even that.
Written up in TODO A50; the owner reads a tie as a problem for the temporal hypothesis.

## DROW records at 12.7 Hz, not 10 Hz, and the sensors' specs are now sourced (2026-09-17)

While looking up DROW's laser for the anomaly checker (TODO A49), the paper's 12.5 Hz disagreed with the repo's measured 10.0 Hz.
**DROW's timestamps are rounded to 0.05 s.** Across all splits there are 263,738 intervals of 0.100 s and 194,481 of 0.050 s, so `1 / median(diff(time))` returns 10 Hz. Scans over elapsed time per file gives 12.41-12.71 Hz (median 12.69), which matches the paper.
Four analysis scripts use the median (`motion_analysis`, `slot_budget`, `horizon_sweep`, `three_horizon_oracle`), so their DROW durations in seconds are 1.27x too long. The fix and re-run are queued in TODO A49 and not yet applied. `memory/dataset-properties.md` is corrected: DROW annotations are 0.39 s apart, not 0.5 s, and a `T=5, dtime=10` window spans 3.9 s, not 5.0 s.
Sensor specs now come from sources in the new `memory/noise-structure.md`:

- UTM-30LX, per Hokuyo: 0.1-30 m, with ±30 mm up to 10 m and ±50 mm up to 30 m.
- S300, per SICK's data sheet: a 30 m measuring range, with no range accuracy published.

The owner's idea to remove the FROG loader's clamp of non-finite ranges to 10 m is recorded in TODO A49, with its retraining blast radius.

## A50 confirmation: five fine taps and none tie, and the screening noise floor was three times too small (2026-09-17)

*keywords: A50, confirmation, seeds, noise floor, run-to-run spread, fine_lags default, step2, screening protocol, retraction*

**What was confirmed.** Owner's call: take the two best fine-horizon arms, 5 taps and no-fine, to three seeds each. The pre-registered rule: the higher mean becomes the `--fine-lags` default, while means within the seed-to-seed range count as a tie, recommended to no-fine for cost and decided by the owner. Seeds 1 and 2 ran with the new `clip_length` default (37 and 33 frames), and every run matched its pre-registered parameter count.

| arm | seed 0 | seed 1 | seed 2 | mean | sd |
| --- | --- | --- | --- | --- | --- |
| no-fine (1 fine tap) | 78.03% | 77.89% | 76.60% | **77.51%** | 0.79 |
| 5 fine taps | 78.56% | 77.32% | 76.63% | **77.50%** | 0.98 |

**A tie.** The means are identical, so the default stays at 3 taps pending the owner's decision, with no-fine recommended for cost (6.54 against 9.27 ms per frame).

**The larger result is the spread.** The per-run sd of 0.79-0.98 points is about three times the 0.30 points that screened every arm on 2026-09-16. That figure came from one pair of runs that also differed in code, so it was never a variance estimate. At ~0.9 points per run, two single runs must differ by ~2.5 points to count.

**Retracted as rankings:**

- dropout 0.1 over the control (+0.70);
- every temporal ablation: no-fine +1.41, no-coarse +0.39, static +0.60, width-matched −0.23, skip +0.99;
- 5 taps over the control (+1.94), and 7 taps under 5 (−1.42);
- the mechanisms argued from them.

**Still standing:**

- the median denoiser's −9.16, far outside any spread;
- the within-run decline of the duration curve (−3.58), though on one seed;
- findings about one fixed model, which carry no seed variance: the stage-3 threshold calibration, the memory diagnosis, and the `--clip-frames` bug.

**The protocol did what it was built for**: confirmation caught a screening winner that was noise. What changes is that single-run screening can only rank effects above ~2.5 points; anything smaller needs seeds from the start (`memory/interpreting-evaluation.md`).

## A50 phase 1b: stage 3's "lost" people were a threshold artefact, and at its own operating point the memory strictly improves on its input (2026-09-16)

*keywords: A50, phase 1b, memory_diagnosis.py, --stage3-threshold, rescore_gate, threshold calibration, distribution shift, false positives per frame, manage_slots, slot retirement, slot capacity*

**Owner's priority**, after item 1 found stage 3 taking only 17.5% of the available headroom while losing 6,076 people it had already detected: "stage 3 underperforming is exactly what we should put an effort into ... do the slots preserve enough information? Is the slot invalidation rule enough? Are there enough slots?"

**Two of the three were already settled and were not re-run.** Slot *capacity* is not the limit: step 4 scored 64 / 128 / 256 slots at 80.12 / 80.20 / 80.10%, and A48's occupancy work found the memory never fills at 256, so no slot is ever evicted for capacity — which is also why `docs/PROPOSAL.md` §5.5's rule-versus-oracle comparison stopped testing anything. The invalidation rule therefore could only fail through *retirement* (value fading to `floor` over `fade_time_s` = 10 s) or *association* (mutual-nearest within `gate_m` = 0.6 m of the velocity-predicted position).

**`utils/memory_diagnosis.py` split the loss by mechanism**, reading only values `ObjectMemory.step` already returns, so it needed no library change and could not disturb a training run. Over the whole test split, of 6,119 lost people: **99.4% to rescoring, 0.6% to anything else.** A live slot sat within the gate for **100%** of them, median value 0.793, matched 0.0 s earlier. Association and retirement are exonerated, and phase 1b's slot-state and hand-coded-association items drop down the list.

**The mechanism is a distribution shift, not selective suppression.** The people stage 3 *kept* were pushed down harder (median **-3.115** logits) than the ones it lost (**-1.394**). `candidate_logit = logit(score) + rescore_gate * rescore(...)` moves the whole distribution down, which leaves rank-based AP better (+3.2) while silently decalibrating a threshold fixed at stage 2's 0.3.

**Priced at a matched error rate, the memory wins on both axes**, which the shared-threshold comparison had hidden:

| | covers of 153,655 | recall | false positives / frame |
| --- | --- | --- | --- |
| stage 2 at 0.30 | 134,196 | 87.34% | **1.910** |
| stage 3 at 0.30 | 130,131 | 84.69% | 1.293 |
| **stage 3 at 0.22** | **135,520** | **88.20%** | **1.767** |
| stage 3 at 0.20 | 136,643 | 88.93% | 1.925 |
| stage 3 at 0.15 | 139,055 | 90.50% | 2.474 |

At 0.22 stage 3 covers 1,324 more people than its own input *and* emits 0.143 fewer false positives per frame. At 0.20, where false positives match, it covers 2,447 more. The first reading of 0.15 — 139,055 covered, the loss collapsing from 6,119 to 609 — was **not** a gain: it buys those people at 2.474 false positives per frame, above stage 2's 1.910, which the tool's own printed criterion rejects.

**The false-positive counting was validated before its conclusions were used**: at 0.3/0.3 it reproduces the published 1.910 and 1.293 to three decimals despite matching people with a gated Hungarian assignment where `_prec_rec_2d` matches greedily by score, and the same pass reproduced 134,196 / 130,131 / 6,119 / 2,054 exactly, confirming the instrumentation changed nothing it measured. Two bugs were caught by the smoke-run-first discipline rather than by a full pass: a `frames` accumulator shadowed by the segment-unpacking variable of the same name, and `duration_curve.py` building networks with default lags so that no ablation checkpoint could load at all (fixed by `network_for`, which rebuilds from the lag lists in each checkpoint's own `args`).

**What follows.** Report stage 3 at its own operating point rather than stage 2's, and treat "the memory buys 8 points of precision for 2.6 of recall" as an artefact of the shared threshold (`memory/performance-log.md` footnote 5, corrected). The remaining work on stage 3 is calibration of the rescore head — not slots, not association, not architecture.

## A50 phase 2: both temporal ablations beat the control, and nothing we checked explains it (2026-09-16)

*keywords: A50, phase 2, --fine-lags, --coarse-lags, step2_no_fine, step2_no_coarse, step2_static, temporal ablation, shuffled history, val scoring, duration_curve.py --split, network_for, balanced artefact test, screening*

**What was screened.** Phase 2's first two ablations needed no model change: `CausalTemporalConv` sizes its convolution as `in_channels * len(lags) + pose_channels`, and `pose_channels` is zero when no lag is non-zero, so `lags=(0,)` degrades cleanly to a 1x1 convolution on the current frame. `step2 --fine-lags` and `--coarse-lags` were added and a third arm run with both removed. All settings match the dropout-0.0 control, so each arm differs from it in exactly one variable.

**Every arm was verified twice before its result was read.** Each ablation changes the parameter count by an amount predictable from the layer it shrinks, and each prediction was registered before the run: 1,179,011 for no-fine (`Conv1d(96,32,1)` -> `Conv1d(32,32,1)`, -2,048), 979,843 for no-coarse (`Conv1d(652,128,3)` -> `Conv1d(128,128,3)`, -201,216), 977,795 for both. All three matched exactly, so no flag was silently ignored. The static arm then scored **identically ordered and shuffled** (77.22% / 2.776 FP / 0.348 FN both ways), which is the behavioural proof: with no temporal taps, shuffling the past cannot change the answer.

| arm | test AP | against control | shuffled-history penalty | parameters |
| --- | --- | --- | --- | --- |
| control, both convolutions | 76.62% | — | 1.18 pp | 1,181,059 |
| fine only (`--coarse-lags 0`) | 77.01% | +0.39 | 0.18 pp | 979,843 |
| **coarse only (`--fine-lags 0`)** | **78.03%** | **+1.41** | 0.65 pp | 1,179,011 |
| static, neither | 77.22% | +0.60 | **0.00 pp** | 977,795 |

**Read against the static arm, not the control.** The **coarse** convolution adds **+0.81 pp** over no temporal information at all (78.03 against 77.22, about 2.7x the 0.30 pp run-to-run spread), so the ~1.2 s horizon earns its place. The **fine** convolution alone lands at 77.01 against static's 77.22, *inside* the noise floor, so nothing is claimed for it either way — but adding it on top of the coarse one costs **-1.41 pp**, the largest effect measured. This is not "less information beats more": it is one horizon helping and the other not.

**A fifth arm isolated why the fine one hurts, and it is the width rather than the temporal content.** `--fine-lags 0 0 0` taps the current frame three times: the same 96-channel width and parameter count as the control (1,181,059, matched exactly), no alignment invoked because lag 0 skips it, and no history at all — confirmed by its shuffled-history penalty falling from the control's 1.18 pp to **0.43 pp**. It scores **76.39%**, sitting with the control's 76.62% (-0.23, inside the 0.30 pp noise floor) and not with no-fine's 78.03% (-1.64, far outside). Three identical copies of the current frame cost as much as three real past frames, which pointed at the layer's width. **A sixth arm refuted that.** `--fine-lags 0 1 2 3 4` is wider still at 160 channels, and the pre-registered prediction was that it would land at or below the control's 76.62%. It scored **78.66%**, the best of every arm (shuffled 77.51%, penalty 1.15 pp; 1,183,107 parameters, matched). What the arms support is the owner's original reading, not width. At 0.08 s, three taps score the same whether real (76.62%) or copies of the present (76.39%), so there is no usable signal at that span. At 0.15 s — DR-SPAAM's window — five taps score 78.66% and rely on history. Why three signal-free taps score ~1.5 points *below* a single tap (78.03%) instead of matching it remains unexplained. A seventh arm, `--fine-lags 0 2 4`, uses three taps over the same 0.15 s at 96 channels. It scored **77.61%**, almost exactly midway between the control (76.62%) and 5-tap (78.66%). So span and tap count each account for about a point, and neither alone explains the 5-tap gain.

**An eighth arm replaced the fine convolution with parameter-free denoising, and it failed badly.** Settings were the owner's: window `0 1 2 3 4`, tolerance 0.3 m, run as `--fine-lags 0 --fine-denoise 0.3`. It scored **68.87%**, 9.16 below no-fine's 78.03% on an otherwise identical network. Shuffled history fell to 43.84%. Implementing it surfaced a real bug first: `align_beams` carries a past value unchanged to its new beam rather than re-measuring distance from the current pose. So `project_ranges` was added, sharing the projection through `_land`, with 7 new tests (466 pass). **The failure is in the outlier rule, not the code.** On real test frames projection leaves consecutive-frame disagreement unchanged (1.30 cm raw, 1.35 cm projected; the robot moves 1.24 cm per frame). But the rule replaces **10.32% of readings near annotated people against 3.45% elsewhere**, 3.0x. A leg crossing a beam appears in only a few of the five frames, so the window median is background and the leg is discarded as an outlier. The ratio *rises* to 3.5x at 1.0 m tolerance, so no tolerance fixes it. A median-based "most recent non-outlier" rule erases exactly the thin moving objects this detector exists to find. **Decided (owner, 2026-09-16): dropped**, since a plain convolution is cheap and already gives the best result. The code and its 16 tests were removed on 2026-09-17, and `align_beams` returned to its original inline body.

**The 5-tap and skip results carry a caveat, found afterwards from their checkpoints.** `--clip-frames` defaults to 35, which is enough history only for three fine taps (`max(coarse_lags) + max(fine_lags) + 1` = 32 + 2 + 1). Both runs had a fine lag of 4 and needed 37, so at the oldest coarse tap the two oldest fine frames were copies of the clip's first frame. Training and test scoring used the same clips, so each number is internally consistent. Streaming inference, which builds stage-3 candidates and would run on the robot, reads the real frames instead. Of the fine-horizon arms only the control and no-fine are clean.

| fine taps | span | ms / frame | fine cache | clip frames needed |
| --- | --- | --- | --- | --- |
| 1 | 0 s | 6.54 | 90 KB | 33 |
| 3 | 0.076 s | 7.92 | 270 KB | 35 |
| 5 | 0.153 s | 9.27 | 450 KB | 37 |
| 7 | 0.229 s | 11.70 | 630 KB | 39 |
| 9 | 0.305 s | 12.28 | 810 KB | 41 |

Measured with streaming `step`, stages 1-2 only, on the RX 9060 XT under ROCm; the frame budget at 26.2 Hz is 38.2 ms. Each tap adds 2,048 parameters (0.17%).

**Re-running at the correct clip lengths settles it: five taps, not seven.** 5 taps at `--clip-frames 37` scored **78.56%**, within noise of the clipped run's 78.66%, so the clipping had not moved that number. 7 taps at 39 scored **77.14%**, 1.42 below (about 4.7x the 0.30 pp spread), and its validation peaked early, at epoch 0.50. That fits the architectural reading: over seven taps (0.229 s) a walking person moves ~0.32 m, more than a leg's width. A per-beam window then stops seeing one surface and starts seeing a trail, which is the coarse convolution's job. `--clip-frames` now defaults to `clip_length(fine_lags, coarse_lags)`, so a lag change cannot leave the clip short again. Single seed; the confirmation stage decides.

**Two explanations were proposed and both failed on checking.** First, that our odometry-aligned taps reproduce a known DROW weakness that DR-SPAAM fixed with attention — but the repo's own table labels DROW3's FROG number `T=1`, a single-scan result that says nothing about temporal fusion, and `docs/RESEARCH.md` records DROW aligning by **rotation only**, where `align_beams` re-projects endpoints through world coordinates, translation included. Our design also already addresses all four of DROW's listed weaknesses. Second, that a lagged tap is a hole-punched resampling — but `test_the_fine_cache_is_almost_hole_free_at_its_own_lag` pins `filled.mean() > 0.98` at exactly this lag. Neither survives. A third — that the taps' near-duplicate content sits at the sensor's noise scale and the optimiser fits it to training-recording specifics — was refuted by **measurement** rather than by reading, by the width-matched arm above. Width was proposed as the positive answer and then refuted by the 5-tap arm above, which is wider and scores best. So four explanations went on reading or measurement, and a fifth on a failed pre-registered prediction. What is left is the owner's framing: a 0.08 s window carries no usable temporal signal and a 0.15 s window does.

**Validation could not adjudicate it.** Scored on every one of val's 122,405 annotated frames (`duration_curve.py --split val`, which needed `network_for` to rebuild each network from the lag lists in its own checkpoint, since a default-shaped network cannot load an ablation's weights at all): control 73.79%, no-fine 73.88%, no-coarse 72.95%, static 72.78%, dropout 73.89%. Val agrees with test that no-fine and dropout are the best two, but cannot resolve the 1.41-point gap between no-fine and control — they are tied on val either way — and disagrees outright elsewhere, ranking static last where test ranks it third of five (`memory/interpreting-evaluation.md`).

**Decided (owner, 2026-09-16): park it and let `balanced` adjudicate.** The temporal ablations are re-run there at the end of phase 3 as an artefact test. Because `balanced` holds 16-41 whole, one evaluation pass yields both its test recordings separately, so the effect can be compared on 16-41 against 15-53 **with identical weights, varying only the frames** — which isolates recording-specificity in a way no cross-model comparison can. Two follow-ups were also queued on the reading that the fine horizon's job is denoising rather than information: five taps instead of three (`--fine-lags 0 1 2 3 4`, DR-SPAAM's window, needs no code), and replacing that convolution with parameter-free outlier rejection (`TODO.md` A50).

**All of it is single-seed screening** against a 0.30 pp spread, and nothing here is confirmed.

## A50 phase 1, item 4: dropout on stage 2, and the first measurement of test AP against training duration (2026-09-16)

*keywords: A50, dropout, --dropout, --checkpoint-every, PeriodicCheckpoint, step2_control, step2_dropout01, duration_curve.py, overfitting on test, screening and confirmation, seeds, EPOCH_TOLERANCE*

**Owner's suggestion**, after item 2 attributed step 3b's regression to stage 2: if stage 2 is overfitting, add or raise its dropout. Checking first found something simpler — **the network had no dropout at all**, while every other trainable detector here trains at `train.py`'s default of 0.5 (as does DR-SPAAM's own published config), and `docs/RESEARCH.md` §5.4 documents 0.1 as this project's intent for its own architectures, an intent never wired in. `docs/PROPOSAL.md` specifies no regularisation for this model either.

**Two opt-in options landed** (433 tests pass), both default-off so every result so far is reproduced and every existing checkpoint still loads, since neither holds parameters:

- `step2 --dropout` applies `nn.Dropout` to the **head's input only**, leaving the decoder features stage 3 pools from deterministic. Every evaluation path (`network_candidates`, `evaluate_calibration`, `evaluate`, `evaluate_joint`) already calls `.eval()`, so cached candidates cannot be contaminated.
- `step2 --checkpoint-every` keeps a checkpoint per interval through a new `PeriodicCheckpoint`, mirroring `BestCheckpoint` rather than extending it, so the "strictly better only" guarantee that protects against optimizer restarts stays intact. An evaluation landing a little early still counts (`EPOCH_TOLERANCE`), because step 2's evaluations land at 0.2499, 0.4998 and so on rather than on exact quarters.

**A control run was necessary, not optional.** The 76.92% baseline cannot be reproduced under current code: its stored args carry no `schedule`, `lr_factor`, `lr_patience` or `min_lr`, so it predates the plateau scheduler, and a lone dropout run would have differed from it in two variables. Running dropout 0.0 on today's code gives a clean comparison, and incidentally shows the baseline reproduces within 0.30 pp — both runs peaked at epoch 1.00, before the scheduler first cut the rate at 2.25, so the scheduler did not select either checkpoint.

| | best val (epoch) | test AP | P | R | FP | FN | shuffled |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original step 2 | 73.78% (1.00) | 76.92% | 58.4% | 87.3% | 1.910 | 0.388 | 75.62% |
| control, dropout 0.0 | 73.88% (1.00) | 76.62% | 50.7% | 88.9% | 2.653 | 0.341 | 75.44% |
| **dropout 0.1** | 73.93% (0.75) | **77.32%** | 60.3% | 87.6% | 1.770 | 0.379 | 75.32% |

**Test AP against training duration, measured for the first time** (`duration_curve.py`, every annotated test frame per checkpoint):

| epoch | control AP | P | R | dropout AP | P | R |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | 75.26% | 48.8% | 87.8% | 75.72% | 46.9% | 90.6% |
| 1.00 | **76.62%** | 50.7% | 88.9% | **78.16%** | 55.6% | 88.5% |
| 2.00 | 75.39% | 62.9% | 82.9% | 75.93% | 65.0% | 81.5% |
| 3.00 | 73.90% | 71.7% | 77.7% | 74.78% | 70.1% | 79.0% |
| 4.00 | 73.04% | 73.5% | 76.5% | — | | |

**Three findings.** *Stage 2 overfits on test*, not only on validation: AP peaks at epoch 1.00 and falls 3.58 pp by epoch 4, while precision climbs from 48.8% to 73.5% and recall falls from 87.8% to 76.5% — the network grows steadily more conservative past the AP-optimal point. This answers the question item 2 could not, because no over-trained checkpoint existed then, and it revises item 2's conclusion. *Dropout lifts the whole curve* — +0.46, +1.54, +0.54 and +0.88 points, all four paired comparisons in the same direction — *but does not flatten it*: the control loses 2.72 pp from epoch 1 to 3 and dropout 3.38 pp, so it improves the peak rather than curing the decline. *Validation mis-selects*: it chose the dropout run's epoch-0.75 checkpoint (77.32% on test) when epoch 1.00 scores 78.16%, costing 0.84 pp. The reported figure stays the val-selected one, since choosing on test would be selection on the test set.

**How much of this is noise.** The two dropout-0.0 runs differ by 0.30 pp of test AP, so dropout's +0.70 pp clears that by roughly 2.3x — but the same pair differs by **0.74 false positives per frame** and **7.7 points of precision** (58.4% against 50.7%), nearly as much as dropout's own 0.88 and 9.6, so the whole operating point at the 0.3 threshold — precision, recall and false positives alike — is suggestive only. The shuffled-history penalty is firmer: the two dropout-0.0 runs agree at 1.30 and 1.18 pp against dropout's 2.00 pp, meaning the regularised network leans *more* on temporal evidence. Validation showed none of the gain at all (73.88% against 73.93%, inside its own >1.5 pp swing between adjacent evaluations).

**Decided (owner, 2026-09-16): no bespoke confirmation run for dropout.** Dropout is one model option among ~14 across phases 2 and 3, and single runs cannot resolve effects of this size, so all of them are judged by one procedure — screening at one seed against the shared control, then confirmation of the survivors alongside the control at further seeds, recorded in `TODO.md` A50. Every number states its seed count; dropout's +0.70 pp is provisional until confirmation.

## A50 phase 1, item 2: step 3b's regression is stage-2 weight drift, and its feedback prior never activated (2026-09-16)

*keywords: A50, step3b, step3b_joint.best.pth, aggregate.weight, prior_channels, weight drift, coarse.conv.weight, BestCheckpoint, best.offer, joint AP selection, evaluate_calibration, prior silenced*

**The question.** Step 3b scored 78.84% on test where step 3a scored 80.10%. Its own numbers localise the loss before any experiment: the memory contributes +3.16 in step 3b against +3.18 in step 3a, so the memory did not regress at all — the whole gap is in the candidates, 76.92% -> 75.68%. Two explanations survived: joint training moved stage 2's weights somewhere worse, or the weights are intact and what the memory draws into the bottleneck is what costs the points.

**Method.** The prior enters only as extra input channels of `aggregate`, so a zero prior contributes `W[:, 2c2:] @ 0 = 0` — algebraically identical to a `prior_channels=0` network holding the trimmed weight `W[:, :2c2]`. Building that network inverts `load_calibration_weights` and lets `evaluate_calibration` score step 3b's stage 2 through the *same* protocol that produced step 2's 76.92% (every annotated test frame, 35-frame clips, batch 64). The run asserts the equivalence numerically before scoring, and the control reproduces `step2_results.json` to 76.9218% against 76.9227%.

| | AP | P | R | FP | FN |
| --- | --- | --- | --- | --- | --- |
| step 2 alone (control) | 76.92% | 58.4% | 87.3% | 1.910 | 0.388 |
| step 3b's stage 2, prior silenced | **75.68%** | 58.4% | **84.5%** | 1.842 | 0.477 |

**It is drift, and the prior is inert.** Silencing the prior changes nothing: 75.68% is *exactly* what step 3b's candidates scored with the prior active, so the feedback path accounts for 0.00 points. Inspecting the weights says why — the prior columns start at exactly zero and ended at norm 0.0485, max |w| 0.0071, against the shared columns' norm 7.08. They never left the neighbourhood of zero, so **step 3b never actually tested feedback**. The damage is entirely recall (87.3% -> 84.5%) at unchanged precision.

**Where the weights moved.** Relative L2 drift is 2.97% over all 1,181,059 shared parameters, concentrated in `coarse.conv.weight` (5.44% over 250,368 params — the ~1.2 s horizon), `head.weight` (5.29%), `down.2.weight` (4.62%) and the GroupNorm biases (5-7.7%).

**Checkpoint selection compounded it.** `best.offer(result["memory"]["ap"], ...)` ranks on the whole detector's AP, never on stage 2's. At epoch 1.0 stage 2 was at its val best (76.12%) with a joint AP of 78.37%; epoch 1.5 won selection at 78.98% joint with stage 2 already down to 75.68%. The criterion preferred the checkpoint whose stage 2 was worse.

**Is it simply "more training"? The evidence leans against it.** `step2_continued` trained stage 2 on for another 3.25 epochs at `lr` 3e-4 with no joint objective and no memory, and its val AP fell monotonically 73.02% -> 69.18%, the same shape as step 2's own decline after epoch 1.0. Only its epoch-0.25 checkpoint was kept; rescored on test through the same protocol, reproducing that run's own recorded 76.91% to the digit:

| | AP | P | R | FP | FN |
| --- | --- | --- | --- | --- | --- |
| step 2's best | 76.92% | 58.4% | 87.3% | 1.910 | 0.388 |
| + 0.25 epochs, ordinary training | **76.91%** | 64.2% | 84.0% | 1.433 | 0.492 |
| + 1.5 epochs, joint training | **75.68%** | 58.4% | 84.5% | 1.842 | 0.477 |

Read alone, that says ordinary continued training **kept AP and only moved the operating point along the curve**, while joint training made the curve itself worse — evidence against "joint training is simply more training". **The duration curve in the next entry overturned that**, and the reason is the caveat recorded when this measurement was made: a quarter-epoch exposure is too short to show the effect, because the decline is slow at first. Scoring a full run every epoch shows ordinary training losing **1.23 pp between epoch 1.00 and 2.00**, as much as joint training's 1.24 pp over 1.5 epochs. "More training" is therefore a sufficient explanation for a loss of this size, and the joint objective is no longer needed to account for it — though not excluded either, since step 3b ran at `lr` 1e-4 against the control's 1e-3, and losing as much at a tenth of the rate may still mean the joint objective contributes something of its own.

**What it means for items 3 and 4.** Item 4's regularisation (owner's suggestion 2026-09-16: this network has no dropout at all, while every other trainable detector here trains at `train.py`'s 0.5, and `docs/RESEARCH.md` §5.4 documents 0.1 as this project's intent for its own architectures) is worth running — but the overfitting case for it is currently **validation-side only**, and that run should keep periodic checkpoints rather than only the best, so test AP against training duration can be measured directly for the first time. Item 3 should rank on a criterion that cannot hide a degrading stage 2, and give the zero-initialised prior a rate that lets it leave zero.

## A50 phase 1, item 1: the temporal oracle replayed on our own candidates (2026-09-16)

*keywords: A50, three_horizon_oracle.py, oracle_ours.json, interpolation bound, recoverable, position-key collisions, official vs balanced split, step3a_object_memory, coverage, bounds*

**The question** (`docs/PROPOSAL.md` §9): how much of the "+9.6 points of recall, from people missed in one frame but detected shortly before or after" does our memory actually recover? That ceiling was measured above **LFE-Peaks**, so it describes LFE's headroom, not ours. `utils/three_horizon_oracle.py` re-measures it above our own stage 2 and, in the same pass, scores what stage 3 did with it.

**The headroom above our own candidates is smaller, and stage 3 takes little of it.** Over the whole FROG `official` test split (50,088 annotated frames, 153,655 person-observations, 672 trajectories):

| | value |
| --- | --- |
| covered by stage 2 (candidates) | 86.1% |
| covered by stage 3 (memory) | 83.5% |
| interpolation ceiling at a 2 s bridge | **92.6%** (+6.5 pp, against +9.6 pp above LFE-Peaks) |
| of that ceiling, taken by stage 3 | **17.5%** (1,769 of 10,080) |
| people stage 2 found and stage 3 dropped | **6,076** (against 2,113 recovered; net −3,963) |
| never detected at all by stage 2 | 0.8% (LFE-Peaks: 3.8%) |

**What it means.** Stage 3's +3.2 points of AP are bought with precision, not with recovered recall: at the 0.3 operating threshold the memory loses nearly three people for every one it recovers. The ceiling is also flat from 10 s to 60 s, so nothing is available beyond 10 s — consistent with the 90%-within-10 s figure measured above LFE-Peaks. This is the evidence item 2 (the step 3b regression) starts from.

**Caveats recorded with the numbers** (`memory/dataset-properties.md`): this coverage is a per-frame gated Hungarian match at one threshold and runs ~1.2 pp below the `recall` column of the result tables; 1.5% of person-observations shared a rounded world position with another person in the same frame and may have coverage attributed to the wrong one; and a single threshold cannot contradict an AP integrated over all of them. **Corrected later the same day:** the ~1.2 pp gap was recorded here as a metric difference, the result tables sweeping a precision-recall curve. It is not — `utils/memory_diagnosis.py` scores the same per-frame coverage without linking trajectories and reproduces the reported recall exactly (87.3% and 84.7%). The gap *is* the position-key collisions, so the two caveats were one cause listed twice. The LFE oracle runs on the `balanced` split and ours on `official`, so the two are recorded separately rather than as rows of one table.

**Built with a test first** (`tests/test_three_horizon_oracle.py`, CPU-only): `bounds` counts a miss as recoverable only when its own trajectory was detected both before and after it within the bridge, measured in seconds against each trajectory's own rate — so a miss past the last detection, or before the first, is never recoverable however long the bridge.

## A48 re-runs after the heading fix; A43 steps 1-2 built, and an annotation-alignment bug caught by the first smoke run (2026-09-15)

*keywords: A48, A43, tracker_sweep, persistence_map, horizon_sweep, motion_analysis, phantom_analysis, slot_budget, slot occupancy spike, displacement oracle, train_three_horizon.py, ObjectMemory, CalibrationNetwork, idet2iscan, enumerate over dict, candidate cache, evaluation windows, GPU memory budget, commit limit, smoke run*

**Owner's request.** Confirmed `.venv` as the one environment; asked to re-run everything downstream of the heading-wrap fixes now, then train (~1 day).

**A48: 13 scripts re-run, all exit 0** (world-frame caches moved aside first -- `.persist_cache`, `.slot_cache` and `.horizon_cache` stored world coordinates computed with the old headings; `.tracker_cache` stores sensor-frame detections and was kept). Against the documented numbers:

| result | documented | re-run |
| --- | --- | --- |
| SORT, LFE-Peaks, `min_hits = 3`: AP / FP / FN per frame | 63.6% / 1.43 / 0.72 | 63.7% / 1.430 / 0.715 |
| SORT exchange rate, LFE-Peaks, `min_hits = 3` | 3.85 | 4.06 |
| recall oracle, LFE-Peaks | 75.9% -> 85.5% | unchanged |
| person trajectories: count / median / max | 1,087 / 4.9 s / 67.7 s | **888 / 6.8 s / 74.8 s** |
| people moved < 0.5 m at 1 / 2 / 5 / 10 / 30 s | 67.3 / 46.6 / 36.0 / 29.5 / 16.3% | 67.4 / 47.0 / 36.5 / 30.3 / 15.3% |
| background model, best exchange rate, LFE-Peaks / LFE-PPN | 1.90 / 3.32 | ~1.80 / 2.92 |
| displacement oracle, best exchange rate | 4.20 at 0.30 m | **2.58 at 0.10 m -- unexplained** |
| slots in use at a 10 s retire window, p99 / max: people, LFE-Peaks, LFE-PPN | 31/40, 55/79, 105/164 | **16/19, 39/44, 73/112** |
| people p99.9 slots, 0 s -> 1 s retire window | 9 -> 28 | **9 -> 10** |
| memory replay at 256 slots, LFE-PPN: mean / p99 / max slots; person-slot replacements | 79.1 / 179 / 256; 2 | **76.0 / 151 / 172; 0** |

**What it means.** SORT got slightly better everywhere and its conclusion stands. Trajectory statistics were being shattered by the wrong headings: fewer, longer tracks now. The slot-occupancy spike left open in `PROPOSAL.md` §15 was the same bug, and the memory never fills at 256 slots, so no replacement happens at all. The displacement oracle's fall from 4.20 to 2.58 is far larger than ~0.14% of wrongly headed frames should cause -- at equal phantom removal it now costs 10.7% of true positives against 6.9% -- and is queued for investigation before `docs/PAPER.md` quotes either number (`TODO.md` A48). DR-SPAAM windowed evaluations (item 6) are still open.

**A43 steps 1 and 2 built, tests first** (`library/follow_the_drow/detectors/three_horizon.py`, `utils/train_three_horizon.py`; 63 tests across `tests/test_three_horizon.py` and `tests/test_train_three_horizon.py`):

- `ObjectMemory` -- stage 3: batched slot bookkeeping with the replay's value rules plus velocity prediction; distance-biased attention with a learned null item for matching, exchange and rescoring; selective recurrence in real time with decays initialised over 1-60 s; a rescoring gate and coasting head that start silent, so the untrained model reproduces the detector exactly.
- `CalibrationNetwork` -- stages 1-2: the U-Net of ConvNeXt blocks, fine and coarse causal temporal convolutions, DROW head; the clip forward encodes only the 5 tapped frames and equals streaming frame by frame.
- `train_three_horizon.py step1 / step2` -- candidate cache, stream sampler for truncated backpropagation, early stopping, windowed validation, test scoring with exchange rate; phase A on 35-frame clips with `train.py`'s DROW head loss and decoding.

**Bug caught by the step-1 smoke run: annotations were attached to the wrong scans on train and val.** Val AP came out at 0.01% for the untouched candidates. Traced: `FROG_Dataset.idet2iscan[s]` is a dict from annotation index to scan index, and both the candidate builder and phase A's frame loader iterated it with `enumerate`, which yields the keys -- so annotation `d` landed on scan `d`. On test every scan is annotated and the two coincide (candidates scored 64.7% AP); on train and val the first scans of each recording are not annotated, so every annotation was shifted (12,574 of 393,820 val people had a candidate within 0.5 m). The unit test missed it because its fake dataset used a plain array for `idet2iscan`; it now mirrors the real dict with annotations starting after scan 0, and a new test covers the frame loader. The broken caches were deleted; caches are now stored as plain dicts of arrays, because the first ones pickled their class as `__main__.SequenceCandidates` and could not be loaded outside the script.

**Two machine limits, found by crashed smoke runs** (`memory/gotchas.md`): one process gets ~10.7 GB of the 16 GB GPU under ROCm on Windows (measured by allocating until failure), and killed processes release VRAM lazily; the machine's commit limit (31 GB) runs ~26 GB full with other programs, so two torch processes plus a dataset load fail. Step 1's batch was halved to 1,048 frame-steps (4.3 GB measured), and the two training runs will be sequential. Stage 3 validation streams 64 evenly spaced 60 s windows after 30 s of warm-up, because val's three recordings are ~65,000 frames each.

**Smoke runs after the fix, then launch.** Step 1 on 3 recordings: val AP of the untouched candidates **81.8%** (was 0.01%) on 4 windows -- a small sample, to be checked against the full validation before it is read as a property of val -- and test 66.6%, the untrained memory reproducing it exactly. Step 2 on the full data for 0.03 epochs: 0.21 s per batch of 32 clips after warm-up (~12 min per epoch, 1.3 GB of GPU memory), val AP 52.7% -> 58.5% -> 59.2% on 64 frames, test on every 500th frame 61.3% in order against 58.8% with shuffled history. A finished smoke run hung at exit for ~7 hours overnight, unnoticed, because it predated the forced exit; the step-1 and step-2 runs were launched at 08:37 with a watcher on progress, errors and stalls.

**The launched run stopped at 10:28, and it lost its best checkpoint first.** Step 1's 1 s stage ran all 40 epochs at ~2.6 min each: val AP rose from 64.3% to a best of **68.8%** at epoch 37, against **60.7%** for the untouched candidates (+8.1 pp), mostly through fewer false negatives (0.61 -> ~0.50 per frame) with false positives flat at ~3.0; it was still creeping up when the epoch cap ended it. The 10 s stage then ran at ~9.5 min per epoch (4 streams x 262 frames, 5.4 GB). Two failures:

- *The best checkpoint was overwritten.* Every chunk-length stage restarted early stopping, and "best" with it, while writing the same file -- so the 10 s stage's first epoch after the optimizer restart (66.6%) replaced the 68.8% model. Fixed with `BestCheckpoint` (tested: a worse later epoch or stage, or an undefined metric, never replaces the best), per-stage best files alongside, and stages resuming from the run's best.
- *The machine slept.* The System log shows a kernel-power shutdown transition and "system initiated reboot from Sleeping (Idle)" at 10:29:54, 110 s after the last log line; no out-of-memory, bugcheck or unexpected-shutdown record. The script now calls `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`, which holds only while the process lives (`memory/gotchas.md`).

Step 1's default budget is now 100 epochs per chunk length (owner's call: 1 s epochs are fast and the stage ended on the cap, not on patience). Step 2 never started.

**Owner's call: go straight to our own network -- step 2, then 3 and 4; step 1 dropped.** Step 2 relaunched 13:24 as a detached process. Its first evaluations: val AP 70.5% at 0.25 epoch, 73.8% at epoch 1, ~11 min per epoch with evaluations, 1.3 GB of GPU memory.

**Steps 3a, 3b and 4 built while step 2 trains, tests first** (405 tests pass):

- *Step 3a*: `network_candidates` streams a recording through the trained stage 2, decodes candidates with DROW's vote grid (owner's choice, so step 3's AP stays comparable with step 2's), and pools each candidate's decoder features from the beams whose votes land within the grid's collect radius (`candidate_pooling_weights`); candidates are cached with float16 features and stage 3 trains on them with step 1's loop.
- *Step 3b*: `render_prior` draws the slots' predicted positions into 2 channels at the bottleneck (largest value and person probability per sector); `CalibrationNetwork(prior_channels=2)` starts them at zero, and `load_calibration_weights` loads step 2's weights into it unchanged -- tested: identical outputs until trained. `joint_step` runs prior -> stage 2 -> decoding -> stage 3 on one frame, with candidate features pooled differentiably so stage 3's loss reaches stage 2 (tested), and `joint_frame_loss` adds stage 2's DROW head loss to stage 3's. One stream, 10 s chunks, learning rate 1e-4.
- *Step 4, evaluation-time ablations only* (owner's choice): SORT replayed on stage 2's candidates at `min_hits` 1 / 3 / 5 (`replay_sort`, ego-motion from the poses as `tracker_sweep.py`), and stage 3 and the whole detector at 64 / 128 / 256 slots.

**Step 2 result: 76.9% test AP, and its history matters.** Stages 1-2 alone (no object memory), weights of epoch 1.0 selected on val AP, every annotated frame of the FROG test recording:

| | AP @ 0.5 m | FP / frame | FN / frame |
| --- | --- | --- | --- |
| clip history in order | **76.9%** | 1.91 | 0.39 |
| clip history shuffled | 75.6% | 1.89 | 0.42 |
| LFE-Peaks / LFE-PPN, measured here | 65.1% / 69.6% | 1.76 / 5.03 | 0.63 / 0.38 |
| DR-SPAAM `T=5`, published | 75.6% | -- | -- |

Both step-2 gates pass: above LFE's published AP, and shuffling the history costs 1.3 pp (this project's SpaceTimeCNN lost 0.5 pp). The margin over DR-SPAAM is our measurement against a published one, one seed -- recorded in `memory/performance-log.md` with that caveat. Val AP peaked at 73.8% at epoch 1.0 and early stopping ended the run at epoch 4.0 with training loss still falling (0.183 -> 0.076): the cosine schedule over 20 epochs had left the learning rate at ~95%. **Owner's call:** continue from the best weights with a plateau schedule (x0.3 per stalled epoch, floor 1e-5, from 3e-4), now step 2's default; that continuation started automatically at 14:24 ahead of steps 3a -> 3b -> 4.

**The continuation did not help.** Val AP 73.0% at epoch 0.25, then falling at every evaluation to 69.2% at epoch 3.25 (training loss 0.132 -> 0.036), through rate cuts to 9e-5 and 2.7e-5; its best weights test at 76.9% / shuffled 75.6%, with fewer false positives (1.43 per frame) and more misses (0.49). Stage 3 therefore builds on the first run's weights (best val 73.8%). A guard script made that swap, but 4 s after step 3a had started loading the checkpoint, so step 3a was restarted on the first run's checkpoint path. Stages 1-2 overfit within about one epoch; regularisation is queued as a future experiment in `TODO.md` A43.

**Vote-grid fix: tied maxima made NaN detections.** Step 3a crashed 7 min into building its candidate cache with `matrix contains invalid numeric entries` from the Hungarian candidate match. Root cause, in `votes_to_detections` (`library/follow_the_drow/utils/drow_utils.py`): two neighbouring grid cells whose blurred votes tie exactly both pass the 3x3 maximum filter, each vote goes to its nearest maximum, and a maximum left without votes was averaged over nothing into a NaN position and probability. A random search over small vote sets reproduced it (8 votes, a diagonal tie); such a peak is now skipped, pinned by `tests/test_vote_decoding.py`. Every decoding through the vote grid changes -- step 2 and 3 of `train_three_horizon.py`, `train.py`'s evaluation, `render_video.py` -- but rarely: NaN comparisons are False, so a NaN detection passes every distance gate and `_prec_rec_2d` could count it as a true positive, yet step 2's test set held 2 among 800,856 detections and its AP moves from 76.9227% to 76.9218%.

**Stage 3 evaluation in slices.** Step 3a trained for 6 h (1 s, 10 s and 30 s chunks; best val AP 77.32% at 30 s, epoch 5, against 73.19% for its stage 2 candidates), then its test pass failed allocating 1.66 GB: `evaluate` gathered 32 whole test recordings of up to 6,796 frames at once. It now gathers 512 frames at a time and carries the slots across slices, pinned by a test that slicing changes no metric; the saved checkpoint is evaluated on test without retraining.

**Step 3a result: the object memory adds 3.2 points on test.** Every test recording played from empty memory:

| | AP @ 0.5 m | FP / frame | FN / frame |
| --- | --- | --- | --- |
| stage 2 candidates alone | 76.92% | 1.91 | 0.39 |
| **stages 1-3, with the object memory** | **80.10%** | **1.29** | 0.47 |
| same, every 5th annotated frame | 80.15% | 1.29 | 0.47 |
| DR-SPAAM `T=5`, published | 75.6% | -- | -- |

The memory removes a third of the false positives at the cost of 0.08 missed people per frame (7.6 removed per person lost). The candidates' row is identical to step 2's own test result, so the two harnesses agree. One run, with stage 2 frozen; joint training (step 3b) follows.

**Precision and recall are now reported beside AP** (owner's request, 2026-09-16), so a result can be read against the recoverable-recall ceiling of `docs/PROPOSAL.md` §2.5: `score_detections` returns `precision` and `recall` at the 0.3 operating point next to the per-frame false-positive and false-negative rates it already had, and every test line of steps 3a, 3b and 4 prints all four. The ceiling itself was measured on LFE-Peaks' detections, so replaying the oracle on our own candidates is queued in `TODO.md` A43 before any "how close are we" claim.

**Step 3b: joint training with feedback is a regression.** Three epochs with stage 2 and stage 3 trained together at 1e-4, the memory's predicted positions drawn into stage 2's bottleneck:

| | AP @ 0.5 m | FP / frame | FN / frame |
| --- | --- | --- | --- |
| step 3a, memory on a frozen stage 2 | **80.10%** | 1.29 | 0.47 |
| step 3b, joint with feedback | 78.84% | 1.24 | 0.55 |
| stage 2 alone | 76.92% | 1.91 | 0.39 |
| stage 2 with feedback, after joint training | 75.68% | 1.84 | 0.48 |

Joint training cost 1.26 points, and stage 2 lost 1.24 points of its own -- while step 3b's validation rose from 74.06% to 76.12%, which says more about its 16 validation windows (6 with no annotated frame) than about the method. Step 3a's weights stay the best model; the candidate explanations (weak validation signal, one learning rate for a fine-tuned stage 2 and a from-scratch memory, 3 epochs) are in `TODO.md` A43.

**Step 4: the gain is the learned memory, not temporal filtering in general.** Every row is the FROG test recording, every annotated frame, at the 0.3 operating point:

| on stage 2's candidates | AP @ 0.5 m | precision | recall | FP / frame | FN / frame |
| --- | --- | --- | --- | --- | --- |
| candidates alone | 76.92% | 58.4% | 87.3% | 1.910 | 0.388 |
| SORT, `min_hits` 1 | 76.66% | 59.1% | 86.8% | 1.844 | 0.404 |
| SORT, `min_hits` 3 | 75.73% | 60.4% | 85.5% | 1.722 | 0.446 |
| SORT, `min_hits` 5 | 74.92% | 61.1% | 84.4% | 1.646 | 0.479 |
| stage 3, 64 slots | 80.12% | 66.3% | 84.9% | 1.325 | 0.463 |
| **stage 3, 128 slots** | **80.20%** | 66.7% | 84.7% | 1.298 | 0.469 |
| stage 3, 256 slots | 80.10% | 66.8% | 84.7% | 1.293 | 0.470 |
| whole detector with feedback, 64 / 128 / 256 slots | 78.86% / 78.91% / 78.84% | ~67% | ~82% | ~1.24 | ~0.55 |

Three findings. **SORT lowers AP at every setting** -- it trades recall for precision about one for one, so filtering harder only costs more (exchange rates 4.15, 3.27, 2.91), while stage 3 gains 3.2 points over the same candidates and beats the strictest SORT by 5 points of precision at a similar recall. **The slot budget does not bind**: 0.1 points separate 64 from 256 slots, for both variants. **Precision is the remaining weakness**: a third of reported detections are still false (66.8% precision at 84.7% recall), where the candidates alone are 58.4% at 87.3%.

## A43 step 0 built: input features, cache alignment, causal temporal convolution -- and two heading-wrap bugs fixed on the way (2026-09-14)

*keywords: A43, A48, step 0, three_horizon.py, beam_features, align_beams, align_sectors, CausalTemporalConv, TemporalCache, test_three_horizon.py, _load_odom, np.unwrap, heading wrap, cutout, drow_utils, odometry jumps, training budget, epochs, patience*

**Owner's request.** Approved the build plan (step 0 interface and caches; step 1 stage 3 on saved LFE-Peaks candidates; step 2 stages 1-2 on ROCm) and asked for epochs and early stopping to be decided per stage before any training, from architecture and epoch time, with changes to them kept as future experiments.

**Built** (`library/follow_the_drow/detectors/three_horizon.py`, 21 tests in `tests/test_three_horizon.py`, written first and watched failing on the missing module):

- `beam_features` -- the 11 per-beam channels of `PROPOSAL.md` §5.3. Missing returns are non-finite or non-positive (FROG test recording: 7.8% of beams are `inf`, 28 `nan`, none `0`); returns beyond 10 m are clipped but stay valid (23% of beams). Tests pin hand geometry, bearing invariance of a leg, and the median channel.
- `align_beams` -- projects each past endpoint through world odometry onto the current beams, nearest return winning, empty beams zero.
- `align_sectors` -- rotation shift of bottleneck sectors plus the past pose in the current frame as 3 channels.
- `CausalTemporalConv` -- clip `forward` for training, `step` with a functional `TemporalCache` for streaming, every past tap re-projected *directly* into the current frame so both paths share one alignment and agree exactly (tested frame by frame at lags `[0, 1, 2]` and `[0, 8, 16, 24, 32]`, plus causality, start-of-clip padding and cache reset).
- Slot eviction stays in `utils/slot_budget.py` with its existing test until step 1 uses it.

**One test expectation was wrong, not the code.** The static-scene test first demanded 90% of beams filled after 0.36 m / 8.6° of motion and got 89.3%. Diagnosed before changing it: filled beams land within 8 mm median, 1.5 cm p90; pure rotation leaves only field-of-view edges empty; the rest are disocclusion and returns spread thinner under translation, growing with motion -- 99.0% filled at 2 frames, 84.0% at 32. The test now checks accuracy under large motion and fill at the fine cache's own lag.

**Real-data check** (`AGENTS.md` rule 4; FROG test recording, 47 sequences, every 25th frame, world endpoints of frame `t - k` aligned onto frame `t`, beams under 10 m; after the fix below):

| lag | re-projected: median / p90 / within 5 cm / filled | no correction: median / p90 / within 5 cm | heading sign flipped: median |
| --- | --- | --- | --- |
| 1 frame, 0.04 s | 1.5 cm / 5.0 cm / 88% / 98% | 1.4 cm / 5.1 cm / 87% | 2.0 cm |
| 2 frames, 0.08 s | **1.6 cm / 6.2 cm / 84%** / 97% | 2.1 cm / 8.9 cm / 77% | 3.3 cm |
| 8 frames, 0.31 s | **2.2 cm / 16 cm / 72%** / 93% | 6.6 cm / 63 cm / 44% | 11.4 cm |
| 32 frames, 1.22 s | **4.4 cm / 72 cm / 54%** / 80% | 25.9 cm / 198 cm / 19% | 46.1 cm |

Re-projection is neutral at one frame and wins from two frames on, by 6x in median at the coarse convolution's 1.22 s reach; the flipped control loses everywhere, so the odometry convention is right on real data. At 32 frames 20% of beams have nothing projected onto them -- a cost of the beam cache only; the coarse cache uses sectors.

**Bug 1: FROG heading was interpolated straight through 0° at every ±180° crossing** (`FROG_Dataset._load_odom`). The first real-data run showed pairs where uncorrected scans agreed to 1-3 cm while re-projection missed by metres. Traced: the published 10 Hz heading steps `180 -> -179°` (a real 1° turn) and `np.interp` on the raw numbers produced loaded scan headings `180.0, 142.6, 109.4, -29.7, -173.5` across 40 ms frames. Fixed by interpolating `np.unwrap`ped heading and wrapping back; test in `tests/test_frog_splits.py`. On the real test recording, consecutive-scan heading changes above 2°: **54 -> 2** (both across 124-248 ms recording gaps); p99.9 **7.73° -> 0.81°**; max 143.8° -> 4.3°. About 25 crossings, ~3 bogus frames each.

**Bug 2: `cutout()` shifted DR-SPAAM-style windows by an unwrapped heading difference** (`drow_utils.py`), so a turn across ±180° moved every window 2π, off the scan. `aligned_raw_scan()` in the same file already wrapped. Fixed and tested (`tests/test_geometry.py`: `170° -> -170°` must equal `0° -> 20°`).

**Blast radius.** Everything that read FROG heading through the loader, at ~0.14% of frames: the tracker's ego-motion (`tracker_sweep.py`, the SORT exchange rates in `docs/PAPER.md`), the world-frame analyses (`persistence_map.py`, `motion_analysis.py`, `phantom_analysis.py`, `temporal_oracle.py`, `horizon_sweep.py`, `slot_budget.py`), and every `--align-scans` training window touching a crossing. Everything that built cutouts with `T > 1` across a crossing, on FROG or DROW. Heading differences in `_EgoMotion` and `train.py` are unwrapped too, but only feed `cos`/`sin`, where 2π is harmless. Queued as `TODO.md` A48; checkpoints in §B.

**Tests: 347 passed** (324 before, 21 + 2 added).

**Training budgets proposed, owner to decide** (`TODO.md` A43): phase A 20 epochs, evaluated every 0.25 epoch, patience 12 evaluations, at least 2 epochs, 24 h cap; step 1 / phase B 40 epochs per chunk length `[1, 10, 30]` s, patience 8, at least 5, 6 h cap; phase C 3 epochs, every 0.125, patience 8, at least 0.5, 24 h cap. Grounded in: FROG authors' DROW3/DR-SPAAM 5 epochs at batch 8, LFE's 100/150 epochs with patience 20, this project's runs peaking at epochs 3-14, and a stride-9 FROG training subsample scoring identically to the full split.

## A43 hardware: ROCm on Windows works and is the fastest device for both stages (2026-09-14)

*keywords: A43, DirectML, torch_directml, CPU, ROCm, RX 9060 XT, torch 2.9.1+rocm7.2.1, .venv, requirements.txt, training throughput, access violation, 0xC0000005, hw_probe, stage 2, stage 3*

**Question.** `PROPOSAL.md` §15 left CUDA availability open, and it decides training chunk lengths. Owner's device order: DirectML, then CPU.

**Found.** torch 2.4.1+cpu with `torch-directml`, AMD RX 9060 XT, 16 GB RAM. No CUDA. ROCm 7.2.1 with PyTorch 2.9 does list the RX 9060 XT on Windows (AMD's Windows support matrix, "the entire ROCm stack is not yet supported on Windows"); it would need its own venv, since `torch-directml` pins torch 2.4.1. Not installed.

**Measured.** The A43 sketch at default sizes (stage 2 decoding only the clip's last frame; stage 3 attention + recurrence, no slot management or matching loss), forward + backward + SGD step, minimum of the last two of three iterations, each case in its own subprocess with 4 CPU threads:

| case | DirectML ms/iter | CPU ms/iter |
| --- | --- | --- |
| stage 2, batch 1 x 3 frames | 42.6 | 21.6 |
| stage 2, batch 1 x 35 frames | 62.9 | 150.0 |
| stage 2, batch 4 x 35 frames | **158.3** (39.6 per clip) | 720.6 (180.2 per clip) |
| stage 3, batch 1 x 26 steps (1 s) | 144.6 | 144.6 |
| stage 3, batch 1 x 262 steps (10 s) | 4143.5 | 1455.0 |
| stage 3, batch 4 x 262 steps | 4765.2 | **4313.1** |

**What it means.** Stage 2 belongs on DirectML: one pass over FROG's 108,356 training frames as 35-frame clips is ~72 min there against ~5.4 h on CPU. Stage 3 runs hundreds of small per-step operations, where DirectML's dispatch overhead cancels its throughput, so it is no faster and is 3x slower at batch 1. The first probe, at batch 8 with every case in one process, died with an access violation (exit 0xC0000005) and no readable output, so batch 8 on DirectML stays untested. The two identical 144.6 ms stage 3 figures are a coincidence to recheck, not a finding.

**ROCm environment prepared, not installed.** Owner's request: a `requirements.txt` for training with ROCm PyTorch, to review before creating the venv. The root `requirements.txt` pins AMD's ROCm 7.2.1 SDK wheels and `torch 2.9.1+rocm7.2.1` (cp312, win_amd64) by direct URL, then `-e ./library` and the runtime packages at the versions working in `.venv312`. All seven AMD URLs answered HTTP 200 (torch wheel 783 MB, SDK wheels 615 + 222 + 467 MB). `numpy` stays 1.x because `library/pyproject.toml` requires `numpy~=1.24`; its `torch~=2.0` is satisfied by the ROCm build, so the library install does not pull a CPU torch. `torchvision`/`torchaudio` are omitted because nothing imports them, and `torch-directml` because it pins torch 2.4.1.

**ROCm installed and measured.** The owner's first venv failed with "is not a supported wheel on this platform": a bare `python` resolves to 3.14 on this machine and every wheel is cp312. Recreated as `.venv` with Python 3.12.10: `torch 2.9.1+rocm7.2.1`, `torch.cuda.is_available()` true on the RX 9060 XT, the library and its C++ extension import, and **324 of 324 tests pass**. `.venv312` no longer exists. The same probe, same sizes and protocol:

| case | ROCm ms/iter | CPU in `.venv` (torch 2.9.1) | DirectML (above) | CPU in `.venv312` (torch 2.4.1, above) |
| --- | --- | --- | --- | --- |
| stage 2, batch 1 x 3 frames | 17.4 | 26.2 | 42.6 | 21.6 |
| stage 2, batch 1 x 35 frames | 33.3 | 175.6 | 62.9 | 150.0 |
| stage 2, batch 4 x 35 frames | **120.4** (30.1 per clip) | 770.7 | 158.3 | 720.6 |
| stage 3, batch 1 x 26 steps | 114.3 | 1388.6 | 144.6 | 144.6 |
| stage 3, batch 1 x 262 steps | 1228.3 | 14483.8 | 4143.5 | 1455.0 |
| stage 3, batch 4 x 262 steps | **1291.4** (323 per chunk) | 53100.7 | 4765.2 | 4313.1 |

**What it means.** ROCm is the device for both stages. Stage 2: 30.1 ms per clip, ~54 min per pass over FROG's training frames. Stage 3: 323 ms per 10 s chunk at batch 4, 3.3x the old CPU, and batch 1 to 4 costs only 5% more time, so larger batches are nearly free. **CPU is not a fallback for stage 3 in this venv**: torch 2.9.1's CPU path is ~10x slower than 2.4.1's on this workload (13.3 s against 1.08 s per chunk) -- cause not investigated.

## PROPOSAL.md finalised for outside readers, built explicitly on three temporal horizons (2026-09-14)

*keywords: docs/PROPOSAL.md, three temporal horizons, calibration, short, long, building blocks, configuration names, deferred alternatives, mermaid diagram, efficiency, references*

**Owner's request**: finalise `docs/PROPOSAL.md` for the owner's review and for collaborating researchers -- efficiency counts, architecture, named configuration parameters with defaults and reasoning, alternatives deferred to experiments, and a mermaid diagram -- written so that someone outside this project can follow it, citing current implementations and naming building blocks.

**The three temporal horizons are now the stated foundation** (owner's framing): **calibration** under 1 s, **short** from 1 to a few seconds, **long** tens of seconds. §3 gives each horizon what it supplies, the measured evidence, the memory that serves it, and why the addressing changes from beam to object between calibration and short.

**What else changed.** A background section cites the datasets and the current detectors with their published numbers and their implementations in this repository; §5.2 lists every building block with its source and the reason it was chosen; configuration settings carry names, meanings, defaults, reasoning and whether they need retraining; §12 lists 20 decisions with the alternatives kept for experiments; internal item numbers and conversation references are replaced with plain explanations.

**One correction.** The state carried between frames is **~1.8 MB, not 0.5 MB**: the coarse temporal convolution reads taps up to 32 frames back, so its cache must hold all 32 frames of bottleneck features, not only the 4 tapped ones. Multiply-add counts are unaffected; the earlier cost entry below is corrected in place.

## Value-ranked slot eviction: 256 slots are enough and the rule matches the oracle (2026-09-14)

*keywords: A43, slot_budget.py --memory, PriorityMemory, slot value, eviction, margin, spawn floor, oracle, coverage, missed people, K = 256*

**Owner's rule**: every slot carries one value -- accumulated certainty, not the latest score, so a hidden person is not evicted by a flicker -- any proposal above a low floor may spawn, and when memory is full a candidate replaces the weakest slot only if clearly more certain. Built as `PriorityMemory` in `utils/slot_budget.py` (15 tests first, `tests/test_slot_memory.py`) and replayed on every frame of `official` test with candidates down to 0.01, against unlimited memory and an oracle that never evicts a slot holding an annotated person while another is evictable.

| detector | memory | slots mean / p99 / max | people's slots evicted | missed people with a slot within 0.5 m |
| --- | --- | --- | --- | --- |
| LFE-Peaks | K = 256 | 21.6 / 73 / 91 | 0 | 71.6% (14.8 pp of all people) |
| LFE-PPN | K = 256, margin 0 | 79.1 / 179 / 256 | 2 in 50,088 frames | 90.6% (11.1 pp) |
| LFE-PPN | K = 256, margin 0.1-0.2 | same | 1 | same |
| LFE-PPN | oracle / unlimited | same / max 290 | 0 | same |

**K = 256 is enough and the rule is as good as the oracle.** LFE-Peaks never fills memory; LFE-PPN reaches the limit only in its densest moments, evicts ~50 times in 50,088 frames, and loses no measurable coverage against unlimited memory. The margin makes no measurable difference at this capacity and stays at 0.1 as a churn guard. `docs/PROPOSAL.md` §6.3 now carries the rule, replacing the earlier "sub-threshold proposals may not spawn" refinement, whose ~17,000-slot argument assumed slots are freed only by a timer.

**Read the coverage column carefully.** A slot within 0.5 m of a missed person counts whether it came from memory or from that frame's own sub-threshold candidate, and whether or not it really is that person. It bounds what the head could use, not recall.

## Operations and memory per frame: cutout models cost 60-290x the proposed model, and not because of attention (2026-09-14)

*keywords: multiply-adds, FLOPs, MACs, FlopCounterMode, parameters, activation memory, training memory, Li2Former OOM, DR-SPAAM streaming, LFE ONNX, A43 cost*

**Owner's question**: compare not only complexity classes but actual operation counts and memory for DR-SPAAM, LFE, Li2Former and the proposed model at default sizes -- a quadratic model on small inputs might still beat a linear one. Counted, not estimated; full table in `docs/PROPOSAL.md` §10.1.

| per frame, 720 beams | multiply-adds | params | training activations |
| --- | --- | --- | --- |
| LFE-Peaks / LFE-PPN | 24 M / 26 M | 53 K / 171 K | — |
| A43 sketch | 303 M | 1.49 M | 22 MB |
| DR-SPAAM, official streaming | 18.4 G | 1.98 M | — |
| DR-SPAAM, our windowed port, `T=5` | 48.0 G | 1.98 M | 1.4 GB |
| Li2Former, `T=5` | 88.2 G | 4.70 M | 2.0 GB |

**The owner's intuition was right in principle and the answer is still one-sided.** At these sizes attention is cheap -- Li2Former attends over five steps per beam, DR-SPAAM's 720 x 720 similarity is ~7 G of 48 G -- and what dominates is a wide CNN applied per cutout: tens to hundreds of thousands of positions per scan, against 720 for a full-scan model. Li2Former's 2.0 GB of saved activations per frame is consistent with its recorded OOM at batch size 2.

**Three measurement traps, recorded because each would have produced a wrong table.**

1. PyTorch's FLOP counter reports **2x multiply-adds** (verified on single Linear and Conv1d layers); mixing it with a hand count of the ONNX graph would have favoured LFE by 2x.
2. Under `no_grad`, `nn.MultiheadAttention` and `TransformerEncoderLayer` take a fused fast path the counter **does not see**; Li2Former's and A43's attention would have dropped out of the count.
3. ONNX shape inference with every symbolic dimension set to 1 collapsed LFE's 720-beam axis and reported 0.1 M. Shapes are now taken from a real inference with every intermediate exposed.

**Also found: our DR-SPAAM port costs 2.6x the official streaming model per frame** (48.0 G against 18.4 G), because it re-encodes the whole window every call -- the cost side of `TODO.md` A47.

## The architecture is settled question by question and written up for the owner's review (2026-09-14)

*keywords: A43, docs/PROPOSAL.md, decisions log, step interface, beam-local offsets, fine and coarse temporal conv, ConvNeXt U-Net, nnU-Net Revisited, feature cache re-alignment, K = 256, spawn rule, feedback, CenterTrack, simulated priors, configuration, inference cost*

**`docs/PROPOSAL.md`, fourth version**, replaces the options-and-recommendations draft with the design the owner settled over three rounds of questions: 19 architectural questions plus configurability and inference cost. §16 is the decisions log; §17 lists what must be verified before building.

**What changed from the recommendations, on the owner's call:**

- **Two temporal scales, not one short window.** The owner proposed consecutive frames for calibration plus a ~1.5 s span; it is a two-layer dilated causal conv, fine per beam at full resolution and coarse jointly over space and time at the U-Net bottleneck, where a walking person stays within reach.
- **Features are cached, not raw scans** -- speed over cache size -- which forced beam-local input offsets, so cached features stay valid while the robot moves, and an explicit re-alignment step for each cache.
- **Feedback from memory into stage 2** (the owner's intent from the start), CenterTrack-style, at the bottleneck first. Staged training survives it through priors simulated from annotations.
- **K = 256**, over the 128 the measurement suggested -- the owner's overshoot rule.

**Checked for the owner before deciding.** The U-Net's successors: nnU-Net Revisited (*MICCAI* 2024) finds well-configured CNN U-Nets, including ConvNeXt variants, still state of the art, and Mamba layers in U-Mamba contributing nothing once a residual U-Net baseline is included -- so the design keeps the U-Net shape with ConvNeXt-style blocks. CenterTrack's training of a prior-track input from static images with heavy augmentation, which is what lets feedback coexist with staged training.

**One refinement proposed and awaiting review: who may spawn a slot.** The owner assumed any prediction can, given the model's short-term memory. Any *detection* can, immediately. Sub-threshold proposals only update existing slots: 64 per frame over a 10 s window would need ~17,000 slots and fill 256 within four frames, after which eviction recycles occluded people.

**`docs/RESEARCH.md` §2 constraints 1 and 2 are lifted for this model only**, as settled in the discussion (recurrence confined to stage 3's slots; Cartesian input). The three existing full-scan architectures keep both.

## Stage 3 needs ~128 slots, not ~10: slots outlive the people in them (2026-09-14)

*keywords: A43, slot_budget.py, slot count, K, object memory, retire window, occupancy, concurrency, turnover, phantom flicker, overshoot*

**Owner's question 3.9**: how many slots should the object memory hold, with the rule *overshoot rather than undershoot*. Measured rather than guessed: `utils/slot_budget.py` counts slots in use per scan when a slot is spawned from every unmatched detection and retired only after `retire` seconds without a match. Tracks by gated Hungarian against each track's last position, no motion model, so detection rows over-count -- the intended direction. 13 tests (`tests/test_slot_budget.py`).

FROG `official` test, every frame (26.1 Hz, gate 0.6 m), slots in use:

| source | retire | mean | p99 | max |
| --- | --- | --- | --- | --- |
| annotated people | in frame | 3.07 | 8 | 10 |
| | 5 s | 4.72 | 20 | 39 |
| | 10 s | 6.14 | 31 | 40 |
| LFE-Peaks detections | in frame | 4.20 | 12 | 18 |
| | 5 s | 12.01 | 45 | 71 |
| | 10 s | 16.71 | 55 | 79 |
| LFE-PPN detections | in frame | 7.72 | 23 | 37 |
| | 5 s | 23.76 | 82 | 140 |
| | 10 s | 32.96 | 105 | 164 |

DROW test (10 Hz, class-agnostic annotations, gate 1.25 m): 1.86 in frame, **2.05** at 10 s, max 9 -- almost flat.

**The count is driven by turnover, not by crowd size.** People in frame peak at 10, but a retire window keeps a slot for every person who recently *left* the view, and a museum crowd with a median trajectory of 4.9 s turns over constantly. Detections add phantom flicker -- most phantom trajectories last one frame, and each still holds a slot for the full window. DROW's sparse care-facility scenes barely turn over, and its curve is flat. Part of FROG's growth is also association fragmentation (no motion model, fixed gate); how much was not separated, and it inflates the count in the direction the owner asked for. The p99.9 jump for annotated people between 0 s and 1 s (9 to 28) looks like bursts of such fragmentation and is unexplained.

**Proposal: K = 128**, configurable. *(Owner chose 256 the same day, as deliberate overshoot.)* Covers LFE-PPN's p99 at 10 s (105) and LFE-Peaks' maximum (79); at 128 synchronisation is ~16k slot pairs per frame, still trivial. Two design consequences, recorded for `PROPOSAL.md`:

1. **An overflow rule is mandatory at any K**: when full, evict the slot longest unmatched (or lowest existence score) rather than refusing to spawn.
2. **The spawn rule moves K more than K moves anything**: most slots are held by one-frame phantom flickers. A spawn threshold above the proposal threshold, or confirmation over a few frames, is the lever -- a knob, not a constant.

## Prior-work survey for the three-stage model: both requested interfaces are published, and our DR-SPAAM port is not streaming (2026-09-14)

*keywords: A43, A47, prior work, continual-inference, streaming convolution, rolling cache, TSM, Mamba, step interface, recurrent state, RVT, TimePillars, Deep Tracking, MeMOTR, StreamPETR, SambaMOTR, LRU, minGRU, DR-SPAAM streaming*

**Owner's request**: before deciding the model, look for published architectures and implementations to reuse -- specifically temporal convolutions over a rolling cache of recent frames for stage 2, and a recurrence of the form `step(input, state) -> (output, state)` with per-frame cost constant. Surveyed by web search; each work checked against its abstract, project page or code. Full tables in `docs/PROPOSAL.md`, "Published work to reuse".

**Stage 2.** Streaming causal convolution is a solved conversion: train on clips, run one frame per call from a per-layer buffer, and the result is exact. `continual-inference` (Hedegaard & Iosifidis; pure PyTorch, Apache-2.0) does it as a drop-in for Conv1d-3d, pooling and GRU/LSTM. The Mamba block is both requirements in one unit -- a cached causal conv followed by a recurrent SSM state. **Decision 3 is revised** from the LFE port to a full-scan backbone with a streaming causal temporal conv; the port's remaining argument is only as a published static control.

**Stage 3.** Object-keyed step-wise memory is published and recent: MeMOTR (per-track exponential memory, *ICCV* 2023), StreamPETR (object queries as hidden state, *ICCV* 2023) and **SambaMOTR** (one selective SSM per tracklet, synchronised by attention, *ICLR* 2025) -- the nearest match to stage 3. Grid-keyed step-wise detectors exist too: RVT and its SSM successor on event cameras, TimePillars on 3D LiDAR, and **Deep Tracking on the Move on 2D laser from a moving vehicle**.

**Two corrections this forced.**

1. **`PROPOSAL.md` had ruled out learned place-keyed memory on the strength of A41/A44**, which measured hand-built statistics only. Deep Tracking on the Move is a working learned grid recurrence on this sensor class and platform type. The rules-out row now says what was measured; object keying stays recommended on the association and mobility evidence.
2. **Our `DrSpaamDetector` is not the streaming model its authors ship.** The official detector keeps its attention template across calls; ours re-encodes a `T`-scan window per call. Every DR-SPAAM number here is the windowed form. Probably a small difference at `alpha = 0.5`, unmeasured -- `TODO.md` A47.

**A trap recorded for stage 2**: cached beam-indexed activations can be re-aligned for rotation by a cyclic shift but not for translation, which reaches ~0.5 m over a 1 s cache on this robot. Either cache Cartesian points and re-encode the short window, or rely on a learned local alignment, as DR-SPAAM's attention does.

## Model extensions written up as one paper section; the proposal becomes three memories inside one detector (2026-09-14)

*keywords: docs/PAPER.md, Model extensions, docs/PROPOSAL.md, A43, three memories, object memory, slots, Cartesian input, normalising memory, linear recurrence, exchange rate, tracker AP cost, LFE-PPN correction*

**`docs/PAPER.md`.** The tracker, the merge-radius control, the horizon oracle, the displacement re-measurement (A45), the persistence map (A41) and the background model (A44) now sit in one section, "Model extensions: temporal reasoning on top of a frozen detector", priced in one currency and closed by a single table:

| stage after the detector | causal | LFE-Peaks | LFE-PPN |
| --- | --- | --- | --- |
| persistence map | yes | 0.57 | — |
| background model | yes | 1.90 | 3.32 |
| trail-length filter | oracle | 2.54 | — |
| SORT, `min_hits = 3` | yes | 3.8 | 9.8 |
| displacement filter | oracle | 4.20 | — |
| merge radius 0.30 -> 0.40 m | yes | **21.9** | **~63** |

LFE-PPN's SORT and radius rates are derived from A40's own figures (0.626 / 0.064 and 0.941 / 0.015) and were not previously tabulated.

**Correction found while consolidating.** The paper's "what tracking buys" table still carried LFE-PPN's **pre-recalibration** column (`nms_radius = 0.8`), although the tracker table beside it had been re-run. Recomputed from the current rows, and the claim built on it does not survive: LFE-PPN does **not** benefit less from tracking. Up to a third of phantoms removed both detectors pay ~0.20 pp of AP per point of reduction, and beyond that LFE-PPN is cheaper (19.0 pp for 55.5% against LFE-Peaks' 25.0 pp for 59.0%). The phantom-density mechanism proposed to explain the old result is withdrawn with it. A37's note that the claim "survives but narrows" was read at `min_hits = 3` only, where the two are equal.

**`docs/PROPOSAL.md`, third version.** Built from the owner's three-stage outline: **Cartesian input**, a **normalising memory** (<= 1 s, beam-indexed) and an **object memory** (seconds to tens of seconds). One refinement: short-term and long-term share one object-keyed recurrent state and differ by decay rate, because the horizon oracle found one horizon rather than two time constants and A44 found the long one cannot be keyed by place. Role C survives as the slow channels, and its ablation is to zero them. Two constraints derived rather than assumed: translation must be compensated even at one second (1.88 cm per frame is ~0.5 m per second), and step 1 can test the object memory on frozen LFE proposals with no port.

**Four owner decisions are open**, two of them reversals of `docs/RESEARCH.md` §2 (non-recursive only; no Cartesian input). Recorded in `TODO.md` A43.

## Role C closes as a measured negative: a moving robot never looks anywhere long enough (2026-09-11)

*keywords: A44, A46, background map, BackgroundMap, scan returns, hit rate, role C closed, observation span, mobile robot, FP per FN, exchange rate*

**Third and last mechanism for role C, and it fails like the other two.** `BackgroundMap` accumulates **raw scan returns** rather than detections -- a chair reflects a beam whether or not anything detects it -- and keys the decision on **hit rate** rather than hit count, which is what separates the three cases a count cannot:

| | rate | span |
| --- | --- | --- |
| a chair | ~1 | long |
| someone standing 5 s | ~1 | **short** |
| a busy corridor cell | **low** | long |

It uses no trails and no association, so A45's association-limited ceiling does not bind it. It still fails. LFE-Peaks, `official` test, all 50,088 frames:

| min span | min rate | FP removed | TP lost | FP per FN |
| --- | --- | --- | --- | --- |
| 1 s | 0.95 | 45.4% | 45.5% | 0.72 |
| 10 s | 0.95 | 5.3% | 2.0% | **1.90** |
| 30 s | 0.30 | 11.3% | 6.4% | 1.29 |
| 100 s | any | **0.0%** | 0.0% | — |

**LFE-PPN confirms**: best **3.32** (removing 3.5% of phantoms for 2.0% of true positives), and **2.53-2.85** wherever the removal is substantial. Marginally better than LFE-Peaks, still under SORT's 3.8 and far under the merge radius.

**Every role-C mechanism tried, against the levers already priced:**

| mechanism | best exchange rate |
| --- | --- |
| A41 detection-frequency map | 0.57 |
| A44 background from scan returns | 1.90 |
| A45 trail-length oracle (*acausal*) | 2.54 |
| A45 displacement oracle (*acausal*) | 4.20 |
| *SORT* | *3.8* |
| *merge radius* | *21.9* |

Only the acausal displacement oracle beats SORT, it is an oracle and therefore unreachable, and it is still **5x below simply widening a merge radius**.

**The unifying reason, which the A44 table shows directly.** At a minimum observation span of 30 s only **11.3%** of phantoms qualify at all, and at 100 s **none do**. **On a mobile robot, how long something has been at a location is dominated by how long the robot looked, not by whether the object is furniture.** The museum robot drives past a chair; it does not sit and watch it. So "persistence at a world location" -- the feature all three mechanisms key on, and the one the 89.7% headline measured -- is mostly a statement about the robot's transit.

That also explains the 89.7% itself: the precision oracle removed everything with a trail shorter than 30 s, which on person-free frames is safe and is most detections. Short-lived is exactly what a *person* is too, so the same filter on populated frames destroys them.

**Role C is closed as a measured negative**, and it is a publishable one: the horizon argument in `docs/PAPER.md` survives -- 786 frames is still the span over which the evidence exists -- but *location-keyed persistence is not the way to reach it on a moving platform*. Any mechanism that does must be **object-keyed**, which is role B.

**What this does not close.** The paper's claim is about the horizon, not about this family of mechanisms. A displacement feature carried per *object* remains the one form with a measured rate above SORT (4.20), and that is role B's territory rather than role C's.

**Next, per the owner's directive**: model engineering, carrying all three memories in priority order **normalising > short-term > long-term** (`docs/PROPOSAL.md`). Role C is retained as *structure* rather than deleted -- a ceiling on three mechanisms is not proof the horizon carries no signal.

## Role C has far less headroom than 89.7% implied, and role B gates it (2026-09-11)

*keywords: A45, discrimination oracle, role C priority, trail length vs displacement, populated frames, exchange rate, FP per FN, association-limited, horizon_sweep --discrimination*

**Owner's question**: measure the role-C bound on *all* frame types, not only person-free ones, to price the role properly. Done -- `horizon_sweep.py --discrimination` -- and it demotes role C.

The precision oracle ran on **person-free** frames, where every detection is a phantom by construction, so a trail filter could not destroy a true positive. It bounds *removal*. On populated frames the same filter must **discriminate**, and both sides become visible. LFE-Peaks, `official` test, all 50,088 frames, 153,655 people, trails built over every detection:

| filter | best exchange rate | at |
| --- | --- | --- |
| **trail length**, `max_gap = 0` | **2.43** FP per FN | 0.10 s |
| **trail length**, `max_gap = 26` (1 s) | **2.54** | 0.25 s |
| **displacement** (world bounding box) | **4.20** | 0.30 m |
| *SORT, for reference* | *3.8* | *`min_hits = 3`* |
| *merge radius, for reference* | *21.9* | *0.30 -> 0.40 m* |

**Three findings, in ascending order of how much they change the plan.**

**1. Trail length was the wrong feature, and `PAPER.md` already said so.** Its separating claim is that *a person eventually displaces and a chair never does* -- displacement, not persistence. The oracle measured persistence because on person-free frames the proxy is never tested against the class it must exclude. Keyed on displacement instead, the same acausal filter improves from 2.54 to **4.20**. At `D = 0.80 m` it removes **36.1%** of phantoms for **6.9%** of true positives.

**2. Even so, role C's ceiling is low.** The best acausal, perfect-hindsight filter on the right feature reaches 4.20 FP per FN -- marginally past SORT's 3.8 and **5x worse than simply widening the merge radius**. The 89.7% headline was a removal figure with no cost column, and the cost column is most of the story. **A41's failure was not an implementation problem**; the concept has little room.

**3. And the bound is association-limited, which makes role B a prerequisite rather than a sibling.** Trails are built by greedy nearest-neighbour association. In FROG's crowds (~3.9 people per frame) that merges a person's detections into a nearby phantom's trail, and the merged trail carries the phantom's low displacement -- so the person's true positives are destroyed by a filter aimed at the chair. Better association would raise this ceiling directly. **Role C cannot be filtered better than its trails can be formed, and forming them is role B.**

**Correction to a claim made earlier the same day.** After A41 this log stated that "a world-cell index cannot supply that discrimination". Too strong, and the reason matters: `build_tracks` is greedy nearest-neighbour association, so the oracle's trails are **object-indexed**, never cell-indexed. A41 built a per-cell *traffic counter*, which sums contributions from distinct objects and is a different mechanism from per-object persistence. What is established is that **detection frequency per location does not discriminate**; a per-cell rule keyed on *continuous presence* has not been tested, and neither has anything built from raw scan returns (A44).

**What this does to the plan.** Role C drops below role B in priority. A44 -- the background map built from raw scan returns rather than from detections -- survives this result specifically because it does **not** depend on detection trails or on association at all, so the ceiling measured here does not apply to it. It is now the only cheap role-C variant with untested headroom.

## The precision oracle bounds removal, not discrimination -- and a world-cell map cannot tell people from furniture (2026-09-11)

*keywords: A41, A44, persistence map, role C, occupancy grid, oracle bound, discrimination, world-indexed, negative result, persistence_map.py*

**The cheapest version of role C was built and priced, and it fails decisively.** `utils/persistence_map.py`: an ego-motion-compensated world-frame map accumulating, per cell, an exponentially-forgetting count of how often a detection has been reported there; detections standing on cells with more than `tau` frames of evidence are suppressed. Two knobs, no training, O(1) per detection per frame. Swept over cell size, horizon and threshold on LFE-Peaks, both splits, every frame.

**On person-free frames it looks like a success.** At 0.50 m cells and a 30 s horizon it removes **77.5%** of phantoms -- 1.391 to 0.312 per frame -- against an oracle bound of 89.7%.

**On populated frames it removes people at the same rate.**

| tau | phantoms removed (person-free) | detections removed (populated) | recall |
| --- | --- | --- | --- |
| — | 0% | 0% | 79.3% |
| 200 | 7.1% | 5.2% | 74.8% |
| 100 | 23.0% | 17.0% | 66.2% |
| 30 | 55.5% | 45.2% | 46.8% |
| 10 | 77.5% | 73.8% | **23.2%** |

The two columns track each other at every setting. **The map is not discriminating; it is suppressing whatever it has seen before.** Its exchange rate is **0.57 false positives removed per false negative added**, against SORT's 3.8 and the merge radius's 21.9 -- **6.7x worse than the tracker and 38x worse than a post-processing constant.**

**Why, and this is the finding worth keeping.** The role-C oracle -- 89.7% of LFE-Peaks' phantoms removable at a 30 s horizon -- was measured **only on person-free frames**. It asked what fraction of phantoms sit on long trails. It never had to separate a phantom from a person, because there were no people. **It is an upper bound on *removal*, not on *discrimination*,** and the difference is invisible until a mechanism is actually built.

A world-cell index cannot supply that discrimination on this data, because **people revisit locations**: museum visitors queue at the same exhibits and pass through the same doorways, so "a detection has been reported here before" conflates high-traffic locations with static furniture. The caveat that the two oracle halves are measured on different frames was already recorded in `dataset-properties.md`; its consequence for any location-indexed rule was not, and now is.

**What survives.** "Phantoms are long-lived and people are not" remains true *as trajectories* -- 67.7% of phantom detections sit on trails of 1 s or more while only 3.6% of person trajectories reach 30 s. What fails is exploiting it through location alone.

**What comes next (A44), and it is a different mechanism rather than a tuned version of this one.** Build the occupancy map from **raw scan returns** instead of from detections, and use it as a background model: a chair returns a persistent *surface* whether or not anything detects it, and the question becomes "is this leg-like return already explained by the background?" rather than "have detections happened here?". That is classical background subtraction, still training-free, and it addresses the identified cause instead of re-tuning around it. The known failure mode is unchanged and must be measured: a person standing still long enough joins the background.

**`PROPOSAL.md`'s stated principal risk was exactly this**, and the hard-threshold form was built *because* it is the cheapest possible version -- the point of pricing a lever is that it is allowed to come out badly.

## Cartesian input ties DROW's depth tunnel and beats LFE's normalisation by 6 pp (2026-09-11)

*keywords: A42, local-Cartesian, _local_cartesian, repr_probe, coordinate conversion, depth tunnel, preprocessing, unmatched control, BEV, rasterising*

**Owner's proposal**: drop polar preprocessing and hand the model Cartesian coordinates from the start. Tested with no training, as a fifth and sixth parameterisation in `utils/repr_probe.py` -- metric offsets from each window's own centre point, rotated into that beam's frame, radial component clipped to +-1 m so it differs from DROW's tunnel in coordinate system and nothing else. 24 points x 2 channels = 48 dims, matched to the cutout.

| angular | depth | 1-NN err |
| --- | --- | --- |
| adaptive | centred (DROW) | **11.08%** |
| adaptive | cartesian | 12.38% |
| fixed | **cartesian** | **13.39%** |
| fixed | centred | 13.56% |
| adaptive | lfe | 18.86% |
| fixed | lfe | 19.60% |

**Cartesian ties the depth tunnel at matched window width** (13.39% against 13.56%), so the simplification is free -- it deletes the range-dependent half-width, the bilinear resample and the out-of-bounds handling while keeping both displacement components instead of one. **Against LFE's global `1 - r/10` it wins by 5.5-6.2 pp**, which is the comparison that matters for a full-scan model of ours. **It does not replace the range-adaptive angular window**: `adaptive/centred` stays best, and Cartesian is 1.30 pp behind it. Going adaptive buys the tunnel 2.48 pp and Cartesian only 1.01 pp -- consistent with metric coordinates already carrying part of the scale information.

**Verdict: take the coordinates, keep the range conditioning.** Cartesian input *plus* a range-conditioned receptive field (A39), not instead of it.

**The first run of this said 14.88% and would have rejected the idea.** It was an unmatched control: the `fixed` cutout row uses a 41-beam half-width resampled to 48 points, while the Cartesian row took +-12 *raw* neighbouring beams -- a third of the context, with extent and dimensionality tied to one knob. Separating them moved the answer from "clearly worse" to "ties". This is the second confound of this class in two days (A38's was an under-explored grid), and both were caught by noticing an asymmetry rather than by anything automatic. `_local_cartesian` now takes `hw_fixed` so extent and dimensionality are separate parameters, and `repr_probe.py`'s cache key carries a version tag because adding a representation silently invalidates every cached sample.

**Also recorded: coordinate conversion is not rasterisation.** Only the latter costs anything -- 720 points into ~40,000 cells at ~2% occupancy -- and the sparsity objection raised against "BEV" does not apply to converting coordinates at all. The rasterised form is a separate question, deferred to A43.

## The merge radius is a better false-positive lever than the tracker (2026-09-11)

*keywords: A40, operating point, --score-by, _frame_metrics, close-pair, duplicates, exchange rate, merge radius, FP per FN, monotone trade, no operating point*

**A38 swept the merge radius against AP. A40 sweeps it against what Tier 0 measured** -- recall, duplicates, phantoms and close-pair misses at a fixed threshold (`nms_sweep.py --score-by operating-point`, 28 cells x 2 detectors x 2 association distances, `official` test every 5th frame, 30,737 people). Three results, in ascending order of importance.

**1. A38 stays rejected, now on the metric it was supposed to win.** At matched recall the constant radius has fewer false positives everywhere it can be compared:

| | recall | FP/frame |
| --- | --- | --- |
| LFE-Peaks, `a=0.30 b=0.030` | 76.9% | 1.445 |
| LFE-Peaks, constant `r=0.45` | 76.9% | **1.380** |
| LFE-PPN, `a=0.30 b=0.030` | 85.9% | 3.079 |
| LFE-PPN, constant `r=0.45` | 85.7% | **3.001** |

The one comparison it does not lose is LFE-PPN at `d = 0.3 m`, where the two are within noise. It never wins.

**2. The "~5 pp of recall in close-pair over-merging" from Tier 0 was overstated, and this is the measurement that caught it.** `range_probe.py` computes its nearest-detection distance *before* thresholding, so "a detection within 0.5 m" counted sub-threshold detections that would never be reported. At the operating point, close-pair misses are **0.033 per frame -- 331 of 30,737 people, 1.1%** -- and they move only between 0.028 and 0.062 per frame across radii from 0.20 m to 0.50 m. **Recall moves 1.8 pp across that entire range** (79.5% to 77.7%) while duplicates fall **18-fold** (0.636 to 0.035). The radius is a precision lever, not the cheap recall it was billed as. `static-detector-diagnosis.md` and `PROPOSAL.md` corrected.

**3. The radius is a far better exchange rate than SORT, at the same job.** Both are cheap post-processing that trades false positives for false negatives along a monotone curve with no optimum -- the *same shape* the tracker sweep found. What differs is the price. At **matched false-positive reduction** for LFE-Peaks (~0.33 per frame):

| | FP/frame removed | FN/frame added | FP per FN |
| --- | --- | --- | --- |
| SORT, `min_hits = 3` | 0.331 | 0.087 | 3.8 |
| merge radius, 0.30 -> 0.40 m | 0.328 | **0.015** | **21.9** |

**5.8x cheaper in false negatives for the same false-positive reduction**, and LFE-PPN agrees (the radius removes 0.941 FP/frame for 0.015 FN/frame where SORT removes 0.626 for 0.064). This does not weaken `docs/PAPER.md`'s argument -- neither lever reaches zero, and both saturate -- but it does mean the tracker is not the strongest cheap baseline available, and the paper should not present it as one.

**Why AP could not see any of this.** AP picked `r = 0.45` for both detectors (A38). On the operating-point view there is no optimum at all: recall falls monotonically as the radius grows and false positives fall with it. AP's preference is just where AP's own weighting of that trade happens to land, not a property a deployment would recognise -- which is the same objection `docs/PAPER.md` raises about average precision and absolute false positives, arriving on a second axis.

**Method note.** `_frame_metrics` assigns one detection per person by Hungarian, and guarantees `tp + duplicates + phantoms == len(detections)` so a radius change cannot move detections into an uncounted category and look like an improvement. 14 tests pin the definitions (`tests/test_operating_point_metrics.py`). The sweep merges once per `(a, b)` and scores every association distance from the same detections.

## A range-aware merge radius loses to a constant one, and AP is why (2026-09-11)

*keywords: A38, A40, merge_radius_slope, nms_radius_slope, _merge_nearby, range-aware radius, nms_sweep.py, _ReMerger, raw-score-floor, tuning on test, close-pair*

**Built, swept, rejected.** `_merge_nearby` now takes a `slope`, making the radius `r(d) = radius + slope * d` at the pair's mean range, floored at zero. `slope = 0.0` is bit-identical to the old single scalar and is the default, guarded by `tests/test_merge_radius.py` (17 tests, including the zero-slope regression at all four radii this project has published a number under).

Motivation was Tier 0 (`memory/static-detector-diagnosis.md`): at 0.30 m the one scalar fails in both directions at once -- ~5 pp of LFE-Peaks' recall spent merging adjacent people, while LFE-PPN emits 1.609 duplicates per frame.

**Result, LFE-Peaks, `official` test every 5th frame, both association distances:**

| | AP@0.5 | AP@0.3 |
| --- | --- | --- |
| `r = 0.30` (default) | 65.1% | 63.4% |
| best range-aware, `a=0.30, b=0.030` | 67.0% | 65.0% |
| `r = 0.40` constant | 67.7% | 65.7% |
| **`r = 0.45` constant** | **68.1%** | **66.0%** |
| `r = 0.60` constant | 66.1% | 63.8% |

**The range-aware optimum averages ~0.45 m over FROG's range distribution -- it is approximating a constant 0.45 m, 1.1 pp worse, with one more parameter.**

**LFE-PPN, controlled identically, agrees**: `r = 0.30` 69.6% / 64.3%, best range-aware cell 71.9% / 65.2%, constant `r = 0.45` **72.3%** / 65.3% and constant `r = 0.40` 72.1% / **65.6%**. Both detectors put the AP optimum at 0.40-0.45 m while feeding very different inputs into the same merge -- 201.5 raw centroids per frame against 5.6 -- so the disagreement with the paper-reproduction rule below is systematic, not sampling noise.

**The first grid was confounded and the control is what caught it.** The intercept grid stopped at 0.30, so "the slope helps" could not be separated from "a bigger radius helps". Extending it to 0.60 at `slope = 0` answered it in one run. Recorded because the confound was in a grid *designed this session* to test a hypothesis formed this session -- exactly what `dos-and-donts.md`'s "write the prediction before running the test" is for, and it was not written down first.

**The sweep is validated, not assumed.** `nms_sweep.py` now caches raw centroids once per detector and replays them through `_merge_nearby` at each `(a, b)`, because the ONNX forward does not depend on the radius -- 640k detector calls become 20k. Replaying at each detector's own radius reproduces that detector's own AP to **0.00 pp** at `--raw-score-floor 0.01`; at 0.05 it does not, costing LFE-PPN 0.49 pp because AP integrates a PR curve whose tail those proposals are. The constant-radius rows also reproduce `LFEPeaksDetector`'s independently measured docstring table (0.30 / 0.35 / 0.40) exactly.

**The finding worth keeping is not about the radius.** AP rises monotonically from 0.30 m to 0.45 m -- past the value that reproduces the published number, and 0.30 was pinned precisely because 0.40 overshoots the paper by +2.1 pp. So **optimising this parameter against AP on the test recording is tuning on test, and it pushes monotonically away from the paper's own configuration.**

And **AP and Tier 0 disagree about the direction.** AP wants more merging; Tier 0 measured over-merging costing ~5 pp of recall at a fixed operating point. Both are right about their own quantity: a duplicate is a full false positive, while a close-pair miss costs only one missed GT, because the surviving merged detection still matches *one* of the two people. AP therefore under-charges exactly the error Tier 0 found. That is the same structural blindness `docs/PAPER.md` argues about false positives, on a different axis -- and it means A38's verdict is a verdict about AP, not about the radius. Re-scoring against Tier 0's own quantities is **A40**.

## Every LFE-PPN false-positive number was stale, and `phantom_analysis.py` had a latent counting bug (2026-09-11)

*keywords: nms_radius, 0.8 to 0.30, LFE-PPN FP/frame, phantom_analysis breakdown, min(n_near, n_gt), Hungarian, stale table, 1.94 vs 3.17, 4.15, longest unbroken run, transferred*

**Found by chasing a 2x discrepancy** between `performance-log.md` (LFE-PPN 1.94 phantoms per person-free frame) and `dataset-properties.md` (4.15) for the same detector on the same split. Neither was a typo and both tools were right about what they measured. Three separate things were stacked on top of each other.

**1. `nms_radius` was recalibrated 0.8 m -> 0.30 m and the false-positive tables were never re-run.** The calibration itself was correct — 0.30 m reproduces LFE-PPN's published AP at *both* association distances the benchmark reports, where 0.8 m is ~3.5pp low on both — but a smaller suppression radius keeps more detections, and every FP figure in `PAPER.md`, `performance-log.md`, `dataset-properties.md` and this file predated it. Measured directly, on 500 person-free frames:

| | nms/merge 0.80 | 0.40 | 0.30 |
| --- | --- | --- | --- |
| LFE-PPN | 3.314 /frame | — | **5.824** /frame |
| LFE-Peaks | — | 2.358 /frame | **2.420** /frame |

**Only LFE-PPN moved**, because only its radius moved far: 0.8 -> 0.30 is 1.76x, while LFE-Peaks' 0.40 -> 0.30 is 1.03x. That is the whole reason one row of every table went stale and the other did not.

**2. The two documents were also measuring different frame sets, and neither said so.** `evaluate.py` walks all 72,533 person-free frames; the oracle path walks the longest *unbroken* run per recording, because a trajectory needs consecutive frames. The unbroken runs are denser in phantoms — a long stretch with nobody in it is a stretch where the robot is parked somewhere cluttered:

| | all 72,533 person-free frames | longest unbroken run (16,475) | ratio |
| --- | --- | --- | --- |
| LFE-Peaks | **1.391** /frame, 51.0% of frames | **1.533** /frame, 60.6% | 1.10x |
| LFE-PPN | **3.174** /frame, 64.0% of frames | **4.150** /frame, 74.4% | 1.31x |

So `1.94 -> 3.174` is cause 1 and `3.174 -> 4.150` is cause 2. Both were needed to explain the gap, and LFE-Peaks' two figures (1.391 and 1.533) were correct all along — the documents simply never said which subset each described. They do now.

**3. `phantom_analysis.py` counted true positives as `min(detections near a person, people present)`.** That shortcut cannot distinguish *one detection each on five people* from *five detections on one person*, so it scores a pile of duplicates as recall. It was harmless while LFE-PPN emitted 0.004 duplicates per frame at `nms_radius = 0.8`; at 0.30 m it emits 1.609, and the shortcut put LFE-PPN's miss rate at **5.0%** where a one-to-one assignment gives **19.3%**. Replaced with `linear_sum_assignment` gated at the association radius.

**The fix is self-checking:** with the assignment in place, `phantom_analysis.py` now reproduces `evaluate.py`'s independently-computed populated figures for LFE-Peaks exactly — 1.759 FP/frame and 0.634 FN/frame against the results table's 1.76 and 0.63. The two tools had been disagreeing for as long as the bug existed, and nothing flagged it.

**Corrected tables.** Populated is `frog_16-41` every 5th frame (10,034 frames, 30,737 people), threshold 0.3, association 0.5 m:

| | AP | true positives | duplicates | **genuine phantoms** | total FP | FN/frame |
| --- | --- | --- | --- | --- | --- | --- |
| LFE-Peaks | 65.1% | 2.429 | 0.372 | **1.387** | 1.759 | 0.634 |
| LFE-PPN | **69.6%** | 2.689 | **1.609** | **3.416** | **5.025** | **0.375** |

| | phantoms/frame, person-free | recur next frame |
| --- | --- | --- |
| LFE-Peaks | 1.391 | 88.3% |
| LFE-PPN | **3.174** | **77.6%** |
| *real people* | — | *99.8%* |

**LFE-PPN's AP also moved, 68.6% -> 69.6%**, for the same reason: at 0.30 m it resolves adjacent people that 0.8 m merged away. Against its published 69.2% that is now a **+0.4pp overshoot** rather than a -0.6pp undershoot, which is small but on the wrong side of the standing overshoot rule and belongs with A32d.

**What survives unchanged.** The paper's first contribution — that the genuine phantom rate barely depends on whether anyone is present — is a comparison of two measurements made the same way, so the radius change moves both sides together. It now reads **0.3%** (Peaks) and **7.6%** (PPN) agreement rather than 0.3% and 1.3%. And the `genuine phantoms` column was never touched by cause 3 at all: a detection near no person needs no assignment to identify. A correction of magnitude, not of method.

**The standing lesson, added to `dos-and-donts.md`:** changing a calibrated constant invalidates every table downstream of it, and nothing in this repo notices. `nms_radius` moved in the working tree; four documents kept quoting numbers from before the move for as long as nobody re-ran them.

## A tracker cannot remove the phantoms, and the paper draft that says so (2026-09-10)

*keywords: docs/PAPER.md, SortTracker, SimpleTracker, hit_streak, min_hits, tracker_sweep.py, phantom_analysis.py, merge_radius, phantom vs duplicate, persistence, FP:FN*

**A paper draft now exists**, `docs/PAPER.md`: theory, dataset design, baseline table, tracking result, and a command per claim so every number can be re-derived.

**The problem is reframed, on the owner's call**, to: *reduce false-positive detections by introducing temporal information, without giving up single-frame accuracy.* Measurable on a quantity the benchmark does not report, and it admits a clean negative result — which is what it got.

**`merge_radius` closed the LFE-Peaks reproduction gap.** Testing the paper's "most common ground truth circle **diameter**" rule (0.8 m) is what found it, by *failing*: at 0.8 m LFE-Peaks loses ~6pp, because that rule governs LFE-PPN's NMS over *person proposals*, while LFE-Peaks merges *legs into a person* ("centroids that are close together are interpreted as legs or part of legs"). At **0.30 m** it reproduces the paper to -0.5pp at `d = 0.5` and +0.2pp at `d = 0.3`. The old 0.40 m was the whole +2.1pp overshoot. LFE-PPN's `nms_radius = 0.8` already matched its rule and stands.

**The phantom rate does not depend on whether anyone is present** — the paper's central claim, and now exact rather than approximate. Separating duplicate detections on already-matched people from genuine phantoms:

| | true positives | duplicates | genuine phantoms | phantoms, person-free | agreement |
| --- | --- | --- | --- | --- | --- |
| LFE-Peaks | 2.576 | 0.224 | **1.387** | **1.391** | 0.3% |
| LFE-PPN | 2.421 | 0.004 | **1.968** | **1.942** | 1.3% |

The raw FP columns differ by up to 27%; the *phantom* columns agree to 1.3%. The difference was duplicates, which cannot exist where there are no people. This is also the answer to "should `transferred` be cut for scope": the similarity is the finding, and `official` alone cannot state it.

**Two trackers, because the first one was not citeable.** `SimpleTracker` was described in this repo as "SORT-style". Audited against the reference implementation it deviates four ways, and only one is forced:

| deviation | necessary? | why |
| --- | --- | --- |
| Euclidean gate, not IoU | **yes** | point detections have no boxes; circle IoU is a monotone function of centre distance |
| ego-motion compensation | **yes** | SORT assumes a static camera; this robot moves 1.88 cm and 0.4 deg per frame |
| alpha-beta filter, not Kalman | no | replaced with a real 4-state constant-velocity KF |
| **confirms on total, not consecutive, hits** | no | SORT resets `hit_streak` on any miss. **This was the one that mattered** -- it had been documented as a property rather than recognised as a deviation, and it is strictly weaker against exactly the failure being studied |
| reports coasted tracks | no | SORT emits only tracks matched this frame; this is why permissive `min_hits` *raised* FP |

`SortTracker` (new, `tracking.py`) is faithful on all four. `SimpleTracker` stays for the deployment path and becomes the robustness ablation.

**The result, and it is negative.** SORT over the published detectors, `min_hits` swept 1-50:

| phantom reduction | costs LFE-Peaks | costs LFE-PPN |
| --- | --- | --- |
| ~4% | 0.1 pp AP | 0.5 pp |
| ~11-13% | 1.5 pp | 2.0 pp |
| ~22-28% | 5.6 pp | 6.2 pp |
| ~51-59% | 25.0 pp | 22.4 pp |

**Both curves are monotone -- there is no operating point, only an exchange rate, and it worsens the harder it is pushed** (3.85 false positives per false negative at `min_hits = 3`, 1.18 at 50). At `min_hits = 50`, which is also 1.9 s of latency before a real person is first reported, **32.6% / 47.1% of person-free frames still carry a phantom.** The cheap reduction (13% for 1.5pp) is close to the measured transient fraction; everything past it is bought by discarding real detections.

**Two predictions recorded in advance, and wrong:**

1. *"LFE-PPN will benefit more, having the larger transient fraction (30% vs 12%)."* It benefits **less** at every setting. The likely mechanism is density -- 2.36 phantoms/frame against 1.38, so a flickering phantom is often near enough to a *different* phantom for the tracker to associate them into one track and sustain the streak. **A denser phantom field is harder to filter temporally, not easier.**
2. *"SORT's stricter `hit_streak` rule may change the conclusion."* At matched false-positive rates SORT and `SimpleTracker` trace the same curve to within ~1pp of AP. The conclusion is not an artifact of tracker design.

**New, and reproducible**: `utils/tracker_sweep.py` (both tables, disk-cached detections, `--tracker sort|simple`) and `utils/phantom_analysis.py` (persistence, phantom-vs-duplicate). `tests/test_tracker.py` is a 26-test audit pinning every clause of the tracker description the paper cites.

## The published FROG numbers we have been quoting are a different metric (2026-09-10)

*keywords: A32d, mAP, association distance, d=0.5, d=0.3, Table 4, AP column, LFE-Peaks overshoot, LFE-PPN localisation, eval_r, _merge_nearby*

**What prompted this.** The owner asked why our reproductions score above the published figures. The answer is mostly that they do not.

**FROG's Table 4 reports nine numbers per model, and its first numeric column is not AP.**
It is **mAP averaged over five association distances**, `d = [0.3 : 0.05 : 0.5]` m, MS-COCO style; `AP` at `d = 0.5 m` and at `d = 0.3 m` follow in separate column groups.
Every FROG figure this project has quoted — 15.8, 49.6, 64.9, 66.5, 73.6, 75.3 — came from the mAP column, while every number we produce is AP at `--eval-r 0.5`.
The two were never the same quantity.

| | mAP [0.3:0.05:0.5] | **AP @ 0.5 m** | AP @ 0.3 m |
| --- | --- | --- | --- |
| LFE-Peaks | 64.9 | **65.6** | 63.2 |
| LFE-PPN | 66.5 | **69.2** | 62.5 |
| DROW3 (T=1) | 73.6 | **73.9** | 73.0 |
| DR-SPAAM (T=5) | 75.3 | **75.6** | 74.7 |

The correction is not uniform — +0.3pp for DR-SPAAM, **+2.7pp for LFE-PPN** — because sensitivity to the association distance is itself a per-detector property, so no single offset repairs the old comparisons.
`memory/performance-log.md`'s FROG column now carries the `d = 0.5 m` figures, and "the number to beat" moves from 75.3% to **75.6%**.

**Rescored against the right column**, on `frog_16-41`, every 5th frame:

| | ours | paper | delta |
| --- | --- | --- | --- |
| LFE-Peaks | 67.7% | 65.6% | **+2.1** |
| LFE-PPN | 68.6% | 69.2% | **-0.6** |

**LFE-PPN no longer overshoots.** LFE-Peaks' +2.1pp is real, stays open (A32d), and is *uniform across association distance* (+2.1 at 0.5 m, +2.5 at 0.3 m) — so a post-processing difference, not a localisation one.
Prime suspect, now that the paper has been read for it: it specifies NMS at "the most common ground truth circle **diameter**", which is **0.8 m** in FROG's own annotation data (median circle radius 0.400 m, 151,611 of 153,655 annotations), and `LFEPeaksDetector` merges at the *radius*, 0.4 m.

**A second defect fell out of the same comparison, pointing the other way.**
Our LFE-PPN matches at `d = 0.5 m` and undershoots by **9.4pp at `d = 0.3 m`** (53.1% against 62.5%), where LFE-Peaks is flat (+2.1 / +2.5).
That signature is mislocalisation — detections near enough for a 0.5 m gate, too far for a 0.3 m one — and it points at the anchor-decode calibration done earlier the same day, which explicitly recorded the angular convention as weakly identified **because AP@0.5 could not distinguish it**. AP@0.3 can: the same candidates span 28.7-56.4% there against 66.6-68.5% at 0.5 m.

**An attempt to solve for the convention from residuals, and why it was discarded.**
Fitting the localisation residual of 98,680 matched detections against the raw offset channel suggested the arc offset is *metric* rather than angular: the implied angular scale falls 13.76° -> 2.94° across depth bins while `scale x depth` holds near-constant at 0.32-0.34 m.
**That result is confounded and was not acted on.** The ground-truth association radius is itself metric (0.6 m), so angular residuals are capped at ~0.6/d by construction — which manufactures exactly that pattern whatever the true convention is.
Recorded here so the next attempt does not repeat it. The convention should be settled by A32c's weight-port, reading the encoding out of the graph, not inferred from outputs.

**Standing rule added**: when calibrating anything positional, use `--eval-r 0.3` as the discriminator. A 0.5 m gate is wider than most of the errors worth finding.

## Three FROG partitions, a metric for the one AUC cannot score, and the premise confirmed (2026-09-10)

*keywords: A30, A31, A32, frog_mode, official transferred balanced, _MODES, false_positive_rate, FP/frame, FP/s, LFEPPNDetector double sigmoid, score_thresh, depth_spacing, lfe_ppn.onnx*

**A30 — one dataset, three partitions, selected by `--frog-mode`.**
`FROG_Dataset(mode=...)` now chooses between `official` (the published benchmark, and the default, so nothing moved for a caller that does not ask), `transferred` (official's train and val exactly, tested on person-free frames) and `balanced` (every split sees crowded, sparse and empty scenes).
Recordings, not files, became the unit: `frog_11-36_12-43_train_val.h5` bundles two recordings ~24.6 h apart and `balanced` sends them to different splits.
A recording a split does not own is **not loaded at all** rather than merely unscored, so no window can draw history across a split boundary.

`balanced`'s composition, measured rather than assumed:

| split | recordings | frames | empty | people / populated frame |
| --- | --- | --- | --- | --- |
| train | 12-43 + 14-57 | 133,367 | 20.6% | 3.70 |
| val | 11-36 + 10-31 | 121,329 | 19.8% | 3.52 |
| test | 16-41 + 15-53 | 110,846 | 19.1% | 3.20 |

Whole recordings, not patches: dealing blocks of one recording across splits would give every split all six environments, but adjacent blocks share venue, lighting and the same people in the same clothes — trading the property this session spent its first half buying back for a diversity gain nothing could verify.
The pairing was chosen from the six possible matchings to minimise the spread of empty frames (20.6/19.8/19.1% against up to 17.0-22.9%); the residual prior shift, 3.70 people per populated frame in train against 3.20 in test, is **smaller than `official`'s own** (3.93 -> 3.07).
`balanced`'s test is `16-41 + 15-53` for two load-bearing reasons: neither recording is in LFE's or DROW's authors' training set, so published baselines are honestly held out there; and it contains `frog_16-41` whole, so the official-test number stays extractable from the same pass.
**`balanced`'s val does contain 11-36, which those authors did train on — never report a published baseline's number on it.**

**A31 — wp-AUC is undefined on a person-free split, not merely low.**
No true positives means precision 0 at every threshold and recall with no denominator.
`false_positive_rate()` (in `drow_utils.py`, beside `_prec_rec_2d`) reports FP/frame, FP/second and the share of frames carrying at least one, at a sweep of operating points rather than at a single number that would be a choice presented as a measurement.
`evaluate.py` switches to it **automatically when the split has no ground truth**, rather than behind a flag someone must remember.

*Headline decided in advance, as A31 required.* The reportable number is **`balanced` test wp-AUC** — whole raw recordings, valid AP, and it charges for empty-scene false positives inside the same number, which is what a deployment feels. `official` test wp-AUC stays the comparability number; `transferred` FP/frame is the sharp diagnostic.

**Two bugs in `LFEPPNDetector`, both found while trying to use it, both invalidating everything downstream of it.**

*The decoder applied a second sigmoid to an already-sigmoided channel.*
Read off the graph, not inferred: `lfe_ppn.onnx` ends `Slice -> Sigmoid -> Concat`, so channel 0 of `grid` is already a probability.
Squashing it again into [0.5, 0.731] broke three things at once — `score_thresh` went inert (0.01 and 0.3 returned bit-identical output), all 3,600 anchors entered the greedy NMS every frame instead of the ~61 that genuinely clear 0.3, and inference cost **224 ms/frame** while emitting **182 "detections" per frame** on scenes holding ~3 people.
Fixed: **1.34 ms/frame, 3.14 detections/frame.**

*The depth-offset channel is normalised by the anchor spacing, not raw metres.*
The paper does not state the convention, so all 16 plausible ones were scored against the published 66.5% AP on 2,525 official-test frames.
The depth axis separates cleanly and under **every** angular convention tried — raw metres 50.5%, anchor-spacing 65.8-68.5% — so this is a choice between two named conventions rather than a tuned constant.
The angular axis is weakly identified by contrast (67.4% with *no* arc offset against 68.1% with one), because a sector is 1.5° wide and the whole correction is under 10 cm at conversational range; `l_offset * sector_angular_width` was taken for symmetry with the depth channel, and half that value is indistinguishable at 68.5%.
This also settles the `-1` question left open in `docs/RESEARCH.md`: endpoint-inclusive anchors score 68.1% against the step form's 67.5%.

**This dissolves A5 rather than completing it.** LFE-PPN's slowness was never its Python decode loop; it was 3,600 spurious proposals per frame entering an O(n²) merge. No vectorization needed.

**Exact parameter counts, replacing the ONNX file-size estimates** (`onnx` is now installed; note it pulls numpy 2.x and the pin had to be restored): LFE-Peaks **53,258**, LFE-PPN **170,903** — against the ~65K/~180K previously carried.

**A32 — the premise holds, and the benchmark ranking inverts.**

On `official` test (10,018 frames, every 5th): LFE-Peaks **67.7%** against its published 64.9%, LFE-PPN **68.6%** against 66.5%. Both reproduce, both ~+2-3pp high — see the standing overshoot rule in `performance-log.md`; the residual is not yet reconciled and is tracked as A32d.

On `transferred` test — all 72,533 person-free frames, 46.1 minutes of real time at 26.2 Hz:

| detector | FP/frame @0.3 | @0.9 | FP/s @0.3 | frames with ≥1 FP @0.3 | @0.9 |
| --- | --- | --- | --- | --- | --- |
| LFE-Peaks | 1.28 | 0.66 | 33.6 | 51.0% | 32.7% |
| LFE-PPN | 1.94 | 1.05 | 50.9 | 64.0% | 43.7% |

**Both published static detectors hallucinate a person in half to two-thirds of frames containing nobody at all**, at their authors' own operating point — and still in a third to a half of them at 0.9 confidence, at 17-28 phantoms per second.

**The result that matters most is the inversion.** LFE-PPN is the *better* detector on the benchmark (68.6% vs 67.7%) and the *worse* one on false positives (1.94 vs 1.28 FP/frame on `transferred`).
The official benchmark cannot see that axis, so a ranking derived from it can be backwards for deployment.

### Correction, same day: "static detectors fail *on empty scenes*" is the wrong mechanism

The paragraphs above were written before the `official` false-positive rate was measured, and framed the result as empty scenes being *specially* hard.
**They are not.** Measured at the same 0.3 threshold, on `frog_16-41` every 5th frame (10,034 frames, 30,737 people):

| | FP/frame official | FP/frame transferred | miss rate official |
| --- | --- | --- | --- |
| LFE-Peaks | **1.43** | 1.28 | 21.2% |
| LFE-PPN | **1.98** | 1.94 | 21.1% |

**The false-positive rate is within 10% of identical whether or not anyone is present.**
Empty scenes do not trigger the failure; they remove the true positives that were concealing it.
The benchmark hides a *constant absolute* error behind a *ratio*: 1.43 phantoms per frame still reads as 63% precision when the frame also holds 3.06 people, and no FROG benchmark frame holds zero.

**And the phantoms are furniture, not noise.** On unbroken person-free runs at 26.2 Hz (ego-motion ~1.2 cm/frame, an order of magnitude under the 0.3 m association radius, so no compensation needed):

| | phantoms/frame | recur within 0.3 m next frame |
| --- | --- | --- |
| LFE-Peaks | 1.31 | **88.0%** |
| LFE-PPN | 2.36 | **69.9%** |
| *ground-truth people* | 3.06 | *99.8%* |

So they are static scene structure correctly read as leg-like, and **short-window persistence or track-confirmation filtering cannot remove them** — a chair confirms as readily as a person.
That reconciles cleanly with this session's earlier ablation: shuffling a five-frame window costs 0.5pp because over ~2 s a standing person and a chair are equally stationary, so there is no evidence in that window to shuffle away.
The separating evidence — a person eventually displaces, a chair never does — only exists at a horizon of tens of seconds.
`docs/PAPER.md` carries the argument this leads to.

**What A32 does *not* yet say.** Whether temporal information closes the gap is untested; these are static baselines establishing that a gap exists, and the persistence measurement above rules out the cheapest way of closing it. `balanced` is not yet run for either model (~2 min each now), and no comparison against our own architectures is valid until A32b holds training data constant.

## Sweep cut to one point; the research question moves from crowded accuracy to sparse-scene value (2026-09-10)

*keywords: A1 scope cut, A21, A22, A24, --eval-zero-history, temporal contribution, zero_history, running order, LFE-Peaks, DR-SPAAM T=1 T=5*

**Owner's call.** The `dtime`/`time_frame` sweep is cut to the single in-progress `T=5, dtime=10` run.
The `dtime in {1,5,15,20,25}` points, the `time_frame in {1,3,10}` axis and both screening phases are **removed, not deferred**.

**The reasoning.** A sweep over `(T, dtime)` measures what the temporal *window* is worth, and the evidence says the model barely uses it: `dtime=0` scored 77.1% against `dtime=5`'s 80.7% (retracted, but indicative within one pipeline), and the detector fires ~2.5 phantoms per person-free frame — the exact failure motion should prevent.
Sweeping a dimension the model ignores yields a table of noise at several GPU-hours per point.

**The published ceiling makes this sharper.** DR-SPAAM's own FROG numbers are `T=1` **73.3%** and `T=5` **75.3%** — the best temporal model in the literature gains **+2.0pp** from five frames, on a benchmark where every frame holds 3-4 people and shape alone gets most of the way.
Meanwhile `LFEPeaksDetector` reaches 64.9% from a single frame with ~65K parameters at 1.76 ms.
If this project's temporal architectures end up as static shape detectors, LFE already does that job far more efficiently, and they have no rationale.

**New running order, recorded at the top of `TODO.md` §A because the chain is load-bearing**: A1 (one run) -> **A21** (does the model use history at all — *one eval pass, no training*) -> A22 (make it use motion, `--diff-channels` first) -> A24 (show where that pays).
A20 is independent.
A21 gates everything: running A24 before A22 would most likely show nothing, because there would be no temporal behaviour to reveal.

**A24 is the reframed claim** (owner's call): stop pushing accuracy on frames that are 100% populated, aim for *no worse than static there* and *better where scenes are sparse or empty*.
Measured on one axis — scene occupancy — with two controls: the same architecture under `--zero-history` (capacity-matched ablation) and `LFEPeaksDetector` (a real published static detector).
Nobody has published a sparse-scene or person-free number for this dataset family, so "temporal costs nothing when crowded and buys X when sparse" would be more novel than another point of crowded AP.
The risk is recorded up front: it may not work, in which case the finding is that a full-scan temporal architecture does not beat a small single-frame CNN on 2D LiDAR — a negative result worth publishing and cheaper to establish than to keep chasing.

**Unblocked A21's cheapest measurement.** `evaluate_auc()` has always accepted `zero_history`, but `evaluate.py` never exposed it, so the one-pass diagnostic was unreachable from the CLI.
Added `--eval-zero-history`, threaded through `eval_nn_model()`, with three tests: the flag exists, an AST check that every `evaluate_auc` call site in `evaluate.py` passes it (the A15 bug shape), and a semantic check that blanking history changes the input while leaving the current frame untouched.
Found and fixed while verifying it: adding measured percentages to `--eval-dtime`'s help text had silently broken `evaluate.py --help` — argparse `%`-formats help strings, so a literal `%` raises at render time and *only* at render time, leaving every test and every real invocation green.
Escaped, recorded in `gotchas.md`, and a regression test now runs `--help` on both CLIs.
134 tests passing.

---

## Val becomes three held-out recordings; training reaches exact parity with the paper (2026-09-10)

*keywords: cross-recording val, _SPLIT_TO_FILE_KEYS, _HELD_OUT_SPLITS, frog_10-31, frog_14-57, frog_15-53, guard bands removed, split == 0, paper parity, memorization probe, train_wp_auc*

**What prompted this.** Epoch 1 of the first fixed-pipeline run reported **val wp-AUC 82.7%** — above DR-SPAAM's published 75.3% and above the 68-75% band predicted in advance for the test number.
The written-in-advance rule said to suspect a leak at >=80%, so that was checked first: of 60,200 scan indices read by val windows, **zero** were training frames (window reach 40 scans against a 150-scan guard, min val-train distance 151).
No leak.
The number was simply not the quantity it was being compared against: val was carved from the *same two recordings* the model trains on, and same-recording holdout is a far easier problem than the cross-recording test set.

**The better design was available and free, and an earlier framing had obscured it.**
When the val geometry was chosen, "use a whole session as val" was described as costing a whole session of training data — true only if the extra recordings would otherwise have been trained on, which the owner's "official sessions only" decision had already ruled out.
The three unused recordings were then checked and are fully annotated: `frog_10-31` (64,238 scans, 62.6% of frames with people, 1.99 people/frame, 26.7 min), `frog_14-57` (70,062 / 60.8% / 1.90 / 29.2 min), `frog_15-53` (60,758 / 65.2% / 2.19 / 25.3 min), all continuous with zero pauses over 0.5 s.

**The split is now a partition of whole recordings**, so no window, no guard band and no near-duplicate frame can bridge two splits:

| split | recordings | annotated frames |
| --- | --- | --- |
| `train` | 11-36 + 12-43, the paper's own `split == 0` frames | **108,356** |
| `val` | 10-31, 14-57, 15-53 — whole | 195,058 |
| `test` | 16-41 — whole | 50,088 |

Two consequences worth stating separately.
**Training now reproduces the paper's set exactly** — 108,356 frames, not the 102,356 the guard bands had left, and not the 120,396 of the whole file (which would have meant beating published numbers partly on 11% more data).
That removes one of the three comparison caveats entirely.
**Early stopping now selects for cross-recording generalization**, the thing the test set actually measures, instead of for same-recording interpolation.

`_assign_split_labels()`, `GUARD_SCANS`, `VAL_BLOCKS` and the GUARD label are deleted rather than left dead; `git log` has them if a same-recording val is ever wanted again.
`FINAL_AUC_MAX_FRAMES = 20000` bounds the end-of-training val pass, since val is now ~4x the test set and scoring all of it would cost hours in the vote-decode loop for a number that only summarises a selection signal.

**Also added, on the owner's request: a memorization probe.**
Every gated epoch now scores a strided subsample of the *training* split beside the val gate and logs `train_wp_auc` and the gap.
It exists because the absence of one is what let four measurement bugs hide: FROG's interleaved holdout scored **identically** to a stride-9 train subsample (78.8% both at matched `dtime`), so no gap could ever appear.
Gated on val only — selecting on the probe would reinstate exactly the failure it exposes, and a test asserts that.
First reading, on the superseded same-recording val: train 85.5% vs val 82.7%, gap **+2.8pp** — a real generalization gap visible for the first time in this project.

**Checkpoint durability, hardened the same day.**
`save_checkpoint()` wrote in place, so a crash during the write left a truncated file with the previous best already gone; it now writes to a `.tmp` and `os.replace()`s it in.
`--resume` reset `best_val_auc` to `-inf`, so the first epoch after a resume overwrote `.best.pth` even when worse; the selection score is now stored as `val_wp_auc` and seeds the bar on resume.

**Tests.** 124 passing.
Nine block-carving tests removed with the code they covered, six cross-recording ones added (`TestCrossRecordingVal`), plus `TestMemorizationProbe` (4) and `TestCheckpointDurability` (3).
Verified against the real files, not only synthetically: train 108,356 / val 195,058 / test 50,088, three splits, no shared recording.

**The first launch of that design OOM'd, and the fix mattered more than the crash.**
`RuntimeError: Not enough memory resources are available` in `evaluate_loss`, ~2 minutes in, on a 17.1 GB machine.
Cause: `LidarFrameDataset` caches every preprocessed frame, and val had gone from 12,040 frames to **195,058** — ~4.9 GB of cache on top of train's ~2.7 GB.

Two things were wrong, and the second is the one worth remembering.
The cache was the proximate cause, fixed by never caching val at all: only a bounded stride of it is ever read, so caching 195k frames to touch 2,000 is pure waste.
But the real problem was that `evaluate_loss` walked the **entire** val split every epoch — nearly twice the training set — to produce a `val_loss` that **nothing gates on any more**, since A17 moved selection to wp-AUC.
That pass is now bounded by the same `--auc-max-frames` budget as the AUC gate, on a *fixed* stride rather than the old fresh-random-sample-per-epoch: a diagnostic read as a trend across epochs has to be the same sample each time to mean anything.

Watch for this shape generally: **when a split grows, every full pass over it grows with it, including the ones nobody is using any more.**
Three tests lock it down (`TestValPassIsBounded`) — the pass is capped, the sample is stable across epochs, and the val frame dataset is not cached while the training one still is.

**And the first cross-recording val was itself wrong, for a reason worth keeping.**
Epoch 1 reported **val wp-AUC 50.8% against a train probe of 85.7% — a +35pp gap**, far past anything cross-recording difficulty explains (DR-SPAAM transfers to a different recording at 75.3%).
Every corruptible axis checked out: odometry coverage 99.77-99.99% on all three extras, scan medians 4.14-4.56 m, annotation ranges and angles all in line with train and test.
The difference was composition. **Both official FROG files are 100% frames-containing-people; the three extras are ~62%.**
The authors curated the benchmark files and left the extras raw — conditioned on being populated the extras average 3.13-3.36 people/frame against test's 3.07, so the *scenes* match and only the empty stretches differ.
Putting ~38% empty frames into val hands a precision-recall metric a mass of pure false-positive opportunity that neither train nor test contains.
Held-out splits now annotate populated frames only, which is a **no-op on the test recording** — that is what makes it a protocol match and not a thumb on the scale.
After the fix: train 108,356 at 3.93 people/frame, val 122,405 at **3.22**, test 50,088 at 3.07.

**The empty-scene false-positive rate, measured — after one wrong turn worth recording.**

First reading: the extras' ~38% person-free frames are ready-made false-positive data.
A probe checkpoint scored 3.65 detections per such frame against 3.85 on populated ones.
Second reading, on noticing the zero-runs are long (up to 9,233 frames, ~6 min) and carry *more* short returns than populated frames (40.4% of beams under 3 m against 26.0%): those frames must be un-annotated rather than empty, so the number measures nothing.
**Third and final reading: the second was wrong, and over-weighted two weak signals.**
Both have innocent explanations — a tour-guide robot can easily spend six minutes somewhere quiet, and somewhere quiet is typically also somewhere narrow, which is what produces short returns.

The direct test settles it.
At a 0 -> N transition the person count jumps by a **median of 1**, occupancy trails down to ~1.1-1.4 *before* a gap and resumes at ~1.4-1.6 *after* it — people walking out of view one at a time, not an annotator stopping mid-crowd.
And within annotated runs the frame-to-frame count dynamics are **indistinguishable from the official test recording**: mean |delta count| 0.014-0.017 for the three extras against 0.022 for `frog_16-41`, with 0.17-0.25% of steps changing by more than one against test's 0.17%.
That second statistic is the load-bearing one for everything else: **the frames this project evaluates on are annotated to the same standard as the benchmark's own.**

**Measured, thresholding each detection on its own confidence the way a deployment would** (probe checkpoint, 450 person-free + 450 populated frames):

| det-conf threshold | FP/frame, person-free | recall, populated | FP/frame, populated |
| --- | --- | --- | --- |
| 0.1 | 5.14 | 67.7% | 3.13 |
| 0.3 | 2.51 | 51.9% | 1.27 |
| 0.5 | 1.39 | 40.5% | 0.62 |
| 0.7 | 0.73 | 30.5% | 0.30 |
| 0.9 | 0.30 | 16.9% | 0.12 |

At a setting recovering half the real people, the detector invents ~2.5 per frame in a scene with nobody in it; even at 0.9 it is one phantom every ~3 frames, ~8 per second at 26 Hz.
Note the shape: FP/frame on person-free scenes runs ~2x the populated rate at every threshold, so this is not a special empty-scene pathology so much as a high false-positive rate generally, with no true positives to offset it.
Knee-height LiDAR and furniture legs is the obvious suspect, unverified.

Caveats: the probe is a *retracted* checkpoint (pre-A18, superseded run), and the recall column uses a greedy 0.5 m match rather than the Hungarian matching `_prec_rec_2d` uses, so it is not comparable to a published AP.
The FP column is the robust half — those frames have no ground truth at all, so no matching convention affects it.

**The consequence for the benchmark stands regardless: no published FROG AP — DR-SPAAM's 75.3% included — penalises this at all**, because neither official file contains a single person-free frame.
`TODO.md` A20 is now actionable rather than blocked: the data exists and is genuine.

**Decided.** Two restarts today were spent on val design; this is the one to let run.

---

## Four measurement bugs found and fixed; every FROG number retracted (2026-09-10)

*keywords: _SPLIT_TO_FILE_KEYS, _assign_split_labels, _split_at_gaps, GUARD_SCANS, SESSION_GAP_S, _load_odom, evaluate_auc dtime, BEAM_BATCH, val wp-AUC early stopping, frog_16-41_test.h5*

**What prompted this.** The owner asked why `dtime=10` "overfit" at 76.7% when an older `dtime=5` run with no odometry had reached 80.7%, why a dataset the size of FROG could overfit at all, and why a 476K-parameter temporal model was barely beating single-frame LFE-PPN.
None of those turned out to be true.
All three are artifacts of four independent measurement bugs, each verified here against the real data and the real `dtime=10` checkpoint rather than read out of the code.

**The decomposition.** Held-out val split, same architecture, same real odometry, same regularization throughout:

| step | wp-AUC | delta |
| --- | --- | --- |
| as reported in `performance-log.md` (epoch-8 weights, eval `dtime`=1) | 76.7% | — |
| the weights actually saved to disk (epoch 3), eval `dtime`=1 | 76.3% | reproduces the bug |
| + pass `dtime` to `evaluate_auc()` (1 -> 10) | 78.8% | **+2.5** |
| + don't early-stop on a noise dip (epoch 3 -> 14) | 79.7% | **+0.9** |
| + score it on the contaminated "test" set, as the log did | 80.1% | +0.4 |

76.3 + 2.5 + 0.9 + 0.4 = 80.1, the number already sitting in `results/eval_frog_realodom_dtime10_test.log`.
Nothing is left over for `dtime` or odometry to explain: the "regression" the previous three entries were investigating does not exist.

**Bug 1 (A11): `split="test"` has always loaded the training file.**
`_load_h5()` globbed every `.h5` in the directory regardless of the requested split, then marked **every** frame as annotated for any split that was not `train`/`val`.
`frog_16-41_test.h5` had never been downloaded, so `split="test"` returned all 120,396 frames of `frog_11-36_12-43_train_val.h5` — the train split plus the val split.
Verified directly: `train` -> 108,356, `val` -> 12,040, `test` -> 120,396, and `test.scans[0][0]` bit-identical to `train.scans[0][0]`.
Every `eval_frog_*_test.log` since 2026-08-13 reports `120396 annotated frames`.
Corroborating signature, found before the fix: LFE-Peaks measures 74.6% here against the paper's own 64.9%, LFE-PPN 67.7% against 66.5% — FROG-trained published weights scored on their own training data.
Fixed with an explicit `_SPLIT_TO_FILE_KEYS` whitelist. It has to be a whitelist: the test file carries no `split` field, so merely downloading it would have made a glob-based `split="train"` swallow all of it through the same `det_mask = ones` branch.
Downloaded all six sessions; `download()` now defaults to `which="all"`.

**Bug 2 (A12): the val split was 99% adjacent to the training split.**
FROG's own `split` field is a per-frame random 10% holdout — 12,040 isolated single frames scattered through the training recordings, **99% of them exactly one scan (~38 ms) from a training frame** (median distance 1, max 3).
Measured consequence on the `dtime=10` checkpoint at matched eval `dtime`: a stride-9 train subsample and the val split score **identically, 78.8% both** (and 76.3% both at eval `dtime`=1), because they are the same frames offset by one scan.
Replaced with `_assign_split_labels()`: 10 evenly-spaced contiguous val blocks per recording, ~10% of frames, each with a 150-scan (~5.7 s) guard band dropped from training on both sides.
The guard is deliberately a fixed width rather than `T * dtime` — a `dtime`-dependent guard would give every sweep point a different split, destroying comparability along the exact axis A1 sweeps.
Verified on real data: nearest train frame to any val frame is now **151 scans**, up from 1; train 102,356 / val 12,040 / test 50,088 annotated frames.

**Bug 3 (A15/A16/A17): the reported number described neither the saved weights nor the training configuration.**
Three separate faults in one code path.
`evaluate_auc()`'s `dtime` defaults to 1 and `train_model()` never passed it, so a model trained on 0.38 s windows was scored on 0.038 s ones (76.3% vs 78.8%).
The final AUC was computed at line 1527 and the best checkpoint loaded at line 1546 — *after* — so the printed number came from the discarded last epoch while the file held the best one; verified, the `dtime=10` run printed 76.7% from epoch-8 weights and saved epoch-3 weights stamped "epoch 8".
And early stopping gated on `val_loss`, which `compute_loss()` scales per *batch* (`pos_weight = n_neg/n_pos`, `weight=None` for person-free frames) and then averages over batches rather than samples: on one checkpoint and 2,000 frames, a contiguous block gives train 0.0908 / val 0.2915 while a random sample of the same data gives train 0.1973 / val 0.1775 — a 3x swing on batching alone.
**There was never any overfitting**: beam-level person AUC at matched `dtime` is train 0.9738 vs val 0.9696, and on every like-for-like loss comparison val is *better* than train.
The direct proof is two runs with identical architecture, `dtime` and data: the 2026-09-08 one improved steadily to epoch 14 and stopped at 19, while the 2026-09-09 one happened to hit `val_loss` 0.1987 at epoch 3 — a value the first did not reach until epoch 19 — which set an unbeatable bar and let `patience` kill it at epoch 8 with `train_loss` still falling.
Fixed: `dtime` passed at every call site and stored in the checkpoint (so `evaluate.py --eval-dtime` defaults to the value the weights were trained at); the best checkpoint is restored *before* the final AUC and stamped with its own epoch rather than the stopping epoch; early stopping and model selection both run on val wp-AUC, computed each epoch on a strided subsample (`--auc-max-frames`, default 2000) with the full split used for the reported number.
Budget raised to `--epochs 100 --patience 15`, roughly matching LFE-Peaks/LFE-PPN's own recipe.

**Bug 4 (A18): a frame's prediction depended on which other frames shared its minibatch.**
`SpaceTimeCNNDetector.forward()` did `x.permute(2, 0, 1).unsqueeze(0)` on a `(B*N_beams, T, 1)` input, folding the batch into the **beam** axis — a batch of 4 scans became one 2880-beam "scan".
GroupNorm statistics, `_SEBlock`'s global channel gate and the beam-axis convolutions (receptive field 177 beams) then all leaked across the seam between frames.
`FullScanTCNDetector` had the same fault in its spatial backbone.
Because evaluation batches are consecutive frames in time, a larger batch leaked more temporal context and scored higher: on the real val split at `dtime`=10, beam-level person AUC ran **0.9644 (batch 1) -> 0.9696 (4) -> 0.9765 (16) -> 0.9810 (64)**.
`evaluate.py` defaults to batch 16, `train.py` used 4, and the robot runs `forward_one()` at batch 1 — so every published figure was inflated relative to what actually gets deployed.
Both models now set `BEAM_BATCH=True` and take `(B, N_beams, T, 1)`, keeping frames independent throughout.
Old checkpoints still load (shapes are unchanged) but compute different outputs, so this invalidates the numbers rather than shifting them.

**Also fixed while here (A13, A14, A19).**
`session_gap_s` lowered from 300 s to 0.5 s: the 60 recording pauses inside the two training recordings (up to **74.3 seconds** long, together 15-20% of wall time) were sitting inside sequences, so a nominal 0.38 s window that straddled one really spanned 74 seconds of two unrelated scenes — 2.4% of `dtime=10` windows did, scaling linearly with `dtime` and so biasing the sweep against the wide windows it was measuring.
Largest in-sequence gap is now 464 ms, down from 74,300 ms.
`_load_or_fake_odom()` became `_load_odom()` and raises instead of silently substituting all-zero odometry on any failure — zero odometry is indistinguishable from a stationary robot, which made every ego-motion conclusion unfalsifiable.
`informational_capacity.py` was calling `aligned_raw_scan()` without `beam_spacing`, so it used DROW's 0.5 deg/beam on FROG's 0.25 deg data and applied half the rotation correction training does; fixed and the whole `(T, dtime)` grid re-run.

**Recorded while investigating, not a bug:** FROG's published heading odometry is **quantized to exactly 1.0 degree** (361 unique values, steps exact multiples of 1 deg, p90 2 deg, max 4 deg) at 10 Hz against 26.2 Hz scans — a 4-beam step with +/-2 beams of quantization noise at 0.25 deg/beam.
Below `dtime~=5` the true rotation is smaller than that quantum, so the correction injects more jitter than it removes.
That is a real mechanism for "real odometry scored below no odometry" at small `dtime`, and a reason to expect odometry to pay for itself only at wider windows.
Translation is unaffected (x/y quantized at ~0.5 cm, against ~1.2 cm of real motion per frame).

**Tests.** 120 passing.
Three new files, written before the fixes and failing against the old code (28 failures at that point): `tests/test_frog_splits.py` (28 cases — split selection, block carving, guard bands, session boundaries, odometry strictness), `tests/test_batch_independence.py` (29 cases — every full-scan model, both heads, batch sizes 2/4/16, batch order, seam beams, `forward_one` parity), `tests/test_train_eval_consistency.py` (10 cases — `dtime` reaching the evaluator including a static AST guard over every call site, reported-AUC-matches-saved-weights, wp-AUC early stopping).
`tests/test_frog_session_splitting.py` was folded into `test_frog_splits.py` and deleted: it asserted the superseded 300 s threshold, the glob-based loading and the zero-odometry fallback.
All four dataset fixes were additionally verified against the real recorded files, per `dos-and-donts.md` — a synthetic pass does not close a dataset-logic item.

**Checkpoint durability, hardened the same day** after the owner asked whether an interrupted run's intermediate weights actually survive.
They did — `.best.pth` carries weights, optimizer state and epoch on every wp-AUC improvement — but two things would have bitten on recovery.
`save_checkpoint()` wrote in place, so a crash *during* the write left a truncated file with the previous best already gone; it now writes to a `.tmp` and `os.replace()`s it in, which is atomic on POSIX and Windows alike.
And `--resume` reset `best_val_auc` to `-inf`, so the first epoch after a resume overwrote `.best.pth` unconditionally, even when it scored worse — destroying the checkpoint the resume existed to preserve.
The selection score is now stored in the checkpoint as `val_wp_auc` and seeds `best_val_auc` on resume.
Three regression tests added (`TestCheckpointDurability`), 123 passing.

**Also answered, and worth recording because the question will recur: FROG's interleaved holdout cannot serve as a second val signal.**
Measured on one checkpoint at two eval `dtime`s, 12,040 frames each side: a stride-9 *train* subsample and the interleaved val split score **identically** — 76.3%/76.3% at `dtime`=1 and 78.8%/78.8% at `dtime`=10.
The interleaved number *is* the training-set number, so it adds nothing a train-split evaluation does not already give for free, and using it as a real holdout would cost ~10% of training frames (about 85% of its frames fall inside what are now train blocks).
The useful thing in that neighbourhood is the **gap** between a train-subsample AUC and the block-val AUC — a direct memorization probe, and one that would have made this whole class of bug visible immediately.

**Retracted tuning sweeps, moved here from `performance-log.md` so the numbers are not lost with the table.**
All three were measured on the contaminated "test" split (FROG's 120,396 *training* frames), so none of their settings should be restored — they are runs to redo, not configuration to keep.

*NMS / vote-clustering sweep:* FullScanTCN `vote_collect_radius` over 0.3-1.3 picked **0.7 -> 69.7%** (+0.2pp over the 0.5 default); LFE-PPN `nms_radius` over 0.7-1.0 picked **0.9 -> 61.2%** (+0.5pp over 0.8).
`min_thresh`/`score_thresh` were non-binding for both across the whole range — the score distributions simply do not fall where either sweep looked.

*Operating-threshold selection* (best-F1 point vs. a 75%-precision target, no retraining), FP saved / FN added: SpaceTimeCNN 968/718 (favourable 1.35:1), TemporalUNet 4179/3567 (1.17:1), LFE-Peaks 1278/936 (1.37:1), FullScanTCN 3944/4382 (unfavourable 0.90:1), LFE-PPN 3966/3951 (break-even).

*Tracker invocation-rate sweep* (SpaceTimeCNN, `SimpleTracker` at `radius=0.5, min_hits=3`): stride 5 (~125 ms) 77.6% -> **78.6%** (+1.0pp); stride 40 (~1 s) 80.4% -> 75.1% (**-5.3pp**).
The stride-40 regression was explained rather than merely observed — `rejected-ideas.md`'s tracker entry has the radius/min_hits grid that isolated it, and that mechanism is unaffected by the retraction.

**Decided.** A1's sweep restarts from scratch on the fixed pipeline; `dtime=5` is dropped as a target (owner's call) in favour of `dtime=10, T=5` or whatever the re-run capacity grid points at.
Every existing checkpoint is worth re-scoring under the fixed protocol before any new GPU time is spent.

---

## Checked whether under-regularization explains the fast overfit and the old-vs-new gap — it doesn't, and an earlier memory note about dropout was wrong (2026-09-10)

*keywords: dropout, weight_decay, amsgrad, base_dr_spaam_drow_cfg.yaml, n_params, DrowDetector, DrSpaamDetector, SpaceTimeCNN, LFE-PPN parameter count*

**What prompted this.** After the `dtime=10` full run's fast overfit (previous entry), the owner suspected under-regularization, reasoning that a bigger model (this project's own architectures vs. LFE-PPN) should need more rigorous regularization to match.

**Self-correction, caught while checking this.** The previous entry's "confirmed the model's dropout is 0.1, light regularization" was wrong — that grep only checked `full_scan.py`'s class constructor defaults, not `train.py`'s actual CLI/`_build_model()` override.
`train.py`'s real default is **dropout 0.5** (`_default_args()`, `parser.add_argument("--dropout", ..., default=0.5)`), applied uniformly to every trainable detector via `_build_model()`'s `dr = getattr(args, "dropout", 0.5)` — confirmed unchanged across all three commits in this repo's history that touch `utils/train.py`, so this was never a regression, just a previously-uncaught documentation error (`docs/RESEARCH.md` §5.4 separately documents 0.1 as this project's own *design intent* for the novel architectures specifically — that intent was simply never wired into the CLI default).

**Checked against the actual papers, not assumed.** Fetched `VisualComputingInstitute/2D_lidar_person_detection`'s own training config (`dr_spaam/cfgs/base_dr_spaam_drow_cfg.yaml`) and optimizer (`dr_spaam/pipeline/optim.py`) directly: `dropout: 0.5` — an exact match to this project's default — and `optim.Adam(model.parameters(), amsgrad=True)` with **no weight decay at all**, versus this project's own `weight_decay=1e-4` default.
Neither the official config (`augment_data: False`) nor this project's own defaults (`beam_shift_aug=0`, `range_jitter_scale=0.0`) use data augmentation.
On every one of these three axes, this project's training is already at parity with or more heavily regularized than the paper it ports from.

**Checked model size directly rather than estimating it.** `sum(p.numel() for p in net.parameters())` on each class: `SpaceTimeCNN` (this sweep's architecture) 476,262; `FullScanTCN` 67,206; `TemporalUNet` 651,521; `DrowDetector` 1,518,918; `DrSpaamDetector` 1,978,054 — `SpaceTimeCNN` is smaller than either official cutout-based architecture (both trained/loaded at the identical 0.5 dropout), though genuinely ~3-7x bigger than LFE-Peaks/LFE-PPN by ONNX file-size estimate (~65K/~180K params, no `onnx` package available in this environment to count exactly).

**Conclusion.** The owner's "bigger model needs more regularization" intuition is directionally correct relative to LFE specifically, but LFE isn't the other half of the actual regression under investigation (old zero-odometry `dtime=5` at 80.7% vs. new real-odometry `dtime=10` at 76.7%) — both of those used the identical `SpaceTimeCNN` architecture and the identical dropout/weight_decay defaults, so regularization strength cannot be what explains that specific gap.
The dtime value and the pipeline change (real odometry alignment, session-splitting fix, angle-wraparound fix) remain the live candidates.
Decided: run `dtime=5` next on the fixed pipeline (not `dtime=9` or a higher-dropout rerun as previously planned) — it isolates the pipeline change from the `dtime` change by holding `dtime` fixed at the old sweep's own peak (`TODO.md` A1).

---

## First full-scale fixed-pipeline training result: `T=5, dtime=10` reaches 76.7% wp-AUC, but overfits fast (2026-09-09)

*keywords: dtime=10, spacetime_cnn_dtime10, wp-AUC, early stopping, overfitting, dropout, checkpoints_frog_dtimesweep_full*

**What.** Following the informational-capacity grid's finding that peak signal sits near `dtime=9` at `T=5` (previous entry), launched the first full-scale (`--epochs 30 --patience 5`, no `--subsample`) training run on the fixed real-odometry pipeline at `T=5, dtime=10`.
`dtime=10` had never actually been screened in phase 1a — both earlier launch attempts were killed mid-run for other investigations — so this run had no screening number to compare against, only `dtime=1`/`5`/`15`'s (76.9%/76.9%/73.0%).

**Result.** Val wp-AUC 76.7%, best checkpoint at epoch 3 of 30 (val_loss 0.1987), early-stopped at epoch 8 once `patience=5` was exhausted (val_loss never beat epoch 3 again: 0.2064, 0.2208, 0.2226, 0.2165, 0.2230).

**Not directly comparable to the old (pre-fix) `dtime=5` number of 80.7%** in `performance-log.md`'s superseded table: different `dtime`, that run used no real odometry, predates session-splitting, and predates the angle-wraparound fix.
This is the first fixed-pipeline full-scale number on file, kept in its own table in `performance-log.md` rather than merged into the old one.

**Open question, not resolved by this one data point**: full (non-subsampled) data converging then overfitting by epoch 3 is faster than expected — full data was expected to be at least as overfitting-resistant as the 20%-subsample screening runs, not more prone to it.
This is also the first point with both a proxy value and a trained `wp-AUC`; not enough points yet for the rank-correlation check `memory/interpreting-evaluation.md` calls for, but tracking starts here.
Investigated (and mostly ruled out) same-day, next entry: whether under-regularization explains it.

---

## Informational-capacity proxy extended to a full `(T, dtime)` grid — the real plateau is governed by total window span, not either axis alone (2026-09-09)

*keywords: informational_capacity.py, informational-capacity-proxy.md, time_frame, dtime, window span, mutual_info_classif, plateau*

**The single-axis proxy (`T=2` only, previous entry below) never found a plateau because it was looking on the wrong axis.**
Extended `utils/informational_capacity.py` (moved out of the scratchpad into a permanent, documented tool) to a `T in {2,3,5,10} x dtime in {1,5,10,15,20,25,35,50}` grid, using per-beam standard deviation across the full `T`-frame aligned window as the feature (generalizing the earlier `T=2` `|diff|` feature — identical up to a monotone transform at `T=2`), plus an actual Shannon mutual-information estimate (`sklearn.feature_selection.mutual_info_classif`, Ross 2014) alongside the AUC measure, not just AUC alone.

**Finding**: every row's peak AUC lands at a total window span (`(T-1) * dtime * 38.15ms`) of roughly 1.1-1.7 seconds, regardless of how that span is split between `T` and `dtime` — `T=2` peaks at `dtime=35` (1.34s span, AUC 0.652), `T=3` at `dtime=15-20` (1.14-1.53s, tied at 0.669), `T=5` at `dtime=10` (1.53s, 0.677), `T=10` at `dtime=5` (1.72s, 0.679, the grid's overall best).
Past its own peak span, every `T` declines monotonically, down to a clear floor by `dtime=50` (`T=10`'s AUC there, 0.598, is barely above the `dtime=1` floor of ~0.57-0.64 across all `T`).

**What it means, and what was decided.**
Total real-time span, not `T` or `dtime` individually, is the variable that actually matters — a finding that also lines up with Winter (1990)'s ~400-600ms single-step timing already cited in this project's own `T=5` design rationale (a full gait cycle, ~0.8-1.2s, sits at the low end of the found peak range).
The training-based sweep (`TODO.md` A1) is revised to concentrate on configurations near this peak span (`T=5, dtime~=10`; `T=10, dtime~=5`) rather than the wider `dtime in {20, 25}` points the original plan weighted toward, which this proxy predicts are already past it.
Full grid, methodology, and citations: `memory/informational-capacity-proxy.md` (new), `memory/performance-log.md`.

**Refined the same day, on the owner's request to check both sides of the peak properly and test wider `T`.**
The grid above had only one or two points below each `T`'s own peak — not enough to confirm a real interior maximum rather than a slow monotonic trend that just hadn't turned yet (exactly the mistake the single-axis `T=2` pass made before this entry's own first fix).
Re-ran with `T in {2,3,5,10,15,20,25}` and `dtime` sampled *per-`T`*, targeting a fixed set of total spans (0.15s to 8s) rather than reusing one `dtime` list for every `T` (which undersamples the peak at large `T`, where it sits at a small `dtime`).
Confirmed: a genuine rise-then-fall at every one of the seven `T` values, peak spans now clustering even more tightly (1.18-1.83s: `T=2` 1.18s, `T=3`/`5`/`10` 1.37s each, `T=15` 1.60s, `T=20` 1.45s, `T=25` 1.83s).
**New finding**: the peak AUC value itself saturates once `T>=5` (0.652, 0.670, 0.678, 0.679, 0.679, 0.678, 0.677 for `T`=2,3,5,10,15,20,25, essentially flat past `T=5`) — stacking more than ~5 frames buys this proxy nothing once `dtime` hits the right span, so the wider `T` values are now a lower priority for real training, not a higher one.
Whether this proxy's peak actually predicts the *trained* network's peak is a separate, explicitly open question, not assumed — `memory/interpreting-evaluation.md` has the reasoning and the concrete check (measure the rank correlation directly, once real training data exists) that resolves it.

---

## Cross-dataset calibration and a no-training informational-capacity proxy, both requested to verify the pipeline before resuming the sweep (2026-09-09)

*keywords: calibration, world-frame Cartesian, nearest-neighbour residual, cKDTree, informational capacity, person-background separation, roc_auc_score, zero-cost proxy*

**Cross-dataset odometry calibration, in world-frame Cartesian coordinates, independent of `aligned_raw_scan()`'s own approximations.**
Converted real DROW and FROG scans to world-frame points using each frame's own odometry pose (a from-scratch rigid transform, not routed through `aligned_raw_scan()`, so this checks the odometry data and the transform math directly rather than that function's specific implementation), then measured the correspondence-free nearest-neighbour residual between consecutive frames' point clouds via `scipy.spatial.cKDTree` — small and comparable between datasets is the expected signature of correct odometry+transform on both sides.
Result: DROW `dtime=1` median 1.00cm (the long-trusted reference), FROG `dtime=1` median 1.56cm (same order of magnitude), FROG `dtime=15` median 5.74cm but p75=16.22cm (grows with real motion as expected, but a heavier tail than ideal — not yet root-caused, not blocking, noted for later).

**No-training "informational capacity" proxy** (the owner's own framing, implemented literally rather than debated in the abstract): for real annotated FROG frames, label each beam "person" (within a few beams of a real annotated position) or "background", and measure the AUC of the raw aligned frame-to-frame beam difference *alone* — no network, no training, one scalar feature — as a person-vs-background classifier, at each `dtime`.
Result: 0.568 (`dtime=1`) -> 0.604 (5) -> 0.624 (10) -> 0.636 (15) -> 0.645 (20) -> 0.650 (25), monotonically increasing with no plateau across the whole tested range; the person/background median-diff separation ratio grows from 1.35x to 4.68x over the same range.

**What it means, and what was decided.**
The proxy is real evidence that wider `dtime` carries *more* decodable signal, not less — which reframes the paused screening result (`dtime=15` scoring below `dtime=1`, `CHANGELOG.md` entry above) as very likely a training-convergence artifact of the cheap screening budget, not evidence that wider windows lack signal.
The sweep (`TODO.md` A1, previously paused pending this check) is resumed on this basis.

---

## `aligned_raw_scan()`'s rotation delta didn't handle angle wraparound; found while investigating why `dtime=15` screened worse than `dtime=1` (2026-09-09)

*keywords: aligned_raw_scan, theta wraparound, arctan2, rotation shift, dtime=15, dtime=1, background frame-to-frame diff*

**The owner asked a fair "does this make sense?" question**: `dtime=15`'s screening AUC (73.0%) came in below `dtime=1`'s (76.9%), which is counter-intuitive if wider windows show more real motion.
Checked directly rather than reasoned about in the abstract: a diagnostic measuring median frame-to-frame beam difference for background (non-person) beams, before and after `aligned_raw_scan()`, found alignment made background *more* inconsistent across frames, not less, at both `dtime=1` (1.40cm raw -> 1.61cm aligned) and `dtime=15` (14.25cm raw -> 16.60cm aligned) — the opposite of what a working ego-motion correction should do.

**Root cause, confirmed, not guessed**: `aligned_raw_scan()`'s rotation-delta computation (`theta[-1] - theta`) never wrapped the result — real FROG odometry's heading spans the full +/-pi range, and ~1.2% of sampled real windows crossed that boundary, turning a true small rotation into a ~2*pi-radian spurious one (hundreds of beams' worth of bogus shift for that frame).
Testing every sign combination of (x, y, theta) first, to rule out a simple sign-flip: none came close to fixing it, which is what pointed at wraparound instead.

**What was NOT the cause, checked and ruled out**: after fixing the wrap, the residual gap (raw 14.25cm vs rotation-only-fixed 16.10cm) barely moved, and isolating translation's own contribution (16.10cm -> 16.60cm) showed it adds very little on top.
The bulk of the residual "alignment adds a few cm of apparent background noise" effect traces to the linear beam-interpolation itself: shifting a scan by a fractional beam count and linearly blending two adjacent beams produces a value that matches neither side of a real depth discontinuity (a doorway edge, a column) — an inherent property of sub-beam linear interpolation on a signal full of sharp edges, not a further bug.
This residual is modest (~10-20% relative to the raw baseline) and does not look large enough to be the primary explanation for the observed AUC gap on its own.

**What changed in the code.**
`library/follow_the_drow/utils/drow_utils.py`'s `aligned_raw_scan()`: the rotation delta is now wrapped via `arctan2(sin(delta), cos(delta))` before being converted to a beam shift.
New regression test `tests/test_geometry.py::TestAlignedRawScan::test_rotation_delta_wraps_across_the_pi_boundary`, an exact analytical case (170deg -> -170deg, a true 20deg turn) that the un-wrapped version gets wrong; full suite now 64/64.

**What it means, and what's still the leading explanation for the actual AUC gap.**
This bug is real and now fixed, and every prior real-odometry training result technically predates it — but at ~1.2% window incidence, it's unlikely to be the dominant cause of a 3.9pp gap between two screening runs, and the interpolation-noise residual above is modest too.
The better-evidenced explanation remains what the same-day screening logs already show directly: `dtime=15`'s val_loss peaked at epoch 6 (0.1543, *better* than `dtime=1`'s epoch-6 value of 0.1682 at that point) then got worse at epoch 7 (0.1764, patience triggered) — a classic overfitting signature on the 20%-subsample screening data — while `dtime=1` improved smoothly through all 8 epochs with patience never triggered.
That pattern is consistent with a mechanism this project has already documented elsewhere (`interpreting-evaluation.md`'s DROW-annotation-scarcity finding): a temporally richer input has more capacity to overfit a small effective training set within a few epochs, while a near-static input is a simpler function that a limited budget can still fit well.
Not fully resolved — phase 2's full-data, full-epoch runs (`TODO.md` A1) are the actual test of this, not this entry.

---

## `train.py --help` crashed on cp1252, blocking a flag check before the A1 sweep launch (2026-09-09)

*keywords: train.py, --help, UnicodeEncodeError, cp1252, epilog=__doc__, RawDescriptionHelpFormatter*

**`train.py --help` crashed with the same `UnicodeEncodeError` class already fixed once in `_download_file()` this session** — found while checking the exact `--dtime`/`--subsample`/`--epochs`/`--patience` flag names before launching A1's first screening run, not by accident.
Previously flagged (an earlier session) as "cosmetic, out of scope"; this time it actively blocked verifying a command before a real run, so it was worth fixing rather than working around.

**Root cause, more specific than the earlier fix**: the parser sets `epilog=__doc__` (the whole module docstring, `RawDescriptionHelpFormatter`), so any non-cp1252 character anywhere in that docstring reaches the console — not just characters in `argparse` `help=` strings, which was the initial (too narrow) hypothesis.
Checked precisely (encode-tested every non-ASCII character against `cp1252`, not just flagged every non-ASCII character): the module docstring and a handful of `help=` strings carried `→` and `≈`; other non-ASCII characters elsewhere in the file (em-dashes, `×`, `§`, `–`) are actually valid cp1252 and were never the problem.

**What changed in the code.**
`utils/train.py`: every confirmed-unsafe character replaced with an ASCII equivalent (`→`->`->`, `≈`->`~`, `°`->`deg`) across the module docstring, `detect_device()`'s docstring, three `help=` strings, and a few internal docstrings/comments not currently printed anywhere but fixed for consistency.
Verified: a script that encode-tests every character in the file against `cp1252` now reports none unsafe; `train.py --help` runs to completion; `pytest tests -q` still 63/63 (unrelated to this file's logic, but confirms nothing broke).

**What it means.**
This class of bug is per-file, not fixed once — anything printed to this Windows console needs the same check before being trusted; `memory/gotchas.md`'s existing entry already states the general rule.

---

## DR-SPAAM published-weights evaluation abandoned: T=5 skipped by decision, T=1 stalled with an unexplained duplicate process (2026-09-09)

*keywords: eval_frog_official_drspaam.log, DrSpaamDetector, stall, duplicate process, AppData Python312, .venv312*

**T=5's own evaluation was skipped by owner decision** once T=1 looked close to finishing — the FROG paper's own Table 4 numbers (T=1: 73.3%, T=5: 75.3%) were already documented and cited, and re-deriving them independently wasn't judged worth the remaining ~5h.

**T=1 then failed to complete either, but not by timing out — it stalled.**
After a consistent ~7 fr/s for 4h40m, progress froze at 120385 of 120396 frames (>99.9%) and stayed there for 12+ minutes with no further movement before being killed.

**A second process was found running the identical script** at the same time: same `-c` script text byte-for-byte, but invoked via `C:\Users\...\AppData\Local\Programs\Python\Python312\python.exe` rather than this project's `.venv312` — and *that* process, not the intended `.venv312` one, held almost all the accumulated CPU time (the `.venv312` process itself showed essentially none, despite being the one whose output was reaching the log file).
**Partially explained, later the same day**: `.venv312\pyvenv.cfg` sets `home = C:\Users\...\AppData\Local\Programs\Python\Python312` — that "unrelated" interpreter is actually this venv's own base install, not a stranger.
The paths being linked isn't a coincidence, but *why* a process ended up executing via the base path instead of the venv's own launcher is still not root-caused; noted in `memory/gotchas.md` for next time this pattern shows up.
Both processes were killed once found.

**What was decided, same day.**
`memory/performance-log.md`'s `DrSpaamDetector`/FROG cell cites the paper's numbers for both T=1 and T=5, alongside the already-passed (and still valid) official-weights-compatibility check (zero missing/unexpected keys loading into this repo's `DrSpaamDetector`).
Not reopened: the explicit call was to put further effort into this project's own novel results (`TODO.md` A1) rather than into reproducing a number the paper already reports.
`TODO.md` A2 is closed on this basis.

---

## `FROG_Dataset.scan_time` lost sub-minute precision to a float32 cast; real per-frame motion measured directly (2026-09-09)

*keywords: scan_time, odoms["t"], float32, float64, Unix epoch precision, dtime, ego-motion, human displacement, det_wp*

**`FROG_Dataset.scan_time` and `odoms["t"]` were stored as `float32` at Unix-epoch magnitude (~1.4e9) — every consecutive raw frame collapsed to an identical timestamp, quantized in ~128-second jumps.**
Found while writing a sanity-check script that reads `scan_time` directly for the first time (nothing in the training/eval pipeline had ever consumed it): `np.diff(scan_time)` was `0.0` for essentially every pair, `128.0` at each precision-boundary crossing.
Checked whether this had silently corrupted anything already trusted: it had not — session-gap detection (`_load_h5`) and the real-odometry `np.interp` alignment (`_load_or_fake_odom`) both already used the full-precision `float64` array read straight from the `.h5` file, before this cast ever happens, so every finding and number that depends on those (the 2026-09-08 entries below) stands.
This was a real trap for the *next* thing to read `scan_time` directly, though — which is exactly what happened here.

**What changed in the code.**
`library/follow_the_drow/datasets/frog_dataset.py`: the `dtype`'s `"t"` field and both `timestamps_s.astype(...)` call sites now use `np.float64` instead of `np.float32`.
Verified: `pytest tests -q` still 63/63; `scan_time`'s consecutive diffs are now the real ~38ms native spacing, not 0/128.

**Sanity check, run once the fix was in place: how much does the robot itself move per frame, at each `dtime`?**
FROG's native rate measures 26.2 Hz (~38.15 ms/frame), not the documented 40 Hz.
Real (not estimated) odometry, median robot displacement per `dtime`-frame step:

| `dtime` | span | median translation | median rotation |
| --- | --- | --- | --- |
| 1 | 38 ms | ~1.15-1.16 cm | ~0 deg |
| 5 | 191 ms | ~6.2 cm | ~0 deg |
| 10 | 382 ms | ~12.4 cm | ~0 deg |
| 20 | 763 ms | ~24.9 cm | ~0-0.07 deg |
| 40 | 1.5 s | ~49.8 cm | ~0-2.2 deg |

The robot barely moves in any of these windows — good news for alignment, since there's little ego-motion for `aligned_raw_scan()` to have to correct in the first place.
(A small number of extreme outliers, up to ~21m at `dtime=1`, are almost certainly an odometry glitch or session-adjacent artifact, not real single-frame motion — not investigated further, doesn't affect the median-based conclusion above.)

**Second sanity check: how much does a *person* actually move per frame** — matched directly against real annotated positions (`det_wp`), not assumed from a textbook walking speed.
Greedy nearest-neighbour match between two frames' person point-sets (sensor-local coordinates, 2.5 m gate against cross-person mismatches), sampled every 25th annotated frame, ~13,000-15,000 matched pairs per `dtime`:

| `dtime` | span | median displacement | mean | p90 |
| --- | --- | --- | --- | --- |
| 1 | 38 ms | 2.12 cm | 3.52 cm | 5.13 cm |
| 5 | 191 ms | 8.78 cm | 14.28 cm | 22.50 cm |
| 10 | 382 ms | 17.16 cm | 26.44 cm | 46.28 cm |
| 20 | 763 ms | 34.78 cm | 47.43 cm | 93.05 cm |
| 40 | 1.5 s | 67.14 cm | 81.56 cm | 160.70 cm |

**What it means.**
Median displacement scales almost exactly linearly at ~1.7 cm per unit of `dtime` across this whole range — a real, human-motion-dominated signal (the robot's own ego-motion above is small enough not to explain it).
This confirms the original intuition behind reconsidering the default: at `dtime=1`, consecutive frames really are close to identical for a person (~2 cm, well under a body width); by `dtime=10` a person has moved ~17 cm (a meaningful fraction of a stride); by `dtime=20`, ~35 cm (close to half a walking stride).
Not yet decided which `dtime` should be the new FROG default — that's exactly what the retrain sweep in `TODO.md` A1 (now widened to test a wider range than originally planned) is for; this entry is the measurement the decision should be checked against, not the decision itself.

---

## Fixed the Makefile's broken `python3`, corrected a false code-duplication claim (2026-09-08)

*keywords: PYTHON, VENV_PYTHON, python3, python, OS Windows_NT, make venv, make test, FollowTheDrow CMake, find_package*

**`make venv`/`make test` were silently broken on Windows** — `python3` resolves to the Microsoft Store's install stub here, which exists as a command but does nothing useful when run, so `python3 -m venv venv` never created a usable environment.
Separately, `make venv` never installed `utils/requirements.txt`, so `pytest` was never available even once that was fixed.

**What changed in the code.**
`Makefile` gained `PYTHON`/`VENV_PYTHON` variables, branched on `$(OS)`: Windows prefers bare `python` (checked with `command -v`, since `python3` there is the trap above), Linux/CI prefers `python3` (the tested path on `ubuntu-latest`, which may not alias `python`).
`venv`/`test` now call `$(VENV_PYTHON)` explicitly instead of relying on `PATH` (which assumed `venv/bin`, wrong for Windows' `venv/Scripts`); `venv` now also installs `utils/requirements.txt`.
Verified with `make -n test` (dry run): resolves to the real `/c/Python314/python`, correct `venv/Scripts/python` invocations throughout.

**What it means, and what was decided.**
`python` on this machine is 3.14, not the 3.12 the prebuilt C++ extension needs (`memory/gotchas.md`) — this fix resolves "the command doesn't exist," not "the command is the wrong version"; a full `make venv` run has not been exercised end-to-end here yet.
Tracked as a separate, still-open item (`TODO.md` A9).

**A previously-recorded finding is retracted: the C++ `AlgorithmicDetector` is not duplicated between `library/cpp_core/` and `deploy/follow_the_drow/`.**
Checked directly (not assumed, this time): `deploy/follow_the_drow/CMakeLists.txt` does `find_package(FollowTheDrow CONFIG REQUIRED)` and links every node against the compiled library; `deploy/follow_the_drow/src/algorithmic_detector.cpp` is a 54-line ROS-node wrapper around the shared `follow_the_drow::AlgorithmicDetector` class, not a second copy of its logic.
The earlier claim (written during the same-day documentation migration, from file-name similarity rather than a diff) has been corrected in `memory/dos-and-donts.md` and `memory/coding-guidelines.md`.
The real, narrower duplication that *is* there: `deploy/follow_the_drow/nodes/drow_detector.py` hardcodes its own `RESULT_CONF` vote-clustering hyperparameters independent of the training/eval pipeline's own defaults (`TODO.md` A10).

---

## Migrated to the agentic-layout-template documentation layout (2026-09-08)

*keywords: agentic-layout-template, AGENTS.md, memory/, verify_memory.py, HIGHLIGHTS.md, EVALUATION_PLAN.md*

**Adopted the `pseusys/agentic-layout-template` five-file documentation layout** (`AGENTS.md` / `README.md` / `TODO.md` / `CHANGELOG.md` / `memory/`), replacing a `docs/` directory that had grown to 827 lines in `RESEARCH.md` alone, mixing enduring architecture rationale, dated bug-fix narratives, rejected experiments, and evaluation caveats in one file.

**What changed.**
`docs/RESEARCH.md`'s Section 8 (evaluation results and their history) was split out: the dated findings below are now in this file, tuning-sweep tables moved to `memory/performance-log.md`, accuracy-number caveats to `memory/interpreting-evaluation.md`, and three tried-and-rejected experiments to `memory/rejected-ideas.md`.
`docs/RESEARCH.md` itself is kept as the citation-grade research write-up (literature review, architecture rationale, the fidelity audit against each paper) — its own audience (an eventual paper/thesis writeup) differs from `memory/`'s (an agent working in this repo), so it was not folded in wholesale.
`docs/HIGHLIGHTS.md` (a one-off summary written the previous session) and `docs/EVALUATION_PLAN.md` (a runbook for evaluation tables that are now already filled in) were retired — their content is now `memory/performance-log.md`, `README.md`'s "Where things stand", and `memory/commands.md` respectively.
The stale `compare/` directory references (two Makefile targets, several `AGENTS.md` mentions) were removed — the directory itself no longer exists in this repo.

**What it means.**
Every fact now has exactly one home: `memory/*.md` for current mechanism, this file for what happened and when, `memory/rejected-ideas.md` for settled negative results, `TODO.md` for what's still open.
Markdown structure (links resolve, both indexes match the files on disk, one-sentence-per-line) is enforced by `memory/scripts/verify_memory.py`, wired into CI via `.github/workflows/lint.yaml`.

## FROG ships real, official per-session odometry that this codebase had never downloaded (2026-09-08)

*keywords: FROG_Dataset.download, frog_<HH-MM>_odom.npz, _session_odom_filenames, session_gap_s, session splitting, integrate_trajectory, max_dt, timestamp gap*

**The whole ICP-pseudo-odometry effort (previous entry below) was working around a self-inflicted gap, not a real one.**
The FROG authors publish real per-session odometry (`frog_<HH-MM>_odom.npz`, confirmed via HTTP HEAD returning 200 for every session) that `FROG_Dataset.download()` never fetched — it only ever downloaded the `.h5` scan files — so `_load_or_fake_odom()` silently fell back to zero odometry on every run to date.
Cross-checked, not assumed: the real odometry files' own session-boundary gap (88,658s) matches the gap independently found in scan timestamps (88,655s) to within 3 seconds.

**Two more bugs found while investigating why the odometry work looked unlike anything expected.**
(1) `integrate_trajectory()` treated a whole `.h5` file's scans as one continuous ~26-40 Hz stream; real recordings have 62 timestamp gaps (61 sub-90s pauses plus one ~24.6-hour cross-session gap), and 31 of the 62 produced a small, *plausible-looking* ICP delta that slipped past the existing magnitude clamp — including the 24.6-hour boundary itself, which looked like an innocuous "0.087m, 3.74°" step.
(2) More fundamentally, `FROG_Dataset` treated one `.h5` file as one *sequence*, so `get_scan()`'s temporal window had no way to know about that same boundary — meaning even zero-odometry runs could pull raw-scan history across two unrelated recordings for frames near it.

**What changed in the code.**
`integrate_trajectory()` (`library/follow_the_drow/utils/odometry_estimation.py`) now takes real `timestamps` and forces zero motion across any gap exceeding `max_dt` (default 0.1s), before asking ICP at all.
`_load_h5()` (`library/follow_the_drow/datasets/frog_dataset.py`) splits into multiple sequences at any gap exceeding `session_gap_s=300.0`, mirroring DROW's own multi-sequence structure.
`download()` now also fetches each session's real odometry file after its `.h5`, and `_load_h5()` prefers a matching real file over the estimated/zero fallback when present, via a new `_session_odom_filenames()` filename-token parser.
Verified against the real files: `FROG_Dataset(split='train')`'s loaded x/y/theta ranges exactly match the real files' own reported ranges.
9 new tests (`tests/test_frog_session_splitting.py`, `tests/test_odometry_estimation.py`); full suite at 63 passing.

**What it means, and what was decided.**
Every real-odometry number produced before this fix (next entry) is retracted as a valid real-odometry measurement — it was trained on estimated pseudo-odometry, a single-session dataset, or both.
A from-scratch `dtime` sweep on the fully-fixed pipeline (real odometry + correct session splitting) is needed before any real-vs-zero-odometry conclusion can be trusted; not yet scheduled (`TODO.md`).

## ICP pseudo-odometry estimator built for FROG, first real-odometry `dtime` sweep run (2026-09-05 to 2026-09-06)

*keywords: odometry_estimation.py, generate_frog_odometry.py, trimmed ICP, Kabsch, dtime=1, dtime=5, real odometry*

**Built a trimmed-ICP scan-matching odometry estimator for FROG, which ships no odometry at all** (confirmed by inspecting its `.h5` files directly at the time — no pose field of any kind) — see the entry above for how that premise turned out to be wrong.
Verified with 10 synthetic-ground-truth unit tests (forward/rotation/combined/sideways/static motion) before running on real data, given this project's own prior sign-error history in exactly this kind of derivation (`memory/dos-and-donts.md`'s evidence section).

**First real-data issue found and fixed during generation**: a large open-space region (~97% of beams out of range) produced a sustained cluster of large, wrong jumps — a classic scan-matching failure mode, not a code bug.
Fixed with a per-step sanity clamp (reject an implausible-velocity delta, fall back to zero motion for that frame) plus a higher minimum-inlier count; max single-step distance dropped from 1.10m to the 0.20m clamp bound.

**First `dtime` sweep on this (later found to be superseded) real odometry**: `dtime=1` measured 78.4% wp-AUC (train and test) vs. 76.7% zero-odometry — a real gain at the time; `dtime=5` measured 79.5-79.6% vs. 80.7% zero-odometry — a real *regression* at the time, train ~ test both cases.
Both numbers are retracted as real-odometry measurements by the entry above — they used a since-fixed pipeline (single-session dataset, no timestamp-gap protection).

**What it means.**
The estimator itself (`library/follow_the_drow/utils/odometry_estimation.py`) remains useful as a fallback for any FROG session without a real odometry file, and its own unit tests are unaffected by the pipeline bugs above.

## Does temporal fusion help on FROG? Error breakdown, NMS sweep, and a wide-window regression (2026-08-24)

*keywords: error_breakdown.py, nms_sweep.py, vote_collect_radius, nms_radius, false positive, SimpleTracker, dtime=20*

**False positives, not false negatives, are the dominant error on FROG for every model tested** — FN/FP ratio never exceeds 0.8x for `spacetime_cnn`, `fullscan_tcn`, `temporal_unet`, `lfe_peaks`, and `lfe_ppn` alike, measured by re-reading the same precision/recall curve every AUC number already comes from, at each model's own best-F1 threshold (`utils/error_breakdown.py`).

**Operating-threshold selection is a free, unused win for 3 of 5 models** — moving to a ~75-80% precision point trades more false positives saved than false negatives added; past ~80% the trade reverses. `FullScanTCN` and `LFE-PPN` don't get this win from threshold selection alone.

**NMS/vote-clustering sweep found two small, real wins**: `FullScanTCN`'s `vote_collect_radius` peaks at 0.7 (+0.2pp over the 0.5 default); `LFE-PPN`'s `nms_radius` peaks at 0.9 (+0.5pp over 0.8).
Two bugs found and fixed along the way: an all-zero detection config crashed `_deep2flat()` and `error_breakdown.py`'s best-F1 finder instead of returning a well-defined empty result.

**A wide real-time window (`--time-frame 10 --dtime 20`, FullScanTCN) was retrained to test whether a clearer motion signal over ~4.5s would help, given NMS tuning alone didn't close FullScanTCN's accuracy gap to SpaceTimeCNN — it regressed to 54.8% wp-AUC (val), ~15 points below the `dtime=1` baseline.**
Root cause: FROG had no odometry at the time (see the entries above), so the wider window gave uncompensated ego-motion 36x longer to accumulate, outweighing any clearer-motion benefit.
Recorded as rejected in `memory/rejected-ideas.md`; results adopted as new defaults (`vote_collect_radius=0.7`, `nms_radius=0.9`).

## `SimpleTracker` built; a real DROW ego-motion sign bug found and fixed (2026-08-26)

*keywords: SimpleTracker, tracking.py, aligned_raw_scan, ego-motion compensation, DROW wp-AUC*

**Built `SimpleTracker`** (`library/follow_the_drow/utils/tracking.py`), a minimal SORT-style tracker (constant-velocity alpha-beta filter, Hungarian assignment against a track's *predicted* position, confirmation via `min_hits`, miss-tolerant coasting up to `max_age`) — the mechanism that replaced an earlier naive-averaging design rejected on inspection (`memory/rejected-ideas.md`).

**Fixed a rotation-sign bug in `aligned_raw_scan()`** discovered while validating the tracker's ego-motion-compensation path on DROW, which has real per-frame odometry.
Fixing it took the DROW-native `SpaceTimeCNN` baseline from 15.0% to 18.5% wp-AUC on test — confirming ego-motion compensation is worth having correctly, not just in principle.

**What it means.**
DROW is the fairer testbed for validating this tracker's core assumptions, since it has real odometry to compensate with; FROG was treated as a static-sensor case by necessity at the time (before the entries above found real FROG odometry existed).

## `dtime=5` found as a real accuracy peak on FROG, not overfitting (2026-09-02 to 2026-09-04)

*keywords: dtime sweep, dtime=0, dtime=1, dtime=5, dtime=10, SpaceTimeCNN 80.7%*

**A proper `T=5` sweep over `dtime` in {0, 1, 5, 10} on FROG (SpaceTimeCNN, full retrain per point, zero odometry, evaluated on both train and test) found a real, non-overfitting peak at `dtime=5`**: 77.1% (dtime=0) -> 76.7% (dtime=1) -> **80.7%** (dtime=5) -> 78.0% (dtime=10), train ~ test at every point.

**What it means.**
The original `dtime=1` (~125ms) window was simply too narrow to capture real human motion at FROG's 40 Hz rate, not evidence that temporal fusion is structurally useless there — reframing the original "why doesn't our temporal architecture beat single-frame LFE-Peaks" question, which motivated re-checking that comparison specifically at `dtime=5` (see the full evaluation run below).
This sweep used zero (not real or estimated) odometry throughout and predates the FROG odometry work above — see `memory/performance-log.md` for the full table and its own retraction note.

## DROW zero-shot transfer beats native training and fine-tuning, for every proposed architecture (2026-08-13 to 2026-08-14)

*keywords: DROW-native, zero-shot transfer, fine-tuned, --init-weights, overfitting, annotation coverage*

**Zero-shot transfer of FROG-trained weights onto DROW wins outright over both training natively on DROW and fine-tuning from the FROG checkpoint, for all three proposed architectures** — SpaceTimeCNN 29.0% (zero-shot) vs. 15.0% (native) vs. 17.2% (fine-tuned); FullScanTCN 22.2% vs. 16.5% vs. 21.4%; TemporalUNet 22.1% vs. 19.4% vs. 12.7%.

**Root cause, checked directly rather than assumed**: DROW's train split has ~17.8x fewer total person-annotation instances than FROG's (6.1x fewer annotated frames, 2.9x fewer people per frame).
A DROW-native `spacetime_cnn` run shows textbook overfitting (val loss rising from epoch 2 onward while train loss keeps falling) under the same regularization that works on FROG — a data-scarcity problem relative to model capacity, not a broken model (a spot-check on real positive frames found it still produces confident, roughly-correct detections).

**What changed in the code.**
Added `--init-weights` to `train.py` (loads weights only, fresh optimiser/epoch count — unlike `--resume`, which also restores the optimiser state and would carry over the wrong learning rate).

**What it means, and what was decided.**
These architectures transfer to a differently-shaped LiDAR reasonably well without retraining, and can be retrained cheaply if desired, but at DROW's data scale retraining currently doesn't pay for itself.
Shrinking the architecture for DROW-scale data specifically has not been tried (`TODO.md`).

## Full evaluation run: two real fidelity bugs found and fixed in published baselines (2026-08-13)

*keywords: window_depth, THRESH_DIST, cutout normalization, LFEPPNDetector anchor count, LFE recall metric, _prec_rec_2d, torch-directml, ABI mismatch*

**DR-SPAAM's published weights measured 25.8% wp-AUC on DROW (paper: 69.6-72%+) — a missing cutout-normalization step, not an attention-mechanism bug.**
Found by cloning the official DR-SPAAM-Detector repo and diffing its real `scans_to_cutout()` source against this repo's `cutout()` (not just comparing config files): the official `dr_spaam.yaml` specifies `window_width=1.0, window_depth=0.5` (narrower/shallower than DROW's own `1.66`/`1.0`) and **divides the centred cutout by `window_depth`** — a step this repo's `cutout()` was missing entirely.
For DROW's own weights (`window_depth=1.0`) the missing division is a no-op, which is why `DrowDetector` looked unaffected throughout.

**What changed in the code.**
`cutout()` (`library/follow_the_drow/utils/drow_utils.py`) now divides by `thresh_dist` (backward-compatible no-op at DROW's default of 1.0); `DrSpaamDetector` gained `WIN_SZ=1.0`/`THRESH_DIST=0.5` class attributes matching the official config.
DROW wp-AUC after the fix: 68.1% (matching the paper's range, now correctly ranking above `DrowDetector`'s 66.5%); FROG (zero-shot transfer): 38.6% -> 67.2%.

**`LFEPPNDetector` had two independent bugs**: it hardcoded 31 depth anchors per sector, but the bundled ONNX model's real output shape has 30 — every evaluation crashed.
And `evaluate.py`'s LFE recall computation divided by its own found true positives rather than the total ground-truth count, structurally unable to penalize a detector for missing people outright — this inflated `LFEPeaksDetector`'s FROG number to 86.2%, ~21 points above its own paper's 64.9%.
Fixed by reading the anchor count from the model's own output shape, and routing LFE through the same `_prec_rec_2d` PR-curve code every other model already uses.
Post-fix: LFE-Peaks 74.6%, LFE-PPN 67.7% (close to the paper's 64.9%/66.5%).

**Environment issues found and fixed before any of the above could even run**: a stale editable `pip install -e ../library` pointing at an unrelated old checkout; a Python 3.14/cp312 ABI mismatch against the prebuilt C++ extension (resolved with a dedicated Python 3.12 venv, `.venv312`); the bundled DROW/DR-SPAAM/LFE weights and dataset were never actually downloaded in this environment; no GPU acceleration installed despite a usable AMD GPU (`torch==2.4.1` + `torch-directml` installed and verified).

**What it means, and what was decided.**
Every number in `memory/performance-log.md` sourced to this evaluation run reflects these fixes; the run is the baseline every later finding in this file builds on.
