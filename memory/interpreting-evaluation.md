# Interpreting evaluation numbers

*keywords:* wp-AUC, AP, zero-shot transfer, DROW-native, fine-tuned, precision, recall, false positive, eval-stride, headline metric, eval-r 0.3, dead vote head, matching radius hides, plateau, converged, epoch-to-epoch noise, training loss as the instrument

By how much should a given accuracy or speed number be discounted, and why.
The standing comparison against published SOTA is in [`performance-log.md`](performance-log.md), and individual runs and sweeps in [`../CHANGELOG.md`](../CHANGELOG.md); this file is about how to read either correctly.




## Training cost is not inference cost, and a queued latency row is not a comparable one (2026-10-24)

**Two mistakes from item 43, both worth not repeating.**

**First: a per-epoch training multiple says nothing about the deployment axis.** Hungarian association cost **4.3x per
epoch** on 1 s curriculum chunks and ~1.8x on 10 s ones, and this project predicted from that, repeatedly and out loud,
that it would be expensive on the CPU. It is not: **+0.08 ms on a 19.46 ms total, inside the noise.** The reason is
structural. Training runs **26 streams** in the 1 s stage and 2 streams of 399-frame chunks in the 10 s stage, so a
per-frame assignment happens thousands of times per optimiser step; **inference is batch 1**, one assignment per frame
over a ~24x22 gated block. Anything whose cost scales with the number of concurrent streams will look far worse in
training than it is in deployment. Measure the axis you intend to claim.

**Second: the queue cannot produce comparable latency rows, and giving it latency jobs invites a wrong one.**
`queue_item16.py` and `queue_item36.py` each paired an arm with a `tracking_baselines.py latency` job, which looked like
good hygiene and produced item 16's wrong `gru_cpu` row -- 23.87 ms against a true 20.57, because its `stage2` control
phase read 10.49 ms instead of ~8.6 under session load. **A latency row is only comparable to rows measured beside it in
the same session**, and a sequential queue runs every job in its own session by construction. So:

- **Do not add latency jobs to a sweep queue.** They will each be a different session.
- **Measure latency as a matched sweep by hand, after the chain, on an idle host**: every arm plus the control, back to
  back, interleaved, two or three rounds.
- **Read the `stage2` phase first.** It is byte-identical work across stage-3 arms, so its spread *is* the noise floor.
  Item 43's three rounds agreed to 0.01 ms on the means, which is what made a 0.10 ms memory difference readable as
  noise rather than signal.

## The architecture search was not exhaustive, and the write-up must say so (owner's call, 2026-10-23)

**Every "we tested X and it did not help" in this project means one arm at one seed, judged against a 1.32 pp
three-sigma bar, and unpromising ideas were dropped early rather than pursued.** That is a deliberate policy and it is
defensible, but only if it is stated. The alternative reading -- that the adopted architecture survived an exhaustive
search -- is one this evidence cannot support, and a reviewer who assumed it would be misled.

**What the policy actually is.** Run each idea **once**. If the result sits inside variance, drop it and move on. Do not
spend a second seed on an arm with no signal, and do not take "inside variance" as equivalence -- it is *no evidence of
a difference*, which is a weaker statement and the one to write down. The policy accepts a known risk: a **hidden gem**,
an idea that would have paid off at a second seed or a tuned setting, is dropped. The owner takes that risk knowingly
rather than by omission.

**So the honest form of the claim is:** *of the architectural variations we tried, none beat the adopted choice by more
than noise; we did not try every variation, and the ones we rejected we rejected on one measurement each.* Not: *this is
the best architecture.*

**Where this bites hardest.** Items 15, 17, 18, 19, 20, 24, 25, 34 and A58/A60 are all single-arm negatives, and items
20 and S4 were closed **without any run at all**, on the grounds that the parameter's own sweep was flat. The three
findings that do carry weight have more behind them: the memory's value over its candidates (three seeds), the removal
of both attention mechanisms (E13, three seeds), and per-object addressability (item 36, a 3.44 pp gap against a 1.32 pp
bar). Quote those as results and the rest as screening.

**This paragraph belongs in the paper's limitations**, not only here (`docs/PAPER.md`).

## A CPU millisecond moves between sessions, so only arms measured back to back can be compared (2026-10-20)

**Found while answering item 16's second half**, and it changes how every future CPU row has to be taken. minGRU's memory
phase came out 0.18 ms above the adopted arm's -- until the one phase that *cannot* differ was checked.

**Stage 2 is a free control in every stage-3 CPU row.** Every stage-3 arm loads the *same* averaged stage-2 checkpoint, so
its `stage2` phase is byte-identical work. Any difference between two rows' `stage2` is therefore pure measurement noise,
and it bounds what a difference in the other phases can mean. Three readings per arm, 2,000 consecutive scans each:

| arm | phase | r1 | r2 | r3 | spread |
| --- | --- | ---: | ---: | ---: | ---: |
| `selective` | stage 2 | 8.47 | 8.81 | 8.78 | 0.34 |
| `selective` | memory | 9.12 | 9.36 | 9.36 | 0.24 |
| `selective` | **total** | **19.15** | **19.75** | **19.77** | **0.62** |
| `mingru` | stage 2 | 8.90 | 8.77 | 8.89 | 0.13 |
| `mingru` | memory | 9.30 | 9.27 | 9.32 | 0.05 |
| `mingru` | **total** | **19.76** | **19.66** | **19.81** | **0.15** |

**The offset is between sessions, not between arms.** `selective`'s r1 is the 2026-10-19 in-chain reading, and it agrees
with item 30's independent 2026-10-11 row to 0.03 ms (19.15 both times). Tonight's session reads **~0.6 ms slower for both
arms alike**, about 3%. So a row is reproducible within a session and carries a few percent of drift across sessions.

**Judged inside one session, which is the only matched comparison** (r2 and r3 of each arm, interleaved):

| phase | `selective` | `mingru` | difference |
| --- | ---: | ---: | ---: |
| stage 2 *(the control -- must be 0)* | 8.79 | 8.83 | **+0.04** |
| decode | 1.56 | 1.55 | -0.01 |
| memory | 9.36 | 9.29 | **-0.06** |
| **total** | **19.76** | **19.73** | **-0.02** |

So **minGRU's CPU cost is indistinguishable from the adopted recurrence's**: its 0.06 ms edge on the memory is barely above
the 0.04 ms the control says is noise, and the totals differ by less than the control does.

**The rules this leaves:**

- **Never compare a CPU row against one from another session.** Re-measure the control arm alongside any new arm; it costs
  about 45 seconds per reading and the protocol is identical, so there is no excuse for the cross-session comparison.
- **Read the `stage2` phase first.** If two rows disagree there, they are not comparable anywhere, however interesting the
  `memory` column looks.
- **A single reading is not a measurement.** Within a session the spread is 0.15 ms; the first cross-session comparison
  attempted here would have reported a 0.18 ms "cost" that does not exist.
- The raw readings are in `results/item16_cpu_repeats.json`.

## A GPU millisecond is not a CPU millisecond, and this project is judged on the CPU (2026-10-19)

**The owner's framing, and it changes which number is the headline.** DROW and LFE targeted cheap, CPU-only robots; nothing
about this project's deployment says otherwise (`memory/deployment.md` -- the robot runs a Docker image on its own
hardware). So **CPU `ms/frame` is the deployment axis, not a secondary one**, and an architecture that wins on AP while
costing more on a CPU has not won.

**The trap that makes this easy to get wrong.** Every `step3a_results.json` carries `memory_ms_per_frame`, which looks like
the answer and is not: `memory_ms_per_frame` times `ObjectMemory.step` on **whatever device trained the run**, which on
this host is the RX 9060 XT. Quoting it as a cost figure silently reports a GPU number. The frozen model's own two
measurements differ by more than a factor of two for exactly this reason -- **4.65 ms on the GPU against 9.14 ms on four
CPU threads** -- and the memory is the *larger* half of the CPU total while being the smaller half of the GPU's.

**The protocol to use instead** is item 29's, which every row of the speed table shares: `tracking_baselines.py latency
--device cpu --threads 4`, 2,000 consecutive FROG test scans, batch 1, median and p99 against the real 25 ms frame period.
Item 30 also settled the thread count that matters: scaling **saturates at two**, so "two CPU cores" is the deployment
claim, and one core misses the budget at 151%.

## Before anything else: every pre-2026-09-10 FROG number is retracted

Four measurement bugs, all fixed (`TODO.md` A11-A19, `CHANGELOG.md`), none of them re-measured yet.
Read `performance-log.md`'s status table before quoting any row — it says precisely which rows the fixes invalidated and which they did not.
The two that generalize past this incident, and are worth carrying as habits:

**A number is only comparable to another number measured the same way.**
The `dtime=10` "regression" decomposed exactly into four protocol differences — evaluation `dtime`, which epoch's weights, which split, which batch size — with nothing left over for the variable actually under study.
Before comparing two runs, check all four.

**An evaluation implementation detail can move the number more than the model does.**
See the batch-size section below.

## `wp-AUC` is this project's headline metric, and is the same thing as the papers' "AP"

Person-class AUC at a 0.5 m matching radius, computed by `_prec_rec_2d`/`_safe_auc` — a direct port of the original DROW-v2 evaluation code, and verified by direct source comparison to compute the identical trapezoidal-AUC-over-a-Hungarian-matched-PR-curve formula the official DR-SPAAM repo calls "AP".
A gap between this project's number and a paper's own number is a real accuracy difference, not a metric-definition artifact.
Never substitute the agnostic (any-class) AUC column for it — see `AGENTS.md` rule 2b for the incident that makes this an explicit rule rather than a preference.

## A cross-dataset number is not one measurement, it's three, and they disagree

Every architecture in `detector-architectures.md` has been evaluated on DROW in up to three different regimes, and they are not interchangeable:

- **Zero-shot transfer** — weights trained on FROG (or, for `drow`/`drspaam`, published weights trained on DROW), evaluated on DROW without any DROW-specific retraining.
- **DROW-native** — trained from scratch on DROW only.
- **Fine-tuned** — DROW-native training initialized from the FROG checkpoint's weights at a lower learning rate (`--init-weights`, not `--resume` — the latter also restores the wrong optimiser state).

The counterintuitive, load-bearing finding: **zero-shot transfer wins outright over both other regimes, for every one of the three proposed architectures.**
This is a genuine result, not a bug — DROW's much smaller effective training set (next section) means "more DROW-specific training" does not currently mean "better on DROW."
Quoting a DROW number without saying which of the three regimes produced it is close to meaningless; the spread between them is 2x or more for every architecture tested.

## DROW's numbers are structurally handicapped by annotation coverage, not by being a harder dataset

DROW's train split has roughly **17.8x fewer total person-annotation instances** than FROG's: 6.1x fewer annotated frames (17,665 vs. 108,356) and 2.9x fewer people per annotated frame (1.35 vs. 3.93).
A DROW-native `spacetime_cnn` run shows textbook overfitting (val loss rising from epoch 2 onward while train loss keeps falling) on the same regularization that works fine on FROG — this is a data-scarcity artifact relative to model capacity, not evidence that the architecture transfers badly to DROW's sensor geometry.
The architectures were sized for FROG's data volume; nobody has yet tried shrinking one for DROW's scale specifically (`TODO.md`).

## LFE numbers on DROW/JRDB are an approximate cross-dataset transfer, not a faithful replication

`LFEPeaksDetector`/`LFEPPNDetector` were trained on FROG's 720-beam, 0.25 deg/beam scans.
Running them on DROW (450 beams, 0.5 deg/beam) or JRDB (541 beams) requires zero-padding shorter scans to 720 — this project's own extrapolation, never described in the LFE paper, and one that does not correct for the different angular resolution.
Treat any LFE DROW/JRDB number as approximate, degraded-by-construction, not comparable one-for-one to a same-dataset number.

## A tracker must be fed every frame, not the frames you happen to score

`SimpleTracker` assumes `dt = 1.0` — one `step()` per sensor frame. Feeding it only the *scored* frames breaks it in two ways that both look like "tracking does not help":

- under `--eval-stride N` its time step silently becomes N frames long, so its velocity estimates and its `match_radius` gate are N times too tight;
- on the `transferred` split the scored frames are the person-free ones, **interleaved in the raw recording with populated frames the split drops** — so consecutive scored frames can be minutes apart, and joining them into a track is meaningless.

The LFE evaluation paths therefore walk the raw sequence at full rate whenever a tracker is active, and score only the subset (`_lfe_frames`). Without a tracker the walk *is* the scored set, so no number measured before this existed moved.

**Pass real ego-motion.** FROG's odometry is not zero — 1.88 cm and up to 0.4° per frame on the test recording. The belief that it was zero dates from `_load_or_fake_odom()` substituting zeros on failure (`TODO.md` A14) and survived in two code comments and a docstring; 0.4°/frame over a track's ~6-frame lifetime is 2.4°, which at 8 m is 33 cm — most of the tracker's 0.5 m gate.

## A FROG number means nothing without its association distance

FROG's Table 4 reports **nine** figures per model: mAP / mPeakF1 / mEER averaged over `d = [0.3 : 0.05 : 0.5]` m, then AP / PeakF1 / EER at `d = 0.5 m`, then the same at `d = 0.3 m`.
**The leading column is the averaged mAP, not AP at any single distance**, and every FROG figure this project quoted before 2026-09-10 was taken from it while every number we produce is AP at `--eval-r 0.5`.

The offset is not a constant you can subtract: +0.3pp for DR-SPAAM, **+2.7pp for LFE-PPN**, because how fast a detector's AP falls as the gate tightens is itself a property of the detector.
`performance-log.md` now carries the `d = 0.5 m` column. **Always say which `d`.**

## FROG test carries 2,254 duplicate labels, and they reward double reporting

1.47% of the test file's circles repeat another circle of the same frame bit for bit (`data-model.md`, found 2026-10-05).
Matching is one to one, so a copy can only be found by a **second** detection beside the original, which is a false positive everywhere else.
Two consequences when reading a FROG test number:

- **Every AP is 0.7-1.05 pp below its de-duplicated value**, and the method that double-reports least loses most: candidates 77.66 -> 78.70, the tuned ByteTrack + CBMOT 80.96 -> 81.90, E13 83.86 -> 84.60.
  A gap between a double-reporting method and a clean one is therefore inflated by up to ~0.3 pp (E13 over E14: +2.90 official, +2.71 de-duplicated, every interval still above zero).
- **A per-person statistic that counts the copies** credits double reporting as recall.
  The A56 "unlinked annotations" the memory "found" 39% of were exactly these copies.

The official labels stay the reported protocol, because every published FROG and LFE number was scored on them.
Quote the de-duplicated figure only as a sensitivity, and only beside the official one.

## Calibrate anything positional at `--eval-r 0.3`, not 0.5

A 0.5 m association gate is wider than most localisation errors worth finding, so it will happily accept a decoder that is systematically off by 10-20 cm.
This is not hypothetical: the LFE-PPN anchor-decode calibration reported its angular convention as "weakly identified" because the candidates spanned 66.6-68.5% at `d = 0.5`. At `d = 0.3` the same candidates span **28.7-56.4%**, and the chosen one undershoots the published figure by 9.4pp.

And do **not** try to recover a decode convention by fitting localisation residuals against the raw offset channel. It was tried, it looked conclusive, and it is confounded: ground-truth association uses a *metric* radius, so angular residuals are capped at ~`r/d` by construction and any convention produces a `scale x depth = const` signature.

**A dead regression head is the limiting case, and Li2Former's own released network is the worked example -- but the lesson is not the one a per-beam estimate predicts** (2026-11-01, item 31). Their `reg_head` ends in `nn.ReLU()`, which their own left-right-flip augmentation drives into its dead zone within two epochs: `pred_reg` becomes identically 0, the gradient through the clamp is exactly 0, and `loss_reg` goes bit-stable (`utils/diagnose_vote_head.py` proves it; `memory/detector-architectures.md` has the mechanism). Every detection is then placed at its own beam's measured point.

Inverting the frozen loss through their AFM puts the typical vote magnitude at only **~0.19 m**, well inside a 0.5 m gate, which **predicts that the loss is nearly free at `d = 0.5`. That prediction is wrong, and the correction is the point:** the votes are consumed by a *clustering* step before anything is matched, and zero votes never converge -- the per-person votes stay smeared across the surface the beams hit, so one person yields several detections instead of one peak and the false positives take the precision. Measured on DROW over nine of twelve epochs: **val** wp-AUC converges near **29.7%** and **train** near **38.5%**, against a 61.9% floor for DROW3 and a published 76.4%. (An earlier reading of this same run put the plateau at 25.7% / 34.1% from two epochs and was withdrawn; the four-point and training-loss standards that replaced it are in the run-to-run spread section above.)

So: a per-beam distance argument prices the matching step only, and underestimates a broken vote head badly. What a 0.5 m gate *does* hide is a decoder that is systematically off, which is the LFE-PPN case above. For either, **check the head directly rather than inferring it from a plausible loss curve or a plausible headline number** -- this curve rose for nine epochs and the total loss fell throughout, with three quarters of that loss a frozen constant the whole time -- and report `d = 0.3` alongside `d = 0.5` whenever the spatial head has not been verified.

## We cite published accuracy numbers; we do not re-derive them

Owner's call, 2026-09-10, now the standing policy (`performance-log.md` holds the notation):
**a table cell takes the original authors' published number wherever they published one.**
This project evaluates a third-party model only to fill a genuine gap — a model/dataset pair nobody published — or to measure `ms/frame`, which is comparable only when every row came off one machine.

Re-deriving a published accuracy figure through this pipeline costs a real run and adds a reproduction risk, for a number that was already authoritative.
This session's audit is the case in point: four measurement bugs silently reshaped every locally-derived FROG figure, while the published ones were never in doubt.

**Standing corollary — if our reproduction of a published number overshoots it, the harness is wrong.**
An undershoot has innocent explanations: a different frame sample, a wrapper detail, a merge radius we had to choose ourselves.
An overshoot on identical published weights does not.
The live example: LFE-Peaks measured **74.6%** here against its authors' own **64.9%** on FROG — a +9.7pp overshoot that was the visible symptom of the train/test contamination for weeks before anyone read it that way.
Check this the moment a reproduction lands, not later.

## Published-weights vs. this project's own training

Every detector falls into exactly one of: **replicated with official weights** (`drow`, `drspaam`, `lfe_peaks`, `lfe_ppn` — genuine published-model inference, no training performed here), **reimplemented with no official weights** (`li2former` — trained from scratch from the paper's description alone, since training never completed, no locally-measured number exists at all), or **novel** (`spacetime_cnn`, `fullscan_tcn`, `temporal_unet` — no prior paper describes them, so no official weights could exist).
A "replicated" score is a reproduction of a published result; a "novel" score is this project's own claim, with nothing to check it against except internal consistency (train ~ test, ablation sanity).
An official-weights evaluation of DR-SPAAM (T=1, T=5) through this project's own pipeline was attempted and abandoned (`TODO.md` A2) — the paper's own numbers are cited instead, deliberately, since the weights-compatibility check (zero missing/unexpected keys) already passed independently.

**A released repository's deployment config is not the configuration its paper's number came from, and transcribing it faithfully is not the same as reproducing the model** (2026-11-01, item 31; cost: ~10 h of GPU on a meaningless arm, plus a day of reasoning about a 30% result as though it described the published model). Li2Former's only published config is `cfgs/ros_li2former.yaml`, a **ROS robot deployment** file: it sets `MODE: TEMPORAL`, which runs their 1-layer temporal encoder alone, never calls their 3-layer spatial encoder, and so has **no cross-beam path** — while the paper's entire contribution is **Omni-Dimension Aggregation**, i.e. both dimensions at once (`SERIAL`/`PARALLEL` in their code). Three quarters of the checkpoint's 29.47M parameters never received a gradient. The transcription was exact and verified against the upstream file; the mistake was upstream of that, in believing the file answered the question. **So before training someone else's network to compare against their number, check that the configuration is the one the number was measured on** — a deployment config is chosen for a robot's CPU budget, an ablation config for a table row, and neither has to be the headline model. Corroborating signal worth trusting next time: the configured model's active parameter count collapsed to ~4M, the size of the hand-written reimplementation it was brought in to replace, and **that coincidence was visible before the run started**. A paper whose contribution is an aggregation *mechanism* and a config that disables half the mechanism is a contradiction that should stop a launch.

The same file then supplied a second one, which is why the lesson is about the *file* and not about `MODE`: `NUM_CTS: 90` is their robot's cutout count, and it also caps their position encoding, so **the deployment config encoded the deployment's sensor as well as its compute budget**. DROW has 450 beams; the corrected mode attends over beams rather than scans, and the first batch died on the mismatch. **When a config comes from a deployment, assume every value in it describes that robot — its sensor geometry, its beam count, its latency budget — and check each against your own dataset, not just the ones the paper happens to discuss.** Three launches were spent on the second value after the first had already taught the lesson.

## Epoch counts across papers are not comparable without converting to minibatches

FROG's own authors trained DROW3/DR-SPAAM for exactly 5 epochs (batch 8) on FROG's full 108,356-frame train split — about 67,700 minibatches — a number chosen deliberately "to match the magnitude of minibatches processed when training on the original DROW dataset," not because 5 epochs is enough in general.
LFE-Peaks/LFE-PPN targeted 100/150 epochs (patience 20) at batch 32/4 instead.
Comparing raw epoch counts across these (or against this project's own runs) is meaningless without converting to `epochs x frames_seen / batch_size`; a screening-scale run here (8 epochs, 20% subsample, batch 4, ~43,300 minibatches) lands in the same order of magnitude as DROW3/DR-SPAAM's own training, while this project's full runs (30 epochs, full data, batch 4, up to ~812,700 minibatches) run roughly 12x longer than that.
A model still improving (val wp-AUC rising, patience not triggered) at a cheap screening run's cutoff is undersold by that run's own number — a real risk specific to short/subsampled runs, not evidence the configuration is actually worse (`CHANGELOG.md`'s `dtime` screening table has a live example).

## What temporal information is worth on FROG, according to the literature

**+1.9pp.** DR-SPAAM's own published FROG numbers at d = 0.5 m are `T=1` **73.7%** and `T=5` **75.6%** (73.3% / 75.3% in the paper's mAP column, `performance-log.md`) — the best temporal model in the literature gains about two points from five frames on this benchmark.
LFE-Peaks reaches 65.6% from a **single frame** with ~65K parameters at 1.76 ms; neither LFE model carries any memory between scans.

**Those five frames span about 0.15 s.** The FROG paper gives only "a temporal window size of 5 scans" and names no stride. It trained DR-SPAAM with the official repository (`VisualComputingInstitute/2D_lidar_person_detection`), whose DR-SPAAM config sets `scan_stride: 1` and whose loader takes scans `scan_idx - i * scan_stride`, i.e. consecutive; FROG's own export (`robotics-upo/2DLaserPeopleBenchmark`, `convert_frog_bags.py`) keeps every scan. Five consecutive scans at FROG's measured 26.2 Hz (`dataset-properties.md`) are 4/26.2 = 0.15 s apart end to end -- inferred from the code, since FROG's own DR-SPAAM config is not published. Even run as a stream, DR-SPAAM's template update (α = 0.5) halves an older scan's weight per scan. So the published temporal gain comes from the span `docs/PROPOSAL.md` gave stage 2's fine temporal convolution (~0.08 s), well inside its calibration horizon (the coarse convolution's ~1.3 s); both were removed after failing their ablations (A50, A60), and the short and long horizons, now the object memory's, have no published counterpart on this sensor.

Keep that ceiling in view before reading any temporal result here as a large effect.
It also frames the risk this project actually runs: every one of its three novel architectures exists *for* the temporal channel, so if the trained model turns out to be a static shape detector, LFE already does that job far more efficiently and the architectures have no rationale (`TODO.md` A21).

The corollary is a reframing worth holding onto: on a benchmark where every frame contains 3-4 people, shape alone gets most of the way, so temporal information has little room to show up as raw AP.
Where it *should* show up is in rejecting **static** false positives — a chair leg and a standing person are near-identical in a 2D knee-height slice, and motion is the only cue separating them.
That is what `TODO.md` A20's person-free false-positive rate measures, and nobody has published it.

## The informational-capacity proxy's peak is a prior for where to train, not a prediction of the trained result

`informational-capacity-proxy.md`'s AUC/MI grid found a clean peak around a ~1.2-1.8s total window span, stable across `T` from 2 to 25 — a real, useful finding about the *raw input*, but it is not the same claim as "this is where the trained `wp-AUC` will peak too."
Expect a real, positive correlation, not an exact match, for reasons that cut in both directions:

**Why a correlation is expected at all.** The proxy is computed on the exact same `aligned_raw_scan()` window a real detector trains on — if the raw per-beam signal has low separability at some `(T, dtime)`, the network has strictly less to work with there, and no amount of training invents information that was never in the input.
This is the same logic behind the "zero-cost NAS proxy" literature (Mellor, Turner, Storkey & Crowley, "Neural Architecture Search without Training", *ICML* 2021; Abdelfattah, Mehrotra, Dudziak & Lane, "Zero-Cost Proxies for Lightweight NAS", *ICLR* 2021) — those proxies are computed from network gradients/Jacobians at initialization rather than raw input statistics, a different mechanism from this one, but published for the same reason: to narrow a search space before paying for full training, and reported there with a *moderate*, not perfect, rank correlation to final trained performance (typically informative enough to discard clearly-bad candidates, not precise enough to pick the exact winner without confirming it).

**Why it won't be exact.** This proxy is univariate — one beam's own feature, in isolation.
A real detector combines many beams nonlinearly (a person spans several beams; correlated motion across adjacent beams is a much stronger cue than any one beam's variance) and applies a fixed architecture whose own temporal kernels (`SpaceTimeCNNDetector`'s 3/5/7-tap time-axis convolutions, sized around `T=5`, `detector-architectures.md`) may not extract a given `(T, dtime)`'s available information equally well regardless of how much is nominally there.
Training-dynamics effects already found in this project (different `dtime` values converging at different *rates*, not just different ceilings — `CHANGELOG.md`'s dtime screening table) are a separate axis entirely, orthogonal to how much raw information exists.

**The honest position, and the concrete check that resolves it**: treat the proxy's peak as a *prior* — worth concentrating training compute near it, worth being suspicious of a config it says is well past its peak — but not as a substitute for the phase-2 training runs themselves.
Once several real `(T, dtime)` points have both a proxy value and a trained `wp-AUC`, compute the actual (Spearman) rank correlation between them directly rather than assume it — this project's own convention throughout (`dos-and-donts.md`'s evidence section) — and record whatever that correlation turns out to be here, since it determines how much weight this proxy should carry in future sweep decisions.

## False positives, not false negatives, are the dominant error on FROG — for every model tested

At each model's own best-F1 operating point, the FN/FP ratio never exceeds 0.8x, for `spacetime_cnn`, `fullscan_tcn`, `temporal_unet`, `lfe_peaks`, and `lfe_ppn` alike.
A proposal to "improve recall" is targeting the wrong side of the error for this dataset; a proposal targeting false positives (threshold selection, NMS tuning, a confirmed-track requirement) is aimed at the actual dominant error.
This also means the best-F1 point on a model's PR curve is not necessarily the right *deployment* threshold: for three of the five models above, moving to a ~75-80% precision operating point trades favorably (saves more false positives than it costs in false negatives); past ~80% the trade reverses.

## wp-AUC is *undefined* on a person-free split, not merely low

With no true positives, precision is 0 at every threshold and recall has no denominator.
`evaluate.py` therefore switches metric with the data rather than with a flag: a split carrying no annotated people is scored by `false_positive_rate()` — FP/frame, FP/second, and the share of frames carrying at least one, at a sweep of operating points — because there every detection *is* a false positive by construction.
The sweep is deliberate: a single operating point would be a choice presented as a measurement.

`duration_s` is `n_frames * frame_period_s`, never the span from the first timestamp to the last: `transferred`'s person-free frames are interleaved with populated ones the split drops, so a span would bill the detector for time it was never asked about and understate its rate.

## A number in a draft is not evidence that it was measured

**Found 2026-10-10.** `docs/PAPER.md` carried LFE-Peaks at **59.8%** and LFE-PPN at **63.5%** on the `balanced` partition, inside a sentence arguing that "the target any temporal architecture of ours must beat is fixed".
No `CHANGELOG.md` entry held those numbers, no file under `memory/` held them, `TODO.md` listed the measurement as still pending — and the owner confirmed that **nothing has ever been run on `balanced`**.
They were deleted rather than re-derived.

**What this changes about reading the documents here.** `verify_memory.py` checks links, indexes and markdown style; **nothing checks that a number in `docs/` corresponds to a measurement**, and the convention that every figure is traceable to a `CHANGELOG.md` entry is enforced by review alone.
So when a number appears only in `docs/` — in `PAPER.md`, `PROPOSAL.md` or `RESEARCH.md` — and nowhere in `CHANGELOG.md` or `memory/`, treat it as unverified until the entry is found, especially where it supports a claim rather than merely illustrating one.
The three partitions make this easy to get wrong in one direction in particular: `official` and `transferred` share their train and val halves by construction, so a result on one often *is* a result on the other, while `balanced` shares nothing and needs its own training run.

## A frame rate is the elapsed time over the frames, never the median interval

**Settled 2026-10-10, and it moved every per-second figure in the repo by 1.53x.**
FROG's recorder stamps scans in bunches — roughly 38 ms, 38 ms, then under 0.1 ms, with **20.2% of intervals under 1 ms** — so the median interval reads **26.2 Hz** where the sensor delivers a scan every 25 ms.
`evaluate.py` took that median, and so every `FP/s` in this repository was 1.53x low, the "38.2 ms real-time budget" the cost claims were set against was really **25 ms**, and every duration converted from frames was 1.53x too long.

One definition now serves every script: `takheliped.utils.drow_utils.frame_period_s(times)`, the mean of the intervals inside each recording, dropping any longer than 0.5 s as a recording pause.
Measured on the real files: FROG **40.01 Hz** (`official` test) and **40.06 Hz** (`transferred`), against the sensor's 25 ms nameplate; DROW **12.70 Hz**, against the paper's 12.5 Hz and the median's 10.00 Hz.

**What this changes when reading an older number.** Any figure per *second*, any duration in seconds converted from a frame count, and any "real-time budget" comparison written before 2026-10-10 is on the 26.2 Hz clock.
Anything per *frame* — AP, FP/frame, recall, ms/frame — is unaffected, because no frame count changed.
`DT_NATIVE_S`, `persistence_map`'s spans, `motion_analysis`'s horizons and `slot_budget`'s first-frame step all read the real rate now; the analyses whose seconds-axes they set are re-run from it.

**A detector that carries memory needs one more column there, and TAKHeLiPeD is the first one measured on this split.**
The three raw recordings interleave populated and person-free frames, and `utils/empty_scene.py` replays the memory over all of them, as a robot would see them — so a detection on a person-free frame can be a *stale memory of a real person who has just left the field of view*, which LFE-Peaks and LFE-PPN cannot produce and are not charged for.
The slots fade over 10 s, so the comparable number is the one restricted to frames where nobody has been in view for longer than that, and `empty_scene.py` reports both.
Quoting the unrestricted count beside a static detector's overstates the memory's phantoms; quoting only the restricted one hides reports the robot would really make.

## `LFEPPNDetector` was never architecturally slow — it was decoding a bug

Recorded here because the opposite claim stood in this file until 2026-09-10, and was quoted.
Its ~250-300 ms/frame was blamed on a nested pure-Python decode loop. That was wrong.
The decoder applied a **second sigmoid** to an objectness channel `lfe_ppn.onnx` had already sigmoided, squashing every score into [0.5, 0.731]; `score_thresh` went inert, and all 3,600 anchors entered an O(n²) greedy merge every frame, yielding 182 "detections" per frame on scenes holding ~3 people.
Fixed at the source, it runs at **1.56 ms/frame** — within 0.1 ms of the paper's own *GPU* figure, on CPU — with the same Python loop untouched.

**The general lesson, which is why this entry stays**: a performance anomaly of two orders of magnitude was explained by the first plausible implementation story and the story was written down as fact. The measurement that would have refuted it — how many detections per frame? — cost nothing and was never taken.

## Run-to-run spread is ~0.9 points of test AP — two single runs must differ by ~2.5 points to mean anything

Measured 2026-09-17 by training two step-2 arms at three seeds each, all else equal:

| arm | seed 0 | seed 1 | seed 2 | mean | sd |
| --- | --- | --- | --- | --- | --- |
| no-fine (1 fine tap) | 78.03% | 77.89% | 76.60% | 77.51% | 0.79 |
| 5 fine taps | 78.56% | 77.32% | 76.63% | 77.50% | 0.98 |

With a per-run sd of ~0.9 points, the difference between two single runs has an sd of ~1.3, so **anything under ~2.5 points between two single runs is noise**. The 0.30-point spread used to screen every architecture arm on 2026-09-16 came from one pair of runs that also differed in code (the plateau scheduler), so it was never a variance estimate. It understated the spread about threefold.

That retracts every single-seed ranking of 2026-09-16 smaller than ~2.5 points: dropout over the control (+0.70), each temporal ablation, 5 taps over 3 (+1.94), 7 taps under 5 (−1.42), and the mechanisms argued from them. **Still standing:** effects far outside the spread (the median denoiser, −9.16), and findings about one fixed model, which carry no seed variance (stage-3 threshold calibration, the memory diagnosis, the `--clip-frames` bug).

The rule going forward: screen at several seeds from the start, or let single-run screening rank only effects above ~2.5 points.

**Seeds do not pair runs; compare means, not per-seed differences.** Measured 2026-09-17. The default architecture was retrained at seeds 0-2 on current code and scored 78.19 / 76.64 / 75.13. The same seeds' earlier runs, rescored with the same decoding, gave 78.40 / 78.28 / 77.09. The code differed only in how targets anchor beyond 10 m. A same-seed gap of up to 1.96 points means a seed does not reproduce a run here: GPU kernels are nondeterministic, and seeding fixes data order but not the arithmetic. So a paired-by-seed t-test is not valid on this machine. Two paired "results" are retracted: static vs the default architecture ("t = 3.4") and limit-beam exclusion ("−0.93, lower on every seed"). The latter reversed to +0.34 against fresh controls. Five three-seed arms now pool to a per-run sd of **~0.95 points** (0.46-1.53 per arm), so the ~2.5-point single-run rule stands. At three seeds a mean difference needs roughly 1.5-2 points to count.

**Stage 3's screening bars are per backbone, and one of them no longer exists.** B0's three seeds on the no-temporal stage 2 gave bars of 1.45 (AP), 1.29 (AP at 0.3 m), 0.86 (recall at matched FP) and 8.19 pp (coasting precision). On A59's adopted recipe the same three-seed reading (A61's E13 runs) gives **1.32, 1.62 and 2.37**, and **coasting precision has no bar at all**: the memory emits 0.0012 / 0.0000 / 0.0020 coasting reports per scored frame against B0's 0.176, so one seed's precision is undefined. Recall at matched false positives is the metric that loosened most, from 0.86 to 2.37, because stage 2 now reaches 95.47% there itself. Judge an arm against its own backbone's bars, and say which one a coasting number came from (`slot-design-evidence.md`).

**Corrected 2026-09-29: the stated reason was wrong, and the rule survives for a different one.** That 2026-09-17 measurement compared runs whose *code differed*, so it could never separate "a seed does not reproduce a run" from "the code change moved the result". A clean test now exists. `a51_b0f_s1` retrained stage 3 from scratch, same seed, same flags, same code, a different output directory and eight days later than `a51_b0_short_s1` -- and reproduced it **exactly**: all **46 of 46** evaluations agree on validation AP to six decimals, and every test metric matches (AP 83.17, AP@0.3 81.52, recall at matched FP 92.53, coasting precision 22.31). Only `ms/frame` differs, as a wall-clock measurement must. So the kernels this project uses **are** deterministic, and a seed **does** reproduce a stage-3 run here.
  What actually breaks reproduction is **`--resume`**. `a51_b0f_s0` was interrupted by a driver bugcheck and resumed; it tracks `a51_b0_short_s0` for **36 of 42** evaluations and diverges at exactly the first epoch after the resume point (10 s stage, epoch 4: 0.7885 against 0.7854), ending 0.41 pp higher in AP and 5.3 pp lower in coasting precision. A resumed run is therefore a **different draw**, not the same run continued -- fine as one sample of the arm, never a reproduction of it.
  **The practical rule does not change: still compare means, not per-seed differences.** An ablation changes the architecture, so it consumes the same seed's random stream differently and its "same seed" shares nothing with the control's beyond the number. Pairing is invalid because the runs are not paired, not because the arithmetic is noisy. What *is* now legitimate is re-running an identical configuration to verify a number, which is worth doing whenever a result looks too good.

**Within a single run, the same caution applies to adjacent epochs -- and the series to read convergence off is the training loss, not the val AUC** (2026-11-01, item 31, cost: one wrong reading reported to the owner and withdrawn two hours later). Val wp-AUC swung +-5 pp between neighbouring epochs on the Li2Former DROW leg: 29.2 -> 28.9 -> **24.1** -> 29.7%, where epoch 8's dip sits between two 29s and epochs 4-5 produced an identical two-point dip that was wrongly reported as a ceiling. **Two points cannot establish a plateau on a noisy metric.** The training loss can, because it is monotone and nearly noiseless -- its per-epoch change on the same run went -0.0072 (epoch 2) -> -0.0012 -> -0.0008 -> -0.0006 -> **-0.0002** (epoch 9), which says "optimisation has stopped" without any smoothing or judgement. So: **call a plateau from the loss curve, or from four AUC points, or from the `.best.pth` write times** (the cheap check that was available and skipped), never from two. The same series separates the two failure modes: a *converged* loss with both AUC series flat and a steady gap is a bad optimum, while the data-scarcity overfit this repo has seen is a *still-falling* train loss with train AUC climbing away from val (`spacetime_cnn` on FROG, 83.8% / 63.6%, gap +20.2 widening to +32.5).


## FROG `official`'s validation split does not track its test split — do not read a val difference as a test result

Three independent measurements on 2026-09-16, all in the same direction, two of them pointing the *wrong* way:

| change | val AP | test AP |
| --- | --- | --- |
| dropout 0.1 against dropout 0.0 | 73.88% -> 73.93% (+0.05) | 76.62% -> 77.32% (**+0.70**) |
| removing the fine temporal convolution | 73.88% -> 73.76% (-0.12) on the sample, 73.79% -> 73.88% (+0.09) on every val frame | 76.62% -> **78.03%** (**+1.41**) |
| the dropout run's own checkpoint choice | picked epoch 0.75 | epoch 1.00 scores 0.84 higher |

Val also sits about 3 points *below* test for identical weights (73.88% against 76.62%), and swings more than 1.5 points between adjacent evaluations of one run, so a val difference under ~1.5 points carries no information at all.

**Scoring every val frame rather than the 2,000-frame sample narrows the charge** (2026-09-16, `duration_curve.py --split val`, five arms at their selected weights): control 73.79%, no-fine 73.88%, no-coarse 72.95%, static 72.78%, dropout 73.89%, against test APs of 76.62 / 78.03 / 77.01 / 77.22 / 77.32. The sample was accurate to about 0.12 points, so its ordering of control against no-fine flipped only because those two are **tied on val either way** — val cannot resolve the 1.4-point difference test sees between them. Full val does agree with test that no-fine and dropout are the best two. But it disagrees outright elsewhere: **static is last on val and third of five on test, control is third on val and last on test**. So the rule is not "val is reversed" — it is that val cannot resolve differences of this size and is actively misleading on some arms.

Two consequences. **Screen and rank on test AP of the val-selected checkpoint, never on val AP itself** — and say which checkpoint a number came from. **Never select on test**, which is why the mis-selection above is recorded rather than corrected: 77.32% stays the reported dropout figure even though 78.16% was available at another epoch.

The cause is most likely the split itself: `official`'s val is the populated-only frames of three recordings, sampled 2,000 frames at a time, while its test is one recording scored on every annotated frame. `balanced`'s val is 121,329 frames including 19.8% empty and is the obvious candidate for a better selector — but only for a model trained on `balanced`, since `balanced`'s val contains 11-36, which is half of `official`'s **train** (`memory/dataset-properties.md`, `TODO.md` A50).

## `--eval-stride` changes which frames were measured, not just how many

FROG's every-frame-annotated-at-40Hz protocol means adjacent frames are highly correlated near-duplicates, so a `--eval-stride 40` number is usually close to the `stride=1` number on FROG specifically — but it is still a different sample, and DROW's already-sparse ~5%-of-scans annotation means this correlation argument does not carry over there.
Never compare a strided FROG number against an unstrided DROW number and call the gap architectural.

**And a stride does not only move the absolute number -- it can get the size of a *difference* wrong by more than an order of magnitude, which is usually the thing being measured** (2026-11-03, item 64). A 41-frame DROW slice (`--eval-stride 60`) put the whole decode-and-score path at **+1.7 pp**; the full 2,428-frame split put it at **+0.1 pp** -- a factor of 17 in the one quantity the item existed to measure, in the direction that flattered it. The slice gave no hint of this, because its *absolute* cells looked entirely reasonable: 44.2% against the split's 48.1%, a plausible few points low. The arithmetic is the reason to expect it: a difference between two strided numbers carries roughly the noise of each, so a slice small enough to leave a point of noise on one cell cannot resolve a tenth of a point between two cells at all. **So treat a strided run as a smoke test of the plumbing -- every cell reachable, no cell raising -- and never as a preview of the comparison, however sane its absolute values look.** A strided delta that is going to be quoted needs its own error bar, which at this scale would swallow it.

## `--eval-batch-size` used to change the answer, and the habit it leaves behind

Until 2026-09-10 the full-scan models folded the minibatch into the **beam** axis, so a batch of B scans became one `B*720`-beam "scan" and normalization statistics, the SE gate and the beam-axis convolutions all leaked between frames sharing a batch.
Evaluation batches are consecutive frames in time, so a larger batch leaked more temporal context and scored higher — measured on the real val split at `dtime`=10, beam-level person AUC ran **0.9644 (batch 1), 0.9696 (4), 0.9765 (16), 0.9810 (64)**.
`evaluate.py` defaults to 16, `train.py` used 4, the robot runs `forward_one()` at batch 1: the deployed model was the 0.9644 one and every published figure was inflated relative to it.

Fixed (`TODO.md` A18), and a regression test now asserts a frame's output is identical alone or inside a batch of 16.
The habit worth keeping: **when a metric moves with a knob that should not affect the model, that is a bug in the harness, not a property of the model.**
Batch size, evaluation `dtime`, split membership and checkpoint selection are all knobs of exactly that kind.

## `dtime` at evaluation must match `dtime` at training

Scoring a `dtime=10` model at `dtime=1` cost **2.5pp wp-AUC** (76.3% vs 78.8%) and was silent — the parameter simply defaulted to 1 and nothing passed it.
`dtime` now lives in the checkpoint and `evaluate.py --eval-dtime` defaults to it, so a hand-passed flag can no longer disagree with how the weights were trained.
Pass the flag only to deliberately evaluate off-distribution, and say so when you report the number.
Checkpoints saved before 2026-09-10 carry no `dtime` and fall back to 1 — for those, pass it by hand and check the training log for what it should be.
