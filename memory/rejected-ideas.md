# Rejected ideas — do not re-propose without new evidence

*keywords:* rejected, tried, did not work, negative result, reopen, settled, consensus filter, dtime, wide window, tracker, attention, GRU

Decisions to NOT do something, recorded so they are not rediscovered and re-argued from scratch.
Open questions live in [`../TODO.md`](../TODO.md); this file is for settled ones.

Each entry states what was proposed, what was measured, and what new evidence would be needed to reopen it.

## A learned association width

*keywords: item 20, D5, association width, gate_m, SlotRules, S4, 99.2%, flat curve*

**Idea.** Make the Gaussian width that matches a candidate to a slot learned and input-dependent, rather than the fixed
`gate_m` in `SlotRules`. It was the last of the "what could become trainable" list to stay open.

**Why rejected without a run (owner's call, 2026-10-22).** An earlier sweep had already covered the parameter it would learn:
**99.1 / 99.2 / 99.2%** of annotated people have a live slot within 0.5 m at gates of 0.3 / 0.6 / 1.0 m. The association
is not the limit -- it reaches essentially everybody, and the curve is flat across a 3x range -- so a learned width has
**0.8% of people to recover** and no gradient in the measurement to suggest where it would find them. S4 was closed as
"not worth a training run" on this same reading, which this question inherits.

**What would reopen it.** A dataset where association demonstrably fails: more crowding than FROG's 3.06 people per
frame, or a frame rate low enough that a person moves further than the gate between scans (DROW at 12.5 Hz is 3.2x
FROG's period, so DROW and JRDB are where to look).

## The full voteless LFE head: a Gaussian heatmap with a peak decoder

*keywords: item 44b, LFE head, peaks decoder, heatmap, no vote regression, dropped without an arm, fixed-weight decomposition*

**Idea.** Replace DROW's per-beam vote head with LFE's published parameterisation: a Gaussian heatmap over beams, a
peak decoder in place of `nms-mean`, and no vote regression at all. The vote head is the one piece this project took
deliberately unchanged, so "did you try the published alternative?" is a fair reviewer question.

**Why it was dropped without its own arm (owner, 2026-10-27), and why that is a measurement rather than a shrug.**
Two independent results bound it:

1. **The decoder half was already measured on fixed weights** (`static-detector-diagnosis.md`): our own network read
   through LFE-Peaks' own decoder scores **71.26% against `nms-mean`'s 79.88%**, and **50.3% against 81.6%** at 8-10 m
   at a tight operating point. Of that 8.62 pp gap, **2.84 pp is centre regression a voteless head cannot recover at
   all**.
2. **The objective half was then measured by the soft target's probe**, which is the part a fixed-weight decomposition
   *cannot* bound -- that page says so itself: "the network's +6.15 is a lower bound, since our probability map was
   trained for votes, not for peaks". Trained *for* a Gaussian target, `peaks` improves **+2.81 pp** (72.54% ->
   75.35%) -- and still trails `nms-mean`'s **82.78%** by **7.4 pp**.

So the published head loses by 7.4 pp **on the objective that favours it**, with its single largest deficit
structurally unavailable to it. An arm would spend a stage-2 run plus a stage-3 arm to confirm a conclusion two
existing measurements already reach from opposite directions.

**What would reopen it.** A dataset where candidate *positions* are not the limiting factor, or a deployment that
cannot afford the vote head's decode. Neither describes FROG, where stage 2 already places 98.1-98.5% of people inside
the 0.5 m match radius.

**What survives.** The soft Gaussian target's half -- `beam_heatmap` and `compute_loss(soft_labels=)` -- is independent of this and
remains a live candidate on its own evidence.

## A learned grid over the room: spatial binding in place of object binding

*keywords: item 22, grid slots, cell_m, spatial binding, occupancy grid, 2x2, room frame, measured not assumed*

**Idea.** Replace object-bound slots with cells of the room. The obvious reviewer question about this architecture --
"why slots rather than an occupancy grid?" -- and the classical robotics answer, never tested here. The global-state ablation had shown
*addressability* matters but never **which kind**, so the paper's central claim was assumed rather than supported.

**Run as sparse "grid slots" (the owner's design), because a dense grid is not the question.** FROG's venue is
120-250 m across, so person-scale cells need ~46,000 of them -- 180x the recall attention's cost -- and 256 room cells
would be ~11 x 17 m against a 0.5 m match radius. Instead: keep the 256 slots, quantise each one's position to a room
cell, never instantiate an empty cell. Capacity, frame, recurrence, recall, head and eviction unchanged, so **one
variable moves** -- an adopted slot's position follows its object, a grid slot's never moves.

**What was measured** (2026-10-27/28, single seeds, all arms on identical candidates at 82.76%):

| arm | memory AP | gain | share |
| --- | ---: | ---: | ---: |
| adopted | 86.12% | +3.88 | 100% |
| cell, room frame | 84.46% | +1.70 | **44%** |
| cell, sensor frame | 83.62% | +0.86 | 22% |
| cell 2 m, room frame *(screen)* | 83.47% | +0.74 | 19% |

**Why rejected: it recovers 44% of the memory's value at the same capacity**, -1.66 pp and outside the 1.32 pp bar.
Coarser cells are worse, so the loss is not a resolution artefact.

**What it bought even though it lost**, and the reason it was worth a day of GPU: the contribution is now a
measurement. The write-up can state that object binding was compared against the spatial alternative at matched
capacity in the same room frame and recovers 2.3x the value -- rather than asserting it.

**Where it lives.** `SlotRules.cell_m` and `--cell-m` on `step3a`; `tests/test_grid_slots.py`; `utils/queue_object_binding.py`.
Off by default and bit-identical when off.

**What would reopen it.** A deployment with no usable odometry, where object binding's dependence on the association's
geometric consistency is a liability -- the sensor-frame row suggests cell binding degrades more gracefully, losing
0.84 pp to the frame where object binding loses 1.31.

## Slot Attention's own update, abandoned on a numerical guard

*keywords: item 42, Slot Attention, Locatello, --iterations, NaN gradients, training guard, abandoned unmeasured*

**Idea.** `docs/RESEARCH.md` credited Slot Attention for the *abstraction* and not the mechanism. The global-state ablation proved the
abstraction load-bearing, and a citable cell was preferred to a re-implemented one, so the published *update* --
iterated competitive attention, softmax over slots, with a GRU state step -- was the conclusion of that logic.

**What happened** (2026-10-25). Implemented as `--iterations N` with 24 tests and a clean two-recording smoke. The arm
trained **19 epochs** and hit the non-finite guard at epoch 20, iteration 79: **finite outputs, finite parameters, NaN
gradients in all 30 parameters**, 11 consecutive times, losses finite throughout. Not divergence.

**Three refuted hypotheses, tested on the failing run's own weights** -- worth keeping so nobody re-derives them: a
degenerate frame (8 probes: no live slot, no valid candidate, zero state, an 80 m bias) has a finite backward; the
`clamp(min=1e-6)` normalisation does **not** amplify gradients (2.1e1 -> 1.7e0 -> 8.4e-1 -> exactly 0 as live slots go
256 -> 3 -> 1 -> 0); and the MLP residual compounded over a full 399-frame chunk leaves the state bounded at 70-80 with
**0 of 30** gradients non-finite, because minGRU's gate is a convex combination.

**Why rejected: a grounded guess from partial results -- not "it crashed, so we dropped it".** The distinction
matters for how the write-up should read. The crash is what *stopped* the arm; what *ended* it was where its curve
sat. At epoch 18 its validation gain was **+2.22 pp** against the simplified baseline's **~+2.70 pp** at the
comparable point, and that baseline finished at +5.09 -- so the arm was running **second to its own control, measured
the same way**, with no sign of the sudden gain the run-everything-once rule exists to catch. Against that, resuming
meant finding a NaN whose cause three hypotheses had failed to explain, then a fresh ~10.5 h arm.

**The evidence is real but weak in a specific way, and the write-up should say so.** A validation curve at epoch 18 of
~50 predicts poorly in absolute terms -- `dim 64` went +4.15 at epoch 20 to +4.96 at 52 -- so the decision rests on
*relative position against a control*, which is the strongest claim a partial curve supports. This is the one arm
where a reviewer could fairly ask for the full run, and the answer is that it was deprioritised on interim evidence,
not that it was unmeasurable.

**Where it lives.** `--iterations N` on `step3a`; `_SlotAttention` in
`library/takheliped/detectors/takheliped.py`; `tests/test_slot_attention.py`. Off by default and bit-identical when
off. The failing state is kept at `checkpoints_three_horizon/a70_slot_attention_s0/step3a_nan.pth`.

**What would reopen it.** A dataset where the hand-coded association actually fails -- S4 put its headroom at ~0.8% of
people on FROG, which is part of why the arm was not worth a second night. DROW at 12.5 Hz, where a person moves 3.2x
further between scans, is the place to look.

## Position refinement: slots regressing a correction to the candidate's position

*keywords: item 46, O3, position refinement, --refine, candidate_xy, matched_person_xy, recall at matched FP, coverage*

**Idea.** The only mechanism on the architecture list that moves a detection rather than rescoring it. The coverage audit gave it
the reason it had lacked for a year: at its own operating point the memory recovers 203 person-observations and loses
2,990, so its entire +3.9 pp of AP is ranking, and a position correction was the one way carried state might become
coverage.

**What was measured** (2026-10-25, one seed, against `a66_mingru_s0`): AP **86.24% against 86.40%**, -0.16 pp inside
the 1.32 pp bar. Raw recall rose to **86.38%** from 84.33% -- but only by moving the operating point, FP 1.065 against
0.911. **At matched false positives recall is 95.45% against 95.84%**, i.e. slightly worse. The coverage question the
audit posed is answered negatively.

**Why rejected: it changes the operating point, not the coverage.** And it cost nothing to find out -- ~605 s per 10 s
epoch against the control's ~700, because its Hungarian runs once per frame over a `[P, G]` of about 64x3.

**Where it lives.** `--refine` and `--refine-weight` on `step3a`; `MemoryOutput.candidate_xy` (equal to the input
positions when the head is absent, so the evaluation path never branches on a flag); `matched_person_xy` and
`refinement_loss` in `utils/train_takheliped.py`. Off by default and bit-identical when off.

**What would reopen it.** A dataset where candidate *positions* are the limiting error rather than candidate *scores*.
FROG's stage 2 already places 98.1-98.5% of people inside the 0.5 m match radius, so there is almost no
position error for a correction to remove; DROW at 12.5 Hz, where a person moves 3.2x further between scans, is where
to look.

## Hungarian association, and four stage-2 shape knobs

*keywords: item 43, item 45, hungarian, linear_sum_assignment, mutual nearest-neighbour, kernel_size, expansion, blocks, bottleneck_blocks, batch 1*

**Hungarian association.** The standard primitive against the adopted mutual-nearest-neighbour rule, over the
gated sub-problem per stream. **86.08% against the control's 86.40%**, -0.32 pp inside the 1.32 pp bar, and **+0.08 ms**
on a 19.46 ms CPU total, inside the noise its identical `stage2` control phase measures. It differs only in operating
point: recall **88.94%** against 84.33%, at FP 1.330 against 0.911, because a globally optimal assignment keeps more
slots matched.

**Why it is dropped:** it changes nothing, not because it is expensive. **This project predicted it would cost on the
CPU and was wrong** -- 4.3x per training epoch on 1 s chunks does not transfer to batch-1 inference
(`interpreting-evaluation.md`). Training total was 1.4x (13.99 h against 10.06, 58 epochs against 64), and the arm
survived two OOMs on `--resume`.

**four shape knobs** -- all already `CalibrationNetwork` arguments, chosen once and never varied. Averaged
weights through `nms-mean`, control mean 82.19 (range 82.08-82.28): `expansion2` **82.27**, `kernel11` **82.15**,
`kernel5` **82.11**, `deeper` **81.87**. All inside or at the range. **Depth is flat just as the sweep found width flat**,
and halving ConvNeXt's inverted bottleneck changes nothing.

**Not run from the same item, deliberately:** `--neighbours` changes `N_BEAM_FEATURES` and the stem's input width, and
`--top-p` is a candidate-cache key whose rebuild costs more than the arm.

**Where they live.** `--association hungarian` in `SlotRules` with `_assign` in the detector; `--kernel-size`,
`--expansion`, `--blocks`, `--bottleneck-blocks` on `step2`, all four in `network_shape`. Off by default.

## Ablating stage 2's whole-scan aggregator, and dropping stage 3's curriculum

*keywords: item 40, item 41, --no-aggregator, whole-scan max, --chunk-s 10, curriculum, averaged weights, best checkpoint, 1.40 pp*

Both were **flat**, and both are kept in the tree because each would simplify the architecture at no measured cost if
the owner wants it.

**the whole-scan max aggregator.** `--no-aggregator` drops `x.amax(dim=-1).expand_as(x)` from the bottleneck's
concatenation, halving `aggregate`'s input width and saving 16,384 parameters. On the averaged weights through
`nms-mean`: **82.24% against the control's 82.19% mean and 82.08-82.28 range**, with the highest recall of the four
(94.7%). Nothing lost.

**And the trap it set.** Its best *single* checkpoint scored 80.84%, which is -1.12 pp and reads as a real effect.
Weight averaging gained this arm **+1.40 pp** where the controls gain +0.1 to +0.4, so the two readings disagree by more
than a point and only the averaged one describes what stage 3 loads. **Score stage-2 arms on averaged weights**; the
best-checkpoint column in `step2_results.json` is not a substitute and can invert the verdict.

**What it leaves open.** The attribution ladder put about half the lead over LFE-Peaks in the output head; items 19a, 19b and 40 have now
found the block interchangeable, the widths flat and the aggregator free. The other half of stage 2's lead has **no
identified owner**, and the aggregator is described as a defining feature in `../docs/RESEARCH.md` §5.2.

**the 1 s curriculum stage.** `--chunk-s 10` alone: **85.97% against the control's 86.40%**, -0.43 pp inside
the 1.32 pp bar, reached in **7.66 h against 10.06** and 38 epochs against 64. The operating point moved (FP 0.678
against 0.911, recall 79.3% against 84.3%) for the same integrated AP, so the two are not interchangeable at a fixed
threshold.

**Where they live.** `--no-aggregator` on `step2` with an `aggregator` key in `network_shape`; `--chunk-s 10` needs no
code at all. Both off by default and bit-identical when off.

## Sensor-frame vote targets in stage 2's head

*keywords: item 18, --vote-frame sensor, swap_vote_frame, make_targets frame, a67_sensor, beam-relative offsets, across along*

**Idea.** DROW's head, which this project inherited, regresses the offset to a person's centre in **each beam's own
rotated frame** -- `(across, along)`, measured from that beam's endpoint. One person is therefore a different target for
every bearing that sees it. The arm gives every beam the sensor's **one shared Cartesian frame** instead, so a person is
one target. The ladder put about half this project's lead over LFE-Peaks in the output head, which is what made the head worth
probing.

**What was measured** (2026-10-21, three seeds, averaged weights, adopted `nms-mean` decoder): **81.42% mean against the
control's 82.19%, -0.77 pp, with every arm seed below every control seed** -- 81.32-81.48 against 82.08-82.28, the two
ranges not touching. The only separated result of the three arms run that day.

**Why rejected: the beam frame is the better target, and readably so.** The head's input is a per-beam feature stack, so a
beam-relative target asks each output for a quantity its receptive field is already centred on; a shared Cartesian target
asks every beam for the same global coordinate and turns the bearing into something the network must carry internally.

**Where it lives, to restore it.** Kept in the tree, off by default and bit-identical when off: `--vote-frame sensor` on
`step2`, `swap_vote_frame` in `library/takheliped/utils/drow_utils.py` (a pure rotation, its own inverse),
`make_targets(frame=...)` in `utils/train.py`, and `beam_frame_votes` in `utils/train_takheliped.py`, which rotates the
head's output back at the decoders' door so no decoder knows which arm produced it.

## A plain residual block, and doubling stage 2's widths

*keywords: item 19, --block residual, ResidualBlock, ConvNeXtBlock, --channels 64 128 256, a67_residual, a67_wide, capacity plateau*

**Idea.** Two untested parts of stage 2: whether the **ConvNeXt block** is the reason it works (depthwise large kernel,
inverted bottleneck) or whether a plain residual block of two full convolutions would do (He et al., CVPR 2016); and
whether **more capacity** helps, since the width arm had only ever priced shrinking.

**What was measured** (2026-10-21, three seeds each, averaged weights, `nms-mean`): the residual block **82.02% against
82.19%**, -0.17 pp with overlapping ranges; the doubled widths **82.25%**, +0.06 pp for roughly **three times the
parameters**.

**Why rejected: the plateau is flat in both directions.** Shrinking to LFE's widths cost 1.4 pp and widening
gains nothing, so the adopted 32/64/128 sits at the top of a flat region and **capacity is not where this detector's score
lives**. The block is interchangeable, so ConvNeXt is kept for being no worse and already trained, not for being
necessary -- which is what a write-up should say about it.

**Where it lives, to restore it.** `--block residual` (`ResidualBlock`, `BLOCKS` in
`library/takheliped/detectors/takheliped.py`) and `--channels 64 128 256`, which needed no new code. Both are recorded in
`network_shape`, so a checkpoint rebuilds into the network it was trained as.

## Stage 2's coarse temporal convolution

*keywords: A60, coarse convolution, CausalTemporalConv, coarse_lags, --coarse-lags, align_sectors, TemporalCache, CalibrationState, calibration horizon, e76aa94, restore*

**Idea.** `docs/PROPOSAL.md` §5.4: a causal space-time convolution at stage 2's bottleneck, five taps eight frames apart (~0.8 s at FROG's 40 Hz), each past frame's sectors re-aligned by odometry, so the detector sees a walking person's motion before the object memory has met them.

**What was measured.** At stage 2 alone it tied or lost: 77.92% (old controls) or 76.65% (fresh) against the no-temporal 77.23%, and on the adopted recipe 81.09% against 81.93%. **End to end** (2026-10-08, E13's stage 3 on each, three seeds, FROG test): **85.01% against 85.87%** AP, -0.86 at AP@0.3, and on people in their first second in view -- the only place a memory cannot help -- **+0.59 pp** against a +2 pp bar written before the runs (`utils/coarse_e2e.py`, `checkpoints_three_horizon/a60/a60.json`). The memory added less on top of a coarse stage 2 (+2.4 to +3.0) than on top of a no-temporal one (+3.3 to +4.1). It cost +201,216 parameters (+20.6%), ~3.3x the training time per epoch, a ~1.8 MB streaming cache, and odometry inside stage 2.

**Why rejected (owner's call, 2026-10-08): the object memory gets more out of the same sub-second span** -- emptied every 0.5 s it already adds +1.75 pp -- and the convolution earns nothing anywhere it was measured.

**Where it lives, to restore it.** Commit **e76aa94**, `library/takheliped/detectors/takheliped.py`: `align_sectors`, `TemporalCache`, `CausalTemporalConv`, `CalibrationState`, `POSE_CHANNELS`, and `CalibrationNetwork`'s `coarse_lags`, `poses` and `step`; `utils/train_takheliped.py`: `clip_length`, `clip_frames`, `clip_batch`, `mirror_clips` (which also mirrored poses), `network_candidates`' poses, `JointState.calibration`, `detach_joint`'s cache copy, and `step2 --coarse-lags/--clip-frames`. One line restores either file: `git show e76aa94:<path> > <path>`. Checkpoints that need it: `default_s*`, `confirm_nofine_s*`, `step2_no_fine`, `step2_control`, `exclude_limit_s*`, `a60_default_c1_s*`.

**Reopen if** a dataset or sensor makes the memory's first second matter more -- a lower frame rate, where 0.8 s is a few scans, or a scene where people mostly appear briefly -- and measure it end to end again, not at stage 2.

## A second, wider depth-centring channel in stage 2

*keywords: A59, F3, median_windows_deg, --median-windows-deg, MEDIAN_WINDOW_DEG, range minus median, blind spot, 33 deg*

**Idea.** `beam_features`' `r - median` channel uses an 11 deg window, sized for a person at 4 m; nearer people fill it, so a person's own beams read ~0 for 98% of people at 0-1 m and 41% at 1-2 m. Add a second channel over 33 deg, where the signal survives.

**What was measured** (2026-10-05, three seeds on the no-temporal stage 2, FROG test every 5th annotated frame): AP 80.23% through `nms-mean` against the control's 79.75% (+0.48), 77.43 against 77.10 through the vote grid; 1-4 m recall +0.6 pp, against a predicted +1 to +3.

**Why rejected (owner's call, 2026-10-06): the blind spot is real in the input and costs almost nothing in the output.** The 1-4 m deficit it was meant to fix turned out to be the vote-grid decoder's, and once the decoder is right the network finds near people from the neighbour offsets and the 3-tap context alone.

**Reopen if** a dataset puts most people within 2 m of the sensor (a service robot in a corridor), where the blind spot covers the bulk of the data rather than 14% of it.

## A merge radius on the object memory's output

*keywords: A58, merge radius, NMS, double report, duplicate detection, coasting slot beside a candidate, merge_detections, oracle bound*

**Idea.** The memory puts a second detection within 0.5 m of 12-15% of people at matched false positives (E13; the candidates
1.7%), and stage 3 has no merge step, while the merge radius is the best-priced precision lever measured here. Merge its output
as the LFE detectors merge theirs.

**What was measured** (2026-10-05, `tracking_baselines.py merge`, the saved baseline frames; radius chosen per output on val from
none / 0.10-0.60 m, test every 5th annotated frame scored once):

| output | val chooses | test AP, none -> chosen | at 0.30 m | at 0.50 m | oracle: every redundant detection removed |
| --- | --- | --- | --- | --- | --- |
| candidates | 0.50 | 77.66 -> 77.60 | 77.66 | 77.60 | 78.04 (+0.38) |
| E14 ByteTrack + CBMOT | none | 80.96 | 80.51 | 79.31 | 81.48 (+0.52) |
| E13, 3 seeds | 0.10-0.15 | -0.04 to -0.05 | -0.56 to -0.63 | -2.3 | +0.43 to +0.55 |
| B0 (3) / E10 (2) | none | unchanged | -0.5 to -0.6 | -1.7 to -2.2 | -- |

**Why rejected: the double reports are a second person's distance away, and they sit low in the ranking.** The two detections
are a median **0.32 m** apart, about two people side by side, so any radius that removes them removes close neighbours too: past
0.15-0.20 m AP falls monotonically for E14 and every memory arm. 85-87% of E13's double reports are a candidate with a coasting
slot beside it (77% for E14's coasted tracks), and they appear at E13's matched-FP threshold (0.14-0.17), low in the ranking. Even an oracle that
removes every detection matching nobody within 0.5 m of a person, knowing the labels, buys only +0.4-0.55 pp, and E14 almost as
much, so E13's margin over it does not move (+2.90 / +3.16 / +2.51 with both merged at their val radii, against +2.95 / +3.20 /
+2.56 unmerged).

**What would reopen it.** An operating point well above the matched-FP threshold (a deployment threshold of 0.5+ where the
doubles may rank higher), or a training change that makes the coasting slot beside a matched person a confident one. The bound
for a fix inside the model, which would stop a slot coasting beside its own person, is the oracle's +0.26-0.41 pp on slots
alone (`checkpoints_three_horizon/a56/merge_oracle.json`).

## A hand-coded rescoring rule in place of the memory's learned head

*keywords: A56, E14, E15, CBMOT, ByteTrack, rule scorer, track confidence, score update, value as score, saturation, decay*

**Idea.** Rescoring before the threshold is the memory's mechanism; a tracker's own confidence rule might do it with no learning.

**What was measured** (FROG test, the same candidates, every setting tuned on val; `experiments/2026-09-tracking-claims.md`):

| rule | test AP | vs candidates (77.66) |
| --- | --- | --- |
| the memory's value rule used directly as the score (E15) | 69.51 | -8.14 |
| ByteTrack + CBMOT confidence, untuned (no decay), on LFE | collapses: LFE-Peaks 65 -> ~51, LFE-PPN 70 -> ~37 | -- |
| ByteTrack + CBMOT, tuned, with a 0.1/s confidence decay (E14) | 80.96 | +3.30 |
| the learned memory (E13) | 83.86 | +6.20 |

**Why it is not the design.** A rule does recover half the gain -- but only with a decay, because without one a persistent phantom
accumulates confidence exactly as a person does and the ranking collapses; and the value rule, below `gain = 1`, demotes a
confident detection to a young slot's value. The learned state's remaining +2.90 pp sits where a rule has nothing to go on:
occluded people and new trajectories.

**What would reopen it.** A platform where a deployed rule must replace a learned head (no GPU, no training data); then E14's tuned
configuration is the starting point, not the value rule.

## Four slot-memory ideas closed by measurement

*keywords: A51, slots, stage 3, S3, S4, T3, D2, more candidates, learned association, slot statistics, accumulated value, capacity*

All four were in the plan, all four were closed without a training run each, and the measurements are in
[`slot-design-evidence.md`](slot-design-evidence.md).

**More candidates per frame, to fill the unused slot capacity (S3).** *Closed 2026-09-22.* The memory uses a median of **24 slots
of 256**, and a live slot already sits on **95.5%** of the people stage 3 fails to report, with a median value of 0.986. Spare
capacity is not what is missing, so feeding more candidates fills a shortage that does not exist. **Reopen if** a dataset appears
where the memory never learns about a meaningful share of people -- on FROG that is 4.5% of the misses and 0.7% of all people.

**A learned association rule (S4).** *Closed 2026-09-21.* A rule-only sweep over 3 association gates (0.3-1.0 m) x 3 fade times
(2-30 s) moved coverage by 0.2 pp: **96.9-97.0%** of annotated people have a slot that matched a candidate, at every setting. There
is nothing left for a learned association to recover, and the discrete spawn and eviction decisions have no gradient anyway.
**Reopen if** a dataset or a denser scene pushes the matched share meaningfully below 97%.

**Normalising the slot statistics fed to the learned update (T3).** *Demoted 2026-09-20.* The input that most needed clamping was
slot speed, which read a median of 3.97 m/s while people walked at 0.56. That was the time step, not the scaling: with the running
`dt` the same statistic reads 0.58 m/s. **Reopen if** an arm shows the remaining statistics (age, time since match) saturating.

**Accumulated value as the rule that holds a slot (D2).** *Not rejected -- unsupported.* The ablation that replaces it with the
latest candidate score **scored 83.94% against B0's 83.34% mean**, so it does not lose. The design argues accumulated value keeps a
briefly hidden person's slot; on FROG the learned slot state appears to cover that already. One seed, so this is "claim it and you
will be asked for evidence you do not have", not "it is wrong".

## A per-beam fine temporal convolution (the "calibration" denoiser) in the three-horizon detector

*keywords: A50, fine convolution, fine_lags, --fine-lags, align_beams, calibration horizon, denoising, jitter, 3 taps, 5 taps, 7 taps, tested hypothesis, not required*

**Status: a tested hypothesis that turned out not to be needed. Removed from the architecture 2026-09-17, owner's call.**

**What was proposed.** `docs/PROPOSAL.md` §5.4: a causal convolution over the last few frames of each beam, re-aligned by odometry, so that repeated scans of one surface average into a clean local shape before the U-Net sees them.

**What was measured.**

- On FROG `official` test (wp-AUC), no fine convolution scored 78.03 / 77.89 / 76.60 at three seeds (mean 77.51) and five taps 78.56 / 77.32 / 76.63 (mean 77.50): a tie. It costs 9.27 against 6.54 ms per frame.
- The 3-tap control (76.62), 7 taps (77.14) and the `0 2 4` skip (77.61) are single runs, all inside the ~0.9 pp per-run spread, so they rank nothing.
- A median denoiser in its place erased legs (next entry).
- The mechanism had nothing to work on. The anomaly checker (`memory/noise-structure.md`) puts consecutive-frame jitter at 9-20 mm under 10 m on FROG and at the 1 cm rounding on DROW, within the sensors' rated accuracy and far below the 0.5 m match radius. Single-frame spikes are rare on FROG and are leg motion on DROW.

**What replaced it.** One normalisation rule (`sanitize_ranges`): a reading that is non-finite, non-positive or at or beyond `--max-range-m` reads as that limit.

**What would reopen it.** A dataset whose checker report shows jitter comparable to the match radius, or spikes that are not motion, with a model trained on it that loses AP to that noise.

## Excluding beams at the range limit from stage 2's training loss: a tie, not needed

*keywords: A50, --limit-beams-in-loss, training_beams, calibration_loss, limit beams, missing returns, loss mask, open space negatives*

**What was proposed.** At decoding, beams at the range limit cast no vote, which gained +0.34 pp on all six checkpoints. So leave them out of training too.

**What was measured.** Three seeds, default architecture, FROG `official` test, default decoding: 77.68 / 76.72 / 76.58 (mean 76.99). Against old controls (78.40 / 78.28 / 77.09) it first looked like −0.93 pp. Controls retrained on the same code scored 78.19 / 76.64 / 75.13 (mean 76.65): **a tie, +0.34, inside the ~1 pp per-run spread.**

**Reading.** No benefit, so not worth an option: the flag was removed 2026-09-17. Since seeds do not reproduce runs here, the first "lower on every seed" reading was noise.

**What would reopen it.** A dataset where limit beams are rare enough that the class balance is not affected, or a variant that keeps them as negatives for classification and drops only their vote loss.

## Median-based temporal denoising of range scans, replacing the fine convolution

*keywords: A50, denoise_ranges, --fine-denoise, project_ranges, most recent non-outlier, median outlier rejection, fine convolution, temporal denoising, thin moving objects, legs*

**Idea.**
The fine temporal convolution's job is denoising ("repeated scans of one surface average into a clean shape"), and the phase-2 ablations showed it failing at that. So replace it with a parameter-free rule: per beam, over the last five frames aligned to the current pose, reject readings far from the window median and keep the most recent one that is not an outlier. No parameters, and no shortcut for the optimiser to fit.

**What was tried.** `--fine-lags 0 --fine-denoise 0.3`: window `0 1 2 3 4`, tolerance 0.3 m, one step-2 run on `official`, seed 0. It was measured against no-fine, which is the identical network without the denoiser.

| arm | wp-AUC | shuffled history |
| --- | --- | --- |
| no-fine | 78.03% | 77.38% |
| **median denoiser** | **68.87%** | **43.84%** |

**Why rejected.**
It erases people. On real test frames the rule replaces **10.32% of readings near annotated people against 3.45% elsewhere, 3.0x**. A leg crossing a beam is present in only a few of the five frames, so the window median is background, and the leg reads as an outlier and is discarded.

Loosening the tolerance makes it *worse*: the ratio is 3.5x at 1.0 m. The erasures are large near-to-far jumps at object edges, so no tolerance separates them from noise.

It is the rule, not the implementation. Projection was checked on the same frames and leaves consecutive-frame disagreement unchanged (1.30 cm raw, 1.35 cm projected, with 1.24 cm of robot motion per frame).

**What would reopen it.**
A rule that cannot erase a near return: **never replace a reading nearer than the window median**. Real objects read near, while spikes and dropouts read far, so a one-sided rule should keep legs and still repair dropouts. That variant is untested. The denoiser code was removed 2026-09-17, so reopening means rebuilding it. One point to keep: the past scans must be *re-measured* from the current pose, not carried unchanged, which `align_beams` does and is why a dedicated `project_ranges` was needed.

**What this does not reject.** Temporal denoising as such, or the reading that the fine horizon's job is denoising. What is rejected is *symmetric* median outlier rejection on a detector whose targets are thin and moving.

## World-location persistence as the false-positive filter (role C)

*keywords: A41, A44, A45, role C, persistence map, background subtraction, occupancy grid, world-indexed, observation span, mobile robot*

**Idea.**
Phantoms are furniture: static, long-lived, and fixed in the world while the robot drives past. People are not. So accumulate evidence per **world location** over tens of seconds and suppress detections standing where something has persisted. The oracle appeared to price this very highly -- **89.7%** of LFE-Peaks' phantom detections and 93.9% of LFE-PPN's are removable at a 30 s horizon.

**What was tried.** Three mechanisms, all built, all measured on populated frames with the recall cost visible:

| mechanism | best FP per FN |
| --- | --- |
| detection-frequency per cell, exponentially forgetting | 0.57 |
| background model from raw scan returns, keyed on hit rate | 1.90 (LFE-PPN 3.32) |
| trail-length filter, *acausal oracle* | 2.54 |
| displacement filter, *acausal oracle* | 4.20 |
| *SORT, for reference* | *3.8* |
| *merge radius, for reference* | *21.9* |

**Why rejected.**
Only the acausal displacement oracle beats SORT, and it is unreachable by construction and still 5x below a merge-radius tweak.

The 89.7% was measured on **person-free** frames, where a filter cannot destroy a true positive. It bounds *removal*, never *discrimination*, and the cost column turns out to be most of the story.

The deeper reason is structural and shows directly in the background map's table: at a 30 s minimum observation span only **11.3%** of phantoms qualify, and at 100 s none do. **On a mobile robot, how long something has been at a location is dominated by how long the robot looked at it.** A museum robot drives past a chair rather than watching it, so world-location persistence is largely a statement about the robot's own transit -- and short-lived is exactly what a person is too.

**What would reopen it.**
A **stationary or slowly-patrolling** platform, where per-location observation span is set by the scene rather than by the robot's path -- the premise fails on mobility, not on the physics.
The published case where place-keyed memory does pay is exactly that one: Luber, Tipaldi & Arras, *Place-dependent people tracking*, IJRR 2011, learns per-place priors over long deployments (`../docs/RESEARCH.md` §7.4.2). Or an **object-keyed** displacement feature rather than a location-keyed persistence one: that is the only variant measured above SORT, and it belongs to role B (`docs/PROPOSAL.md`).

**What this does not reject.** The horizon argument itself. 786 frames is still the span over which the separating evidence exists; what is rejected is reaching it through location.

## Two candidate decoders in parallel, fused by stage 3

*keywords: item 34, union_decode, parallel decoders, complementarity, pool coverage, candidate floor, 98.41%, oracle at the wrong operating point, two score scales*

**Idea** (the owner's, 2026-10-15). `nms-mean` and DROW's vote grid are each better where the other is worse, so decode every
scan with both, hand the union to the object memory, and let the learned rescoring head decide which to trust. The pool has
room -- never more than ~17 of its 64 slots are used -- and stage 3 already rescores candidates, so no new machinery.

**It had a measured ceiling, and that is why it was built.** At LFE-Peaks' 1.784 false positives per frame, over three seeds,
each decoder covered about **900 people of 30,283 that the other missed**: union 27,875 / 28,009 / 27,813 against the better
alone at 26,997 / 27,168 / 26,907, a headroom of **+2.90 / 2.78 / 2.99 pp**.

**What was measured, and why it is rejected.**

**1. As a detector, the union realises 3% of its own ceiling.** Put both decoders through *one* threshold at matched false
positives and the union gains **+35 / +30 / +18 people** against the oracle's +878 / +841 / +906; 8-10 m moves 81.6 -> 81.8%
and close pairs 89.7 -> 89.8%. Same cause as the range split above: the two decoders put their scores on **different
scales**, and one global threshold cuts a mixed set unevenly.

**2. At the candidate floor -- the lens the pipeline actually uses -- there is nothing left to gain.** Candidates are not
detections. They enter stage 3 at `CANDIDATE_FLOOR = 0.01`, the pool is capacity- rather than score-limited, and the memory's
job is to rescore what it is handed. Measured there:

| people with any candidate, of 30,283 | s0 | s1 | s2 | non-person candidates / frame |
| --- | ---: | ---: | ---: | ---: |
| vote-grid | 28,662 (94.65%) | 28,695 (94.76%) | 28,485 (94.06%) | 4.89-5.21 |
| **`nms-mean` (adopted)** | 29,800 (**98.41%**) | 29,829 (**98.50%**) | 29,714 (**98.12%**) | 8.67-9.30 |
| union | 29,802 (98.41%) | 29,820 (98.47%) | 29,705 (98.09%) | 8.63-9.26 |

**+2 / -9 / -9 people, at the same cost.** `nms-mean` alone already puts **98.1-98.5%** of people in front of the memory, and
the vote grid -- which reaches 4 pp *fewer* people at this floor -- holds none of the remaining 1.6%. The two seeds that go
*backwards* are the displacement risk this was built knowing about: the 0.30 m dedup keeps the higher-scoring of a coincident
pair, and where that is the grid's, its slightly worse position can fall outside the 0.5 m match radius.

**The lesson is worth more than the experiment, and it generalises.** The +2.9 pp ceiling was real and was measured
correctly -- at **1.784 false positives per frame**. The candidate stage does not operate there; it operates at a floor of
0.01, where one decoder already has almost everyone. **A complementarity ceiling measured at one operating point does not
transfer to a pipeline that runs at another**, and the two decoders' "disjoint misses" were an artefact of thresholding
rather than a difference in reach. Measure the ceiling *at the operating point the consumer actually uses* -- the same family
of error as `AGENTS.md` rule 3's cross-regime comparisons.

**Reopen if** stage 2's pool coverage at the floor falls well below 98% on another dataset, which is the only condition under
which a second decoder would have people left to contribute. `union_decode` and `pool_coverage` stay in the tree, tested, so
checking that costs one run.

## A range-split candidate decoder, and the NMS radius with it

*keywords: item 13, item 17, range_split_decode, split-6m, split-8m, nms-mean-0.4, nms-mean-0.6, decoder geometry, score scale, matched false positives*

**Idea.** Each candidate decoder is better exactly where the other is worse. At LFE-Peaks' false-positive rate the vote grid
finds **85.3-85.6%** of people at 8-10 m where the adopted `nms-mean` finds **80.9-81.6%**, because aggregating many weak
votes helps where few beams are on target; `nms-mean` finds **86.6-89.7%** of close pairs where the grid finds **66.8-70.5%**.
Close pairs are a near-range problem and sparse returns a far-range one, so run `nms-mean` within R metres of the sensor and
the vote grid beyond it, and keep both wins. The NMS radius was swept beside it as the control, which the range-aware radius had already taught.

**What was measured** (2026-10-15, `stage2_probe.py`, the adopted recipe's three seeds, `official` test every 5th annotated
frame, every decoder scored at LFE-Peaks' own 1.784 false positives per frame):

| recall at matched FP | overall | 8-10 m (13%) | close pairs 0-0.6 m (9%) |
| --- | ---: | ---: | ---: |
| vote-grid | 89.1 / 89.4 / 88.5 | **85.6 / 85.4 / 85.3** | 70.5 / 69.4 / 66.8 |
| **`nms-mean` (adopted, 0.5 m)** | 89.1 / 89.7 / 88.9 | 81.6 / 80.9 / 81.6 | 89.7 / 89.6 / 86.6 |
| `nms-mean` at 0.4 m | 86.8 / 87.4 / 87.0 | 76.9 / 76.5 / 78.0 | **91.8 / 92.6 / 91.9** |
| `nms-mean` at 0.6 m | 89.0 / 89.1 / 88.0 | 83.5 / 82.1 / 82.4 | 76.8 / 74.1 / 71.5 |
| split at 6 m | 89.2 / 89.5 / 89.0 | 79.9 / 78.7 / 80.1 | 86.2 / 86.1 / 82.8 |
| split at 8 m | 89.2 / 89.7 / 88.9 | 78.6 / 77.5 / 78.9 | 88.9 / 88.5 / 85.8 |

**Why rejected, and there are two separate reasons.**

**1. The split is dominated: it is worse at far range than either parent.** Every seed puts split-6m at 78.7-80.1% at
8-10 m, *below* plain `nms-mean`, in the stratum where it is running the vote grid that scores 85.3-85.6% there.
**The table shows why.** At 6-8 m on seed 0, split-6m uses the vote grid exactly, and scores **83.8%** where the vote grid
alone scores **88.1%** -- same decoder, same people, different answer. The only thing that differs is the threshold the
false-positive match picks: **0.670 for the split against 0.476 for the grid**. The two decoders put their scores on
different scales, so one global threshold cuts a mixed set unevenly, and the far-range detections pay for it. A split would
need the two score distributions calibrated onto each other first -- machinery for a prize that reason 2 says is not there.

**2. Overall recall at matched false positives does not move.** vote-grid, `nms-mean`, 0.6 m, split-6m and split-8m all land
within **88.0-89.7%**, inside the seed spread. Decoder geometry **redistributes** recall between range and crowding without
changing the total: 0.4 m buys 2-5 pp on close pairs and loses 4-5 pp at 8-10 m, 0.6 m does the reverse. There is no setting
that is better, only settings that are differently wrong, and the adopted 0.5 m sits in the middle of the trade.

**This also closes the pool-size question**, which merged here once its own precondition showed the 64-candidate pool is never full
(`CHANGELOG.md` 2026-10-15): the remaining lever was the decoder's suppression, and the suppression has no free prize in it.

**Reopen if** a dataset is crowd-dominated rather than range-dominated. `nms-mean` at 0.4 m is the best close-pair decoder
measured here, by 2-5 pp on all three seeds, and on data where most people have a neighbour under 0.6 m that trade reverses.
FROG has 9%.

## Range-aware merge/NMS radius, `r(d) = a + b*d`

*keywords: A38, merge_radius_slope, nms_radius_slope, range-aware radius, close-pair, over-merging, tuning on test*

**Idea.**
Tier 0 (`static-detector-diagnosis.md`) measured one scalar radius failing in both directions at 0.30 m: ~5 pp of LFE-Peaks' recall spent merging adjacent people into one detection, while LFE-PPN emits 1.609 duplicates per frame. Since a detector's localisation error grows with range (9 beams on a person at 9 m against 42 at 2 m) and the separation between two people does not, make the radius range-dependent -- tight where the scan is dense, loose where it is sparse.

**What was tried.**
Implemented (`_merge_nearby(..., slope=)`, `slope = 0.0` bit-identical to the old behaviour, guarded by `tests/test_merge_radius.py`) and swept over a 5x5 `(a, b)` grid on `official` test, every 5th frame, scored at **both** association distances. `utils/nms_sweep.py --skip fullscan_tcn lfe_ppn`.

**Why rejected: it does not beat a constant radius of the same average size.**

| LFE-Peaks | AP@0.5 | AP@0.3 |
| --- | --- | --- |
| `r = 0.30` (the default) | 65.1% | 63.4% |
| **best range-aware**, `a=0.30, b=0.030` -> r(2m)=0.36, r(8m)=0.54 | 67.0% | 65.0% |
| `r = 0.40` constant | 67.7% | 65.7% |
| **`r = 0.45` constant** | **68.1%** | **66.0%** |
| `r = 0.60` constant | 66.1% | 63.8% |

The range-aware optimum averages ~0.45 m over FROG's range distribution, so it is *approximating a constant 0.45 m* -- and scoring 1.1 pp below it at both association distances. The extra parameter buys nothing.

**The grid that produced the +1.9 pp headline was confounded**, and the control is what found it: the intercept grid stopped at 0.30, so "slope helps" could not be separated from "a bigger radius helps". It was the second.

**LFE-PPN, controlled the same way, agrees:**

| LFE-PPN | AP@0.5 | AP@0.3 |
| --- | --- | --- |
| `r = 0.30` (the default) | 69.6% | 64.3% |
| best range-aware, `a=0.30, b=0.030` | 71.9% | 65.2% |
| `r = 0.40` constant | 72.1% | **65.6%** |
| **`r = 0.45` constant** | **72.3%** | 65.3% |
| `r = 0.60` constant | 70.8% | 63.1% |

A constant radius beats the best range-aware cell on both detectors and at both association distances. **And both detectors put the AP optimum at the same 0.40-0.45 m**, despite feeding wildly different inputs into the same merge -- 201.5 raw centroids per frame for LFE-PPN against 5.6 for LFE-Peaks. Two separately post-processed models agreeing on one radius is the same class of evidence that fixed the 0.30 m default in the first place, which makes the next paragraph harder to dismiss as noise.

**What the sweep found instead, which matters more.** AP rises monotonically from 0.30 m to 0.45 m -- well past the value that reproduces the published number, and the repository pinned 0.30 precisely because 0.40 overshoots the paper by +2.1 pp. **Optimising this parameter against AP on the test recording is tuning on test, and it pushes monotonically away from the paper's own configuration.**

More sharply: **AP and the Tier 0 measurement disagree about the direction.** AP wants *more* merging; Tier 0 measured over-merging costing ~5 pp of recall at a fixed operating point. Both are right about their own quantity. AP charges fully for a duplicate and barely at all for a close-pair miss, because the surviving merged detection still matches *one* of the two people -- so the second person's loss shows up only as a single missed GT, against a duplicate's full false positive. This is the same structural blindness `docs/PAPER.md` argues about false positives, on a different axis.

**Confirmed on the operating-point metric too (same day).** The re-run scored against recall, duplicates, phantoms and close-pair misses at threshold 0.3 rather than AP. At matched recall the constant still wins: LFE-Peaks at 76.9% recall costs 1.380 FP/frame constant against 1.445 range-aware, LFE-PPN at ~85.8% costs 3.001 against 3.079. The single comparison it does not lose is LFE-PPN at `d = 0.3 m`, where the two are within noise. **It loses or ties on both metrics, both detectors and both association distances, and never wins.**

**And the motivation itself did not survive.** The ~5 pp of recall this was built to recover is not there: recall moves only **1.8 pp** across radii from 0.20 m to 0.50 m, while duplicates fall 18-fold. Tier 0's close-pair figure counted detections *before* thresholding; at the operating point the class is 1.1% of people. The radius is a precision lever.

**What would reopen it.**
A detector whose localisation error scales differently with range -- the premise is about localisation, and both detectors tested here share a backbone family. The code stays in place, defaulting to `slope = 0.0`, so that costs nothing to set up.

## Naive late-fusion temporal consensus (averaging confidence/position across recent frames)

*keywords: consensus filter, late fusion, temporal averaging, SimpleTracker*

**Idea.**
Smooth a detector's raw per-frame output by averaging confidence, and matching by nearest-neighbour to recent historical positions, to suppress the false positives that dominate every model's error (`interpreting-evaluation.md`).

**What was tried.**
Nothing was built — the design was rejected on inspection before implementation.

**Why rejected.**
Rejected on design grounds.
Averaging *positions* assumes the target is static, which is false for a moving person; the mechanism can only suppress false positives, never recover a false negative, so it addresses one side of a two-sided error at best.

**What would reopen it.**
It would not reopen as stated — `SimpleTracker` (a SORT-style tracker with a predicted, not raw-averaged, position and a miss-tolerant confirmed-track state) was built instead and is the current mechanism; see `detector-architectures.md`.

## Wide real-time temporal window (`dtime=20`, FullScanTCN, FROG)

*keywords: dtime, temporal stride, FullScanTCN, ego-motion, wide window*

**Idea.**
Widen the temporal window from `T=5` consecutive frames (~125 ms of real time) to `T=10` frames spaced `dtime=20` apart (~4.5 s), on the reasoning that a person's own displacement becomes more visible to the network over a longer real-time span.

**What was tried.**
A full retrain of `FullScanTCNDetector` on FROG with `--time-frame 10 --dtime 20`, otherwise identical hyperparameters to the existing `fullscan_tcn.best.pth` baseline.
**`FullScanTCNDetector` was retired on 2026-10-31**; to re-run this, the class lives at commit `98b0716`.
Early-stopped at epoch 8 (best at epoch 3), final val-set wp-AUC 54.8% — a ~15-point regression from the `dtime=1` baseline's 69.5-69.7%.

**Why rejected.**
It did not hold up: FROG ships no real odometry at the time this was tried, so every temporal window was raw, unaligned frame-stacking, and widening the window gave the robot's own uncompensated motion 36x longer to accumulate — a cost that outweighed the intended benefit.
A second, independent test (the `SimpleTracker` invocation-rate sweep, next entry) found the same pattern at a wider real-time gap, reinforcing that this is a structural mismatch, not a single bad hyperparameter.

**What would reopen it.**
FROG's real, official per-session odometry is now wired into the loading pipeline (`data-model.md`) — a `dtime=5`/`10`/`20` sweep retrained on real (not estimated) odometry, with ego-motion actually compensated, is the natural re-test and has not been run yet (`TODO.md`).

## `SimpleTracker` at wide invocation strides (FROG, stride=40, ~1s gap)

*keywords: SimpleTracker, match_radius, min_hits, invocation rate, stride=40*

**Idea.**
Since the fine-stride tracker (`stride=5`, ~125 ms between invocations) gave a real +1.0pp win, run it at a much wider invocation gap (`stride=40`, ~1s) on the reasoning that motion is more visible, and the tracker more useful, the further apart its invocations are.

**What was tried.**
A four-point `match_radius`/`min_hits` grid at stride=40 (`{1.6m, 0.5m} x {2, 3}`), against a no-tracker baseline of 80.4% wp-AUC.
Every configuration tested was worse than the baseline: the best (`radius=1.6m, min_hits=2`) reached 77.9%, and the tight radius carried over from the fine-stride setting collapsed to 34.8-47.5%.

**Why rejected.**
It did not hold up outside the fine-stride case it was tuned on: a single fixed match radius cannot simultaneously be tight enough to avoid confusing two nearby people in FROG's dense crowds (~3.9 people/frame) and loose enough to follow one person's own displacement over ~1s, because per-person speed and inter-person spacing both vary independently.

**What would reopen it.**
A per-track adaptive gate — scaled by that track's own estimated speed, or a proper covariance/Mahalanobis gate as in a full Kalman filter — rather than one global constant.
Not attempted; a meaningfully larger piece of work than a parameter tweak (`TODO.md`).

## Global self-attention / GRU temporal fusion for the full-scan architectures (`FullScanTransformerDetector`, `FullScanCNNDetector`)

*keywords: FullScanTransformerDetector, FullScanCNNDetector, BeamSelfAttention, GRU, attention, recurrence*

**Idea.**
Two of this project's four original full-scan detector designs used components stronger than a plain CNN/TCN: `FullScanTransformerDetector` used global multi-head self-attention across all beams, and `FullScanCNNDetector` used a GRU for temporal aggregation after a dilated 1D backbone.
Both are, on paper, more expressive than the non-recursive designs that replaced them.

**What was tried.**
Both were built and benchmarked once, on the same footing as the other candidates.
`FullScanTransformerDetector` was both the slowest of the four (~1.5s/scan on CPU) and the lowest-scoring.

**Why rejected.**
Rejected on design grounds, confirmed by measurement: a GRU or global self-attention layer violates this research's own non-recursive-only constraint (`docs/RESEARCH.md` §2), which exists to keep every architecture feed-forward, fully parallel, and DirectML-safe — and in the one head-to-head benchmark available, the more expressive design was worse on both accuracy and speed, not merely disqualified on principle.

**What would reopen it.**
It would not reopen under the current research question; a project whose scope explicitly permitted recurrent or attention-based full-scan designs would need to restate the comparison from scratch, not resurrect these two classes as-is — both were deleted from `full_scan.py`.

## Timing the authors' own Li2Former network under this project's CPU protocol

*keywords: item 65, CPU latency, ms/frame, OfficialLi2FormerDetector, cpu_latency.py, speed table, 1675.24, parameter count*

**Idea.** Time `OfficialLi2FormerDetector` the way every other detector is timed -- 2,000 consecutive scans, batch 1, four
threads, against FROG's 25 ms period -- so the speed table's `Li2Former` row describes the published architecture instead of
this project's smaller class. Filed 2026-11-03 as the last unmeasured cell for this detector, and dropped the same day unrun.

**Why rejected (owner's call, 2026-11-03).** Not for the obvious reason, which does not hold on its own: latency is a
property of the graph and the input shape, not of the weights, so the 28.3 pp accuracy gap does not by itself make a timing
meaningless -- the reproduction items put that gap in the training, and measured the decode-and-score path at +0.1 pp. What
does hold is that **there is no single "their time" to measure.** Their code builds **16.85 M parameters in the `PARALLEL`
mode trained here and 29.47 M as their `__init__` builds it**, and reaching the paper's model at all took three stated
departures from their `ros_li2former.yaml` (`MODE`, `add_bf=False`, `NUM_CTS: 720`), one of which switches off a stream they
hardcode on. A CPU number would be the cost of *that* configuration, and the paper publishes no CPU figure to compare it
against -- their 14.95 / 17.54 ms are GPU.

**And the question it was filed to answer is already answered in prose.** `performance-log.md`'s speed table states that the
`Li2FormerDetector` row is this project's class and that the published model is **slower than that row, not equal to it**,
with training putting the authors' network at **~3.5x** our class's CPU cost per frame. The ordering is fixed; an hour of CPU
would refine a number no decision depends on, for a detector that is not a deployment candidate.

**What would reopen it.** A published CPU figure from the authors to compare against, or a decision that puts Li2Former on a
robot. Either way, time it with **`cpu_latency.py --detectors`**, which is what times a detector; `tracking_baselines.py
latency` takes a `--memory` checkpoint and times TAKHeLiPeD's three stages, and the dropped item named it by mistake.
