# TODO — work queue, tiered by when it can happen

**Open work only.**
Settled decisions live in [`CHANGELOG.md`](CHANGELOG.md) for what was done, and in [`memory/rejected-ideas.md`](memory/rejected-ideas.md) for what was deliberately not done, both with the evidence.

Every item states **What / How / Why** so a future agent can act on it cold, without reconstructing the reasoning.

## The tiers

| Tier | Meaning |
| --- | --- |
| ⚠️ NOT DEPLOYED | State, not work. What is in the repo but inert on the robot. |
| §A NOW | Actionable this session or in the next few days. |
| §B UPON A FULL RETRAIN SWEEP | Held deliberately — batched behind the next multi-checkpoint retrain, since each one costs real GPU time. |
| §C BLOCKED | Cannot progress by effort. Waiting on elapsed time, or on data nobody has yet. |
| §D LATER | Unblocked and understood, just not worth the cycles now. |
| §E PROJECT EVOLUTION | Direction changes, not tasks. Needs a decision before it becomes work. |

Started 2026-09-08, alongside the documentation-layout migration.

---

## ⚠️ In the repo but NOT deployed

No merged change currently has a different effect on the robot than what's already running: the DR-SPAAM cutout-normalization fix (`CHANGELOG.md`) doesn't touch `DrowDetector`'s own preprocessing (it's a no-op at DROW's own `window_depth=1.0`), and none of the odometry, tracker, or full-scan-architecture work is wired into any ROS node at all — `deploy/follow_the_drow/nodes/drow_detector.py` only ever runs the original `DrowDetector`.
That's a structural gap, not a pending deploy — see §E below.

| Change | File | Production effect when deployed |
| --- | --- | --- |
| *(none currently queued)* | — | — |

---

## §A NOW

> ### Reorganised 2026-09-10, around a measurement that changed the question
>
> Four history ablations on the epoch-1 checkpoint, identical weights, whole `frog_16-41` test recording (50,088 frames). The decomposition is clean:
>
> | history | wp-AUC | vs full | what the step removes |
> | --- | --- | --- | --- |
> | full | **67.1%** | — | — |
> | shuffled | **66.6%** | **-0.5** | **ordering and direction — i.e. motion** |
> | frozen | **63.0%** | -4.1 | the independent observations — i.e. aggregation |
> | zeroed | 42.0% | -25.1 | nothing further; frozen and zero carry the same information, so the extra 21pp is pure out-of-distribution damage |
>
> **The temporal window is worth 4.1pp, and essentially none of it is motion.** Destroying ordering costs 0.5pp; removing the independent looks costs 3.6pp.
> `SpaceTimeCNN` is a five-frame **averager**, not a temporal detector.
>
> **And its spatial half is weak.** `frozen` — the model's effective single-frame score — is **63.0%**, *below* LFE-Peaks' published 64.9% and LFE-PPN's 66.5%.
> So a 476K-parameter five-frame model beats a 180K single-frame one by **0.6pp**, and does it by averaging rather than by seeing motion.
>
> This project exists to show that temporal information helps. On the official benchmark, with this architecture, it does not — and the benchmark is structurally unable to reward it, because every frame contains 3-4 people and shape alone answers most of the question.
> The response is not to keep tuning: it is to **evaluate where motion is the only available cue**, and to **find an architecture that can use it**.
>
> Previous items A20, A24, A25 (dataset framing) and A22, A26, A27 (architecture) are folded into the phases below rather than left as separate threads.

### Running order

| phase | item | gate | cost |
| --- | --- | --- | --- |
| **1 — data** | ~~A30 three loading modes~~ | **done 2026-09-10** | — |
| | ~~A31 the metric each mode needs~~ | **done 2026-09-10** | — |
| **2 — premise** | **A32** LFE across the modes | **`official` and `transferred` done**; `balanced` left | ~2 min |
| | **A32b** hold training data constant — the confound that would invalidate every mode comparison | — | design |
| | **A32c** port LFE to PyTorch from its ONNX graph, for a retrainable control | A32b | code + weight-port check |
| | **A32d** our LFE reproductions overshoot their published AP | **mostly resolved 2026-09-11**: Peaks -0.5pp, PPN +0.4pp after both merge radii were recalibrated | — |
| **3 — architecture** | **A33** a *validated* ranking test | — (can start now) | code + validation |
| | **A34** candidate survey | A33's filters | reading |
| | **A35** build, rank, train the survivors | A33/A34 | the expensive part |
| **4** | **A36** re-triage everything above | A32/A35 | decision |
| **post-Tier-0** | ~~**A37** re-run everything downstream of LFE-PPN's `nms_radius` change~~ | **done 2026-09-11** | — |
| | ~~**A38** range-aware merge/NMS radius~~ | **rejected 2026-09-11** — loses to a constant 0.45 m | — |
| | ~~**A40** re-score the radius against Tier 0's quantities, not AP~~ | **done 2026-09-11** — A38 stays rejected; the radius beats SORT 5.8x on exchange rate | — |
| | ~~**A41** classical world-frame persistence map~~ | **done 2026-09-11 — negative.** 0.57 FP per FN; location does not discriminate | — |
| | ~~**A45** role-C bound on populated frames~~ | **done 2026-09-11** — 4.20 FP per FN at best; role C demoted, role B gates it | — |
| | ~~**A44** background subtraction from raw scan returns~~ | **done 2026-09-11 — negative.** 1.90 FP per FN; **role C closed** | — |
| | ~~**A42** local-Cartesian input representation~~ | **done 2026-09-11** — ties DROW's tunnel, beats LFE by 6 pp, does not replace the adaptive window | — |
| | **A39** locally range-centred input channel — Tier 0's headline candidate | **unblocked** (A38 closed) | code + one training run |
| | **A43** own architecture carrying all three memories, priority A > B > C | **FROG chain done 2026-09-16: 80.2% test AP** (stage 2 alone 76.9%, DR-SPAAM published 75.6%); owner's review of `docs/PROPOSAL.md` still pending | follow-up work is A50 |
| | **A47** our `DrSpaamDetector` re-encodes a `T`-scan window per call; the official detector keeps its template across calls | found 2026-09-14 by the A43 survey | one streaming evaluation |
| | **A48** re-run everything downstream of the two heading-wrap fixes | found 2026-09-14 by A43 step 0's real-data check | minutes per script; cutout evaluations longer |
| | **A49** the three-horizon detector on FROG, DROW and JRDB, with its parameters stated for each | owner's goal 2026-09-15; starts after A43's FROG results | settings in physical units, a dataset-agnostic training harness, two more training chains |
| | **A50** post-processing of A43's FROG result, then the deferred heavy ablations, then architecture experiments | owner's plan 2026-09-16, in that order | phase 1 cheap to one short retrain each; phases 2-3 one training chain per variant |

**Next session (owner's call 2026-09-17): stage-3 slot experiments** (A50 phase 1b). Start from these facts:

- **Stage 3 needs a new stage 2.** The fine convolution is gone, and `step2_calibration.best.pth` (3 taps), with every stage-3 checkpoint built on it, no longer loads. Use `checkpoints_three_horizon/default_s0/step2_calibration.best.pth` (78.19% test, trained on current code 2026-09-17; `default_s1`/`default_s2` are its sister seeds). Stage-3 candidate caches are now keyed `-limit-silenced`, so they rebuild with the new decoding.
- **Naming from 2026-09-17:** with `--fine-lags` gone, the runs formerly called "no-fine" are the **default** architecture (coarse convolution only), and "static" is **no-temporal** (`--coarse-lags 0`). Directory names (`step2_no_fine`, `confirm_nofine_s*`, `step2_static`, `confirm_static_s*`) keep the old words because the logs refer to them.
- **Measure the time-step suspect first.** It is cheap and needs no stage-2 training. FROG timestamps are bunched, stage 3 clips `dt` to 1 ms, and `manage_slots` divides displacement by it. Log slot speeds against walking speed.
- **FROG is 40 Hz, not 26.2** (A49): `FRAME_PERIOD_S` sets step 3's chunk and validation window lengths 1.53x shorter than their flags say. Decide whether to fix that before or after the slot experiments, because either order changes step-3a numbers.
- The slots-versus-coarse-convolution hypothesis is parked in §D, and it needs a working stage 3 first.
- Single runs must differ by ~2.5 pp to count; confirm with 3 seeds.

**Phase 2 answered its question, and reframed it.** Published static detectors emit **1.4-3.4 genuine phantoms per frame** — and at close to the same rate whether or not anyone is present (`official` 1.387/3.416 against `transferred` 1.391/3.174).
So empty scenes do not *cause* the failure; the benchmark's ratio metric was *concealing* it, since 1.43 phantoms still reads as 63% precision beside 3.06 real people.
**And the phantoms persist** — 88.3% / 77.6% recur within 0.3 m next frame against 99.8% for real people — so they are furniture, and no short-window persistence filter removes them.
The claim that survives is sharper than the one filed: *the separating evidence between a chair and a standing person exists only at a horizon of tens of seconds*, which is why a shuffled five-frame window costs just 0.5pp.
Argument in `docs/PAPER.md`; numbers in `CHANGELOG.md` and `memory/performance-log.md`.

### A32. Finish the premise check: LFE on `balanced`

**Done, and it reframed the question.** LFE-Peaks and LFE-PPN on `official` test (**65.1%** / **69.6%** AP, **1.76** / **5.03** FP/frame, **0.63** / **0.38** FN/frame) and on `transferred`'s 72,533 person-free frames (**1.39** / **3.17** FP/frame at 0.3). *(LFE-PPN re-measured 2026-09-11 after its `nms_radius` recalibration — `CHANGELOG.md`.)*
The false-positive rate barely moves between them: the benchmark was concealing a constant absolute error inside a ratio, not failing to provoke one.
See `docs/PAPER.md`.

**Left to run.** Both models on `balanced` test, now ~2 minutes each since the LFE-PPN fix.
Expect degradation against `official`: the split is 19.1% person-free, so the same false positives that `transferred` isolates now land inside a valid AP.
That number is the **headline metric** (A31), so it is the one our own architectures will have to beat.

**Read it as a transfer result, not an in-domain one.** LFE's published weights were trained on `official`'s train set; `balanced`'s test recordings (16-41, 15-53) are held out from that, which is why the comparison is honest — but a `balanced`-trained model of ours would be in-domain there and LFE would not. That gap is exactly what A32b forbids papering over and A32c fixes.

**Not a re-derivation of a published number**, so no conflict with the citation policy (`memory/performance-log.md`): nobody has published a FROG number on person-free or mixed-occupancy splits.

### A37. Re-run everything downstream of LFE-PPN's `nms_radius` recalibration

**Done on 2026-09-11**: the false-positive tables, the phantom/duplicate decomposition, the persistence measurement, LFE-PPN's AP, and `phantom_analysis.py`'s true-positive counting (`CHANGELOG.md`).

**Still stale, all LFE-PPN only:**

- ~~**`tracker_sweep.py`**~~ **done 2026-09-11.** LFE-Peaks' rows came back unchanged to the digit — a fourth independent confirmation of the corrected baseline. LFE-PPN's curve is **steeper** than the stale one, not shallower: `min_hits = 50` now costs 47.2 pp of AP against the old 22.4 pp, while leaving 34.7% of person-free frames with a phantom against the old 47.1%. The qualitative claim (PPN benefits less from tracking despite a larger transient fraction) survives but narrows, 12.8% against 13.4% at `min_hits = 3` where it was 10.7% against 13.4%.
- **`temporal_oracle.py` / `horizon_sweep.py` precision halves** — the trail-length curves in `dataset-properties.md` start from 4.15 phantoms/frame, which *is* the current radius, so these are **not** stale; they were run after the change. Confirmed, not assumed: the 4.15 figure reproduces exactly.
- **`error_breakdown.py`** if any of its numbers are quoted anywhere.

**The recall-side oracle numbers are unaffected** — they were measured on LFE-Peaks and LFE-PPN detections at the current radius.

### A38. Range-aware merge/NMS radius -- REJECTED 2026-09-11

Built, swept, and beaten by a constant radius of the same average size: best range-aware cell 67.0% / 65.0% AP against a constant 0.45 m at 68.1% / 66.0%. Full reasoning, and the confound in the original grid, in `memory/rejected-ideas.md`. The code stays, defaulting to `slope = 0.0`.

### A40. Re-score the merge radius against Tier 0's quantities, not AP -- DONE 2026-09-11

Three findings, in `CHANGELOG.md`. A38 stays rejected on this metric too. Tier 0's "~5 pp in close-pair over-merging" was overstated -- it counted detections before thresholding, and the real figure is 1.1% of people; recall moves only 1.8 pp across the whole radius range while duplicates fall 18-fold. And **the merge radius removes false positives 5.8x more cheaply than SORT does at matched reduction** (0.015 FN/frame against 0.087 for the same 0.33 FP/frame), which is a `docs/PAPER.md` result and is now in its tracking section.

**Left open deliberately: the 0.30 m default has not moved.** AP says 0.45 m, the paper-reproduction rule says 0.30 m, and the operating-point view says there is no optimum -- only an exchange rate. Picking one is a decision about what the detector is *for*, and it is the owner's.

### A41. Classical world-frame persistence map -- DONE 2026-09-11, NEGATIVE

Built (`utils/persistence_map.py`), swept, and beaten by everything it was compared against: **0.57 false positives removed per false negative added**, against SORT's 3.8 and the merge radius's 21.9. Removing 73.8% of detections costs recall 79.3% -> 23.2%.

**The cause is identified and it is not a tuning problem.** The role-C oracle bound was measured only on person-free frames, so it bounds *removal* and never *discrimination*. People revisit locations, so a per-cell count of "detections have happened here" cannot separate a queue from a chair. Full numbers: `CHANGELOG.md`.

The code stays: the accumulator is reusable for A44, and the negative result is a result.

### A45. The role-C bound on populated frames -- DONE 2026-09-11

**Role C is demoted.** The 89.7% headline was a removal figure measured where nothing could be destroyed. On populated frames the best acausal trail-length filter reaches **2.5 FP per FN**; keyed on displacement -- `PAPER.md`'s own separating feature, and the right one -- **4.20**, against SORT's 3.8 and the merge radius's 21.9.

**Role B now gates role C.** The ceiling is association-limited: greedy nearest-neighbour association merges a person into a nearby phantom's trail, so filtering the chair destroys the person. Trails cannot be filtered better than they can be formed.

Full numbers and the correction to an earlier overstatement: `CHANGELOG.md`.

### A44. Background subtraction from raw scan returns -- DONE 2026-09-11, NEGATIVE

Built (`BackgroundMap`), measured, and below every lever already priced: best **1.90 FP per FN** against SORT's 3.8 and the merge radius's 21.9.

**Role C is now closed across three mechanisms** (A41 0.57, A44 1.90, A45's acausal oracles 2.54 and 4.20). The unifying cause is in A44's table: at a 30 s minimum observation span only 11.3% of phantoms qualify and at 100 s none do -- **on a mobile robot, persistence at a location measures how long the robot looked, not whether the object is furniture.** Written up in `memory/rejected-ideas.md` with the conditions that would reopen it.

**The horizon argument survives**; what is rejected is reaching it through location rather than through objects.

### A47. Evaluate DR-SPAAM in its official streaming form

**What.** The official DR-SPAAM detector is stateful: in inference mode `_SpatialAttentionMemory` keeps `self._memory` across calls (`memory = alpha * new + (1 - alpha) * attend(memory)`) and `reset()` clears it. Our `DrSpaamDetector.forward` instead re-encodes a `T`-scan window from scratch on every call (`architectures.py`, the template is initialised from `x[:, :, 0]` each time). Every DR-SPAAM number in this repository is therefore the windowed form.

**How.** Add a streaming evaluation path that carries the template between consecutive frames of a sequence and resets it at sequence boundaries; compare against the windowed form on the DROW test set with the published weights. Write the test first: a streamed run over frames `0..T-1` must equal the windowed forward on the same `T` frames.

**Why.** Fidelity of a replicated baseline (`AGENTS.md` rule 3), and it is the in-domain precedent for A43's `step` interface. With `alpha = 0.5` old evidence halves every frame, so the gap is expected to be small -- which is a prediction to check, not a reason to skip it.

### A48. Re-run everything downstream of the two heading-wrap fixes

**What.** Two bugs fixed 2026-09-14 (`CHANGELOG.md`): `FROG_Dataset._load_odom` interpolated heading straight through 0° at every ±180° crossing (54 single-frame jumps above 2° on the test recording, up to 144°; now 2), and `cutout()` shifted windows by an unwrapped heading difference (a 2π shift at a crossing). Every stored number computed through either is stale, by an amount expected to be small (~0.14% of FROG frames) but unmeasured.

**How.** Re-run and diff against the documented numbers, cheapest first; a change beyond the third significant digit is itself a finding to write up:

1. `tracker_sweep.py` -- the SORT exchange rates (3.8 / 9.8) in `docs/PAPER.md`.
2. `persistence_map.py` and `--background` (A41 0.57, A44 1.90).
3. `temporal_oracle.py`, `horizon_sweep.py`, `horizon_sweep.py --discrimination` (A45 2.54 / 4.20).
4. `motion_analysis.py --people / --phantoms / --headroom`, `phantom_analysis.py` (`memory/dataset-properties.md`).
5. `slot_budget.py` and `--memory` (`PROPOSAL.md` §5.5).
6. DR-SPAAM windowed evaluations with `T > 1` on DROW and FROG (overlaps A47 -- do them together).

Trained `--align-scans` checkpoints saw the bad headings in the windows touching a crossing; they are not worth a retrain for that alone and go stale with the next retrain sweep (§B).

**Status 2026-09-15: items 1-5 re-run (13 scripts, all exit 0; logs to be written up); item 6 open.** First read against the documented numbers:

- SORT (`tracker_sweep.py`): AP up 0.1-0.9 pp at every `min_hits`, FN slightly down; LFE-Peaks at `min_hits = 3` now 63.7% AP, 1.430 FP / 0.715 FN per frame, exchange rate 4.06 against the documented 3.85. Qualitative claims hold.
- Unchanged: recall oracle +9.6 pp to 85.5%; phantom trails (67.7% of detections on trails of 1 s or more, 89.7% removable at 30 s, LFE-Peaks' longest 36.7 s); persistence map (0.50 m, 30 s: 73.9% removed, recall 23.2%); phantom recurrence 88.3% / 77.6%; people moved < 0.5 m (67.4 / 47.0 / 36.5 / 30.3 / 15.3% at 1-30 s, documented 67.3 / 46.6 / 36.0 / 29.5 / 16.3%); motion headroom within 0.2 pp.
- Changed: person trajectories 1,087 -> 888, median 4.9 s -> 6.8 s, max 67.7 s -> 74.8 s, 3.6% -> 4.6% reaching 30 s (fewer broken tracks); background model best 1.90 -> ~1.80 (LFE-PPN 3.32 -> 2.92).
- **Slot occupancy fell sharply and `PROPOSAL.md` §15's unexplained spike is gone** -- it was the heading bug breaking association. At a 10 s retire window, p99 / max slots: people 31 / 40 -> **16 / 19**, LFE-Peaks 55 / 79 -> **39 / 44**, LFE-PPN 105 / 164 -> **73 / 112**; the people p99.9 jump from 0 s to 1 s is now 9 -> 10, not 9 -> 28. Memory replay at 256 slots: LFE-Peaks 21.6 / 73 / 91 -> **20.4 / 49 / 54**, LFE-PPN 79.1 / 179 / 256 -> **76.0 / 151 / 172**, and **no slot is ever replaced** (LFE-PPN person-slot replacements 2 -> 0), so the rule-versus-oracle comparison in `PROPOSAL.md` §5.5 no longer tests anything; coverage of missed people unchanged (71.7% / 90.6%).
- **Unexplained, to investigate before `PAPER.md` quotes it:** the displacement oracle fell from **4.20 to 2.58** with the same command. At equal phantom removal (36%) it now costs 10.7% of true positives against 6.9%, and the removal that used to happen at 0.80 m now happens at 0.30 m -- the displacements themselves changed, far more than ~0.14% of wrongly-headed frames should move them.

**Why.** `AGENTS.md` rule 3 and the evidence rules: the paper's exchange-rate tables and the §2.4 numbers the proposal is argued from must be measured on correct odometry before they are quoted.

### A50. After the first FROG result: post-processing, deferred ablations, architecture experiments

**Owner's plan (2026-09-16), in three phases in this order.** Phase 1 explains and cleans up what A43's chain produced; phase 2 runs the §9 ablations that were delayed because each needs a training run; phase 3 tries the alternatives `docs/PROPOSAL.md` §12 kept for experiments, which are now comparatively cheap because each stage has a measured baseline of its own.

**Phase 1 -- post-processing of the FROG result** (each item is cheap, or one short retrain):

| # | item | why, and what it needs |
| --- | --- | --- |
| 1 | ~~replay the oracle on **our** candidates~~ **done 2026-09-16** (`utils/three_horizon_oracle.py`, `checkpoints_three_horizon/oracle_ours.json`) | the ceiling above our own stage 2 is **+6.5 pp at a 2 s bridge** (86.1% -> 92.6%), not the +9.6 pp `docs/PROPOSAL.md` §9 quotes from LFE-Peaks, and **stage 3 takes 17.5% of it** while losing 6,076 of the people stage 2 had found (+2,113 recovered, net -3,963). The memory's +3.2 AP is bought with precision, not recovered recall. Feeds item 2 |
| 2 | ~~find the cause of step 3b's regression~~ **diagnosed 2026-09-16** | **stage-2 weight drift, and the feedback prior never activated.** Stage 2 alone lost 1.24 points (76.92% -> 75.68% with the prior silenced, which is *exactly* its score with the prior active), all of it recall (87.3% -> 84.5%, precision flat at 58.4%). The prior's own weights never left zero — norm 0.049 against the shared columns' 7.08 — so step 3b never tested feedback at all. Selection compounded it: `best.offer` ranks on the *joint* AP, and epoch 1.5 won (78.98%) over epoch 1.0 (78.37%) even though stage 2 was better at epoch 1.0 (val 76.12% vs 75.68%). Drift is diffuse, 2.97% overall, concentrated in `coarse.conv.weight` (5.44%), `head.weight` (5.29%) and the norm biases. **Revised by item 4's duration curve:** ordinary training loses 1.23 pp between epoch 1.00 and 2.00, as much as joint training's 1.24 pp over 1.5 epochs, so "more training" now suffices to explain a loss of this size — not excluded as the joint objective's doing, since step 3b ran at a tenth of the control's rate, but no longer requiring it. Feeds items 3 and 4 |
| 3 | retry joint training | a lower rate for stage 2 than for the from-scratch memory, or stage 2 frozen for the first epoch; more validation windows, placed only on annotated stretches (6 of 16 scored nothing) |
| 4 | regularise stages 1-2 | both step 2 runs peaked near epoch 1 while training loss kept falling — but that decline is **validation-side only**, and the one over-trained checkpoint on disk (`step2_continued`, 0.25 epochs past the best) scores **76.91% on test against step 2's 76.92%**, so test-set overfitting is not yet demonstrated. **Dropout landed 2026-09-16** (owner's suggestion) as `step2 --dropout`: the head's input only, default 0.0 so every result so far is reproduced, `docs/RESEARCH.md` §5.4 intends 0.1; scan mirroring, range jitter and weight decay still open. `step2 --checkpoint-every` landed the same day (also default 0.0) so a run keeps **periodic** checkpoints and not only the best, which is what lets test AP be measured against training duration for the first time — the gap that left "does stage 2 overfit on test?" unanswered. **The 76.92% baseline cannot be reproduced under current code**: its stored args carry no `schedule`, `lr_factor`, `lr_patience` or `min_lr`, so it predates the plateau scheduler, and a lone dropout run would differ from it in two variables. The dropout run is therefore paired with a **control at dropout 0.0 on today's code** (`step2_control` and `step2_dropout01`, launched 2026-09-16), and the comparison is control against treatment. **Control finished: test AP 76.62%** (val best 73.88% at epoch 1.00, stopped at 4.00) against the historical 76.92% — today's code reproduces the baseline within **0.30 pp**, and both runs peaked at epoch 1.00 while the plateau scheduler first cut the rate only at 2.25, so the scheduler did not pick the checkpoint and the gap is run-to-run nondeterminism. **Read any dropout effect below ~0.3 pp on test as noise**; val is noisier still, swinging >1.5 pp between adjacent evaluations. 16 periodic checkpoints (0.25-4.00) are kept, so the duration curve is measurable as soon as the GPU frees. **Dropout 0.1 finished 2026-09-16: test AP 77.32%** against the control's 76.62% — **+0.70 pp**, about 2.3x the measured run-to-run spread, and above even the historical 76.92%. False positives fell by a third (2.653 -> 1.770 per frame) for a small recall cost (FN 0.341 -> 0.379) — but **the operating point is itself noisy**, and this is suggestive only: the two dropout-0.0 runs differ by 0.74 FP/frame (1.910 against 2.653), nearly as much as dropout's own 0.88. The shuffled-history penalty is firmer evidence, since the two dropout-0.0 runs agree at 1.30 and 1.18 pp against dropout's 2.00 pp, so the regularised network leans *more* on temporal evidence. **Validation showed none of it** (73.88% -> 73.93%, inside val's own >1.5 pp swing), one more reason not to rank stage 2 on this val sample. One pair at one seed, so it stays **provisional**: the confirming seeds are deferred to the shared confirmation stage below (owner's call 2026-09-16) rather than run bespoke for dropout, since dropout is one model option among many and all of them are judged the same way. **Duration curve, the first on test** (`duration_curve.py`, `checkpoints_three_horizon/duration_curve.json`): control 75.26 / 76.62 / 75.39 / 73.90 / 73.04% at epochs 0.25 / 1 / 2 / 3 / 4 and dropout 75.72 / 78.16 / 75.93 / 74.78% at 0.25 / 1 / 2 / 3. **Stage 2 overfits on test**, peaking at epoch 1 and losing 3.58 pp by epoch 4 as precision climbs 48.8 -> 73.5% and recall falls 87.8 -> 76.5%; dropout lifts every point (+0.46, +1.54, +0.54, +0.88) **without flattening the decline**. Val mis-selected the dropout run: it chose epoch 0.75 (77.32% test) where epoch 1.00 scores 78.16% |
| 5 | decaying rate for step 3a | its cosine schedule spans `--max-epochs` (100) per chunk length, so it hardly decays in the 9-37 epochs a stage runs; plateau schedule as in step 2 |
| 6 | remove the time budget | owner's call: runs are bound by epoch budgets and early stopping anyway (`--max-hours` and its checks, every step) |
| 7 | report the remaining §9 **metrics** (these are measurements, not ideas; the architecture ideas are phase 3 below) | AP at 0.3 m as well as 0.5 m; milliseconds per frame for each arm (none of the 2026-09-16 ablations has one, which AGENTS.md rule 2b treats as an incomplete answer); the 72,533 person-free frames; people standing still, reported separately |

**Phase 1b -- why stage 3 underperforms** (owner's call 2026-09-16, after item 1: "stage 3 underperforming is exactly what we should put an effort into ... do the slots preserve enough information? Is the slot invalidation rule enough? Are there enough slots?").

*Already settled, do not re-run.* **Slot count is not the limit**: step 4 scored 64 / 128 / 256 slots at 80.12 / 80.20 / 80.10%, a 0.1-point spread, and A48's occupancy work found the memory never fills at 256, so **no slot is ever evicted for capacity** — which is also why `docs/PROPOSAL.md` §5.5's rule-versus-oracle comparison stopped testing anything. The invalidation rule therefore cannot be failing through eviction; only through **retirement** (value fades to `floor` in `fade_time_s` = 10 s, then the slot dies) or **association** (mutual-nearest within a fixed `gate_m` = 0.6 m of the velocity-predicted position).

*Found by accident 2026-09-17: beams with no return may be casting false votes.* After the loader clamp was removed, `evaluate_calibration` still zeroed the probability of every non-finite beam, as it always had, but only now did FROG test's `+inf` beams reach it. Scored that way, seed 0's folded checkpoints rose from 78.03 to **78.25%** (no-fine; FP 1.508 -> 1.512 at nearly unchanged precision) and from 77.22 to **77.47%** (static; FP 2.776 -> 2.741). The accident was then fixed: decoding now anchors on sanitised ranges, reproducing the old scoring. **What:** a dataset-agnostic decoding rule, where a beam at the range limit, with no surface to stand on, casts no vote. **How:** evaluation only, no retraining. Score the existing no-fine and static checkpoints at all three seeds with and without the rule. Six evaluations at ~4 min each; +0.2 pp needs all three seeds to agree to count. Then decide whether training targets should exclude those beams too. **Why:** it is free at inference time, and it fits the owner's goal of handling missing returns inside the model, not the loader.
**Result 2026-09-17** (`duration_curve --silence-limit-beams`, same weights, current decoding; `checkpoints_three_horizon/silence_limit_beams.json`):

| run | plain | silenced | change |
| --- | --- | --- | --- |
| default (dirs `step2_no_fine`, `confirm_nofine_s1/s2`) seed 0 / 1 / 2 | 78.11 / 77.92 / 76.70 | 78.40 / 78.28 / 77.09 | +0.29 / +0.36 / +0.39 |
| no-temporal (dirs `*static*`) seed 0 / 1 / 2 | 77.22 / 77.04 / 76.40 | 77.60 / 77.40 / 76.68 | +0.38 / +0.36 / +0.28 |

**All six improve, by +0.28 to +0.39 pp (mean +0.34).** The spread across runs is irrelevant here, because each pair scores the same weights and scoring is deterministic. The gain is recall, not precision: FN per frame falls 0.007-0.012 while FP barely moves. Votes from beams with no surface were pulling vote-grid peaks away from real people rather than creating phantoms. **Adopted as the default 2026-09-17 (owner's call):** `voting_beams` silences beams at the range limit in candidate decoding (stage-3 caches and the joint step) and in evaluation; the scoring option and `duration_curve --silence-limit-beams` were removed once the experiment was done. Three-seed test means under the default decoding: **default 77.92, no-temporal 77.23** (previously recorded 77.51 / 76.82 under the old far-range anchoring without the rule). **Superseded the same day by fresh controls: a tie, not a loss.** The owner asked for the controls to be rerun on current code (`step2`, default flags, seeds 0-2, `checkpoints_three_horizon/default_s*`). They scored **78.19 / 76.64 / 75.13 (mean 76.65, sd 1.53)** against exclusion's 76.99 (sd 0.60), a difference of +0.34 in exclusion's favour and well inside the spread. The earlier −0.93 came from comparing with old controls that were 0.2-2.0 pp higher at the same seeds. **Seeds do not reproduce runs on this machine** (`memory/interpreting-evaluation.md`), so the per-seed pairing below was invalid. **Decision under the pre-registered rule: a tie, training stays `train`.** Exclusion brings no benefit, so **`--limit-beams-in-loss`, `training_beams` and `calibration_loss` were removed 2026-09-17 (owner's call)**. The `exclude_limit_s*` checkpoints still load, since the option held no parameters. **Also open:** current code's default-architecture mean (76.65) sits 1.27 below the old controls' 77.92. That is inside noise at three seeds (sds 1.53 / 0.70), but worth watching: the only training change is target anchoring beyond 10 m. **Stage 2 for the slots session:** `default_s0` (78.19%, trained on current code, no folding).
**Excluding those beams from training too: run 2026-09-17, first read as rejected** (`step2 --limit-beams-in-loss exclude`, default architecture, seeds 0-2, 1,177,923 parameters; `checkpoints_three_horizon/exclude_limit_s*`). Test wp-AUC under the default decoding was **77.68 / 76.72 / 76.58 (mean 76.99)**, against the controls' 78.40 / 78.28 / 77.09 (mean 77.92). It is lower on every seed, by −0.72 / −1.56 / −0.51 (mean −0.93, paired t = −2.9, df 2). The pre-registered rule keeps training unchanged. One caveat: the controls were trained before `anchor_ranges` existed, with beams beyond 10 m targeted at their true range, and that confound is small next to a 0.9 pp loss. A plausible reading is that limit beams are useful negatives: they teach the network what open space looks like even though they never vote. Removed as a tested option that was not needed (`memory/rejected-ideas.md`).

*New suspect, found 2026-09-17 while checking noise: stage 3's time step.* FROG's timestamps come in bunches (TODO A49), so the stamped interval is under 0.1 ms on roughly a quarter of frames. Stage 3 clips it to `MIN_DT_S` = 1 ms. `manage_slots` then updates velocity by `velocity_smoothing * ((match_xy - position) / step - velocity)`. Dividing a few-cm candidate wobble by 1 ms gives tens of m/s; after the 0.1 smoothing that is still a kick of several m/s. The corrupted velocity moves the predicted position, which gates association, and `slots.velocity.norm` is an input to the learned value update. **Not yet measured.** **How:** log slot speed against annotated walking speed on the test split. Then re-run step 3a with a fixed `dt` of 25 ms, or with stamps smoothed over a short window, and compare. **Why:** it could explain part of the association and rescoring loss below, and the fix does not need retraining stage 2. Under the owner's goal of a model that accepts raw data, the model should not trust a raw stamped interval.

*The open question.* Item 1 showed stage 3 **loses 6,076 people stage 2 had already found** while recovering 2,113, so the failure is not forgetting — something actively destroys good detections. The prime suspect is the rescoring itself: `candidate_logit = logit(score) + rescore_gate * rescore(...)` is a **signed** additive correction on every candidate, with nothing constraining it to only raise confidence, so it can buy precision by pushing marginal true positives under the 0.3 threshold.

| # | item | what it needs |
| --- | --- | --- |
| 1 | split the 6,076 lost people into *rescoring pushed it under threshold*, *association failed*, or *the slot had retired* | evaluation only, no library change: the delta is `candidate_logit - logit(cand_score)` from values already returned, and `MemoryOutput.slots` carries value, age, since-match and alive. **Answered on the whole test split 2026-09-16** (`utils/memory_diagnosis.py`, `checkpoints_three_horizon/memory_diagnosis.json`): of 6,119 lost, **99.4% to rescoring and 0.6% to anything else**. A live slot sat within the gate for **100%** of them — median value 0.793, matched 0.0 s earlier — so neither the 0.6 m gate nor the 10 s fade is implicated, and phase 1b items 3 and 4 drop down the list. The same pass also **corrects the oracle's coverage figures**: counting per frame without linking trajectories gives 134,196 / 153,655 = 87.3% for stage 2 and 130,131 / 153,655 = 84.7% for stage 3, exactly the reported recall, so the oracle's 86.1 / 83.5 were depressed by its position-key collisions rather than by any metric difference (`memory/dataset-properties.md`). **The reframing matters more than the split:** the people stage 3 *kept* were pushed down harder than the ones it lost (median delta **-3.292** logits against **-1.497**). The head is not selectively suppressing true positives, it is **shifting the whole score distribution downward** — which leaves rank-based AP better (+3.2) while silently decalibrating a fixed 0.3 threshold. So the 6,076 may be mostly a threshold artefact rather than lost capability. **Test that before changing anything in the model**: score stage 3 at the threshold matching stage 2's false-positive rate and see whether the loss survives. Caveat on the head's apparent boosting of non-covering candidates (+0.853 median on the full split): one-to-one matching counts a duplicate on a real person as covering nobody, and the set is dominated by near-floor candidates where the nudge is meaningless. **Threshold test, the decisive one (2026-09-16, `--stage3-threshold 0.15`):** scoring stage 3 at 0.15 while stage 2 stays at 0.3 collapses the loss from 6,119 to **609** and lifts coverage from 130,131 to **139,055 of 153,655 (90.5%)**, *above* stage 2's own 134,196 (87.3%). The "lost" people were overwhelmingly a **threshold artefact**, not lost capability, exactly as the distribution-shift reading predicted — so the fix is calibration, not architecture, and phase 1b items 3 and 4 stay parked. **Now priced, and it is not a win.** At 0.15 stage 3 emits 5.250 detections per frame against stage 2's 4.590, running at **2.474 false positives per frame against stage 2's 1.910** — the recovered people are bought *above* stage 2's own error rate, which is a different operating point rather than a gain. The counting is trustworthy: at 0.3/0.3 it reproduces the published 1.910 and 1.293 to three decimals, and the same pass reproduces 134,196 / 130,131 / 6,119 / 2,054 exactly. So the "threshold artefact" reading survives only in the weak sense that the people return at a lower threshold; the strong claim that the memory recovers 5,468 of them does not. **Answered by bracketing, and favourably.** At **0.22 stage 3 strictly dominates stage 2**: 135,520 people covered against 134,196 (+1,324, recall 88.20% against 87.34%) at **1.767 false positives per frame against 1.910** — better on both axes at once. At 0.20, where false positives essentially match (1.925 against 1.910), it covers **136,643, or +2,447 people** (+1.59 pp recall). So the memory genuinely recovers people net of its own error rate, and the entire apparent recall loss at 0.3 was **decalibration**: the operating threshold was set for stage 2's score distribution, not stage 3's. **What follows:** report stage 3 at its own operating point rather than stage 2's, and treat "the memory buys precision for recall" as an artefact of the shared 0.3 threshold (`memory/performance-log.md` footnote 5). The fix is calibration of the rescore head, not slots, association or architecture |
| 2 | compare the rescoring delta on true-positive candidates against non-matching ones | measures the recall-for-precision trade the head is actually making, and whether a one-sided or per-slot gate would do better |
| 3 | does slot state survive what matters? | a retired slot loses everything — `state *= alive`, and a refilled slot gets fresh `spawned` state — so a person re-detected after a 10 s gap starts blank. The oracle says little is recoverable past 10 s on FROG, so this may not bind here; check before changing it |
| 4 | ablate the hand-coded association | `manage_slots`' 0.6 m mutual-nearest gate governs value, spawn and retire, while the *learned* cross-attention governs only rescoring. The hand-coded half has never been ablated |

**Phase 2 -- the heavy §9 ablations**, one training run each (the evaluation-only ones are done: SORT on our candidates, 64/128/256 slots, stage 2 alone, and step 3a itself is "no feedback"):

| ablation | what it tests |
| --- | --- |
| remove the fine temporal convolution | the calibration horizon (~0.08 s). **Screening launched 2026-09-16** as `step2 --fine-lags 0`, all else matching the control. **The ablation is verified by the parameter count**, 1,179,011 against the control's 1,181,059 — exactly the 2,048 that `Conv1d(96, 32, 1)` -> `Conv1d(32, 32, 1)` predicts, so the flag reached the model rather than being silently ignored. Check every later ablation the same way. **Result: test AP 78.03% against the control's 76.62%, +1.41 pp** — about 4.7x the measured run-to-run spread, and above even the dropout run's 77.32%. Removing the calibration horizon *helps*, and it cannot be a capacity effect, since the ablation takes 2,048 parameters away (0.17%) and still gains; it also won despite selecting a later checkpoint (epoch 1.50 against 1.00) where the duration curve says test AP is already falling. The shuffled-history penalty drops 1.18 -> 0.65 pp, so the network still reads history, but only through the coarse convolution. **Validation could not see it**: 73.76% against the control's 73.88% on the 2,000-frame sample, and 73.88% against 73.79% when every val frame is scored — a tie either way, against a 1.41-point difference on test (`memory/interpreting-evaluation.md`). One seed, screening only — confirm before quoting, because it contradicts `docs/PROPOSAL.md` §5.3-§5.4's premise that the ~0.08 s horizon carries signal, and DR-SPAAM's published +1.9 AP is credited to a comparable window |
| remove the coarse temporal convolution | the ~1.2 s motion scale. `step2 --coarse-lags 0`, expected parameter count **979,843**: the conv is `Conv1d(128*5 + 4*3, 128, 3)` = 250,496 params against 49,280 ablated (cross-checked against the drift inspection, which reported `coarse.conv.weight` at 250,368 weights plus 128 bias). **Capacity confound, recorded before the result:** this strips 201,216 parameters, 17% of the network, where the fine ablation strips 2,048 (0.17%), so a fall in AP cannot be read as the ~1.2 s span alone. The capacity-matched control is `--coarse-lags 0 0 0 0 0` — five taps on the current frame, keeping the conv 640 inputs wide against 652, which removes the temporal information without the capacity. **Result: test AP 77.01% against the control's 76.62%, +0.39 pp** (val best 73.01% at epoch 0.75; parameter count 979,843 as predicted). **The pre-registered confound does not bite**: it was registered against a *fall*, and this is a rise — losing 17% of the parameters and still gaining is the harder direction, so capacity cannot explain it away, and the matched control is not needed to defend this result. The shuffled-history penalty collapses 1.18 -> 0.18 pp, leaving the network almost static: with the coarse conv gone its whole history is the fine conv's 3 frames (0.08 s). Training is also 3.5x faster, 46 s against 165 s per epoch. One seed, screening only |
| cap every decay time at 1 s | whether the long horizon carries anything |
| sensor-frame x, y instead of beam-relative offsets | the input description and cache validity |
| then repeat the headline runs on the `balanced` split | §10 step 4: `official` first, then `balanced`. **Never mix partitions** (checked 2026-09-16): `balanced`'s val is 11-36 + 10-31 and **11-36 is half of `official`'s train**, so validating there while training on `official` would score the model on its own training data. Stay inside `balanced` for train/val/test — its test is 16-41 + 15-53 and holds `frog_16-41` *whole* precisely so the official-test subset is extractable from the same evaluation pass, giving the balanced number and a published-comparable one in one run. Its val is 121,329 frames with 19.8% empty against official-val's populated-only sample, which is the weakness that mis-selected the dropout checkpoint (owner's point 2026-09-16: use `balanced` where `official` fails, but never quote a published baseline on it). State the deviation when quoting: published baselines trained on 11-36 + 12-43 where `balanced` trains on 12-43 + 14-57, so the test recording is held out for everyone but the training sets differ. **Owner's call 2026-09-16: re-run the temporal ablations here, at the very end, as an artefact test.** On `official` they came out backwards — removing the fine convolution gained +1.41 pp and removing the coarse one +0.39, while validation ranked the arms in nearly the opposite order — and neither the literature nor the architecture explains it (`CHANGELOG.md`). If the same ablations behave differently on `balanced`, what we measured is an `official`-split artefact rather than a property of the model. `balanced` tests both suspects at once: **two** test recordings against `official`'s one, and a val split of 121,329 frames with empties against a 2,000-frame populated-only sample. **Sharper still, and the reason to prefer this over a plain replication:** because `balanced` holds 16-41 whole, one evaluation pass yields both subsets, so the ablation effect can be compared on 16-41 against 15-53 **with identical weights, varying only the frames**. If it appears on 16-41 alone it is pinned to that recording; if on both, the finding is real and `official` was never the problem |

**The four temporal arms, screened 2026-09-16** on `official`, seed 0, dropout 0.0, every one parameter-count verified:

| arm | test AP | against control | shuffled-history penalty | parameters |
| --- | --- | --- | --- | --- |
| control, both convolutions | 76.62% | — | 1.18 pp | 1,181,059 |
| fine only (`--coarse-lags 0`) | 77.01% | +0.39 | 0.18 pp | 979,843 |
| **coarse only (`--fine-lags 0`)** | **78.03%** | **+1.41** | 0.65 pp | 1,179,011 |
| static, neither | 77.22% | +0.60 | **0.00 pp** | 977,795 |

The static arm scores **identically ordered and shuffled** (77.22% / 2.776 FP / 0.348 FN both ways), which is the behavioural proof that the ablation is complete: with no temporal taps, shuffling the past cannot change the answer.

**This is not "less information beats more" — read the arms against the static one, not against the control.** The **coarse** convolution adds **+0.81 pp** over no temporal information at all (78.03 against 77.22, about 2.7x the run-to-run spread), so the ~1.2 s horizon earns its place. The **fine** convolution alone lands at 77.01 against static's 77.22, a gap *inside* the noise floor, so nothing is claimed for it either way — but adding it on top of the coarse one costs **-1.41 pp**, the largest effect measured. **The mechanism was then measured, and it is the width, not the temporal content.** A capacity-matched arm, `--fine-lags 0 0 0`, taps the current frame three times — same 96-channel width and parameter count as the control, no alignment (lag 0 skips it), no history at all, confirmed by its shuffled-history penalty falling from 1.18 to 0.43 pp — and scores **76.39%**, with the control's 76.62% (-0.23, inside the noise floor) rather than with no-fine's 78.03% (-1.64). Three identical copies of the current frame cost as much as three real past frames. That looked like the layer's width doing the harm, but the 5-tap arm refuted it: wider still at 160 channels, it scored **78.66%**, the best of all. So the 0.08 s window carries no usable signal, while a 0.15 s window does (see the fine-convolution section below). All single-seed, and the `balanced` re-run above is the artefact test.

**Why the fine convolution hurts is still open, and one explanation is already ruled out** (owner's question 2026-09-16: the 3-frame layer still contains the current frame, so how can it be worse?). It is a strict superset — `Conv1d(96, 32, 1)` on `[x_t, aligned x_{t-1}, aligned x_{t-2}]` represents the static `Conv1d(32, 32, 1)` by zeroing the 64 lagged channels — so it cannot be worse in expressivity, only in what training finds.

*Ruled out: alignment artefacts.* `tests/test_three_horizon.py::test_the_fine_cache_is_almost_hole_free_at_its_own_lag` pins `filled.mean() > 0.98` at this lag's own motion (~4 cm, 0.8°), so the tap is close to a clean copy, not a hole-punched resampling. Beam-index rounding, nearest-wins collisions and field-of-view losses all stay small at 0.8°. Do not re-propose this without new evidence.

*Also ruled out, 2026-09-16: the content of the lagged taps.* The hypothesis was that over 0.08 s the taps' only new content is ~5 cm of motion, at the sensor's noise scale, which the optimiser then fits to training-recording specifics. **The width-matched control refutes it.** `--fine-lags 0 0 0` taps the current frame three times — identical 96-channel width and parameter count to the control, no alignment invoked (lag 0 skips it), no temporal content at all — and scores **76.39%**, sitting with the control's 76.62% (-0.23, inside the 0.30 noise floor) rather than with no-fine's 78.03% (-1.64, far outside). Its shuffled-history penalty collapsed from 1.18 to **0.43 pp**, confirming the taps really do carry no history, and the parameter count matched 1,181,059 exactly.

*Width was proposed next, and the 5-tap run refuted it.* Three identical copies of the current frame cost as much as three real past frames, which suggested the harm was the layer widening from 32 to 96 inputs. The pre-registered test was that `--fine-lags 0 1 2 3 4`, wider still at 160 channels, would land at or below the control's 76.62%. **It scored 78.66%, the best of every arm** (shuffled 77.51%, penalty 1.15 pp; val best 74.30% at epoch 1.00; 1,183,107 parameters as predicted). A wider layer won, so width is not the explanation.

*What the five arms do support, which is the owner's original reading.* Three taps over 0.08 s score about the same whether they are real past frames (76.62%) or copies of the present (76.39%), so at that span there is effectively no temporal signal. Five taps over 0.15 s — DR-SPAAM's window — score 78.66% and lean on history (1.15 pp penalty), so at that span there is. **Still unexplained:** why three taps with no usable signal score ~1.5 points *below* one tap (no-fine, 78.03%) instead of matching it. Single seed throughout: 5-tap's edge over control (+2.04) is ~7x the 0.30 pp spread, its edge over no-fine (+0.63) only ~2x.

*Two next arms (owner's call 2026-09-16).* **What:** `--fine-lags 0 2 4`, three taps over the same 0.15 s as the 5-tap arm. **How:** one step-2 run, flag only, 1,181,059 parameters (identical to the control). **Why:** it separates tap *count* from *span* at fixed 96-channel width. A result near 78.66% means span is what matters, near 76.62% means count is. **Result: 77.61%** (shuffled 77.32%, penalty 0.29 pp; 1,181,059 parameters as predicted; val best 73.26% at epoch 1.50). It lands almost exactly midway: +0.99 above the control and -1.05 below 5-tap, each about 3x the 0.30 pp spread. So **span and count both contribute** and neither prediction held. It still trails no-fine (78.03%); only 5-tap beats a single tap, and only marginally. **Both skip and 5-tap were trained and scored with clipped history**, found afterwards from their checkpoints. `--clip-frames` defaults to 35, which is `max(coarse_lags) + max(fine_lags) + 1` only for three fine taps. With a fine lag of 4 both needed 37, so at the oldest coarse tap the two oldest fine frames were copies of the clip's first frame. Training and test scoring agree with each other, but streaming inference (stage-3 candidates, the robot) reads the real frames. **Only the control (76.62%) and no-fine (78.03%) are clean.** Measured streaming cost per tap count, stages 1-2, RX 9060 XT, against a 38.2 ms frame budget: 1 / 3 / 5 / 7 / 9 taps take 6.54 / 7.92 / 9.27 / 11.70 / 12.28 ms. Each tap adds 90 KB of fine cache and 2,048 parameters, and training clips need 33 / 35 / 37 / 39 / 41 frames. **Re-run at the correct clip lengths, 2026-09-16/17:** 5 taps at `--clip-frames 37` scored **78.56%** (shuffled 77.45%, penalty 1.11 pp; 1,183,107 parameters), within noise of the clipped run's 78.66%, so the clipping had not moved that result. 7 taps at 39 scored **77.14%** (shuffled 75.90%, penalty 1.24 pp; val best 73.61% at epoch 0.50), **1.42 below 5 taps**, about 4.7x the spread. **5 taps is the fine horizon's best setting and 7 is past it**, one seed, pending confirmation. **Confirmation launched 2026-09-17 (owner's call):** seeds 1 and 2 for 5 taps and no-fine, added to seed 0 for three seeds per arm, run with the new `clip_length` default (37 and 33). The 3-tap control is not re-run: it trails 5 taps by 1.94 pp, far outside any spread seen. **Pre-registered decision:** the arm with the higher three-seed mean becomes the `--fine-lags` default. If the two means sit within the arms' seed-to-seed range, it is reported as a tie, no-fine is recommended for cost (6.54 against 9.27 ms per frame), and the owner decides. Dropout's +0.70 pp was screened on 3 taps, so it is re-screened on whichever default wins rather than confirmed now. **Result 2026-09-17: a tie, and the screening noise floor was wrong.** Three seeds each: no-fine 78.03 / 77.89 / 76.60 (mean **77.51**, sd 0.79), 5 taps 78.56 / 77.32 / 76.63 (mean **77.50**, sd 0.98). By the pre-registered rule this is a tie. No-fine is recommended for cost (6.54 against 9.27 ms per frame), and **the owner decides the default**, which stays at 3 taps until then. The per-run spread of ~0.9 pp is about three times the 0.30 pp used for screening, so two single runs must differ by ~2.5 pp to count. Every single-seed architecture ranking of 2026-09-16 below that size is retracted (`memory/interpreting-evaluation.md`), including 5 taps over the 3-tap control (+1.94) and 7 taps under 5 (−1.42). **Next, the owner's call on 2026-09-17: static at seeds 1 and 2**, which puts the memoryless arm on three seeds beside no-fine's 77.51% mean. The owner's reading: if static ties, something is fundamentally wrong with the temporal hypothesis, because motion within ~1.5 s is not negligible by definition. The runs use `--clip-frames 35` to match the seed-0 static run exactly. **Seed 1: 76.92%**, identical ordered and shuffled, with 977,795 parameters. **Seed 2: 76.31%**, also identical ordered and shuffled. **Result: a tie by the pre-registered rule.** Static at three seeds is 77.22 / 76.92 / 76.31 (mean **76.82**, sd 0.46) against no-fine's 78.03 / 77.89 / 76.60 (mean **77.51**, sd 0.79); the two seed ranges overlap. Per seed, no-fine was ahead on all three (+0.81, +0.97, +0.29), but seeds do not reproduce runs on this machine, so that pairing is not evidence (retracted 2026-09-17, `memory/interpreting-evaluation.md`); only the means compare. **By the owner's pre-stated reading, a tie means the temporal hypothesis is in trouble:** 1.2 s of history buys at most ~0.7 pp here. Next is the owner's call. **Done 2026-09-17:** `--clip-frames` now defaults to `clip_length(fine_lags, coarse_lags)` = `max(coarse_lags) + max(fine_lags) + 1`, so a lag change cannot leave the clip short again (3 new tests; 453 pass). At 9 taps (lag 8, 0.305 s) the fine convolution reads exactly the coarse convolution's first past tap, so the two scales stop being separate. The second arm is **the denoiser replacing the fine convolution**, plan confirmed by the owner 2026-09-16: window `0 1 2 3 4`, tolerance 0.3 m, the convolution replaced. **Implemented** as `step2 --fine-lags 0 --fine-denoise 0.3`. `project_ranges` re-measures past returns from the current pose, which `align_beams` does not, since it only carries a value to its new beam. `CalibrationState.denoise` caches raw ranges and poses; `detach_joint` and `duration_curve.network_for` carry the setting. **Deferred:** `load_split` and `memory_diagnosis.py` build default networks for candidate caches, so a stage-3 run on a denoise stage-2 would silently drop the denoiser. Wire that in before any such run. **Result: 68.87% wp-AUC, 9.16 below no-fine** (78.03%), the same network without the denoiser. Shuffled history gives 43.84%, a 25 pp penalty. Val best was 64.15% at epoch 0.75, and the parameter count of 1,179,011 matched. **The cause is the outlier rule, not a bug.** Projection is sound on real data: consecutive-frame disagreement is 1.30 cm raw against 1.35 cm projected, with 1.24 cm of robot motion per frame. But the rule replaces **10.32% of readings near annotated people against 3.45% elsewhere, 3.0x**. A leg crossing a beam appears in only a few of the 5 frames, so the median is background and the leg is discarded as an outlier. Loosening the tolerance makes the ratio *worse* (3.5x at 1.0 m), so retuning cannot fix it. **Owner's call 2026-09-16: denoising dropped.** A plain convolution is cheap and already gives the best result (5-tap, 78.66%), so the one-sided variant (never replace a reading nearer than the window median) is not being pursued; it stays recorded in `memory/rejected-ideas.md` as what would reopen the idea. **Open:** `denoise_ranges`, `project_ranges`, `CalibrationState.denoise` and `--fine-denoise` are now unused, and `memory/coding-guidelines.md` says to generate nothing that is not used. **Removed 2026-09-17**, confirmed by the owner. `denoise_ranges`, `project_ranges`, `CalibrationState.denoise`, `--fine-denoise` and their 16 tests are gone. `align_beams` is back to its original inline body, and 453 tests pass. **Closed 2026-09-17 (owner's call): the fine convolution is removed from the architecture, a tested hypothesis that turned out not to be needed** (`memory/rejected-ideas.md`). Its job was denoising, and the anomaly checker found nothing to denoise: jitter sits at the sensor's rated accuracy, far below the 0.5 m match radius (`memory/noise-structure.md`). Five taps tie no fine convolution at three seeds. Missing returns are handled by one rule instead, the owner's "simple normalisation": `sanitize_ranges` with a configurable `--max-range-m`. Gone: `align_beams`, beam alignment in `CausalTemporalConv`, ranges in `TemporalCache`, the validity input channel, `CalibrationState.fine` and `--fine-lags`. `clip_length` is now `max(coarse_lags) + 1`. Old checkpoints with one fine tap (no-fine, static) fold exactly into the new stem (`fold_legacy_calibration`). Multi-tap checkpoints no longer load: the 3-tap control, 5- and 7-tap, skip, width, dropout, `step2_calibration.best.pth`, and the stage-3 runs built on it. Their numbers stand as recorded, and stage 3 is rebuilt on the new default anyway (plan item 3).

**Two follow-ups (owner's proposal 2026-09-16, scheduled *before* the `balanced` retrain rather than after it — owner's call the same day)**, on the reading that the fine horizon's job is **denoising, not information** — which is this project's own framing of the calibration horizon, "repeated scans of one surface average into a clean shape" — so the ablation says it fails at its one stated job:

1. **Five frames instead of three.** `--fine-lags 0 1 2 3 4` spans 0.153 s at 26.2 Hz, almost exactly DR-SPAAM's 5-scan window and its reported +1.9 AP. **Needs no code**, the flag exists. Because lags are arbitrary integers, separate the two things DR-SPAAM conflates: tap **count** (averaging power) against **span** (reach) — `0 1 2 3 4` (5 taps, 0.15 s) against `0 2 4 6 8` (5 taps, 0.31 s) against today's `0 1 2` (3 taps, 0.08 s). Caveat: DR-SPAAM's 5 scans *replace* a single scan, while ours sit on top of a coarse convolution already spanning 1.22 s, so a null result here does not contradict their number.
2. **Replace the convolution with parameter-free denoising**: per beam, over the aligned last N frames, reject outliers and take the **most recent non-outlier**. It targets the stated purpose directly, adds no parameters and gives the optimiser no shortcut — which is what the four-arm result suggests this horizon should be. Design notes: operate on **ranges before `beam_features`**, not on stem features, so all 11 channels are computed from a cleaned scan; treat alignment holes (~2% at this lag) as *missing*, not zero; keep most-recent-non-outlier rather than a median, which would smear moving returns.

**Phase 3 -- architecture experiments**: every alternative `docs/PROPOSAL.md` §12 deferred to experiments, listed in full at the owner's request 2026-09-16. Each is now comparable against a measured baseline for its own stage, and is judged by the screening-then-confirmation protocol above.

**Stages 1-2**, input and calibration:

| idea | instead of | status |
| --- | --- | --- |
| first layer gathering neighbours within a fixed metric width (DROW's adaptive window) | distance and distance-minus-median channels | open; §2.6 puts it at ~1 point |
| cache raw scans and recompute each frame, or scans for the first layer and features deeper | caching motion-corrected features | open |
| shifting part of each layer's channels in time (TSM) | the fine temporal convolution | open; the fine horizon itself is now measured (5 taps best, 7 running) |
| coarse span, e.g. 9 frames between taps (~1.5 s) | 8 frames (~1.3 s) | open. **Interacts with the fine horizon:** the fine ceiling is the coarse convolution's first past tap, so changing the span moves it |
| DR-SPAAM's running template with local attention | dilated causal convolution | open; only worth it if cache alignment proves costly |
| plain residual U-Net, or widening 2 | ConvNeXt blocks, widening 4 | open |
| range anchors, as LFE-PPN (+10.2 points over LFE-Peaks at 8-10 m) | per-beam score and offset | open; aimed at far-range recall |

**Stage 3**, object memory:

| idea | instead of | status |
| --- | --- | --- |
| every beam as a candidate | the 64 best after merging | open |
| 720 slots exchanging only with nearby slots | 256 slots | open; 64 / 128 / 256 already measured within 0.1 point |
| fixed decays (LRU, S5), minGRU, or GRU | selective diagonal recurrence | open |
| Hungarian matching with pseudo-identities, or slots predicting their own position (MOTR) | learned soft attention | open; FROG has no identities, and pseudo-identities are unreliable beyond 10 s |
| sensor coordinates | room coordinates | open |
| confirmation over several frames, or a higher creation threshold | creating a slot from any candidate above 0.01 | open |
| **where memory plugs in**: feedback at the detector input, or no feedback | feedback at the bottleneck | open. **Effectively untested:** step 3b's prior never left zero, so the bottleneck variant was never exercised either |
| a learned grid over the room (Deep Tracking on the Move) | memory addressed per object | open; hand-built grids failed (§2.4), a learned one is untested |

**Training and evaluation**, what is trained and how it is judged:

| idea | instead of | status |
| --- | --- | --- |
| everything trained together from the start | phases A, B, C | open |
| an added identity loss from pseudo-identities | detection and slot-value losses | open |
| memory always fresh | carried across chunks with 20% random resets | open |
| LFE ported to PyTorch as a published baseline | the same model with memory reset every frame | open; only if reviewers require a published baseline |
| `balanced` first | `official` first | **decided**: `balanced` re-runs at the end, as the artefact test |

**Order and bookkeeping:** phase 1 items 4-6 change training settings, so run them before phases 2-3 and state the settings with every later number; A49's cross-dataset runs use whatever phase 1 settles on.

**The documentation refresh waits for the end of phase 3** (owner's call 2026-09-16). `docs/SHOWCASE.md` is already stale on two points — it calls the joint-training regression "under investigation", which item 2 closed, and it presents step 3b as a test of feedback when that run's prior never left zero and contributes 0.00 points — and phases 2-3 will stale its cost table and finding bullets further, so the pages are revised once at the end rather than after each result.

**How every model option is judged** (owner's call 2026-09-16: dropout is one option among many, so one procedure covers them all). Single runs cannot resolve the effect sizes at stake — two identically configured runs differ by ~0.3 pp of test AP and 0.74 false positives per frame, against dropout's own +0.70 pp — and with ~14 options across phases 2 and 3, chance alone would produce several convincing-looking winners. So:

1. **Screening.** Each option gets one run at seed 0 with all other settings shared, reported as test AP with precision, recall, FP and FN per frame and the shuffled-history penalty, against the shared dropout-0.0 control (76.62%). Phase 2's "one training run each" *is* this stage.
2. **Confirmation.** Only options moving test AP by more than the current spread estimate advance; they are re-run alongside the control at two or three further seeds, and reported as a mean and spread. Only what survives is adopted into later settings.

Every quoted number states how many seeds it rests on, and single-seed numbers are labelled provisional — dropout's +0.70 pp included. Because confirmation re-runs the control anyway, the spread estimate tightens as a by-product, so the threshold sharpens as it is used and is never paid for separately.

### A49. The three-horizon detector on all three datasets, parameters stated for each

**Owner's goal (2026-09-15): an adaptive model.** Train it and report numbers on FROG, DROW and JRDB, each with an explicit table of the parameters used. Order: FROG first (A43 steps 3a -> 3b -> 4, running), then DROW, then JRDB (§C).

**What the current code ties to FROG** (checked 2026-09-15):

- `utils/train_three_horizon.py` hard-codes `FROG_Dataset`, `frog_laser_angles(720)` and `FRAME_PERIOD_S = 1/26.2`, which sets chunk lengths, validation windows and the first frame's time step.
- Stage 2's temporal lags are counted in frames (fine 0, 1, 2; coarse 0, 8, 16, 24, 32): 0.08 s and 1.22 s at 26.2 Hz, but 0.16 s and 2.5 s at DROW's 12.7 Hz. `docs/PROPOSAL.md` principle 9 asks for seconds.
- The resampling of every scan to one angular spacing (`docs/PROPOSAL.md` §5.3) is not built, so a kernel spans 0.25° per beam on FROG and 0.5° on DROW.
- Stage 3 already uses the real time step, and its slots live in metres.

**What differs per dataset** (`memory/dataset-properties.md`):

| | FROG | DROW | JRDB |
| --- | --- | --- | --- |
| beams / field of view / spacing | 720 / 180° / 0.25° | 450 / 225° / 0.5° | to measure: the loader assumes 541 / 270°; the official DR-SPAAM repo merges two lasers into 1091 points / 360° |
| scan rate | 26.2 Hz | 12.7 Hz (not 10.0: timestamps rounded to 0.05 s) | to measure |
| annotated scans | every one | every 5th (5.0%) | to measure |
| people per annotated frame | 3.06 | 0.81 | to measure |
| classes | pedestrian | pedestrian, wheelchair, walker | pedestrian |

**Dataset anomaly checker (owner's call 2026-09-17).** **What:** a general-purpose script that reports unexpected range values per dataset, with counts and examples: zeros, NaNs, infinities, values outside the sensor's rated range, and large jumps within 3 frames. **How:** thresholds come from each laser's technical characteristics; it runs on FROG, DROW and JRDB; the plan goes to the owner before implementation. **Why:** it tests the hypothesis that FROG's scans are clean enough that temporal denoising has nothing to remove. The 1.30 cm median consecutive-scan disagreement offered in support is **biased low**: the loader rewrites missing returns to 10.0, so a beam missing in both frames scores zero. The checker must therefore read raw scans from before that clamp (`memory/gotchas.md`). It answers the same question for the datasets A49 moves to next. **Owner's framing, 2026-09-17:** "denoiser" means every denoising mechanism, and each dataset should use whichever one the kind of noise found calls for. So the checker reports by the categories that decide the choice:

- missing-return encodings and sentinels call for validity masking of the input, not denoising;
- single-frame A–B–A spikes call for a spike filter that replaces the middle reading only when its neighbours agree;
- range jitter that is large next to the 0.5 m match radius calls for temporal averaging, i.e. the fine convolution, already switchable with `--fine-lags`;
- none of these means no denoising.

A mechanism is implemented only when some dataset needs it. A–B–B steps are real motion and must never be filtered. The median denoiser was removed on 2026-09-17, and `memory/rejected-ideas.md` records what a rebuild would need.

**Checker spec, approved 2026-09-17.** `utils/scan_anomalies.py --dataset {frog,drow,jrdb}`, tests first, run only when no training is live. It reads raw scans (FROG `h5["scans"]` before the clamp; DROW CSV). It reports:

- NaN, ±inf, zero and negative counts per beam and per recording;
- readings outside the rated range, from the sourced spec table in `memory/noise-structure.md`;
- frequent exact values, which reveal sentinels;
- A–B–A spikes against A–B–B steps on beams valid in all three frames, binned by range; a spike is a middle reading more than 0.3 m from both neighbours while the neighbours agree within 0.1 m.

Examples accompany each category. Output is JSON plus a printed report grouped by the mechanism categories above.

**Remove the FROG loader's missing-return clamp (owner's idea, 2026-09-17).** **What:** `frog_dataset.py` rewrites every non-finite range to 10.0, a value inside the sensor's rated range. Drop the rewrite, or write a sentinel such as -1. **How:** the three-horizon detector needs nothing more, because `beam_features` already masks non-finite and non-positive ranges into its validity channel; check `align_beams` for NaN first. The LFE, DROW and DR-SPAAM baselines read FROG through the same loader and have no validity channel, so they need the sentinel or their own handling. The encoding is chosen after the checker shows what FROG actually contains. **Why:** the network should learn to handle invalid returns rather than depend on a preprocessor that makes them look real. **Blast radius:** every stage-2 checkpoint and candidate cache was trained on clamped input, and the baselines' FROG numbers were measured on it, so this means retraining. Do not edit the file while a run imports it. Details are in `memory/noise-structure.md`.

**Owner's extension (2026-09-17): heal instead of flag.** **What:** remove the `valid return` channel from `beam_features`. Replace an invalid reading with the same beam's most recent valid reading from the last few frames. Where the beam is invalid in all of them, fall back to a default (-1, 10 m or another value). **Precondition:** know what "invalid" means per dataset. FROG is annotated only to 10 m, and the authors' detector cuts at 10 m, so its non-finite readings may be "beyond 10 m" rather than dropouts. DROW encodes no return as 29.96 m (11.9 % of readings). Healing suits dropouts; for open space it fabricates a surface, and a constant is the faithful choice. **How:**

1. Add to the checker, for invalid readings: run length in frames, run length in beams, the largest finite value, and the fraction of invalid readings within 0.5 m of an annotated person.
2. Pick per dataset: a constant, or healing with fallback. Healing must re-project past readings through odometry (`align_beams`) and use a one-sided rule (`memory/rejected-ideas.md`), or a moving leg leaves a stale copy behind.
3. Test healing without a flag against the current input at 3 seeds each, since the noise floor is ~0.9 pp per run.

**Why:** one encoding, learned rather than hand-tuned per dataset, if the data supports it.

**Checker result, 2026-09-17 (`memory/noise-structure.md`):** in both datasets invalid readings are open space: long runs, wide stretches, and rarer on people than elsewhere. Neither dataset has sensor noise worth a spike filter or temporal averaging. **Healing is not supported by the data**, since at most 6-10 % of invalid readings are flickers. FROG writes no return as `+inf` in test and the extras but as **61.0 m** in train/val; DROW writes 29.96/29.98/29.99.
**Implemented 2026-09-17 (owner approved, the range limit made configurable as `--max-range-m`),** together with the removal of the fine convolution (A50). On FROG at 10 m the new input equals the old one, so no retraining is needed for this change; the 3-seed comparison proposed below is dropped as a comparison of identical inputs. **The proposal as approved:** one dataset-agnostic rule in `beam_features` and nothing in the loaders. A reading that is non-finite, non-positive, or at or beyond the model's range limit becomes "nothing within range", i.e. the range limit itself. The FROG clamp is removed, and the validity channel is dropped as the owner proposed. This handles inf, 61.0 and real 45 m walls identically without knowing the dataset. DROW's 29.96 needs no special case as long as the range limit is under 29.96 m. The range limit becomes a stated per-deployment parameter; 10 m drops 7.3 % of DROW's labels. Changing the input channels means retraining stage 2, so compare against the current input at 3 seeds. The baselines keep their own preprocessing.

**FROG runs at 40 Hz, not 26.2 Hz (found 2026-09-17 by `utils/scan_anomalies.py`).** **What:** FROG's scans are stamped in bunches, roughly 38 ms, 38 ms, then under 0.1 ms. The median interval gives 26.2 Hz; the mean over linked intervals is 25.0 ms in all five files, matching the sensor's 25 ms per scan, and no two consecutive scans are identical. **Blast radius:**

- `train_three_horizon.FRAME_PERIOD_S = 1/26.2` sets chunk lengths and validation windows, so every one is 1.53x shorter in real time than its flag says.
- The coarse lags span 0.80 s, not 1.22 s. The "~1.5 s" in the temporal hypothesis is really ~0.8 s.
- Stage 3's time step comes from the stamps, clipped to 1 ms, and the slot velocity divides by it (see A50, stage 3 time step).
- `evaluate._frame_period_s` takes the median, so every FROG `fp_per_s` is 1.53x too low.
- `motion_analysis` (`FRAME_HZ = 26.2`), `persistence_map` (`--hz 26.2`, `SPANS`), `slot_budget`, `horizon_sweep`, `three_horizon_oracle` and `informational_capacity` (`DT_NATIVE_S`) are affected, along with the tests that pin 26.2.
- The 38.2 ms real-time budget is really 25 ms; the measured 6.5-12.3 ms per frame still fits.

**How:** one shared helper for the rate (mean over linked intervals), used everywhere; re-run the DROW and FROG analysis rows; restate the FROG seconds in `memory/dataset-properties.md`. **Why:** every FROG figure in seconds, and every seconds-based flag of the three-horizon trainer, is off by 1.53x.

**DROW runs at 12.7 Hz, not 10 Hz (found 2026-09-17).** **What:** DROW's timestamps are rounded to 0.05 s, so the median interval reads 0.1 s. The per-file mean is 12.4-12.7 Hz, matching the paper's 12.5 Hz. **How:** `utils/motion_analysis.py`, `slot_budget.py`, `horizon_sweep.py` and `three_horizon_oracle.py` all take `1 / median(diff(time))`. Switch them to the mean over each unbroken sequence, and check that FROG's 26.2 Hz survives the same change. Then re-run their DROW rows, whose durations in seconds are 1.27x too long. **Why:** every DROW figure stated in seconds is affected, including the lag conversions in the table above.

**Decide before training beyond FROG:**

- One model trained per dataset with the same settings in physical units, a FROG-trained model evaluated zero-shot, or both; each result says which.
- DROW labels every 5th scan (20x fewer annotated frames than FROG), which thins stage 3's supervision; densifying by interpolating between real annotations is already a research item (`memory/dataset-properties.md`).
- AP does not compare across datasets (person density differs 3.8x): each dataset's result is set against that dataset's published baselines only.
- **`--max-range-m` per dataset.** FROG is annotated only to 10 m; DROW has 3,952 labels (7.3 %) beyond 10 m and 475 beyond 15 m. A 10 m model silently loses those people on DROW. Choose the limit per dataset, state it in the parameter table, and score beyond-limit labels separately.
- **Four DROW labels are errors**, at 316-716 m (`results/scan_anomalies_drow.json` lists them; all in train). Drop or ignore them in training targets before DROW training.

**Follow-ups from the anomaly checker and the missing-return change (2026-09-17):**

- **Move the baselines' clamp into their own preprocessing.** `FROG_Dataset` still defaults to `missing_return_m=LEGACY_MISSING_RETURN_M` (10 m), so no baseline number moves. The owner's view is that loaders should not preprocess. Put the clamp where LFE, DROW and DR-SPAAM prepare their inputs, flip the loader default to `None`, and re-run one baseline to confirm its number is unchanged. That baseline must see a train/val file too, because 61.0 m is never clamped.
- **Odometry-compensated jitter.** The approved spec allowed restricting jitter to near-stationary frames; it was not built, so jitter figures are upper bounds that include the robot's motion. Build it only if a dataset's jitter comes near the match radius.
- **Run the checker on JRDB** once it is on disk. It first needs the two LMS500s' specs in `SENSORS` and a reader.

**Deliverable per dataset:** AP on its standard test split beside its published baselines, and a parameter table -- temporal lags in seconds and frames, chunk lengths, angular spacing, slot count, epoch budgets and early stopping, checkpoint selection.

### A43. Own architecture: three memories inside one detector -- DESIGN SETTLED 2026-09-14, STEP 0 DONE 2026-09-14, training budgets awaiting owner

**What.** One streaming model, `detections, state = step(scan, odometry, state)`, in three stages (`docs/PROPOSAL.md` is the full design; §3 the three temporal horizons it is built on, §5.2 its building blocks, §12 the alternatives deferred to experiments):

1. **Input** -- per-beam range, range minus local median, valid mask, and offsets to beams `i+-1`, `i+-2` in beam `i`'s local frame.
2. **Normalising memory** -- a 1D U-Net of ConvNeXt-style blocks with LFE's global max aggregator; a fine causal temporal conv per beam (~0.1 s) and a coarse causal space-time conv at the bottleneck (~1.3 s); feature caches re-aligned for rotation and translation; DROW-style head; top-64 proposals below threshold included.
3. **Object memory** -- K = 256 slots in the odometry frame; cross-attention association, synchronisation across slots, selective diagonal recurrence in real time steps; one accumulated value per slot -- any proposal above a 0.01 floor spawns, the weakest slot is replaced only by a clearly more certain candidate, slots retire when their value fades; rescoring, coasting, and **feedback into stage 2's bottleneck**.

**Why.** Every stage built after a frozen detector failed on a hand-set association radius or on place-keyed memory, and none beat the merge radius (`docs/PAPER.md`, "Model extensions"). This design moves memory inside, keys it by object, learns association and acts before the threshold. Owner's outline and priority A > B > C; every component answered question by question with the owner on 2026-09-14.

**How.** `PROPOSAL.md` §10:

0. ~~**Interface and caches, tests first**~~ **done 2026-09-14**: `library/follow_the_drow/detectors/three_horizon.py` (`beam_features`, `align_beams`, `align_sectors`, `CausalTemporalConv` with `TemporalCache`), 21 tests; streaming equals the clip frame by frame; checked on the real FROG test recording (`CHANGELOG.md`). Value-ranked eviction keeps its existing test in `utils/slot_budget.py` until step 1 needs it in the model.
1. **Built 2026-09-14, owner-approved to train:** stage 3 alone on frozen LFE-Peaks proposals, on ROCm -- `ObjectMemory` in `three_horizon.py`, `utils/train_three_horizon.py step1`. Gate: beats SORT on the same proposals and the merge radius at matched recall.
2. **Built 2026-09-14, owner-approved to train:** stages 1-2, phase A, on ROCm -- `CalibrationNetwork`, `train_three_horizon.py step2`. Gate: A35's two criteria (single-frame path at LFE's published AP; shuffled history hurts).
3. Stage 3 on the new stage 2, then joint with feedback.
4. Ablations (`PROPOSAL.md` §9).

**Training launched 2026-09-15 08:37**: step 1 then step 2, sequentially, logs and `progress.txt` in `checkpoints_three_horizon/`. Both smoke runs passed first -- step 1 end to end (val candidates 81.8% AP on 4 windows, test 66.6%), step 2 at ~0.21 s per batch of 32 clips (~12 min per epoch, 1.3 GB GPU) with validation rising 52.7% -> 59.2% in 0.03 epochs and shuffled history already costing 2.5 pp on a 100-frame test sample. About 7 hours were lost overnight to a finished smoke run hanging at exit (started before the `os._exit` fix) with nobody watching; the runs are now watched for stalls.

**That run stopped at 10:28 and is not resumable.** The 1 s stage finished its full 40 epochs (best val AP **68.8%** at epoch 37 against **60.7%** for the untouched candidates); the machine then went to sleep and rebooted at 10:29:54, one epoch into the 10 s stage -- no error, no out-of-memory or bugcheck record. That epoch (66.6%) had already overwritten the 68.8% checkpoint, because each chunk-length stage restarted "best" along with patience. Fixed and tested 2026-09-15: `BestCheckpoint` keeps the run's best whatever epoch or stage produced it, each stage also keeps its own best file, stages resume from the run's best, the script keeps Windows awake while it runs, and step 1 defaults to 100 epochs per chunk length (owner, 2026-09-15). **Decided 2026-09-15 (owner): go straight to our own network -- step 2, then 3, then 4; step 1 (stage 3 on LFE-Peaks) is dropped.** Step 2 relaunched 13:24 as a detached process (a background task of the agent session dies with the session), watched for evaluations, errors and stalls.

- **Step 3a (phase B)** -- run the trained stage 2 over every train/val/test frame, decode candidates with **DROW's vote grid** (owner's choice: step 3's AP stays comparable with step 2's) and pool each candidate's decoder features from the beams whose votes land within the grid's collect radius, cache them (float16), and train stage 3 from scratch on them with step 1's loop and budget (100 epochs per chunk length, patience 8).
- **Step 3b (phase C)** -- joint streaming training with feedback: 2 prior channels at the bottleneck drawn from the slots' predicted positions (slot value, person probability), zero-initialised so the trained stage 2 is unchanged until trained; TBPTT chunks through `CalibrationNetwork.step` and `ObjectMemory.step`, loss on both; ~6 GB per 10 s chunk at one stream; budget as approved (3 epochs).
- **Step 4 -- only the evaluation-time ablations for now** (owner's choice): SORT on step 2's candidates instead of stage 3, 64 / 128 slots without retraining, shuffled history. The retrain ablations of `PROPOSAL.md` §9 wait for step 3's result.

**Steps 3a, 3b and 4 built 2026-09-15, tests first** (405 tests pass): `CalibrationNetwork` returns its decoder features and takes `prior_channels` (zero-initialised; `load_calibration_weights` loads step 2's checkpoint into it unchanged); `render_prior` draws live slots into bottleneck sectors; `ObjectMemory(feature_dim=...)` takes candidate features; `train_three_horizon.py step3a / step3b / step4`. Choices made while building, to confirm: step 3b trains **one stream** over **10 s** chunks (streams in a batch must start together, and ~6 GB per chunk), evaluates every **0.25** epoch (not 0.125) on **16** val windows, because the joint evaluation streams the whole network and decodes every candidate on the CPU; its learning rate is 1e-4 for both models together. **Validation cadence and learning rate approved by the owner 2026-09-15.** GPU memory, owner's concern: the step-3b smoke run peaked at 1.0 GB for a 2 s chunk at one stream, so a 10 s chunk should need ~5 GB against the ~10.7 GB one process gets -- to confirm on the first real evaluation line, with 5 s chunks as the fallback. **Chain queued to run unattended after step 2 ends cleanly**: step 3a -> step 3b -> step 4, stopping at the first failure. Smoke runs passed 2026-09-15 on one recording each: step 3b end to end (training, validation, test; 1.0 GB for 2 s chunks) and step 4 (SORT at `min_hits` 1/3/5, 64 slots, whole detector). They caught and fixed three failures a real run would have hit: validation windows with no annotated frame (scoring now returns undefined metrics), no checkpoint when every validation was undefined (the last weights are tested, with a note), and a crashed ROCm process hanging at exit (always `os._exit`; the chain launcher also stops a process that lingers 3 min after its log says it finished). Step 3a's candidate caches take ~24 ms per frame to build, ~2.4 h for all 365,542 frames. **Step 2 result (2026-09-15, 14:24): test AP 76.9%, history shuffled 75.6% -- both gates pass** (above LFE's published 65.6% / 69.2%; shuffling costs 1.3 pp, against 0.5 pp for this project's earlier SpaceTimeCNN). 1.91 FP / 0.39 FN per frame at 0.3, against LFE-Peaks 1.76 / 0.63 and LFE-PPN 5.03 / 0.38. Best val AP 73.8% at epoch 1.0; early stopping at epoch 4.0 while training loss kept falling (0.183 -> 0.076), i.e. overfitting at the undecayed rate. **Step 2's learning rate never decayed** (owner's question 2026-09-15): cosine over the 20-epoch budget left it at ~95% by epoch 2.75, while val AP had stalled since epoch 1.0 (73.8%) and early stopping was due near epoch 4. **Owner's call: continue from the best weights with a plateau schedule** -- rate x0.3 after 4 evaluations (1 epoch) without improvement, floor 1e-5, starting at 3e-4 -- written to `checkpoints_three_horizon/step2_continued/`; plateau is now step 2's default (`--schedule`, `--init`). The chain was stopped before step 2 ended and relaunched to run the continuation first, then steps 3a -> 3b -> 4 on its weights. **Known waste, to fix later:** 6 of step 3b's 16 validation windows fall on unannotated stretches (val is 62.8% annotated; recording 0's first annotated frame is 9,233), so they cost evaluation time and score nothing. **Continuation result (2026-09-15, 15:12): no gain -- stage 3 builds on the first run's weights.** Val AP 73.0% at its first evaluation (epoch 0.25), then down every evaluation to 69.2% at epoch 3.25 while training loss fell 0.132 -> 0.036; the rate cuts to 9e-5 and 2.7e-5 did not stop it. Its best weights test at 76.9% / shuffled 75.6% (FP 1.43 / FN 0.49 per frame), the first run's AP at a different operating point. A checkpoint guard compared the two runs' best val AP (73.0% < 73.8%) and put the first run's weights in the continuation's place, but only 4 s after step 3a had started reading it, so step 3a was stopped and the chain relaunched on `checkpoints_three_horizon/step2_calibration.best.pth` directly. **Lesson:** a continuation's `BestCheckpoint` starts empty, so it must be seeded with the initial weights' val metric, or the chain must pick the better checkpoint before launching the next step, not race it. **Future experiment:** stages 1-2 overfit the 120k training frames within ~1 epoch at any tried rate -- try augmentation (scan mirroring, range jitter), weight decay and dropout before any longer schedule. **Step 3a crashed 7 min into its candidate cache (2026-09-15, 15:20) on a vote-grid bug, now fixed:** two neighbouring cells whose blurred votes tie exactly are both maxima in `votes_to_detections`, every vote goes to the nearer one, and the other became a detection averaged over no votes -- NaN position and probability -- which broke the Hungarian candidate match. Fixed at the source (a peak with no votes is skipped), with `tests/test_vote_decoding.py` (reproduced by a random vote search, watched failing first; 411 tests pass). NaN detections pass every distance gate, so `_prec_rec_2d` could have counted them as true positives; re-scoring step 2's test set found 2 among 800,856 detections and AP 76.9227% -> 76.9218%, so the step 2 result stands. The chain was relaunched on the fixed decoder at 15:36. **Step 3a's budget, as run (2026-09-15):** `--max-hours 6` counts from the start of training across all chunk lengths, not per chunk length (`train_memory`'s `start`, and the break after each stage). The 1 s stage stopped early at epoch 37 (1.5 h), the 10 s stage at epoch 9, and the 30 s stage is cut by the total at about epoch 9, while still setting new bests (77.32% val at its epoch 5). **Owner's calls (2026-09-15), for after the current chain:** (1) **remove the time budget** (`--max-hours` and its checks) from every step of `train_three_horizon.py`: runs are bound by their epoch budgets and early stopping anyway. (2) **Give step 3a a decaying rate**: its cosine schedule spans `--max-epochs` (100) per chunk length and so hardly decays in the 9-37 epochs a stage actually runs -- the same undecayed-rate pattern step 2's continuation addressed; a plateau schedule as in step 2 is the obvious candidate. **Step 3a's test pass ran out of host memory (2026-09-15, 22:40), after training finished:** `evaluate` gathered 32 whole test recordings at once (up to 6,796 frames x 64 candidates x 32 features, 1.66 GB for the features alone) with the machine near its commit limit; validation windows are short, so training never hit it. Fixed: frames are gathered in 512-frame slices with the slots carried across (`tests/test_train_three_horizon.py::TestEvaluate`, sliced equals whole; 412 tests pass). `evaluate_joint` already reads one frame per step. The checkpoints were intact (best: 30 s chunks, epoch 5, val 77.32%), so the saved best is evaluated on test without retraining, and the chain continues with steps 3b and 4 (relaunched 22:48). **Step 3a test result (2026-09-15, 23:09): stages 1-3 reach 80.10% AP on every annotated test frame (every 5th: 80.15%)**, against 76.92% for stage 2's candidates on the same frames -- +3.2 points; false positives 1.91 -> 1.29 per frame, missed people 0.39 -> 0.47, exchange rate 7.6. One run, stage 2 frozen; recorded in `memory/performance-log.md`. **Owner's request (2026-09-16): report precision, recall and the false-positive and false-negative rates beside AP**, so results can be read against the recoverable-recall ceiling of `docs/PROPOSAL.md` §2.5. `score_detections` now returns precision and recall at the 0.3 operating point alongside the per-frame rates it already had, so every result file carries all four. **Still to do:** that ceiling (+9.6 points of recall recoverable, 4.4 below threshold, 14.1 blind) was measured on LFE-Peaks' detections, not ours -- replay the oracle on the three-horizon detector's own candidates before quoting how close we are to it. Step 3b's test pass was already running when the fields were added, so `step3b_results.json` carries only AP and the per-frame rates; step 4 and everything later carry all four, and step 4 then re-measured every variant (candidates, SORT, stage 3 and the joint model at each slot count) with all four, so no separate re-scoring pass is needed. **Step 4 result (2026-09-16, 05:34, after 131 min; chain ALL DONE):** SORT on our own candidates *lowers* AP at every setting (76.66% / 75.73% / 74.92% at `min_hits` 1/3/5, against 76.92% for the candidates alone), while stage 3 reaches 80.20% at 128 slots -- so the gain belongs to the learned memory, not to temporal filtering in general. Slots do not bind (0.1 points from 64 to 256, both variants), and precision is the weak spot: 66.8% at 84.7% recall, against 58.4% at 87.3% for the candidates. Full table in `CHANGELOG.md`. **Step 3b result (2026-09-16, 03:22, after 250 min): joint training with feedback is a regression, 78.84% test AP against step 3a's 80.10%** (FP 1.235 against 1.293, FN 0.546 against 0.470); its stage 2 with feedback tests at 75.68%, *below* the 76.92% it started from, although step 3b's own validation rose from 74.06% to 76.12% over the run. Step 3a's weights remain the best model. **Why it may have regressed, to try before concluding the feedback design is wrong:** validation was 16 windows, 6 of them with no annotated frame, so checkpoint selection had a weak and biased signal (fix: place windows only on annotated stretches, and use more of them); both models trained at one rate (1e-4), so stage 2 was fine-tuned as fast as the memory was trained from scratch -- a lower rate for stage 2, or freezing it for the first epoch, is the obvious next experiment; and 3 epochs may be too few to recover what the early updates cost.

**Choices made while building, to confirm at the review round:**

- *Runs are sequential*, step 1 (6 h cap) then step 2 (24 h cap): 16 GB of RAM and one GPU make parallel runs contend, and a wall-clock cap is only meaningful without contention.
- *Slot bookkeeping matches by mutual nearest neighbour within 0.6 m* (batched on the GPU), where the replay used gated Hungarian; the two agree whenever the scene is unambiguous. Learned matching still decides what the memory believes -- the bookkeeping only decides where a slot sits.
- *Stage 3 validates on 64 evenly spaced 60 s windows after 30 s of warm-up*, not whole recordings: val's three recordings are ~65,000 frames each, which would take 10-15 min per evaluation. Test is scored on whole recordings.
- *Phase A trains without memory priors.* The prior only matters once stage 3 feeds it (phase C), and its corruption settings are still "to tune" (`PROPOSAL.md` §6.4); prior channels are added then, zero-initialised, without disturbing trained weights.
- *Phase A reuses `train.py`'s DROW head pipeline* -- `make_targets`, `compute_loss`, `votes_to_detections` with `evaluate_auc`'s settings -- rather than the proposal's merge-and-top-64 decoding, so its AP is measured the way every other model here was.

**Training budgets -- proposed 2026-09-14, approved by the owner 2026-09-15 with the go-ahead to train.** Whichever limit is hit first stops a run; selection on val wp-AUC / AP, never loss.

| setting | phase A: stage 2, 35-frame clips | step 1 / phase B: stage 3, cached candidates | phase C: joint |
| --- | --- | --- | --- |
| one epoch | one pass over the training frames, ~54 min on ROCm | recordings cut into 10 s chunks at a fresh random offset, ~414 chunks, ~2-3 min | one pass over 10 s chunks, timed at start |
| `max_epochs` | 20 | 40 per chunk length `[1, 10, 30]` s | 3 |
| `eval_every_epochs` | 0.25 | 1 | 0.125 |
| `patience_evals` | 12 | 8 | 8 |
| `min_epochs` | 2 | 5 | 0.5 |
| `max_hours` | 24 | 6 | 24 |
| learning rate | AdamW 1e-3, cosine | AdamW 1e-3, cosine per chunk length | 1e-4 |

*Why:* published budgets span 5 epochs (FROG authors' DROW3/DR-SPAAM) to 100-150 (LFE); a stride-9 FROG subsample scores like the full split, so a pass holds far fewer independent looks than its frames; this project's runs peaked at epochs 3-14 and one early spike once stopped a run, hence `min_epochs`. Stage 3 sees only ~414 independent chunks, so it evaluates often with short patience. *To experiment later* (`PROPOSAL.md` §12 at the review round): epoch definition (full pass vs stride), patience and cadence, selection metric (AP vs exchange rate vs recall at fixed false positives), schedule, chunk lengths, stage 3 batch size.

**Open** (`PROPOSAL.md` §15): owner's review of the whole design; the coarse dilation; `slot_budget.py`'s unexplained 99.9th-percentile spike.

**Answered 2026-09-14, to fold into `PROPOSAL.md` at the review round:**

- **Hardware.** No CUDA. Owner's device order: DirectML, then CPU. ROCm 7.2.1 with PyTorch 2.9 *does* list the RX 9060 XT on Windows (AMD's matrix), but it needs a separate venv (`torch-directml` pins torch 2.4.1). **Installed 2026-09-14 as `.venv` from the root `requirements.txt`; 324 tests pass; ROCm is the device for both stages** (stage 2 30.1 ms per 35-frame clip, stage 3 323 ms per 10 s chunk at batch 4; CPU in this venv is ~10x slower than torch 2.4.1's on stage 3, so not a fallback). Owner confirmed 2026-09-14 that `.venv` is the one environment; `AGENTS.md`, `README.md`, `memory/` and `docs/PAPER.md` updated.
- **Canonical grid.** FROG 720 beams / 180° / 0.25° at 26.2 Hz; DROW 450 / 225° / 0.5° at 10 Hz (`memory/dataset-properties.md`), so `angular_resolution_deg = 0.25`.
- **Training throughput of the sketch** (forward + backward + SGD step; `CHANGELOG.md` 2026-09-14): stage 2 on 35-frame clips 39.6 ms per clip on DirectML at batch 4 against 180 ms on CPU; stage 3 over a 10 s chunk ~1.1 s per chunk on either device at batch 4, and 3x *slower* on DirectML at batch 1. An earlier probe at batch 8 crashed with an access violation; which case is unknown.
- **What history each baseline uses** (owner's question, 2026-09-15). DR-SPAAM `T=5` on FROG uses 5 consecutive scans, about 0.15 s end to end: the FROG paper names no stride, but it trained with the official DR-SPAAM repository (`scan_stride: 1`) on an export that keeps every scan -- inferred from the code, FROG's own config being unpublished. That is the span of stage 2's fine convolution (~0.08 s), inside the calibration horizon (coarse ~1.3 s); LFE-Peaks and LFE-PPN carry no memory at all. So the published temporal gain (+1.9 points) belongs to the calibration horizon alone, and the short and long horizons have no published counterpart on this sensor -- worth stating in §2 and §3 when comparing information used and claiming novelty. Sources: `memory/interpreting-evaluation.md`.

**Measured for it so far**: FROG carries no person identities; slots in use reach 105 at p99 and 164 at maximum (LFE-PPN detections, 10 s retire window); the value-ranked eviction rule, replayed down to 0.01, evicts an annotated person's slot twice in 50,088 frames and matches an oracle and unlimited memory on coverage -- `CHANGELOG.md` 2026-09-14.

### A42. Local-Cartesian input representation -- DONE 2026-09-11

**Take the coordinates, keep the range conditioning.** At matched window width Cartesian ties DROW's depth tunnel (13.39% against 13.56% 1-NN error) and beats LFE's global normalisation by 5.5-6.2 pp, but does not reach `adaptive/centred`'s 11.08%. Full table and the unmatched-control mistake that nearly reversed the verdict: `memory/static-detector-diagnosis.md`, `CHANGELOG.md`.

**What it changes for A39 and A43.** A39's range-centred channel should be built on *Cartesian* input rather than on normalised ranges, and the range conditioning stays -- the two are complementary, not alternatives.

### A32b. The confound that would invalidate every mode comparison: hold training data constant

**The flaw.** The three-mode claim compares *our* architecture against LFE.
But LFE's published weights were trained on `official` data — populated frames only, no empty scenes ever.
If we train our model on `balanced` and compare it against LFE-as-published on `balanced` or `transferred`, **we are measuring two things at once**: architecture (temporal vs static) *and* training data (saw empty scenes vs did not).

That confound would flatter us enormously, and in the most tempting direction: the entire `transferred` false-positive advantage could come from having trained on person-free frames, with the temporal channel contributing nothing.
Given this project's 2026-09-10 history, that is exactly the kind of result that would look like a finding for weeks.

**The fix, and it is not optional: every comparison holds training data constant.**
For each dataset mode, train a **static control on that same mode** and compare against it.
The control should be **LFE itself, retrained** (A32c) rather than something built here: it is smaller and single-frame so it trains faster, and being a published baseline it cannot be accused of having been crippled to flatter us.
`--history-mode frozen` at train time is the cheap fallback if the port stalls.
LFE-as-published then serves only as the `official` reference point, where its training data *is* matched.

| comparison | valid? |
| --- | --- |
| ours (`official`-trained) vs LFE published, on `official` | yes — matched data |
| ours (`balanced`-trained) vs LFE published, on `balanced` | **no** — confounded |
| ours (`balanced`-trained) vs static control (`balanced`-trained), on `balanced` | yes |
| ours (`balanced`) vs ours (`official`), on the same eval | yes — isolates the *data* effect, which is A25's old question |

**Why it matters beyond bookkeeping.** Separating the two effects is itself a result worth reporting: *"how much of empty-scene robustness comes from seeing empty scenes, and how much from architecture?"* is a question nobody has answered for this task, and answering it needs exactly the four-cell design above.

### A32c. Port LFE to PyTorch from its own ONNX graph — the static control, and the foundation for A34

**What.** A32b needs a static control trained on the *same* data as whatever it is compared against.
Owner's proposal, 2026-09-10, and it is better than building one from scratch: **retrain LFE itself**.
It is small (**53,258 / 170,903** parameters, counted from the ONNX initializers, against `SpaceTimeCNN`'s 476,262), single-frame — so no `T`-frame window to build and no `aligned_raw_scan()` per sample, which is real time in this project's loader — and it is a *published, credible* baseline that nobody can accuse us of having crippled.

**The blocker, checked rather than assumed.** The bundled LFE weights are **ONNX inference artifacts**, loaded through `onnxruntime.InferenceSession` (`lfe_detector.py`).
There is no PyTorch module, no training code and no published training repo.
Retraining requires a reimplementation.

**Why that is far safer here than it sounds, and how to keep it honest.**
This project already has one architecture reimplemented from a paper's prose — `Li2FormerDetector` — and it has never produced a trustworthy number, because there were no reference weights to check it against.
LFE is the opposite case: **the ONNX file is a complete architectural specification**, and the published weights exist to validate against.

1. ~~`pip install onnx` and enumerate the graph.~~ **`onnx` is installed and the graphs are read** (2026-09-10). Note it pulls numpy 2.x over this project's `numpy~=1.24` pin — reinstall the pin after (`gotchas.md`).
   What the enumeration already gave, for free: LFE-Peaks is **190 nodes / 53,258 parameters**, LFE-PPN **133 nodes / 170,903** — exact counts replacing the file-size estimates.
   Op histograms: Peaks `Conv x29, Relu x15, MaxPool x2, ReduceMax x4, Sigmoid x1`; PPN `Conv x19, Relu x10, MaxPool x3, ReduceMax x3, Sigmoid x1`.
   The `ReduceMax` nodes are the "global aggregator" `docs/RESEARCH.md` describes, and there is **no BatchNorm op** in either graph — it is folded into the convolutions, so a from-scratch port has to decide whether to reintroduce it.
   Both take `scan` as `(batch, N_beams, 1)`; Peaks emits `(batch, N_beams, 1)`, PPN `(batch, N_sectors, 30, 3)` where channel 0 is **already sigmoided inside the graph**.
2. Build the PyTorch equivalent.
3. **Port the ONNX weights in and check the outputs match.** If a PyTorch LFE loaded with the authors' own weights reproduces the ONNX model's outputs to numerical tolerance, the reimplementation *is* LFE rather than an approximation — and the Li2Former failure mode cannot recur.
4. Only then train from scratch, per dataset mode.
5. Sanity gate before using it as a control: **from-scratch training on `official` must land near the published 64.9% / 66.5%.** If it does not, the control is not a control, and that must be discovered here rather than inside a comparison.

**Dual purpose, which is what makes it worth the effort.**
A34 proposes "LFE's own backbone with a temporal axis added" — the architecture `TemporalUNetDetector` was mistakenly believed to be.
That needs exactly this PyTorch port as its starting point, so the work pays for both the control and the most promising candidate.

**The risk dropped since this was filed.** Two decoder bugs were found and fixed by reading the graph rather than the paper (`CHANGELOG.md`), which is direct evidence that the graph *is* a sufficient specification — the premise this item rests on.
It also means the wrapper around the port is now trustworthy, so step 3's output comparison tests the port alone.

**Cost.** One reimplementation plus a weight-port check, against retraining a 476K temporal model per mode as the alternative control.
The port is bounded work with a hard correctness test at the end, which is the kind this project should prefer.

### A32d. Two residuals from the reproduction check, after the metric mismatch was removed

**Largely resolved.** The reported overshoot was mostly a **column error**: FROG's Table 4 leads with mAP over `d = [0.3:0.05:0.5]`, not AP@0.5, and this project had been quoting that column against its own AP@0.5 numbers (`CHANGELOG.md`).
Against the right column: LFE-PPN **-0.6pp** (innocent), LFE-Peaks **+2.1pp**.

**Residual 1 — LFE-Peaks read +2.1pp high. RESOLVED: it was our merge radius.**
`merge_radius` was `_PERSON_RADIUS` (0.40 m); at **0.30 m** LFE-Peaks reproduces the paper to **-0.5pp at `d = 0.5` and +0.2pp at `d = 0.3`**, inside the harness's own sampling noise. Default changed.

The paper's stated "most common ground truth circle **diameter**" rule (0.8 m, from a median annotated radius of 0.400 m) turned out **not** to apply here, and testing it was how this got found: at 0.8 m LFE-Peaks *loses* ~6pp by merging genuinely distinct people.
The rule belongs to LFE-PPN, whose NMS deduplicates whole person proposals; LFE-Peaks' merge does a different job — "centroids that are close together are interpreted as **legs or part of legs**" — so its scale is the gap between one person's legs, not a person's width.
~~LFE-PPN's own `nms_radius = 0.8` already matched the rule and stays.~~ **Overturned:** calibrating LFE-PPN the same way puts it at **0.30 m** too (69.3 / 63.9 against its published 69.2 / 62.5; at 0.8 m it is ~3.5pp low at both). Two separately post-processed models landing on one radius is better evidence than either alone — the paper's sentence describes design intent, not the value used. Changing it invalidated every LFE-PPN false-positive number in the repo; see `CHANGELOG.md` 2026-09-11.

**Residual 2 — our LFE-PPN is mislocalised, and only `d = 0.3` shows it.** -0.6pp at 0.5 m, **-9.4pp at 0.3 m** (53.1% against 62.5%): detections near enough for a wide gate, too far for a tight one.
It points at the anchor-decode calibration, which recorded its angular convention as weakly identified precisely because AP@0.5 could not separate the candidates (66.6-68.5%). At AP@0.3 they span **28.7-56.4%**.
**Do not re-calibrate by fitting residuals** — that was tried and is confounded, because a metric association radius caps angular residuals at ~`r/d` and fakes a metric-looking convention regardless of the truth (`CHANGELOG.md`).
Settle it in **A32c** instead, by reading the target encoding out of the graph during the weight-port.

**Standing rule this leaves**: calibrate anything positional at `--eval-r 0.3`. A 0.5 m gate is wider than most errors worth finding.

### A33. A defensible ranking test — and it must be validated before it is trusted

**What.** Phase 3 wants to consider many architectures and train few.
That needs a cheap ranking test, and this project has already been burned twice by proxies used before they were checked: a cheap `--subsample 0.2` screening run ranked `dtime=15` below `dtime=1` purely because they converge at different *rates*, and the informational-capacity grid's predictive power is *still* explicitly open (`interpreting-evaluation.md`).

**The rule that makes this defensible: the ranking test must first reproduce the ordering of architectures whose real numbers we already have.**
`SpaceTimeCNN`, `FullScanTCN` and `TemporalUNet` all have measured results.
**A proxy that cannot rank three known architectures correctly cannot rank unknown ones**, and should be discarded rather than explained.

**How — a funnel, cheapest stage first.**

1. **Stage 0, free, seconds.** Hard filters, no training: parameter count in ~100-500K, CPU latency in single-digit ms/frame (LFE-Peaks is 1.76 ms, `SpaceTimeCNN` 12.1 ms), no recurrence, no attention (`rejected-ideas.md`).
   This eliminates most of the field before anything is read closely.
2. **Stage 1, minutes, no training.** Zero-cost NAS proxies at initialisation — the literature this project already cites (Mellor et al., *ICML* 2021; Abdelfattah et al., *ICLR* 2021) — **plus a task-specific probe worth more than any of them here: temporal-order sensitivity at initialisation.**
   Feed a window and its shuffled counterpart to the *untrained* network and measure how much the output changes.
   An architecture that mean-pools across time before any temporal mixing is *structurally incapable* of encoding order and will score ~0 — which is exactly the defect the 0.5pp shuffle result just exposed in `SpaceTimeCNN`, and it is detectable in seconds without training anything.
3. **Stage 2, hours.** Fixed small training budget for survivors, ranked on `balanced` val wp-AUC **and** the frozen-vs-full gap.
   Carry forward anything still improving at the cutoff, per the convergence-rate lesson above.
4. **Stage 3.** Full training for the best two or three.

**Why.** It makes "we tried many and trained few" defensible rather than arbitrary, and stage 1's order-sensitivity probe is a genuinely novel screening signal for this problem — one that directly targets the failure just measured.

### A34. Candidate architecture survey

**What.** Which architectures enter the funnel.
The design space is not ResNet-versus-U-Net.

**State the problem shape correctly**: the input is an `(N_beams x T)` grid with **one** channel; the output is per-beam class plus offset.
That is **1D dense prediction with a temporal axis** — far closer to semantic segmentation and video understanding than to bounding-box detection, whose priors (anchors, box NMS, size-driven feature pyramids) do not transfer.

**What is in use today, named honestly**: `SpaceTimeCNNDetector` is a **ResNet + Inception + Squeeze-Excitation** hybrid — residual skips (2015), parallel multi-scale kernel branches (2015), an SE channel gate (2018).
A 2015-2018 design. **ConvNeXt** (Liu et al. 2022) exists because modernising exactly that lineage — large depthwise kernels, inverted bottleneck, far fewer norm and activation layers — buys real accuracy at equal FLOPs.
LFE already sits closer to that modern shape (depthwise-separable convs at kernels [9, 7, 5], roughly constant width, few layers), which may be much of why ~180K parameters reach 66.5%.

| family | why it might suit |
| --- | --- |
| **SlowFast** | two pathways, slow-with-spatial-detail and fast-with-motion. **Literally the strong-spatial-plus-separate-temporal decomposition, as an architecture** |
| **Two-stream / R(2+1)D** | explicit motion stream, or factorised space-then-time convolutions |
| **HRNet** | keeps high resolution throughout rather than encode-then-decode; beam-level localisation is what this task is scored on |
| **ConvNeXt** | the cheapest "is the current design simply dated?" test |
| **U-Net** | strong on small data; the family LFE is nearest to |
| **DeepLab / atrous** | the shape `SpaceTimeCNN` already has — a fair control for "family is fine, execution is dated" |

**Also worth building regardless: LFE's own backbone with a temporal axis added.**
It inherits a *known* single-frame score, so "does not lose on spatial" is satisfied by construction and any gain is unambiguously temporal.
`docs/RESEARCH.md` records that `TemporalUNetDetector` was believed to be this and **is not** — LFE is not a textbook U-Net, and the "global aggregator" (global max-pool concatenated at every position) that likely does much of its work has no counterpart here.

### A35. Build, rank, train the survivors

**What.** Run the funnel over A34's candidates; full-train the best two or three; report all three dataset modes for each.

**Acceptance criteria, fixed in advance** — both required, or the architecture is not worth keeping:

1. **Does not lose on spatial**: single-frame (or `frozen`) score at least LFE-PPN's **66.5%**.
   `SpaceTimeCNN` **fails this at 63.0%** — below LFE-Peaks' 64.9% as well.
2. **Strictly benefits from ordered time**: shuffled score meaningfully *below* full score.
   `SpaceTimeCNN` **fails this at 0.5pp**.

Failing both is what makes A35 a rebuild rather than a tuning exercise.
Note the one encouraging number in the ablation: **aggregation alone is worth 4.1pp** on this backbone.
If that transfers, bolting an aggregation term onto a backbone that already scores 66.5% single-frame would land near 70% without using motion at all — and a genuine motion term would then be additive on top.

**Why both.** A model that beats LFE only by aggregating frames is not a temporal architecture, and a model that uses motion but starts from a weak spatial baseline is compensating rather than adding.
The claim worth publishing needs both halves.

### A36. Re-triage once phases 2 and 3 report

**What.** Revisit every item above and below, and expect to discard some.
Candidates for deletion depending on results: the `dtime`/`time_frame` sweep entirely (already cut to one point, and pointless if ordering is unused), `--diff-channels` as a fix (may be superseded by an architecture that uses motion natively), and the whole benchmark-comparison framing if `transferred` turns out to be the more meaningful axis.

**Why it is an item rather than an afterthought.** This session has already reorganised twice on new measurements, and both times the reorganisation was worth more than the work it replaced.
Budget for it.

---

## §S STALLED and SUPERSEDED — parked 2026-09-10

Nothing here is abandoned; it is parked with a stated reason and a stated trigger.
The 2026-09-10 reorganisation (§A) redirected the project, and most of what was queued assumed the old direction.
**Do not pick anything up from here without checking whether §A's phase 2 has changed its premise.**

### Superseded — the content moved into §A

| item | went to | why |
| --- | --- | --- |
| A20 empty-scene FP rate | **A31** | it is not a one-off measurement any more, it is the `transferred` mode's metric |
| A24 sparse-scene claim | **A30** | the claim is now the three-mode claim, stated in advance |
| A25 FROG re-partition | **A30** | this *is* the `balanced` mode |
| A26 replace the backbone | **A35** | acceptance criteria carried over verbatim |
| A27 architecture survey | **A34** | carried over, with the problem shape corrected |
| A1 `dtime`/`time_frame` sweep | — | cut to one point, run, stopped at epoch 13; a sweep over an axis the model provably ignores (0.5pp under shuffle) measures nothing. **Do not resume without a reason from §A** |
| A21 temporal contribution | — | **answered.** 67.1% full / 66.6% shuffled / 63.0% frozen / 42.0% zeroed. The result is the reason §A exists; see `CHANGELOG.md` |
| A5 vectorize LFE-PPN's decode loop | — | **dissolved, not done.** The 244-316 ms/frame was never the Python loop: the decoder applied a second sigmoid to a channel the ONNX graph had already sigmoided, so all 3,600 anchors entered an O(n²) merge every frame. Fixed at the source — **1.56 ms/frame** — and there is nothing left to vectorize |
| A30 three loading modes | — | **done 2026-09-10** |
| A31 the metric each mode needs | — | **done 2026-09-10** |

### Stalled — still open, still valid, wrong moment

**A22. Make the architecture use motion (`--diff-channels`, collapse operator, auxiliary motion loss).**
Stalled because §A phase 3 may replace the architecture wholesale, which would make a fix to *this* one wasted work.
*Revive if*: A35 keeps `SpaceTimeCNN` as a candidate, or A32 shows the thesis holds and a cheap intermediate result is wanted before the funnel finishes.

**A23. Cut the false-positive rate (`pos_weight` sweep, DROW negatives).**
Stalled because the `balanced` mode supplies person-free training frames from the *same* sensor and venue, which is strictly better than DROW's different geometry — so half of this item is absorbed and the other half should wait for a model worth tuning.
*Revive if*: A32 confirms false positives are the dominant failure and a quick win is wanted.

**A7. Clean up `scratch_frog_weights/`.** Housekeeping, no trigger; do it when convenient.

**A9. Makefile Python 3.12 check.** Housekeeping, independent of research direction.

**A10. Stop hardcoding `RESULT_CONF` in the ROS node.** Deployment-side; nothing reaches the robot until someone rebuilds the image anyway.

**AL1-AL7 (CI, ruff, shellcheck, hadolint, per-detector smoke tests, markdown strictness).**
All independent of the research direction and all still worth doing; parked only because §A is where the value is right now.
AL6 (a smoke test per detector class) is the one that would pay off soonest, since A35 will add architectures.

---

## §B UPON A FULL RETRAIN SWEEP — **also stalled 2026-09-10**

Both items below assume the current architecture family is what gets retrained.
§A phase 3 may replace it, and §A phase 2 may show the premise does not hold at all, so neither is worth a GPU session yet.
*Revive if*: A35 settles on an architecture and a batch of variants is worth one session.

Batched because each item needs a from-scratch training run, and this project's convention is to batch several detector/hyperparameter combinations behind one GPU session rather than retrain once per idea.

- Also retrain `FullScanTCN`/`TemporalUNet` at real-odometry `dtime=5`, not just `SpaceTimeCNN` (A1 covers `SpaceTimeCNN` only).
- Test a capacity-shrunk `SpaceTimeCNN` variant (`--channels`, `--n-spatial-stages`, higher dropout) sized for DROW's ~17.8x-smaller effective dataset, rather than reusing FROG-scale capacity (`memory/interpreting-evaluation.md`).
- Every existing FROG checkpoint trained with `--align-scans` before 2026-09-14 saw wrongly interpolated headings in the windows touching a ±180° crossing (A48); the next sweep retrains on the fixed loader, and no number from the old checkpoints should be compared with a new one without saying so.

---

## §C BLOCKED — waiting on time or data

**JRDB evaluation for every detector.**
**Unblocked 2026-09-15: the owner has JRDB access. Queued after the three-horizon model's FROG results (A43), as the third dataset of A49; the owner's interest is a new frame rate and venue structure, not only new baseline numbers.**
Before any result: extract into `library/follow_the_drow/include/JRDB-data/`, then measure the real format. `JRDB_Dataset` assumes a DROW-compatible 541-beam, 270° scan (`memory/data-model.md`), never checked against real files, while the official DR-SPAAM repository evaluates on merged front and rear scans of 1091 points over 360°. Its README reports AP@0.5 of 82.9% (DROW3) and 84.9% (DR-SPAAM) on JRDB, above Li2Former's 81.4%, so the published JRDB numbers likely use different protocols (scan merging, split, AP threshold): check each paper's protocol before calling anything comparable. Measure the scan rate too (`memory/dataset-properties.md` has none for JRDB), since the three horizons are set in seconds.
Previously blocked on registration requiring a university email; no public mirror of the 2D-LiDAR portion was found or sought further.

---

## §D LATER

### Do the object slots already cover the coarse horizon? (owner's hypothesis, 2026-09-17)

**What:** at three seeds each, the static stage 2 (no temporal convolutions) ties no-fine (76.82 against 77.51% wp-AUC; TODO A50). The owner's reading: the two horizons may not be different things. Stage 3's slots are updated every frame and hold each object's state over seconds, so they may carry everything the 1.2 s coarse convolution extracts. **How:** compare end to end, not stage 2 alone. Train stage 3 on the static stage 2 and on the no-fine stage 2, three seeds each. If the slot stage closes the gap, the coarse convolution is redundant with memory. What the slots cannot cover: people not yet detected, meaning first appearances and candidates below threshold, where beam-level motion is the only cue. So also compare recall on each track's first frames. **Precondition:** stage 3 must work first (its prior feedback path never left zero, and recall loss was threshold decalibration, TODO A50 phase 1b); otherwise a tie says nothing. **Why:** if confirmed, the architecture loses a module and ~200k parameters, and the "three horizons" become two.

### C++ modernization: compile check, Meson migration, `clang-tidy`, `clang-format`

**What.** Pulled out of the active alignment plan (originally `AL6`) on 2026-09-08: not the current priority since there's no physical robot to deploy against right now, but real work this project will come back to.
Bundles: (1) confirm the C++ side still actually compiles — `.github/workflows/build-lidar-vision-image.yml`'s `build-cpp-static-library` job already runs `sudo make build-lib` on `ubuntu-latest`, but its current pass/fail status hasn't been checked this session, so verify it before assuming new work is needed; (2) migrate the build system to Meson; (3) `clang-tidy` (config plus `compile_commands.json` via `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, or Meson's native `compile_commands.json` output if (2) lands first); (4) `clang-format`, with a `.clang-format` config.

**How.** Scope properly when picked up rather than now — in particular, (2) needs a real decision on scope: `library/cpp_core/` is a plain CMake target and a reasonable Meson migration candidate, but `deploy/follow_the_drow/` is a **catkin** ROS package, and catkin is built on CMake specifically — ROS Noetic has no supported Meson build path, so a full migration likely means "migrate the standalone library, leave the ROS package on CMake," not a uniform switch, unless someone finds a workable catkin+Meson bridge worth the risk.

**Why.** Grouped as one item rather than broken into an AL-style sequence because it's explicitly deferred, not because it's small — real scoping (especially the Meson/catkin question) happens when this is picked back up, per the "don't invent a rule/plan for something not yet earned" principle this documentation layout otherwise follows.

### Small tooling gaps found 2026-09-17

- **ruff is not installed in `.venv`**, so the Python lint step of the development routine could not run on `utils/scan_anomalies.py` or the fine-convolution removal; only `py_compile` and the tests did. Install it, or document the intended runner in `memory/commands.md`.
- **`duration_curve.py --help` prints a `SystemExit: 0` traceback and exits 1.** Its `__main__` wrapper catches `BaseException`. Catch `Exception` or let `SystemExit` through, as the ROCm hang workaround allows.

### Per-track adaptive tracker gating

**What.** `SimpleTracker`'s one fixed `match_radius` cannot work at both fine and wide invocation strides (`memory/rejected-ideas.md`).

**How.** Scale the gate by a track's own estimated speed, or use a proper covariance/Mahalanobis gate.

**Why.** Would reopen the wide-invocation-stride tracker result — a real piece of engineering, not a parameter tweak, so not urgent against A1-A5 above.

### Investigate LFE-PPN's residual DROW gap

**What.** After fixing the padding-value and beam-index-as-angle bugs, `LFEPPNDetector` on DROW measures 24.0% wp-AUC — right at the "maybe still broken" threshold that started that investigation, and a smaller absolute improvement than `LFEPeaksDetector` got from the identical input-side fix despite sharing the same preprocessing.

**How.** `LFEPPNDetector` has already had two other, unrelated bugs found in it (a hardcoded anchor count, a broken recall metric) — that track record is reason enough to suspect a third rather than assume the residual gap is pure cross-dataset transfer penalty.

**Why.** Not investigated further at the time because it wasn't urgent; revisit if `LFEPPNDetector`'s DROW number matters for a later comparison.

### DROW-native accuracy ideas that avoid heavy (DR-SPAAM-style) per-beam preprocessing

**What.** DR-SPAAM's *slowness* comes from ~450 independent per-beam network passes over pre-made cutouts; its *accuracy* advantage plausibly comes from a separate thing, each crop being locally centered/scale-normalized before that pass.
The two are separable, and none of the ideas below reintroduce per-beam-independent inference.

**How**, in the recommended order (#1-#2 first — cheapest, most direct, and each targets an already-verified mechanism rather than an unconfirmed hypothesis):

1. **Beam-index shift augmentation** — randomly roll the whole scan and its labels by a random beam offset each training batch, directly attacking the "beam 200 = the doorway in this room" absolute-position memorization mechanism (`raw_scan()`/`aligned_raw_scan()` preserve absolute beam position end-to-end).
   Zero architecture change, zero inference cost.
2. **Frame-to-frame difference channels** — feed the current frame plus `(T-1)` delta channels (`frame[t] - frame[t-1]`, post-alignment) instead of `T` raw stacked frames, making "temporal" explicitly about motion rather than repeated absolute position.
3. **Local/relative range normalization** — subtract a small local-window mean/median from each beam once, before the single full-scan conv pass; the DR-SPAAM-inspired centering idea without DR-SPAAM's per-beam architecture.
4. **Range/scale jitter augmentation** — randomly scale/offset a whole frame's range values during training, forcing invariance to absolute distance rather than removing that information via a fixed transform.
   Complements #3.
5. **Frozen-backbone fine-tuning from FROG** — an earlier full (unfrozen) fine-tune attempt regressed *below* zero-shot transfer (14.6% vs. 26.7-28.0% wp-AUC on that run, a different, cruder attempt than the `--init-weights` result in `memory/interpreting-evaluation.md`, plausibly catastrophic forgetting on DROW's small training set).
   Freezing the early backbone and tuning only later/output layers is a cheap flag change that might recover fine-tuning's promise without that failure mode.
6. **Class-imbalance check** (lower priority) — verify whether the classification loss already weights the heavily-imbalanced person-vs-background beam classes; a class-balanced or focal loss is a standard, cheap lever if it doesn't.

**Why.** Targets DROW-native accuracy specifically, as an alternative to A1's real-odometry retrain and §B's capacity-shrinking experiment — not started, queued behind those.

---

## §E PROJECT EVOLUTION

### Should any of this project's newer detectors, the tracker, or the real-odometry pipeline ever be wired into the ROS deployment?

Today the robot runs only `DrowDetector`, unchanged in spirit since before this research began (`memory/deployment.md`).
Wiring in `SpaceTimeCNN` (the current accuracy leader) or `SimpleTracker` would need real ROS-node integration work — a new node or a modified `DROW_detector`-equivalent, plus on-robot CPU inference-speed validation, since the training-side ms/frame numbers in `memory/performance-log.md` are batched, not single-frame-real-time measurements.
Needs an owner decision: is on-robot deployment of the research results a goal at all, or does this project's scope end at the research comparison?
