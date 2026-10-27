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

No merged change currently has a different effect on the robot than what's already running: the DR-SPAAM cutout-normalization fix (`CHANGELOG.md`) doesn't touch `DrowDetector`'s own preprocessing (it's a no-op at DROW's own `window_depth=1.0`), and none of the odometry, tracker, or full-scan-architecture work is wired into any ROS node at all — `deploy/takheliped/nodes/drow_detector.py` only ever runs the original `DrowDetector`.
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

### The FROG pre-print list, set 2026-10-10

**The goal that orders this list:** after these, a pre-print on **FROG alone** can be written -- every number in it defensible,
every design choice measured, every baseline fair. The other datasets (**A62**, **A63**) then run in parallel with the writing
rather than blocking it.

**Everything here is FROG, unblocked, and can start now.** One decision is the owner's and gates the longest job:

> **The decision (owner's, one answer, gates ~2 GPU-days).** A51's six single-seed arms -- D2 `--latest-value`, D3
> `--fixed-decay-s 8`, T1 `--no-rescore-gate`, T2 `--schedule plateau`, O1 bounded rescoring, O2 `--coast-weight 3` -- were
> screened on the superseded no-temporal backbone. Either **(a) re-screen them on the adopted recipe** (~5 h each, two GPU-days,
> and the ablation table then describes the model being proposed) or **(b) report them as they stand**, each labelled with the
> backbone it ran on. **(b) costs nothing and is defensible**, because none of the six changed the adopted design; **(a) is what
> a reviewer would prefer** on noticing the backbone moved under the table. Nothing else waits on this.

**Numbers are permanent and the gaps are deliberate.** A finished item's row is deleted once `CHANGELOG.md` carries its entry (the house rule in `AGENTS.md`), but the numbering never shifts: `item 29` means the same item in every commit message, `memory/` page and `docs/` reference. To read what a missing number did, grep `CHANGELOG.md`.

| # | action | blocks the pre-print? | cost | plan |
| --- | --- | --- | --- | --- |
| ~~4~~ | ~~**Refresh the pages the pre-print is built from**~~ **DONE 2026-10-19.** `docs/PAPER.md` was done 2026-10-10; the speed table was rewritten under one protocol (item 29); and `memory/performance-log.md`'s own headline now leads with the frozen model. Four things in it had outlived their configuration: the stages-1-3 cell (85.9% -> **86.0%**, with the with-coast and old-recipe figures kept beside it), the margin over DR-SPAAM (**7.7 -> 10.4 pp**, a sentence that outlived its number by three weeks), the claim that "the code still spells it `three_horizon`" three weeks after the rename landed, and three of our own rows reading "re-measuring" for a `T=5, dtime=10` run that was never going to happen -- a temporal window is not what the project proposes any more | **yes** -- it is a pre-print source table | done | A50's documentation note |
| ~~7~~ | ~~**A59 phase 3b: LFE retrained with its authors' recipe**~~ **DROPPED 2026-10-10 inside the owner's time box.** The port is mechanical except for two blocks whose `Expand`/`Concat` shapes need a shape-inference pass; what the attempt established -- LFE-Peaks' graph, convolution by convolution -- is now `docs/RESEARCH.md` §3.4. The pre-print states the limitation instead: we cannot separate LFE's architecture from LFE's training | -- | dropped | A59 phase 3 (b) |
| ~~10~~ | ~~**A53: rewrite `docs/RESEARCH.md` around TAKHeLiPeD**~~ **DONE 2026-10-19.** Section 5 was titled "Proposed Architectures" and proposed the **three superseded full-scan architectures**; it now proposes TAKHeLiPeD, stage by stage, with a citation audit in the same style -- ConvNeXt and U-Net faithful, DROW's vote head deliberately not novel so the comparison is architectural, Slot Attention credited for the abstraction and not the mechanism, Mamba for the selective diagonal decay and not the block (**and the real-time `dt` term is not in Mamba at all**), two components with no source claimed because there is none. The old sections moved to §6 whole, renumbered 6.1-6.6, with every cross-reference chased: they are the project's first line of work, and the `SpaceTimeCNN` history ablation inside them is why the proposed detector reads one scan | no, but it is the pre-print's related-work and fidelity draft | done | A53 |

**Items 13 and up run in parallel with the writing.** They are the FROG experiments that could still improve the model, none of
them blocking a pre-print, ordered by how well a measurement already motivates them. Each is one training run (~1 h stage 2,
~5 h stage 3), and each slots into a table the pre-print already has:

| # | action | why this one | plan |
| --- | --- | --- | --- |
| ~~15~~ | ~~**A39's locally range-centred input channel**, and the metric-width first layer~~ **ANSWERED 2026-10-18: it costs.** A first layer reading a fixed 1.66 m of arc, resampled per beam from its own range, scored **1.42 pp below** the adopted stage 2 (test AP 80.70% against 82.12%) and **0.21 pp below** the control's floor end to end (85.28% against 85.49-86.35). Tier 0's probe priced the adaptive window at +2.5 pp, concentrated at long range -- but it measured what a *capacity-free* 1-NN classifier gains from scale normalisation, and a trained 977k-parameter network learns that itself. **Second time a Tier 0 prize has not survived a trained model**, after A59's F1 arm. `--gather-width-m` stays in the tree, off by default and bit-identical when off | A39, A50 phase 3 |
| ~~17~~ | ~~**Every beam as a candidate**~~ **CLOSED 2026-10-15 with item 13's leftover, negative** (`memory/rejected-ideas.md`). Its precondition killed the pool-size framing first: over **245,146 frames not one reaches the 64-candidate cap** (median 17, max 59), so `TOP_P` never binds and the 4,068 people with no candidate are the decoder's doing, not the pool's. The decoder was then swept -- three NMS radii and three range-split boundaries, three seeds, at matched false positives -- and **overall recall does not move** (88.0-89.7% across every sane arm). The split is **dominated**, worse at far range than either parent, because the two decoders' scores are on different scales and one global threshold cuts the mixed set unevenly: at 6-8 m split-6m runs the vote grid and scores **83.8%** where the grid alone scores **88.1%**. Geometry redistributes recall between range and crowding; it does not add any. **The adopted `nms-mean` at 0.5 m stays** | A50 phase 3 |
| 47 | **The three-seed freeze of the adopted architecture** -- the last thing before other datasets, and the row that replaces every published number | **STAGE 2 IS ALREADY BUILT (2026-10-27):** `queue_item44a_confirm.py`'s trio won and is trained and averaged, so `queue_freeze.py` shows all six stage-2 jobs `done` and only the three stage-3 arms remain -- **~22 h, not 24**. **The headline is the superseded model's.** 86.01% is the *selective* recurrence on the aggregator-ful stage 2 with the curriculum; the adopted architecture is slots + minGRU with neither **and a 64-wide slot state** (item 49, adopted 2026-10-27), and its one-seed reference is **86.12%**. Items 38, 40, 41 and 49 each tied inside the bar, so the published figures are not wrong -- they measure a model the project no longer proposes. `docs/PAPER.md`'s table and `docs/RESEARCH.md` section 7 cannot be quoted until this runs | **PREPARED 2026-10-25 as `utils/queue_freeze.py`, not launched -- the owner will say when.** `--list` shows the plan without running it: stage-2 seeds 1 and 2 (seed 0 is reused in place as `a69_no_aggregator_s0`), each averaged, then the three stage-3 arms. **Three stage-2 seeds first** (`--no-aggregator`, `--checkpoint-every 0.25`, then `average_checkpoints.py`; ~20 min each, and only seed 0 exists), **then three stage-3 arms** (`--no-exchange --no-learned-match --no-coast --recurrence mingru --chunk-s 10`). **~24 h total, corrected 2026-10-25** -- the earlier ~13 h priced a stage-3 arm at ~4 h, but the simplified baseline ran **7.34 h** (52 epochs, 404 s train + 103 s eval); the 507 s in the changelog is the per-epoch *total*, not train time. Plus ~1.3 h of candidate-cache rebuilds for the two new stage-2 seeds. Then one **matched, interleaved** CPU latency sweep on an idle host (item 29's protocol), and `empty_scene.py` at three seeds for the phantom column |
| 48 | **Calibrate the rescore head** -- **accepted by the owner 2026-10-25, scheduled AFTER the freeze.** It cannot block: done post-hoc (a temperature and bias fitted on val) it is a **monotone** transform of the scores, and AP is rank-based, so it provably cannot move the frozen headline -- it only fills the fixed-threshold columns. Implemented as a *training* loss it would change the weights and would have to precede the freeze, which is the reason to do it post-hoc. **No code yet.** | **This is the repo's own conclusion, never acted on.** A51 phase 1b ends: "the fix is calibration of the rescore head, not slots, association or architecture", and phase 0c still lists calibration as unmeasured. The measurement behind it: the head does not suppress true positives selectively, it **shifts the whole score distribution down** (median -3.292 logits on people it keeps against -1.497 on people it loses), which leaves rank-based AP better while silently decalibrating a fixed threshold. At **0.22 stage 3 strictly dominates stage 2** -- 135,520 people against 134,196 at 1.767 false positives per frame against 1.910, better on both axes -- so the capability is already there and only the operating point hides it. **What was done about it was a reporting change** (score each stage at its own threshold); nothing in the model was ever built | A calibration term on the rescore head so the logits mean what a fixed threshold assumes: a proper scoring rule beside the focal loss, or a learned temperature. **Deployment-relevant in a way AP is not** -- `deploy/conf.env` runs a fixed threshold, and the ROS node cannot pick an operating point per recording. Cheap: one stage-3 arm, no new decode path, and the metric already exists (`own_threshold` / `own_recall` / `own_fp_per_frame` from phase 0c) |
| ~~49~~ | ~~**Stage 3's `dim` and `heads`**~~ **CLOSED 2026-10-27.** `dim 64` **adopted**: ties (86.12% against 86.20%) and takes **28% off the CPU memory phase**. `heads 8` **dropped**: 85.99%, a tie, and a head split changes neither parameters nor cost. `CHANGELOG.md` carries both -- *proposed 2026-10-25, awaiting the owner* | The last shape knobs in the project that have never been varied. Item 45 swept stage 2's four and said so explicitly: "`dim`/`heads` are stage-3 and wait on the cheap half". The cheap half is done and they are still unrun, so the adopted 128/4 is a choice nobody has priced -- and unlike stage 2's knobs, `dim` sets the width of every slot's state, which is the component item 36 proved load-bearing | **RESULT (2026-10-27).** `dim64` test stride 5 memory AP **86.12%** against the control's 86.40% and its own baseline's 86.20%, recall at matched FP **96.5%** against 96.43% -- a tie inside the 1.32 pp bar. **CPU, matched and interleaved on an idle host:** the `memory` phase falls **9.30 -> 6.70 ms (-28%)** and the total **20.06 -> 17.73 ms (-11.6%)**, with p99 margin against FROG's 25 ms period nearly doubling (2.3-2.8 -> 4.3-4.8 ms); the `stage2` noise floor is 0.53 ms, so the effect is 5x the floor, and the floor also shows a 0.31 ms ordering bias *against* `dim 64`. Stage 3's checkpoint drops 706 KB -> **195 KB**. **Recommend adopting**, on E13's principle and the owner's 2026-10-19 call that CPU decides. `heads 8` is converging on the same tie (val +4.62 against `dim64`'s +4.96) and cannot change the decision. **Earlier pause note:**  `heads8` stopped at **epoch 33 of 100** with `step3a_resume.pth` written at 22:45 and **patience 8 of 8 already used**, so it was about to stop and run its ~20 min test pass -- resuming finishes it almost immediately. The queue runner was ended *before* the trainer so the job could not be scored as failed and retried, and the trainer was killed with `Invoke-CimMethod -MethodName Terminate` (`taskkill /F /T` does not work on ROCm processes). **To resume:** `schtasks /run /tn "lidar_prefreeze"` -- the queue skips the four finished jobs and `--resume` picks `heads8` up at epoch 33. **`dim64` is COMPLETE and answered:** test stride 5 memory AP **86.12%** against the control's 86.40% and its own baseline's 86.20%, recall at matched FP **96.5%** against 96.43% -- a tie inside the 1.32 pp bar at **half** the slot-state width. Its CPU figure is the only thing missing and needs an idle host, which there now is. **LAUNCHED 2026-10-26 in `utils/queue_prefreeze.py`** (`dim64` then `heads8`, after item 44(a)'s probe), because anything adoptable must precede the ~24 h freeze or be formally dropped -- item 40's lesson. ~~**`--dim` and `--heads` do not exist as `step3a` flags**~~ **wired 2026-10-26**, with the `object_memory` shape key that stops a resumed `--dim 64` arm rebuilding at 128 and refusing its own weights -- they are `ObjectMemory` constructor arguments only, so this needs the same small plumbing `--iterations` got (argparse, the constructor call, and a `network_shape`/`object_memory` key so a resumed arm rebuilds the right width), plus tests. **Recommend one arm, `--dim 64`**, at ~7.3 h: `--dim 192` is the third attempt at a capacity question already answered flat (A59's W arm, item 19b's widths, slot capacity), and `--heads 8` is the same parameter count with no efficiency upside. **Expect a tie** -- A59's W arm priced stage-2 widths at 1.4 pp with no speed gain, and capacity was flat -- but `dim` is also the one knob that could cut CPU cost, which is the axis that decides deployment, so a tie at `--dim 64` would itself be a win |
| ~~44~~ | ~~**An LFE-style Gaussian heatmap head**~~ **CLOSED 2026-10-27.** (a) **adopted** -- the soft Gaussian beam target replicated at three seeds, **82.79% against 82.10%** with non-overlapping ranges and **18.5% fewer false positives** at the fixed threshold. (b) **dropped** without an arm. `CHANGELOG.md` and `memory/rejected-ideas.md` carry both. Original framing | The vote head is the one piece taken deliberately unchanged, to keep the comparison against DROW3 and DR-SPAAM an *architecture* comparison -- and A59 attributes about **half** this project's lead over LFE-Peaks to it. A heatmap over beams is the published alternative and `make_heatmap_target` already exists. **Re-read 2026-10-25, and the row's old premise -- "the least tested component" -- was wrong:** `memory/static-detector-diagnosis.md`'s decomposition already measured the *decoder* half on fixed weights, and the peak decoder loses heavily. Our own network decoded LFE's way scores **71.26% against `nms-mean`'s 79.88%**, a **-8.62 pp** gap made of -2.84 (centre regression), -3.17 (vote-grid aggregation) and -2.61 (NMS); at 8-10 m and a tight operating point it is **50.3% against 81.6%**. **A head that emits no vote cannot recover the -2.84 at all.** What is genuinely untested is the *objective*, and that same page says so: "the network's +6.15 is a lower bound, since our probability map was trained for votes, not for peaks" | **Split into the two halves the decomposition separates, because they are not equally open -- awaiting the owner's pick (2026-10-25).** **(a) LAUNCHED 2026-10-26 as `soft_label` in `queue_prefreeze.py`** -- `--label-sigma-m 0.3` on a `--no-aggregator` stage 2, averaged, then one `stage2_probe.py` pass against `a69_no_aggregator_s0` (its control, *not* A59's aggregator-ful seeds, which would conflate two changes). ~1 h, and it needs no stage-3 arm unless it wins. **It belongs before the freeze**: it changes stage 2's weights, so adopting it later would invalidate all three stage-2 seeds and every stage-3 arm on them. **(a) the objective, untested -- CODE LANDED 2026-10-25** (`beam_heatmap` and `compute_loss(soft_labels=)` in `utils/train.py`, 19 tests; **no arm queued, nothing trained**, and `step2` has no flag wired to it yet): Gaussian heatmap targets over beams as the per-beam classification loss, **keeping** the vote head and `nms-mean`, so the only change is what trains the probability map. Its support is identical to `make_targets`' positives and its sigma is metric, both so the arm varies one thing. **One thing found while writing it:** a soft target fights the ~50:1 `pos_weight`, because weighted BCE is minimised at `w*t/(w*t+1-t)` rather than at `t` -- so a hard 1 is closer to the weighted optimum than the Gaussian, which is a reason this arm might tie and worth saying before it runs. This is the half the decomposition calls a lower bound, it needs no new decode path, and it is the cheaper code. ~~**(b) the published head**~~ **DROPPED 2026-10-27 (owner), answered without an arm:** trained *for* a Gaussian target the `peaks` decoder improves +2.81 pp (72.54% -> 75.35%) and still trails `nms-mean`'s 82.78% by **7.4 pp** -- it loses on the objective that favours it, and 2.84 pp of its deficit is centre regression a voteless head cannot recover. `memory/rejected-ideas.md` carries it. Original plan: `--head heatmap` with a peak decoder and no vote regression, as the row originally said -- the full LFE parameterisation, whose decoder half is already measured at -8.62 pp on fixed weights. Worth one arm under the owner's run-everything-once rule, but expected to lose, and it should be reported as a confirmation rather than a candidate. **The owner picked (a) first, then (b) as a confirmation (2026-10-25).** (a) isolates the one unmeasured variable, and if the heatmap objective does not improve the map under the *favourable* decoder it cannot win under the unfavourable one; (b) then answers "did you try the published head?" cleanly, and is reported as a confirmation rather than a candidate. One stage-2 run each, then a stage-3 arm only on whichever holds up |
| 21 | **An identity loss**, and **training everything together** | both are chains, and both should wait until the headline numbers are frozen | A50 phase 3 **The identity loss belongs in A63 (2026-10-10):** FROG has no identities, A56 found pseudo-identities unreliable beyond 10 s, and the memory already beats a tuned ByteTrack + CBMOT. It becomes interesting where real tracks exist, i.e. JRDB |
| 22 | **A learned grid over the room: is spatial binding enough?** -- **RUNNING 2026-10-27 as `queue_item22.py` (`grid1m`, `grid2m`)** | **The owner reopened this on 2026-10-27 and was right to.** My reasoning for closing it -- "it replaces the contribution, so it is a different paper" -- is the *same* argument item 36's row rejects in as many words: *"the strongest available attack on this project's own contribution... that is exactly why it is worth running."* Item 36 then produced the sweep's one load-bearing result. **And item 36 only proved that *addressability* matters, not that *object-centric* addressability beats *spatial* addressability**, because nothing spatial was ever run -- so the paper's actual claim is assumed, not supported. "Why slots rather than an occupancy grid?" is the obvious reviewer question and a learned grid is the classical answer. The counter-evidence in this row was about **hand-built** grids, which says little about a learned one | **Grid slots, the owner's design, and it is the sharper experiment.** A dense grid is not runnable as the question: FROG's venue is **120-250 m across** (measured 2026-10-27), so person-scale cells need **~46,000** of them -- 180x the recall attention's cost -- while 256 room cells would be ~11 x 17 m against a 0.5 m match radius. Instead keep the **256 slots**, give each a quantised room cell, and never instantiate an empty one. Capacity, room frame, recurrence, recall, head and eviction all unchanged, so **one variable moves**: an adopted slot's position follows its object, a grid slot's is a cell centre that never moves. `--cell-m` on `step3a`; 1.0 m headline (the match radius is 0.5 m, the gate 0.6 m), 2.0 m as the sensitivity check so a loss cannot be blamed on cell size. Scored against `a71_dim64_s0`'s **86.12%** and the 1.32 pp bar. **A tie would be the important outcome** -- it would mean the object binding this paper rests on does nothing a fixed lattice cannot, which must be known before the freeze, not after publication |


**Items 23 and up: proposed 2026-10-10 from what the FROG measurements now say, parked here to be remembered.** They came out of
reviewing items 13-22 against the evidence: the model and what we know about it have both moved since `docs/PROPOSAL.md` §12 was
written, and these are what that evidence suggests instead. Ordered by value per hour, not appended in the order they arose.

| # | action | why the measurements point here | cost |
| --- | --- | --- | --- |
| ~~24~~ | ~~**Does the coasting head earn its parameters?**~~ **ANSWERED 2026-10-15: no** (`a64_no_coast_s0`, `--no-coast`, one step-3a run on the adopted recipe). Ablated **AP 86.08%** against the matched seed's **86.35%** and the three-seed control range **85.49-86.35** -- inside it, and inside E13's own 1.32 pp three-sigma bar. Every other metric lands inside the control's spread too (FP/frame 0.885 against 0.849-1.239, recall 82.63% against 81.17-87.76). **The head is 137 parameters of 173,579, so the case for removing it is not size -- it is that the write-up should not describe a mechanism that never acts**: it fires 0.0012-0.0020 times per populated frame and **2 / 0 / 0** times across 72,533 person-free frames (item 27). **DROPPED from the architecture 2026-10-16 on the owner's call.** `coast` defaults off; `--coast` restores the head, because every number published before that date was produced with it -- the same reason item 32 kept `SpaceTimeCNNDetector`. One subtlety handled: checkpoints written before the flag existed carry `coast.*` and say nothing about it, so `object_memory` reads the **weights** rather than the default, or the adopted model's own weights would be refused. **Consequence to carry:** the published 85.87% describes the *with-head* model, and step 2's multi-seed runs on the frozen architecture are what will produce the head-free headline -- no extra cost, since those runs are already planned | one run, plus a flag | done |
| ~~25~~ | ~~**Finish O1: one-sided rescoring**~~ **ANSWERED 2026-10-18: bounding is not the fix.** A +-4 logit tanh does not stabilise the head that diverged twice with NaN -- it **silences** it. Memory AP **82.22%, exactly its candidates' 82.22%**, FP 3.045, recall 93.37%: the raw candidate operating point, untouched. A bound tight enough to be stable is tight enough to be useless, so the divergences stay unexplained and **the paper states them as a limitation** rather than a solved problem | one run |
| 26 | **Person-free frames as training negatives** -- **scheduled into the `balanced` step (owner, 2026-10-15), and it has to be there.** `transferred`'s train and val splits are *identical to `official`*; only its **test** split is person-free, so `balanced` (every split 19-21% empty) is the repo's only source of person-free *training* frames. **The detector has never seen a room with nobody in it**, which is the likeliest reason stage 2 emits **2.62 phantoms per empty frame** before the memory filters them to 0.74 -- and item 27 closed the loop on that: every person-free phantom is provably a stage-2 candidate, not a memory invention. **The owner's framing: report `official` and `balanced` side by side in the final table, so the difference between them justifies `balanced`'s existence.** That makes this a column rather than a replacement, so it cannot stale the frozen architecture's ablation table | one run, plus a loader path, after A62 |

**Items 29-31: CPU inference time, the secondary metric (owner, 2026-10-10).** Accuracy is measured for every detector and
speed is not -- not comparably. `memory/performance-log.md`'s speed table has a CPU row for eight of them, but those rows were
taken around 2026-09-10 by a different script than the one that timed ours on 2026-10-08/10, at an unstated thread count and
with no percentile, so the table satisfies its own stated rule ("one machine") while breaking the one it needs ("one protocol").

| # | action | why | cost |
| --- | --- | --- | --- |
| 31 | **Li2Former's accuracy on FROG, or its removal from the comparison.** It is the one detector whose numbers we cite but have never measured: its *inference* runs here (**1675.24 ms** per frame on CPU, p99 1758.21 -- 70x the frame period, item 29) and only its *training* failed, on DirectML, three separate ways (`AGENTS.md` I4) -- and that was before this machine had ROCm, so the premise of that rule no longer holds for it | A comparison table with a cited-but-unmeasured row invites exactly the cross-regime confusion `AGENTS.md` rule 3 exists to prevent. Either it trains here and gets a measured row, or it is dropped to a cited-published-only footnote like PeTra | one training chain, unknown until attempted |

**PeTra stays cited-only, and the reason is recorded so the question does not come back:** there is no PeTra implementation or weights in this repository -- only its published GPU figure (28.17 ms) -- so it has never run here and cannot be timed. It is not a failed attempt.

**Item 34 (the owner's parallel-decoders question) closed 2026-10-15, negative** -- and the reason it failed is the more
useful half (`memory/rejected-ideas.md`).

| # | action | outcome |
| --- | --- | --- |
| ~~34~~ | ~~**Two candidate decoders in parallel, fused by stage 3**~~ | **The ceiling was real and unreachable.** At LFE-Peaks' 1.784 FP/frame each decoder covered ~900 people of 30,283 the other missed (+2.90 / 2.78 / 2.99 pp) -- but at matched FP the union realises **+35 / +30 / +18**, and **at the 0.01 candidate floor the pipeline actually uses, +2 / -9 / -9**: `nms-mean` alone already puts **98.1-98.5%** of people in the pool, and the vote grid reaches 4 pp fewer there. **A complementarity ceiling measured at one operating point does not transfer to a pipeline running at another** |

**Items 32-33, owner's calls 2026-10-11.** One is a removal to carry out; the other is a proposal I would answer differently
than it was put, and the reasons are below rather than in a commit message.

| # | action | why | cost |
| --- | --- | --- | --- |
| 32 | **Retire `FullScanTCNDetector` and `TemporalUNetDetector`**, and **keep `SpaceTimeCNNDetector`** -- the owner's refinement: the one architecture the project's decisions rest on stays, so those decisions stay reproducible | Nothing has been trained on any of the three since 2026-09, but SpaceTimeCNN is the one whose history ablation reorganised the whole project and whose 63.0% is what A35's acceptance criteria were written against. The other two influenced no decision | ~1 h, and the cascade below **disappears** |

**Item 32's scope, after the owner's refinement.** Keeping SpaceTimeCNN removes every consequence that made this risky:
**exactly six definitions go** -- `FullScanTCNDetector`, `TemporalUNetDetector` and the four blocks only they use
(`DilatedScanBackbone`, `_TemporalTCNBlock`, `_UNetEncoderBlock`, `_UNetDecoderBlock`), while **`_gn`, `_SEBlock`,
`_JointConvBlock` and `_MultiScaleBlock` stay** because SpaceTimeCNN reaches them, `_SEBlock` only through `_MultiScaleBlock`,
which is why that had to be checked rather than assumed. **The raw-scan path stays**, since SpaceTimeCNN is its user, so
`aligned_raw_scan` and `--no-align-scans` keep their purpose and the old harness can still train a full-scan architecture.
**No test is deleted and `utils/ablation_zero_history.py` stays** -- which is what makes the decision it produced reproducible
rather than merely recorded, the owner's reason for keeping the architecture. Nine scripts list the two removed detectors as
`--detector` choices or in skip-lists: `train.py`, `train_all.py`, `evaluate.py`, `nms_sweep.py`, `render_video.py`,
`informational_capacity.py`, `error_breakdown.py`, `finetune_on_drow.py`, `cpu_latency.py`.

**The superseded scope, for the record:**

- `library/takheliped/detectors/full_scan.py` goes whole: the three detectors and the blocks only they use
  (`DilatedScanBackbone`, `_TemporalTCNBlock`, `_SEBlock`, `_JointConvBlock`, `_MultiScaleBlock`, the two U-Net blocks, `_gn`).
- **They are the only users of `INPUT_MODE = "raw_scan"`.** Every surviving trainable detector in `train.py` is cutout-based, so
  the old harness becomes cutout-only and its raw-scan path -- `aligned_raw_scan`, `raw_scan`, `--no-align-scans` and the beam
  alignment behind them -- is dead code in that harness. TAKHeLiPeD reads raw scans through its own trainer, not this one. That
  is a simplification worth having, but it is an architectural change rather than a tidy-up.
- `utils/ablation_zero_history.py` exists only to run the history ablation on `SpaceTimeCNN` and goes with it.
- `tests/test_batch_independence.py` is **about** `SpaceTimeCNNDetector`'s forward pass -- it locks down a batch-leakage bug in
  that class -- so it goes too; a bug cannot recur in deleted code. But `tests/test_train_eval_consistency.py` merely *uses*
  that detector as a stand-in while testing shared train/eval machinery, in 16 places, and must be **repointed rather than
  deleted** -- which means repointing it to a cutout detector and checking that none of those tests was secretly about the
  raw-scan path.
- Ten scripts offer them as `--detector` choices or skip-lists and need updating: `train.py`, `train_all.py`, `evaluate.py`,
  `nms_sweep.py`, `render_video.py`, `informational_capacity.py`, `error_breakdown.py`, `finetune_on_drow.py`, `cpu_latency.py`.

**The two removed detectors' measurements stay in the record** -- `memory/performance-log.md` marks their rows removed rather
than deleting them, and the removal records the commit where the code last existed, as A60 did for the coarse convolution
(`LAST_TEMPORAL_COMMIT`). **SpaceTimeCNN's stay reproducible**, which is the point of keeping it. For why that matters: the
2026-09-10 history ablation that reorganised §A found `SpaceTimeCNN` was a five-frame *averager* -- shuffling its window cost
**0.5 pp** while freezing it cost 4.1 -- and its single-frame score of **63.0%** is what A35's acceptance criteria were written
against and what `docs/PAPER.md` argues from. Those numbers were measured and remain true; after this they simply cannot be
*reproduced* from this tree. The removal therefore records the commit where the code last existed, as A60 did for the coarse
convolution (`LAST_TEMPORAL_COMMIT`), and `memory/performance-log.md` marks the rows removed rather than deleting them.

**Item 33: three findings that change the proposal.**

1. **It is not an algorithmic detector.** The `people` stack's `leg_detector` extracts hand-crafted features from laser clusters
   -- point count, width, linearity, circularity, boundary regularity and so on -- and feeds them to a **trained OpenCV forest**
   shipped with the package. It is a classical-ML baseline, not a rule-based one. Prof. Aycard's detector is the only
   genuinely rule-based point in our comparison, and it is also the fastest at **0.20 ms** per frame. Swapping one for the other
   does not replace like with like; it removes a category.
2. **Its number here would be zero-shot, and this repository has a rule about that.** Its shipped classifier was trained on the
   authors' own data at their own sensor height, so a FROG score would be transfer, not in-domain -- the exact mixing of regimes
   `AGENTS.md` rule 3 exists to stop. Reporting it next to our in-domain numbers needs the caveat stated every time.
3. **Getting a number costs days.** It is a ROS 1 catkin package, so either FROG is replayed as `LaserScan` messages inside the
   Docker image and its output captured, or its feature extraction and forest are ported and verified against it. And
   "maintained" is arguable in the other direction: ROS 1 is end-of-life and `leg_detector` has no ROS 2 successor in that stack.

**Owner's resolution, 2026-10-11: mention it, do not build it.** It cannot be reproduced in this setup, so both the swap and
the addition are off; `docs/RESEARCH.md`'s related work names it as the standard ROS answer and says why no number for it
appears here. **Revisit only alongside real-ROS deployment** (§E), where its own stack is available and the comparison would sit
in its natural setting rather than ported into ours. Aycard's detector therefore stays as the rule-based baseline, measured,
credited, and the fastest row in the table.

**Out of scope for the FROG pre-print, deliberately:**

- **Every other partition and dataset: A62** (make the code dataset-agnostic) and **A63** (DROW, JRDB -- and **`balanced`**, moved
  there 2026-10-10 on the owner's call, because it is functionally another dataset: every model would need retraining on it, even
  though it shares FROG's sensor and venue).
- **The ROS deployment gap** (§E), which is a separate project rather than a queued task.
- **Owner-side and not blocking a pre-print:** A54's formal citation for prof. Aycard's `AlgorithmicDetector` (cite as personal
  communication until it arrives), the pending review of `docs/PROPOSAL.md`, and A55's driver revert.

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
| | ~~**A50** post-processing of A43's FROG result, then the deferred heavy ablations, then architecture experiments~~ | **phase 1 closed 2026-10-09**; phase 2's temporal ablations are moot since A60 removed the last temporal path, and phase 3's open rows are re-tiered in the item | the one measurement left, person-free frames, is queued |
| | **A51** stage-3 slots: performance, trainability and output | **the design is settled** (E6+E7, state is the mechanism); what is left is a decision: re-screen the single-seed arms on the adopted recipe, or report them on the backbone they were run on | ~2 GPU-days if re-screened, nothing if not |
| | **A52** finish the migration to the name **TAKHeLiPeD** | docs done 2026-09-23; the code half waits for a gap between training runs | mechanical renames, one commit per step |
| | **A53** rewrite `docs/RESEARCH.md` around TAKHeLiPeD | header note added 2026-09-23; the body still argues the superseded framing | a day of writing, no compute |
| | **A54** a formal citation for prof. O. Aycard's `AlgorithmicDetector` | credit stated everywhere 2026-09-23; **blocked on the owner** for a reference | one question |
| | **A55** revert the GPU driver to Adrenalin 26.2.2, the version AMD requires for ROCm 7.2.1 | queue stopped for it 2026-09-26 04:51; **owner installing** | ~30 min by hand, then one long run as the test |
| | ~~**A56** tracking-literature baselines and the paper's claims~~ | **done 2026-10-03**: E13 +2.90 pp over a tuned ByteTrack + CBMOT, which itself gains +3.30; the unlinked annotations were FROG duplicate labels (2026-10-05); E17-E18 wait for A49 | -- |
| | ~~**A58** a merge radius on the memory's output~~ | **done 2026-10-05, negative**: val picks no merge or 0.10-0.15 m, test moves under 0.06 pp; even an oracle buys only ~0.5 pp | -- |
| | ~~**A57** replace the eight PowerShell launchers with one typed Python queue runner~~ | **done 2026-10-09**: `utils/run_queue.py`, explicit dependencies, a `BLOCKED` outcome, 40 tests with the spawn behind a seam | -- |
| | **A59** what stage 2's score is made of, and lifting it | **phases 1, 2 and 3a done 2026-10-08**: recipe adopted (stage 2 82.2%, whole detector 85.9%); left: the attribution ladder F1/W/R, re-running A56's tracking baselines on the new candidates, and 3b | ~1 h per ladder arm; A56 rerun ~1 day |
| | ~~**A60** the coarse convolution end to end; drop it if it earns nothing~~ | **done 2026-10-08: dropped and removed** -- AP -0.87, first-second recall +0.59 (bar +2); code at e76aa94 | -- |

**Stage-3 slot experiments (owner's call 2026-09-17) are planned in A51**, which supersedes the backbone choice below: the owner chose the **no-temporal** stage 2, not `default_s0`. Start from these facts:

- **Stage 3 needs a new stage 2.** The fine convolution is gone, and `step2_calibration.best.pth` (3 taps), with every stage-3 checkpoint built on it, no longer loads. Use `checkpoints_three_horizon/default_s0/step2_calibration.best.pth` (78.19% test, trained on current code 2026-09-17; `default_s1`/`default_s2` are its sister seeds). Stage-3 candidate caches are now keyed `-limit-silenced`, so they rebuild with the new decoding.
- **Naming from 2026-09-17:** with `--fine-lags` gone, the runs formerly called "no-fine" are the **default** architecture (coarse convolution only), and "static" is **no-temporal** (`--coarse-lags 0`). Directory names (`step2_no_fine`, `confirm_nofine_s*`, `step2_static`, `confirm_static_s*`) keep the old words because the logs refer to them.
- **Measure the time-step suspect first.** It is cheap and needs no stage-2 training. FROG timestamps are bunched, stage 3 clips `dt` to 1 ms, and `manage_slots` divides displacement by it. Log slot speeds against walking speed.
- **FROG is 40 Hz, not 26.2** (A49): fixed in the stage-3 trainer 2026-09-17, before the slot experiments (A51 0b).
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

**Left to run, and moved to A63 step 0 on 2026-10-10** (owner: `balanced` is functionally another dataset, so our side of the
comparison needs retraining there). Both models on `balanced` test, ~2 minutes each since the LFE-PPN fix.
**Never run, despite a draft that said otherwise:** `docs/PAPER.md` carried LFE-Peaks at 59.8% and LFE-PPN at 63.5% on
`balanced` until 2026-10-10, when the owner confirmed nothing has ever been exercised on that partition. The figures were
deleted from the draft.
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

### A47. Evaluate DR-SPAAM in its official streaming form -- **DONE 2026-10-10 via §A item 6**

**Closed.** `DrSpaamDetector.stream()` plus `evaluate_auc(stream=True)`, six tests pinning prefix equivalence exactly.
On DROW test with the published weights: **windowed 67.84%, streaming 69.08%, +1.24 pp**, which moves our reproduction
of the authors' 69.6% from -1.5 to **-0.52 pp**. So every DR-SPAAM accuracy figure in `memory/performance-log.md`
understates the baseline by about that much, while its speed row *overstates* its cost -- the windowed form encodes ten
scans per frame where the published one encodes one (item 29's table states both directions). **Timing the streaming
form is the open remainder**, and was out of item 29's scope. The plan below is kept for the mechanism it records.

**What.** The official DR-SPAAM detector is stateful: in inference mode `_SpatialAttentionMemory` keeps `self._memory` across calls (`memory = alpha * new + (1 - alpha) * attend(memory)`) and `reset()` clears it. Our `DrSpaamDetector.forward` instead re-encodes a `T`-scan window from scratch on every call (`architectures.py`, the template is initialised from `x[:, :, 0]` each time). Every DR-SPAAM number in this repository is therefore the windowed form.

**How.** Add a streaming evaluation path that carries the template between consecutive frames of a sequence and resets it at sequence boundaries; compare against the windowed form on the DROW test set with the published weights. Write the test first: a streamed run over frames `0..T-1` must equal the windowed forward on the same `T` frames.

**Why.** Fidelity of a replicated baseline (`AGENTS.md` rule 3), and it is the in-domain precedent for A43's `step` interface. With `alpha = 0.5` old evidence halves every frame, so the gap is expected to be small -- which is a prediction to check, not a reason to skip it.

### A62. Make the detector dataset-agnostic -- **the first of the two postponed dataset items (owner, 2026-10-09)**

**What.** Everything that has to change before a second dataset can be trained at all, and nothing that depends on having one.
The list is A49's "what the current code ties to FROG", unchanged in substance:

1. `utils/train_takheliped.py` hardcodes `FROG_Dataset` and `frog_laser_angles(720)`: take the dataset and its angles as
   arguments, with FROG as the default so no FROG result moves.
2. **Settings in physical units, not frames or beams.** `docs/PROPOSAL.md` principle 9: a lag, a chunk, a window and a decay are
   seconds; a gate and a radius are metres. Stage 3 already works this way, stage 2 does not.
3. **One angular spacing.** `docs/PROPOSAL.md` §5.3's resampling is not built, so a kernel spans 0.25 deg per beam on FROG and
   0.5 deg on DROW -- the same convolution sees twice the arc.
4. **The per-dataset parameter table** the owner asked for, printed by the trainer and stored in the checkpoint: beams, field of
   view, spacing, rate, annotated share, people per frame, classes, range limit.
5. The range limit is already `--max-range-m`; state it per dataset (10 m drops 7.3% of DROW's labels).

**How.** Tests first, FROG as the regression: a FROG run on the ported code must reproduce a current FROG run's candidates
bit for bit where the settings are equivalent, which is the only check that keeps the headline numbers intact. No retraining is
needed for the port itself.

**Why.** It is the whole reason a second dataset is cheap once it is done, and it is the item that makes "adaptive" a property of
the code rather than a claim in the paper.

**Cost.** ~a day of code, no GPU. **Gated on nothing** -- it can start whenever the FROG list is not using the machine.

### A63. Train and report on DROW, then JRDB -- **the second of the two postponed dataset items (owner, 2026-10-09)**

**Reporting rule for `balanced`, owner's call 2026-10-15.** The final comparison table carries **both** `official` and `balanced` results, side by side. The difference between them is not a caveat to explain away -- it is the result that justifies the `balanced` partition existing at all, and it is also why item 26 (person-free training negatives, which only `balanced` can supply) cannot stale the frozen architecture's ablation table: it adds a column rather than replacing the model.

**What.** The results half of A49, plus the two A56 rows that were waiting on it.

| step | what | notes |
| --- | --- | --- |
| 0 | **FROG `balanced`** -- the cheapest "other dataset" and the one to do first: same sensor, same venue, same loader, **two** test recordings against `official`'s one, and a 121,329-frame val split with 19.8% empty frames against a 2,000-frame populated-only sample. Train the adopted recipe there at three seeds, score LFE-Peaks and LFE-PPN on its test split (~2 min each, never measured, A32's last open row), and extract the `official`-test subset from the same pass, since `balanced` holds `frog_16-41` whole | **Moved here 2026-10-10, owner's call:** it is functionally another dataset, because every model needs retraining on it. **Never mix partitions**: `balanced`'s val is 11-36 + 10-31 and 11-36 is half of `official`'s train, so validating there while training on `official` scores the model on its own training data. State the deviation when quoting: published baselines trained on 11-36 + 12-43, `balanced` trains on 12-43 + 14-57 **Nothing has ever been run on `balanced`** (owner, 2026-10-10). `docs/PAPER.md` had quoted LFE-Peaks 59.8% and LFE-PPN 63.5% there; those figures had no measurement behind them and were **deleted** from the draft, not re-derived. So this step starts from nothing: the partition is defined in the loader and has never been exercised. |
| 1 | **DROW**: train stage 2 and stage 3 with the settings in physical units, three seeds, and report against DROW's own published baselines | 12.7 Hz, every 5th scan annotated (20x less stage-3 supervision than FROG), three classes |
| 2 | **FROG -> DROW zero-shot**, beside it -- **and this is now also the dt-aware mechanism's last defence (item 37, 2026-10-22)**: DROW runs at 12.5 Hz against FROG's 40.01 Hz, so a model that reads real elapsed time should transfer across the rate change where one taking fixed steps cannot. It needs no new training, which is why it beats the drop-augmented retraining item 37 declined. Report `selective` against `gru`/`mingru` on this row specifically | the owner's 2026-09-15 question: one model per dataset, one transferred, or both -- each result says which |
| 3 | **JRDB**: measure the loader's geometry, rate and annotation density first (the table's "to measure" row), then repeat | the official DR-SPAAM repo merges two lasers into 1091 points / 360 deg; our loader assumes 541 / 270 deg |
| 4 | **A56's E17**: the slots scored as a tracker -- HOTA, MOTA, IDF1, OSPA(2) -- on JRDB's 3D tracks projected to the ground plane | needs step 3; FROG has no identities, which is why these metrics were never computed |
| 5 | **A56's E18**: self-supervised memory, or DROW label densification from slot tracks | needs step 1; densifying DROW's every-5th annotation is itself a research item |

**Rules that already apply.** **AP does not compare across datasets** (person density differs 3.8x), so each dataset's result is
set against that dataset's published baselines only; and every number states its range limit, since 10 m is not a neutral choice
on DROW.

**Cost.** `balanced` ~1 day of GPU at three seeds plus the two LFE evaluations; DROW ~2 days plus the zero-shot pass; JRDB
unknown until step 3's measurements. **Step 0 is gated on nothing** -- it needs no part of A62, since it shares FROG's loader,
sensor and angular grid -- so it can run beside the FROG pre-print list whenever the machine is free. Steps 1-5 are gated on A62.

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

### A61. A56's tracking baselines on the adopted recipe -- **DONE 2026-10-09: the claim holds, +2.36 pp over the tuned tracker, every seed's interval above zero**

**What.** Every A56 row (SORT, ByteTrack as published, ByteTrack + CBMOT tuned on val, the rule scorer E15, the LFE detectors, GOSPA, strata, first-detection latency) is on the old recipe's vote-grid candidates and E13 at 83.86%. The paper's "learned state beats a tuned tracker" claim has to be re-read against today's detector: stage 2 C1 averaged (`a59_c1_recipe_dropout01_s{k}/step2_calibration.averaged.pth`, `nms-mean`) and its memory (`a60_e13_static_s{k}`, 85.87%).

**Launched 2026-10-09** as the second half of `run_queue_ladder_a61.ps1` (`QUEUE a61_*` lines), after the ladder; it starts even if a ladder step fails. Every candidate cache it reads already exists (A60 built them). `--arms` and `summary` are built and tested.

**How.** Each memory seed was trained on its own stage-2 seed, and A56 compares trackers on identical candidates, so **each seed is its own replay**: `tracking_baselines.py {rule,track,arms,report} --stage2 <C1 seed k averaged> --out checkpoints_three_horizon/a61/s{k} --arms E13:a60_e13_static_s{k}`, then a `summary` step that averages the three reports. Needs `--arms` (today `ARMS` is hardcoded to the A51 runs) and `summary`, tests first. ~3.4 h per seed (rule scorer 2.5 h, trackers 0.5 h, arms and report 0.4 h), ~10 h, one resumable Task Scheduler chain after the ladder.

**Predicted, before running** (old recipe: candidates 77.66, E14 80.96, E13 83.86): the tracker's gain shrinks with the candidates' gap, as the memory's did (+6.20 -> +3.65): **E14 +1.5 to +2.5 over the candidates**; **E13 - E14 +1.0 to +2.5 pp**, the interval above zero on at least two of three seeds; the rule scorer E15 still below the candidates; the learned state's lead still largest on occluded people and in a trajectory's first 0.5 s.

**Seed 0, 2026-10-09** (`checkpoints_three_horizon/a61/s0/report.json`; every 5th annotated test frame, 95% block-bootstrap intervals): candidates 82.22%; SORT `min_hits` 1 82.36 (+0.14); ByteTrack as published 81.28 (-0.94); **ByteTrack + CBMOT tuned on val (E14) 83.52, +1.30 [+1.03, +1.56]**; rule scorer (E15) 71.67 (-10.55); **E13 86.35, +4.13 over the candidates and +2.84 [+1.99, +3.73] over E14**. GOSPA: E13 0.447, E14 0.482, candidates 0.492. Recall at matched FP over E14: occluded people 80.6% against 73.7% (+6.9), partial 90.9 against 88.0, visible a tie; a trajectory's first 0.5 s 94.4 against 90.5; detected on the first frame 94.8% against 82.8% (candidates 91.7%). *One seed:* E14's gain is below its predicted +1.5 to +2.5, and E13 - E14 sits above its predicted +1.0 to +2.5 -- the stronger stage 2 narrowed the tracker's gain, not the memory's lead.

**Seed 1, 2026-10-09** (`a61/s1/report.json`, same reading): candidates 82.11%; SORT `min_hits` 1 82.23 (+0.12); ByteTrack as published 81.00 (-1.11); **E14 83.46, +1.35 [+1.13, +1.58]**; E15 72.31 (-9.80); **E13 85.77, +3.66 over the candidates and +2.31 [+1.28, +3.36] over E14**. GOSPA: E13 0.462, E14 0.498, candidates 0.494. Recall at matched FP, E13 against E14: occluded 75.5 against 74.2, partial 90.1 against 87.0, visible 96.8 against 97.0; first 0.5 s 94.4 against 89.4; first frame 93.2% against 82.4% (candidates 90.7%).

**Seed 2, 2026-10-09** (`a61/s2/report.json`): candidates 82.23%; E14 83.55, +1.32 [+1.11, +1.54]; E15 72.09; **E13 85.49, +3.26 over the candidates and +1.94 [+0.94, +3.01] over E14**; at matched FP E13's recall is 94.56% against E14's 95.40.

**RESULT, three seeds** (`checkpoints_three_horizon/a61/summary.json`; FROG test, every 5th annotated frame, 95% block-bootstrap intervals per seed; recall at the candidates' own false positives per frame at 0.3):

| what reads the stage-2 candidates | AP, 3 seeds | vs candidates, per seed | recall at matched FP | GOSPA |
| --- | --- | --- | --- | --- |
| nothing | 82.19% | -- | 94.13% | 0.494 |
| SORT `min_hits` 1 | 82.31% | +0.14 / +0.12 / +0.12, all above zero | 94.19% | 0.494 |
| ByteTrack as published | 81.17% | -0.94 / -1.11 / -1.00 | 94.29% | 0.494 |
| the memory's bookkeeping with CBMOT's score (E15) | 72.02% | -10.55 / -9.80 / -10.14 | 94.58% | 0.619 |
| ByteTrack + CBMOT, tuned on val (E14) | 83.51% | +1.30 / +1.35 / +1.32, all above zero | **95.56%** | 0.493 |
| **learned state, no attention (E13)** | **85.87%** | **+4.13 / +3.66 / +3.26; over E14 +2.84 [+1.99, +3.73] / +2.31 [+1.28, +3.36] / +1.94 [+0.94, +3.01]** | 95.47% | **0.454** |

**Predictions scored.** E14 +1.5 to +2.5 over the candidates: **missed low**, +1.30 to +1.35. E13 - E14 +1.0 to +2.5 with the interval above zero on at least two seeds: **held**, mean +2.36, all three above zero. E15 below the candidates: **held**, by ~10 pp. The lead largest on occluded people and in a trajectory's first 0.5 s: **held** -- recall at matched FP over E14, occluded 78.3 against 74.9 (+3.4), partial 89.5 against 87.3 (+2.2), first 0.5 s 93.4 against 90.0 (+3.4), beyond 8 m 95.4 against 92.2 (+3.2); first-frame detection 93.0% against E14's 83.0% and the candidates' 91.2%.

**Not predicted, and it changes a sentence of the paper: at one matched false-positive rate the memory no longer finds more people than the tuned tracker** (95.47 against 95.56%; on the old recipe +1.0, 93.16 against 92.17). It gives back visible, standing and near people (visible 96.4 against 96.9, still 91.4 against 93.2, 0-2 m 93.4 against 94.9) for the occluded, new and far ones. Its AP lead comes from ranking over the whole curve, and GOSPA agrees (0.454 against 0.493). The 2,254 copied test labels sit in the "unknown" stratum (E13 64.1% against 54.9%), which is double reporting, not people (A56).

**Why.** The tracker comparison is one of the paper's two claims, and it rested on a stage 2 four points weaker than the one the paper reports. It survives: the stronger stage 2 shrank what a tuned tracker adds (+3.30 -> +1.32) more than what the memory adds over it (+2.90 -> +2.36).

### A60. The coarse convolution: test it end to end, then drop it if it earns nothing -- **DONE 2026-10-08: DROPPED AND REMOVED**

**What.** Decide whether stage 2 keeps `CausalTemporalConv` (the ~0.8 s coarse space-time convolution at the bottleneck) now that the object memory exists. **The owner's instruction (2026-10-05): run the tests below; if the convolution turns out useless, drop it, and record in `CHANGELOG.md` exactly where it lived, so it can be restored or reused later.** This replaces the §D item *Do the object slots already cover the coarse horizon?* (2026-09-17), whose precondition -- a working stage 3 -- A51 met.

**The evidence so far, none of it end to end:**

- Stage 2 alone ties at best: default 77.92 (old controls) or 76.65 (fresh `default_s*`) against no-temporal 77.23; through `stage2_probe.py`, every 5th test frame, vote grid 76.75 against 77.27 and DR-SPAAM's decoder 79.42 against 79.88 (A59).
- It helps no group of people: at LFE's false-positive rates the two architectures' recall agrees within ~2 pp in every range bin and every crowding bin (A59). It does read history -- shuffling it costs `default_s0` 0.67 pp -- and buys nothing measurable with it.
- The memory gets more out of the same span: emptied every 0.5 s it still gains +1.75 pp over its candidates, every 1 s +2.80 (A51 E1), against the convolution's at most ~0.7 at stage 2.
- Every stage-3 result already runs without it, by the owner's call of 2026-09-17.
- Its cost: +201,216 parameters (+20.6%), ~3.3x the training time per epoch (139 s against 42 s per quarter epoch), a 1.8 MB streaming cache, and odometry inside stage 2.

**How -- the tests, ~1.5 days of GPU, unattended:**

1. Candidate caches for `default_s0..2` and for A59's current-code `a59_static_ckpt_s0..2` (`load_split --stage2`, ~1-1.5 h each).
2. E13's stage 3 on each: `step3a --stage2 <ckpt> --chunk-s 1 10 --no-exchange --no-learned-match --test-strides 5`, ~4.4 h each, one stage-3 seed per stage-2 seed so that both seeds vary.
3. **Written 2026-10-06, tests written, NOT YET RUN** (owner's call: write now, test later -- nothing that imports torch runs beside the chain): `utils/coarse_e2e.py`, 9 tests in `tests/test_coarse_e2e.py`. It defines a person's first second without the memory -- annotations linked into trajectories in world coordinates (`motion_analysis.build_tracks`, 0.6 m, A56's age strata), FROG's copied labels left out -- and reads every recall at one budget, the no-temporal arm's mean false positives per frame at 0.3. **Before reading any A60 number:** on an idle machine, `pytest tests/test_coarse_e2e.py`, fix until green, then the full suite, then `python coarse_e2e.py`. The paragraph below is the plan it implements. Score test AP and AP@0.3 (every 5th annotated frame), recall at matched false positives, and **recall in each person's first second in view** -- the people a slot cannot have seen yet, where beam-level motion is the only cue a memory lacks. FROG has no identities, so a person's first second is the frames before `memory_diagnosis.py --recovery` first finds a live slot on them; build that as a small option there, tests first.
4. Optionally, the per-beam latency of stage 2 with and without it, on CPU at 4 threads, on an idle machine (`tracking_baselines.py latency`).

**Owner's call 2026-10-06: run it on the new stage-2 recipe (A59: `nms-mean`, mirror, nearest-person labels), merged with A59 phase 3a.** The default-architecture stage 2 is retrained with that recipe at three seeds; the no-temporal side is the combination run's stage 2; stage 3 (E13) is trained on all six with `nms-mean` candidates. The no-temporal half is then the stage-3 retrain whichever way the rule falls, so nothing is trained twice. **Launched 2026-10-06 as named, resumable steps** (owner's call: cancellable, and resumable after a bug), `checkpoints_three_horizon/run_a60.ps1`, Task Scheduler task `lidar_a60`: each step is skipped when its output exists, training steps resume from their own state, `-Steps a,b` runs chosen ones; cancel with `schtasks /end /tn lidar_a60`, resume with `schtasks /run /tn lidar_a60`. It trains the default architecture on C1's recipe (`step2 --dropout 0.1 --checkpoint-every 0.25`, three seeds), averages it, saves every stage 2's val-chosen average as `step2_calibration.averaged.pth` (`utils/average_checkpoints.py`; the no-coarse side's three are already saved), then trains E13 on all six, arms alternating; candidate caches are now keyed by run directory too (`candidate_cache_path`, 2 tests), since every run writes identically named checkpoints.

**Stopped and resumed** (2026-10-06 21:49 -> 2026-10-07 05:17): the chain ended with `0xC000013A` at keyboard or mouse input, as on 2026-10-05 (`memory/gotchas.md`), during `stage3_static_s0`'s 10 s stage; relaunched by the owner's call with its console hidden, it skipped the four finished steps and resumed that run from its state (a different draw from here, fine for one arm's seed).

**Stage 2 on the new recipe, done 2026-10-06** (`a60_default_c1_s*`, every annotated test frame, `nms-mean`, before averaging): the default architecture 81.21 / 81.23 / 80.83% (mean **81.09**) against the no-temporal C1's 82.12 / 81.80 / 81.86 (mean 81.93) -- **-0.84 pp at stage 2**, as on the old recipe a tie or worse. Stage 3 decides.

**First stage 3, 2026-10-07** (`a60_e13_static_s0`, resumed once, every 5th annotated test frame): candidates 82.22% -> **memory 86.35%** (AP@0.3 85.24%, recall at its own FP rate 96.0%, 5.17 ms/frame), against E13's 83.86% on the old recipe. One seed. **`a60_e13_default_s0`** (coarse arm, 2026-10-07, its 10 s stage running 15+ epochs): candidates 82.51% -> memory **84.89%** (AP@0.3 83.73%), the memory adding +2.4 where the no-temporal arm's adds +4.1; seed 0 of each, so nothing is concluded. **`a60_e13_static_s1`** (2026-10-07): candidates 82.11% -> memory **85.77%** (AP@0.3 84.65%). **`a60_e13_default_s1`** (2026-10-08): candidates 82.15% -> memory **85.05%** (AP@0.3 83.95%). **`a60_e13_static_s2`** (2026-10-08): candidates 82.23% -> memory **85.49%** (AP@0.3 84.16%); **the no-temporal arm is complete, 86.35 / 85.77 / 85.49% (mean 85.87)**. With the default arm at 84.89 / 85.05, its seed 2 would need ~92% to clear the 1.5 pp AP bar, so the AP half of the rule is settled for *drop*; first-second recall (`coarse_e2e.py`) decides the rest.

**VERDICT 2026-10-08: DROP** (`utils/coarse_e2e.py`, `checkpoints_three_horizon/a60/a60.json`; its 9 tests passed on first run and the full suite, 673, before it was run; it reproduces every run's recorded test AP exactly):

| full detector, FROG test | AP | AP@0.3 | recall at 1.056 FP/frame | first-second recall (n = 24,477) |
| --- | --- | --- | --- | --- |
| no-temporal stage 2, 3 seeds | **85.87%** (86.35 / 85.77 / 85.49) | **84.68%** | **85.26%** | 75.11% |
| default (coarse) stage 2, 3 seeds | 85.01% (84.89 / 85.05 / 85.08) | 83.82% | 84.61% | 75.71% |
| default - no-temporal | **-0.87** | -0.86 | -0.65 | **+0.59** (bar +2.0) |

Every no-temporal seed beats every default seed on AP. **Both predictions held**: AP within ~1 pp, first-second recall within ~2 pp. The one place the convolution could have earned its keep -- people the memory has not yet seen -- moves +0.59 pp. **Removed 2026-10-08 (owner's call)**, in the commit after e76aa94, which is the last that holds it (`memory/rejected-ideas.md` has every symbol, the flags, the checkpoints that need it and the one-line restore). **The no-temporal half is A59 phase 3a's retrain**: the full detector on the adopted recipe scores **85.87%**, +2.01 over E13's 83.86% on the old recipe.

**Decision rule, written before the runs.** *Drop* if the default's three-seed mean test AP is not above the no-temporal mean by more than **1.5 pp** (the three-seed resolution, `memory/interpreting-evaluation.md`) **and** its first-second recall is not above by more than 2 pp. *Keep* otherwise, and then measure what it buys per frame of latency. *Predicted:* AP within ~1 pp, first-second recall within ~2 pp, so drop.

**The removal, done 2026-10-08 as planned:**

- `library/takheliped/detectors/takheliped.py`: `align_sectors`, `TemporalCache`, `CausalTemporalConv`, `CalibrationState`, `POSE_CHANNELS`, and `CalibrationNetwork`'s `coarse_lags`, its `poses` argument and its `step` caches. Stage 2 becomes one frame in, `forward(ranges)`; stage 3 keeps its poses, which the room frame needs.
- `utils/train_takheliped.py`: `--coarse-lags`, `--clip-frames`, `clip_length`, `clip_frames`, the clip and pose plumbing of `clip_batch`, `network_candidates`, `evaluate_calibration` and `joint_step`; `duration_curve.py` and `stage2_probe.py` follow.
- **What must survive:** the no-temporal stage 2 still runs a plain 3-tap spatial convolution at the bottleneck (today `coarse.conv`, `Conv1d(128, 128, 3)`), and every stage-3 result rests on checkpoints of it, so those must keep loading and scoring bit-identically -- pin that with a test against a saved output before removing anything. The `default_s*`, `confirm_nofine_s*` and `step2_no_fine` checkpoints stop loading; say so.
- **The history entry**, in the same commit: the last commit that contains the convolution (`git log -1 --format=%H -- library/takheliped/detectors/takheliped.py` before the removal), every file and symbol above with its line range at that commit, the flags, the checkpoint directories that need it, and the one-line restore (`git show <hash>:library/takheliped/detectors/takheliped.py`). Then add the convolution to `memory/rejected-ideas.md` with the numbers and what would reopen it (a dataset where the memory cannot see first appearances, e.g. a lower frame rate, A49). The docs naming "three horizons" (`docs/PROPOSAL.md`, `memory/detector-architectures.md`) change with it.

**Why.** If it earns nothing, it costs a fifth of stage 2's parameters, a third of its training speed and its odometry dependence; and the paper should claim two temporal horizons, not three, unless the third is measured.

### A59. What stage 2's score is made of, and lifting it -- **phases 1, 2 and 3a done; the F1/W/R ladder and its F1+W cell done 2026-10-11; 3b dropped**

**W's latency, measured 2026-10-09** (`results/a59_w_latency.json`; one scan at a time, network then `nms-mean` decoding, 2,000 test scans, same process; `tracking_baselines.py latency` cannot time it, since E13's memory reads C1's 32-channel decoder features): **W is no faster.** CPU at 4 threads, median total 11.14 ms (network 8.53) against C1's 11.46 (8.78, matching A60's 8.75); GPU 9.04 against 8.80. At batch 1 stage 2's time is fixed per-scan overhead, not width, so a 64k-parameter stage 2 buys memory, not speed, and costs 1.3-1.5 pp.

**Paused and resumed (history).** The owner paused the tests at ~18:14 (the arms chain ended with 0xC000013A; no crash, no reboot). Finished: phases 1, L1, L2, F3 and M at three seeds, and D at seeds 0-1 (78.96 / 78.07%). Left: D seed 2, saved at epoch 1.75 (`a59_d_dropout01_s2/step2_resume.pth`), then F3/M/D's weight averaging and probe. **To resume**, on an idle machine (`memory/scripts/machine_is_free.py`): launch `checkpoints_three_horizon/run_a59_arms_resume.ps1` through the Task Scheduler as `memory/long-running-tasks.md` shows (~10 min of training, then ~2 h of evaluation), and re-arm the event and clock watchers. The resumed seed is a different draw, which is fine for a screening arm.

**The owner's questions (2026-10-05).** (1) Are the coarse convolutions worth keeping now that the slots exist, or can stage 2 take one frame per call? That is now A60. (2) Why does stage 2 alone, with no temporal input, beat DR-SPAAM (published 73.7% at `T=1`, 75.6% at `T=5`) and LFE (65.6 / 69.2%), and is it below its potential? The paper needs a measured answer.

**Measured 2026-10-05, evaluation only** (`utils/stage2_probe.py`, FROG `official` test, every 5th annotated frame; three no-temporal seeds and three default seeds; `results/stage2_probe.json`):

- **The decoder is a handicap, not the advantage.** The same probabilities and votes through the official DR-SPAAM decoder (`nms_predicted_center`, greedy, 0.5 m) score higher on all six checkpoints, +2.33 to +3.03 pp: no-temporal 77.27 -> **79.88%**, default 76.75 -> **79.42%** (AP@0.3 75.79 -> 76.58, 75.04 -> 75.82). LFE-Peaks' peak decoder scores 71.3%, and the same peaks at their beam's vote 74.1%, so the centre vote is worth ~3 pp over a surface point. Our vote grid scores a detection by the *plain mean* of every voter down to p = 0.001.
- **Not capacity against the cutouts:** stage 2 has 976,707 parameters against DROW3's 1,518,918 and DR-SPAAM's 1,978,054 (our ports). Against LFE (53,258 / 170,903) it is 6-18x larger, untested.
- **Not training compute:** the selected checkpoint sits at epoch ~1 at batch 32, 108k samples seen, against DR-SPAAM's 5 epochs at batch 8 (541k) and LFE's up to 100. `memory/performance-log.md`'s "~38x the compute, in our favour" is about the retired full-scan models, not this one.
- **Where it wins, at LFE's own false positives per frame** (people counted once per frame): far and isolated people. At LFE-Peaks' 1.78 FP/frame, 8-10 m recall 50.3% -> 83.2% and 6-8 m 75.3 -> 85.1, people 1-2 m from the next 78.6 -> 92.8; at 1-4 m it is 1-3 pp *worse*.
- **Where it loses, through the vote grid: close pairs.** People within 0.6 m of another (9.3% of test people, 4.9% of train) are found 59.9% of the time against LFE-Peaks' 83.2% and LFE-PPN's 84.0%. 1.2% of test people have a neighbour under 0.3 m, so these are mostly real pairs, not twice-annotated people. **Corrected the same day: this is the vote grid's, not the network's** -- through NMS the same outputs find 82.3% of them (phase 1, item 2).
- **A label flaw first read as the cause** (it is not -- see phase 1, item 2): `make_targets` lets the last-listed person within 0.6 m overwrite a beam's label and vote, so on FROG train **3.3% of positive beams vote for a person who is not their nearest** and 12% of people lose beams to a neighbour (6.9% of positive beams sit in two circles). DR-SPAAM assigns each beam to the nearest person, within 0.35 m (0.6 m is its *wheelchair* radius): there the share falls to 0.4%.
- **Training duration costs more than any architecture arm measured:** test AP of the no-temporal seed 0 peaks at 77.60% at epoch 1.00 and falls to 72.36% by epoch 4.00 (every annotated frame, `step2_static/duration_curve_test.json`), precision rising 50.9 -> 69.4% while recall falls 88.8 -> 76.5%: the network memorises the two training recordings. The baselines cannot see this: DR-SPAAM ran a fixed 5 epochs, and LFE early-stopped on loss over `split == 1` frames *interleaved* with its training frames (`2DLaserPeopleBenchmark/train_segmentation.py`).
- **LFE's official training never shuffled its data:** `SegDataGenerator.__call__` shuffles a copy and iterates the original, and the `tf.data` pipeline adds no shuffle, so the LFE backbone trained on batches of 32 consecutive scans in recording order. LFE-PPN fine-tuned that backbone with shuffling, on scans holding at least 3 people.
- **The gap to LFE-Peaks, decomposed on fixed weights** (no-temporal, three-seed means): LFE-Peaks 65.11% -> our network through LFE-Peaks' own decoder 71.26 (**+6.15, the network and its training**) -> the same peaks at the peak beam's vote 74.10 (**+2.84, centre regression**) -> the vote grid 77.27 (**+3.17, many beams' votes aggregated**) -> DR-SPAAM's NMS 79.88 (+2.61, scoring by the best beam). So about half of our lead over LFE-Peaks is the output head, and half the network; the +6.15 is a lower bound for the network, since our probability map was trained for votes, not peaks.
- **The depth-centring input goes blind up close.** `r - median` over 11 deg (`MEDIAN_WINDOW_DEG`, sized for a person at 4 m) reads ~0 on a person's own beams for 98% of people at 0-1 m and 41% at 1-2 m, and only -0.19 m at 2-3 m, because the person fills the window (FROG test, every 50th frame, a scratch analysis). Over 33 deg the signal is -1.2 m at 1-2 m and -2.5 m at 2-3 m. It was read as the cause of stage 2's 1-4 m deficit against LFE; that deficit too is the vote grid's (phase 1, item 2), so the blind spot is real in the input but its cost is unmeasured.
- **Weight averaging is free accuracy, on one seed:** the mean of the checkpoints from epochs 0.5-1.5 (chosen on val, 74.06%) scores **78.53%** on test against the selected checkpoint's 77.58%, and every window tried beats it (+0.52 to +0.95; `results/swa_step2_static.json`).

**Phase 1, running now (each under 2 h, no change to the trainer).** Predictions written before the results:

1. ~~**Weight averaging at three seeds.**~~ **Done 2026-10-05.** `step2 --coarse-lags 0 --checkpoint-every 0.25 --seed {0,1,2}` into `checkpoints_three_horizon/a59_static_ckpt_s*`, then the average over the val-chosen window (`checkpoints_three_horizon/a59_swa_probe.py`, `results/swa_a59_*.json`, every 5th annotated test frame). *Predicted:* +0.5 to +1.0 pp on every seed. Also gives three duration curves and a fresh three-seed no-temporal baseline on current code. **Scored: missed on "every seed".** Over the nine A59 runs the mean gain is +0.54 pp but two lose: control +0.96 / -0.27 / +0.82, L1 +0.36 / -0.75 / +0.26, L2 +2.09 / +0.89 / +0.48. A modest, unreliable gain, not seed 0's +0.95. **L2 averaged is the strongest stage 2 so far, 79.39 / 79.38 / 79.12% (mean 79.30) against the averaged control's 77.61, +1.69 pp** -- above the bar, but not the comparison registered in advance, so suggestive until the probe reads it under `nms-mean`.
2. ~~**Choosing the decoder on val.**~~ **Done 2026-10-05.** `stage2_probe.py --split val --stride 25` over the six checkpoints, then every decoder through the matched-FP breakdown on test (`results/stage2_probe_val.json`, `stage2_probe_decoders.json`). *Predicted:* NMS with each detection at the probability-weighted mean of the votes it suppresses (`nms-mean`) wins on val at both radii, and gains at least +2.5 pp on test on all six. Its close-pair recall stays below the vote grid's, since NMS cannot emit two detections within 0.5 m, which would leave the labels as the cause of the close-pair deficit. **Scored: the first two held, the third missed outright.** `nms-mean` wins on val on all six (mean 75.34% against the vote grid's 73.44%; AP@0.3 72.87 against 71.40) and gains +2.60 to +3.15 pp on test on all six: no-temporal **80.07%**, default 79.61%, AP@0.3 77.94 / 76.96. But it *recovers* the close pairs, 59.9% -> 82.3% (LFE-Peaks 83.2%), and erases the 1-4 m deficit (1-2 m 87.2 -> 91.0%, 2-3 m 88.9 -> 92.1%, 3-4 m 85.9 -> 90.0%, all now above LFE-Peaks): **both of stage 2's weaknesses against LFE were the vote grid merging neighbours**, not the labels or the input. It gives some back far away and alone (8-10 m 83.2 -> 77.7%, people over 2 m from the next 82.2 -> 78.4%), where aggregating a few beams helps. **Proposed default decoder: `nms-mean`**, the owner's call; adopting it changes every stage-3 input, so stage 3 is retrained on it (phase 3a) before any full-model number moves.

**Phase 2: training arms on the no-temporal stage 2**, three seeds each, ~15-25 min per seed, each an opt-in `step2` flag that reproduces today's network when off. **Owner's call 2026-10-05: L1 and L2 only, for now; the rest stay queued here.** Built tests first as `step2 --label-assign {last,nearest} --label-radius` (`make_targets(nearest=)`, 4 tests in `TestNearestLabelAssignment`), and queued behind phase 1 as `checkpoints_three_horizon/a59_l{1,2}_nearest_s{0,1,2}`, with a checkpoint every 0.25 epoch so they average like the controls (`a59_static_ckpt_s*`).

| arm | change | predicted |
| --- | --- | --- |
| L1 | nearest-person label assignment at 0.6 m | close-pair recall 60% -> at least 70%, AP +0.5 to +1.5 (written before item 2 showed the close-pair loss is the decoder's; read L1/L2 under `nms-mean` too) |
| L2 | L1 at 0.4 m, FROG's own circle radius | as L1, and fewer phantoms beside walls |
| M | mirror augmentation: reverse the 720 beams, bearings `phi -> -phi - 0.25 deg` (exact on FROG's grid) | a higher and later peak, +0.5 to +1.0 |
| D | `--dropout 0.1` (exists) | +0.3 to +0.7, the old single-seed +0.70 shrinking |
| F1 | input `1 - r/10` alone (LFE's) | -1 to -3, mostly beyond 6 m: the neighbour offsets are linear in the five neighbouring ranges, so a learned stem loses only the median channel |
| F3 | a second depth-centring channel, `r - median` over 33 deg (proposed 2026-10-05, after the blind-spot measurement above) | 1-4 m recall +1 to +3 pp, AP +0.3 to +1.0 |
| W | LFE-sized: widths (8, 16, 32), 63,891 parameters | -2 to -4 |
| R | DR-SPAAM's recipe: 5 epochs, batch 8, LR decaying 1e-3 -> 1e-6, last weights | 73-75%, the baselines' own range |

**L1 trained 2026-10-05** (vote grid, every annotated test frame): **78.64 / 78.66 / 77.68%**, mean 78.33 (sd 0.56), against the current-code control's 77.29 / 77.18 / 76.60 (mean 77.02, sd 0.37): **+1.31 pp, under the ~1.5 pp three-seed bar**, so suggestive only. **L2** (nearest within 0.4 m): **77.37 / 78.55 / 78.56%**, mean 78.16 (sd 0.68), **+1.14 pp**, also under the bar. Both point the same way, all six above the control's mean, and neither clears it alone. **Read under every decoder 2026-10-05** (`results/stage2_probe_a59.json`, every 5th annotated frame): under `nms-mean` control 79.75% (78.83 / 80.45 / 79.96), **L1 80.89% (80.95 / 81.12 / 80.61), +1.14**, L2 80.48%, +0.73; AP@0.3 77.46 / 78.80 / 79.08, so the clearest gain is localisation, +1.3 to +1.6 pp. Under both NMS decoders every L1 seed beats every control seed. **Predictions scored:** L1's AP +0.5 to +1.5 held (+1.23 vote grid, +1.14 `nms-mean`), still under the three-seed bar; L1's close-pair recall to at least 70% missed (vote grid 62.0 -> 64.7%, `nms-mean` 83.6 -> 84.4%), as item 2 had already shown the close-pair loss to be the decoder's; L2's "fewer phantoms beside walls" was not measured. **Recommendation, the owner's call:** adopt L1's nearest-person assignment at the existing 0.6 m radius. It corrects a documented flaw, matches DR-SPAAM's rule, costs nothing, and points the same way in six runs under three decoders; the best stage 2 is now **L1 + `nms-mean`, 80.89%**.

**Phase 2, all five arms scored 2026-10-06** (FROG test, every 5th annotated frame, three seeds each; `results/stage2_probe_a59.json`, `stage2_probe_a59_arms.json`; recall at LFE-Peaks' false positives per frame):

| arm | vote grid | `nms-mean` | AP@0.3 | 1-4 m | close pairs | 8-10 m |
| --- | --- | --- | --- | --- | --- | --- |
| control | 77.10% | 79.75% | 77.46% | 90.9% | 83.6% | 78.2% |
| L1 nearest, 0.6 m | 78.33% | 80.89% (+1.14) | 78.80% | 92.1% | 84.4% | 78.5% |
| L2 nearest, 0.4 m | 78.14% | 80.48% (+0.73) | 79.08% | 91.8% | 85.3% | 80.3% |
| F3 median over 33 deg too | 77.43% | 80.23% (+0.48) | 78.19% | 91.5% | 84.1% | 78.4% |
| **M mirror** | 77.88% | **81.49% (+1.74)**, 81.48 / 81.95 / 81.03 | 79.29% | 90.0% | 79.4% | **81.4%** |
| D dropout 0.1 | 78.38% | 80.45% (+0.71) | 78.89% | 90.3% | 84.6% | 79.0% |

**Predictions scored.** L1 AP held (+1.1 to +1.2), its close-pair recall missed; F3 missed -- AP +0.33 / +0.48 and 1-4 m recall +0.5 / +0.6 pp against the predicted +1 to +3, so the blind spot costs little once the decoder is right; M held through the vote grid (+0.78, later selected epochs) and **clears the three-seed bar under `nms-mean`, +1.74 with every M seed above every control seed** -- it gains far away (8-10 m +3.2 pp) and gives some back on close pairs (-4.2); D exceeded its +0.3 to +0.7 through the vote grid (+1.28) and gains on every seed from averaging (79.50%, above). **Six arms under two decoders is twelve comparisons**, so only M under `nms-mean` is read as a result; the rest are leads.

**DECIDED 2026-10-06 (owner): adopt `nms-mean` as stage 2's decoder, mirror augmentation, and nearest-person label assignment at 0.6 m; drop F3** (`memory/rejected-ideas.md`). Dropout 0.1 and weight averaging stay leads, settled by the combination run. Making the three the defaults, and removing F3's code, is the first step below; every stage-3 result so far rests on the old recipe and is reported as such until phase 3a retrains it. Was: adopt `nms-mean` and mirror augmentation; adopt L1's nearest assignment as the corrected label rule; drop F3. Next, one run of the combination -- `--mirror --label-assign nearest --dropout 0.1` with averaging, three seeds (~75 min) -- since the arms act on different things and may stack. Then phase 3a, stage 3 retrained on the best stage 2's `nms-mean` candidates, which is the only way the full detector's number moves.

**The combination run (owner's call 2026-10-06), on the new defaults** -- no-temporal, val selection now by `nms-mean` AP, a checkpoint every 0.25 epoch, three seeds each: **C0** the adopted recipe alone (mirror + nearest labels), which is also the new no-temporal baseline and phase 3a's stage 2 candidate; **C1** the same + `--dropout 0.1`. Then weight averaging and the probe. Read against the control's 79.75% and M's 81.49% (`nms-mean`, every 5th annotated frame). *Predicted, before the runs:* C0 81.5-82.5%, every seed above the control's best seed (80.45), so mirror and nearest labels add partly; C1 within +-0.5 of C0 as selected, and averaged C1 at least +0.3 over averaged C0, since dropout's gain showed mainly through averaging. The stage 2 for A60 and phase 3a is the arm with the higher three-seed mean, averaged if averaging gains on every seed of it, the owner confirming.

**Combination run, done 2026-10-06** (`results/stage2_probe_a59_combination.json`, `swa_a59_c*.json`; `nms-mean`, every 5th annotated frame, same frames as every arm above): **C0 81.29 / 82.36 / 81.16% (mean 81.60, +1.85 over the control)**, averaged 81.90; **C1 82.17 / 81.82 / 81.86% (mean 81.95, +2.20)**, averaged **82.28 / 82.08 / 82.22% (82.20)**, averaging gaining on every C1 seed (+0.11 / +0.26 / +0.36). C1 is also best at AP@0.3 (80.43 against 79.52 and the control's 77.46) and on close pairs (86.0% against mirror alone's 79.4% and the control's 83.6%): nearest labels add little AP over mirroring (+0.11) but recover part of its close-pair loss, and dropout the rest. **All four predictions held**: C0 in 81.5-82.5 with every seed above the control's best; C1 within +-0.5 of C0 (+0.35); averaged C1 over averaged C0 by +0.30, exactly the predicted minimum. **By the rule written above, A60's stage 2 is C1, averaged. Confirmed by the owner 2026-10-06, which adopts dropout 0.1 and weight averaging into stage 2's recipe** (`step2 --dropout` now defaults to 0.1; averaging is `utils/average_checkpoints.py`, which reproduced the scratch analysis's choice and val AP exactly on all three C1 seeds).

**The attribution ladder F1 / W / R: owner's go-ahead 2026-10-09.** Each arm changes one thing in today's stage 2 (C1: no-temporal, `nms-mean`, mirror, nearest labels, dropout 0.1) toward LFE's or DR-SPAAM's setting; three seeds each, into `checkpoints_three_horizon/a59_{f1_lfe_input,w_lfe_width,r_drspaam_recipe}_s{0,1,2}`. **Control: C1 itself**, 81.95% as selected and 82.20% averaged (`nms-mean`, every 5th annotated frame). Not retrained: today's trainer draws the same permutation and mirror flips as the code C1 ran on, and loads its weights bit-identically, so a fresh control would differ by GPU noise only.

- **F1** `step2 --input-features lfe --checkpoint-every 0.25`: LFE's `_normalize_scan`, `1 - r/10` clipped to 0.2-10 m, as the only input channel. Averaged and probed like C1.
- **W** `step2 --channels 8 16 32 --checkpoint-every 0.25`: LFE's widths, everything else ours. Averaged and probed like C1.
- **R** `step2 --batch 8 --max-epochs 5 --min-epochs 5 --schedule exponential --lr 1e-3 --min-lr 1e-6 --weight-decay 0 --select last`: DR-SPAAM's protocol on FROG (5 epochs at batch 8, per-iteration exponential decay 1e-3 -> 1e-6, Adam without weight decay, last weights, no averaging); AMSGrad, which DR-SPAAM also used, is left out. Its val-best checkpoint is scored too, which separates the duration-and-selection loss from the batch-and-schedule one.
- Code, tests first: `CalibrationNetwork(input_features=)` and `lfe_features`, `calibration_network` rebuilding `input_features` and `channels` from a checkpoint's args (`network_shape`), `exponential_schedule`, `--select {best,last}` saving `step2_calibration.last.pth`. **Built 2026-10-09** (19 tests with A61's, red first; full suite 681 passed; a real-data smoke run of every flag at once rebuilt exactly through the probe). **Launched 2026-10-09 00:48** as the first half of `checkpoints_three_horizon/run_queue_ladder_a61.ps1` (Task Scheduler `lidar_queue`, `progress.txt` lines `QUEUE ladder_*`): F1, W, R at three seeds, averaging, then `stage2_probe.py` over all 18 checkpoints into `results/stage2_probe_a59_ladder.json` (~4 h). To resume after a stop: `schtasks /run /tn lidar_queue`; finished steps are skipped.

**F1 trained 2026-10-09** (as selected, every annotated test frame, `nms-mean`): **81.60 / 81.19 / 81.60%, mean 81.46**, against C1's 82.12 / 81.80 / 81.86 (81.93): **-0.47 pp**, inside the three-seed bar and short of the predicted -1 to -3. Averaging and the range breakdown come from the probe. **W trained 2026-10-09** (63,891 parameters, same reading): **80.81 / 80.40 / 80.70%, mean 80.64, -1.29 pp**, every W seed below every C1 seed but the mean inside the bar, and short of the predicted -2 to -4: a 15x smaller network keeps all but ~1.3 pp. **R trained 2026-10-09** (last weights, same reading): **79.92 / 80.49 / 79.89%, mean 80.10, -1.83 pp**, the only arm past the three-seed bar, every R seed below every C1 seed; far above the predicted 75-78%. Its last weights trade recall for precision at the 0.3 threshold (1.98 FP and 0.38 FN per frame, three-seed means, against C1's 3.11 and 0.20). Its val-best checkpoints come from the probe.

**THE LADDER, SCORED 2026-10-09** (`results/stage2_probe_a59_ladder.json`; `nms-mean`, every 5th annotated test frame, the frames of every A59 arm; C1 from `stage2_probe_a59_combination.json` and `swa_a59_c1_*.json`; recall at LFE-Peaks' 1.78 FP/frame):

| arm, 3 seeds | parameters | selected | averaged | vs C1 | 6-8 m | 8-10 m | close pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 control | 976,707 | 81.95% | 82.20% | -- | 86.7% | 81.7% | 86.0% |
| F1 LFE's input alone | 976,419 | 81.47% | **82.34%** | -0.48 / **+0.14** | 85.3% | 79.4% | 86.6% |
| W LFE's widths | 63,891 | 80.66% | 80.70% | -1.29 / -1.50 | 84.5% | **76.9%** | 88.4% |
| R DR-SPAAM's protocol, last weights | 976,707 | 80.11% | -- | -1.84 | 83.9% | 80.2% | 81.7% |
| R, its val-best checkpoint (epoch 0.75-1.0) | 976,707 | 81.35% | -- | -0.60 | 86.3% | 80.3% | 86.2% |
| *LFE-Peaks, published weights* | 53,258 | 65.1% | -- | -- | 75.3% | 50.3% | 83.2% |

**Predictions scored.** F1 missed: -0.48 as selected and +0.14 averaged against the predicted -1 to -3; as selected its loss sits beyond 6 m (8-10 m -2.3) but also at 1-2 m (-1.7), and averaging erases the far-range loss. W missed on size (-1.3 to -1.5 against -2 to -4) but held on where: its loss is the far range (8-10 m -4.8, 6-8 m -2.2) while it *gains* on close pairs (+2.4). R's last weights missed far on the high side (80.11% against 75-78); its val-best checkpoint held (-0.60, within 1.5). **Every arm lost less than predicted.**

**What the ladder says, for the paper.** Neither LFE's input nor LFE's size nor DR-SPAAM's training protocol explains much of the lead: each costs our network at most ~1.8 pp, and every arm still scores 80-82% against LFE-Peaks' 65% and DR-SPAAM's published 73.7-75.6%. The input features are worth nothing measurable once weights are averaged; width is worth ~1.4 pp, all of it on far people; of R's -1.84, -1.24 is taking the last weights instead of the best on val (the memorisation curve: 1.98 FP and 0.38 FN per frame at 0.3, against C1's 3.11 and 0.20) and -0.60 is batch 8, the decaying rate and no weight decay. With phase 1's decomposition (about half the lead over LFE-Peaks is the output head: centre votes, aggregation, NMS), **what is left is the network's form and the training recipe** -- a ConvNeXt U-Net with a whole-scan aggregator reaches 80.7% at LFE's 64k parameters -- not its input, its size or its compute. The converse test, LFE retrained with its authors' code and its shuffling fixed (phase 3b), is what remains to separate LFE's architecture from LFE's training. **W's ms/frame, measured 2026-10-09:** no faster at batch 1. **F1 + W together, measured 2026-10-11 (item 12): the two costs do not add, because only one of them is real.** Three seeds, averaged and probed identically: **80.68 / 80.66 / 80.51, mean 80.62** -- which is W alone (80.70) to within **0.08 pp**. Simple additivity would predict 80.84, so the interaction is -0.22 pp, inside the spread. **LFE's input contributes nothing on top of its width, exactly as it contributed nothing alone**, and the width is the whole 1.5 pp. The ladder's conclusion therefore strengthens rather than softens: taking LFE's input *and* its width at once, our network still scores **80.62% against LFE-Peaks' 65.1%** -- a 15.5 pp gap that neither ingredient explains, nor their combination, nor DR-SPAAM's training protocol.

*Predictions restated on C1, 2026-10-09, before running* (the table above was written against the 2026-10-05 control): **F1** -1 to -3 pp, the loss concentrated beyond 6 m; **W** -2 to -4; **R** last weights 75-78% (-4 to -7), its val-best checkpoint within 1.5 pp of C1's 81.95, so the loss is duration and selection rather than batch or schedule.

F1, W and R are the attribution ladder the paper needs: each removes one candidate reason from our stage 2 toward LFE's or DR-SPAAM's setting. L1, L2, M, D and F3 are improvements. **Owner's call 2026-10-05, second round: run F3, M and D as well, then decide what to adopt.** Built as `step2 --median-windows-deg 11 33` (`beam_features`/`CalibrationNetwork(median_windows_deg=)`, extra channels appended after the ten, rebuilt from checkpoint args) and `step2 --mirror` (`mirror_clips`, `mirror_people`), tests first; D uses the existing `--dropout`. Queued after L1/L2 and their evaluation as `a59_{f3_median33,m_mirror,d_dropout01}_s{0,1,2}`, gated on the full test suite run while nothing else is (659 passed, 2026-10-05 15:07). F1, W and R wait for the owner. **F3 trained 2026-10-05**: 77.20 / 78.15 / 76.82% (vote grid, every annotated test frame), mean 77.39, +0.37 over the control, inside the noise; its 1-4 m recall and `nms-mean` reading come from the arms chain's probe. **M trained 2026-10-05**: 78.13 / 78.38 / 77.01%, mean 77.84, +0.82, under the bar; its selected checkpoints come later, at epochs 2.25 / 2.25 / 2.75 against the control's 0.75 / 2.00 / 1.00, so the "later peak" half of its prediction held and the size is the predicted +0.5 to +1.0. **D trained 2026-10-05/06** (seed 2 resumed from epoch 1.75, a different draw): 78.96 / 78.07 / 77.91%, mean 78.31, **+1.29**, under the bar and above its predicted +0.3 to +0.7. **Averaged** (val-chosen window, every 5th annotated frame; `results/swa_a59_*.json`): D **79.77 / 79.66 / 79.06%, mean 79.50, +1.89 over the averaged control (77.61)**, gaining on every seed (+0.71 / +1.54 / +1.10); M 78.39 (+0.78), F3 78.13 (+0.53). Over all eighteen A59 runs averaging gains in fifteen, +0.66 on average. Not comparisons registered in advance: suggestive until the probe reads them under `nms-mean`.

**Phase 3.** (a) **Done 2026-10-08 as A60's no-temporal half: the whole detector on the new recipe, 86.35 / 85.77 / 85.49%, mean 85.87.** A56's tracking baselines on the new candidates are still to re-run, so the paper's tracker comparison is on the old recipe until then. Was: re-cache stage-3 candidates with the chosen decoder from the best stage 2, and retrain E13 at three seeds (~1 h cache plus ~4.4 h per seed): the full detector's number. Every A51/A56/A58 figure rests on vote-grid candidates, so this changes the headline and must be reported beside it, not instead of it. (b) Retrain LFE with its authors' published code, shuffling fixed and selected on our val AP: the converse of R. This supersedes A32c's port (below). **Plan proposed 2026-10-09, awaiting the owner's go-ahead:**

- **What was checked.** `include/FROG-data/frog_11-36_12-43_train_val.h5` *is* the authors' training file (`export_dataset.py`: train recordings 11-36 and 12-43, a random 10% of their frames as val). Their recipe (`train_segmentation.py`, `utils/models.py`): five residual blocks of depthwise-separable convolutions (32 channels, kernels 9/7/5, BatchNorm, dropout 0.2, global max on blocks 2-5), loss `0.5 * (Dice + weighted CE, beta 1.5)`, AdamW 1e-3 with weight decay 0.004, batch 32, scans with at least one person, up to 100 epochs, early stop on val *loss* with patience 20, best weights restored; `SegDataGenerator` never shuffles. It needs TensorFlow <= 2.15 (Keras 2 API), so Python 3.9 here and CPU only. The ONNX graph keeps BatchNorm as `Mul`/`Add` nodes, so a port can load the published weights exactly.
- **Proposed route: a PyTorch port of LFE-Peaks**, tests first, proven by loading the published ONNX weights and matching the ONNX outputs (~1.5-2 h). Then train it on our GPU with their recipe, scored by `stage2_probe.py`'s LFE-Peaks decoder on the frames of every A59 arm. Three arms, one seed each first: as published (no shuffle, interleaved val, early stop on loss; must reproduce ~65% or nothing else counts), shuffle fixed, and shuffle fixed plus selection by AP on FROG's held-out val recordings. ~3-6 h unattended. The alternative is their Keras code in a Python 3.9 + TF 2.15 environment, CPU only, several hours per arm, plus an export to score it.

**Why.** The paper must say why a single-frame network scores 77-80%, and right now it can say only where (far and isolated people) and what it is not (the decoder, capacity against the cutouts, compute). Phase 2's ladder supplies the why. Phase 1 and the label fix are free or nearly free accuracy, and stage 3 inherits all of it.

### A58. A merge radius on the memory's output -- **DONE 2026-10-05, NEGATIVE** (`memory/rejected-ideas.md`)

**Result** (`checkpoints_three_horizon/a56/merge.json`, `merge_oracle.json`): **no output gains from a merge.** Val picks no merge for E14, B0 and E10 and 0.10-0.15 m for E13's seeds; test AP moves -0.04 to -0.06 pp where anything is merged at all, and falls monotonically past 0.15-0.20 m (E13 -0.6 at 0.30 m, -2.3 at 0.50 m). The two detections of a double report sit a median 0.32 m apart, a neighbour's distance, so a radius that catches them also removes close neighbours. 85-87% are a candidate with a coasting slot beside it. **An oracle bounds any fix**: removing every detection that matches nobody within 0.5 m of a person, with the labels known, buys E13 +0.43 to +0.55 pp and E14 +0.52; slots alone +0.26 to +0.41. E13 over E14 with both merged: +2.90 / +3.16 / +2.51, all intervals above zero.

**Predictions scored:** (1) missed: no radius in 0.2-0.4 m is chosen and nothing is gained; (2) held, vacuously: the candidates move -0.06 and E14 is not merged at all; (3) held: candidate + slot is 85-87% of E13's double reports (77% of E14's); (4) held: the smallest seed is +2.51.

**What.** The memory emits two detections within 0.5 m of one person far more often than anything before it: at matched false positives, 12-15% of people for E13, 8-12% for B0, 5% for E10, 6.5% for the tuned ByteTrack + CBMOT, against 1.7% for the stage-2 candidates (`checkpoints_three_horizon/a56/duplicates/unlinked.json`). Off FROG's duplicate labels every second detection is a false positive, and stage 3 has no merge step; the stage-2 decoder and both LFE detectors do, and the merge radius is the best-priced precision lever measured here (21.9 FP per FN, `memory/rejected-ideas.md`).

**How.** `tracking_baselines.py merge`, on the saved A56 frames, no retraining. `merge_detections` is `lfe_detector._merge_nearby`'s greedy merge (highest score first, drop anything within `r` of a kept detection), vectorised and pinned to it by a test. For every output -- candidates, E14, E10 x2, E13 x3, B0 x3 -- sweep `r` in {none, 0.10 ... 0.60} m on val (every 5th annotated frame, as E14 was tuned), pick each output's best, score test once; the full test curve is reported but not used for choosing. Block bootstrap of merged against unmerged per output, and of each learned arm merged against E14 merged, on the official labels and with the duplicate labels dropped. Then the anatomy of a double report at the matched-FP threshold: two candidates, a candidate and a coasting slot, or two slots (memory frames list candidates at their own positions first, so the split is exact).

**Predicted before running:** (1) the memory arms choose `r` in 0.2-0.4 m and gain +0.3 to +1.0 pp test AP, interval above zero; (2) the candidates gain under 0.1 pp (already merged at decoding) and E14 less than E13; (3) most double reports are a candidate beside a coasting slot, which would mean association lost the person and a slot coasted on next to them; (4) E13 over E14, both merged, stays at least +2.5 pp.

**Why.** If it moves AP, the fix is a post-merge (cheap, and it changes the headline) or a training target that stops two outputs claiming one person (the mechanism). If it does not, the double reports sit low in the ranking, which is worth knowing too.

### A57. Replace the PowerShell launchers with one typed Python queue runner -- **DONE 2026-10-09**

**What landed.** `utils/run_queue.py`: a `Job` dataclass (`name`, `cli`, `outputs`, `needs`, `cwd`), a `StrEnum` `Outcome` of
`DONE` / `SKIP` / `FAILED` / `BLOCKED`, and `run(jobs, *, spawn, progress, log_root, gate, steps)`. A queue is now a committed
Python module that builds jobs and calls `main`, which takes `--list`, `--steps`, `--progress` and `--log-root`; the module is in
git, so the exact queue behind a result outlives the session that ran it, which is the gap that left T1 and T2 unverifiable.

**The rules it enforces, each one a copied launcher's mistake:** a job runs only when every `needs` has finished, so **a failed
job's dependents are recorded `BLOCKED` and never start** (the copies waited for an idle machine, and a stopped chain makes the
machine idle); a job whose `outputs` all exist is skipped without spawning; a job with no outputs is judged by its exit code,
which is how the test gate works; each job gets three attempts with `--resume`, except that a **training guard in the log stops it
after one**; every attempt keeps its own log, `<job>.<stamp>.attempt<n>.log`; and the progress lines are byte-identical to the
PowerShell format (`QUEUE <job> attempt N START/END/RETRY`, `SKIP, already done`, `STOPPED, a training guard fired`), so the
watchers in `memory/long-running-tasks.md` work unchanged. One line is new: `QUEUE <job> BLOCKED, <dep> did not finish`.

**40 tests** in `tests/test_run_queue.py`, all with the process spawn behind a seam, so none needs a GPU: dependency order against
declaration order, transitive blocking, an independent branch surviving another's failure, the gate running first and not being
retried, skip-if-outputs (and only when *every* output exists), `--steps`, the retry limit, a guard stopping a retry, one log per
attempt, the progress markers, and `spawn_process` itself against real processes (exit code, stdout and stderr in the log, the
working directory, and a job that lingers after writing its outputs being waited out and killed rather than polled forever).

**One behaviour deliberately improved rather than ported:** the PowerShell loop slept a fixed 30 s between checks, so a job that
finished in a second still cost 30; the runner waits on the process with a 30 s timeout instead, which is why the spawn tests run
in under a second.

**Original plan, kept for the reasons it was written:**

**What.** `utils/run_queue.py`: one program that takes a list of jobs and runs them all, replacing the ~60-line PowerShell
launcher that has now been hand-copied eight times (`run_arms.ps1`, `run_arms2.ps1`, `run_b0fresh.ps1`, `run_e10.ps1`,
`run_e12.ps1`, `run_e12ab.ps1`, `run_s2.ps1`, `run_s3.ps1`). Do it when the machine is free, not beside a run.

**Why.** Every incident in this session traces to the launcher rather than to the research:

| incident | what the duplicated script did |
| --- | --- |
| S3 trained ahead of a failed test gate | each copy waits for an *idle machine*, and a chain that stops on failure makes the machine idle |
| a 6.8 h run died silently at 08:48 | the launcher was a child of the agent session, so a teardown killed the tree |
| `\r` became a carriage return, twice | the scripts are built by string substitution rather than written as code |
| a wrong completion marker nearly missed a finish | each copy greps a different hand-typed string |
| `Wait-Finished` hardcoded `step3a` | copied for a `step3b` run and silently never matched |

**How**, following `memory/coding-guidelines.md` -- type hints throughout, `from module import name`, constants at module level,
one docstring per function, standard library only so it runs on a fresh clone:

- A `Job` dataclass: `name`, `step` (`step3a` / `step3b`), `cli`, and `needs: tuple[str, ...]` -- explicit dependencies rather
  than the implicit "wait until nothing is running". **Order within the queue carries no meaning** (`completion-over-ordering`):
  the runner takes anything whose dependencies are met and whose results file is absent.
- `run(jobs, *, gate=pytest)`: runs the gate once, then each job up to three times with `--resume`, stopping that job on a
  training-guard match and recording the outcome. **A failed job never releases the machine to a job that depended on it.**
- One `Outcome` per job -- done, skipped, failed, blocked -- printed as a table at the end and appended to `progress.txt` in the
  existing format, so the watchers keep working unchanged.
- Started detached (`memory/long-running-tasks.md`), so it outlives the session that launched it.
- Tests in `tests/test_run_queue.py`: dependency resolution, the skip-if-results rule, retry limits, guard classification,
  blocked-on-failed-dependency, and marker formatting -- with the process spawn behind a seam so none of it needs a GPU.

**Tier.** After the S2/S3 queue drains. It is tooling, not research, so it waits for a gap rather than taking machine time from
an experiment.

### A51. Stage-3 slots: performance, trainability and output

> ## Closing out the slots work -- scope set by the owner, 2026-09-30
>
> **Three items, and deliberately not more.** An earlier draft of this block scheduled the tracking-baseline work (A56) and a
> coarse-convolution arm as well; the owner cut both back to later sessions. **A56 stays awaiting the owner and is not
> approved.** Machine time is single-threaded -- nothing that imports torch runs beside a training job
> (`memory/gotchas.md`) -- so these run in order.
>
> | # | question | cost | status |
> | --- | --- | --- | --- |
> | ~~**S1**~~ | Does E10's headline survive a second seed? | ~5 h | **done 2026-10-01: partly.** AP -1.40 is now **inside** its 1.45 bar; AP@0.3 (-1.30), recall at matched FP (-1.60, 1.9x) and coasting precision (-13.80, 1.7x) clear. Three of four, not four |
> | ~~**S2**~~ | Can the coasting head tell a remembered person from a remembered phantom, if it is told what is occluded? | ~5 h + code | **done 2026-10-01/02: no end to end, though the ranking did improve.** At matched false-positive cost the coasting head recovers **6-13% more people**; but the rest of the model turns conservative, covering 2,240 fewer before coasting starts, so recall at each model's own operating point is flat (92.58 against 92.47). Timing cleared: **4.74 ms at 256 slots** on a quiet host, not the 7.76 measured under load. Original fixed-threshold read: coasting precision 27.38 against 25.43 (+1.95, bar 8.19), reporting **58% fewer** slots to get it -- conservative, not discriminating. Every barred metric inside its bar. **Two follow-ups queued:** the coasting sweep on this checkpoint, for precision at a *matched* report rate rather than a fixed threshold; and a re-timing of its 7.76 ms/frame on a quiet host |
> | ~~**S3**~~ | Does training with jittered odometry repair the room frame? (E11) | ~5 h + code | **done 2026-10-02: partly, and for free.** Room-frame AP **+1.75 pp at 0.05 m/s and +4.34 at 0.1**, -0.38 at zero; every clean-odometry metric inside its bar. **The crossover with the sensor frame does not move** (~0.015 m/s), so D1 stays conditional |
>
> **Left for later sessions, with their plans intact:** A56's tracking baselines and metrics (E14-E22; approved and run
> 2026-10-02/03, see A56), the coarse-convolution interaction (§D / A51 phase 4), extra seeds for the single-seed screening arms (D2, D3, O1,
> O2, T1, T2), and the cross-dataset work (A49).
>
> ~~**Still open, small and cheap:**~~ **done 2026-10-09:** `slot_experiments.py` took a hardcoded output path, and on 2026-10-02
> the `odometry` and `cost` modes overwrote E2's `a51_odometry.json` and E4's `a51_cost.json`. Every mode now takes `--out`, the
> default is still one file per mode, and the experiments are reached through an `EXPERIMENTS` table the CLI's own choices come
> from, so a mode cannot be added without a name. Both sets of overwritten numbers survive -- E2's in `results/a51_odometry.log`,
> E4's in `CHANGELOG.md` and `memory/slot-design-evidence.md`.
>
> ## Re-based on the adopted recipe, 2026-10-09 -- and one bar has disappeared
>
> **Every arm in this item was screened against B0 on `step2_static`, and that backbone is superseded twice over** (A59's recipe,
> A60's removal of the coarse convolution). A61 already produced the replacement control: **E13 on C1**, three seeds, read from
> `a60_e13_static_s{0,1,2}/step3a_results.json`:
>
> | metric | seed 0 / 1 / 2 | mean | sd | three-sigma bar | B0's old bar |
> | --- | --- | --- | --- | --- | --- |
> | AP | 86.35 / 85.77 / 85.49 | 85.87 | 0.44 | **1.32** | 1.45 |
> | AP at 0.3 m | 85.24 / 84.65 / 84.16 | 84.68 | 0.54 | **1.62** | 1.29 |
> | recall at matched false positives | 96.00 / 95.86 / 94.56 | 95.47 | 0.79 | **2.37** | 0.86 |
> | coasting precision | 33.3 / none emitted / 10.0 | -- | -- | **gone** | 8.19 |
>
> Its candidates are 82.22 / 82.11 / 82.23 (mean 82.19, sd 0.07), the memory's own operating threshold has fallen to 0.071-0.097,
> and it costs 5.17-5.22 ms per frame.
>
> **Coasting has gone inert, and that retires a bar.** The memory emits **0.0012 / 0.0000 / 0.0020 coasting reports per scored
> frame** against B0's 0.176 -- a factor of ~100, with seed 1 emitting none at all, which is why its precision is undefined.
> Nothing is wrong: stage 2's recall at its own operating point rose to 95.47%, so there is almost nobody left for a slot to
> report on its own. **Three consequences.** (1) The screening protocol is now three bars, not four, and the tightest is AP's
> 1.32. (2) **The arms that aimed at coasting have nothing left to aim at** -- O2 (`--coast-weight 3`) and S2 (free-space
> evidence for the coasting head) were already nulls end to end, and on this recipe the quantity they improve is ~0.001 reports
> per frame. (3) Any paper sentence about the coasting head must say which backbone it was measured on.
>
> **The decision this item now needs (owner's call).** The deferred single-seed arms -- D2 `--latest-value`, D3
> `--fixed-decay-s 8`, T1 `--no-rescore-gate`, T2 `--schedule plateau`, O1 bounded rescoring, O2 `--coast-weight 3` -- cannot be
> "given extra seeds": on this recipe they are *new screens* against a new control, about 5 h each, so roughly **two GPU-days for
> the six**. Either:
>
> - **(a) re-screen them on the adopted recipe** -- every design-choice number in the paper then describes the model being
>   proposed, and the three-bar protocol applies cleanly; or
> - **(b) report them as they stand**, each labelled with the backbone it was run on, and say plainly that the design choices were
>   screened on a superseded stage 2 while the adopted configuration (E6+E7, i.e. E13) was confirmed at three seeds on the current
>   one.
>
> **(b) is defensible and costs nothing**, because none of the six changed the adopted design: D2 and O2 were nulls, D3 and T2
> were inside their bars, T1 was inside on AP, and O1 never finished. **(a) is what a reviewer would prefer** if they notice the
> backbone changed under the ablation table. The write-up cannot stay silent on it either way.
>
> **Still open inside the item, unaffected by the decision:** D4 (`step3a --learn-value`, implemented 2026-09-20, never run) and
> D5 (a learned association width, not implemented), which are the live form of A50 phase 1b item 4 -- whether the hand-coded
> bookkeeping's constants are the right ones. D4 is one run; D5 needs code first, and A51's own reading is that it is "only worth
> it if D4 moves".
>
> **Closed, with reasons, so they are not reopened by accident:**
>
> - **Feedback / re-injection.** Measured null in three configurations, cause measured (0.012-0.045% of the layer's input),
>   ceiling measured (1.91% of people, all already coasting). A widened re-test is possible and is **not** scheduled: it chases a
>   fraction of that ceiling against a coasting head that already reaches the same people. **S2 is the same prize by a shorter
>   route.**
> - **Deleting the feedback code.** Kept deliberately (owner, 2026-09-30) so E12's three runs stay reproducible until the paper's
>   numbers are frozen.
> - **Slot capacity.** 24 of 256 slots in median use, p99 84; not the limit.


> **Where this stands, 2026-09-30 (session close).** The architecture question is **settled**; one deployment item is left.
>
> | question | answer | evidence |
> | --- | --- | --- |
> | Does the learned slot state earn its place? | **Yes, and it is the only part that does** -- though AP alone is inside its bar. -1.40 AP (bar 1.45), -1.30 AP@0.3, **-1.60 recall at matched FP (1.9x)**, **-13.80 coasting precision (1.7x)** | E10, two seeds vs a three-seed control |
> | Is the attention needed? | **No.** Ties on every metric at three seeds, -43% parameters, -18% ms/frame | E6, E7, E13 |
> | Does feeding memory back into stage 2 work? | **No, in three configurations including a ground-truth prior with no competitor.** Cause measured: the prior carries 0.012-0.045% of that layer's input | E12a, E12b, E12ab |
> | Was there anything for feedback to win? | **1.91% of people at most**, and every one already held by a coasting slot | `memory_diagnosis.py --recovery`, no training |
>
> **What to propose:** E6+E7 -- rule-based association, learned per-slot recurrence, rescoring head -- with the full model as the
> ablation and the feedback path removed along with the `step3b` joint stage it existed for.
>
> **Provenance of every arm, resolved 2026-09-30.** `judge_arms.py` used to flag T1 and T2 as unverified, because their launcher
> scripts are no longer on disk. The checkpoints answer it better than a launcher could: each stores its own CLI, and
> `torch.load(<run>/step3a_object_memory.best.pth)["args"]` shows **T1 and T2 share the control's settings exactly** --
> `step3a` on `step2_static`, `chunk_s [1.0, 10.0]`, `test_strides [5]`, `max_hours 24`, 256 slots, seed 0, cosine schedule --
> differing only in their own intervention (`rescore_gate False` for T1, `schedule plateau` for T2). The flag is removed. Read
> args from the checkpoint, not from a script that may not survive the session.
>
> **Running 2026-09-30: `a51_e10_norecurrent_s1`, a second seed for E10.** E10 became the headline claim after its first seed,
> and its AP margin is 1.50 against a 1.45 bar -- 1.03x. This session already watched a two-seed reading of the tightest-barred
> metric flip when the third seed landed (E13's recall at matched FP, +1.08 -> +0.68). One seed under a headline is not enough.
>
> **Still open in this item:** **E11** (train with jittered odometry), which is deployment robustness rather than accuracy, and
> the one place the design still carries a caveat (D1/E2: room coordinates win only below ~0.02 m/s of drift).
>
> **Not queued, owner's call:** a fair re-test of feedback with a widened, densified injection. Now priced -- it chases a
> fraction of 1.91 pp against a coasting head that already reaches the same people and fails on precision, not coverage.
> **Coasting precision is the cheaper door to that same recall.**
>
> **Method findings that outlive the item:** stage-3 training is bit-reproducible from its seed here and `--resume` is what
> breaks it (`memory/interpreting-evaluation.md`, corrected); nothing that imports torch can run beside a training job
> (`memory/gotchas.md`, the commit limit); and a two-seed reading of the tightest-barred metric was wrong once this session
> before the third seed corrected it.


**Owner's plan, confirmed 2026-09-17.** It takes over A50 phase 1b and the stage-3 rows of A50 phase 3; A50 keeps the rest for later.

**What.** Make stage 3 (`ObjectMemory`) train reliably, report well-calibrated detections, and take more of the room the oracle leaves: +6.5 pp coverage at a 2 s bridge above stage 2, of which stage 3 took 17.5% (A50 phase 1 item 1).

**Scope decisions (owner, 2026-09-17):**

- **Backbone: the no-temporal stage 2** (`step2_static`, `--coarse-lags 0`; seeds 0/1/2 at 77.60 / 77.40 / 76.68% under current decoding). Coarse convolutions stay out of every run in this item, because their benefit is not established. The §D slots-versus-coarse-convolution comparison is therefore out of scope here.
- **The 40 Hz frame rate and `dt` are fixed before any experiment.** No stage-3 result exists on this backbone yet, so fixing now costs no comparability.
- **Joint training and feedback (step 3b) are out of scope.** They need stage 2 retrained. With a no-temporal backbone, feedback is the only way memory reaches stage 2, so it is the natural follow-up.
- **Screening uses a shortened curriculum** (1 s + 10 s chunks), and confirmation uses the full one, provided B0 shows the 30 s stage buys nothing beyond noise.

**How every arm is judged:** A50's screening-then-confirmation procedure, against the **stage-3 noise floor measured in B0**, not stage 2's ~0.95 pp per run. Each arm reports wp-AUC at 0.5 m and 0.3 m, ms/frame, FP and FN per frame at 0.3 **and** at stage 3's own threshold (the one matching stage 2's FP per frame), coasting detections and their precision, and the rescoring shift on true positives against the rest. Changes are adopted cumulatively: each screen runs against the best configuration so far.

**Phase 0 -- harness and baseline**

| # | item | how | why |
| --- | --- | --- | --- |
| 0a | ~~measure the `dt` suspect~~ **done 2026-09-17: confirmed, and worse than suspected** | `memory_diagnosis.py --slot-speed` replays the rule-based `manage_slots` on the old test cache (50,088 frames) with stamped `dt` and with the mean period; reference speed is each slot's displacement over >= 1 s | see the result below the table |
| 0b | ~~fix the frame rate and `dt`~~ **done 2026-09-17** | `FRAME_PERIOD_S` removed. Chunk and validation lengths come from `mean_frame_period` (elapsed time over frames) of their own split. Stage 3's `dt` is `frame_periods`: a causal running average of the stamped intervals, the plain mean for the first `DT_AVERAGE_FRAMES` = 32 and an exponential average with rate 1/32 after (owner's choice over a last-N-frames window: one number per stream), computed once per recording in `pack` and in the joint path. Speed clamping moved to T3, since a sane `dt` gives sane speeds | chunk and validation lengths were 1.53x off; velocity and its input were corrupted |
| 0c | ~~extend stage-3 evaluation~~ **done 2026-09-17** | `evaluate` returns `own_threshold` / `own_recall` / `own_fp_per_frame` / `own_fn_per_frame` (budget: the candidates' FP per frame), `rescore_shift_person` / `_other`, `coast_per_frame` / `coast_precision`, and `ap_0.3m` via `extra_radii`; `memory_ms_per_frame` at one stream; step 3a's test report prints all of them. Val selection is unchanged | calibration and coasting are unmeasured; AGENTS rule 2b needs ms/frame |
| 0d | ~~candidate cache for `step2_static` seed 0~~ **done 2026-09-17**: test 402 s, train 976 s, val 1,535 s (log `results/cache_static_s0.log`) | `load_split` with `--stage2` (~1 h) | the old caches belong to a stage 2 that no longer loads |
| 0e | ~~B0 baseline and noise floor~~ **done 2026-09-19 at three seeds** (`checkpoints_three_horizon/a51_b0_short_s*`); the full-curriculum run was stopped 1.9 h in, inside its first stage, by the owner's call. Two nights of GPU were lost first: one to the `device.type` crash, one because the run was scheduled for an evening and never launched, so the relaunch was stopped after its smoke step passed and the chain is queued for the evening; the cheap phase-3 sweeps run in the meantime. **First launch died in its smoke step 2026-09-17 19:05** (`memory_ms_per_frame` read `device.type` on a string; `memory/gotchas.md`), so the chain stopped and the GPU idled overnight; **relaunched 2026-09-18 14:12** as a detached chain (session scratchpad `run_b0.ps1`, progress in `checkpoints_three_horizon/progress.txt`): it waits for 0d, runs a smoke step (`a51_smoke`), then `a51_b0_short_s0/s1/s2` (`--chunk-s 1 10`) and `a51_b0_full_s0`, all `--max-hours 24` so early stopping, not the old 6 h cap that ended the first step 3a, decides; each writes to `checkpoints_three_horizon/<name>/` | one full-curriculum step 3a run, then short-curriculum runs at stage-3 seeds 0/1/2 on the same cache (~13 h) | the noise floor every screen is judged against; decides whether 30 s chunks earn their 3.5 h |

**0a result** (`checkpoints_three_horizon/slot_speed.json`; mean frame period 24.99 ms, 20.2% of stamped intervals clipped to 1 ms):

| `dt` | slot matches per annotated frame | velocity state near a person, p50 / p90 / p99 | windowed reference near a person, p50 / p90 | windowed, slots elsewhere, p50 / p90 |
| --- | --- | --- | --- | --- |
| stamped (training today) | 8.51 | 3.97 / 8.63 / 15.22 m/s, 63.9% over 3 m/s | 0.66 / 1.67 m/s | **10.83 / 25.27 m/s** |
| mean period | 8.84 | 0.58 / 1.16 / 1.66 m/s, 0.0% over 3 m/s | 0.56 / 1.10 m/s | 0.59 / 1.75 m/s |

With stamped `dt`, a matched slot's velocity is ~7x a walking pace. Unmatched slots then coast away at ~11 m/s, so a slot that misses one frame is predicted metres off and loses its object. With the mean period, velocity agrees with the displacement reference to 0.02 m/s and matches rise 3.9%. The earlier stage-3 results (80.10%) were trained and scored with this corruption.

**0b result.** Replayed the same way, the running average gives 8.84 matches per frame and velocity p50 / p90 0.58 / 1.17 m/s near people against the reference's 0.56 / 1.10, indistinguishable from the constant mean period. On the test split its `dt` stays within 24.23-25.82 ms (p0.1-p99) once 32 intervals have passed; a recording that starts with a sub-ms interval begins at the 1 ms floor for its first few frames. The first slot-speed replay's "mean period" row is superseded by this "running average" row in `checkpoints_three_horizon/slot_speed.json`.

**B0 result, 2026-09-19.** Short curriculum (1 s + 10 s chunks), no-temporal stage 2 seed 0, whole `official` test split, every 5th annotated frame. Every row shares the same candidates: 77.66% AP, 88.8% recall at 2.620 false positives per frame.

| seed | memory AP | vs candidates | AP at 0.3 m | recall at the candidates' FP rate | own threshold | coasting | hours |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 82.96% | +5.30 | 81.36% | 92.2% | 0.178 | 0.083/frame, 27.4% | 4.1 |
| 1 | 83.17% | +5.51 | 81.52% | 92.5% | 0.227 | 0.292/frame, 22.3% | 5.0 |
| 2 | 83.88% | +6.22 | 82.17% | 92.7% | 0.156 | 0.153/frame, 26.6% | 9.7 |
| **mean** | **83.34%** (sd 0.47) | **+5.68** | 81.68% | 92.5% | — | — | — |

- **The screening threshold for A51 is ~1.5 pp of test AP** (three seeds, sd 0.47, range 0.92), about half of stage 2's ~0.95 pp per-run spread. State it with every later arm.
- **The memory pays on both axes now**: +5.68 pp of AP over its own candidates against +3.2 pp for the old chain, and +3.7 pp of recall at matched false positives. 83.34% against the old 80.10% is **not** a clean comparison -- different backbone, and the old number predates the running time step and the limit-beam decoding.
- **Stage 3 alone costs 4.5-4.6 ms per frame** at one stream, inside FROG's 25 ms budget beside stage 2's 6.5 ms.
- **The rescoring shift survives training** at every seed (people -2.8 to -3.5 logits), so T1 and O1 still have a target; **coasting is alive** (0.08-0.29 reports per frame at 22-27% precision) where it was dead at one epoch, so O2 is about improving it rather than starting it.
- **Cost varies 2.4x across seeds** (4.1-9.7 h) through early stopping alone, and validation is 160 s against 70 s of training per epoch in the 1 s stage -- about 70% of the run is scoring. **T4 now covers cutting that** before the screens.
- **Open, deliberately:** whether 30 s chunks add anything over 1 s + 10 s. The full-curriculum run is ~1 GPU day and was stopped; revisit only if an arm looks chunk-length sensitive.

**Phase 1 -- trainability**, one screening run each:

| # | item | why |
| --- | --- | --- |
| T1 | replace the scalar `rescore_gate` with a zero-initialised last rescore layer | **screened 2026-09-20: 84.39% against B0's 83.34% mean** (+1.05, inside the ~1.5 pp threshold, so no win on AP), but **recall at matched false positives 93.4% against 92.2 / 92.5 / 92.7%** and **coasting precision 32.2% against 22.3-27.4%**, both above every B0 seed. One seed. 367 min, 306,442 parameters against B0's 306,443 -- exactly the dropped scalar. **Implemented 2026-09-18** as `step3a --no-rescore-gate`; the default keeps the gate, so B0 is unaffected. A test pins the reason: with the gate, the rescoring head's own gradient is exactly zero over the first steps; without it, it is not. The gate stayed at +0.03-0.05 through step 3a, so a -3 logit shift needed MLP outputs near -70 |
| T2 | plateau learning-rate schedule, as in step 2 | **screened 2026-09-20: 83.89% against B0's 83.34% mean** (+0.55, inside the threshold and inside B0's own range), recall at matched false positives **93.2%** and coasting precision **32.0%**, both above every B0 seed; 500 min, the longest screen yet. **Implemented 2026-09-20** as `step3a --schedule plateau` (`--lr-factor`, `--lr-patience`, `--min-lr`; `memory_schedule` picks it and says whether it steps per iteration), running after T1 as `a51_t2_s0`. Cosine over `--max-epochs` hardly decays in the 9-37 epochs a stage runs (A50 phase 1 item 5) |
| T3 | normalise slot statistics: log-scale age and time since match, clamp speed | unscaled inputs to the learned update. **Demoted 2026-09-20**: the running time step (0b) already brought slot speeds to 0.6 m/s, so the input that most needed clamping is no longer extreme |
| T4 | cut the screening cost | **partly done 2026-09-20**: `--test-strides` lets a screen score the test split once (stride 5) instead of twice, saving ~25 min per run and changing nothing about training. **Shrinking validation is deliberately not done yet**: it changes checkpoint selection, so arms would stop being comparable with B0 unless B0 is re-run under the cheap protocol (3 runs, ~8 h). Owner's call 2026-09-20 -- screen broad sweeps on a smaller fraction first, full evaluation for the finalists -- so re-baseline once if more than about four arms are queued at once. 30 s chunks at several streams only if the long chunks come back |

**Phase 1 read, 2026-09-20.** Both screens behave the same way, which is itself the finding:

| arm | test AP | recall at the candidates' FP rate | coasting precision | shift on non-people | hours |
| --- | --- | --- | --- | --- | --- |
| B0, 3 seeds | 82.96 / 83.17 / 83.88 (mean 83.34) | 92.2 / 92.5 / 92.7 | 22.3 / 26.6 / 27.4 | about +0.2 | 4.1-9.7 |
| T1, no scalar gate | 84.39 (+1.05) | **93.4** | **32.2** | -0.87 | 6.1 |
| T2, plateau schedule | 83.89 (+0.55) | **93.2** | **32.0** | -0.71 | 8.3 |

- **Neither moves AP past the ~1.5 pp threshold**, and T2's 83.89 sits inside B0's own range.
- **Both beat every B0 seed on the calibration metrics**, by almost the same amount, from two unrelated changes. Two readings, and this session cannot separate them: either any change that lets the rescoring head train harder produces this, or the two metrics are noisier across runs than B0's three seeds suggest (their B0 range is only 0.5 pp for recall and 5.1 pp for coasting precision). **No threshold was ever pre-registered for the calibration metrics**, which is the gap to close before either arm is adopted.
- **What follows:** neither is adopted yet. Confirmation seeds for T1 -- the arm with a mechanism behind it, and the cheaper of the two -- belong in phase 4 alongside the winner's re-run, not in a bespoke pair of runs now. T3 stays demoted.

**Phase 2 -- output**

| # | item | why |
| --- | --- | --- |
| O1 | bounded rescoring against the signed one | **unparked 2026-09-25 (owner's call): re-run it once the queue frees.** Three things changed since it was parked: the loop now reports the first non-finite iteration with the offending tensors, parameters and gradients and saves that chunk (`nan_diagnosis`), `--resume` means a crash costs one evaluation, and the launcher retries a crash but **not** a guard, so a genuine divergence stops immediately with its diagnosis instead of burning hours. Run it as `--no-rescore-gate --rescore-limit 3`, the configuration whose failure is cleanest to read. **Failed twice 2026-09-22, cause not established.** Both attempts reached the 10 s chunk stage and went NaN there (4.4 h with the gate, 2.4 h without), while the 1 s stage ran 29-62 epochs cleanly at val AP 77-80%. The gate explanation is **wrong**: the gateless arm diverged too. The decay-layer explanation is **also weak**: at the 10 s stage O1's `decay.weight` norm is 16.06 against B0's 14.52, with near-identical implied decay times. **The honest state: O1 failed 2 of 2 while eight other runs completed, but the machine also had two genuine hardware failures in the same window, and those stories are not separable from this evidence.** The discriminating test is whether the known-good D1 seeds now complete, and **it has not answered yet**: the first attempt after O1 died at 3 h of host memory exhaustion because an agent ran `pytest` beside it (`memory/gotchas.md`), which is self-inflicted and says nothing about the machine or the arm. Relaunched 2026-09-22 20:07. Do not spend a third O1 attempt before it answers. First attempt detail: `--rescore-limit 3` with the scalar gate kept ran 62 epochs at val AP ~80%, during which **the gate grew to +1.55 against B0's +0.05**, then the loss went NaN at 4.4 h. Bounding the correction removes what kept the gate small, leaving two multiplicative scales (`gate x limit x tanh`), so it is re-run with T1's parameterisation (`--no-rescore-gate --rescore-limit 3`). The candidates' AP held at 72.85% throughout, so this was the model diverging, not the device fault of 2026-09-21. **Implemented 2026-09-18** as `step3a --rescore-limit L`, which bounds the correction to +-L logits through a tanh; 0, the default, is today's unbounded correction. The head shifts the whole score distribution down (A50 phase 1b item 1), by -3.36 logits on people against +0.04 on everything else in the 2026-09-18 smoke run |
| O2 | coasting: make the head separate a remembered person from a remembered phantom | **measured 2026-09-22** (`--recovery`): a live slot sits on 95.5% of the people stage 3 fails to report, median value 0.986, while only 24 of 256 slots are in use -- so capacity is not the limit and reporting is. Thresholding cannot fix it: 3.0 FP per person recovered at 0.3, 4.6 at 0.2, 11.1 at 0.1, and the learned head already beats ranking by the rule's value. `step3a --coast-weight 3` ran 2026-09-26 (433 min): **AP 84.04, recall at matched FP 92.86, coasting precision 32.00 at 0.107 reports per frame** -- **a null on every bar** (+0.70 / +0.39 / +6.57 against 1.45 / 0.86 / 8.19). Weighting the supervision three times higher is not what the head is short of. Note it emits *fewer* reports than B0 (0.107 against 0.176) at the same precision, so the weight traded recall for nothing. **Next lever, if any: the head's inputs, not its loss** -- it reads `state` and the slot statistics, and the statistics it would need to separate a remembered person from a remembered chair (how far the slot has moved, its hit rate) are already there, which argues the limit is the 2-frame-sparse supervision rather than the design |
| O3 | position refinement: slots regress a correction to the candidate's position, L1 loss (`docs/PROPOSAL.md` §8) | outputs are raw candidate positions; should show in AP at 0.3 m |

**Phase 3 -- slot mechanics**, evaluation-only sweeps before any retraining:

| # | item | why |
| --- | --- | --- |
| S1 | ~~sweep `gate_m` and `fade_time_s`~~ **done 2026-09-18, and nothing binds** (`memory_diagnosis.py --slot-rules`, `checkpoints_three_horizon/slot_rules.json`) | see the table below |
| S2 | ~~re-detections after gaps over 10 s~~ **done 2026-09-18, and the case is thin** | see below |
| S3 | higher creation threshold or confirmation; more than 64 candidates | **deprioritised 2026-09-22**: the memory uses a median of 24 slots of 256 and already holds 95.5% of the people it fails to report, so more candidates would fill capacity that is not scarce. Only worth revisiting if a later result shows people the memory never learns about at all |
| S4 | ablate the hand-coded association -- **not worth a training run** on S1's evidence: association already puts a live slot on 99.2% of annotated people, and a *matched* one on 97.0%, so there is nothing for a learned association to recover | A50 phase 1b item 4 |

**S1 and S2 result, 2026-09-18** (rule-only replay over the whole FROG `official` test split, 153,655 annotated people, no trained model; the no-temporal seed-0 candidates):

| gate / fade | people with a live slot within 0.5 m | of those, matched this frame | matches/frame | spawns/frame | losses/frame | slots p50/p99 | returns/frame | return gap p50 / p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 m / 2 s | 99.1% | 96.9% | 9.33 | 0.25 | 0.23 | 20/47 | 0.14 | 0.79 / 5.65 s |
| **0.6 m / 10 s (default)** | **99.2%** | **97.0%** | 9.44 | 0.13 | 0.10 | 40/89 | 0.02 | 2.99 / 14.08 s |
| 1.0 m / 30 s | 99.2% | 97.0% | 9.47 | 0.11 | 0.05 | 67/172 | 0.00 | 2.86 / 14.94 s |

**Neither rule binds.** Coverage moves 0.2 pp across a 3x range of gate and a 15x range of fade, and the matched share never leaves 96.9-97.0%. This extends A50 phase 1b item 1 (a live slot sat within the gate for 100% of the people stage 3 lost) from the lost people to all of them: **the bookkeeping already places a matched slot on 97% of annotated people, so every remaining loss is downstream, in what the learned head does with that slot.** Priority therefore moves to phase 2 (O1 calibration, O2 coasting) and phase 1.
Coverage counts any live slot within 0.5 m, so it is an upper bound, as in `docs/PROPOSAL.md` §5.5; with 40 slots alive at the default some coincidence is included.
**Returns are rare and short** (S2): at the default, a slot is lost 0.10 times per frame and something reappears where it stood 0.02 times per frame, with a median gap of 3.0 s and 17.8% of returns beyond the 10 s fade. Against ~3 annotated people per frame, carrying state across retirement addresses a very thin slice, which agrees with the oracle finding that little is recoverable past 10 s. **S2 stays parked unless a later result points back at it.**
**One capacity note for later:** at 0.3 m / 30 s the memory reaches 242 live slots of 256 at p99, so a longer fade with a tight gate would start to hit the capacity that 64/128/256 never did.

**Phase 3b -- the design decisions the write-up has to defend (proposed 2026-09-20, awaiting the owner).** The owner's reason for
prioritising slots is that the mechanism is the novel part, so every choice in it needs a stated reason and a measurement. Phases 1
and 2 are implementation quality; these are the choices themselves. Each is one training run behind a flag, and each is expected to
*lose* -- that is what makes it evidence for the decision.

| # | the decision | the ablation | why it is the one to run |
| --- | --- | --- | --- |
| D1 | slots live in room coordinates, so a chair keeps zero velocity and a person does not (`docs/PROPOSAL.md` §5.5) | ~~run~~ **settled 2026-09-26 at three seeds**: 82.02 / 82.57 / 82.01 (mean 82.20, sd 0.32) against B0's 83.34, **-1.14 pp, Welch t = 3.40, p = 0.034**. Seed 1 took five attempts and 835 min; seed 2 took 321 min. Recall at matched false positives moves only -0.17 pp (p = 0.41), so the effect is in ranking, and the ablation still bundles the association loss the sensor frame causes. `step3a --no-room-frame` | the central claim of the design, and the reason memory sits inside the model rather than behind a tracker; never ablated. **A test found the first consequence already**: in the sensor frame the robot's own motion moves every object, so at 0.2 m of travel a static object gains velocity, and at 1 m per frame it leaves the 0.6 m gate entirely and the slot is lost. The ablation therefore costs association as well as the velocity feature |
| D2 | a slot is held by its accumulated value, not the latest score, so a hidden person keeps their slot | **run 2026-09-21: 83.94% against B0's 83.34% mean, +0.60 pp**, recall at matched false positives 92.7% (B0's best seed), coasting precision 31.7%, 700 min. **The ablation does not lose: accumulated value earns nothing measurable on FROG.** `step3a --latest-value` | argued in §5.5 and never tested; the replay only showed capacity is ample (A50). Both rules create a slot at the candidate's score; from there accumulation climbs past it towards certainty while the ablation copies each frame's score |
| D3 | each slot decays at a learned, input-dependent rate over 1-60 s | **run 2026-09-21 at 8 s: 82.44% against B0's 83.34% mean, -0.90 pp**, recall at matched false positives 92.1%, below every B0 seed; 288,907 parameters against 306,443, exactly the decay layer and its bias; 352 min. Direction supports the learned decay, inside the threshold. `step3a --fixed-decay-s T` | prices the selective recurrence, the part closest to published recurrent memories. Pick `T` as the geometric middle of the initial range, ~8 s, and state it |

**The bookkeeping itself is not trainable (owner's question 2026-09-20), and that choice has never been measured.** `manage_slots`
runs under `@no_grad`: association (mutual nearest within `gate_m`), value accumulation, spawn, replacement, retirement, velocity and
hit rate are all fixed rules. What learns is everything that *reads* them -- the candidate encoder, the matching, exchange and recall
attention, the selective decay and state update, rescoring and coasting. The rules' outputs enter the learned path as inputs (value,
age, time since match, displacement, speed, hit rate, matched flag, observed score, and slot positions as an attention bias), but as
constants, so the loss cannot shape which slots survive.

| # | what could become trainable | how, without breaking the discrete parts | expected |
| --- | --- | --- | --- |
| D4 | the **value rule's constants**: `gain` and `fade_time_s` | **implemented 2026-09-20** as `step3a --learn-value`: `manage_slots` keeps its discrete decisions under `no_grad` and takes the constants as detached floats, while the value the network reads is recomputed with gradient, so the rule moves as the parameters train (+2 parameters) | one run; tests whether the two hand-set constants are the right ones |
| D5 | the **association width** | the 0.6 m gate is a hard cut inside a discrete assignment, so it has no gradient; give the *value* update a distance weight `exp(-d^2 / 2 sigma^2)` with learned `sigma` while the assignment stays discrete | one run; only worth it if D4 moves |
| -- | the spawn and eviction decisions themselves | `argmin`/`argsort` over slots, so a relaxation (Gumbel, straight-through) or a soft assignment would be needed -- a research-scale change, and `docs/PROPOSAL.md` §12 already lists the MOTR-style alternative | not now |

**Phase 3b read so far, 2026-09-21** (one seed each, B0 mean 83.34%, threshold ~1.5 pp):

| ablation | test AP | vs B0 | recall at matched FP | reading |
| --- | --- | --- | --- | --- |
| D1, sensor coordinates | 82.02% | **-1.32** | 92.2% | room coordinates earn their place, but inside the threshold on one seed |
| D2, latest score | 83.94% | **+0.60** | 92.7% | **accumulated value earns nothing measurable**; the ablation is, if anything, ahead |
| D3, one fixed 8 s decay | 82.44% | **-0.90** | 92.1% (below every B0 seed) | direction supports the learned decay; 17,536 parameters cheaper |

**The honest summary of phase 3b: the memory as a whole is worth +5.68 pp over its own candidates, but no single design choice
inside it clears the threshold on one seed.** D1 is closest and is the claim the write-up most needs, so it gets the confirmation
seeds first. D2's null and D3's small loss are reported as they are.
**Seed 1 has now failed four times** -- twice on 2026-09-21 (host memory, then device-side corruption), once on 2026-09-22 when an agent ran `pytest` beside it, and once at 23:52 that night when the **AMD driver bugchecked the machine** (`amdkmdag.sys`, `memory/gotchas.md`) 3.5 h into a healthy run. **Queue relaunched 2026-09-24 23:05** with a launcher that retries each step up to three times with `--resume`, stopping instead on a training guard; the owner found no supported way to change the driver, so the machine stayed on 26.8.1 and resilience replaced prevention -- until 2026-09-26, when the queue was stopped to revert to 26.2.2 (A55). `--resume` was built on 2026-09-23 and **verified end to end on 2026-09-24** by killing a run at epoch 3 and resuming it (`CHANGELOG.md`). The owner's tests of the machine ran uninterrupted, but the driver is still the known-bad 26.8.1. Detail on the first two: **2026-09-21** (`memory/gotchas.md`): first a host allocation failure 3.5 h in (stage 3 held the train, val and test caches at once on a 16 GB machine; test now loads only after training), then, on the relaunch, device-side corruption at epoch 12 -- the candidates' own val AP fell from 72.85% to 3.40% with the loss at exactly zero, and it died with a HIP OOM that reported 14.87 GiB free. A GPU sanity check afterwards passed, so the fault is transient. `training_fault` now stops a run at the first such evaluation. **D1 therefore still rests on seed 0 alone**, and the machine should be rebooted before the seeds are attempted again.

- **D2 is the more interesting result, and it is a null.** It agrees with S1 (the bookkeeping rules do not bind) and suggests the learned slot state already carries what the value rule was designed to protect -- a person hidden for a moment. Before the write-up claims the value rule, either confirm the null at more seeds or state the rule as a design choice that measurement does not support on this dataset.
- **It also weakens D4's prospects**: learning the constants of a rule that does not matter is unlikely to matter either.
- **D1's number bundles two effects** (the test found the sensor frame also breaks association when the robot moves further than the gate in one frame), so it is a lower bound on the value of room coordinates, not a clean isolation of the velocity feature.
- **The mid-run slowdown was transient**: D2's 10 s-chunk epochs reached 1,383 s at one point against ~640 s for B0, then returned to 629 s, and its inference cost is unchanged at 4.44 ms per frame. Nothing about the rule is slower.

**An interaction worth knowing before reading D4 (found by its test, 2026-09-20).** With T1's zero-initialised rescoring head, the
head's last layer starts at zero, so at initialisation *no* gradient reaches anything it reads -- the slot statistics, and therefore
the value constants, included. D4's parameters only start moving once the head itself has. So D4 is run on whichever rescoring
parameterisation wins T1, and a null D4 result on an untrained head would mean nothing.

S1 predicts little headroom on FROG -- the rules already put a matched slot on 97.0% of annotated people -- so the likely outcome is
that learning them changes nothing, which is itself the evidence the write-up needs for keeping them fixed.

Already-strong companion evidence, for the same section: SORT on the same candidates loses at every setting (A43 step 4), 64/128/256
slots tie within 0.1 pp (A50), the bookkeeping rules do not bind (A51 S1), and the time step does (A51 0a).

**The baseline may sit low, and this repo has been burned by that before (2026-09-26).** Every arm run since B0 lands above it on
every metric at once:

| arm | AP | recall at matched FP | coasting precision |
| --- | --- | --- | --- |
| B0, 3 seeds | 83.34 | 92.47 | 25.43 |
| T1, no gate | +1.05 | **+0.88 (passes)** | +6.72 |
| T2, plateau | +0.56 | +0.71 | +6.59 |
| O2, coast x3 | +0.70 | +0.39 | +6.57 |

Three unrelated changes cannot all be mildly good in the same way. Either the B0 seeds sit low by chance -- three single runs above a
three-seed mean is about a one-in-eight coincidence -- or something since 2026-09-19 shifted the numbers. Nothing in the changes
since then should touch training numerics (the flags added default to the old behaviour, and `--coast-weight 1.0` multiplies by
one), but **that is exactly the reasoning that failed for stage 2's limit-beam arm**, where fresh controls turned a 0.93 pp loss
into a tie (`CHANGELOG.md` 2026-09-17). **Before any of these arms is claimed, re-run B0 at three seeds on today's code**; ~15-20 h,
and it also refreshes the spread estimates every bar depends on. T1's recall pass (+0.88 against a 0.86 bar) is far too thin to
survive a baseline shift.

**Phase 5 -- what a write-up about the slot memory still needs (proposed 2026-09-25, awaiting the owner).** Phases 1-3b answer
"is each internal choice justified". These answer "is the mechanism itself established", which is what a reader will ask. The first
four need no training at all.

| # | experiment | why it is needed | cost |
| --- | --- | --- | --- |
| ~~E1~~ | ~~**memory-horizon curve**~~ **done 2026-09-27: the slots use about 10 seconds.** AP 79.41 / 80.46 / 81.51 / 82.65 / 83.53 / 83.73 / 83.88 at 0.5 / 1 / 2 / 5 / 10 / 30 s / never, against candidates at 77.66 -- ~95% of the gain by 10 s, still climbing steeply through 2-10 s, and `fade_time_s = 10` sits exactly where it flattens | The project's thesis is that a long horizon carries the signal, and **we have never measured how long a memory the model actually uses.** The coarse convolution's horizon was measured; the slots' was not | evaluation only, one pass per N |
| ~~E2~~ | ~~**odometry robustness**~~ **done 2026-09-27, and it qualifies D1.** Room frame 83.88 / 82.96 / 81.25 / 78.27 / 73.10 at 0 / 0.005 / 0.02 / 0.05 / 0.1 m/s of drift; the sensor frame is flat at 82.57 because it never reads poses. **The room frame wins only below ~0.01-0.02 m/s and loses above it**, and wheel odometry at FROG's ~0.5 m/s typically drifts 0.005-0.025 m/s, so the crossover is inside the realistic range. Training with jittered poses is the untried repair | Room coordinates are the central claim, and they buy their advantage *from odometry*. A robot's odometry drifts; a reader will ask what happens then, and D1 currently answers only the perfect-odometry case | evaluation only |
| ~~E3~~ | ~~**crowding**~~ **done 2026-09-27: the gain is flat, not crowd-driven.** +6.22 / +5.87 / +6.13 / +6.62 pp at 0-1 / 2-3 / 4-5 / 6+ people per frame. Largest in the most crowded frames, nearly as large when nobody is there (where candidates are weakest at 65.21) -- so memory helps everywhere, a weaker claim than the design's | The design argues memory helps where detection fails, which is crowds and occlusion. If the gain is flat in crowd size, the story is wrong even though the average is good | evaluation only |
| ~~E4~~ | ~~**cost scaling**~~ **done 2026-09-27: flat.** 4.37 / 4.37 / 4.32 / 4.37 / 4.46 ms per frame at 64 / 128 / 256 / 512 / 720 slots -- 11x the slots for 2% of the time, so the O(K^2) exchange is invisible at these sizes on this GPU. State the GPU with the claim; a weaker embedded part may differ | A robot paper needs the cost curve, not one number; 4.4-4.6 ms is measured only at 256. **It also prices the design's only super-linear term** (owner's framing 2026-09-25: linear-time memory for embedded use): the slot-to-slot exchange is O(K^2) per frame, ~65k pairs at 256 slots, while everything else is constant per frame. With E6 it decides whether the quadratic term can be dropped | evaluation only |
| ~~E5~~ | ~~a per-track GRU baseline~~ **dropped 2026-09-25 (owner's call)**, and it costs almost nothing: **E6 and E7 run together are that arm**. Remove the slot exchange and the learned matching and what is left is per-slot recurrence plus rescoring, which is the "plain per-track recurrent memory" baseline, with no new code. **Two corrections to the reasoning, for the write-up:** Li2Former's failures were DirectML-era (`gotchas.md` I4) and stage 3 *is* attention that trains fine on ROCm at 4.4 ms per frame, so "the setup cannot train attention" no longer holds; and the memory **does** contain recurrence -- `state = keep * state + (1 - keep) * update` with a learned per-slot decay is a diagonal RNN of the S5/minGRU family. The defensible claim is **object-keyed memory carried as state at constant per-frame cost, instead of re-encoding a T-scan window**, which is what makes Li2Former and DR-SPAAM expensive | none |
| ~~E6~~ | ~~**ablate the slot-to-slot exchange**~~ **done 2026-09-27: it earns nothing.** AP 83.58 against B0's 83.34 and recall 92.63 against 92.47, both inside the bars, while coasting precision rises to 37.35% (+11.9, bar 8.19) and the arm is **66,560 parameters smaller, 9% faster (3.93 ms) and uses 40% less GPU memory (3.2 GiB)**. One seed, and scored from the epoch-21 checkpoint the run was interrupted at, so it is slightly under-trained -- which strengthens the null. **Dropping it removes the design's only super-linear term.** Its own guard failure is in `gotchas.md` | One of the three learned mechanisms and never ablated. `docs/PROPOSAL.md` justifies it by crowds, which E3 also probes. **It is also the design's only O(K^2) term**, so a null here buys a strictly linear-in-slot-count memory -- the cleanest form of the efficiency claim | one training run |
| E7 | **ablate the learned matching** (cross-attention from slots to candidates), leaving the hand-coded association to carry the value | Separates "learned soft association" from "bookkeeping", which S1 showed is already sufficient for coverage | one training run |
| E8 | **DROW** (A49) -- **postponed 2026-09-25 (owner's call)**: the architecture is designed on FROG and then carried to DROW and JRDB by changing **parameters only, never the architecture**, so the cross-dataset work waits until the FROG design is settled. A49 already requires the settings in physical units, which is what makes that possible | a training chain, later |
| ~~E9~~ | ~~spread for the calibration metrics~~ **done 2026-09-25**: sd 0.48 (AP), 0.43 (AP at 0.3 m), **0.29 (recall at matched FP)**, **2.73 (coasting precision)**; three-sigma bars 1.45 / 1.29 / 0.86 / 8.19 pp (`slot-design-evidence.md`). **It re-judges phase 1:** T1's recall gain (+0.93) is the only arm effect that clears its bar, and only just; T2's (+0.73) does not, and neither arm's coasting gain (+6.8, +6.6) comes near the 8.19 pp bar. So the two arms are **not** equivalent after all -- T1 edges past on recall, T2 clears nothing | done |

Recommended order: E9 (minutes), then E1-E4 (a day, no training), then E5 and E6, then E8. E7 only if E6 or S1 suggests the learned
half matters. A qualitative figure -- a person occluded and recovered by a coasting slot -- is worth adding once E1 fixes the
horizon claim.

**What phase 5 has established, and what it costs the story (2026-09-28).** The mechanism is in better shape than its internals:
the memory is worth +5.68 pp over its candidates at three seeds, uses about **10 seconds** of history, helps at every crowd size,
and costs 4.3-4.5 ms per frame whatever the slot count. Two of its parts are now known to be unnecessary -- the slot exchange (E6)
and, by measurement, the accumulated-value rule (D2) -- and one is conditional: room coordinates (D1) win only while odometry drift
stays under ~0.02 m/s. The remaining arms (E7, the both-off recurrent baseline, O1) are running.

**E10, the experiment the simplification makes necessary (owner's challenge 2026-09-28).** E6 and E7 show the attention adds
nothing, which does **not** weaken the project's actual claim -- an object-keyed memory *inside* the detector, acting before the
threshold, at constant per-frame cost, against a field that uses a detector plus a post-hoc tracker or re-encodes a window. That
claim rests on evidence we have: +5.68 pp over its own candidates at three seeds, SORT losing at every setting on the *same*
candidates, place-keyed memory failing three ways, and ~10 s of usable horizon (E1).

What the simplification does expose is how close the remainder sits to "a tracker with a learned scorer": mutual-nearest
association, an exponential value decay, and a rescoring head. Three things separate it, and only one is measured.

| distinguishing feature | evidence |
| --- | --- |
| rescoring acts before the threshold | strong: SORT on identical candidates loses at every `min_hits` |
| the slot state is learned, not hand-coded | weak: only D3, where a fixed decay costs 0.90 pp, inside its bar |
| memory feeds back into the detector | none: step 3b's prior never left zero, so feedback was never exercised |

**E10 -- done 2026-09-29, and it is the favourable answer.** AP **81.84** (-1.50, bar 1.45), AP@0.3 80.26 (-1.42, bar 1.29), recall at matched FP **90.80** (-1.67, bar 0.86), coasting precision **11.35** (-14.08, bar 8.19): **the only arm of eleven that clears its bar on every metric, and all four downward**, at 55,691 parameters and 2.70 ms/frame. The memory is **not** a tracker with a learned scorer; the row "the slot state is learned, not hand-coded" moves from weak to strongest. Coasting precision collapsing by 14 pp locates the mechanism: with no current measurement only carried state separates an occluded person from a stale slot. **E13 therefore proposes E6+E7, not E10.** Original framing: rescore from the bookkeeping statistics alone -- value, age, time since match, displacement, hit rate -- with no `state`
and no recurrence. A match says the memory is a tracker with a scorer, and the write-up should say so; a drop says the learned
state is the mechanism. One training run, after the B0 control.

**Phase 6 -- agreed 2026-09-28, in priority order.**

| # | what | why it is worth a run |
| --- | --- | --- |
| **E12** | **exercise the feedback path**: step 3b with the memory prior forced to matter -- simulated priors as `docs/PROPOSAL.md` §8 intends, or the prior's weights initialised away from zero and its contribution logged per epoch | **The owner's view (2026-09-28): feeding memory back into the network is the proposal's biggest novelty, and it has never been tested.** Step 3b's prior never left zero, so "memory changes what the detector sees, before any threshold" is currently an unsupported claim. Everything else in phase 5 measured the memory as a *post*-detector rescorer |
| ~~**E13**~~ **done 2026-09-30: the simplification is free.** E6+E7 at three seeds ties B0 on every metric -- AP 83.86 (+0.53, bar 1.45), AP@0.3 82.14 (+0.46, bar 1.29), recall at matched FP 93.16 (+0.68, bar 0.86), coasting precision 29.81 (+4.38, bar 8.19) -- at **173,579 parameters against 306,443 (-43%)** and **3.72 ms against 4.53 (-18%)**. Nothing clears in either direction, and every point estimate is slightly above the control. **Correction: a two-seed reading had recall at matched FP clearing its bar at +1.08; seed 2 pulled it to +0.68 and inside.** With E10 the architecture is decided: state is the mechanism, attention is not; propose E6+E7 with the full model as the ablation. Original framing: the minimal architecture as the proposal, at three seeds: no exchange, no learned matching, and whatever E10 says about the state, judged against the fresh B0 control | The owner's framing: propose the most efficient solution **and** the simplest of the ones that work. E6+E7 already matches B0 at 43% fewer parameters; if it survives three seeds it is the architecture to publish, with the full model as the ablation rather than the reverse |
| **E11** | **train with jittered odometry** | E2 showed the room frame loses to the sensor frame above ~0.02 m/s of drift, inside the range real wheel odometry reaches. Domain randomisation over pose error is the standard repair and is untried; without it the design's central claim carries a deployment caveat |

**The fresh B0 control is finished as a question, 2026-09-29: it reproduces the old control exactly, and changes no verdict.**
`a51_b0f_s1` retrained stage 3 from scratch at the same seed, flags and code as `a51_b0_short_s1`, in a different directory eight
days later, and matched it on **46 of 46** evaluations to six decimals -- AP 83.17, AP@0.3 81.52, recall at matched FP 92.53,
coasting precision 22.31, every figure identical. `a51_b0f_s0` does **not** match, and the reason is exact: it was resumed after a
driver bugcheck, tracks `a51_b0_short_s0` for 36 of 42 evaluations, and diverges at the first epoch after the resume point.

- **Stage-3 training is bit-reproducible from its seed here**, which `memory/interpreting-evaluation.md` denied on evidence that
  had confounded a code change with the seed. Corrected there.
- **`--resume` produces a different draw**, not the same run continued -- relevant to `a51_o1_bounded_s0c` (three attempts) and
  `a51_e6_noexchange_s0`, which are valid single samples but not reproductions.
- **Every arm verdict recorded against the E9 bars stands**, so the results table does not need rewriting. `utils/judge_arms.py`
  (new, stdlib only, safe beside a running job) regenerates the whole comparison in one command.
- **The prediction was confirmed and then the run was stopped (owner's call, 2026-09-29 09:19).** `a51_b0f_s2` matched its
  predicted trajectory on **39 of 39** evaluations before it was killed at 4.9 h; its remaining 22 evaluations would have cost
  about 5 more GPU hours to reproduce numbers already on disk from `a51_b0_short_s2` (83.88 / 82.17 / 92.72 / 26.61). 39/39 across
  two chunk stages is the evidence; the last 22 would have added nothing. **Seed 2 of the fresh control therefore does not exist,
  and the control of record stays `a51_b0_short_s*`.**
- **Running instead, from 09:19:** `a51_e10_norecurrent_s0`, then `a51_e67_recurrent_s1` and `_s2`. The latter two were added
  because the E6+E7 arm is the architecture E13 would propose and it rests on a single seed; they are worth their time whatever
  E10 returns. **E10 is 55,691 parameters against B0's 306,443 -- an 82% reduction**, so a match here is a much stronger
  simplification result than E6+E7's 43%.
- **Open question for the owner:** three seeds per arm still buys real run-to-run variation, since different seeds give different
  initialisations -- determinism only means each is repeatable. So E13's three-seed plan stands unchanged. But an arm that must be
  *verified* rather than *sampled* can now be re-run identically, which is cheaper than adding a seed.

**How E12 must be built, worked out 2026-09-29 from step 3b's own log** (owner's approval wanted before it is coded).
The claim "step 3b's prior never left zero" is now backed by a number rather than an assertion: step 3b's log reports
**stage 2 with feedback at 75.68%** on test, and A50 phase 1 item 2 separately scored **the same network with the prior
algebraically silenced at 75.68%** -- equal to two decimals, so the feedback channels carry no signal at all.

The mechanism matters for how E12 is designed, and it is not a bug. Gradient does reach `aggregate.weight[:, 2*c2:]` even
from a zero initialisation, because `dL/dW = dL/d(bottom) * prior` and the prior is not zero. What is missing is *pressure*:
**the feedback path and the rescoring head compete for the same signal, and the rescoring head wins.** Both read the same
slots; the head applies them directly to a candidate's score one operation before the loss, while the prior has to travel
through the bottleneck, the decoder, the vote grid and the candidate decoder to have any effect. With 3 epochs at lr 1e-4
the optimiser takes the short path, and the long one stays silent. So an E12 that only "initialises the weights away from
zero" will measure the same null, slower.

| option | what it does | what it would prove |
| --- | --- | --- |
| **E12a (recommended)** | step 3b with the **rescoring head frozen at identity** -- the memory's only route to the loss is the prior. Log the prior's contribution per epoch as `\|W_prior . prior\|` against `\|bottom\|` | whether memory *can* change what the detector sees. The one configuration where a null is informative: if the prior stays silent with no competitor, feedback does not work here |
| E12b | oracle priors: draw the prior from ground truth instead of from slots, rescoring head left alone | an upper bound on what any feedback could buy. Cheap and diagnostic, but proves nothing about the real system |
| E12c | initialise `W_prior` away from zero and log its contribution | the option `TODO` recorded before this analysis; expected to reproduce the null for the reason above |

**Approved by the owner 2026-09-29: E12b first, then E12a**, and E12c is dropped. E12b is one run and tells us how to read
E12a -- a null with a perfect prior means feedback cannot help here and we report that; a large gain with a perfect prior means
the mechanism works and only training fails to find it, which is a different claim. E12a is **two** runs, not one: the prior-off
control is required because joint training drifts stage 2's weights by itself, which already cost 1.24 pp (76.92 -> 75.68) in the
original step 3b, and without the control that drift is indistinguishable from the feedback's effect. **E12a's whole-detector AP
will be below B0 by construction**, since it gives up the rescoring worth +5.68 pp; the only number it reports is stage 2's AP
with the prior minus stage 2's AP without it. **Built 2026-09-29, and the design got cheaper on the way.** Both arms were going to need a matched control run -- a second
training to compare against -- which would have made E12 four runs, not three. A **within-network ablation** answers the question
better and in half the time: train once with the prior, then score *those same weights* twice on test, once with the prior as
trained and once with it fed zeros. The difference is what the feedback actually contributes, and unlike a retrained
`prior_channels=0` control it cannot be muddied by the control compensating elsewhere. Every step-3b run now prints that
comparison as a `feedback contribution` line and stores `test_prior_silenced` beside `test` in its results.

What landed: `ObjectMemory(rescore=False)` drops the correction entirely (E12a); `oracle_prior()` draws a perfect prior from the
annotations (E12b); `joint_step(..., prior=...)` lets a caller override what the slots would have drawn; `evaluate_joint(...,
prior_mode=)` picks `slots` / `oracle` / `zero` and refuses anything else; `--oracle-prior`, `--no-rescore` and
`--prior-channels` on step3b. Tests written first, 13 of them, in `TestRescoringOff`, `TestOraclePrior`,
`TestJointStepPriorOverride`, `TestPriorModes` and `TestCalibrationNetworkWithoutPrior`.

**Queued as `run_e12.ps1`, armed behind the E10 chain: `a51_e12b_oracle` then `a51_e12a_norescore`, ~4 h each.** The chain runs
`pytest` itself once the machine is free and refuses to start either run if the suite does not pass, because the tests could not
be run while E10 held the GPU. **Coverage gap to close:** `evaluate_joint` has no test at all -- it needs a `FrameSplit` fixture
nothing else builds -- so its `prior_mode` dispatch is covered only by the guard test, not by an end-to-end pass.

**E12 answered 2026-09-30, and the answer is about the implementation, not the idea.** Three cells, all null:

| arm | prior | rescoring | stage 2 AP | prior silenced | contribution |
| --- | --- | --- | --- | --- | --- |
| `a51_e12b_oracle` | ground truth | on | 76.4517% | 76.4506% | +0.0012 pp |
| `a51_e12a_norescore` | the slots' own | off | 77.5909% | 77.5931% | -0.0022 pp |
| `a51_e12ab_oracle_norescore` | ground truth | **off** | 77.36% | 77.35% | +0.02 pp |

**Two hypotheses died on the way, and the third is measured.** The *competition* hypothesis -- the rescoring head out-competes
the prior -- predicted the path would wake once the head was removed; E12a and the ceiling cell both say it does not. The
*cancellation* hypothesis -- large weights parked in an unidentifiable direction -- was tested by `verify_oracle_prior.py` and is
false: under an oracle prior the two channels are identical, so their gradients are identical and the difference stays exactly
zero (measured `0.000000e+00`), leaving the whole weight in the **acting** direction, sum rms 3.56e-2 against the body's 4.07e-2.

**What is actually true, measured by `utils/measure_prior_share.py` on real frames:** the prior carries
**0.012% to 0.045% of `aggregate`'s input magnitude**. `aggregate` is a 1x1 convolution over `cat([x, x.amax(), prior])` --
**256 body channels against 2 prior channels** -- and `render_prior` marks one sector per person, so the prior is non-zero in
**2% of sectors**. A per-weight rms of 0.44 becomes a per-contribution share of 0.045%. **The feedback path was never given
enough capacity to matter, whatever it was told.**

**So E12 refutes this injection, not the proposal's idea.** The write-up must say that precisely: "memory fed back into the
detector, as two sparse channels entering a 258-channel 1x1 convolution at the most downsampled layer, contributes nothing
measurable, and the measurement shows why." A fair test of the idea would need a wider path -- project the prior to `c2`
channels rather than 2 -- a denser prior -- `render_prior` gives a person a single sector with no spatial extent -- injection
above the bottleneck, or a multiplicative gate instead of additive concatenation. **Not queued; it is a design change, not an
ablation, and it is the owner's call whether the claim is worth another architecture.**

**Superseded framing, kept for the record:**

| arm | prior | rescoring | stage 2 AP | same weights, prior silenced | contribution |
| --- | --- | --- | --- | --- | --- |
| `a51_e12b_oracle` | ground truth | on | 76.4517% | 76.4506% | **+0.0012 pp** |
| `a51_e12a_norescore` | the slots' own | **off** | 77.5909% | 77.5931% | **-0.0022 pp** |

**E12a is the load-bearing one.** Removing the rescoring head leaves the prior as the memory's only route to the loss -- the
competition hypothesis' own prediction was that the path would then wake up -- and it contributed **-0.0022 pp**. So the
original step 3b's silence was not the rescoring head out-competing the prior. The path does not work here.

**A side finding worth keeping.** Stage 2 ends at **77.59%** in E12a against **76.45%** in E12b, and candidates alone score
77.66%. So joint training's ~1.2 pp drift in stage 2 (the A50 phase 1 finding) tracks the **rescoring head**, not the prior:
with rescoring off, stage 2 finishes where it started.

**E12b's original write-up, and the check it was waiting on.** `a51_e12b_oracle` trained step 3b for 3
epochs with stage 2 fed a **ground-truth** prior. Scoring those same weights with the prior against themselves with it silenced:

| | stage 2 AP | whole detector AP |
| --- | --- | --- |
| oracle prior | 76.4517% | 81.692% |
| the same weights, prior silenced | 76.4506% | 81.637% |
| **contribution** | **+0.0012 pp** | +0.055 pp |

Stage 2 learned to take **essentially nothing** from a prior that told it exactly where every annotated person was. The two
evaluations are *not* bit-identical, so gradient did reach `aggregate.weight[:, 2*c2:]` and the weights did move -- they moved to
something worth a thousandth of a percentage point.

**The check passed, 2026-09-30 08:55, so E12b's null stands.** `utils/verify_oracle_prior.py` on 400 recorded FROG test
frames: **400 of 400** draw a non-zero prior, 1.59 sectors each, and **638 of 638** people's sectors -- recomputed from
`people_polar`'s bearing, a path that never touches `render_prior`'s xy arithmetic -- land on a sector the prior marked, with
none outside the field of view. The oracle is real.

**And the trained weights are large, which makes the null stranger than expected.** `aggregate.weight[:, 2*c2:]` after training:
rms 1.78e-2 against the body's 4.07e-2, **44% of it**, all 256 entries non-zero. So the network put substantial weight on the
feedback channels and still changes its answer by a thousandth of a point when they are silenced. The likely explanation is a
degeneracy: under an oracle prior the two channels are **identical** (both 1.0 wherever a person is), so only their **sum** can
act and their difference is a free direction the optimizer can park weight in. `verify_oracle_prior.py` now reports the sum and
difference separately; run it on all three checkpoints once the machine is free. Note this degeneracy cannot explain **E12a**,
whose prior is the slots' own, where value and person differ.

**The original check, for the record.** A silently empty oracle -- `people_xy` in the wrong frame, or an axis
convention mismatch against `render_prior` -- would produce a null that looks exactly like this one. The unit tests cover
`oracle_prior` on synthetic points, and `AGENTS.md` rule 4 is explicit that synthetic tests are not enough for anything that
reads recorded data. **Before this is written up, load `a51_e12b_oracle`'s checkpoint and confirm on real FROG frames that the
rendered oracle prior is non-zero in the sectors that hold people, and that `aggregate.weight[:, 2*c2:]` is non-zero.** It cannot
run while a training job holds the machine (`memory/gotchas.md`, the commit limit), so it is queued for the gap after E12a.

**A gap in the design, worth naming now.** E12b left the rescoring head on, so its null is consistent with two different stories:
feedback is architecturally useless here, or the rescoring head simply out-competed it again. E12a removes the competitor but
feeds a *real* prior. The cell that would separate them -- **oracle prior and no rescoring** -- is neither run. **Queued unconditionally 2026-09-30 (owner's call), as `run_e12ab.ps1` behind E12a**, rather than waiting on E12a's
outcome: `a51_e12ab_oracle_norescore`, a perfect prior with no competitor for it. Two nulls with different possible causes are
weaker evidence than one clean ceiling.

The chain runs **`utils/verify_oracle_prior.py` first and refuses to train if it fails**, because a null from an empty oracle is
indistinguishable from a null from a useless feedback path. The script checks three things on real FROG frames: how many
annotated frames draw a non-zero prior, whether each person's sector recomputed from `people_polar`'s **bearing** -- a path that
never touches `render_prior`'s xy arithmetic, so the two disagree exactly when the axis convention is wrong -- is a sector the
prior marked, and how large `aggregate.weight[:, 2*c2:]` grew during training. It also retro-verifies E12b, whose null is
provisional until it passes.

**Literature check, done 2026-09-28** (`docs/RESEARCH.md` §3.5), extended the same day after the owner asked whether JRDB's
separate tracking task has its own literature -- it does, and the closest system is now pinned down: **DR-SPAAM + Norfair,
tracking-by-detection with no feedback, 20 Hz on a Jetson Xavier NX, MOTA 85.45%** (arXiv 2412.15000). Cite it as the baseline
this project must beat *in kind*, and note what it cannot do: a SORT-style tracker never sees a detection that fell below the
threshold. *(Corrected 2026-09-28 by the tracking survey below: this read "which is 99.1% of the people our memory recovers", but
99.1% is the share of the people stage 3 **loses** at 0.3 that rescoring pushed under the threshold; and ByteTrack-style trackers do
see sub-threshold detections.)* Three findings change how the contribution must be stated:
memory inside a 2D-LiDAR detector is **not** new (DR-SPAAM's auto-regressive template is exactly that, at ~0.15 s); object-level
memory fused into detection exists in other modalities (**MAD**, CVPR 2025, memory bank of forecasts, +2.5-7.6 AP on Waymo/AV2;
**Tokmakov et al.**, ICCV 2021, ConvGRU memory in CenterTrack); and a learned 2D-LiDAR detector **with** a multi-object tracker
already runs at 20 Hz on a Jetson Xavier NX at MOTA 85.45% (arXiv 2412.15000), while JRDB ships tracking benchmarks -- image-plane
and point-cloud ones, none on its 2D laser (corrected 2026-09-28). The
defensible claim is object-keyed memory inside the detector, before the threshold, at constant per-frame cost, over a measured
~10 s horizon -- and the finding that its attention is unnecessary.

**Transfer to the embedded tracker paper's recordings (owner's question 2026-09-28).** Their setup is an unusually good match
for a zero-shot transfer test: **Hokuyo UTM-30LX-EW, 0.25 deg resolution -- identical to FROG's -- 270 deg and 1080 points at
20 Hz, mounted at 45 cm**, with **Vicon mocap at 100 Hz for both people and robot** and odometry at 50 Hz, which is better ground
truth than FROG has. Three recordings, about 7 minutes total, three participants.

- **The model already takes it.** Nothing in `takheliped.py` hard-codes 720 beams; the network is convolutional over beams and
  receives `angles`, and 1080 divides by 8 for the U-Net's three halvings. Only the trainer passes `frog_laser_angles(720)`.
- **What must change first:** the temporal lags are counted in **frames** (0, 8, 16, 24, 32), so at 20 Hz the coarse horizon
  silently doubles from 0.8 s to 1.6 s. A49's "settings in physical units" item becomes a precondition, not a nicety. Then a
  loader and a per-deployment `--max-range-m`.
- **The data is not public**: the paper names no repository or release, so this needs an email to the authors. **Drafted
  2026-09-28 at `email-data-request-plozza-et-al.md`** (Plozza, Marty, Scherrer, Schwartz, Zihlmann, Magno; SAS 2024) -- it
  asks for the three recordings with mocap and odometry, offers evaluation-only use with citation, and falls back to two
  questions that are useful even if they decline. Name, affiliation and contact fields are left blank for the owner.
- **What it can and cannot prove:** 7 minutes, three participants, one indoor space is a *robustness demonstration* -- "trained on
  FROG, runs on another robot and sensor untouched" -- not a benchmark row.
- **The public route meanwhile is JRDB** (A49's third dataset), which is 2D laser and is what that paper itself developed against.

**Phase 4 -- confirmation.** The final configuration against B0 on `confirm_static_s1` and `s2` (two more caches). Settings stay in seconds, so the result carries over to A49.

**Why.** Stage 3 is the part of the design that should show temporal information pays, and its only result (80.10%, on a stage 2 that no longer loads) came from one run with a badly conditioned gate, a corrupted time step and a miscalibrated threshold.

### A56. Tracking-literature baselines, metrics and the paper's claims -- **DONE 2026-10-03** (E14-E16, E20-E22); E17-E18 wait for A49

**What it was.** The experiments that decide which paper claims survive the person-*tracking* literature (`docs/RESEARCH.md` §7.4), pre-registered in `memory/experiments/2026-09-tracking-claims.md` before they ran; that file now carries the outcome, every prediction scored, and what changes in the paper. Code: `utils/tracking_baselines.py` (`rule`, `track`, `arms`, `report`, `latency`), `ByteTracker` and `cbmot_update` in `takheliped/utils/tracking.py`; results in `checkpoints_three_horizon/a56/`.

**Result, FROG test, every 5th annotated frame, 95% block-bootstrap intervals:**

| what reads the stage-2 candidates | AP | vs candidates |
| --- | --- | --- |
| nothing | 77.66 | -- |
| SORT (`min_hits` 1) / ByteTrack as published | 77.40 / 76.49 | -0.25 / -1.17 |
| the memory's bookkeeping with its value as the score (E15) | 69.51 | -8.14 |
| ByteTrack + CBMOT, tuned on val (E14) | 80.96 | +3.30 [+2.83, +3.77] |
| learned scorer, no state (E10, 2 seeds) | 81.94 | +4.28 |
| **learned state, no attention (E13, 3 seeds)** | **83.86** | **+6.20; +2.90 over E14, every interval above zero** |

- **The pre-registered decision landed in the middle row**: a tuned track-confidence rule recovers about half the gain, the learned state the rest. The rule needs a confidence decay; without it persistent phantoms saturate as people do.
- **Where the learned state beats the rule**: occluded people (+2.9 pp recall at matched FP over the rule, +0.8 for visible ones) and a trajectory's first 0.5 s (+3.0). The gain is flat across range and motion.
- **Latency**: no confirmation delay (first-frame detection 85.8% against the candidates' 84.1%; the tuned rule 74.1%); the whole streaming detector runs at 12.9 ms per frame on the GPU and 19.5 ms on four CPU threads.
- **GOSPA** ranks the memory best (0.502 against the candidates' 0.530), a smaller margin than predicted.
- **On the published LFE detectors** ByteTrack trades much better than SORT on LFE-PPN (27.6 against 10.4 false positives per false negative) and the merge radius still beats both (`docs/RESEARCH.md` §7.4.7).

**Left open:**

| # | what | waits for |
| --- | --- | --- |
| E17 | the slots scored as a tracker: HOTA, MOTA, IDF1, OSPA(2) on JRDB's 3D tracks projected to the ground plane, and Plozza et al.'s mocap if they share it | A49's JRDB loader |
| E18 | self-supervised memory, or DROW label densification from slot tracks (Tokmakov 2022, Jia 2021, Qi 2021) | A49 |
| ~~--~~ | ~~the test annotations that link to no trajectory~~ | **explained 2026-10-05: duplicate labels, not people** (below) |

**The 2,254 unlinked test annotations are FROG's own duplicate labels** (2026-10-05, `checkpoints_three_horizon/a56/duplicates/`). The test file carries 2,254 circles that repeat another circle of the same frame in all six columns, and they are exactly the annotations the report could not place on a trajectory, one for one (its reverse lookup takes the first of two identical positions). They come in runs (54 runs, median 16 consecutive annotated frames) across 19 of 47 recordings: a tracked label copied by the annotation tool. The memory's "39%" on them was double reporting, a second detection beside a person matching the copy: at matched false positives E13 puts a second detection within 0.5 m of 12-15% of people against the candidates' 1.7%. **Every A56 claim survives the clean labels**: E13 over E14 +2.74 / +3.02 / +2.36 per seed, all intervals still above zero (from +2.95 / +3.20 / +2.56); every AP rises 0.7-1.05 pp, the double-reporting methods least. Facts and counts per file: `memory/data-model.md`; how to read a FROG number with them: `memory/interpreting-evaluation.md`. The double reporting itself was tested as a precision lever the same day and is not one (A58).

E19 ran as A51 S2. The Mahalanobis gate turned out to need a re-tuned Kalman filter first (§D, per-track adaptive gating).

### A50. After the first FROG result: post-processing, deferred ablations, architecture experiments -- **phase 1 closed 2026-10-09**

> ### Triage, 2026-10-09 (owner asked for the remainder of this item)
>
> **Phase 1 is closed.** Items 1, 2 and 5 were already answered; 3, 4 and 6 are answered or closed below; item 7's last piece,
> the person-free frames, now has a script (`utils/empty_scene.py`) and is the one measurement this item still owes.
>
> **Phase 2's temporal ablations are moot, not deferred.** A60 removed stage 2's coarse temporal convolution on 2026-10-08, and
> the fine one went on 2026-09-17, so **stage 2 reads one frame** and there is no temporal horizon left to cap, shift or respan.
> Every row of phase 2 and every stage-1-2 row of phase 3 that names a lag, a tap, a span or a cache is struck below for that
> reason, with the measurement that closed it. The `balanced` re-run was scheduled as the **artefact test for those ablations**;
> the arms no longer exist, so what is left is a second-partition check of the adopted recipe -- a smaller, different claim,
> re-tiered to §D.
>
> **What stage 3 kept** moved to A51 (D1-D5, E6-E13) and is settled there; phase 1b's items 3 and 4 are its ancestors and are
> marked accordingly.
>
> **The person-free measurement is built and queued**, as `utils/queue_a50_person_free.py` -- the first queue module of A57's
> runner, one job per seed of the adopted recipe, about 45 minutes each (28 of them building that stage 2's candidate cache over
> all 195,058 frames of the three raw recordings, ~1 GB per seed). `--list` says what is left; `--steps person_free_s0` runs the
> headline seed alone.
>
> **What is genuinely left in this item:** the person-free frames (item 7), and the phase-3 rows that name no removed component:
> range anchors for far people (now motivated -- A59 found width's whole gain is on far people), the metric-width first layer, a
> plain residual U-Net, every-beam candidates, slot confirmation, a learned room grid, training everything together, and an
> identity loss. None of them is blocked; all are one training run each, and none is scheduled.

**Owner's plan (2026-09-16), in three phases in this order.** Phase 1 explains and cleans up what A43's chain produced; phase 2 runs the §9 ablations that were delayed because each needs a training run; phase 3 tries the alternatives `docs/PROPOSAL.md` §12 kept for experiments, which are now comparatively cheap because each stage has a measured baseline of its own.

**Phase 1 -- post-processing of the FROG result** (each item is cheap, or one short retrain):

| # | item | why, and what it needs |
| --- | --- | --- |
| 1 | ~~replay the oracle on **our** candidates~~ **done 2026-09-16** (`utils/takheliped_oracle.py`, `checkpoints_three_horizon/oracle_ours.json`) | the ceiling above our own stage 2 is **+6.5 pp at a 2 s bridge** (86.1% -> 92.6%), not the +9.6 pp `docs/PROPOSAL.md` §9 quotes from LFE-Peaks, and **stage 3 takes 17.5% of it** while losing 6,076 of the people stage 2 had found (+2,113 recovered, net -3,963). The memory's +3.2 AP is bought with precision, not recovered recall. Feeds item 2 |
| 2 | ~~find the cause of step 3b's regression~~ **diagnosed 2026-09-16** | **stage-2 weight drift, and the feedback prior never activated.** Stage 2 alone lost 1.24 points (76.92% -> 75.68% with the prior silenced, which is *exactly* its score with the prior active), all of it recall (87.3% -> 84.5%, precision flat at 58.4%). The prior's own weights never left zero — norm 0.049 against the shared columns' 7.08 — so step 3b never tested feedback at all. Selection compounded it: `best.offer` ranks on the *joint* AP, and epoch 1.5 won (78.98%) over epoch 1.0 (78.37%) even though stage 2 was better at epoch 1.0 (val 76.12% vs 75.68%). Drift is diffuse, 2.97% overall, concentrated in `coarse.conv.weight` (5.44%), `head.weight` (5.29%) and the norm biases. **Revised by item 4's duration curve:** ordinary training loses 1.23 pp between epoch 1.00 and 2.00, as much as joint training's 1.24 pp over 1.5 epochs, so "more training" now suffices to explain a loss of this size — not excluded as the joint objective's doing, since step 3b ran at a tenth of the control's rate, but no longer requiring it. Feeds items 3 and 4 |
| ~~3~~ | ~~retry joint training~~ **split and closed 2026-10-09** | The joint stage existed to carry feedback, and **A51 measured feedback null in three configurations with the cause measured** (the prior carries 0.012-0.045% of its layer's input), so the owner's adopted design removes step 3b. The half of this item that was about validation rather than feedback is **done**: `evaluation_segments` now places windows where the annotations are. On the 2026-10-09 val cache **6 of 64 windows scored nothing**; after the change, 63 windows, **none empty**, 10,091 scored frames against 9,924 -- more signal from less replay. What joint training leaves untested is whether training stage 2 *with* the memory shapes better candidates, which is phase 3's "everything trained together" row |
| 4 | regularise stages 1-2 | both step 2 runs peaked near epoch 1 while training loss kept falling — but that decline is **validation-side only**, and the one over-trained checkpoint on disk (`step2_continued`, 0.25 epochs past the best) scores **76.91% on test against step 2's 76.92%**, so test-set overfitting is not yet demonstrated. **Dropout landed 2026-09-16** (owner's suggestion) as `step2 --dropout`: the head's input only, default 0.0 so every result so far is reproduced, `docs/RESEARCH.md` §5.4 intends 0.1; scan mirroring, range jitter and weight decay still open. `step2 --checkpoint-every` landed the same day (also default 0.0) so a run keeps **periodic** checkpoints and not only the best, which is what lets test AP be measured against training duration for the first time — the gap that left "does stage 2 overfit on test?" unanswered. **The 76.92% baseline cannot be reproduced under current code**: its stored args carry no `schedule`, `lr_factor`, `lr_patience` or `min_lr`, so it predates the plateau scheduler, and a lone dropout run would differ from it in two variables. The dropout run is therefore paired with a **control at dropout 0.0 on today's code** (`step2_control` and `step2_dropout01`, launched 2026-09-16), and the comparison is control against treatment. **Control finished: test AP 76.62%** (val best 73.88% at epoch 1.00, stopped at 4.00) against the historical 76.92% — today's code reproduces the baseline within **0.30 pp**, and both runs peaked at epoch 1.00 while the plateau scheduler first cut the rate only at 2.25, so the scheduler did not pick the checkpoint and the gap is run-to-run nondeterminism. **Read any dropout effect below ~0.3 pp on test as noise**; val is noisier still, swinging >1.5 pp between adjacent evaluations. 16 periodic checkpoints (0.25-4.00) are kept, so the duration curve is measurable as soon as the GPU frees. **Dropout 0.1 finished 2026-09-16: test AP 77.32%** against the control's 76.62% — **+0.70 pp**, about 2.3x the measured run-to-run spread, and above even the historical 76.92%. False positives fell by a third (2.653 -> 1.770 per frame) for a small recall cost (FN 0.341 -> 0.379) — but **the operating point is itself noisy**, and this is suggestive only: the two dropout-0.0 runs differ by 0.74 FP/frame (1.910 against 2.653), nearly as much as dropout's own 0.88. The shuffled-history penalty is firmer evidence, since the two dropout-0.0 runs agree at 1.30 and 1.18 pp against dropout's 2.00 pp, so the regularised network leans *more* on temporal evidence. **Validation showed none of it** (73.88% -> 73.93%, inside val's own >1.5 pp swing), one more reason not to rank stage 2 on this val sample. One pair at one seed, so it stays **provisional**: the confirming seeds are deferred to the shared confirmation stage below (owner's call 2026-09-16) rather than run bespoke for dropout, since dropout is one model option among many and all of them are judged the same way. **Duration curve, the first on test** (`duration_curve.py`, `checkpoints_three_horizon/duration_curve.json`): control 75.26 / 76.62 / 75.39 / 73.90 / 73.04% at epochs 0.25 / 1 / 2 / 3 / 4 and dropout 75.72 / 78.16 / 75.93 / 74.78% at 0.25 / 1 / 2 / 3. **Stage 2 overfits on test**, peaking at epoch 1 and losing 3.58 pp by epoch 4 as precision climbs 48.8 -> 73.5% and recall falls 87.8 -> 76.5%; dropout lifts every point (+0.46, +1.54, +0.54, +0.88) **without flattening the decline**. Val mis-selected the dropout run: it chose epoch 0.75 (77.32% test) where epoch 1.00 scores 78.16% **Closed 2026-10-09.** Every lever in this row is now a default or measured away: **dropout 0.1** and **weight decay 1e-2** are the defaults, **scan mirroring** is on by default since 2026-10-06 (A59's recipe, `--no-mirror` to turn it off), and **range jitter is not worth running** -- `scan_anomalies.py` puts consecutive-frame jitter at 9-20 mm under 10 m, far below the 0.5 m match radius, which is the same measurement that closed the fine convolution (`memory/noise-structure.md`, `memory/rejected-ideas.md`). Stage 2's recipe was then settled end to end by A59 (82.2% stage 2, 85.9% whole detector) |
| ~~5~~ | ~~decaying rate for step 3a~~ **done 2026-09-20** | Its cosine schedule spans `--max-epochs` (100) per chunk length, so it hardly decays in the 9-37 epochs a stage runs. `step3a --schedule plateau` landed with `--lr-factor`, `--lr-patience` and `--min-lr`, and was screened as A51's **T2: 83.89% against B0's 83.34% mean** (+0.55, inside the threshold and inside B0's own range), with recall at matched false positives and coasting precision above every B0 seed. **The default stays cosine** on that reading |
| ~~6~~ | ~~remove the time budget~~ **done 2026-10-09** | Owner's call: runs are bound by epoch budgets and early stopping anyway. `--max-hours` is gone from `step2`, `step3a` and `step3b`, and `EarlyStopping` no longer takes or checks hours. It only ever truncated a healthy run -- A51's first full-curriculum step 3a died inside its first chunk-length stage at the 6 h default, and every A51 launcher then had to pass `--max-hours 24` by hand to avoid it. A run that should stop now stops on `--max-epochs` or `--patience` |
| 7 | report the remaining §9 **metrics** (these are measurements, not ideas; the architecture ideas are phase 3 below) | **Three of four done.** AP at 0.3 m and milliseconds per frame have been in every stage-3 report since A51 phase 0c and in every A59/A60/A61 table. **People standing still: done by A56/A61's E20**, which splits recall at matched false positives by occlusion, range, trajectory age and stationary-or-moving -- and the prediction *missed*: the memory's gain is **+3.5 pp for stationary people against +4.1 for moving** ones (`memory/experiments/2026-09-tracking-claims.md`). **The 72,533 person-free frames are the one piece left**, and now have a script: `utils/empty_scene.py` counts detections per frame, per second, the share of frames holding any, and the share recurring within 0.3 m in the next frame, for candidates and for the memory at five thresholds, against LFE-Peaks' 1.39 and LFE-PPN's 1.94 (`memory/performance-log.md`'s three `gap` rows). **It also splits by time since anyone was in view**, because the three raw recordings interleave populated and person-free frames and a 10 s memory can report someone who has just left -- an explanation a static detector cannot have, so the count is reported again over the frames beyond the slots' fade time **DONE 2026-10-09, three seeds** (`results/empty_scene_s{0,1,2}.json`): on all 72,533 person-free frames the whole detector emits **0.740 FP/frame against stage 2's 2.622**, a **72% cut**, and **34.1% of frames hold one against 67.2%** -- stage 2 alone is worse than both published detectors (1.39 LFE-Peaks, 3.17 LFE-PPN) and the memory takes it **below LFE-Peaks**. At 0.9 confidence: **0.010 FP/frame, 0.9% of frames**, against the published detectors' third to a half. **Not a stale-memory artefact**: beyond the 10 s fade the cut is larger (-76%). What it keeps is furniture (92.2% recur next frame against the candidates' 84.9%), which feeds Tier 2 item 13. **Found en route:** LFE-PPN's person-free row in `memory/performance-log.md` was the stale pre-recalibration 1.94 while its frames-with-a-phantom cell had been updated; corrected to 3.17, and its "with SORT" cell marked stale. |

**Phase 1b -- why stage 3 underperforms** (**its open items continue in A51**; owner's call 2026-09-16, after item 1: "stage 3 underperforming is exactly what we should put an effort into ... do the slots preserve enough information? Is the slot invalidation rule enough? Are there enough slots?").

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
| ~~2~~ | ~~compare the rescoring delta on true-positive candidates against non-matching ones~~ **done** | `rescore_shift_person` and `rescore_shift_other` have been in every step-3a report since phase 0c. On the adopted recipe (E13, three seeds) they are **-3.61 / -3.88 / -4.17 logits on candidates covering a person against -0.54 / -0.38 / -1.43 on the rest**: the head pushes the whole distribution down and pushes *people* down hardest, which is phase 1b item 1's distribution shift rather than selective suppression of true positives. A one-sided gate was then tried as A51's O1 and diverged twice (NaN at the 10 s chunk stage, cause not established) |
| 3 | does slot state survive what matters? | A retired slot loses everything -- `state *= alive`, and a refilled slot gets fresh `spawned` state -- so a person re-detected after a 10 s gap starts blank. **Parked, with a measurement against it:** A51's `--recovery` pass found a live slot already sitting on **95.5%** of the people stage 3 fails to report, median value 0.986, so retirement is not what loses them; and E1's horizon sweep puts 95% of the memory's gain inside 10 s. Reporting is the limit, not survival |
| ~~4~~ | ~~ablate the hand-coded association~~ **moved to A51, where it was run** | `manage_slots`' 0.6 m mutual-nearest gate governs value, spawn and retire under `@no_grad`. A51 ablated the learned half instead and found it is the load-bearing one (E6 removes the slot-to-slot exchange for nothing, E7 the learned matching for nothing, **E10 removes the learned state and that is the only ablation that costs anything**). The rule-based half's own ablations live there as D4 (`--learn-value`, implemented, never run) and D5 (a learned association width, not implemented) |

**Phase 2 -- the heavy §9 ablations**, one training run each (the evaluation-only ones are done: SORT on our candidates, 64/128/256 slots, stage 2 alone, and step 3a itself is "no feedback"):

| ablation | what it tests |
| --- | --- |
| remove the fine temporal convolution | the calibration horizon (~0.08 s). **Screening launched 2026-09-16** as `step2 --fine-lags 0`, all else matching the control. **The ablation is verified by the parameter count**, 1,179,011 against the control's 1,181,059 — exactly the 2,048 that `Conv1d(96, 32, 1)` -> `Conv1d(32, 32, 1)` predicts, so the flag reached the model rather than being silently ignored. Check every later ablation the same way. **Result: test AP 78.03% against the control's 76.62%, +1.41 pp** — about 4.7x the measured run-to-run spread, and above even the dropout run's 77.32%. Removing the calibration horizon *helps*, and it cannot be a capacity effect, since the ablation takes 2,048 parameters away (0.17%) and still gains; it also won despite selecting a later checkpoint (epoch 1.50 against 1.00) where the duration curve says test AP is already falling. The shuffled-history penalty drops 1.18 -> 0.65 pp, so the network still reads history, but only through the coarse convolution. **Validation could not see it**: 73.76% against the control's 73.88% on the 2,000-frame sample, and 73.88% against 73.79% when every val frame is scored — a tie either way, against a 1.41-point difference on test (`memory/interpreting-evaluation.md`). One seed, screening only — confirm before quoting, because it contradicts `docs/PROPOSAL.md` §5.3-§5.4's premise that the ~0.08 s horizon carries signal, and DR-SPAAM's published +1.9 AP is credited to a comparable window |
| remove the coarse temporal convolution | the ~1.2 s motion scale. `step2 --coarse-lags 0`, expected parameter count **979,843**: the conv is `Conv1d(128*5 + 4*3, 128, 3)` = 250,496 params against 49,280 ablated (cross-checked against the drift inspection, which reported `coarse.conv.weight` at 250,368 weights plus 128 bias). **Capacity confound, recorded before the result:** this strips 201,216 parameters, 17% of the network, where the fine ablation strips 2,048 (0.17%), so a fall in AP cannot be read as the ~1.2 s span alone. The capacity-matched control is `--coarse-lags 0 0 0 0 0` — five taps on the current frame, keeping the conv 640 inputs wide against 652, which removes the temporal information without the capacity. **Result: test AP 77.01% against the control's 76.62%, +0.39 pp** (val best 73.01% at epoch 0.75; parameter count 979,843 as predicted). **The pre-registered confound does not bite**: it was registered against a *fall*, and this is a rise — losing 17% of the parameters and still gaining is the harder direction, so capacity cannot explain it away, and the matched control is not needed to defend this result. The shuffled-history penalty collapses 1.18 -> 0.18 pp, leaving the network almost static: with the coarse conv gone its whole history is the fine conv's 3 frames (0.08 s). Training is also 3.5x faster, 46 s against 165 s per epoch. One seed, screening only |
| ~~cap every decay time at 1 s~~ | ~~whether the long horizon carries anything~~ **moot since 2026-10-08**: A60 removed the coarse convolution and the fine one went 2026-09-17, so stage 2 holds no decay time to cap. The question moved to the slots, where A51's E1 answered it: 95% of the memory's gain is inside 10 s |
| sensor-frame x, y instead of beam-relative offsets | the input description and cache validity. **Still open and no longer about the cache**, which went with the temporal convolutions: it is now purely a question about the vote target, one training run, and A59's decomposition puts about half the lead over LFE-Peaks in the output head, which is where this would act |
| then repeat the headline runs on the `balanced` split | §10 step 4: `official` first, then `balanced`. **Never mix partitions** (checked 2026-09-16): `balanced`'s val is 11-36 + 10-31 and **11-36 is half of `official`'s train**, so validating there while training on `official` would score the model on its own training data. Stay inside `balanced` for train/val/test — its test is 16-41 + 15-53 and holds `frog_16-41` *whole* precisely so the official-test subset is extractable from the same evaluation pass, giving the balanced number and a published-comparable one in one run. Its val is 121,329 frames with 19.8% empty against official-val's populated-only sample, which is the weakness that mis-selected the dropout checkpoint (owner's point 2026-09-16: use `balanced` where `official` fails, but never quote a published baseline on it). State the deviation when quoting: published baselines trained on 11-36 + 12-43 where `balanced` trains on 12-43 + 14-57, so the test recording is held out for everyone but the training sets differ. **Owner's call 2026-09-16: re-run the temporal ablations here, at the very end, as an artefact test.** On `official` they came out backwards — removing the fine convolution gained +1.41 pp and removing the coarse one +0.39, while validation ranked the arms in nearly the opposite order — and neither the literature nor the architecture explains it (`CHANGELOG.md`). If the same ablations behave differently on `balanced`, what we measured is an `official`-split artefact rather than a property of the model. `balanced` tests both suspects at once: **two** test recordings against `official`'s one, and a val split of 121,329 frames with empties against a 2,000-frame populated-only sample. **Sharper still, and the reason to prefer this over a plain replication:** because `balanced` holds 16-41 whole, one evaluation pass yields both subsets, so the ablation effect can be compared on 16-41 against 15-53 **with identical weights, varying only the frames**. If it appears on 16-41 alone it is pinned to that recording; if on both, the finding is real and `official` was never the problem **Moved to A63 step 0 on 2026-10-10** (owner: `balanced` is functionally another dataset). Was re-tiered to §D 2026-10-09: it was scheduled as the artefact test for the temporal ablations, and both convolutions have since been removed, so the arms in question no longer exist. What is left -- does the adopted recipe hold on a second partition? -- is a smaller claim for just before publication. |

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

*Ruled out: alignment artefacts.* `tests/test_takheliped.py::test_the_fine_cache_is_almost_hole_free_at_its_own_lag` pins `filled.mean() > 0.98` at this lag's own motion (~4 cm, 0.8°), so the tap is close to a clean copy, not a hole-punched resampling. Beam-index rounding, nearest-wins collisions and field-of-view losses all stay small at 0.8°. Do not re-propose this without new evidence.

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
| ~~cache raw scans and recompute each frame, or scans for the first layer and features deeper~~ | ~~caching motion-corrected features~~ **moot since 2026-10-08**: with no temporal convolution there is no feature cache to choose a representation for |
| ~~shifting part of each layer's channels in time (TSM)~~ | ~~the fine temporal convolution~~ **moot since 2026-09-17**: the fine convolution is gone, and five taps tied no taps at three seeds before it went |
| ~~coarse span, e.g. 9 frames between taps (~1.5 s)~~ | ~~8 frames (~1.3 s)~~ **moot since 2026-10-08**: A60 measured the coarse convolution end to end against the slots (AP -0.87, first-second recall +0.59 against a +2 bar) and removed it |
| ~~DR-SPAAM's running template with local attention~~ | ~~dilated causal convolution~~ **moot since 2026-10-08**: there is no cache to align, so the condition this was gated on cannot arise |
| plain residual U-Net, or widening 2 | ConvNeXt blocks, widening 4. **Open, and half-priced by A59's W arm**: at LFE's widths (8/16/32, 63,891 parameters) the same network loses 1.3-1.5 pp, all of it on far people, and is **no faster at batch 1** (11.1 ms against 11.5 on CPU) -- so capacity is worth about 1.4 pp and shrinking it buys nothing. Widening 2 and a residual block remain untried |
| range anchors, as LFE-PPN (+10.2 points over LFE-Peaks at 8-10 m) | per-beam score and offset. **Open, and the best-motivated row here**: A59 found width's entire gain sits on far people, and A61 found the memory's lead largest beyond 8 m, so far-range recall is where two independent measurements already point. One training run |

**Stage 3**, object memory:

| idea | instead of | status |
| --- | --- | --- |
| every beam as a candidate | the 64 best after merging | open |
| ~~720 slots exchanging only with nearby slots~~ | ~~256 slots~~ **closed 2026-09-27**: the exchange earns nothing to localise. E6 removes it altogether for AP 83.58 against B0's 83.34, 66,560 parameters cheaper and 9% faster, and capacity never binds (24 of 256 slots in median use, p99 84) |
| fixed decays (LRU, S5), minGRU, or GRU | selective diagonal recurrence. **Open as a family comparison, with two points measured**: one fixed 8 s decay costs -0.90 pp (D3, 17,536 parameters cheaper) and removing the learned state altogether costs the most of any ablation (E10: -1.60 recall at matched FP at 1.9x its bar, -13.80 coasting precision at 1.7x). So the decay being learned is worth little and the state being learned is worth everything |
| Hungarian matching with pseudo-identities, or slots predicting their own position (MOTR) | learned soft attention. **Open, and the learned attention it was to replace is gone**: E7 removed the learned matching for nothing, so the comparison is now against a rule-based assignment. A56/A61 priced the published trackers instead (ByteTrack + CBMOT tuned on val reaches +1.32 of the memory's +4.13), which is the baseline a reviewer asks for |
| ~~sensor coordinates~~ | ~~room coordinates~~ **closed 2026-10-02, conditionally**: room coordinates win at three seeds (D1, -1.14 pp for the sensor frame, p = 0.034) and the crossover with drift sits near 0.015 m/s (E2). Training with jittered odometry (S3) buys +1.75 pp at 0.05 m/s and +4.34 at 0.1 for -0.38 at zero, which is free robustness but **does not move the crossover**, so the claim stays conditional and a deployment with worse odometry should prefer the sensor frame |
| confirmation over several frames, or a higher creation threshold | creating a slot from any candidate above 0.01 | open |
| ~~**where memory plugs in**: feedback at the detector input, or no feedback~~ | ~~feedback at the bottleneck~~ **closed 2026-09-30**: measured null in three configurations (E12a with the rescoring head frozen out, E12b with a ground-truth prior and no competitor, E12ab with both), **cause measured** -- the prior carries 0.012-0.045% of its layer's input -- and **ceiling measured**: 1.91% of people, every one already held by a coasting slot. The adopted design removes the path and the joint stage it existed for |
| a learned grid over the room (Deep Tracking on the Move) | memory addressed per object | open; hand-built grids failed (§2.4), a learned one is untested |

**Training and evaluation**, what is trained and how it is judged:

| idea | instead of | status |
| --- | --- | --- |
| everything trained together from the start | phases A, B, C. **Open, and the only live descendant of phase 1 item 3**: feedback is closed, but whether training stage 2 *with* the memory shapes better candidates is untested. It needs stage 2 retrained, so it is a chain rather than a run, and it should not be attempted before the paper's numbers are frozen |
| an added identity loss from pseudo-identities | detection and slot-value losses | open |
| memory always fresh | carried across chunks with 20% random resets. **Open, and bounded by E1**: emptying the memory every 0.5-30 s at test time shows 95% of its gain inside 10 s, so the training reset rate can matter at most over that span |
| LFE ported to PyTorch as a published baseline | the same model with memory reset every frame. **This is A59 phase 3b**, proposed 2026-10-09 and awaiting the owner: retrain LFE with its authors' code, shuffling fixed, selected on our val AP. It supersedes A32c's ONNX port |
| `balanced` first | `official` first | **decided**: `balanced` re-runs at the end, as the artefact test |

**Order and bookkeeping:** phase 1 items 4-6 change training settings, so run them before phases 2-3 and state the settings with every later number; A49's cross-dataset runs use whatever phase 1 settles on.

**The documentation refresh waits for the end of phase 3** (owner's call 2026-09-16). It is now `docs/PAPER.md` and `memory/performance-log.md` only: **`docs/SHOWCASE.md` was deleted 2026-10-10** (owner's call), because a standalone one-page summary of the result duplicates what the write-up says and had already gone stale on two points — it called the joint-training regression "under investigation", which phase 1 item 2 closed, and it presented step 3b as a test of feedback when that run's prior never left zero.

**How every model option is judged** (owner's call 2026-09-16: dropout is one option among many, so one procedure covers them all). Single runs cannot resolve the effect sizes at stake — two identically configured runs differ by ~0.3 pp of test AP and 0.74 false positives per frame, against dropout's own +0.70 pp — and with ~14 options across phases 2 and 3, chance alone would produce several convincing-looking winners. So:

1. **Screening.** Each option gets one run at seed 0 with all other settings shared, reported as test AP with precision, recall, FP and FN per frame and the shuffled-history penalty, against the shared dropout-0.0 control (76.62%). Phase 2's "one training run each" *is* this stage.
2. **Confirmation.** Only options moving test AP by more than the current spread estimate advance; they are re-run alongside the control at two or three further seeds, and reported as a mean and spread. Only what survives is adopted into later settings.

Every quoted number states how many seeds it rests on, and single-seed numbers are labelled provisional — dropout's +0.70 pp included. Because confirmation re-runs the control anyway, the spread estimate tightens as a by-product, so the threshold sharpens as it is used and is never paid for separately.

### A49. The three-horizon detector on all three datasets, parameters stated for each -- **split 2026-10-09 into A62 and A63**

> **This item is now the record, not the queue.** The owner's call of 2026-10-09 postponed every other-dataset task and asked
> for it as two clear items: **A62** makes the code dataset-agnostic, **A63** trains and reports on DROW and then JRDB. The
> findings below -- the anomaly checker's results, FROG at 40 Hz, DROW at 12.7 Hz, the missing-return rule, and the per-dataset
> table -- stay here because other documents cite them, and both new items inherit them.

**Owner's goal (2026-09-15): an adaptive model.** Train it and report numbers on FROG, DROW and JRDB, each with an explicit table of the parameters used. Order: FROG first (A43 steps 3a -> 3b -> 4, running), then DROW, then JRDB (§C).

**What the current code ties to FROG** (checked 2026-09-15):

- `utils/train_takheliped.py` hard-codes `FROG_Dataset` and `frog_laser_angles(720)`. Its frame period is measured from the data since 2026-09-17 (A51 0b).
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

- ~~`train_takheliped.FRAME_PERIOD_S = 1/26.2` set chunk lengths and validation windows 1.53x short~~ fixed 2026-09-17 (A51 0b).
- The coarse lags span 0.80 s, not 1.22 s. The "~1.5 s" in the temporal hypothesis is really ~0.8 s.
- Stage 3's time step comes from the stamps, clipped to 1 ms, and the slot velocity divides by it (see A50, stage 3 time step).
- `evaluate._frame_period_s` takes the median, so every FROG `fp_per_s` is 1.53x too low.
- `motion_analysis` (`FRAME_HZ = 26.2`), `persistence_map` (`--hz 26.2`, `SPANS`), `slot_budget`, `horizon_sweep`, `takheliped_oracle` and `informational_capacity` (`DT_NATIVE_S`) are affected, along with the tests that pin 26.2.
- The 38.2 ms real-time budget is really 25 ms; the measured 6.5-12.3 ms per frame still fits.

**How:** one shared helper for the rate (mean over linked intervals), used everywhere; re-run the DROW and FROG analysis rows; restate the FROG seconds in `memory/dataset-properties.md`. **Why:** every FROG figure in seconds, and every seconds-based flag of the three-horizon trainer, is off by 1.53x.

**DROW runs at 12.7 Hz, not 10 Hz (found 2026-09-17).** **What:** DROW's timestamps are rounded to 0.05 s, so the median interval reads 0.1 s. The per-file mean is 12.4-12.7 Hz, matching the paper's 12.5 Hz. **How:** `utils/motion_analysis.py`, `slot_budget.py`, `horizon_sweep.py` and `takheliped_oracle.py` all take `1 / median(diff(time))`. Switch them to the mean over each unbroken sequence, and check that FROG's 26.2 Hz survives the same change. Then re-run their DROW rows, whose durations in seconds are 1.27x too long. **Why:** every DROW figure stated in seconds is affected, including the lag conversions in the table above.

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

0. ~~**Interface and caches, tests first**~~ **done 2026-09-14**: `library/takheliped/detectors/takheliped.py` (`beam_features`, `align_beams`, `align_sectors`, `CausalTemporalConv` with `TemporalCache`), 21 tests; streaming equals the clip frame by frame; checked on the real FROG test recording (`CHANGELOG.md`). Value-ranked eviction keeps its existing test in `utils/slot_budget.py` until step 1 needs it in the model.
1. **Built 2026-09-14, owner-approved to train:** stage 3 alone on frozen LFE-Peaks proposals, on ROCm -- `ObjectMemory` in `takheliped.py`, `utils/train_takheliped.py step1`. Gate: beats SORT on the same proposals and the merge radius at matched recall.
2. **Built 2026-09-14, owner-approved to train:** stages 1-2, phase A, on ROCm -- `CalibrationNetwork`, `train_takheliped.py step2`. Gate: A35's two criteria (single-frame path at LFE's published AP; shuffled history hurts).
3. Stage 3 on the new stage 2, then joint with feedback.
4. Ablations (`PROPOSAL.md` §9).

**Training launched 2026-09-15 08:37**: step 1 then step 2, sequentially, logs and `progress.txt` in `checkpoints_three_horizon/`. Both smoke runs passed first -- step 1 end to end (val candidates 81.8% AP on 4 windows, test 66.6%), step 2 at ~0.21 s per batch of 32 clips (~12 min per epoch, 1.3 GB GPU) with validation rising 52.7% -> 59.2% in 0.03 epochs and shuffled history already costing 2.5 pp on a 100-frame test sample. About 7 hours were lost overnight to a finished smoke run hanging at exit (started before the `os._exit` fix) with nobody watching; the runs are now watched for stalls.

**That run stopped at 10:28 and is not resumable.** The 1 s stage finished its full 40 epochs (best val AP **68.8%** at epoch 37 against **60.7%** for the untouched candidates); the machine then went to sleep and rebooted at 10:29:54, one epoch into the 10 s stage -- no error, no out-of-memory or bugcheck record. That epoch (66.6%) had already overwritten the 68.8% checkpoint, because each chunk-length stage restarted "best" along with patience. Fixed and tested 2026-09-15: `BestCheckpoint` keeps the run's best whatever epoch or stage produced it, each stage also keeps its own best file, stages resume from the run's best, the script keeps Windows awake while it runs, and step 1 defaults to 100 epochs per chunk length (owner, 2026-09-15). **Decided 2026-09-15 (owner): go straight to our own network -- step 2, then 3, then 4; step 1 (stage 3 on LFE-Peaks) is dropped.** Step 2 relaunched 13:24 as a detached process (a background task of the agent session dies with the session), watched for evaluations, errors and stalls.

- **Step 3a (phase B)** -- run the trained stage 2 over every train/val/test frame, decode candidates with **DROW's vote grid** (owner's choice: step 3's AP stays comparable with step 2's) and pool each candidate's decoder features from the beams whose votes land within the grid's collect radius, cache them (float16), and train stage 3 from scratch on them with step 1's loop and budget (100 epochs per chunk length, patience 8).
- **Step 3b (phase C)** -- joint streaming training with feedback: 2 prior channels at the bottleneck drawn from the slots' predicted positions (slot value, person probability), zero-initialised so the trained stage 2 is unchanged until trained; TBPTT chunks through `CalibrationNetwork.step` and `ObjectMemory.step`, loss on both; ~6 GB per 10 s chunk at one stream; budget as approved (3 epochs).
- **Step 4 -- only the evaluation-time ablations for now** (owner's choice): SORT on step 2's candidates instead of stage 3, 64 / 128 slots without retraining, shuffled history. The retrain ablations of `PROPOSAL.md` §9 wait for step 3's result.

**Steps 3a, 3b and 4 built 2026-09-15, tests first** (405 tests pass): `CalibrationNetwork` returns its decoder features and takes `prior_channels` (zero-initialised; `load_calibration_weights` loads step 2's checkpoint into it unchanged); `render_prior` draws live slots into bottleneck sectors; `ObjectMemory(feature_dim=...)` takes candidate features; `train_takheliped.py step3a / step3b / step4`. Choices made while building, to confirm: step 3b trains **one stream** over **10 s** chunks (streams in a batch must start together, and ~6 GB per chunk), evaluates every **0.25** epoch (not 0.125) on **16** val windows, because the joint evaluation streams the whole network and decodes every candidate on the CPU; its learning rate is 1e-4 for both models together. **Validation cadence and learning rate approved by the owner 2026-09-15.** GPU memory, owner's concern: the step-3b smoke run peaked at 1.0 GB for a 2 s chunk at one stream, so a 10 s chunk should need ~5 GB against the ~10.7 GB one process gets -- to confirm on the first real evaluation line, with 5 s chunks as the fallback. **Chain queued to run unattended after step 2 ends cleanly**: step 3a -> step 3b -> step 4, stopping at the first failure. Smoke runs passed 2026-09-15 on one recording each: step 3b end to end (training, validation, test; 1.0 GB for 2 s chunks) and step 4 (SORT at `min_hits` 1/3/5, 64 slots, whole detector). They caught and fixed three failures a real run would have hit: validation windows with no annotated frame (scoring now returns undefined metrics), no checkpoint when every validation was undefined (the last weights are tested, with a note), and a crashed ROCm process hanging at exit (always `os._exit`; the chain launcher also stops a process that lingers 3 min after its log says it finished). Step 3a's candidate caches take ~24 ms per frame to build, ~2.4 h for all 365,542 frames. **Step 2 result (2026-09-15, 14:24): test AP 76.9%, history shuffled 75.6% -- both gates pass** (above LFE's published 65.6% / 69.2%; shuffling costs 1.3 pp, against 0.5 pp for this project's earlier SpaceTimeCNN). 1.91 FP / 0.39 FN per frame at 0.3, against LFE-Peaks 1.76 / 0.63 and LFE-PPN 5.03 / 0.38. Best val AP 73.8% at epoch 1.0; early stopping at epoch 4.0 while training loss kept falling (0.183 -> 0.076), i.e. overfitting at the undecayed rate. **Step 2's learning rate never decayed** (owner's question 2026-09-15): cosine over the 20-epoch budget left it at ~95% by epoch 2.75, while val AP had stalled since epoch 1.0 (73.8%) and early stopping was due near epoch 4. **Owner's call: continue from the best weights with a plateau schedule** -- rate x0.3 after 4 evaluations (1 epoch) without improvement, floor 1e-5, starting at 3e-4 -- written to `checkpoints_three_horizon/step2_continued/`; plateau is now step 2's default (`--schedule`, `--init`). The chain was stopped before step 2 ended and relaunched to run the continuation first, then steps 3a -> 3b -> 4 on its weights. **Known waste, to fix later:** 6 of step 3b's 16 validation windows fall on unannotated stretches (val is 62.8% annotated; recording 0's first annotated frame is 9,233), so they cost evaluation time and score nothing. **Continuation result (2026-09-15, 15:12): no gain -- stage 3 builds on the first run's weights.** Val AP 73.0% at its first evaluation (epoch 0.25), then down every evaluation to 69.2% at epoch 3.25 while training loss fell 0.132 -> 0.036; the rate cuts to 9e-5 and 2.7e-5 did not stop it. Its best weights test at 76.9% / shuffled 75.6% (FP 1.43 / FN 0.49 per frame), the first run's AP at a different operating point. A checkpoint guard compared the two runs' best val AP (73.0% < 73.8%) and put the first run's weights in the continuation's place, but only 4 s after step 3a had started reading it, so step 3a was stopped and the chain relaunched on `checkpoints_three_horizon/step2_calibration.best.pth` directly. **Lesson:** a continuation's `BestCheckpoint` starts empty, so it must be seeded with the initial weights' val metric, or the chain must pick the better checkpoint before launching the next step, not race it. **Future experiment:** stages 1-2 overfit the 120k training frames within ~1 epoch at any tried rate -- try augmentation (scan mirroring, range jitter), weight decay and dropout before any longer schedule. **Step 3a crashed 7 min into its candidate cache (2026-09-15, 15:20) on a vote-grid bug, now fixed:** two neighbouring cells whose blurred votes tie exactly are both maxima in `votes_to_detections`, every vote goes to the nearer one, and the other became a detection averaged over no votes -- NaN position and probability -- which broke the Hungarian candidate match. Fixed at the source (a peak with no votes is skipped), with `tests/test_vote_decoding.py` (reproduced by a random vote search, watched failing first; 411 tests pass). NaN detections pass every distance gate, so `_prec_rec_2d` could have counted them as true positives; re-scoring step 2's test set found 2 among 800,856 detections and AP 76.9227% -> 76.9218%, so the step 2 result stands. The chain was relaunched on the fixed decoder at 15:36. **Step 3a's budget, as run (2026-09-15):** `--max-hours 6` counts from the start of training across all chunk lengths, not per chunk length (`train_memory`'s `start`, and the break after each stage). The 1 s stage stopped early at epoch 37 (1.5 h), the 10 s stage at epoch 9, and the 30 s stage is cut by the total at about epoch 9, while still setting new bests (77.32% val at its epoch 5). **Owner's calls (2026-09-15), for after the current chain:** (1) **remove the time budget** (`--max-hours` and its checks) from every step of `train_takheliped.py`: runs are bound by their epoch budgets and early stopping anyway. (2) **Give step 3a a decaying rate**: its cosine schedule spans `--max-epochs` (100) per chunk length and so hardly decays in the 9-37 epochs a stage actually runs -- the same undecayed-rate pattern step 2's continuation addressed; a plateau schedule as in step 2 is the obvious candidate. **Step 3a's test pass ran out of host memory (2026-09-15, 22:40), after training finished:** `evaluate` gathered 32 whole test recordings at once (up to 6,796 frames x 64 candidates x 32 features, 1.66 GB for the features alone) with the machine near its commit limit; validation windows are short, so training never hit it. Fixed: frames are gathered in 512-frame slices with the slots carried across (`tests/test_train_takheliped.py::TestEvaluate`, sliced equals whole; 412 tests pass). `evaluate_joint` already reads one frame per step. The checkpoints were intact (best: 30 s chunks, epoch 5, val 77.32%), so the saved best is evaluated on test without retraining, and the chain continues with steps 3b and 4 (relaunched 22:48). **Step 3a test result (2026-09-15, 23:09): stages 1-3 reach 80.10% AP on every annotated test frame (every 5th: 80.15%)**, against 76.92% for stage 2's candidates on the same frames -- +3.2 points; false positives 1.91 -> 1.29 per frame, missed people 0.39 -> 0.47, exchange rate 7.6. One run, stage 2 frozen; recorded in `memory/performance-log.md`. **Owner's request (2026-09-16): report precision, recall and the false-positive and false-negative rates beside AP**, so results can be read against the recoverable-recall ceiling of `docs/PROPOSAL.md` §2.5. `score_detections` now returns precision and recall at the 0.3 operating point alongside the per-frame rates it already had, so every result file carries all four. **Still to do:** that ceiling (+9.6 points of recall recoverable, 4.4 below threshold, 14.1 blind) was measured on LFE-Peaks' detections, not ours -- replay the oracle on the three-horizon detector's own candidates before quoting how close we are to it. Step 3b's test pass was already running when the fields were added, so `step3b_results.json` carries only AP and the per-frame rates; step 4 and everything later carry all four, and step 4 then re-measured every variant (candidates, SORT, stage 3 and the joint model at each slot count) with all four, so no separate re-scoring pass is needed. **Step 4 result (2026-09-16, 05:34, after 131 min; chain ALL DONE):** SORT on our own candidates *lowers* AP at every setting (76.66% / 75.73% / 74.92% at `min_hits` 1/3/5, against 76.92% for the candidates alone), while stage 3 reaches 80.20% at 128 slots -- so the gain belongs to the learned memory, not to temporal filtering in general. Slots do not bind (0.1 points from 64 to 256, both variants), and precision is the weak spot: 66.8% at 84.7% recall, against 58.4% at 87.3% for the candidates. Full table in `CHANGELOG.md`. **Step 3b result (2026-09-16, 03:22, after 250 min): joint training with feedback is a regression, 78.84% test AP against step 3a's 80.10%** (FP 1.235 against 1.293, FN 0.546 against 0.470); its stage 2 with feedback tests at 75.68%, *below* the 76.92% it started from, although step 3b's own validation rose from 74.06% to 76.12% over the run. Step 3a's weights remain the best model. **Why it may have regressed, to try before concluding the feedback design is wrong:** validation was 16 windows, 6 of them with no annotated frame, so checkpoint selection had a weak and biased signal (fix: place windows only on annotated stretches, and use more of them); both models trained at one rate (1e-4), so stage 2 was fine-tuned as fast as the memory was trained from scratch -- a lower rate for stage 2, or freezing it for the first epoch, is the obvious next experiment; and 3 epochs may be too few to recover what the early updates cost.

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
There is no PyTorch module, so retraining here requires a reimplementation.
**Corrected 2026-10-05 (A59): the training code *is* published**, in Keras, at `robotics-upo/2DLaserPeopleBenchmark` (`utils/models.py`, `train_segmentation.py`, `train_localization.py`); the earlier "no published training repo" was wrong.
It specifies the port outright -- five residual blocks of 32 channels, depthwise-separable kernels 9/7/5, max-pool by 2 then 3, the global max on blocks 2-5, BatchNorm and dropout 0.2 -- and it carries one defect a port must not copy: `SegDataGenerator` never shuffles, so the backbone trained on consecutive scans in recording order (A59).

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

### ~~A33-A36. The architecture funnel~~ -- **CLOSED 2026-10-10: TAKHeLiPeD is the rebuild, and it passes A35's own criteria**

These four items are one thread: a cheap ranking test (A33), a candidate survey (A34), build-rank-train the survivors (A35), and a
re-triage afterwards (A36). They were written on 2026-09-10, when the measurement that reorganised the project showed
`SpaceTimeCNN` was a five-frame *averager* whose spatial half scored 63.0%, below LFE-Peaks' published 64.9%. **A35 then fixed two
acceptance criteria in advance, and both are now met** -- by a detector built directly rather than by a funnel:

| A35's pre-registered criterion | `SpaceTimeCNN`, which failed it | TAKHeLiPeD |
| --- | --- | --- |
| **1. Does not lose on spatial**: single-frame score at least LFE-PPN's **66.5%** | **63.0%** -- below LFE-Peaks' 64.9% too | **82.20%**: stage 2 alone, one frame per call, no temporal input of any kind (A59's adopted recipe, three seeds, `nms-mean`). **+15.7 pp over the bar** |
| **2. Strictly benefits from ordered time**: the score with its history destroyed must be meaningfully below the score with it | **0.5 pp** -- destroying the order cost almost nothing | **4.47 pp**: emptying the memory every 0.5 s scores 79.41 against 83.88 when it is never emptied, of the 6.22 pp the memory adds over its own candidates (E1, reset sweep 0.5 / 1 / 2 / 5 / 10 / 30 s / never = 79.41 / 80.46 / 81.51 / 82.65 / 83.53 / 83.73 / 83.88). **About 9x `SpaceTimeCNN`'s penalty** |

**Why each item closes rather than completes.**

- **A33's ranking proxy is not needed**, because the project stopped screening many architectures and built one. Its own rule --
  "a proxy that cannot rank three known architectures correctly cannot rank unknown ones" -- was never satisfied by any proxy
  tried, and the informational-capacity grid's predictive power remains explicitly unvalidated (`interpreting-evaluation.md`).
  **Nothing that was built was ever ranked by a proxy**, which is the outcome A33 was written to protect.
- **A34's survey informed the design rather than feeding a funnel.** Of the families it listed, the one built is **ConvNeXt
  blocks in a U-Net** with a whole-scan aggregator -- A34's own "cheapest *is the design simply dated?* test" and "strong on small
  data" rows -- and A59 then priced the alternatives from the measured model instead of from a proxy: LFE's input features are
  worth nothing once weights are averaged, LFE's widths cost 1.4 pp, DR-SPAAM's training protocol costs 1.8 pp.
- **A35's rebuild happened as A43/A50/A59/A60**, and its two criteria are the table above.
- **A36's re-triage happened on 2026-10-09/10**: A50's remainder was triaged item by item, and §A is now ordered around a
  pre-print.

~~**One caveat the write-up must keep.**~~ **Closed 2026-10-11 (item 28): the criterion holds on the adopted recipe.** The sweep
was re-run there: emptying the memory every 0.5 s scores 83.16 against 86.35 when it is never emptied, so **+3.19 pp of the
memory's 4.13 over its candidates depends on history older than half a second** -- about **six times** `SpaceTimeCNN`'s 0.5 pp
shuffle penalty, where B0's backbone gave 4.47 pp and nine times. The margin narrowed with the better stage 2, as everything
about the memory has, and the claim now rests on the model being proposed rather than on a superseded one.

### A33. A defensible ranking test — and it must be validated before it is trusted (the plan, kept for its rule)

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

### A52. Finish the migration to the name TAKHeLiPeD -- **DONE 2026-10-11**

> **Done, and wider than planned.** The owner extended it mid-rename from the `three_horizon` identifiers to the whole
> `follow_the_drow` package, which was the project's last name from the superseded line of work. What moved: the Python package
> (`library/takheliped/`, 52 files, 157 references, the distribution in `pyproject.toml`, the editable install), the C++ core
> (namespace, header tree, CMake target and config file), the ROS package (`deploy/takheliped/`, its inner script package, the
> launch file, the image name, the compose mounts and roslaunch path, the Makefile, `.dockerignore`, `.gitignore`,
> `requirements.txt`, the GitHub workflow), every `utils/` script and test file, and the prose.
>
> **The C++ half was compiled rather than assumed.** MSVC 2022 builds and links the extension clean against the renamed
> namespace, and the rebuilt `cpp_binding` loads with `AlgorithmicDetector` constructing. `pybind11` was missing from the venv
> and is now installed, since without it the extension cannot be rebuilt at all.
>
> **The ROS half is unverified by any check here**, because catkin builds in Docker: the nodes are parse-checked and the
> references swept, and the next image build is the real test. Nothing reaches the robot until someone rebuilds (`AGENTS.md`
> rule 5), so the consequence is that the next build uses the new names and the `roslaunch` path changed to
> `src/takheliped/config/takheliped.launch`.
>
> **Step 4 was deliberately skipped and should stay skipped.** `checkpoints_three_horizon/` keeps its name because every
> checkpoint stores the absolute path of the stage 2 it was trained on, and `coarse_e2e.py` resolves it -- renaming the directory
> would dangle ~30 runs' provenance in exchange for a name nobody outside this repository sees. A52 warned that stopping short
> leaves a third spelling in play; the answer is the note now in `README.md` and `AGENTS.md` rather than broken data.
>
> **One detail worth keeping:** the extension's init symbol is `PyInit_cpp_binding` and `algorithmic_detector.py` imports it
> relatively, so the prebuilt `.pyd` kept working under the new package name even before the rebuild. The package prefix in
> `Pybind11Extension` only decides where the built file is placed.

**The original plan, for the record:**

**What.** The owner named the detector on 2026-09-23: **T**(emporal) **A**(daptive) **K**(nee-)**He**(ight) **Li**(dar) **Pe**(rson) **D**(etector), *TA-KHé-Li-PeD*.
The **documentation** half is done (`CHANGELOG.md` 2026-09-23): `README.md`, `AGENTS.md`, `memory/performance-log.md`, `memory/detector-architectures.md`, `memory/keywords.md`, `memory/commands.md`, `docs/PROPOSAL.md`, `docs/PAPER.md` and `docs/RESEARCH.md`'s header (and `docs/SHOWCASE.md`, deleted 2026-10-10).
The **code** half is not, deliberately: every identifier still says `three_horizon`, so no checkpoint, cache or log path moved.

Left to rename, in the order that minimises breakage:

1. `library/takheliped/detectors/takheliped.py` -> `takheliped.py`, and its imports.
2. `utils/train_takheliped.py`, `utils/takheliped_oracle.py`, and `utils/.takheliped_cache/`.
3. `tests/test_takheliped.py`, `tests/test_takheliped_oracle.py`, `tests/test_train_takheliped.py`.
4. `checkpoints_three_horizon/` -> `checkpoints_takheliped/` (gitignored, so this is a local move plus every documented command that names it).
5. The prose in `CHANGELOG.md` and this file, which still say *three-horizon* throughout, and the `*keywords:*` lines that index them.

**How.** One commit per numbered step, tests green between each. Steps 1-3 are mechanical renames with no behaviour change; step 4 breaks every `--stage2`/`--memory`/`--out-dir` path in flight, so it waits for a moment with no run going. Step 5 is prose only and is the one to do last, because the old word is what makes the history greppable until everything else has moved.

**Why.** One name, one spelling. Two spellings for one detector is exactly the kind of drift that makes a grep for past evidence miss half of it — and the reason to do it as staged renames rather than a single sweep is that a checkpoint directory rename during a training run costs a run.

**Why not now.** Steps 1-3 are safe at any time; step 4 is not, and doing 1-3 without 4 leaves a third spelling in play. Worth batching behind the A51 slot experiments.

### A55. Revert the GPU driver to Adrenalin 26.2.2, then prove it on one long run — **owner's hands, started 2026-09-26**

**What.** Replace Adrenalin 26.8.1 (`32.0.31041.1004`) with **26.2.2** (`32.0.23027.2005`), the exact driver AMD requires for the ROCm 7.2.1 PyTorch wheels in `requirements.txt`. The A51 queue was stopped for it at 2026-09-26 04:51 (`checkpoints_three_horizon/progress.txt`); O2's `step3a_resume.pth` (chunk stage 1, epoch 7, 3.86 h) is intact.

**How.** The owner does the install: download 26.2.2 from AMD first, clean-uninstall the current driver (AMD Cleanup Utility), install 26.2.2 with networking off, and stop Windows Update from replacing it (Device Installation Settings -> No). Then `python -c "import torch; print(torch.cuda.is_available())"` in `.venv`, and relaunch the queue, which resumes O2 by itself. **Pass:** no `0x7E` in the System log across the next A51 runs (26.8.1 managed about one per week under ROCm load). **Fail:** a `0x7E` on 26.2.2 means the driver branch was not the cause; then test the hardware (memory test, EXPO/XMP off, PCIe slot forced to Gen4) before anything else.

**Owner's decision 2026-09-26: no revert until the cause is known.** Kernel dumps are on (since the 05:58 boot) and the queue was relaunched at 06:03 on 26.8.1; the next bugcheck is the evidence. Leading suspect is a display **wake** under load (both crashes came as the owner logged in; `memory/gotchas.md`). A wake test (automated display off/on cycles under a ROCm load, ~300 cycles overnight) runs only in a gap between training runs, with the owner's go-ahead, since a hit kills the run.

**Earlier the same day: test done, revert back on.** The dumps put both recent bugchecks in the driver's power-state handler (`memory/gotchas.md`), but a 10-minute display-off test under ROCm load showed no GPU power-down and no crash, so keeping the display on or disabling BACO (`PP_GPUPowerDownEnabled`) targets nothing observed; leave both alone so the revert's effect stays readable. **Before the revert, switch crash dumps to kernel dumps** (elevated: `CrashDumpEnabled = 2` plus a `DedicatedDumpFile` on `C:`, since the page file is on `S:`; reboot), so that a recurrence shows which power request (sleep, shutdown, or the driver's own) triggers it.

**Why.** Five bugchecks since May, all on the `31xxx` branch, three at the same code offset on 26.8.1; none on the `23xxx` branch 26.2.2 belongs to (`memory/gotchas.md`). Going back one release (26.7.x) would stay on the branch that crashed. O2 resumes across the driver change; note it in its CHANGELOG entry, since kernels can differ numerically.

### A54. Get a formal citation for `AlgorithmicDetector` from prof. O. Aycard — **owner's to answer**

**What.** The rule-based detector this repo ports is **prof. O. Aycard's** algorithm and implementation; the only attribution that exists is the header comment in `library/cpp_core/sources/detector.cpp` ("Person detector using 2 lidar data / Written by O. Aycard"). The credit is now stated in `README.md`, `AGENTS.md`, `docs/RESEARCH.md` §2.1 and §2.2, `memory/performance-log.md` (footnote 0), `memory/detector-architectures.md`, `memory/deployment.md`, `detector.hpp` and the Python wrapper's docstring.

**What is missing:** a bibliographic reference — a paper, a technical report, a thesis, or an explicit "unpublished, cite as personal communication". Nothing of the sort is in the repository, and inventing one would be worse than having none.

**How.** Ask him. Then put the reference in `docs/RESEARCH.md` §2.1 and §9, and replace the "still to be obtained" notes that point here.

**Leads found 2026-09-28, none of them the reference.**
His laser tracking work: Vu & Aycard, *Laser-based detection and tracking moving objects using data-driven Markov chain Monte Carlo*, ICRA 2009 (single-layer laser, vehicles); Gómez, Aycard & Baber, *Efficient Detection and Tracking of Human Using 3D LiDAR Sensor*, Sensors 23(10) 4720, 2023.
Neither describes the two-2D-LiDAR leg-and-chest detector, so they are context for the question, not an answer to it.

**Why.** Every other detector in the comparison is cited to its paper; this one is cited to a source-file comment. That asymmetry is fine internally and not fine in anything published.

**Blocked on the owner**, not on effort.

### A53. Rewrite `docs/RESEARCH.md` around TAKHeLiPeD

**What.** `RESEARCH.md` is still titled and framed as *Full-Scan, Non-Recursive Convolutional Architectures*, which names the superseded line of work. A header note added 2026-09-23 says so and points at `PROPOSAL.md`/`PAPER.md`, but the body below it has not moved.

**How.** Keep §1 (problem and metric), §3 (baselines and the fidelity audit), §4 (preprocessing) and §7 (related work), which are all still authoritative. Replace §2's goal and constraints with TAKHeLiPeD's, fold §5 and §6 into one provenance section, and fill §8 from `memory/performance-log.md` rather than restating numbers that will go stale.

**Why.** It is the document a reader outside this repo is pointed at, and it currently argues for a design the project has moved past. It is also the only place where the fidelity audit against each published paper lives, so it cannot simply be retired.

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
Before any result: extract into `library/takheliped/include/JRDB-data/`, then measure the real format. `JRDB_Dataset` assumes a DROW-compatible 541-beam, 270° scan (`memory/data-model.md`), never checked against real files, while the official DR-SPAAM repository evaluates on merged front and rear scans of 1091 points over 360°. Its README reports AP@0.5 of 82.9% (DROW3) and 84.9% (DR-SPAAM) on JRDB, above Li2Former's 81.4%, so the published JRDB numbers likely use different protocols (scan merging, split, AP threshold): check each paper's protocol before calling anything comparable. Measure the scan rate too (`memory/dataset-properties.md` has none for JRDB), since the three horizons are set in seconds.
Previously blocked on registration requiring a university email; no public mirror of the 2D-LiDAR portion was found or sought further.

---

## §D LATER

### ~~Do the object slots already cover the coarse horizon?~~ Moved to §A as A60 (2026-10-05)

The owner's 2026-09-17 hypothesis, now planned and with a decision rule: see A60.

### ~~The adopted recipe on a second FROG partition (`balanced`)~~ -- moved to A63 step 0 on 2026-10-10

**The owner's call:** `balanced` is functionally another dataset, because every model needs retraining on it, so it belongs with
A63's other-dataset work rather than in the FROG queue. It is the *first* step there, being the cheapest -- same loader, same
sensor, same angular grid, so it needs no part of A62. The plan below is kept for its partition rules, which A63 inherits.

### Why it was here (kept for the partition rules)

**What.** Train and score the adopted recipe on FROG `balanced`, which holds two test recordings against `official`'s one and a
121,329-frame val split against a 2,000-frame populated-only sample, and which keeps `frog_16-41` whole so the official-test
subset falls out of the same evaluation pass.

**Why it moved.** It was scheduled as the **artefact test for the temporal ablations** -- on `official` they came out backwards,
removing the fine convolution gaining +1.41 pp and the coarse one +0.39, with validation ranking the arms in nearly the opposite
order. Both convolutions have since been removed from the architecture (2026-09-17 and A60), so the arms whose artefact was in
question no longer exist. What is left is a smaller and different claim -- does the adopted recipe hold on a second partition --
worth doing before publication and not before the headline numbers are frozen.

**Never mix partitions:** `balanced`'s val is 11-36 + 10-31 and **11-36 is half of `official`'s train**, so validating there
while training on `official` scores the model on its own training data. State the deviation when quoting: published baselines
trained on 11-36 + 12-43 where `balanced` trains on 12-43 + 14-57.

### C++ modernization: compile check, Meson migration, `clang-tidy`, `clang-format`

**What.** Pulled out of the active alignment plan (originally `AL6`) on 2026-09-08: not the current priority since there's no physical robot to deploy against right now, but real work this project will come back to.
Bundles: (1) confirm the C++ side still actually compiles — `.github/workflows/build-lidar-vision-image.yml`'s `build-cpp-static-library` job already runs `sudo make build-lib` on `ubuntu-latest`, but its current pass/fail status hasn't been checked this session, so verify it before assuming new work is needed; (2) migrate the build system to Meson; (3) `clang-tidy` (config plus `compile_commands.json` via `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, or Meson's native `compile_commands.json` output if (2) lands first); (4) `clang-format`, with a `.clang-format` config.

**How.** Scope properly when picked up rather than now — in particular, (2) needs a real decision on scope: `library/cpp_core/` is a plain CMake target and a reasonable Meson migration candidate, but `deploy/takheliped/` is a **catkin** ROS package, and catkin is built on CMake specifically — ROS Noetic has no supported Meson build path, so a full migration likely means "migrate the standalone library, leave the ROS package on CMake," not a uniform switch, unless someone finds a workable catkin+Meson bridge worth the risk.

**Why.** Grouped as one item rather than broken into an AL-style sequence because it's explicitly deferred, not because it's small — real scoping (especially the Meson/catkin question) happens when this is picked back up, per the "don't invent a rule/plan for something not yet earned" principle this documentation layout otherwise follows.

### Small tooling gaps found 2026-09-17

- ~~**ruff is not installed in `.venv`**~~ **installed 2026-10-02 (A56)**; `memory/commands.md` has the interim invocation. A project-level ruff config for `library/` and `utils/` is still to be written.
- **`duration_curve.py --help` prints a `SystemExit: 0` traceback and exits 1.** Its `__main__` wrapper catches `BaseException`. Catch `Exception` or let `SystemExit` through, as the ROCm hang workaround allows.

### Per-track adaptive tracker gating

**What.** `SimpleTracker`'s one fixed `match_radius` cannot work at both fine and wide invocation strides (`memory/rejected-ideas.md`).

**How.** Scale the gate by a track's own estimated speed, or use a proper covariance/Mahalanobis gate.

**Checked 2026-10-02 (A56), and the covariance gate is not available off the shelf.** `SortTracker`'s and `ByteTracker`'s point Kalman filter carries SORT's noise scaling, which was set for pixel boxes: measurement noise `R = I` is 1 m^2 here, so the innovation covariance is at least 1 m^2 and a 99% chi-squared gate is at least 3 m wide -- useless in FROG's crowds. A Mahalanobis gate first needs the filter's `R` and `Q` set in metres for a 2D LiDAR point (centimetre-scale measurement noise, walking-pace process noise), and then the SORT rows would no longer be SORT. That is its own experiment, not a flag.

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
