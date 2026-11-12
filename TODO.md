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

### The open register

**This is the whole of the open work.** Every row below is something nobody has done yet. Everything further
down §A is either an open row's own What / How / Why block, or a record kept because something still cites its
mechanism. Twenty-nine sections were deleted on 2026-11-05 under rule 4b, together with the three running-order
tables and six status quotations that had outlived them, after each one's residue became a row here and the two findings only this file held were moved out
(`memory/performance-log.md` footnote 3, `memory/interpreting-evaluation.md`).

**Two numbering series, and they are not interchangeable.** `A<n>` numbers the items opened before 2026-10-10;
a bare `item <n>` numbers the FROG pre-print list of 2026-10-10 and everything added since. **`A37` and `item
37` are different items** and both are cited in `CHANGELOG.md` -- A37 is the LFE-PPN re-run list, item 37 the
dt-aware mechanism's robustness screen. Numbers are never reused and never renumbered, so the collision is
stated rather than fixed: a citation meaning the older series writes the `A`.

**Items 72 and 75 were deleted on 2026-11-05** once `CHANGELOG.md` carried both entries, per rule 4b -- the
DROW headline cell (46.01%, the memory worth +22.56 pp against FROG's +3.49) and the capacity arms that ruled
capacity out. The numbers are the record's, not the queue's; grep `CHANGELOG.md` for either number.

**Item 85's row and item 73's leftover rationale were deleted on 2026-11-06** once `CHANGELOG.md` carried both.
The figure outlived the rows: FROG at 0.5 deg is **1.02 pp** of whole-detector AP behind 0.25 deg (85.28% against
86.30%), not the 0.88 pp the mis-ranked arm reported, and that 0.88 was **not** the upper bound it was called --
fixing stage 2 raised the candidate column 0.44 pp while the whole detector fell 0.14. The memory's lift at half
the beams is **+3.10 pp** against +3.51 at full, so the "decimation does not touch the memory" reading is
withdrawn. Mechanism and caveats in `memory/performance-log.md`; the script-level trap in `memory/gotchas.md`.

**A row whose condition has fired is acted on, not kept.** A37's -- re-run `error_breakdown.py` *only if* any of
its numbers is quoted somewhere -- fired on 2026-11-05 the first time anyone checked: nothing in `docs/` or
`memory/` prose quotes it, only `memory/automation-scripts.md`'s script listing. So the row was deleted instead
of run, which is the point of writing the condition down rather than the task.

**How to read the columns.** *Gate* is what has to land first, `--` for nothing. *Excludes* is what the row is
**not** -- the column exists because the duplicates this register replaced were all of the form "two items that
sound like one". *Condition* is the rule that decides the outcome or the trigger that starts it, written down
before the run so a result cannot be read generously afterwards.

#### 1. The dataset chain -- A63, and the live front

| id | what | gate | excludes | condition | cost |
| --- | --- | --- | --- | --- | --- |
| **79** | give every `published` cell of the accuracy table a citation that can be checked | -- | not a re-measurement -- a sourcing pass over figures already quoted | a cell names paper, table, row, split, radius and the configuration it was published at, or it is deleted | ~half a day, no GPU |
| **91** | **the stopping rule is the variance source on DROW** -- price it, then fix it | **open, 2026-11-07**, straight out of item 90 | the three arms took their best val AP at epochs 3.74, 7.74 and 7.99 under `--patience 8` at 0.25-epoch evaluation granularity. Two cheap moves to compare against the 3.19 pp spread: evaluate every 0.5 epochs rather than 0.25 (half as many chances to fire into a dip), and raise `--min-epochs` so no DROW run can end before epoch 8. Re-run the headline at both seeds under each | **this is the half of the noise the project controls**, unlike the seed. **Constraint from 95(d), 2026-11-08**: removing the stopping rule costs 10.9 pp, because DROW's stage 2 overfits after ~8 epochs, so both arms here must keep early stopping and only change how it reads val. DROW val is 3,919 annotated frames against FROG's order-of-magnitude more, which is why the same rule is quiet there. If it halves the spread, every DROW lever below 3.5 pp becomes measurable at two seeds instead of unmeasurable at any number | ~1 h for four arms |
| **78** | beam-roll augmentation | the augmentation itself, which does not exist | **absorbs §D's DROW idea 1**, which was the same proposal | a free control on item 75 -- both attack overfitting by unrelated means, so agreement is evidence and disagreement is a finding | ~15 min per arm + code |
| **A63.3** | JRDB: measure the loader's geometry, rate and annotation density, then repeat steps 1-2 -- **and shard the candidate cache per recording first**, since DROW's 341,147-frame split already needs 1.7 GB twice over at `pack` (item 72) and JRDB's is larger | the data on disk -- **met 2026-11-11**: the 2019 train split, unpacked by `utils/unpack_jrdb.py` into `library/takheliped/include/JRDB-data/train_dataset/`. It is the only split with 2D laser at all -- the download page has no test rosbags (`memory/data-model.md`) -- and the 2D-lidar train split is what the loader is checked against | **absorbs §C's JRDB block**, deleted 2026-11-11 once its prerequisite was met | no JRDB number is quoted before the loader is checked against real files: ours assumes 541 beams / 270 deg where the official DR-SPAAM repo merges two lasers into 1091 / 360 deg, and its README's 82.9 / 84.9% are on that merge, scored on a 10-sequence hold-out of the train split. Li2Former's 81.4% has an unchecked protocol. The scan rate matters because the three horizons are set in seconds | unknown |
| **A63.4** | the slots scored as a tracker -- HOTA, MOTA, IDF1, OSPA(2) -- on JRDB's 3D tracks projected to the ground plane (A56's E17) | A63.3 | FROG cannot host it: it has no identities, which is why these metrics were never computed | -- | -- |
| **A63.5** | self-supervised memory, or DROW label densification from slot tracks (A56's E18, and item 21's identity loss) | 72 | densifying DROW's every-5th annotation is itself a research item, not a preprocessing step | FROG has no identities and A56 found pseudo-identities unreliable beyond 10 s, so an identity loss only becomes testable where real tracks exist | -- |

#### 1b. DROW's stage 2: our pipeline, our recipe, or our architecture -- **approved by the owner 2026-11-08**

**Why this group exists, in one comparison.** At **one frame** the same two families swap places by about 40 pp
between the datasets. On FROG our stage 2 beats DROW3 at `T=1`, **80.89% against 73.9%** (seed 0 through step 2's own
scoring pass; the 82.80% first quoted here is the chain's three-seed candidate column, a different measurement). On DROW it trails it,
**27.86%** (the headline's two seeds, 29.45 / 26.26) against **59.4%** (Beyer et al.'s own `T=1`) and **66.6%**
(DR-SPAAM's retrained `T=1` baseline) -- both from DR-SPAAM's Table I, DROW test, AP at 0.5 m (arXiv 2004.14079).
So the whole detector's 20.5 pp understates the deficit, and the "temporal input" suspect the four-lever entry left
open cannot be it: neither side of this comparison has any. **And nothing trained in this repository on DROW has
ever been checked against a known answer.** Every DROW baseline reproduced here is a published weight file (DROW3
-5.2 pp, DR-SPAAM -1.2 pp); the two networks trained here, ours and Li2Former's own code (48.1% against 76.4%),
both land ~30 pp short. Rows 92-94 decide whether the pipeline is sound before any lever is read; 95-98 are the
levers and, if the gap survives them, the explanation. **92-94 all reported on 2026-11-08 and cleared the pipeline**:
the scoring path and the labels (93, 94), and training itself -- DROW3 at `T=1` retrained here on DR-SPAAM's recipe
scores 63.20% against their 66.6% (92). The 35 pp between that cutout and our stage 2 on identical frames is ours, and
96 says it is transfer from train to test rather than anything stage 2 fails to fit (`CHANGELOG.md`).

**Agreed with it (owner, 2026-11-08):** no further DROW stage-3 night (76, 77, another head arm) until 92-96 report -- a
memory measured on a stage 2 ~35 pp under a single-frame cutout measures how much it compensates, which is also why
the +22.56 pp lift is not yet a DROW finding. **The direction of that worry is now measured, and it is the other way:** across the three DROW
chains the lift *falls* with the candidate footing (+22.56 / +17.97 / +17.69 off candidates 23.45 / 20.73 / 19.82), so a
weak stage 2 shrinks the memory's contribution rather than inflating it -- which makes the 20.5 pp gap a stage-2
problem end to end and these rows the whole of it. And A63.3's loader work can start beside this, with no JRDB training
before 92 and 93 report, since a pipeline defect would follow us there. **That trigger fired on 2026-11-08** (92-96 all
reported); keeping 76 and 77 parked until 97 reports is the recommendation, since the gap is stage 2's transfer and a
stage-3 night on today's stage 2 measures compensation again -- **the owner's call**.

**The order from here (owner, 2026-11-12): 108, 109 and 110 agreed; 111-113, 103 and 76 / 77 proposed and waiting on the owner; 105 written once 108 decides the stage 2.** Item 107, the investigation that found the per-beam norm and the field, is closed into `CHANGELOG.md` (2026-11-10 / 11); its proposals are these rows. 102 names the
property; if it shows the problem is locality, 104 changes the architecture to supply it -- **not necessarily with
convolutions**, since the U-Net already has plenty, but with whatever primitive the identified property calls for.
103 (LFE on DROW, its own venv approved) is the fallback for locating the cause more precisely. Row 100 was measured and
adopted (DROW's own radii, `CHANGELOG.md`). **Everything else is parked until this line reports**: 76 and 77, and 97's remaining cells -- the cutout
trained on FROG and the receptive-field dose-response, which becomes one of 104's candidates rather than a row of its own.

**How this line ends -- the two outcomes and what each obliges (owner, 2026-11-09).**

- **A. The cause is found exactly *and* 104 finds an architectural change that closes the gap.** Then one decision,
  the owner's, and both branches have a cost written down now:
  - **A1, the change becomes the proposed model's new default.** Every FROG figure it touches is recomputed on it --
    the three-seed stage 2 (82.80% candidates, 80.84% step-2 pass), the frozen whole detector (86.29%), the CPU latency
    sweep, the slot-design table where the change reaches stage 3's inputs -- and `docs/ARCHITECTURE.md` section 5,
    `memory/performance-log.md` and `docs/PAPER.md` are rewritten to the new numbers. The frozen architecture is the
    citable artifact, so this is a pre-print decision as much as a DROW one.
  - **A2, the change is a drop-in, off by default.** Then the condition that enables it must be stated precisely, as a
    *measured property* of the data or the deployment -- the one 102 identified, with its threshold -- not as "on
    DROW". It needs evidence both ways: on where the property holds (DROW), off where it does not (FROG, where it must
    be shown unnecessary or harmful), and a way for a user to check the property before choosing.
- **B. No exact cause, or no change that closes the gap.** 103 (LFE on DROW) runs to locate the cause more precisely,
  and in any case the honest DROW result is prepared for the paper (row 105): why the cause could not be pinned, what
  was tried and measured, what is suspected and why it stays unclear; and why no fix worked, what was tried and
  measured, and what further reasons may lie behind it.

| id | what | gate | excludes | condition | cost |
| --- | --- | --- | --- | --- | --- |
| **97** | **the explanation -- the gap survived 95 (2026-11-08: every lever between 27.9% and 30.2% against the cutout's 63.20%)**: the four-cell design -- {full-scan stage 2, cutout} x {FROG, DROW}, all four trained here under one protocol -- plus a receptive-field dose-response on both datasets. **96 has already narrowed it**: our stage 2 fits DROW's train split (63.6% at the val-best epoch, 82.6% at epoch 40) and transfers to test at 29.5% / 18.1%, where FROG keeps test within 13 pp of train (93.7% / 80.9%) -- so the cell to measure first was the cutout's own train-test gap on DROW, **measured 2026-11-08: 66.50% train against 63.20% test (1.05), where our stage 2 is 63.6% / 29.5% (2.16)**. The cutout learns what carries from DROW's train recordings to its test ones and ours does not; what 97 still owes is *which* part of the full-scan network does the memorising, which is the dose-response. **101 ruled capacity out on 2026-11-08** -- train/test stays at 1.94-2.47 down to 16,667 parameters while test AP falls -- so the receptive field is the first thing to vary. **The DROW pair exists since 2026-11-08**: cutout 63.20% (item 92) against our stage 2's 27.86%, both trained here on identical frames. Still owed: the cutout trained here on FROG (its published `T=1` cell is 73.9%; at FROG's 120k frames the cutout cache is ~19 GB, so it needs `--no-frame-cache` or a frame stride), then the dose-response. **Parked 2026-11-09 by the owner** behind 102 and 104 | 102, 104 | not a tuning sweep: "unsuitable for DROW" is an interaction claim, and an interaction needs all four cells | context that helps on FROG and hurts on DROW makes the context scale a swappable, per-dataset part with a measured reason, the way `--block` and `--head` are; no interaction means the gap is not architectural. Published hint: DR-SPAAM narrowed DROW's window from 1.66 m, sized for walking aids, to 1.0 m "to tightly fit the footprint of a person", and its `T=1` baseline reads 66.6% against the original's 59.4% | ~1 day + code |
| **102** | **name the DROW property that makes locality matter** (owner, 2026-11-08), so locality can become an option keyed to a measured condition rather than to a dataset's name. **Explained 2026-11-11 by row 107** -- the whole-scan `GroupNorm` and the field's reach; earlier: **reading 2026-11-09, the trigger revised by row 106** (`CHANGELOG.md`, the four 2026-11-09 item-102 entries): the three candidates set first -- scene novelty, look-alikes, repeated backgrounds -- all failed their rules; what is left is a mechanism and its trigger. **Mechanism**: inside a scan our stage 2 finds ~80% of what the cutout finds, but its score level moves between scans, so pooled AP falls to 42-47% of the cutout's (pooled over within-scan hits 0.47-0.50 on DROW test, 1.03 on FROG test). **Trigger**: scan geometry outside the training distribution -- DROW test is enclosed (median return 1.74 m, 98% of scans under DROW train's 25th percentile of 3.25 m), DROW train open (4.33 m), FROG test on FROG train (3.72 / 3.61 m); on DROW val, which spans both, ours' cross-scan comparability is 0.81-0.88 on typical geometry and 0.54-0.71 on enclosed (cutout 0.94-0.97), with within-scan hits flat at 0.80 / 0.81 of the cutout's, each rule written before its run and met in both scopes. **Correlational**, on one split, val's typical bin mostly one recording and the checkpoints val-selected. **Row 106's repartition, 2026-11-09: by the rule written first, geometry is not the trigger** -- balancing it lowers our train / test ratio from 2.63 to 1.76, not to 1.4 -- so the row stays open: the mechanism stands, its trigger is the official test recordings as a set, of which enclosed geometry is one difference. **Phase 2, the cutout on the same partition: 53.75% test, train / test 1.27** -- ours there is 78% of the cutout's test AP (11.9 pp behind) against 46% on official DROW (34.2 pp), and still transfers worse (1.76). The two item-104 arms leave the gap where it was, and the local level shows the cross-scan failure is not a per-scan additive offset; what is left to explain is those 11.9 and 34.2 pp | -- | not a lever sweep: it names a property a deployment can check before choosing a model -- the share of its scans whose median return range falls under the training set's 10th percentile, computable from unlabelled scans | a property that separates DROW from FROG and tracks the failure is what locality gets keyed to -- met by geometry, pending 104's intervention | done |
| **108** | **the per-beam norm at kernel 5, end to end on FROG** (owner, 2026-11-12, agreed) -- item 107's change (`--norm position --kernel-size 5`: DROW stage 2 29.03 -> 41.79%, FROG stage 2 81.30 -> 79.80%) through the frozen detector's whole pipeline: three stage-2 seeds on the adopted recipe (seeds 0-1 exist as `frog_field_k5_s0` / `_s1`, trained with exactly it; seed 2 new), each averaged over its val-chosen window, then stage 3 at the frozen recipe (`--recurrence mingru --chunk-s 10 --no-exchange --no-learned-match --no-coast`) on each, so the only difference from `a71_freeze_s0-2` (candidates 82.80%, whole detector **86.29%**) is stage 2's norm and kernel; then the matched CPU sweep end to end, interleaved with the frozen detector, on an idle host | none | not the adoption itself: every default stays as it is until the owner decides A1 or A2 on these numbers | the owner's A1 / A2 call, made on the three-seed whole-detector FROG AP against 86.29% (its seed spread 0.22 pp), with the CPU ms/frame beside it and DROW's +12.8 pp at stage 2 | ~25 min of stage 2, three averages (~1 h), three stage-3 arms at ~7.3 h -- about a GPU-day; candidate caches on disk, kept small while JRDB downloads |
| **109** | **the multi-class head on the new stage 2** (owner, 2026-11-12, agreed) -- DROW's worst part after item 107 is scans where a wheelchair or walker stands near nobody (0.39-0.55 of the cutout's AP): confident detections on look-alikes when few people outrank them. The binary head trains wheelchairs and walkers only as background; `--head multiclass --class-norm softmax` (DROW3's own) teaches them as classes. Two seeds on official DROW at the per-beam norm, kernel 5, DROW's default radii, read part by part (reading 17) | none | not the ignore region, which leaves look-alikes unlabelled rather than named | **prediction, written first:** if the look-alike false positives come from never being taught wheelchairs and walkers as their own classes, that part rises toward the cutout's while the other parts hold; no rise says the binary head is not where they come from | ~25 min + a reading |
| **110** | **the new stage 2 on `geometry-balanced`** (owner, 2026-11-12, agreed: "how much of the difference is the geometrical asymmetry between train and test, and how much is genuine architecture property") -- two seeds of the per-beam norm at kernel 5 on `geometry-balanced`, read against DROW3's cutout on the same partition (53.75%, train / test 1.27) and part by part, beside the same comparison on official DROW (ours 41.79% over the cutout's 63.20%, 0.66) | none | not a new benchmark: the partition is read only against the cutout trained on it | **prediction, written first:** if the remaining official gap is mostly the train / test geometric asymmetry, ours over the cutout on `geometry-balanced` rises close to 1 (0.9 or more) and its train / test approaches the cutout's 1.27; whatever stays below is the architecture's own share. The old stage 2 there read 0.78 | ~25 min + a reading |
| **111** | **a receptive field sized in metres, not beams** (proposed 2026-11-11, not agreed) -- past 3 m our stage 2 still reaches 0.39-0.54 of the cutout's AP, and even the smallest beam field tried (58 beams) spans 3.6 m of arc at 7 m against the cutout's 1 m; on FROG's 0.25-deg grid the same kernels span half the angle, which is why kernel 3 cost FROG another 0.85 pp. A field that holds a fixed metric width at every range -- the metric gather without its scan-wide norm, at more than the first layer, or range-dilated kernels | the owner's approval; after 108 / 110 | not another beam-count field | far-range DROW AP up and far-range train AP toward the cutout's, with FROG no worse than kernel 5's | code ~half a day + two seeds on each dataset |
| **112** | **a fused `layer_norm` for the per-beam norm** (proposed 2026-11-11, not agreed) -- `ChannelNorm` computes its mean and variance by hand and costs stage 2 0.4-0.7 ms on the CPU; `torch.nn.functional.layer_norm` over the channel axis does the same in one kernel | the owner's approval; best before 108's CPU sweep | not a change of arithmetic: outputs must agree with `ChannelNorm` to float tolerance, tested | stage 2's CPU ms back toward the scan-wide norm's, measured in the matched sweep | ~1 h |
| **113** | **geometry augmentation** (proposed 2026-11-09, refocused 2026-11-11, not agreed) -- training scans with far returns replaced by near synthetic walls, the beams on people left as recorded, so open and enclosed versions of a scene are both seen; a training-side answer to whatever share of the gap 110 attributes to the geometric asymmetry | the owner's approval; after 110 | rescaling the ranges, which would rescale the people too | official DROW AP up with FROG unchanged; worth its cost only if 110 puts a real share of the gap on the asymmetry | code + two seeds |
| **103** | **LFE trained on DROW** -- **proposed again 2026-11-11 as an optional check, not agreed: item 107 found the causes in our own architecture, so this is now a second whole-scan model's reading rather than the fallback; after JRDB finishes downloading, since the TensorFlow install is network-heavy; its own venv approved 2026-11-09** -- the owner's strongest evidence (2026-11-08): a *published* full-scan model failing to transfer on DROW would show DROW suits full-scan models poorly independently of ours. Train LFE on DROW with its own released trainer (`robotics-upo/2DLaserPeopleBenchmark`, Keras), score it on DROW train and test, and score its released FROG weights on FROG's train split, so LFE has a train/test ratio on both datasets. LFE is the right test: 53,258 parameters, whole-scan context through a global max broadcast into every block (`docs/RESEARCH.md`'s graph) | a Keras environment beside `.venv` (unmeasured: TensorFlow has no ROCm path on Windows, so CPU, plausible at 53k parameters), or A32c's PyTorch port | not 101's LFE-shaped arm, which is LFE's input and widths on our topology. Their trainer runs as released, defects reported rather than fixed (`SegDataGenerator` never shuffles, `memory/static-detector-diagnosis.md`) | LFE's DROW ratio near ours (2.16) and far from the cutout's (1.05) -> full-scan models transfer poorly on DROW whoever designed them; near the cutout's -> the problem is our architecture's own, and a global-max context is not what memorises | ~half a day for the environment and the export, plus training (unmeasured) |
| **104** | **locality as an optional stage-2 extension** (owner, 2026-11-08; refocused 2026-11-09), chosen by 102's measured property. **Row 107 found the change this row was looking for (2026-11-11): ConvNeXt's per-beam norm at kernel 5, DROW 29.0 -> 41.8%, FROG -1.5 pp -- the A1 / A2 decision is now about it.** **Run now, phase 2 overnight after (owner, 2026-11-09). Plan confirmed the same day -- both arms, official DROW first, and no adoption bar: the results are read, nothing is adopted automatically:** two arms, each behind a flag that leaves the network bit-identical when off, two seeds each on official DROW at the default radii -- **N, the scan-normalised range**: input channel 0, the beam's absolute range, becomes its range over the scan's median return range, the nine local channels kept metric so a person's size still reads in metres (caveat: the across offsets still carry distance as r times the beam step, so this removes the explicit size channel, not all of it); **L, the local level**: a cutout-like branch -- the 1 m depth clip and a 1 m metric window, `MetricGather` -- gives each beam its logit, and the U-Net's logit is added **centred per scan** (its mean over the scan's beams removed), so whole-scan context can re-rank beams inside a scan but cannot move a scan's level, which is the mechanism 102 measured; votes stay the U-Net's. **Reference figures, not a bar (owner):** an arm would plainly close the gap if its two-seed DROW test AP is at least 35% (headline 28.92%, +6 pp, about three times a two-seed difference's SE) **and** its pooled AP over within-scan hits is at least 0.70 (0.47-0.50 now, the cutout 0.86); then FROG at two seeds decides A1 against A2. +6 pp without the comparability moving is a gain for another reason, recorded as one; neither arm closing it sends the line to phase 2 and then 103. **Built 2026-11-09** (both flags bit-identical when off; the local branch could not reuse `MetricGather`, whose `GroupNorm(1, C)` pools over the whole scan -- caught by the locality test); `utils/queue_drow_locality_arms.py` runs them, read by `drow_locality_property.py --readings arms`. **Measured 2026-11-09: neither arm closes the gap** -- two-seed test AP N 28.59%, L 27.84% against the headline's 29.03% at the same seeds, the cutout 63.20% -- and L, whose U-Net can no longer shift a scan's mean logit, leaves pooled over within-scan hits at 0.49-0.50 (headline 0.50-0.51): the cross-scan failure is not an additive per-scan offset. Next by this plan: phase 2 overnight (row 106), then 103 or further experiments (`CHANGELOG.md`). **The owner's direction: if 102 identifies the problem directly, change the architecture with whatever primitive supplies what is missing -- not necessarily more convolutions, which the U-Net already has in plenty.** Candidates to weigh against the property found: (i) a receptive-field cap, 97's dose-response knob; (ii) a local, depth-normalised branch beside the U-Net -- the cutout's own evidence, a resampling primitive rather than a convolution; (iii) context dropout in training, so the network must be able to decide from local evidence alone; (iv) a gate that lets each beam choose how much whole-scan context to take. And, separately, the two removed **temporal** convolutions as DROW-specific extensions on top: the fine per-beam one is local in space and is DROW3's own `T=5` mechanism; the coarse bottleneck one's reopen condition, "a lower frame rate", is met by DROW's 12.70 Hz (`memory/rejected-ideas.md`) | 97, 102 | **not** restoring the coarse or fine convolution as the locality fix: both were temporal, both sat on top of the whole-scan U-Net, so neither limited what a beam's decision sees in space, and both were measured on FROG alone | a spatial option earns its place if it closes DROW's train/test ratio and its gap to the cutout without costing FROG (93.7% / 80.9%) -- and, since 102 tied the failure to geometry outside the training distribution, **if it lifts our populated AP over within-scan hits on enclosed scans** (DROW val under train's 10th percentile: 0.54-0.71, the cutout 0.94-0.97) **without lowering it on typical ones** (0.81-0.88) **or on FROG** (1.03). That is the causal test of 102's reading. Two primitives the trigger adds to the four above: (v) scale-normalising the whole-scan path by the scan's own median range, so its context reads shape rather than size; (vi) geometry augmentation in training -- far returns replaced by near synthetic walls, the beams on people left as recorded, so open and enclosed versions of a scene are both seen (rescaling the ranges would rescale the people too). For A2 the enabling property is measurable before deployment: the share of scans whose median return range falls under the training set's 10th percentile; a temporal one only on top of a stage 2 that already transfers, measured end to end as A60 was. **A success ends in outcome A1 or A2 above, never in an unlabelled flag**: what FROG costs it decides which one is even on the table | code + ~1 day |
| **105** | **the honest DROW section of the paper** (owner, 2026-11-09) -- prepared beside the experiments rather than after them so nothing tried is lost. **Since item 107 (2026-11-11) it is the fix and the remaining limitation, in six parts:** (W1) the cause -- every block's `GroupNorm(1, C)` pooled over the whole scan, a deviation from ConvNeXt the docs misdescribed, and beyond it a wide field that reaches far people from training-scene context new recordings do not share, which FROG's same-venue test does share; (W2) DROW numbers resolved by range and by people per scan, parity where it holds (crowded scans, people under 2 m); (W3) train against test by range as the memorisation evidence; (W4) DROW's measurement caveats, not defects -- val unrepresentative and leaky, 42% person-free test scans, seed spread 2-3 pp, same-seed reruns up to 2.8 pp apart; (W5) the negative results -- local level, scan-normalised range, context dropout as a window crop, more training places, the context probe; (W6) DROW kept as a fair cross-recording test, no grounds to retire it. Earlier framing, kept: Two parts: **(1) the cause** -- what was ruled out and how (the pipeline: 92-94; recipe and schedule: 95; supervision count: 98; capacity: 101; the candidate properties: 102), what is suspected (the whole-scan view memorising DROW's training scenes: train/test 2.16 against the cutout's 1.05), and why it stays unclear; **(2) the fix** -- every 104 arm with its measurement, why none closed the gap, and what further reasons may lie behind it | 102, 104 (and 103 if it runs) | not a limitations footnote: DROW is the second dataset of the claim, and a result that does not carry is reported with its evidence, the way the FROG numbers are | written into `docs/PAPER.md` from `CHANGELOG.md`'s 2026-11-08 entry onward, every number traceable to a row there | ~half a day of writing |

**The composition of DROW's splits** -- train a third stationary, val one recording, test all driving runs -- moved
to `memory/dataset-properties.md` on 2026-11-08 with the owner's leave; it is the evidence 97's mechanism starts from.

#### 2. FROG residue -- single measurements the closed items left behind

| id | what | gate | excludes | condition | cost |
| --- | --- | --- | --- | --- | --- |
| **A48.6** | DR-SPAAM windowed evaluations at `T > 1` on DROW and FROG, on corrected odometry | -- | **absorbs A47's remainder**: time the **streaming** form in the same pass, since the windowed row overstates its cost by encoding ten scans per frame where the published detector encodes one | a change beyond the third significant digit is itself a finding to write up | hours |
| **A48.x** | the displacement oracle fell **4.20 -> 2.58** under the same command after the heading-wrap fix | -- | not a number to re-run -- an unexplained one: ~0.14% of frames had wrong headings and the displacements moved far more than that allows | resolve before `docs/PAPER.md` quotes either figure | hours |
| **A40** | the merge radius default: 0.30 m or 0.45 m | -- | not a tuning question: AP says 0.45 m, the paper-reproduction rule says 0.30 m, and the operating-point view says there is no optimum, only an exchange rate | **owner's** -- it is a decision about what the detector is for | one answer |
| **A39** | a locally range-centred input channel, built on Cartesian input | -- | **absorbs §D's DROW idea 3**; A42 settled that the coordinates and the range conditioning are complementary rather than alternatives | -- | code + one run |
| **A51.E11** | train with jittered odometry | -- | deployment robustness, not accuracy | the one caveat the design still carries: room coordinates win only below ~0.02 m/s of drift | one run |
| **A32c** | port LFE to PyTorch from its own ONNX graph | -- | the architecture funnel that also wanted it closed 2026-10-10; what is left is the **retrainable static control** of the four-cell rule (`memory/interpreting-evaluation.md`), and since 2026-11-08 a route to 103's LFE-on-DROW cell | **attempted and stopped, and this is the resume point**: 29 convolutions reproduce in order, but two blocks' dataflow needs a shape-inference pass over the graph first (`docs/PAPER.md`). Until that is done every LFE comparison is against LFE as published, which is the fair comparison and stated there as a limitation. Gate before use as a control: from-scratch training on `official` must land near the published 64.9 / 66.5% | a shape-inference pass, then the weight-port check |
| **A32d.2** | our LFE-PPN is mislocalised -- -0.6 pp at `d = 0.5` but **-9.4 pp at `d = 0.3`** | A32c | do **not** re-calibrate by fitting residuals: tried, and a metric association radius caps angular residuals at ~`r/d` and fakes a metric-looking convention whatever the truth | settled by reading the target encoding out of the graph during the weight port, not by a sweep | with A32c |
| **A49.a** | move the baselines' missing-return clamp into their own preprocessing | -- | the loaders stop preprocessing; `DatasetSpec` already states the limit per dataset | one baseline re-run must come back unchanged, and it must read a train/val file too, since 61.0 m is never clamped | ~1 h |
| **A49.b** | odometry-compensated jitter | -- | today's jitter figures are upper bounds that include the robot's own motion | build it **only if** some dataset's jitter comes near the 0.5 m match radius; FROG's is 9-20 mm, so not FROG | -- |
| **A49.c** | run `scan_anomalies.py` on JRDB | A63.3 | needs the two LMS500s' specs in `SENSORS` and a reader first | -- | minutes once the data is there |

#### 3. Writing, and what is the owner's

| id | what | gate | condition |
| --- | --- | --- | --- |
| **pre-print** | the FROG pre-print itself | nothing -- the other datasets run **beside** the writing and never gate it | the one caveat it must carry: the six design arms ran with the `selective` recurrence, which minGRU replaced on 2026-10-23 on a 0.52 pp tie (`memory/slot-design-evidence.md`) |
| **A53** | rewrite `docs/RESEARCH.md` around TAKHeLiPeD | -- | the header note landed 2026-09-23; the body still argues the superseded framing |
| **A54** | a formal bibliographic reference for prof. O. Aycard's `AlgorithmicDetector` | -- | **owner's.** Cite as personal communication until it arrives; the credit itself is already stated everywhere |
| **review** | the owner's read of `docs/PROPOSAL.md`, and the pre-print's own read-through | -- | **owner's** |
| **A33's rule** | *(not a task)* a ranking test is not trusted until it is validated against a known-good ranking -- the one thing the closed funnel leaves behind | -- | applies to any future cheap screen, which is why it outlived A33-A36 |

#### 4. Parked, each with its trigger

**The two transfer regimes were parked here on 2026-11-06, by the owner.** The DROW cell is **native-only**:
item 72 already trained from scratch on DROW with no pre-existing weights, which is the measurement the owner
wants, and the two rows that would have initialised from FROG cost ~13 h plus code to answer a question about
**DROW's training-set size** rather than about this architecture.

| id | what | trigger that un-parks it |
| --- | --- | --- |
| **74** | FROG -> DROW zero-shot, plus the `selective` dt row | a reviewer asking for the transfer column, or JRDB giving a second dataset worth transferring *between*. **Its real cost is that it parks item 37's last defence** -- the dt-aware recurrence has still only ever been fed a 1%-spread constant, and a rate change is the only test of it. The stake is small: `selective` is the bespoke cell minGRU already replaced on a 0.52 pp tie |
| **76** | DROW chunk length, `--chunk-s 35` -- blocked 2026-11-06 on host memory twice, and an **11.5 h** arm rather than 7 h, because fewer streams cost 2.52x per epoch; the first length that reaches a second annotation burst | **parked 2026-11-08 by the owner, until items 92-96 report; proposed 2026-11-11 to run on the new stage 2 once 108 settles it** (not agreed): a memory measured on a stage 2 ~35 pp under a single-frame cutout measures how much it compensates. It then still needs free RAM for ~12 h or a smaller-depth variant (its block below) |
| **77** | DROW slot capacity, K = 64 and 128 -- a replay screens K under a frozen policy, the trained arm answers it | **parked 2026-11-08 by the owner, until items 92-96 report**, for the same reason; **proposed 2026-11-11 to run on the new stage 2 once 108 settles it** (not agreed) |
| **80** | the **fine-tuned** regime, initialised from item 73's weights | items 81, 84 and 76 reporting first. Every transfer number is read against the native arm, and 2026-11-05 established that the native arm is **measurably mishandled** -- 56.4% of DROW's test objects trained as background, and no gradient crossing an annotation gap. Measuring a regime comparison against a baseline known to be wrong spends 7 h to produce a number that has to be re-run |

§S holds what was stalled by the 2026-09-10 redirection, §B what waits for a retrain sweep, §C what waits for
data, §D what is understood and not worth the cycles. **Three merges happened there on 2026-11-05**, so each
idea now has one home: §D's frame-to-frame difference channels absorbed A22, §D's class-imbalance check
absorbed A23's `pos_weight` half, and §B's capacity row is item 75 above. **One removal**: §B's row asking for
`FullScanTCN` and `TemporalUNet` at real-odometry `dtime=5` is gone, since both detectors were retired
2026-10-31.

### The FROG pre-print list, set 2026-10-10

**The goal that orders this list:** after these, a pre-print on **FROG alone** can be written -- every number
in it defensible, every design choice measured, every baseline fair. The other datasets run beside the writing
rather than blocking it, which is why they are a separate chain in the register above and not a gate here.

**Nothing in this list is open.** Items 57, 48, 32, 58, 56, 31 and 64 are closed and their entries are in
`CHANGELOG.md`; the one decision that gated the list was **answered 2026-11-03 (owner): report the six design
arms as they stand**, because the re-screen it offered had already been run on 2026-10-18 -- all six on the
adopted recipe's own stage 2, one seed each against the three-seed control's 85.49-86.35% range. No GPU time
was owed and none was deferred. The caveat the write-up carries is the register's pre-print row.

**Two items from this list live on elsewhere, and neither is FROG work.** Item 21's identity loss is register
row **A63.5**, because FROG has no identities; item 26's person-free training negatives were measured on
2026-11-04 and the entry is in `CHANGELOG.md`.

**Three decisions that must not be reopened have moved to the pages that hold decisions**, so this queue stops
restating them: the ROS `leg_detector` question (reopened by the owner 2026-10-11, resolved *mention it, do not
build it*, and revisit only alongside real-ROS deployment) is `docs/RESEARCH.md` §2.2's related work; item 34's
parallel decoders are `memory/rejected-ideas.md`; and PeTra's status -- cited only, never run here, no
implementation or weights in this repository, so **not** a failed attempt -- is a footnote on
`memory/performance-log.md`'s own PeTra row.

### A63. Train and report on DROW, then JRDB -- **the second of the two postponed dataset items (owner, 2026-10-09)**

**Reporting rule for `balanced`, owner's call 2026-10-15.** The final comparison table carries **both** `official` and `balanced` results, side by side. The difference between them is not a caveat to explain away -- it is the result that justifies the `balanced` partition existing at all, and it is also why item 26 (person-free training negatives, which only `balanced` can supply) cannot stale the frozen architecture's ablation table: it adds a column rather than replacing the model.

**What.** Numbers for the proposed architecture on every dataset that is not FROG, each against that
dataset's own published baselines, plus the two tracking-metric rows that could never run on FROG. It is the
results half of A49, which split on 2026-10-09 into the dataset-agnostic port (done 2026-11-05) and this.

**Steps 1 and 2 are planned in the DROW list below** -- the owner's four decisions and items 72-80, set
2026-11-04 and extended with item 80 on 2026-11-05 (item 71 was the fifth decision and is closed).
Step 0 (`balanced`) is done. One correction to step 2's note from writing that plan: the dt-aware row is **not** free.
Every `selective` checkpoint predates the frozen architecture, so it costs one stage-3 seed before the transfer is
inference only.

**The register above holds every gate.** This table is what each step *is*; steps 1-2 are the DROW list
below, and steps 3-5 are register rows A63.3-A63.5.

| step | what | notes |
| --- | --- | --- |
| 0 | **FROG `balanced`** -- the cheapest "other dataset" and the one to do first: same sensor, same venue, same loader, **two** test recordings against `official`'s one, and a 121,329-frame val split with 19.8% empty frames against a 2,000-frame populated-only sample. Train the adopted recipe there at three seeds, score LFE-Peaks and LFE-PPN on its test split (~2 min each, never measured, A32's last open row), and extract the `official`-test subset from the same pass, since `balanced` holds `frog_16-41` whole | **Moved here 2026-10-10, owner's call:** it is functionally another dataset, because every model needs retraining on it. **Never mix partitions**: `balanced`'s val is 11-36 + 10-31 and 11-36 is half of `official`'s train, so validating there while training on `official` scores the model on its own training data. State the deviation when quoting: published baselines trained on 11-36 + 12-43, `balanced` trains on 12-43 + 14-57 **Done 2026-11-03/04**, and the deleted figures came back exactly: `docs/PAPER.md` had quoted LFE-Peaks 59.8% and LFE-PPN 63.5% on `balanced` with no measurement behind them, and they were struck from the draft rather than re-derived -- then measured at **59.8% and 63.5%**. A retracted number reproducing to the decimal is luck, not vindication, and the partition's rationale is now sourced from the FROG paper itself (`CHANGELOG.md`). |
| 1 | **DROW**: train stage 2 and stage 3 with the settings in physical units and report against DROW's own published baselines. **One seed**, per decision 4 below -- the three-seed wording here predates it | 12.70 Hz, every 5th scan annotated (6.1x fewer annotated frames and 2.9x fewer people in each, so 17.8x less supervision than FROG), three classes |
| 2 | **FROG -> DROW zero-shot**, beside it -- **and this is now also the dt-aware mechanism's last defence (item 37, 2026-10-22)**: DROW runs at a measured 12.70 Hz against FROG's 40.01 Hz, a 3.2x rate change, so a model that reads real elapsed time should transfer where one taking fixed steps cannot. It needs no new training, which is why it beats the drop-augmented retraining item 37 declined. Report `selective` against `gru`/`mingru` on this row specifically | the owner's 2026-09-15 question: one model per dataset, one transferred, or both -- each result says which |
| 3-5 | **JRDB**, then the slots scored as a tracker, then the self-supervised memory | register rows A63.3, A63.4, A63.5 |

**Rules that already apply.** **AP does not compare across datasets** (person density differs 3.8x), so each dataset's result is
set against that dataset's published baselines only; and every number states its range limit, since 10 m is not a neutral choice
on DROW.

**Cost.** DROW ~2 days plus the zero-shot and fine-tuned passes; JRDB unknown until step 3's measurements.
Step 0 is **done**. Steps 1-5 were gated on the dataset-agnostic port, **done 2026-11-05** (`CHANGELOG.md`):
the dataset is a flag, the per-dataset table prints and is stored in every checkpoint, and the DROW loader is
verified against the real files.

**Where the per-dataset parameter table went.** A49 used to hold one here, and by 2026-11-05 it disagreed with
the code in the same breath it was quoted in -- 26.2 Hz for FROG two lines above the measurement correcting it
to 40.01 Hz. There is now exactly one: `DatasetSpec` in `utils/train_takheliped.py`, which `parameter_table`
prints at the head of every run and every checkpoint stores. The measured values behind it are
`memory/dataset-properties.md`. Do not restate it in prose anywhere.

### The DROW list, set 2026-11-04 -- A63 steps 1-2, with four decisions taken

**What the owner asked for (2026-11-04):** DROW numbers for the proposed architecture from **both** a natively trained
model and a FROG-trained one, set against the DROW figures this repository already holds, plus whatever further
experiments and parameter changes the evidence suggests.

**The four decisions, owner's call 2026-11-04.** Each changes what the number means, which is why they were asked
before anything was written:

1. **Range limit: DROW's own sensor limit, 29.99 m**, not FROG's 10 m. `sanitize_ranges` maps every reading at or
   beyond `max_range_m` onto that value, so a 10 m model cannot represent the 3,952 DROW labels (7.3%) past 10 m at
   all, and the published DROW baselines score every label. The **10 m-restricted-GT column is reported beside it**,
   because that is the only footing the zero-shot arm can be read on -- its weights carry FROG's 10 m.
2. **One angular grid, and it is DROW's: 0.5 deg per beam.** The k=7 kernel spans 1.75 deg on FROG and 3.5 deg on
   DROW. Rather than interpolate DROW up to 0.25 deg -- which invents a return on every depth
   edge, in a dataset whose invalid readings are already open space -- **FROG is decimated to 0.5 deg (720 -> 360
   beams over 180 deg, divisible by 8 exactly) and one FROG chain is retrained there** (item 73). Nothing is
   invented, the zero-shot arm then differs from its source in **rate and venue only**, and the arm answers a
   question of its own. The 0.25 deg FROG headline numbers are untouched by it.
3. **The DROW3 overshoot is read before any DROW number is quoted. DONE the same day, and it was a mis-citation.**
   Our weights are `...-odom.rot-trainval-...`, which is the DROW paper's "DROW3x + validation" curve at **71.7%**,
   not the unsourceable 61.9% this repository had been comparing against -- so 66.5% is a **-5.2 pp undershoot** and
   the overshoot rule never fired. The DROW column is read as a floor, not a match, and the comparison inside it is
   fair because one pipeline costs every detector the same 1-5 pp. `CHANGELOG.md` carries it; the DR-SPAAM cell it
   left uncited is item 79.
4. **One seed first.** The spread, and which parameters further seeds should carry, are decided from it rather than
   committed to now.

| # | action | why this one | cost |
| --- | --- | --- | --- |
| 72 | **DROW-native, one seed** -- the adopted recipe trained on DROW | A63 step 1, the headline cell, and the owner's goal since 2026-09-15 | ~8 h |
| 73 | **FROG at 0.5 deg**, one chain | removes the kernel-arc confound from item 74 **and** prices angular resolution on FROG, which nothing has measured | ~6 h |
| 74 | **FROG -> DROW zero-shot**, plus the dt-aware recurrence row | A63 step 2, and **item 37's last defence** | ~1 h inference + ~5 h for one extra stage-3 seed |
| 75 | **DROW capacity**: stage 2 sized for DROW's data, not FROG's | the cheapest arm here and the likeliest to move; §B's row, now actionable | ~15 min per arm |
| 76 | **DROW chunk length**: 10 s delivers 16x less supervision there | the one way to equalise gradients per rollout without touching the architecture | ~7 h per arm |
| 77 | **DROW slot capacity**: K = 256 is a 3.8x larger overshoot there | slot count is CPU cost, and CPU is the headline cost metric | ~7 h per arm; a replay screens it |
| 78 | **Beam-roll augmentation** | attacks absolute-beam-position memorisation where the training set is smallest | ~15 min per arm |
| 79 | **Source every `published` cell in the comparison table** | item 71 found two DROW cells naming figures no primary source holds, and the rest have never been checked | no GPU, ~half a day |
| 80 | **The fine-tuned regime** -- DROW training started from item 73's FROG weights | added 2026-11-05 on the owner's call: it is the third of the three regimes a DROW number can be quoted in, and the only one this project has never measured for this architecture | ~7 h + ~1 h of code |

**Three rows of this ranking no longer hold, and the register above is what is live.** On **2026-11-06 the
owner parked both transfer regimes**, so rank **5** (item 74) and rank **6** (item 80) leave the queue and the
DROW cell is native-only -- the register's parked group carries the triggers and the reasons. Rank **4** (item
73) survives, but not for the reason given below: it was 74's gate, and what pays for it now is the resampler
question alone. Ranks 1-3 are closed (items 72, 75 and 78's screen reported 2026-11-05; 79 is open and
unchanged), and items 81, 82 and 84 arrived after this table was written, so they are in the register only.

**Ranked by value and decisiveness, owner's request 2026-11-05.** The two axes disagree, which is why both are
listed: *value* is how much of what this project claims rests on the answer; *decisiveness* is whether the outcome
changes a decision, and whether it reads unambiguously in **both** directions. The headline item is not the most
decisive one.

| rank | # | value | decisive on its own? | cost |
| --- | --- | --- | --- | --- |
| **1** | **72** DROW-native | highest -- the only cell that exists for this architecture on DROW, and the thing every other item here is an arm off | **no**, and that is the finding below: a low number has two readings and item 72 cannot separate them | ~8 h |
| **2** | **75** + **78** capacity and beam-roll | medium alone | **yes, and they are what make item 72 decisive**: they separate "the architecture does not carry" from "17.8x less data" | ~45 min for three arms |
| **3** | **79** provenance | high -- it sets how every cell in the column is read, ours included | total: a cell gets a source or it is deleted | no GPU, runs beside anything |
| **4** | **73** FROG at 0.5 deg | medium -- a FROG result, and item 74's gate | yes: if 0.5 deg costs little, `docs/PROPOSAL.md` section 5.3's resampler was a theory rather than a requirement, which is worth knowing before JRDB's 1091 beams | ~6 h |
| **5** | **74** transfer + dt row | high -- item 37's last defence, and it answers the owner's 2026-09-15 question directly | **highest per claim**: the dt-aware cell survives a 3.2x rate change or it does not | ~6 h, gated on 73 |
| **6** | **80** the fine-tuned regime | high -- without it the DROW column has a hole, and `memory/interpreting-evaluation.md` says a DROW number quoted without its regime is close to meaningless | yes, and the prior is against it: zero-shot beat fine-tuning outright for all three superseded architectures, so a **win** is the informative outcome and a loss confirms a standing finding | ~7 h, gated on 73 |
| **7** | **76** chunk length | medium | partly -- a win makes the adopted 10 s dataset-dependent, which complicates the recipe rather than confirming it | ~7 h per arm |
| **8** | **77** slot capacity | low -- a CPU-cost parameter, not an accuracy one | weakest: a replay prices K under a *frozen* policy, so it confirms only where the policy cannot adapt to it | ~7 h per arm |

**What the ranking changed, and it is not a reordering of equals.** Items 75 and 78 were phase 4, to run *after*
item 72 reported. They belong **before item 72's stage 3**, because they are a pre-flight check on item 72's own
stage 2 rather than a follow-up to its result. Stage 2 trains on annotated frames only -- 17,665 frames on
DROW -- so item 75's two arms cost about half an hour together, and three quarters of an hour with item 72's own
stage 2 beside them. All three are read for the **train/val gap** rather than for val AP. Item 72's stage 3 is the 6-7 h half. Spending that on a stage 2 whose train AP is already climbing away
from val is the one avoidable waste on this list, and it is also the confound the headline number would otherwise
carry permanently: a DROW figure well below FROG's 86.3% has two readings -- the architecture does not carry, or
DROW's training set is 17.8x smaller -- and only an arm sized for DROW's data tells them apart.

**Item 72 still runs the adopted recipe.** Whatever the sweep says, its stage 3 trains on section 5.1's
(32, 64, 128), because the headline cell has to be the published parameter table carried over unchanged -- that is
the claim being tested. The sweep's own arms earn a stage 3 only if the gap they show is worth 7 h.

**Running order lives in the register above**, in one place, as its `gate` column. The phase table that used
to sit here said the same thing a third time and was deleted on 2026-11-05.

**Item 78 is not in the live queue, and that is deliberate.** The ranking puts it beside item 75 as the other
cheap overfitting arm, and it is one -- but beam-roll is an augmentation that does not exist yet, where item
75's arms are two flags that already do. So `utils/queue_drow.py` carries item 75 and item 78 follows the small
piece of code it needs, rather than holding up a run that is ready.

**Item 79. Give every published cell a citation it can be checked against.**

**What.** Each `plain` cell of `memory/performance-log.md`'s accuracy table carries its paper, table, row, split,
matching radius and the configuration the figure was published at -- and, where this project runs the authors'
weights, the row matching **those weights** rather than the paper's headline.

**How.** Item 71's method applied to the rest: read the filename and the loader of whatever `library/setup.py`
downloads, find which published row that configuration is, and record it. Two cells are already known to need it.
**DR-SPAAM on DROW**: 69.6% matches neither the paper's **70.3** nor the repository's retrained **72.3**, and our
wrapper's 56-point cutouts straddle the two, so the streaming reproduction's gap is -1.2 or -3.2 pp rather than the
recorded -0.52. **DROW3 and DR-SPAAM on JRDB**: taken from a repository README that states neither split nor class.

**Why.** A cell sat in that table for two months naming a figure that exists in no primary source, and the error
**inverted the sign of a reproduction gap** -- what read as a +4.6 pp overshoot, which the standing rule says means
the harness is broken, was a -5.2 pp undershoot, which it says is ordinary. The rule only works if the reference is
real.

**Item 74. FROG -> DROW zero-shot, and the dt-aware recurrence row.** *(parked 2026-11-06, owner -- the
register's §4 holds the trigger; the design below is kept because it is the only test of the dt-aware cell
left and whoever un-parks it should not re-derive the confound)*

**What.** Item 73's model run on DROW test with no DROW training, scored on the 10 m-restricted label set beside item
72's native arm; and `--recurrence selective` against the adopted minGRU on that same row.

**How.** Inference only for the transfer. The recurrence row is **not free, contrary to A63 step 2's note**: every
`selective` checkpoint predates the frozen architecture -- the six-arm design screen carried `selective`, and minGRU
replaced it on 2026-10-23 on a 0.52 pp tie -- so setting one against a frozen minGRU arm across a rate change would
confound the recurrence with the backbone. It costs **one** stage-3 seed with `--recurrence selective` on item 73's
stage 2, after which both arms transfer by inference.

**Why.** A63 step 2, and **item 37's last defence**. DROW runs at 12.70 Hz against FROG's 40.01 Hz, so a cell reading
real elapsed time should survive the rate change where one taking fixed steps cannot. Item 37 found the mechanism has
never been fed anything but a 1%-spread constant, so this is the only test of it left; and the result answers the
owner's 2026-09-15 question directly -- one model per dataset, one transferred, or both.

**Item 80. The fine-tuned regime, and the hole it fills.** *(parked 2026-11-06, owner -- after items 81, 84
and 76 give it a native baseline worth comparing against; the `step3a --init` design below is the part that
survives, since item 80 is the only thing that needs it)*

**What.** DROW training initialised from item 73's FROG weights rather than from scratch: stage 2 through the
existing `step2 --init`, stage 3 through a `step3a --init` that does not exist yet. Scored on DROW test beside
item 72's native arm and item 74's zero-shot one, on both label sets.

**How.** One arm, at a lower learning rate than a from-scratch run, off **item 73's** chain and not the 0.25 deg
headline -- otherwise the regime is confounded with the kernel arc the way item 74 would have been.
`step3a --init` is the one piece of code owed: about ten lines mirroring `step2 --init`, which loads weights
only and leaves a fresh optimiser, and the test is that a `--init` run with a matching architecture reproduces
the loaded checkpoint's own candidates before it trains. Note that initialising stage 3 from FROG's memory is a
warm start on a **different** candidate distribution, since stage 2 has moved; that is the regime as defined,
not a flaw in it, and it is why the fallback arm below exists.

**If it regresses below zero-shot, the fallback arm is §D's frozen-backbone variant**, which that section
proposed for exactly this failure: an earlier full fine-tune of a superseded architecture fell to 14.6% against
26.7-28.0% zero-shot, which reads as catastrophic forgetting on a small training set. Freezing the early
backbone and tuning the later layers is a flag, not a design change.

**Why.** `memory/interpreting-evaluation.md` states three regimes -- zero-shot, native, fine-tuned -- and that
quoting a DROW number without saying which one produced it is close to meaningless, with a spread of 2x or more
between them. Items 72 and 74 produce two of the three. This produces the third, and it is the only one never
measured for this architecture.

**What to expect, written down first.** The standing finding is that **zero-shot wins outright over both other
regimes, for every one of the three superseded architectures** -- a genuine result, driven by DROW's 17.8x
smaller effective training set rather than by a bug. So the informative outcome here is a **win**: it would mean
the architecture's capacity is matched to DROW's data where the older ones' was not, which is the same question
item 75 asks of width. A loss confirms a finding this project already rests on, and costs 7 h to confirm it.

**Item 84. A swappable decision head, with the binary one as an option rather than the architecture.**
*(the owner's proposal, 2026-11-05)*

**What.** The output head becomes a named, swappable component. `--head binary` is today's: one person logit
plus a two-channel vote, bit-identical to every FROG number ever measured. `--head multiclass` emits a class
score per annotated class plus the same two-channel vote, so one model detects pedestrians, wheelchairs and
walkers where the dataset labels them.

**Why this is the right shape for the claim, not just for DROW.** The **A** in TAKHeLiPeD is adaptivity, and
today it means one thing: the *parameters* in `docs/ARCHITECTURE.md` section 5.1 are carried to a new dataset
by changing values, never the architecture. `DatasetSpec` made that a fact about the code on 2026-11-05. But
the **output structure** has never adapted -- `DatasetSpec` has recorded `classes=("pedestrian", "wheelchair",
"walker")` and printed all three in every parameter table while the head could only ever emit the first. A
dataset whose labels this architecture cannot express is a limit on the claim, and DROW is that dataset:
**56.4% of the objects it annotates in test are not pedestrians** (`memory/dataset-properties.md`). This makes
the head one more thing the spec selects, which is the same move `--block`, `--input-features` and
`--vote-frame` already made for the stem and the encoder.

**How.** A `HEADS` tuple beside the existing `VOTE_FRAMES`, `BLOCKS`, `RECURRENCES` and `ASSOCIATIONS`, which is
this file's idiom for a swappable part. Today's head is one 1x1 convolution emitting three channels, split in
`_decode` into `out[:, 0]` and `out[:, 1:]`, so the seam already exists and the change is to make the channel
count and the split a property of the head rather than literals. Concretely:

| | channels | forward returns | notes |
| --- | --- | --- | --- |
| `binary` (default) | `1 + 2` | `logit [B, N]`, `votes [B, N, 2]` | exactly today, so FROG is untouched |
| `multiclass` | `C + 2` | `logits [B, N, C]`, `votes [B, N, 2]` | **shared** vote regression, per-class scores |

**The vote stays shared, and that is the published mechanism rather than our invention.** DROW3 and DR-SPAAM
carry exactly this split -- `conv_cls (n_cls)` and `conv_reg (2)` in `architectures.py` -- so a wheelchair and
a pedestrian on the same beam regress to the same centre and differ only in score. Preferring their shape on a
design with no measured winner is the standing rule.

**Both design questions were answered by the owner on 2026-11-06.**

**1. Softmax and independent sigmoids, both, as two arms.** So `multiclass` takes its normalisation as a
parameter rather than a decision: `softmax` is DROW's own and makes the published comparison exact, `sigmoid`
makes `binary` a strict special case so a one-class run is numerically identical and a binary checkpoint loads
into a multi-class network with the extra columns at zero, the way `prior_channels` already does. The arms cost
~15 minutes each on DROW stage 2 and they answer a question nothing here has measured: whether DROW's three
classes behave as mutually exclusive in the loss. **Report them beside each other**, and if they tie, the
standing rule picks `softmax` as the published mechanism.

**2. The object memory stays class-agnostic.** Its job is persistence, persistence is class-independent, and the
project's claim is about the memory -- so it is held still while the head varies, which also keeps every FROG
number and the whole slot ablation table valid.

**The owner's reason for 2 checks out, and it is worth recording why rather than assuming it.** The question was
whether the slot vector can hold class information anyway. It can: the memory is **not** fed a bare score. Each
candidate reaches it as `cand_features`, a `[B, P, C]` pool of **stage 2's own decoder features** over the beams
that voted for that candidate (`joint_step`), which under a `multiclass` head are exactly the features the class
logits are computed from. So the class signal is present in the slot's input by construction and no widening is
needed. What does *not* follow automatically is **pressure**: with a class-agnostic rescoring target the slot
encoder keeps class-correlated features only where they help persistence, which is the intended behaviour and
not a shortfall.

**The one thing this choice costs, named now so it is not discovered later.** A class-agnostic memory cannot
temporally smooth a **class label**: two objects at one place get one persistence score, and the class is
decided by the head on the current frame alone, so a label that flickers between wheelchair and pedestrian
frame to frame stays flickering. That is invisible to `wp-AUC`, which scores one class at a time -- it would
show up only in a label-stability or confusion measure, and **per-class slot state is the fix if it ever
matters**. It is a limitation to state in `docs/RESEARCH.md`, not a reason to reopen the decision.

**Checkpoint discipline, which is not optional here, and decision 1 makes it two rules rather than one.**
`head`, the normalisation and the class count change which weights exist and what they mean, so all three
belong in `network_shape` beside `channels` and `aggregator`, and in the stored args of every checkpoint --
otherwise a multi-class checkpoint silently rebuilds as binary and scores nonsense. A binary checkpoint loads
into a **`sigmoid`** multi-class network with the new columns zeroed; into a **`softmax`** one it must be
**refused** rather than padded, because the denominator changes and the loaded network would not compute what
the checkpoint did. A test per direction, both written before the loader is touched.

**Cost.** The head, the registry, `network_shape`, the loss, and the tests: most of a day of code with no GPU.
Then **two** DROW stage-2 arms at ~15 minutes each (`softmax` and `sigmoid`) against item 81's mask, and a FROG
regression run that must reproduce 82.80% exactly on `binary` -- which decision 1 makes a sharper test than it
was, since `sigmoid` at one class must reproduce it *numerically* and not merely closely.

**Order.** After item 81, and gated on nothing else. Item 81 is the cheap half of the same question -- whether
the wheelchairs are hurting us -- and if a loss mask recovers most of the precision then this item is about the
architecture's generality rather than about DROW's number, which is a different and better reason to build it.
**When it is built, the design belongs in `docs/RESEARCH.md`**, not here: it describes an architecture.

**Item 83. The kernel's arc in metres, on DROW's grid.** *(proposed 2026-11-05, weakest of the three)*

**What.** `--kernel-size 5` and `3` against the adopted 7, on DROW stage 2.

**How.** One ~15 minute stage-2 arm each, beside item 81's.

**Why, and why it is ranked last.** `k = 7` spans 1.75 deg on FROG and **3.5 deg on DROW**, which at 10 m is 61 cm
against 31 cm -- wider than a person, so each layer's receptive field covers two people where on FROG it covers
most of one. That follows directly from the one-angular-grid decision. **The argument against it is already
measured**: the four-shape-knob sweep varied `k` from 5 to 11 on FROG and found every arm inside the control's
range (`memory/rejected-ideas.md`), which is weak evidence that the arc does not matter much. It is a different
physical quantity on DROW, so it is not a re-run of a rejected idea -- but it is the one of these three the
existing evidence argues against.

**Item 76. DROW chunk length -- and its arms were wrong until 2026-11-05.**

**What.** `--chunk-s 35` and `65` against the adopted 10 s, on DROW, as stage-3 arms off item 72's stage 2.

**How.** One arm each, ~7 h, after item 72 reports.

**Why, corrected.** The chunk is specified in seconds and transfers by construction; what it *delivers* does
not. On FROG 10 s is ~400 frames and all 400 are annotated. On DROW it is **127 frames carrying about 6**
annotated ones -- a 60x deficit per rollout, not the 16x this item claimed, because the 16x came from assuming
DROW annotates uniformly at every 5th scan. **It does not.** It annotates in **bursts of 20 at every 5th scan,
spanning 7.5 s, with 24 s of nothing between them**, measured identically across all three splits
(`memory/dataset-properties.md`).

**That is why the arms changed from 20 s and 30 s to 35 s and 65 s.** The burst period is **400 scans =
31.5 s**, so 10 s is 0.32 of a period, 20 s is 0.64 and 30 s is 0.95: all three sit inside **one** burst, and
lengthening the chunk to 30 s buys almost nothing but unannotated frames. **35 s is the first length that spans
a second burst** (1.11 periods) and 65 s spans about three (2.06). An arm that cannot reach a second burst
cannot test the hypothesis.

**And a caveat the arm has to carry**: the 24 s gap exceeds the slots' 10 s fade, so a slot spawned in one burst
is retired before the next one starts. A longer chunk therefore hands stage 3 more supervised frames but still
**no gradient that crosses a gap**, unless the fade time is raised with it. Worth one arm at `--chunk-s 65`
with a longer fade, if 65 s alone moves nothing.

**ATTEMPTED AND BLOCKED 2026-11-06 at 35 s**, as `drow_chunk35` in `utils/queue_resolution_cost.py`, off item 72's
averaged stage 2 with the frozen memory and `--seed 0` -- an arm off the headline, written to its own directory so
the 10 s figure stays readable beside it. 65 s is **not** queued, on the owner's call: 35 s is the first length
that reaches a second burst at all and so carries the information, and 65 s is only worth 7 h once 35 s says there
is an effect. It trains on today's binary stage 2 deliberately, because the mechanism it measures is supervision
density and does not depend on the head item 84 will change.

**A null result has two readings**, and that is the caveat above rather than a hedge: either supervision density
is not what costs DROW its 20 pp, or the arm could not test it because the fade retires a slot before the next
burst arrives. The fade arm is what separates them, and it is the follow-up a null makes worth running.

**It did not get that far. `RuntimeError: bad allocation` inside `loss.backward()`, twice** -- epoch 6, then
epochs 7-8 after `--resume`. A **host** failure, not a device one: the GPU sat at 1.1 GB and the machine has
~5.3 GB free of 16 GB before anything of this project starts. The token budget is held constant
(`streams = tokens // length`), so a longer chunk does not raise frames per iteration -- it trades streams for
**depth**, holding a sequential graph 3.5x longer through one backward. It is the ceiling already on record from
2026-09-21, reached sooner (`memory/gotchas.md`). `--resume` buys a couple of epochs per relaunch, which looks
like progress and is not a path to a finished run.

**And the cost estimate was wrong independently of the crash.** Fewer streams means less parallelism per
iteration, so the same frames per epoch cost **2.52x** the wall clock: 1,147 s against the 10 s arm's 456 s. This
arm is **11.5 h** to the epoch its 10 s twin peaked at and **14 h** to run out its patience, not the ~7 h it was
queued at. Budget a long-chunk arm from `streams`.

**Eight epochs landed and they say nothing.** Neck and neck with the 10 s arm through epoch 5 (36.45% against
35.09% val), behind at 6-8 (34.16% against 39.14%) -- but the 10 s arm's best was **47.43% at epoch 36** and it
gained 8.3 pp after epoch 8. This item is **not** answered and not partly answered; reading the partial curve
either way would be the mistake the arm exists to avoid.

**What it needs to run.** Either host memory freed for ~12 h -- the owner's call, since it means not using the
machine for anything else -- or a variant that cuts depth rather than tokens, which is a design question rather
than a flag: `--tokens 500` would give one stream and halve the retained graph, but it also halves the token
budget the 10 s headline was trained at and so confounds the arm with its batch.

**Item 77. DROW slot capacity: K = 256 is a 3.8x larger overshoot there.**

**What.** K = 64 and 128 against the adopted 256, on DROW.

**How.** The replay is the cheap screen and the trained arms are the answer -- a replay prices a parameter under a
*frozen* policy, so it only confirms where the policy cannot adapt to it (`memory/interpreting-evaluation.md`).

**Why.** K was chosen as a deliberate overshoot of FROG's measured p99 slot occupancy. DROW has **0.81** people per
annotated frame against FROG's 3.06, so the same 256 slots overshoot by nearly four times as much, and slot count is
a CPU cost -- which is the headline cost metric here (owner, 2026-10-19).

**Item 78. Beam-roll augmentation.**

**What.** §D's first DROW-native idea, moved here because DROW is where it should pay: roll the whole scan and its
labels by a random beam offset each training batch.

**How.** Beside `--mirror`, which already does the one-off reflection and whose geometry test generalises to any
uniform grid. Zero architecture change, zero inference cost, one stage-2 arm.

**Why.** It attacks absolute-beam-position memorisation -- "beam 200 is the doorway in this room" -- directly, and
that failure mode is worst where the training set is smallest and the venue is a single room. It is also a free
control on item 75: both arms address overfitting, by unrelated means, so agreement between them is evidence and
disagreement is a finding.

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

### Superseded -- nothing here is work

A20, A24, A25, A30 and A31 became the three loading modes and their metrics, and both are **done 2026-09-10**.
A26 and A27 pointed at A35 and A34, which the architecture funnel closed on 2026-10-10. **A21 is answered** --
the four-way history ablation is now `memory/performance-log.md` footnote 3, moved there on 2026-11-05. **A5
dissolved rather than being done**: the 244-316 ms/frame was never the Python loop but a second sigmoid applied
to a channel the ONNX graph had already sigmoided, so 3,600 anchors entered an O(n²) merge every frame; fixed
at the source at **1.56 ms/frame**, with nothing left to vectorise.

**A1, the `dtime`/`time_frame` sweep, is the one row with a rule attached**: it was cut to one point, run, and
stopped at epoch 13, because a sweep over an axis the model provably ignores (0.5 pp under shuffle) measures
nothing. **Do not resume it without a reason from the register.**

### Stalled — still open, still valid, wrong moment

**A22 and A23 were merged into §D on 2026-11-05 and no longer exist here.** A22's `--diff-channels` is §D's
frame-to-frame difference channels; A23's `pos_weight` half is §D's class-imbalance check. Both were stalled on
a funnel that closed and an architecture that was replaced, so what is left of each is the §D idea, which is
written against the architecture that exists. A23's other half is closed: `balanced` supplies person-free
training frames from the same sensor and venue, and they were measured on 2026-11-04.

**A7. Clean up `scratch_frog_weights/`.** Housekeeping, no trigger; do it when convenient.

**A9. Makefile Python 3.12 check.** Housekeeping, independent of research direction.

**A10. Stop hardcoding `RESULT_CONF` in the ROS node.** Deployment-side; nothing reaches the robot until someone rebuilds the image anyway.

**AL1-AL7 (CI, ruff, shellcheck, hadolint, per-detector smoke tests, markdown strictness).**
All independent of the research direction and all still worth doing; parked only because §A is where the value is right now.
AL6 (a smoke test per detector class) is the one that would pay off soonest, since A35 will add architectures.

---

## §B UPON A FULL RETRAIN SWEEP

Batched because each item needs a from-scratch run, and this project batches several combinations behind one
GPU session rather than retraining once per idea. **Both of this section's experiments have left it**, so what
remains is one standing caveat rather than a queue:

- The capacity-shrink experiment is **register item 75** as of 2026-11-04, against TAKHeLiPeD's stage 2 rather
  than the retired `SpaceTimeCNN`, and it is no longer expensive: stage 2 trains on annotated frames only,
  17,665 of them on DROW against FROG's 108,356, so an arm is ~15 minutes. It is running as this is written.
- ~~Retrain `FullScanTCN` and `TemporalUNet` at real-odometry `dtime=5`~~ **removed 2026-11-05**: item 32
  retired both detectors on 2026-10-31, so there is nothing to retrain.

**The caveat that stays.** Every FROG checkpoint trained with `--align-scans` before 2026-09-14 saw wrongly
interpolated headings in the windows touching a +-180 deg crossing (A48). The next sweep retrains on the fixed
loader, and no number from an old checkpoint is compared with a new one without saying so.

## §C BLOCKED — waiting on time or data

*Empty since 2026-11-11.* JRDB was the last item here, and its data is now on disk; the work is register row A63.3.

---

## §D LATER

### Two questions that left this section, kept as pointers

The owner's 2026-09-17 hypothesis -- *do the object slots already cover the coarse horizon?* -- was answered as
A60 on 2026-10-08: **the coarse convolution was dropped and removed**, AP -0.87 and first-second recall +0.59
against a +2 bar. The `balanced` partition left for A63 step 0 on 2026-10-10 and is **done**.

**The partition rule both leave behind, which is the part still worth having:** never mix partitions.
`balanced`'s val split is 11-36 + 10-31 and **11-36 is half of `official`'s train**, so validating there while
training on `official` scores the model on its own training data. State the deviation when quoting: the
published baselines trained on 11-36 + 12-43, where `balanced` trains on 12-43 + 14-57.

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

1. ~~**Beam-index shift augmentation**~~ **is register item 78** since 2026-11-04 and was deleted from this
   list on 2026-11-05, so the idea has one home.
2. **Frame-to-frame difference channels** — **absorbed §S's A22 on 2026-11-05**, which asked for the same thing
   (`--diff-channels`, a collapse operator, an auxiliary motion loss) against an architecture since replaced — feed the current frame plus `(T-1)` delta channels (`frame[t] - frame[t-1]`, post-alignment) instead of `T` raw stacked frames, making "temporal" explicitly about motion rather than repeated absolute position.
3. ~~**Local/relative range normalization**~~ **is register row A39** (a locally range-centred input channel,
   built on Cartesian input after A42 settled that the two are complementary), merged there 2026-11-05. The
   mechanism is the same: the DR-SPAAM centering idea without DR-SPAAM's per-beam architecture.
4. **Range/scale jitter augmentation** — randomly scale/offset a whole frame's range values during training, forcing invariance to absolute distance rather than removing that information via a fixed transform.
   Complements #3.
5. **Frozen-backbone fine-tuning from FROG** — **now register item 80's fallback arm** rather than a separate
   idea (2026-11-05). An earlier full, unfrozen fine-tune of a superseded architecture regressed *below*
   zero-shot transfer (14.6% against 26.7-28.0% wp-AUC on that run, a cruder attempt than the `--init-weights`
   result in `memory/interpreting-evaluation.md`), which reads as catastrophic forgetting on DROW's small
   training set. Freezing the early backbone and tuning only the later layers is a flag, so it is the thing to
   try if item 80's full fine-tune repeats that failure.
6. **Class-imbalance check** (lower priority) — verify whether the classification loss already weights the
   heavily-imbalanced person-vs-background beam classes; a class-balanced or focal loss is a standard, cheap
   lever if it does not. **Absorbed §S's A23 `pos_weight` half on 2026-11-05**, which was the same question
   asked of a retired architecture.

**Why.** Targets DROW-native accuracy specifically. **Two of the six have been promoted and two merged**, so
what is left here is ideas 2, 4 and 6 -- each unstarted, each behind the register's DROW chain, and each now
written in only one place.

---

## §E PROJECT EVOLUTION

### Should any of this project's newer detectors, the tracker, or the real-odometry pipeline ever be wired into the ROS deployment?

Today the robot runs only `DrowDetector`, unchanged in spirit since before this research began (`memory/deployment.md`).
Wiring in `SpaceTimeCNN` (the current accuracy leader) or `SimpleTracker` would need real ROS-node integration work — a new node or a modified `DROW_detector`-equivalent, plus on-robot CPU inference-speed validation, since the training-side ms/frame numbers in `memory/performance-log.md` are batched, not single-frame-real-time measurements.
Needs an owner decision: is on-robot deployment of the research results a goal at all, or does this project's scope end at the research comparison?
