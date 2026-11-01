# Detector architectures — how each one turns a scan into detections

*keywords:* detector, cutout, full-scan, TAKHeLiPeD, three-horizon, calibration network, object memory, SpaceTimeCNN, FullScanTCN, TemporalUNet, DrowDetector, DrSpaamDetector, LFE, AlgorithmicDetector, Aycard, DETECTOR_REGISTRY, source of truth

The full research write-up — literature review, citations, and the fidelity audit against each paper — is [`../docs/RESEARCH.md`](../docs/RESEARCH.md).
This file is the short, operational version: what an agent needs to predict a detector's behaviour without reading that whole document.

## The core idea

Every detector in this project answers the same question — for each beam in a 2D LiDAR scan, is there a person/wheelchair/walker there, and where exactly — but takes one of two structurally different routes to it.
**Cutout-based** detectors (`DrowDetector`, `DrSpaamDetector`, `Li2FormerDetector`) extract a fixed-width polar window around each beam and run a shared per-beam CNN over it independently, with no or only local cross-beam communication.
**Full-scan** detectors (`SpaceTimeCNNDetector`, and the ONNX-only `LFEPeaksDetector`/`LFEPPNDetector`) process the entire raw scan at once, with dilated convolutions standing in for the cutout's fixed window.
(`FullScanTCNDetector` and `TemporalUNetDetector` were full-scan too and were **retired on 2026-10-31**, `TODO.md` item 32 — see *Lifecycle* below.)
The research question this project started from: can a full-scan, non-recursive (no RNN, no attention) design match or beat the cutout-based state of the art, while being far cheaper per frame.
**That framing was superseded on 2026-09-14.** `TAKHeLiPeD` (below) keeps the full-scan half and deliberately lifts the non-recursive constraint, because the evidence that separates a chair from a standing person only exists at a horizon of tens of seconds, which no feed-forward window can reach (`../docs/PAPER.md`).

## Lifecycle

1. **Ingest** one or `T` consecutive scans (`get_scan()` on the active dataset, or `LiveDataset.push_measure()` for ROS).
2. **Align** (optional, `--align-scans`, default on) — historical frames are rotated by the accumulated odometry delta into the current frame, via `aligned_raw_scan()` (full-scan) or the cutout's own built-in rotation (cutout-based).
   With zero or absent odometry this is a no-op, not an error.
3. **Extract input** — a `(N_beams, T, N_SAMP)` cutout tensor, or a `(N_beams, T, 1)` raw-range tensor.
4. **Forward pass** — per-beam classification logits + a 2D vote offset (the DROW-style head), or a per-beam heatmap value (the LFE-style head, also available on Architectures A/B via `--head heatmap`).
5. **Cluster** votes/heatmap peaks into `(x, y, class, confidence)` detections (`votes_to_detections()` / `find_peaks`).
6. **(optional) Track** — `SimpleTracker` (training/eval side, `library/takheliped/utils/tracking.py`) or `PersonTracker` (the ROS node) turns the per-frame detection list into one persistent, ID-stable position, predicting through missed frames up to `max_age` and requiring `min_hits` real matches before reporting a track at all.

Every stage after preprocessing is shared by every detector; only the extraction (step 3) and the head (step 4) differ by family.

## TAKHeLiPeD — the current detector

**T**(emporal) **A**(daptive) **K**(nee-)**He**(ight) **Li**(dar) **Pe**(rson) **D**(etector), named 2026-09-23 — *TA-KHé-Li-PeD*.
Everything written before that date calls it **the three-horizon detector**, and the code still spells it `three_horizon`: `library/takheliped/detectors/takheliped.py`, `utils/train_takheliped.py`, `checkpoints_three_horizon/`.
It is the only detector here that is **stateful across frames**, so it does not go through `DETECTOR_REGISTRY`, `train.py` or `evaluate.py` at all — it has its own trainer with its own steps (`commands.md`).

It breaks the lifecycle above in two places: there is no `T`-scan window (one scan per call), and there is no post-hoc tracker (the memory replaces it, and beats it — `performance-log.md`).

| Stage | What it is | Parameters | Cost |
| --- | --- | --- | --- |
| 1, features | 10 pose-invariant per-beam features from one raw scan; `sanitize_ranges` maps non-finite, non-positive and beyond-`max_range_m` readings to `max_range_m`, so no dataset-specific preprocessing is needed | — | — |
| 2, calibration network | 1D ConvNeXt U-Net over **one scan**, a 3-tap convolution across bottleneck sectors and a global max; a DROW vote head, decoded by `nms-mean` into 64 candidates, sub-threshold ones included. No state, no pose: its coarse temporal convolution was removed 2026-10-08 (A60) | 0.98 M | 8.75 ms + 2.77 ms decoding, CPU, 4 threads |
| 3, object memory | 256 slots in **room** coordinates and a dt-aware selective diagonal recurrence (Mamba-style, decay 1-60 s), rescoring the candidates. **What the frozen model uses is a subset of what the class can do**: cross-attention association and slot self-attention both exist and are both **off** (E7, E6), there is **no coasting head** (item 24), and the decay is learned rather than fixed (D3). The full configuration, and what each alternative cost, is below | 0.17 M | ~4.4 ms/frame |


### The frozen architecture (2026-10-18; the recurrence changed 2026-10-23)

**Slot capacity, replayed (2026-10-28, `TODO.md` item 50).** `a71_dim64_s0` played at other slot counts with its
weights untouched, scored on the soft-label stage 2:

| slots | 32 | 64 | **128** | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| memory AP | 84.78% | 84.67% | **85.01%** | 85.01% | 85.01% |
| gain | +2.03 | +1.91 | **+2.25** | +2.26 | +2.26 |

**Re-run on fixed weights and its own stage 2, there is no knee at all** (2026-10-28): 32 / 64 / 128 / 256 / 512
score 85.61 / 85.71 / 85.75 / 85.73 / 85.73%, so even 32 slots loses 0.12 pp. The "cliff at 64" in the first replay
was an artefact of playing a hard-label memory against soft-label candidates. **For 128 the replay says +0.02 pp
where the trained arm says -0.74: the replay is wrong by 0.76 pp**, in the direction that would have caused an
adoption.

**The replay said 128 is the knee -- and a trained arm contradicted it (2026-10-28).** Trained at 128 the memory
scores **84.99% against 256's 85.73%**, a **0.74 pp loss**, where the replay had them tied to two decimals. The
replay evicts under the *same trained policy*; a model trained at 128 learns a more conservative one (precision
68.7% -> 74.1%, false positives 1.220 -> 0.871, recall 87.4% -> 81.2%), which a replay cannot see. **256 is kept.**
The original replay reading is left below because the contradiction is the finding: A flat replay curve is weak evidence for *adding* capacity -- the
weights adapted to 256 -- but is decent for *removing* it, since a model that needed those slots would lose
immediately, and at 64 it does. **Confirmation by a trained arm is `TODO.md` item 50** (`queue_item50.py`), against
`a71_dim64_s0`'s 86.12% on the same hard-label stage 2, so capacity is the only variable.

**Read the absolute numbers with care**: this replay plays a memory trained on the *hard-label* stage 2 against
*soft-label* candidates, which costs 1.11 pp (85.01% here against the arm's own 86.12%). The curve is valid because
all five points share that mismatch; the level is not comparable to a trained arm. **The results JSON was lost to a
duplicate-key crash in the writer after the sweep had run** -- the numbers above come from the run's stdout, and the
bug is fixed, so a re-run will produce `results/item50_slots.json`.

## The architecture sweep, consolidated (2026-09 to 2026-10-28)

**Fourteen architectural questions were measured. Two were load-bearing.** Every row below is FROG `official` test at
stride 5, memory AP, seed 0 unless stated. **Read the `gain` column, not the AP**: arms sit on two different stage 2s
(candidates 82.22-82.24 before item 44(a), 82.76 after), so AP is not comparable across that line while the memory's
own contribution is.

| question | item | arm | memory AP | gain | verdict |
| --- | --- | --- | ---: | ---: | --- |
| **per-object addressability** | 36 | one global state | 82.44% | **+0.22** | **LOAD-BEARING** -- 6% of the slots' gain |
| **object vs spatial binding** | 22 | grid slots, 1 m | 84.46% | **+1.70** | **LOAD-BEARING** -- 44% of it |
| ” | 22 | grid slots, sensor frame | 83.62% | +0.86 | 22% |
| ” | 22 | grid slots, 2 m *(screen)* | 83.47% | +0.71 | 19% |
| recurrence rule | 16 | gru / mingru / selective | 85.9-86.4% | ~+4 | tie; minGRU adopted as citable |
| position refinement | 46 | `--refine` | 86.24% | +4.03 | tie; coverage unchanged |
| association | 43 | hungarian | 86.08% | +3.86 | tie on AP **and** CPU |
| the 1 s curriculum | 41 | `--chunk-s 10` | 85.97% | +3.75 | tie, 24% cheaper -- **removed** |
| whole-scan aggregator | 40 | `--no-aggregator` | (stage 2) 82.24% | -- | tie -- **removed** |
| slot-state width | 49 | `--dim 64` | 86.12% | +3.87 | tie, **-28% CPU** -- **adopted** |
| attention heads | 49 | `--heads 8` | 85.99% | +3.74 | tie, no cost change -- dropped |
| stage-2 beam target | 44a | soft Gaussian | (stage 2) 82.79% | -- | **+0.69 pp at 3 seeds -- adopted** |
| the published head | 44b | peaks, no votes | (stage 2) 75.35% | -- | dropped without an arm, -7.4 pp |
| Slot Attention's update | 42 | `--iterations 3` | -- | +2.22 val @ e18 | **NOT TRAINABLE, AND 31.45 ms CPU** (item 53) -- see below |

**Item 42's arm has no score because the published rule is unstable when the slots persist** (item 53, measured
2026-10-28, from the batch the trainer saved). Slot Attention re-initialises its slots on every input, so it
backpropagates through `state = state + mlp(state)` exactly `iterations` times. Our slots persist across the chunk, so
`--iterations 3` applies that residual about 1,200 times in one backward graph. The gradient arriving at the carried
state grows **2.1x per frame** (5.3e-1 at frame 290 to 1.1e16 at frame 240), reaches `inf` ~60 frames back and `NaN` 40
frames later; the forward loss stays finite and the state magnitude saturates near 347, which is why three earlier
forward-side checks cleared it. A length bisect is monotone: 50 frames -> 8.81, 150 -> `inf`, 300 -> `NaN`. The
LayerNorms in the block normalise the state *entering* the attention and the MLP, never the state carried between
frames. **`GRAD_CLIP` runs after `backward` and so cannot prevent it.** Scoring the row therefore needs a stabiliser --
a per-frame gradient clip on the carried state is the only one that leaves the architecture, chunk and loss untouched
-- which is a second change beyond the ablation as specified. That clip now exists as `--state-grad-clip` (off by
default) and holds over all 399 frames. **A carried-state LayerNorm was tried first and made it worse**, 3.3x per
frame against 2.1x, because the growth is in the Jacobian product and LayerNorm's `1/sigma` backward strips the
damping a growing state had supplied.

**And the arm cannot be deployed even once it trains.** A matched CPU sweep of `iterations` 1 against 3 -- identical
otherwise, same stage 2, interleaved, 3 repeats, stage-2 floor 0.2 ms -- gives memory 9.15 -> 20.77 ms and total
**19.65 -> 31.45 ms against FROG's 25 ms period**. Scaled onto the adopted `dim 64` that is ~26.2 ms, still over;
`iterations=2` would fit near 22.0 ms but is not the published setting. So item 42 is a ledger row, not a candidate.
| block / widths / depth / kernel / expansion | 19, 45 | five stage-2 knobs | ~82.2% | -- | all tie |
| sensor-frame vote targets | 18 | `--vote-frame sensor` | -- | -- | **worse** |
| recall width `sigma_m` | 39 | replay sweep | -- | -- | dose-response; confirms 36 |

**The two that mattered answer the same question from opposite sides**, and together they are the contribution:
item 36 says the memory must be *addressable* (one global state keeps 6% of the gain), item 22 says the addressing
must be *to objects* (binding to room cells at the same capacity keeps 44%). Nothing else moved the number.

**Nine ties is the real finding about the rest.** Recurrence, association, refinement, curriculum, aggregator, width,
heads, block and the stage-2 shape knobs all sit inside E13's 1.32 pp bar. That is why the adopted model is *smaller*
than the one the project started with: every tie was resolved towards the cheaper or simpler option, which is E13's
standing rule.

**One arm has no *test* number, and the reason it was dropped is a judgement that should be stated as one.** Item 42
trained **19 epochs** and died on a non-finite-gradient guard whose cause was never found. It was not abandoned
because of the crash: it was abandoned because of **where its curve was when it crashed**. At epoch 18 its validation
gain was **+2.22 pp** against the simplified baseline's **~+2.70 pp** at the comparable point, and that baseline went
on to +5.09 -- so the arm was tracking *behind* its own control with no sign of the sudden gain the
run-everything-once rule exists to catch. Restarting meant a debugging session plus a fresh ~10.5 h arm to pursue a
line already running second.

**Reopened on 2026-10-28 (the owner's call): it will be debugged and re-run rather than left as a guess.** The
reasoning below stands as the description of *why it was deprioritised at the time*, but it is no longer the final
word -- "better to spend a few more hours than be insecure about the results later on". `TODO.md` item 53 carries the
plan, and the arm will be re-run on the adopted recipe so it is comparable to `a71_dim64_s0`.

**The honest description of the original decision is "a grounded guess from partial results", not "unknown error, dropped".** The evidence
is real but weak in a specific way: a validation curve at epoch 18 of ~50 is a poor predictor -- `dim 64` itself went
+4.15 at epoch 20 to +4.96 at epoch 52 -- so the decision rests on *relative position against a control measured the
same way*, which is the strongest reading a partial curve supports and weaker than any finished arm here. It belongs
in the write-up as an idea **deprioritised on interim evidence**, and it is the one row where a reviewer could
reasonably ask for the full run.

**A caveat that governs the freeze.** `--dim 64` was measured on the **hard-label** stage 2 (candidates 82.24) and the
soft Gaussian target was adopted afterwards, so **the adopted combination -- soft labels + `dim 64` -- has never been
run end to end.** Each component tied or won against its own control, but items 40 and 41 are the precedent for why
that is not sufficient: both tied individually and "both at once" still needed its own measurement. The freeze is that
measurement.

**Stage 2 trains against a soft Gaussian beam target as of 2026-10-27** (`TODO.md` item 44(a)): each beam's
classification target is `exp(-d^2 / 2 sigma^2)` in **metres** with `sigma = 0.3 m`, clipped to 1 where people overlap,
over exactly the beams the hard label marked positive (`--label-radius`, 0.6 m). Only the values change, hard 1 to a
smooth falloff; the vote head, `nms-mean` and the class balance are untouched, so the arm varied one thing.
**Replicated at three seeds against three**: mean `nms-mean` AP **82.79% against 82.10%** with non-overlapping ranges
(worst soft seed 82.70% beats best hard seed 82.24%), and false positives at the fixed 0.3 threshold down **18.5%**
(2.841 against 3.488) for 0.46 pp of recall. The width is metric and not LFE's fixed two beams because a person
subtends 42 beams at 2 m and 9 at 9 m. Costs nothing at inference -- same architecture, same 960,323 parameters.

**The slot state is 64-wide as of 2026-10-27** (`TODO.md` item 49, the owner's call), not 128. It ties on accuracy
-- test stride 5 memory AP **86.12%** against `dim 128`'s 86.20% and the older control's 86.40%, with recall at matched
false positives **96.5%** against 96.43%, all well inside the 1.32 pp bar -- and it takes **28% off the memory phase**
on the CPU, 9.30 -> 6.70 ms in a matched interleaved sweep whose own noise floor was 0.53 ms. Total CPU falls
20.06 -> 17.73 ms and p99 margin against FROG's 25 ms frame period nearly doubles (2.3-2.8 -> 4.3-4.8 ms); stage 3's
checkpoint drops from 706 KB to **195 KB**. **The freeze's own sweep reproduces the ratios** (item 47, 2026-10-30, the
three frozen seeds against the same `dim 128` control, and repeated on a cold host as item 57): memory phase
10.08 -> 7.23 ms (**-28.3%**, 34x that session's 0.084 ms floor) and total 22.34 -> 19.32 ms (-13.5%), with the control
at **24.65 ms p99 against the frozen architecture's 21.67** -- inside the 25 ms budget by 0.35 ms, which is to say with
nothing left for a slow frame. The absolute levels differ from the figures above by ~12% because those are a different
session -- which is exactly why only the ratios are quoted across them, and the ratios agree. **The two 2026-10-30
sweeps, warm and cold, agree with each other to 0.4% on the total**, so the level is a property of this environment and
not of thermal state; the gap to the older sessions is unexplained and is not the checkpoint either (both stage 2s are
960,323 parameters and aggregator-free, hence identical at inference). It is the clearest efficiency result of the whole architecture sweep, and
it is on the axis the owner made decisive on 2026-10-19: an architecture that wins on accuracy and loses on CPU has not
won, so one that ties on accuracy and takes a quarter off the dominant phase has. **`heads` stays at 4**, measured: the `heads 8` arm scored **85.99%** against the
baseline's 86.20%, a tie inside the bar, and a head split redistributes attention without adding any -- so it leaves
the parameter count and the CPU cost untouched and has no efficiency case to rescue a tie.

**The architecture was simplified twice on 2026-10-25** (`TODO.md` items 40 and 41, the owner's call): stage 2 has
**no whole-scan aggregator** and stage 3 has **no 1 s curriculum stage**. Measured together as `a70_baseline_s0`:
**86.20%** against the previous 86.40%, inside the 1.32 pp bar, for 16,384 fewer parameters and 28% less training
time -- and with recall at matched false positives *up*, 96.43% against 95.84%. The recipe lives in
`utils/queue_architecture_sweep.py`'s `simplified()` helper, **not** in constructor defaults: `network_shape` falls back
to a constructor default for checkpoints written before a flag existed, so defaulting `aggregator=False` would make
every older stage-2 checkpoint rebuild without it and refuse to load.

**The per-slot recurrence is minGRU as of 2026-10-23** (`TODO.md` item 38, the owner's call), not the bespoke selective
diagonal decay. Item 16 found the three recurrences tie on AP, item 37 found the dt-aware rule the *most* fragile under
dropped frames and the `dt` it reads a 1%-spread constant, and item 36 found that the **slots** are what carries the
gain -- a global state reaches 82.44% against 85.88-86.08. So the citable cell replaces the re-implemented one and the
contribution, per-object addressability, stays. **The headline below is still the selective variant's**: minGRU has one
seed, and until seeds 1 and 2 run every number must name its variant.

**Step 1 ran every architectural question to an answer and nothing beat the adopted choice**, so this is the model the
pre-print proposes. Written out in full because "the adopted recipe" has meant three different things over this project's
life, and `TODO.md` item 29 exists because a number once outlived the configuration that produced it.

| stage | what it is | what was ruled out, and by what |
| --- | --- | --- |
| **1-2** | A59's C1 recipe over a **fixed-beam** ConvNeXt 1-D U-Net: dropout 0.1, nearest-person labels at 0.6 m, mirror augmentation, plateau schedule, weight averaging over the val-chosen window. Widths (32, 64, 128), 976,707 parameters | **No metric gather** (item 15: 1.42 pp worse at stage 2, 0.21 pp below the control's floor end to end, against a Tier 0 probe that priced it at +2.5 pp). **No LFE input channel** (A59 F1). **Not LFE's widths** (A59 W: 1.4 pp for no speed). **No temporal convolution**, fine or coarse (A50, A60) |
| **3** | E13's object memory: 256 slots in **room** coordinates, **accumulated** value, a **minGRU** per-slot recurrence (Feng et al. 2024; item 38, 2026-10-23 -- was a learned input-dependent decay over 1-60 s), no slot-to-slot exchange, no learned matching, a scalar rescore gate, **unbounded** rescoring, cosine schedule | **No coasting head** (item 24: removing it costs nothing measurable and it fired 2/0/0 times across 72,533 person-free frames). **Not the latest score** holding a slot (D2, interchangeable, so the design's own rule stays). **Not one fixed decay** (D3: 1.04 pp). **Not a bounded rescoring** (O1: a +-4 logit bound silences the head -- its AP equals its candidates'). **No slot-to-slot exchange** (E6), **no learned matching** (E7) |
| decoding | `nms-mean` at 0.5 m: greedy over beams by probability, each detection at the probability-weighted mean of the votes it suppresses | **Not the vote grid** (2.1 pp of AP), **not a range-split or unioned decoder** (items 13, 17, 34), **not another radius** (0.4 m and 0.6 m trade range against crowding without moving the total) |

**Two choices are load-bearing and the rest are not.** Of eight arms measured at one seed each against a three-seed
control, five sat inside the control's own spread -- what holds a slot, how the rescoring is gated, the schedule,
coasting's weight, coasting's existence. Only the **learned decay** and the **unbounded rescoring** cost anything to
change. That robustness is worth stating in the paper rather than hiding: the memory's performance does not rest on its
knobs.

**The headline, measured on an architecture that has since been superseded twice** -- it is the *selective* variant's,
on the aggregator-ful stage 2, with the curriculum, while the adopted model is slots + minGRU with neither and has
**one** seed at 86.20% (2026-10-19, three seeds, FROG `official` test every 5th annotated frame): **wp-AUC 86.01%** (86.08 / 86.08 / 85.88), **0.958 false positives per frame** (0.885 / 0.903 / 1.086), recall
84.03%, over candidates of 82.19% -- so the memory is worth **+3.82 pp** over its own input. On the CPU at four threads
it costs **19.15 ms per frame, p99 22.07** -- 88% of FROG's 25 ms period (item 30).

The **85.87%** quoted before this date described the same model *with* a coasting head: 0.14 pp apart, inside the
1.32 pp three-sigma bar, so removing it cost nothing measurable. **The seed spread fell from 0.86 to 0.21 pp** and the
recall spread from 6.59 to 3.64 -- suggestive of a tighter model, but three seeds against three cannot settle a
variance claim and the paper should not make one.

Only stage 3 carries state between frames: its slots, ~138 KB per stream (the coarse convolution's ~1.8 MB cache went with it), and the per-frame cost does not grow with how far back the memory reaches. The whole detector streams at 20.8 ms per frame (p99 24.1) on this machine's CPU at 4 threads, inside FROG's 25 ms scan period (`performance-log.md`).
**`manage_slots` runs under `no_grad`** — association, value accumulation, spawn, replacement and retirement are fixed rules; only what reads the slots learns.

**What A51 settled about stage 3 (2026-09-29/30), and what the write-up should propose.**

- **Keep the learned slot state, and keep it addressable per object.** E10 priced removing the state at −1.50 pp AP, −1.67 pp recall at matched false positives and −14.08 pp coasting precision, all past their bars; item 36 (2026-10-23) priced removing the *addressing* -- one global scene state instead of 256 slots, strongest pooling, published cell -- at **82.44% against 85.88-86.08**, capturing 5% of the slots' gain. **Which rule updates that state does not matter** (item 16: three recurrences inside 0.52 pp), which is why item 38 took the citable one.
- **Drop both attention mechanisms.** Rule-based association plus the per-slot recurrence ties the full model on every metric at three seeds, at **173,579 parameters against 306,443 (−43%)** and **3.72 ms against 4.53 (−18%)** (E6, E7, E13). This is the architecture to propose; the full model is the ablation.
- **Drop the feedback path.** `render_prior` and `prior_channels` contribute 0.00 pp in three configurations, because the prior carries 0.012–0.045% of `aggregate`'s input and its exclusive ceiling is 1.91 pp of recall anyway (E12). Dropping it also drops the `step3b` joint stage, which already scored worse than `step3a` (78.84% against 80.10%) — this session localised that loss to the **rescoring head's** drift on stage 2, not the prior.
- **Still there, still conditional:** room coordinates win only below ~0.02 m/s of odometry drift (D1, E2); training with jittered poses is the untried repair (`../TODO.md` E11).
Design rationale: [`../docs/PROPOSAL.md`](../docs/PROPOSAL.md). Evidence for each slot decision, or an explicit note that there is none: [`slot-design-evidence.md`](slot-design-evidence.md). Numbers: [`performance-log.md`](performance-log.md).

**Stage 2's recipe, decided 2026-10-06 (owner, `../TODO.md` A59):** decode with `nms-mean` (NMS over the beams' votes, each detection at the probability-weighted mean of the votes it suppresses), train with mirror augmentation, and assign each training beam to its nearest person within 0.6 m. Together these take the no-temporal stage 2 from 77.1% to ~81.5% on FROG test; a second median channel (F3) was tested and dropped (`rejected-ideas.md`). **With dropout 0.1 and weight averaging, also adopted 2026-10-06** (`step2 --dropout 0.1 --checkpoint-every 0.25`, then `utils/average_checkpoints.py`), the no-temporal stage 2 reaches **82.20%**. **They are the trainer's defaults since 2026-10-06** (`decode_frame`, `step2 --label-assign nearest --mirror --dropout 0.1`; averaging is its own step); every stage-3 number until A59 phase 3a rests on the old recipe -- vote grid, no mirror, last-listed labels.

**What A59 found about stage 2 (2026-10-05)** (`static-detector-diagnosis.md`, last section):

- **Its decoder understates it.** The official DR-SPAAM decoder (`nms_predicted_center`, 0.5 m) on the same outputs scores +2.33 to +3.03 pp higher on all six stage-2 checkpoints.
  The vote grid stays the default until a decoder is chosen on val and stage 3 is retrained on the new candidates.
- **Its gain over LFE is far and isolated people.** Through the vote grid it also loses close pairs (60% recall against LFE's 83%) and 1-4 m, but both are the vote grid merging neighbours: through `nms-mean` (chosen on val on all six) it reaches 82% of close pairs and beats LFE at every range, at **80.07%** AP.
- **The coarse convolution: dropped by A60's rule and removed (2026-10-08)** -- end to end, on the new recipe, three seeds each, the default architecture scores 85.01% against the no-temporal 85.87%, and +0.59 pp on people in their first second against a +2 pp bar (`../TODO.md` A60). Its code is at commit e76aa94 (`rejected-ideas.md`). Before that measurement it already earned nothing measured: a tie at stage 2 under either decoder, the same recall in every range and crowding bin, at +20.6% parameters, ~3.3x the training time, a 1.8 MB streaming cache and odometry inside stage 2.
  The end-to-end test, with the rule that drops it, is `../TODO.md` A60; until it runs, the no-temporal stage 2 -- one frame per call -- is the one every stage-3 result uses.

**Older architecture names in the logs.** Between 2026-09-17 and 2026-10-08 the runs called "no-fine" or "default" kept the coarse convolution and "static" or "no-temporal" did not (`--coarse-lags 0`); directory names keep the old words. Only the no-temporal kind still loads (its one-lag `coarse.conv` folds into `context`), and it is what stage 2 now is; the others are refused with a pointer to e76aa94.

## The three earlier full-scan architectures — superseded

Preceded TAKHeLiPeD; every accuracy number they carried is retracted (`performance-log.md`), so they are provenance, not results.
**Two of the three were retired on 2026-10-31** (`TODO.md` item 32): nothing had trained on `FullScanTCNDetector` or `TemporalUNetDetector` since 2026-09 and neither influenced a project decision, so their code and their four private blocks (`DilatedScanBackbone`, `_TemporalTCNBlock`, the two U-Net blocks) are gone, last existing at commit **`98b0716`**.
**`SpaceTimeCNNDetector` was deliberately kept** (the owner's refinement): the 2026-09-10 history ablation that reorganised `TODO.md` §A was measured on it, and a decision is only reproducible while the code behind it is.
The four blocks it shares — `_gn`, `_SEBlock`, `_JointConvBlock`, `_MultiScaleBlock` — stayed with it, `_SEBlock` reached only through `_MultiScaleBlock`.

| Detector | Beam communication | Temporal fusion order | Cost | Best FROG wp-AUC |
| --- | --- | --- | --- | --- |
| `SpaceTimeCNNDetector` (A) | local, joint with time from layer 1 | joint | ~9-12 ms/frame | retracted (`performance-log.md`) |
| ~~`FullScanTCNDetector`~~ (B) **retired** | local, growing via dilation | time-then-space (causal TCN first) | ~2-3 ms/frame | retracted |
| ~~`TemporalUNetDetector`~~ (C) **retired** | global (bottleneck) + local (skips) | joint (T input channels) | ~2-3 ms/frame | retracted |

A vs. B is a deliberate ablation: does fusing space and time jointly from the first layer beat collapsing time first and mixing space afterward?
The accuracy ordering that used to sit in this table (A > B > C) is **retracted pending re-measurement** — every FROG number behind it carries at least one of the four 2026-09-10 measurement bugs, and A18 in particular hit A and B but not C, so the three were not even measured the same way.
The speed ordering (B ~ C < A) is unaffected: it is a timing measurement, not an accuracy one.

**All three set `BEAM_BATCH = True` and take `(B, N_beams, T, C)`.**
Every one of them convolves along the beam axis, so a flattened `(B*N_beams, T, C)` minibatch splices B frames into one long scan and lets GroupNorm statistics, the SE gate and the convolutions themselves leak between frames that share a batch.
A and B did exactly that until 2026-09-10; C never could, because it pools over the beam axis.
Measured before the fix, real val split at `dtime`=10: beam-level person AUC 0.9644 (batch 1) -> 0.9696 (4) -> 0.9765 (16) -> 0.9810 (64), because evaluation batches are consecutive frames and a bigger batch leaked more temporal context.
`tests/test_batch_independence.py` now asserts a frame's output is identical alone or inside a batch of 16, for every model and both heads.
Old checkpoints still load — shapes are unchanged — but compute different outputs.
Two earlier designs (`FullScanTransformerDetector`, `FullScanCNNDetector`) used global self-attention and a GRU respectively and were removed for violating the non-recursive constraint; see `rejected-ideas.md` for why they were also worse, not merely disqualified on principle.

## Source of truth

`DETECTOR_REGISTRY` (`library/takheliped/detectors/__init__.py`) is the authoritative list of what detectors exist and their canonical `--detector`/CLI key.
A detector class not in this dict cannot be trained or evaluated through `train.py`/`evaluate.py`, regardless of whether the class itself exists — absence from the registry means "not wired up," not "does not exist."
**TAKHeLiPeD is the standing exception** and is deliberately absent: it is stateful across frames, so it does not fit the registry's stateless `(scan, T) -> detections` contract, and it is trained and scored by `utils/train_takheliped.py` instead.
`library/takheliped/utils/torch_utils.py` looks like it should own device selection and does not — see `gotchas.md`.

## Components

- **`takheliped.py`** — TAKHeLiPeD: `beam_features`/`sanitize_ranges`, `CalibrationNetwork` (single-frame; `fold_legacy_calibration` carries older checkpoints over; `input_features="lfe"` swaps the ten beam features for LFE's one `lfe_features` channel, A59's attribution arm F1 only), `ObjectMemory` + `Slots`/`SlotRules`/`manage_slots`, and `calibration_network()`/`object_memory()` (in the trainer), which rebuild either stage the way its checkpoint was trained -- stage 2's range limit, input and widths through `network_shape`. Not in `DETECTOR_REGISTRY` — it is stateful and has its own trainer.
- **`architectures.py`** — `DrSpaamDetector` (cutout + auto-regressive spatial attention).
- **`drow_detector.py`** — `DrowDetector` (cutout + fixed temporal sum); the only detector the ROS deployment currently runs (`deployment.md`).
- **`full_scan.py`** — `SpaceTimeCNNDetector`; this project's own first novel design, and the only survivor of the three this file held until item 32.
- **`li2former.py`** — `Li2FormerDetector` (cutout + temporal Transformer); no published weights exist, and training has never completed on this project's hardware (`gotchas.md` I4).
- **`lfe_detector.py`** — `LFEPeaksDetector`/`LFEPPNDetector`, ONNX inference only, single-scan (no temporal window at all); the closest prior art to Architectures A-C.
- **`algorithmic_detector.py`** — wraps the C++ `AlgorithmicDetector` (rule-based leg+chest clustering + tracking); the only detector with no confidence score, so it reports F1/precision/recall instead of AUC.
  **The algorithm is prof. O. Aycard's, not this project's** (`cpp_core/sources/detector.cpp`, "Written by O. Aycard") — credit it as his wherever it is reported; only the binding, the wrapper and the harness are ours (`performance-log.md` footnote 0).
- **`odometry_estimation.py`** — trimmed-ICP scan matching, used only as a fallback when a dataset has no real odometry file; see `data-model.md` for when that fallback actually triggers.
- **`tracking.py`** — `SimpleTracker`, the SORT-style post-hoc tracker described in the lifecycle above.

## Tunables

| Tunable | Value | Read in | What it does |
| --- | --- | --- | --- |
| `time_frame` (`T`) | 5 (default) | `train.py --time-frame`, dataset `get_scan()` | Number of scans in the temporal window. |
| `dtime` | 1 (default) | `train.py --dtime`, `get_scan()` | Real-time stride between the `T` frames — `dtime=5` was FROG's measured accuracy peak on the pre-fix pipeline, since retracted (`performance-log.md`); a wider re-sweep (`{1,5,10,15,20,25}`) on the now-fixed real-odometry pipeline is in progress (`TODO.md` A1), since the old "wider regresses" finding (`rejected-ideas.md`) was caused by uncompensated ego-motion that real odometry now corrects for. |
| `align_scans` | on (default) | `train.py --no-align-scans`, `_extract_input()` | Whether historical frames are odometry-rotation-corrected before extraction. |
| `SimpleTracker.match_radius` | 0.5 m (fine stride) | `tracking.py`, `evaluate_auc(tracker_kwargs=...)` | Max distance to associate a detection with a predicted track position — a single global value, known not to generalize across invocation rates (`rejected-ideas.md`). |
| `SimpleTracker.min_hits` | 3 (default) | same | Real matches required before a track is reported at all. |
| `LFEPeaksDetector.merge_radius` / `LFEPPNDetector.nms_radius` | **0.30 m** both | `lfe_detector.py`'s `_merge_nearby()` | How close two centroids must be to count as one person. Calibrated against each detector's published AP at *both* association distances, not taken from the paper's sentence — which implies the 0.8 m person diameter and is ~3.5pp low for LFE-PPN and ~6pp low for LFE-Peaks. Changing either invalidates every false-positive number in the repo; that has already happened once (`CHANGELOG.md` 2026-09-11). |
| `merge_radius_slope` / `nms_radius_slope` | **0.0** (default = the single scalar above) | same | Makes the radius range-dependent, `r(d) = radius + slope * d`, evaluated at the pair's mean range and floored at zero. Exists because Tier 0 measured one scalar failing in both directions at 0.30 m — ~5 pp of LFE-Peaks' recall spent merging adjacent people, while LFE-PPN emits 1.609 duplicates per frame (`static-detector-diagnosis.md`, `TODO.md` A38). `0.0` is bit-identical to the old behaviour and is guarded by `tests/test_merge_radius.py`. |
| session-split threshold | 300 s | `frog_dataset.py`'s `_load_h5()` | Any real timestamp gap wider than this starts a new sequence, so temporal windows never cross it — chosen to isolate FROG's one genuine ~24.6h recording-session boundary without splitting on the 61 sub-90s in-session pauses. |
| `dropout` | **0.5** (`train.py --dropout`, `_default_args()`) | `_build_model()` | Applied uniformly to every trainable detector — `drow`, `drspaam`, `li2former`, and all three full-scan models alike — via `_build_model()`'s single `dr = getattr(args, "dropout", 0.5)`, which overrides each class's own constructor default (`full_scan.py`'s three classes default to 0.1 when instantiated directly, bypassing `train.py`). §5.4 of `docs/RESEARCH.md` documents 0.1 as the *design intent* for the novel full-scan architectures specifically — that intent was never wired into `train.py`'s actual default, so every full-scan training run through the CLI has used 0.5, not 0.1 (verified by reading `_build_model()` and `_default_args()` directly, 2026-09-10 — an earlier memory note claiming "0.1, light regularisation" was wrong, based on checking only the class constructor, not the CLI override). 0.5 exactly matches the official DR-SPAAM/DROW training recipe's own `dropout: 0.5` (`VisualComputingInstitute/2D_lidar_person_detection`'s `base_dr_spaam_drow_cfg.yaml`, fetched directly 2026-09-10) — so this project's dropout is not under-regularized relative to the papers it ports. |
| `weight_decay` | **1e-4** (`train.py --weight-decay`) | `_make_optimizer()` | The official DR-SPAAM/DROW repo's own optimizer (`dr_spaam/pipeline/optim.py`) uses plain `Adam(..., amsgrad=True)` with **no weight decay at all** — this project's default is already *more* regularized on this axis than the papers it ports, not less. Neither this project nor the official recipe (`augment_data: False` in the same yaml) applies data augmentation by default. |
