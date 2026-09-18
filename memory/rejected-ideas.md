# Rejected ideas — do not re-propose without new evidence

*keywords:* rejected, tried, did not work, negative result, reopen, settled, consensus filter, dtime, wide window, tracker, attention, GRU

Decisions to NOT do something, recorded so they are not rediscovered and re-argued from scratch.
Open questions live in [`../TODO.md`](../TODO.md); this file is for settled ones.

Each entry states what was proposed, what was measured, and what new evidence would be needed to reopen it.

## A per-beam fine temporal convolution (the "calibration" denoiser) in the three-horizon detector (TODO.md A50)

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

## Excluding beams at the range limit from stage 2's training loss: a tie, not needed (TODO.md A50)

*keywords: A50, --limit-beams-in-loss, training_beams, calibration_loss, limit beams, missing returns, loss mask, open space negatives*

**What was proposed.** At decoding, beams at the range limit cast no vote, which gained +0.34 pp on all six checkpoints. So leave them out of training too.

**What was measured.** Three seeds, default architecture, FROG `official` test, default decoding: 77.68 / 76.72 / 76.58 (mean 76.99). Against old controls (78.40 / 78.28 / 77.09) it first looked like −0.93 pp. Controls retrained on the same code scored 78.19 / 76.64 / 75.13 (mean 76.65): **a tie, +0.34, inside the ~1 pp per-run spread.**

**Reading.** No benefit, so not worth an option: the flag was removed 2026-09-17. Since seeds do not reproduce runs here, the first "lower on every seed" reading was noise.

**What would reopen it.** A dataset where limit beams are rare enough that the class balance is not affected, or a variant that keeps them as negatives for classification and drops only their vote loss.

## Median-based temporal denoising of range scans, replacing the fine convolution (TODO.md A50)

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
| detection-frequency per cell, exponentially forgetting (A41) | 0.57 |
| background model from raw scan returns, keyed on hit rate (A44) | 1.90 (LFE-PPN 3.32) |
| trail-length filter, *acausal oracle* (A45) | 2.54 |
| displacement filter, *acausal oracle* (A45) | 4.20 |
| *SORT, for reference* | *3.8* |
| *merge radius, for reference* | *21.9* |

**Why rejected.**
Only the acausal displacement oracle beats SORT, and it is unreachable by construction and still 5x below a merge-radius tweak.

The 89.7% was measured on **person-free** frames, where a filter cannot destroy a true positive. It bounds *removal*, never *discrimination*, and the cost column turns out to be most of the story.

The deeper reason is structural and shows directly in A44's table: at a 30 s minimum observation span only **11.3%** of phantoms qualify, and at 100 s none do. **On a mobile robot, how long something has been at a location is dominated by how long the robot looked at it.** A museum robot drives past a chair rather than watching it, so world-location persistence is largely a statement about the robot's own transit -- and short-lived is exactly what a person is too.

**What would reopen it.**
A **stationary or slowly-patrolling** platform, where per-location observation span is set by the scene rather than by the robot's path -- the premise fails on mobility, not on the physics. Or an **object-keyed** displacement feature rather than a location-keyed persistence one: that is the only variant measured above SORT, and it belongs to role B (`docs/PROPOSAL.md`).

**What this does not reject.** The horizon argument itself. 786 frames is still the span over which the separating evidence exists; what is rejected is reaching it through location.

## Range-aware merge/NMS radius, `r(d) = a + b*d` (TODO.md A38)

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

**Confirmed on the operating-point metric too (A40, same day).** The re-run scored against recall, duplicates, phantoms and close-pair misses at threshold 0.3 rather than AP. At matched recall the constant still wins: LFE-Peaks at 76.9% recall costs 1.380 FP/frame constant against 1.445 range-aware, LFE-PPN at ~85.8% costs 3.001 against 3.079. The single comparison it does not lose is LFE-PPN at `d = 0.3 m`, where the two are within noise. **It loses or ties on both metrics, both detectors and both association distances, and never wins.**

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
