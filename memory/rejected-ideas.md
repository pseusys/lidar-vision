# Rejected ideas — do not re-propose without new evidence

*keywords:* rejected, tried, did not work, negative result, reopen, settled, consensus filter, dtime, wide window, tracker, attention, GRU

Decisions to NOT do something, recorded so they are not rediscovered and re-argued from scratch.
Open questions live in [`../TODO.md`](../TODO.md); this file is for settled ones.

Each entry states what was proposed, what was measured, and what new evidence would be needed to reopen it.

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
