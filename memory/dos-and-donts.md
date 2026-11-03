# DOs and DON'Ts

*keywords:* coding standards, conventions, workflow, markdown, linting, evidence, blast radius, what not to commit

House rules for working in this repo.
Most exist because breaking them has already cost something here.

---

## Code

These are the workflow rules — what to change and when.
How the code itself is written, per language, is in [`coding-guidelines.md`](coding-guidelines.md), along with the linter that enforces each set.

**DO prefer editing over adding.**
Extend an existing function rather than creating a new one, unless the abstraction is used in at least two places.

**DO delete code eagerly.**
Whenever a piece of code becomes obsolete, legacy, no longer used: it should be deleted.
Note, that deleting it does not mean removing it forever: the code will stay in git history and can be restored any time.

**DON'T abstract prematurely.**
Three similar lines beat a helper used once.

**DON'T add defensive padding.**
No error handling for scenarios that cannot happen.

**DON'T add comments to code you didn't touch.**
Add one only where the logic is genuinely non-obvious.

**DON'T leave backward-compatibility shims.**
Remove deleted code completely — no `# removed` markers, no re-exports, no `_old_` aliases.

**DO keep long narrative out of code.**
A docstring says what the thing does and points at `memory/` for why.
When you find yourself writing the third paragraph, it belongs in a doc file.

**The ROS build does *not* duplicate `AlgorithmicDetector`'s C++ source — checked directly, not assumed.**
`deploy/takheliped/CMakeLists.txt` does `find_package(Takheliped CONFIG REQUIRED)` and links `algorithmic_detector` against that compiled library; `deploy/takheliped/src/algorithmic_detector.cpp` is a 54-line ROS-node wrapper that constructs and calls the shared `takheliped::AlgorithmicDetector` class, not a second copy of it.
A `library/cpp_core/sources/` change reaches the robot on the next `make build-lib` + catkin rebuild, with nothing to keep in sync by hand.

**`deploy/takheliped/nodes/drow_detector.py` does duplicate one thing: its vote-clustering hyperparameters.**
It imports the real shared functions (`cutout`, `prepare_prec_rec_softmax`, `votes_to_detections` from `drow_utils`), so the *logic* isn't forked — but its `RESULT_CONF` dict hardcodes its own `blur_sigma`/`bin_size`/`vote_collect_radius`/`min_thresh`/`class_weights` values, independent of whatever `evaluate_auc`'s own `v2d_conf` defaults are on the training/eval side.
A hyperparameter re-tune landing in `utils/`/`library/` does not reach this node automatically; see `TODO.md`.
See `deployment.md` for what is and isn't currently wired to the robot.

**DO run the cheap check after editing**, and the full suite before committing.
Both are in [`commands.md`](commands.md), along with the linter for each language.

**DO follow the development routine in [`../AGENTS.md`](../AGENTS.md)** for anything feature-sized: plan, confirm, tests first, implement, work out what was invalidated, re-run the cheap part and file the rest, test, lint, report.

After every change, scan imports, function parameters and comments for staleness, and grep for symbols you deleted.

---

## Documentation

The docs are load-bearing, so they get linted like code.

**DO write one sentence per line.**
No hard wrapping at a column.
A changed sentence then produces a one-line diff instead of a reflowed paragraph.

**DO keep it short.**
One precise sentence beats three thorough ones.
Give a rule its reason only where the reason changes what someone does, and cut restatements, second examples and closing aphorisms.
When a doc grows, look for what to delete before what to add.

**DO keep the two indexes in sync with the files.**
Adding a `memory/` doc is three edits in one commit: the file, a row in [`README.md`](README.md), and a row in [`keywords.md`](keywords.md).
Deleting one is the same three in reverse.

**DO give every `memory/` doc a `*keywords:*` line** directly under its title, holding the symbols someone would grep.

**DO verify before committing a documentation change:**

```bash
python memory/scripts/verify_memory.py
```

It checks links, both indexes, and the markdown rules below.
Structural problems fail the run; style problems warn unless you pass `--strict`.
CI runs the plain (non-strict) form — `docs/RESEARCH.md` predates the one-sentence-per-line convention below and is out of scope for a full reformat, so its style warnings are expected and don't block a push; `memory/`, `AGENTS.md`, `README.md`, `TODO.md`, and `CHANGELOG.md` should stay clean of them.

The full rule set is in [`../.markdownlint.jsonc`](../.markdownlint.jsonc), for editors and for `markdownlint-cli2` if Node is available.
`verify_memory.py` enforces the subset that matters with no dependencies at all.
The rules worth knowing without opening the config:

- ATX headings (`##`), never underlines
- table pipes padded with single spaces, delimiter rows written `| --- |`
- the `*keywords:*` label italic, the terms after it plain, so the line is not read as a heading
- fenced code blocks with backticks, always with a language tag
- dashes for bullets, two-space indent for nesting
- one blank line between blocks, never two
- no trailing whitespace, no tabs, newline at end of file
- no line-length limit, because sentences are the unit

---

## Evidence

**DO follow the internal rules in [`../AGENTS.md`](../AGENTS.md).**
They live there, next to the responding rules, so both are in the file every agent reads first.

**DO record rejections**, measured or on design grounds.
A tried-and-rejected idea goes in [`rejected-ideas.md`](rejected-ideas.md) with what was measured, so it is not re-argued six weeks later.

**DON'T report a change as working on the strength of the check you ran to build it.**
Name what would have failed if it were wrong, and run that.
A synthetic unit test passing has already been mistaken for "the logic is correct" twice in this project's real history — once for a scan-alignment sign error, once for the odometry timestamp-gap bug — and both were only caught by checking against real recorded FROG data, not by any test that was green beforehand.

**DON'T let a test size be chosen only for speed: check it against the configuration's own limits.**
A size picked to keep the suite fast can sit on the safe side of a cap, and then the cap is the one thing the suite cannot
falsify. Every case in `tests/test_li2former_port.py` ran 16 beams; Li2Former's `NUM_CTS` caps the beams at 90 by sizing
its position encoding, whose `forward` slices rather than raises — so 24 green tests, a `ruff` pass and a clean import all
held while the real 450-beam DROW scan crashed on the first batch, three launches in a row (2026-11-01, item 31).
**Ask of every dimension a test fixes: is the real value larger, and is there a configured maximum between them?**
A cheap standing answer is one case at the real size — the new class runs a whole scan at 450, 541 and 720 beams, the three
sensors in this repository, and costs about five seconds.

**DO smoke-run a long job at `--subsample` before committing hours to it.**
One epoch on a percent of the data exercises the real loader, the real criterion and a real backward pass, and it reaches
the first batch in about a minute — against 13 minutes per failed attempt for the full launch, three of which were spent
discovering a crash that a smoke run would have shown before the watchdog's first check (2026-11-01, item 31).

**DO mutation-test a guard whose red phase you never watched.**
A test that has only ever been seen to pass is a claim, not a check.
Three behavioural guards in `tests/test_upstream_scoring.py` were written before the implementation but first
*run* after it existed, so each was verified by breaking the implementation on purpose and re-running: two failed as they
should, and **one passed with the guard it exists to protect deleted** -- numpy raised `ValueError` from a reshape inside
the upstream file by itself, so the test had been pinning numpy's accident rather than our behaviour (2026-11-02, item 64).
It asserts the message now.
Deleting the line under test takes a minute, restoring it takes a `diff -q` against a copy, and it is the only way this
class of test is ever found.

**DON'T write a test that asserts a property of an untrained network's random draw.**
It measures the draw, not the behaviour, and the seed decides whether it passes.
This has happened three times here and cost a test each time: item 40's aggregator test asserted that concatenating a
whole-scan max would change a specific output, which an untrained draw does not guarantee (two versions were written and
both were wrong, so the test was deleted rather than tuned); item 42's first version asserted that two co-located slots
would *split* a candidate's attention, true only of the particular initialisation.
**Assert the guarantee instead.** For item 42 that is "waking a second slot takes weight away from the first", which
holds for any weights because a softmax's denominator grows — and which is also the property that *discriminates* the
mechanism, since a candidate-axis softmax would leave the first slot's weight untouched.

**DON'T clamp a sentinel index without checking a mask discards the result.**
`manage_slots` writes `-1` for "this slot got no candidate", and `.clamp(min=0)` turns that into a read of **element
0** -- which `decode_candidates` makes the frame's *strongest* candidate, because it sorts by descending score. So the
failure mode is not a crash or an obvious garbage value: it is the most plausible-looking wrong answer in the batch,
which is why item 54 survived from E7 until someone read the code against a diagram.
**The question to ask at every such site is "what discards this?"** -- a `where(mask, …)` around the result, a
multiplication by the mask, or a filter on `matched` before indexing. Item 54's two sibling lines had one and it did
not. A 2026-10-28 audit of all ten sites in the detector found exactly one unguarded, and the list is in
`CHANGELOG.md` so it does not have to be rebuilt.
**In NumPy this class is worse than in torch**: `-1` is a legal index that silently wraps to the *last* element, with
no clamp needed and no error.

**DON'T read "the forward is finite" as evidence that the backward is.**
They are different properties, and for masked attention the backward is the fragile one.
Item 42 shipped four degenerate-frame tests -- no live slot, no valid candidate, an empty memory, a twelve-frame run --
and **all four asserted `isfinite` on the forward only**.
The arm then died at epoch 20 with finite outputs, finite parameters and NaN gradients in every parameter, which is a
failure mode none of those four tests could have seen.
Where a `softmax` over a masked axis, a `clamp`ed denominator or an `amax` can be degenerate, assert on `.grad` after a
`backward()`, not just on the output.

**DON'T measure a gradient through a head that starts silent.**
Two of this project's heads are deliberately inert at initialisation — the rescoring gate starts at zero, and item 46's
refinement head has a zero-initialised last layer — so `0 * correction` gives every parameter behind them a gradient of
**exactly zero** and `abs()` has no subgradient at a zero residual.
A connectivity test written against that passes on a disconnected graph.
Wake the head first (`rescore_gate.fill_(1.0)`, or a target offset from the prediction), and say in the test why.
Both mistakes were made and caught here, in items 46 and 42.

**DON'T change a calibrated constant without re-running every table downstream of it.**
Nothing in this repo notices, and the tables do not say which constant they were measured under.
`LFEPPNDetector.nms_radius` moved from 0.8 m to 0.30 m in the working tree — a *correct* recalibration, against the published AP at both association distances — and for as long as nobody re-ran the false-positive passes, `PAPER.md`, `performance-log.md`, `dataset-properties.md` and `CHANGELOG.md` all kept quoting LFE-PPN figures from before the move. They were wrong by up to 2x. It surfaced only because two documents disagreed with each other and someone chased it (`CHANGELOG.md` 2026-09-11).
The cheap habit: when a default changes, grep the docs for the numbers that default produces, before committing.

**DON'T write a matching shortcut that a future parameter change can invalidate silently.**
`phantom_analysis.py` counted true positives as `min(detections-near-a-person, people-present)` — exact whenever a detector emits no duplicates, nonsense when it emits many, and there is nothing in the expression to say which regime you are in.
It gave correct answers for months, then reported a 5.0% miss rate where a one-to-one assignment gives 19.3%, the moment a radius change made the detector duplicate-heavy.
Use `linear_sum_assignment` and pay the microseconds.

**DO write the prediction before running the test.**
An analysis run first and interpreted afterwards will find something.
This project has produced at least three findings that looked strong or plausible at first and did not survive an independent check: a verification script's own sign error, a 10x arithmetic slip in a dtime-to-distance extrapolation, and roughly half of all real-data timestamp gaps producing a plausible-*looking* but wrong odometry delta.

**DO replicate on an independent slice** before acting on any per-segment result — and **DO verify the slice is actually independent** before reading anything into the agreement.
The dtime sweep was trusted for weeks because "train ≈ test ruled out overfitting".
It did not: `split="test"` was loading the training file, so the two columns were two samples of one dataset, and even the val split was FROG's own per-frame holdout sitting 38 ms from its training neighbours (`data-model.md`, `TODO.md` A11-A12).
**Agreement between two slices is evidence of independence failing at least as often as it is evidence of generalization.** Measure the distance between them before quoting the agreement.

**DON'T trust a metric that moves when you change something that cannot affect the model.**
Batch size, evaluation `dtime`, checkpoint selection and split membership are harness knobs; if the number tracks one of them, the harness is what you are measuring.
Measured here: wp-AUC rose monotonically with evaluation batch size (beam-AUC 0.9644 at batch 1 to 0.9810 at batch 64) because the minibatch was folded into the beam axis, and fell 2.5pp when the evaluator silently used a different `dtime` than training did (`interpreting-evaluation.md`, `TODO.md` A15/A18).
Both looked like properties of the model for months.

**DO check that the reported number and the saved artefact are the same object.**
`train.py` computed its final AUC *before* reloading the best checkpoint, so for its whole history the printed number described a network that was then discarded — and the file it wrote was stamped with the stopping epoch rather than the epoch whose weights it held (`TODO.md` A16).

**No minimum-sample threshold is set for this project.**
Treat any DROW-native result as lower-confidence by construction: DROW's train split has ~17.8x fewer total person-annotation instances than FROG's (`interpreting-evaluation.md`), not because of an agreed cutoff below which a result doesn't count.

**DO record measured performance moves** with the control arm beside them, and the provenance that makes the row comparable to the row above it.
`performance-log.md` is where these accumulate; see its own convention for what "comparable" means here.

---

## Changing production behavior

**DO consider the blast radius.**
After any change, ask which stages it affects, and re-run only those.
The stages, in order: ingest -> preprocess (cutout / full-scan, odometry alignment) -> detect (classification + vote regression, or heatmap) -> cluster into detections -> (optional) track -> (ROS only) follow-me behaviour.
A change to preprocessing invalidates every downstream stage's cached numbers; a change to the detection head alone does not invalidate preprocessing.

**DON'T tune `deploy/conf.env` against the real robot and call it a measurement.**
Test a parameter change with `make launch-docker-local` first — nothing about the robot's network, sensor noise, or the RobAIR-specific FoV is reproducible on a laptop, but a crash or an obviously wrong topic name is cheaper to find there than on hardware.

**DO check whether a config knob has a code reader** before relying on it.
None in `deploy/conf.env` is known to be dead as of this writing — if you find one, name it here.

**The full rebuild sequence** is `pip install ./library` then `make build-lib`; see `commands.md`'s Install and build section.

---

## Working with the user

**DO chain long-running actions.**
Waiting on a permission prompt can cost hours.
Ask for the approved list to be extended when something you need is missing.

**DO write a script the second time you do a chore by hand.**
The rule and the conventions are in [`automation-scripts.md`](automation-scripts.md).

**Never commit datasets, weights, checkpoints, logs, or downloaded scratch files.**
`library/takheliped/include/` (datasets + bundled weights, populated on install), `checkpoints*/`, `results/`, `runs/`, `utils/plots/`, `utils/*.pth`, and any ad hoc scratch directory for downloaded weights are all gitignored — there are no secrets in this project, the reason is size and reproducibility, not leakage.
