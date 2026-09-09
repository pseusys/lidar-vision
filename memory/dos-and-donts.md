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
`deploy/follow_the_drow/CMakeLists.txt` does `find_package(FollowTheDrow CONFIG REQUIRED)` and links `algorithmic_detector` against that compiled library; `deploy/follow_the_drow/src/algorithmic_detector.cpp` is a 54-line ROS-node wrapper that constructs and calls the shared `follow_the_drow::AlgorithmicDetector` class, not a second copy of it.
A `library/cpp_core/sources/` change reaches the robot on the next `make build-lib` + catkin rebuild, with nothing to keep in sync by hand.

**`deploy/follow_the_drow/nodes/drow_detector.py` does duplicate one thing: its vote-clustering hyperparameters.**
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

**DO write the prediction before running the test.**
An analysis run first and interpreted afterwards will find something.
This project has produced at least three findings that looked strong or plausible at first and did not survive an independent check: a verification script's own sign error, a 10x arithmetic slip in a dtime-to-distance extrapolation, and roughly half of all real-data timestamp gaps producing a plausible-*looking* but wrong odometry delta.

**DO replicate on an independent slice** before acting on any per-segment result.
This project's own convention: check both the train and test fold, not test alone — the dtime sweep (`docs/RESEARCH.md`) was only trusted once train ≈ test ruled out overfitting as the explanation.

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
`library/follow_the_drow/include/` (datasets + bundled weights, populated on install), `checkpoints*/`, `results/`, `runs/`, `utils/plots/`, `utils/*.pth`, and any ad hoc scratch directory for downloaded weights are all gitignored — there are no secrets in this project, the reason is size and reproducibility, not leakage.
