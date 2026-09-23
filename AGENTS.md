# AGENTS.md — orientation for AI coding assistants

Read this first, then follow a pointer.
Depth lives in [`memory/`](memory/README.md), not here — this file is the map.

## Before you start

Reading everything every session is as wrong as reading nothing.
Read what this task makes relevant, before acting rather than after:

- **Scan the request against [`memory/keywords.md`](memory/keywords.md)** and open what it routes you to.
- **Before proposing an idea**, grep [`memory/rejected-ideas.md`](memory/rejected-ideas.md).
  One grep is cheaper than one investigation.
- **Before debugging anything environment-shaped** — encoding, paths, shell, DirectML, a library that will not import — read [`memory/gotchas.md`](memory/gotchas.md).
- **Before running a command you are inventing**, check [`memory/commands.md`](memory/commands.md) for the one that exists.
- **Before quoting an accuracy or speed number**, read [`memory/interpreting-evaluation.md`](memory/interpreting-evaluation.md).
- **Before calling a change done**, re-read the evidence rules in [`memory/dos-and-donts.md`](memory/dos-and-donts.md).

## Development routine

Every feature goes through these steps, in order.
Skipping one is a decision to be stated out loud, not a default.

1. **Plan.**
   Do not invent an answer to a question that would change the design; put the question in the plan instead.
2. **Present the plan and wait** for the user to confirm.
   A plan that was never shown is a plan nobody agreed to.
3. **Write the tests first**, so they fail before anything is implemented.
   A test written afterwards tests the implementation, not the requirement.
4. **Implement**, following [`memory/coding-guidelines.md`](memory/coding-guidelines.md).
5. **Work out what the change invalidated**: trained checkpoints, cached preprocessing, generated odometry sidecars, downstream evaluation numbers, and which tests cover the blast radius.
   Write the list down before running anything, to notice what you would otherwise skip.
6. **Re-run the cheap half of that list now, queue the expensive half.**
   Anything long, costly or irreversible (a retrain, a full dataset re-evaluation) becomes a `TODO.md` item with What / How / Why, in the tier matching when it can happen.
7. **Run the tests.**
8. **Run the linters** for every language the change touched — see the linter table in [`memory/coding-guidelines.md`](memory/coding-guidelines.md).
9. **Report**, in the shape below, naming: the feature, the tests added and whether they pass, which invalidated steps were re-run and what they returned, which were deferred and where, and what changed in `TODO.md`.

Documentation is updated in the same step as the change, not after step 9.

## What this is

A research project on person detection from knee-height 2D LiDAR data, structured as one shared pipeline with several interchangeable stages:

1. **Ingest** a raw LiDAR scan (or `T` consecutive scans) from a recorded dataset file or a live ROS topic.
2. **Preprocess**: either extract a fixed-width per-beam polar cutout (DROW, DR-SPAAM, Li2Former), or feed the whole raw scan to a full-scan CNN/TCN (this project's own architectures), optionally ego-motion-compensating historical frames with odometry.
3. **Detect**: per-beam classification + 2D vote-offset regression (the DROW-style head), or a Gaussian heatmap over beams (the LFE-style head) — either way, a per-beam confidence plus a spatial estimate.
4. **Cluster** the per-beam votes/heatmap peaks into discrete person/wheelchair/walker detections.
5. **(optional) Track**: a SORT-style tracker (`SimpleTracker`, training/eval side) or `PersonTracker` (the ROS node) turns per-frame detections into one persistent, ID-stable position.
6. **(ROS deployment only)** the follow-me behaviour consumes the tracked position.

**The project's own detector is `TAKHeLiPeD`** — **T**(emporal) **A**(daptive) **K**(nee-)**He**(ight) **Li**(dar) **Pe**(rson) **D**(etector), *TA-KHé-Li-PeD*, named 2026-09-23 and called *the three-horizon detector* in everything written before that date; the code still spells it `three_horizon`.
It does not fit the stage list above and is the standing exception to it: one scan per call, no `T`-scan window, no post-hoc tracker, and a learned object memory carried as state instead of stages 5-6.
It is absent from `DETECTOR_REGISTRY` and is trained and scored by `utils/train_three_horizon.py`, not `train.py`/`evaluate.py` — see [`memory/detector-architectures.md`](memory/detector-architectures.md).

**Storage** is files, not a database: datasets and published weights are downloaded into `library/follow_the_drow/include/` (gitignored, populated by `pip install ./library`); trained checkpoints go to `checkpoints*/` directories (gitignored); evaluation logs go to `results/` (gitignored).
**Working directory** is the repo root for most commands, but `cd utils` first for `train.py` / `evaluate.py` / `render_video.py` — their relative paths (`../checkpoints`, `../results`) assume it.
**Runtime**: Python 3.12, in the `.venv` virtualenv — activate with `.venv/Scripts/activate` (Windows) or call `.venv/Scripts/python.exe` directly.
**Secrets**: none.
This project has no API keys, tokens, or credentials anywhere in the repo; `deploy/conf.env` is committed runtime configuration (topic names, thresholds, node enable flags), not a secret.

## Where to look next

| I need to… | Go to |
| --- | --- |
| Know which docs this task needs | [`memory/keywords.md`](memory/keywords.md) |
| Know the house rules before editing | [`memory/dos-and-donts.md`](memory/dos-and-donts.md) |
| Write code in one of this project's languages | [`memory/coding-guidelines.md`](memory/coding-guidelines.md) |
| Understand a detector's architecture or the research rationale | [`memory/detector-architectures.md`](memory/detector-architectures.md) |
| **Discount an accuracy or speed number correctly** | [`memory/interpreting-evaluation.md`](memory/interpreting-evaluation.md) |
| See where this project stands against published SOTA | [`memory/performance-log.md`](memory/performance-log.md) |
| Compare a tuning sweep against past runs | [`CHANGELOG.md`](CHANGELOG.md) |
| Understand a dataset file format, or what a rebuild invalidates | [`memory/data-model.md`](memory/data-model.md) |
| Run something | [`memory/commands.md`](memory/commands.md) |
| Avoid a known platform or environment trap | [`memory/gotchas.md`](memory/gotchas.md) |
| Check whether an idea was already tried | [`memory/rejected-ideas.md`](memory/rejected-ideas.md) |
| Deploy to the robot, or understand what's running on it | [`memory/deployment.md`](memory/deployment.md) |
| Automate a task I am about to do by hand twice | [`memory/automation-scripts.md`](memory/automation-scripts.md) |
| Find when and why something changed | [`CHANGELOG.md`](CHANGELOG.md) (grep the `*keywords:*` lines) |
| Know what's open right now | [`TODO.md`](TODO.md) |

Full index: [`memory/README.md`](memory/README.md).

## Repository layout

```text
lidar-vision/
├── AGENTS.md / README.md / CHANGELOG.md / TODO.md
├── memory/                     ← the knowledge base (start at memory/README.md)
├── docs/RESEARCH.md            ← the research write-up: architecture design, citations, related work (mechanism only — results live in memory/, history in CHANGELOG.md)
├── deploy/conf.env             ← ROS runtime config, committed — no secrets in this project
│
├── library/                    ← the `follow_the_drow` Python/C++ package
│   ├── follow_the_drow/
│   │   ├── detectors/          ← nine detector implementations — memory/detector-architectures.md
│   │   ├── datasets/           ← DROW_Dataset, FROG_Dataset, JRDB_Dataset, LiveDataset — memory/data-model.md
│   │   ├── utils/              ← preprocessing, odometry estimation, tracking, torch device selection
│   │   └── include/            ← gitignored: datasets + bundled weights, downloaded on first install
│   ├── cpp_core/                ← C++ AlgorithmicDetector + pybind11 binding
│   └── setup.py                 ← downloads datasets/weights the first time include/ is missing
│
├── utils/                       ← train.py, evaluate.py, render_video.py, one-off research scripts
├── tests/                       ← pytest, CPU-only, no GPU or dataset needed
│
└── deploy/follow_the_drow/      ← the ROS Noetic package that actually runs on the robot — memory/deployment.md
    └── nodes/                    ← live_loader, file_loader, visualizer, algorithmic_detector, DROW_detector, data_annotator, tracker
```

## The rules that matter most

1. **Prefer editing over adding**; no premature abstraction, no defensive padding, no compatibility shims.
2. **Long narrative goes in `memory/`**, not in a docstring or in this file.
3. **Don't trust a cross-regime accuracy comparison at face value.**
   A "zero-shot transfer" number and a "natively retrained" number sit in the same table but are not the same measurement, and one AUC column has already been silently swapped for another once (`train_all.py` mislabeled the agnostic/any-class column as "Test AUC", inflating native-DROW numbers by 15-20pp).
   Caveats and regimes are in [`memory/interpreting-evaluation.md`](memory/interpreting-evaluation.md).
4. **A dataset-logic fix is not verified until it's checked against real recorded data, not just a synthetic unit test.**
   Sign errors and a session-boundary timestamp-gap bug both passed clean synthetic tests and were only caught by direct inspection of the real FROG files — see the evidence rules in [`memory/dos-and-donts.md`](memory/dos-and-donts.md).
5. **Nothing in `library/`/`utils/` reaches the robot until someone manually rebuilds and relaunches the Docker image.**
   There is no CI auto-deploy; the ROS pipeline only ever runs whatever was last built, and today that's `DrowDetector` alone — none of this project's newer architectures, odometry work, or tracker are wired into it.
   See [`memory/deployment.md`](memory/deployment.md) and `TODO.md`'s NOT DEPLOYED section.

Full versions of all of these: [`memory/dos-and-donts.md`](memory/dos-and-donts.md).

## Responding rules — how to report back

The owner (`pseusys`, the sole maintainer) reads every answer.
Long ones are expensive to read and bury the decision.
Rules 1-5 are hard; rule 6 is a preference — follow it by default, break it when there is a reason.

**1. Be precise, not verbose.**
Say the thing.
Cut the preamble, the recap of what you were asked, and the narration of what you are about to do.

**2. Use this shape, one paragraph per section, in this order.**
`Running` is one line, not a paragraph.
Anything less important goes *below* it, clearly marked as detail.

| section | content |
| --- | --- |
| **Done** | What you actually did. |
| **Results** | The numbers — **as a table or chart** wherever one fits. |
| **Needs attention** | What is wrong, risky, or awaiting a decision from the owner. Say plainly if there is nothing. |
| **Next** | What you suggest doing, and why that and not something else. |
| **Running** | **One line.** Every background job in flight at the moment of writing, with its progress. `none` if there are none. |

**2b. `wp-AUC` (person-class AUC at 0.5 m matching radius) leads any accuracy answer.**
Never substitute the agnostic/any-class AUC column for it — that exact substitution once inflated native-DROW results by 15-20pp (`train_all.py`'s mislabeled summary column).
Report `ms/frame` alongside it as the secondary axis whenever both exist for the row being discussed; a speed number without its accuracy counterpart, or vice versa, is an incomplete answer here.

**3. Ask early rather than guessing** when an answer would change what you build.
Do not let that suppress your own proposals — suggest freely, and say which option you would pick.

**4. Keep the session's plan in `TODO.md` §A as you go**, not at the end.
An idea agreed to and not written down is an idea lost.
Add the item the moment it is agreed, with What / How / Why filled in.

**5. Update the docs in the same step as the change**, not later.
[`TODO.md`](TODO.md) and [`CHANGELOG.md`](CHANGELOG.md) always, plus whichever of `memory/` the change touches, plus [`memory/keywords.md`](memory/keywords.md) if you added or removed a doc.
A finding that is not written down did not happen.

**6. Prefer to drain `TODO.md` §A before wrapping up a session.** *(Preference, not a rule.)*
§A is the working queue: put work there as soon as it is agreed — **including work you could do now but do not strictly need now** — and try to finish or re-tier it before you stop.
Items legitimately stay in §A when they are owned by someone else, blocked on an answer, or simply still queued.
The point is that §A reflects reality at session end, not that it is empty.

## Internal rules — how to work

Separate from the responding rules above: these govern what you do, not what you say.
Each is earned by a specific failure in this repo — the cost is in the "because" clause, and it is not hypothetical.

**I1. Stage explicit paths, never `git add -A`.**
*Because:* other sessions may share this working tree, and `-A` sweeps a parallel session's uncommitted work into an unrelated commit.

**I2. Never truncate output you have not read.**
Redirect to a file, then read the file.
*Because:* a `tail -80` on a long run discards the headline results, which then have to be reconstructed from secondary output — if they can be at all.

**I3. ASCII only in anything printed to a console.**
*Because:* a Unicode arrow (`→`) in a progress-print crashed `_download_file()` with `UnicodeEncodeError` on Windows' default cp1252 console, after a 189 MB file had already fully downloaded, and an earlier `--help`-text crash had the identical root cause.
Use `->` instead of `→`, plain words instead of check marks.

**I4. Don't spend more than one workaround attempt on a DirectML crash before considering it an environment limit, not a code bug.**
*Because:* `Li2FormerDetector` training hit three separate DirectML crash modes (allocator OOM at batch_size=2, an access violation, a segfault at batch_size=1) across in-process and subprocess attempts, and no code-side fix resolved any of them — the eventual conclusion was a driver/version limitation of `torch-directml==0.2.5.dev240914`, not a bug in this repo.
Check whether CUDA/ROCm is available first; if only DirectML is, and it crashes once, say so and move on rather than iterating blindly.

**I5. Use `.venv/Scripts/python.exe` directly (or activate `.venv`) — do not run bare `python3` or `make venv`/`make test` and assume they work.**
*Because:* on this machine `python3` resolves to the Windows Store's install stub, not a real interpreter, so `make venv`'s `python3 -m venv venv` does not create a usable environment — the actual working interpreter was created by hand as `.venv` (Python 3.12, required for ABI compatibility with the prebuilt C++ extension, see `memory/gotchas.md`).
Fixing the Makefile itself is `TODO.md`'s item A6; until then, the manual venv is the source of truth.
