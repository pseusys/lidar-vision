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

### A1. Re-run the `dtime`/`time_frame` sweep on the fixed real-odometry pipeline — screen before committing full runs

**IN PROGRESS 2026-09-09**: phase 1a's first point (`dtime=15`) launched, was interrupted mid-epoch-5 when the agent session restarted (4 epochs had already completed and checkpointed cleanly — `val_loss` still improving at epoch 4, 0.2122->0.1958->0.1727->0.1657 — so resumed from that checkpoint with `--epochs 4` rather than restarting from scratch, to finish at exactly the planned 8-epoch budget instead of losing the completed work).
If this run is found stopped again with no completion record, check `checkpoints_frog_dtimesweep_screen/spacetime_cnn_dtime15.best.pth`'s stored `epoch` before deciding whether to resume or restart — a background job here does not survive an agent session restart on its own.

**What.** Two swept axes on `SpaceTimeCNN`/FROG, using the now-fixed `FROG_Dataset` (real per-session odometry + correct session splitting, `CHANGELOG.md` 2026-09-08; `scan_time` precision, 2026-09-09):
- `dtime` in `{1, 5, 10, 15, 20, 25}` at the current default `time_frame=5` (widened from the originally-planned `{1, 5, 10}`).
- `time_frame` (`T`, frame count given to the model) in `{1, 3, 10}` — `T=5` is already covered by the `dtime` sweep above — added 2026-09-09 per the owner's suggestion, to compare against LFE's single-frame design (`T=1`) directly.

Both axes together, at this dataset's scale, could plausibly cost ~15h+ cumulative if run in full (today's DR-SPAAM eval alone is running ~9-10h) — too expensive to commit to blind, so this is three phases, not one sweep.

**How, phase 1a — `dtime` screening at `T=5`.** `python train.py --detector spacetime_cnn --dataset frog --subsample 0.2 --epochs 8 --patience 3 --dtime N`, one run per `dtime`, order `15, 1, 5, 10, 20, 25` (middle of the range first, so early results already bracket where the peak might be).
`--subsample 0.2` (~1/5 of train) + `--epochs 8 --patience 3` (short) — this phase ranks `dtime` values relative to each other, it isn't meant to produce a trustworthy absolute AUC.
Take the winning `dtime` (call it `dtime*`) forward to 1b; its window span `S = 4 * dtime* * 38.15ms` is the reference span for 1b.

**How, phase 1b — `time_frame` screening, span-matched.** Sweeping `T` at a *fixed* `dtime` would conflate "more frames" with "a much longer total window" (span = `(T-1) * dtime`), which is exactly the axis the old `T=10`/wide-`dtime` regression sits on (`memory/rejected-ideas.md`) — so match the span from 1a instead of reusing `dtime*` directly:
- `T=1`: one standalone run, `dtime` irrelevant (a single frame has no stride to speak of).
- `T=3`: `dtime ≈ round(S / (2 * 38.15ms))`.
- `T=10`: `dtime ≈ round(S / (9 * 38.15ms))`.

Same cheap settings as 1a (`--subsample 0.2 --epochs 8 --patience 3`).

**How, phase 2 — full runs on the best candidates.** Take the best 2-3 `(T, dtime)` combinations across 1a+1b and retrain each at full scale (`--subsample 1.0 --epochs 30 --patience 5`, this project's normal defaults), then `evaluate.py` for the real number.
Compare against `memory/performance-log.md`'s existing (retracted) real-odometry numbers and the per-frame motion sanity check there.

**Why.** Every real-odometry number on file predates at least one of the two session-boundary bugs fixed 2026-09-08 — the `dtime` half of this is the direct, previously-deferred follow-up.
`dtime` widened past `10` (now to `25`) because the 2026-09-09 sanity check found real per-frame human displacement scales linearly at ~1.7cm per unit of `dtime` (measured: 2cm at `dtime=1`, ~35cm — close to half a walking stride — at `dtime=20`; `25` extrapolates to ~43cm), and because the mechanism that made the *old* wide-`dtime` attempt regress (uncompensated ego-motion) is exactly what real odometry now corrects for.
`time_frame` added because `T=1` is a genuine, cheap ablation directly comparable to LFE's single-frame design (confirmed technically sound by reading `SpaceTimeCNNDetector`'s code: the temporal conv branches use same-padding on the time axis, so arbitrary `T` runs without changes — at `T=1` they degenerate toward single-frame behavior by construction, not a special case).
Span-matching 1b rather than fixing `dtime` keeps "more frames" and "longer window" as separable questions instead of one confounded axis.
The screening phases exist so widening the search doesn't multiply full-run cost by the number of points tested — a `--subsample 0.2`/`--epochs 8` run should cost a small fraction of a full one.

### A2. ~~Evaluate DR-SPAAM published weights~~ — **DONE (settled) 2026-09-09: cite the paper, don't replicate**

Both T=1 (73.3%) and T=5 (75.3%) now cite the FROG paper's own Table 4 numbers in `memory/performance-log.md`, rather than re-deriving them through this repo's own (twice-failed, once by decision and once by an unexplained stall — see `CHANGELOG.md` 2026-09-09) evaluation pipeline.
The official-weights compatibility check (zero missing/unexpected keys loading into this repo's `DrSpaamDetector`) already passed independently and stands regardless.
Explicit owner call: effort goes toward this project's own novel results (`A1`), not toward reproducing a number the paper already reports.
Not reopening unless a specific reason to distrust the paper's number surfaces.

### A3. Decide on a DROW3 published-weights evaluation path

**What.** `drow_on_frog.pth`'s parameters don't match the existing `DrowDetector` class — it's the same conv-block/gate family as DR-SPAAM, minus the `gate` submodule.

**How.** Would need a new or adapted architecture class to load it.

**Why.** Same goal as A2, for `DrowDetector`'s row; not started, needs an owner decision on whether it's worth a new class for one checkpoint.

### A4. Decide on a PeTra evaluation path

**What.** `petra_frog.h5`/`petra_frog_mixedloss.h5` weights exist but need an entirely separate evaluation harness — PeTra's task framing (256x256 occupancy-grid segmentation + external leg-pairing) doesn't fit the shared beam-classification pipeline.

**How.** Not scoped yet.

**Why.** Same goal as A2/A3; a bigger lift than either.

### A5. Vectorize `LFEPPNDetector`'s decode loop

**What.** Replace the nested pure-Python `(sector x anchor)` loop in `detect()` with a vectorized NumPy threshold-mask, matching `LFEPeaksDetector`'s single vectorized call.

**How.** `library/follow_the_drow/detectors/lfe_detector.py`.

**Why.** The ~150x speed gap to LFE-Peaks is a decode-loop artifact, not architectural (`memory/interpreting-evaluation.md`) — a low-risk fix that doesn't touch weights or accuracy.

### A6. ~~Fix the Makefile's `venv`/`test` targets~~ — **DONE 2026-09-08**

`PYTHON`/`VENV_PYTHON` variables added, OS-branched (`$(OS)`): prefers bare `python` on Windows (`python3` there resolves to the Store install stub, which `command -v` finds even though it does nothing — `memory/gotchas.md`), prefers `python3` on Linux/CI (the tested path, since `ubuntu-latest` may not alias `python`).
`test`/`venv` now invoke `$(VENV_PYTHON)` explicitly rather than relying on `PATH` picking up the right `venv/bin` vs `venv/Scripts`; `venv` also now installs `utils/requirements.txt` (previously missing — `pytest` was never installed by `make venv` at all, so `make test` couldn't have worked even once `python3` resolved).
Verified with `make -n test` (dry run): resolves to the real `/c/Python314/python`, correct `venv/Scripts/python` invocations.

**Residual, real limitation — see A9 below**: `python` on this machine is 3.14, not 3.12, so `make venv` will very likely still fail at the C++ extension build/import step (`memory/gotchas.md`'s ABI-mismatch entry) — the underlying `python3`-doesn't-resolve bug is fixed, but a full `make venv` run has not been exercised end-to-end since a working Python 3.12 isn't `python`/`python3` on this machine.

### A9. Make the Makefile check for a working Python 3.12, not just *a* Python

**What.** `make venv`'s new `PYTHON` selection (A6) picks up whatever `python`/`python3` resolves to — on this machine that's 3.14, which will fail the C++ extension's ABI requirement (`memory/gotchas.md`) the same way the old bare `python3` call would have, just further into the run instead of immediately.

**How.** Either add a version check in the `venv` target (fail fast with a clear message if not 3.12), or support `make venv PYTHON=<path-to-3.12>` and document it in `memory/commands.md`.

**Why.** A6 fixed "the command doesn't exist"; this fixes "the command exists but is the wrong version" — a different failure mode with the same symptom (`make test` doesn't work), not yet addressed.

### A10. Stop hardcoding `RESULT_CONF` in the ROS `drow_detector.py` node

**What.** `deploy/follow_the_drow/nodes/drow_detector.py` imports the real shared preprocessing/postprocessing functions (`cutout`, `votes_to_detections`, etc.) but hardcodes its own `RESULT_CONF` dict (`blur_sigma`, `bin_size`, `vote_collect_radius`, `min_thresh`, `class_weights`) independent of whatever the training/eval pipeline's own `v2d_conf` defaults are.

**How.** Either read these from a shared constant/config the training pipeline also uses, or explicitly document why the ROS node's values are deliberately different (e.g. tuned for real-time robot use vs. offline evaluation) rather than leave it looking like an oversight.

**Why.** A vote-clustering hyperparameter re-tune landing in `utils/`/`library/` does not reach this node automatically, and nothing currently flags that it should be checked (`memory/dos-and-donts.md`).
Correction: an earlier pass through this codebase also flagged the C++ `AlgorithmicDetector` as duplicated between `library/cpp_core/` and `deploy/follow_the_drow/` — checked directly and that claim was wrong (the ROS build links against the compiled library via `find_package(FollowTheDrow)`, it doesn't recompile a copy); `memory/dos-and-donts.md` and `memory/coding-guidelines.md` have been corrected.

### A7. Clean up `scratch_frog_weights/`

**What.** ~79 MB of downloaded published `.pth` weight files sitting in the repo root (gitignored, but not the designated scratch location).

**How.** Relocate to the proper scratch directory or delete, once A2-A4 finish reading from it.

**Why.** Repo-root clutter.

### A8. Fetch the remaining real FROG odometry files

**What.** `frog_16-41_odom.npz`, `frog_10-31_odom.npz`, `frog_14-57_odom.npz`, `frog_15-53_odom.npz` (the four sessions not inside `train_val.h5`) aren't placed in the real dataset directory yet — only the two sessions inside `train_val.h5` are.

**How.** `FROG_Dataset.download()` already fetches them automatically; run it against the other split files.

**Why.** Completeness, and needed before A1's sweep can use real (not estimated) odometry for every session it touches.

### Project alignment plan (AL1-AL6, in this order)

Requested 2026-09-08: bring the codebase up to `memory/coding-guidelines.md`'s stated rules, with real enforcement behind each one, plus close the test-coverage and CI gaps found while writing that file.
Ordered by dependency — each step assumes the ones before it are done, and doing them out of order means redoing work (a language-specific config written before the shared root config exists gets rewritten once AL2 lands).

### AL1. Add a CI job that runs the test suite

**What.** `pytest tests -q` (63 tests) currently only runs by hand — there is no GitHub Actions job for it at all.

**How.** A new job (its own workflow, or added to `build-lidar-vision-image.yml`), `paths`-scoped to `library/**`, `utils/**`, `tests/**`; `actions/setup-python@v6` pinned to `python-version: '3.12'` (matching the ABI requirement, `memory/gotchas.md`), then `pip install -e library && pip install -r utils/requirements.txt && pytest tests -q`.

**Why.** Nothing currently catches a regression in `library/`/`utils/` automatically — the 63 tests this project already has are only as good as someone remembering to run them.

### AL2. Move the `ruff` config to the project root

**What.** `memory/scripts/ruff.toml` currently governs only `memory/scripts/`; once `library/`/`utils/` are linted too (AL3), two separate configs would have to be kept in agreement by hand.

**How.** Create `ruff.toml` at the repo root (a plain root file is simpler here than adding `[tool.ruff]` under `library/pyproject.toml`, since `utils/` isn't part of that package), delete `memory/scripts/ruff.toml`, and update its four references: the Python section and linter table in `memory/coding-guidelines.md`, the lint command in `memory/automation-scripts.md`, and the lint command in `memory/commands.md`.

**Why.** "Two configs that must agree will not" — do this before AL3, not after, so there is only ever one config to write in the first place.

### AL3. Set up `ruff` enforcement for `library/` and `utils/`

**What.** This project's own Python — by far the largest surface in the repo — has no linter running against it at all yet.

**How.** Run `ruff check` with AL2's root config against `library/` and `utils/`; fix what's cheap, add narrow `per-file-ignores` (each with a comment naming the reason, per `ruff.toml`'s own convention) for anything that's a deliberate existing style choice rather than an oversight; add a `paths`-scoped CI job once the tree is clean.

**Why.** `memory/coding-guidelines.md` already states the intended rules; this is where they become real. This run doubles as the "does the existing code actually match the guidelines" check — no separate pass needed.

### AL4. Add `shellcheck` for `deploy/docker/entrypoint.sh`

**What.** The one shell script in the repo isn't linted.

**How.** Create `.shellcheckrc` at the repo root (`shell=bash` / `enable=all`), fix whatever it flags in the 3-line script, add a CI job.

**Why.** Per `memory/coding-guidelines.md`'s own rule, a linter config lands alongside the first file it lints, not speculatively — the file already exists, so this is due now.

### AL5. Add `hadolint` for `deploy/docker/Dockerfile`

**What.** The Dockerfile isn't linted.

**How.** Create `.hadolint.yaml`, run hadolint, and fix or deliberately (and visibly, with a comment) ignore each finding.
**Do not** blindly reuse a generic ignore list for version-pinning rules (`DL3007`/`DL3008`/`DL3013`) — those make sense for a disposable eval image, not for the image this project actually deploys onto a physical robot, where pinned versions are generally the safer default.

**Why.** Same "config alongside the first file" reasoning as AL4.

### AL6. Add at least one smoke test per detector class

**What.** `tests/` covers geometry, odometry estimation, session-splitting, and augmentation — no detector (`DrowDetector`, `DrSpaamDetector`, `SpaceTimeCNN`, etc.) has even a basic forward-pass/shape test.

**How.** One lightweight test per class: construct with small/default params, feed a synthetic scan, assert the output shape — the novel architectures need no real weights or dataset for this; the published-weight detectors can use their bundled weights.

**Why.** A detector-level regression (a shape bug, a crash on construction) currently has no automated check at all; the existing 63 tests are algorithm/geometry-level, not detector-level.

---

## §B UPON A FULL RETRAIN SWEEP

Batched because each item needs a from-scratch training run, and this project's convention is to batch several detector/hyperparameter combinations behind one GPU session rather than retrain once per idea.

- Also retrain `FullScanTCN`/`TemporalUNet` at real-odometry `dtime=5`, not just `SpaceTimeCNN` (A1 covers `SpaceTimeCNN` only).
- Test a capacity-shrunk `SpaceTimeCNN` variant (`--channels`, `--n-spatial-stages`, higher dropout) sized for DROW's ~17.8x-smaller effective dataset, rather than reusing FROG-scale capacity (`memory/interpreting-evaluation.md`).

---

## §C BLOCKED — waiting on time or data

**JRDB evaluation for every detector.**
Registration at jrdb.erc.monash.edu requires verifying a university email, which isn't available right now (not currently university-affiliated) — checked whether a public mirror of just the 2D-LiDAR portion exists (HuggingFace, the DR-SPAAM/DROW repos) and found none; didn't search further, since redistributing this dataset outside its own registration gate isn't this project's call to make.
Unblock condition: either university affiliation becomes available, or the JRDB team (`erc-jrdb@monash.edu`) confirms a non-university registration path for research-only, non-submission use — not yet asked.
Until then, extraction into `library/follow_the_drow/include/JRDB-data/` stays undone and every JRDB column in this project's own results stays `n/a`, which is already the documented, correct behavior (`memory/gotchas.md`), not a bug.

---

## §D LATER

### C++ modernization: compile check, Meson migration, `clang-tidy`, `clang-format`

**What.** Pulled out of the active alignment plan (originally `AL6`) on 2026-09-08: not the current priority since there's no physical robot to deploy against right now, but real work this project will come back to.
Bundles: (1) confirm the C++ side still actually compiles — `.github/workflows/build-lidar-vision-image.yml`'s `build-cpp-static-library` job already runs `sudo make build-lib` on `ubuntu-latest`, but its current pass/fail status hasn't been checked this session, so verify it before assuming new work is needed; (2) migrate the build system to Meson; (3) `clang-tidy` (config plus `compile_commands.json` via `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, or Meson's native `compile_commands.json` output if (2) lands first); (4) `clang-format`, with a `.clang-format` config.

**How.** Scope properly when picked up rather than now — in particular, (2) needs a real decision on scope: `library/cpp_core/` is a plain CMake target and a reasonable Meson migration candidate, but `deploy/follow_the_drow/` is a **catkin** ROS package, and catkin is built on CMake specifically — ROS Noetic has no supported Meson build path, so a full migration likely means "migrate the standalone library, leave the ROS package on CMake," not a uniform switch, unless someone finds a workable catkin+Meson bridge worth the risk.

**Why.** Grouped as one item rather than broken into an AL-style sequence because it's explicitly deferred, not because it's small — real scoping (especially the Meson/catkin question) happens when this is picked back up, per the "don't invent a rule/plan for something not yet earned" principle this documentation layout otherwise follows.

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
