# Changelog

Step-by-step history of this repo: what changed, when, what was tested, what the result was, and the decision that came out of it.
Source files carry a short pointer here instead of the full history inline.
Explanations of how things *currently* work live in [`memory/`](memory/README.md), not here.

Organized by date, **newest first**.
Entries before 2026-09-08 are a one-time backfill from `docs/RESEARCH.md`, written during the documentation-layout migration on that date — dates come from file modification times and the evaluation run's own stated date, not from separate per-day commits.

## How to navigate this file

Every entry carries a `*keywords:*` line right under its heading, holding the identifiers, flags and constants that distinguish that entry from the rest of the file.
That is the fast path:

```bash
grep -n "^\*keywords:.*<identifier>" CHANGELOG.md                        # entries touching it
grep -n "^##.* <YYYY-MM>" CHANGELOG.md                                   # everything from a month
```

Topic index — grep the phrase in the right column to land on the entries:

| Looking for | Grep |
| --- | --- |
| the DR-SPAAM cutout-normalization bug | `window_depth` |
| the FROG odometry / session-splitting bugs | `session_gap_s` |
| the LFE-PPN anchor/recall bugs | `LFEPPNDetector` |
| the dtime sweep | `dtime=5` |
| the tracker's invocation-rate finding | `SimpleTracker` |
| this documentation layout itself | `agentic-layout-template` |

Decisions to *not* do something are in [`memory/rejected-ideas.md`](memory/rejected-ideas.md).
Open work is in [`TODO.md`](TODO.md).

## Rotation

This file covers the current development period; this project has not yet had a release to rotate against.
When it does, move this file's entries to `memory/changelog-archive/CHANGELOG-v<X>.md`, copy this header to the top of the archived file, and leave this file with the header and no entries — see the template this layout was adapted from for the full procedure, preserved in `memory/changelog-archive/` once the first rotation happens.

---

## `train.py --help` crashed on cp1252, blocking a flag check before the A1 sweep launch (2026-09-09)

*keywords: train.py, --help, UnicodeEncodeError, cp1252, epilog=__doc__, RawDescriptionHelpFormatter*

**`train.py --help` crashed with the same `UnicodeEncodeError` class already fixed once in `_download_file()` this session** — found while checking the exact `--dtime`/`--subsample`/`--epochs`/`--patience` flag names before launching A1's first screening run, not by accident.
Previously flagged (an earlier session) as "cosmetic, out of scope"; this time it actively blocked verifying a command before a real run, so it was worth fixing rather than working around.

**Root cause, more specific than the earlier fix**: the parser sets `epilog=__doc__` (the whole module docstring, `RawDescriptionHelpFormatter`), so any non-cp1252 character anywhere in that docstring reaches the console — not just characters in `argparse` `help=` strings, which was the initial (too narrow) hypothesis.
Checked precisely (encode-tested every non-ASCII character against `cp1252`, not just flagged every non-ASCII character): the module docstring and a handful of `help=` strings carried `→` and `≈`; other non-ASCII characters elsewhere in the file (em-dashes, `×`, `§`, `–`) are actually valid cp1252 and were never the problem.

**What changed in the code.**
`utils/train.py`: every confirmed-unsafe character replaced with an ASCII equivalent (`→`->`->`, `≈`->`~`, `°`->`deg`) across the module docstring, `detect_device()`'s docstring, three `help=` strings, and a few internal docstrings/comments not currently printed anywhere but fixed for consistency.
Verified: a script that encode-tests every character in the file against `cp1252` now reports none unsafe; `train.py --help` runs to completion; `pytest tests -q` still 63/63 (unrelated to this file's logic, but confirms nothing broke).

**What it means.**
This class of bug is per-file, not fixed once — anything printed to this Windows console needs the same check before being trusted; `memory/gotchas.md`'s existing entry already states the general rule.

---

## DR-SPAAM published-weights evaluation abandoned: T=5 skipped by decision, T=1 stalled with an unexplained duplicate process (2026-09-09)

*keywords: eval_frog_official_drspaam.log, DrSpaamDetector, stall, duplicate process, AppData Python312, .venv312*

**T=5's own evaluation was skipped by owner decision** once T=1 looked close to finishing — the FROG paper's own Table 4 numbers (T=1: 73.3%, T=5: 75.3%) were already documented and cited, and re-deriving them independently wasn't judged worth the remaining ~5h.

**T=1 then failed to complete either, but not by timing out — it stalled.**
After a consistent ~7 fr/s for 4h40m, progress froze at 120385 of 120396 frames (>99.9%) and stayed there for 12+ minutes with no further movement before being killed.

**A second process was found running the identical script** at the same time: same `-c` script text byte-for-byte, but invoked via `C:\Users\...\AppData\Local\Programs\Python\Python312\python.exe` rather than this project's `.venv312` — and *that* process, not the intended `.venv312` one, held almost all the accumulated CPU time (the `.venv312` process itself showed essentially none, despite being the one whose output was reaching the log file).
**Partially explained, later the same day**: `.venv312\pyvenv.cfg` sets `home = C:\Users\...\AppData\Local\Programs\Python\Python312` — that "unrelated" interpreter is actually this venv's own base install, not a stranger.
The paths being linked isn't a coincidence, but *why* a process ended up executing via the base path instead of the venv's own launcher is still not root-caused; noted in `memory/gotchas.md` for next time this pattern shows up.
Both processes were killed once found.

**What was decided, same day.**
`memory/performance-log.md`'s `DrSpaamDetector`/FROG cell cites the paper's numbers for both T=1 and T=5, alongside the already-passed (and still valid) official-weights-compatibility check (zero missing/unexpected keys loading into this repo's `DrSpaamDetector`).
Not reopened: the explicit call was to put further effort into this project's own novel results (`TODO.md` A1) rather than into reproducing a number the paper already reports.
`TODO.md` A2 is closed on this basis.

---

## `FROG_Dataset.scan_time` lost sub-minute precision to a float32 cast; real per-frame motion measured directly (2026-09-09)

*keywords: scan_time, odoms["t"], float32, float64, Unix epoch precision, dtime, ego-motion, human displacement, det_wp*

**`FROG_Dataset.scan_time` and `odoms["t"]` were stored as `float32` at Unix-epoch magnitude (~1.4e9) — every consecutive raw frame collapsed to an identical timestamp, quantized in ~128-second jumps.**
Found while writing a sanity-check script that reads `scan_time` directly for the first time (nothing in the training/eval pipeline had ever consumed it): `np.diff(scan_time)` was `0.0` for essentially every pair, `128.0` at each precision-boundary crossing.
Checked whether this had silently corrupted anything already trusted: it had not — session-gap detection (`_load_h5`) and the real-odometry `np.interp` alignment (`_load_or_fake_odom`) both already used the full-precision `float64` array read straight from the `.h5` file, before this cast ever happens, so every finding and number that depends on those (the 2026-09-08 entries below) stands.
This was a real trap for the *next* thing to read `scan_time` directly, though — which is exactly what happened here.

**What changed in the code.**
`library/follow_the_drow/datasets/frog_dataset.py`: the `dtype`'s `"t"` field and both `timestamps_s.astype(...)` call sites now use `np.float64` instead of `np.float32`.
Verified: `pytest tests -q` still 63/63; `scan_time`'s consecutive diffs are now the real ~38ms native spacing, not 0/128.

**Sanity check, run once the fix was in place: how much does the robot itself move per frame, at each `dtime`?**
FROG's native rate measures 26.2 Hz (~38.15 ms/frame), not the documented 40 Hz.
Real (not estimated) odometry, median robot displacement per `dtime`-frame step:

| `dtime` | span | median translation | median rotation |
| --- | --- | --- | --- |
| 1 | 38 ms | ~1.15-1.16 cm | ~0 deg |
| 5 | 191 ms | ~6.2 cm | ~0 deg |
| 10 | 382 ms | ~12.4 cm | ~0 deg |
| 20 | 763 ms | ~24.9 cm | ~0-0.07 deg |
| 40 | 1.5 s | ~49.8 cm | ~0-2.2 deg |

The robot barely moves in any of these windows — good news for alignment, since there's little ego-motion for `aligned_raw_scan()` to have to correct in the first place.
(A small number of extreme outliers, up to ~21m at `dtime=1`, are almost certainly an odometry glitch or session-adjacent artifact, not real single-frame motion — not investigated further, doesn't affect the median-based conclusion above.)

**Second sanity check: how much does a *person* actually move per frame** — matched directly against real annotated positions (`det_wp`), not assumed from a textbook walking speed.
Greedy nearest-neighbour match between two frames' person point-sets (sensor-local coordinates, 2.5 m gate against cross-person mismatches), sampled every 25th annotated frame, ~13,000-15,000 matched pairs per `dtime`:

| `dtime` | span | median displacement | mean | p90 |
| --- | --- | --- | --- | --- |
| 1 | 38 ms | 2.12 cm | 3.52 cm | 5.13 cm |
| 5 | 191 ms | 8.78 cm | 14.28 cm | 22.50 cm |
| 10 | 382 ms | 17.16 cm | 26.44 cm | 46.28 cm |
| 20 | 763 ms | 34.78 cm | 47.43 cm | 93.05 cm |
| 40 | 1.5 s | 67.14 cm | 81.56 cm | 160.70 cm |

**What it means.**
Median displacement scales almost exactly linearly at ~1.7 cm per unit of `dtime` across this whole range — a real, human-motion-dominated signal (the robot's own ego-motion above is small enough not to explain it).
This confirms the original intuition behind reconsidering the default: at `dtime=1`, consecutive frames really are close to identical for a person (~2 cm, well under a body width); by `dtime=10` a person has moved ~17 cm (a meaningful fraction of a stride); by `dtime=20`, ~35 cm (close to half a walking stride).
Not yet decided which `dtime` should be the new FROG default — that's exactly what the retrain sweep in `TODO.md` A1 (now widened to test a wider range than originally planned) is for; this entry is the measurement the decision should be checked against, not the decision itself.

---

## Fixed the Makefile's broken `python3`, corrected a false code-duplication claim (2026-09-08)

*keywords: PYTHON, VENV_PYTHON, python3, python, OS Windows_NT, make venv, make test, FollowTheDrow CMake, find_package*

**`make venv`/`make test` were silently broken on Windows** — `python3` resolves to the Microsoft Store's install stub here, which exists as a command but does nothing useful when run, so `python3 -m venv venv` never created a usable environment.
Separately, `make venv` never installed `utils/requirements.txt`, so `pytest` was never available even once that was fixed.

**What changed in the code.**
`Makefile` gained `PYTHON`/`VENV_PYTHON` variables, branched on `$(OS)`: Windows prefers bare `python` (checked with `command -v`, since `python3` there is the trap above), Linux/CI prefers `python3` (the tested path on `ubuntu-latest`, which may not alias `python`).
`venv`/`test` now call `$(VENV_PYTHON)` explicitly instead of relying on `PATH` (which assumed `venv/bin`, wrong for Windows' `venv/Scripts`); `venv` now also installs `utils/requirements.txt`.
Verified with `make -n test` (dry run): resolves to the real `/c/Python314/python`, correct `venv/Scripts/python` invocations throughout.

**What it means, and what was decided.**
`python` on this machine is 3.14, not the 3.12 the prebuilt C++ extension needs (`memory/gotchas.md`) — this fix resolves "the command doesn't exist," not "the command is the wrong version"; a full `make venv` run has not been exercised end-to-end here yet.
Tracked as a separate, still-open item (`TODO.md` A9).

**A previously-recorded finding is retracted: the C++ `AlgorithmicDetector` is not duplicated between `library/cpp_core/` and `deploy/follow_the_drow/`.**
Checked directly (not assumed, this time): `deploy/follow_the_drow/CMakeLists.txt` does `find_package(FollowTheDrow CONFIG REQUIRED)` and links every node against the compiled library; `deploy/follow_the_drow/src/algorithmic_detector.cpp` is a 54-line ROS-node wrapper around the shared `follow_the_drow::AlgorithmicDetector` class, not a second copy of its logic.
The earlier claim (written during the same-day documentation migration, from file-name similarity rather than a diff) has been corrected in `memory/dos-and-donts.md` and `memory/coding-guidelines.md`.
The real, narrower duplication that *is* there: `deploy/follow_the_drow/nodes/drow_detector.py` hardcodes its own `RESULT_CONF` vote-clustering hyperparameters independent of the training/eval pipeline's own defaults (`TODO.md` A10).

---

## Migrated to the agentic-layout-template documentation layout (2026-09-08)

*keywords: agentic-layout-template, AGENTS.md, memory/, verify_memory.py, HIGHLIGHTS.md, EVALUATION_PLAN.md*

**Adopted the `pseusys/agentic-layout-template` five-file documentation layout** (`AGENTS.md` / `README.md` / `TODO.md` / `CHANGELOG.md` / `memory/`), replacing a `docs/` directory that had grown to 827 lines in `RESEARCH.md` alone, mixing enduring architecture rationale, dated bug-fix narratives, rejected experiments, and evaluation caveats in one file.

**What changed.**
`docs/RESEARCH.md`'s Section 8 (evaluation results and their history) was split out: the dated findings below are now in this file, tuning-sweep tables moved to `memory/performance-log.md`, accuracy-number caveats to `memory/interpreting-evaluation.md`, and three tried-and-rejected experiments to `memory/rejected-ideas.md`.
`docs/RESEARCH.md` itself is kept as the citation-grade research write-up (literature review, architecture rationale, the fidelity audit against each paper) — its own audience (an eventual paper/thesis writeup) differs from `memory/`'s (an agent working in this repo), so it was not folded in wholesale.
`docs/HIGHLIGHTS.md` (a one-off summary written the previous session) and `docs/EVALUATION_PLAN.md` (a runbook for evaluation tables that are now already filled in) were retired — their content is now `memory/performance-log.md`, `README.md`'s "Where things stand", and `memory/commands.md` respectively.
The stale `compare/` directory references (two Makefile targets, several `AGENTS.md` mentions) were removed — the directory itself no longer exists in this repo.

**What it means.**
Every fact now has exactly one home: `memory/*.md` for current mechanism, this file for what happened and when, `memory/rejected-ideas.md` for settled negative results, `TODO.md` for what's still open.
Markdown structure (links resolve, both indexes match the files on disk, one-sentence-per-line) is enforced by `memory/scripts/verify_memory.py`, wired into CI via `.github/workflows/lint.yaml`.

## FROG ships real, official per-session odometry that this codebase had never downloaded (2026-09-08)

*keywords: FROG_Dataset.download, frog_<HH-MM>_odom.npz, _session_odom_filenames, session_gap_s, session splitting, integrate_trajectory, max_dt, timestamp gap*

**The whole ICP-pseudo-odometry effort (previous entry below) was working around a self-inflicted gap, not a real one.**
The FROG authors publish real per-session odometry (`frog_<HH-MM>_odom.npz`, confirmed via HTTP HEAD returning 200 for every session) that `FROG_Dataset.download()` never fetched — it only ever downloaded the `.h5` scan files — so `_load_or_fake_odom()` silently fell back to zero odometry on every run to date.
Cross-checked, not assumed: the real odometry files' own session-boundary gap (88,658s) matches the gap independently found in scan timestamps (88,655s) to within 3 seconds.

**Two more bugs found while investigating why the odometry work looked unlike anything expected.**
(1) `integrate_trajectory()` treated a whole `.h5` file's scans as one continuous ~26-40 Hz stream; real recordings have 62 timestamp gaps (61 sub-90s pauses plus one ~24.6-hour cross-session gap), and 31 of the 62 produced a small, *plausible-looking* ICP delta that slipped past the existing magnitude clamp — including the 24.6-hour boundary itself, which looked like an innocuous "0.087m, 3.74°" step.
(2) More fundamentally, `FROG_Dataset` treated one `.h5` file as one *sequence*, so `get_scan()`'s temporal window had no way to know about that same boundary — meaning even zero-odometry runs could pull raw-scan history across two unrelated recordings for frames near it.

**What changed in the code.**
`integrate_trajectory()` (`library/follow_the_drow/utils/odometry_estimation.py`) now takes real `timestamps` and forces zero motion across any gap exceeding `max_dt` (default 0.1s), before asking ICP at all.
`_load_h5()` (`library/follow_the_drow/datasets/frog_dataset.py`) splits into multiple sequences at any gap exceeding `session_gap_s=300.0`, mirroring DROW's own multi-sequence structure.
`download()` now also fetches each session's real odometry file after its `.h5`, and `_load_h5()` prefers a matching real file over the estimated/zero fallback when present, via a new `_session_odom_filenames()` filename-token parser.
Verified against the real files: `FROG_Dataset(split='train')`'s loaded x/y/theta ranges exactly match the real files' own reported ranges.
9 new tests (`tests/test_frog_session_splitting.py`, `tests/test_odometry_estimation.py`); full suite at 63 passing.

**What it means, and what was decided.**
Every real-odometry number produced before this fix (next entry) is retracted as a valid real-odometry measurement — it was trained on estimated pseudo-odometry, a single-session dataset, or both.
A from-scratch `dtime` sweep on the fully-fixed pipeline (real odometry + correct session splitting) is needed before any real-vs-zero-odometry conclusion can be trusted; not yet scheduled (`TODO.md`).

## ICP pseudo-odometry estimator built for FROG, first real-odometry `dtime` sweep run (2026-09-05 to 2026-09-06)

*keywords: odometry_estimation.py, generate_frog_odometry.py, trimmed ICP, Kabsch, dtime=1, dtime=5, real odometry*

**Built a trimmed-ICP scan-matching odometry estimator for FROG, which ships no odometry at all** (confirmed by inspecting its `.h5` files directly at the time — no pose field of any kind) — see the entry above for how that premise turned out to be wrong.
Verified with 10 synthetic-ground-truth unit tests (forward/rotation/combined/sideways/static motion) before running on real data, given this project's own prior sign-error history in exactly this kind of derivation (`memory/dos-and-donts.md`'s evidence section).

**First real-data issue found and fixed during generation**: a large open-space region (~97% of beams out of range) produced a sustained cluster of large, wrong jumps — a classic scan-matching failure mode, not a code bug.
Fixed with a per-step sanity clamp (reject an implausible-velocity delta, fall back to zero motion for that frame) plus a higher minimum-inlier count; max single-step distance dropped from 1.10m to the 0.20m clamp bound.

**First `dtime` sweep on this (later found to be superseded) real odometry**: `dtime=1` measured 78.4% wp-AUC (train and test) vs. 76.7% zero-odometry — a real gain at the time; `dtime=5` measured 79.5-79.6% vs. 80.7% zero-odometry — a real *regression* at the time, train ~ test both cases.
Both numbers are retracted as real-odometry measurements by the entry above — they used a since-fixed pipeline (single-session dataset, no timestamp-gap protection).

**What it means.**
The estimator itself (`library/follow_the_drow/utils/odometry_estimation.py`) remains useful as a fallback for any FROG session without a real odometry file, and its own unit tests are unaffected by the pipeline bugs above.

## Does temporal fusion help on FROG? Error breakdown, NMS sweep, and a wide-window regression (2026-08-24)

*keywords: error_breakdown.py, nms_sweep.py, vote_collect_radius, nms_radius, false positive, SimpleTracker, dtime=20*

**False positives, not false negatives, are the dominant error on FROG for every model tested** — FN/FP ratio never exceeds 0.8x for `spacetime_cnn`, `fullscan_tcn`, `temporal_unet`, `lfe_peaks`, and `lfe_ppn` alike, measured by re-reading the same precision/recall curve every AUC number already comes from, at each model's own best-F1 threshold (`utils/error_breakdown.py`).

**Operating-threshold selection is a free, unused win for 3 of 5 models** — moving to a ~75-80% precision point trades more false positives saved than false negatives added; past ~80% the trade reverses. `FullScanTCN` and `LFE-PPN` don't get this win from threshold selection alone.

**NMS/vote-clustering sweep found two small, real wins**: `FullScanTCN`'s `vote_collect_radius` peaks at 0.7 (+0.2pp over the 0.5 default); `LFE-PPN`'s `nms_radius` peaks at 0.9 (+0.5pp over 0.8).
Two bugs found and fixed along the way: an all-zero detection config crashed `_deep2flat()` and `error_breakdown.py`'s best-F1 finder instead of returning a well-defined empty result.

**A wide real-time window (`--time-frame 10 --dtime 20`, FullScanTCN) was retrained to test whether a clearer motion signal over ~4.5s would help, given NMS tuning alone didn't close FullScanTCN's accuracy gap to SpaceTimeCNN — it regressed to 54.8% wp-AUC (val), ~15 points below the `dtime=1` baseline.**
Root cause: FROG had no odometry at the time (see the entries above), so the wider window gave uncompensated ego-motion 36x longer to accumulate, outweighing any clearer-motion benefit.
Recorded as rejected in `memory/rejected-ideas.md`; results adopted as new defaults (`vote_collect_radius=0.7`, `nms_radius=0.9`).

## `SimpleTracker` built; a real DROW ego-motion sign bug found and fixed (2026-08-26)

*keywords: SimpleTracker, tracking.py, aligned_raw_scan, ego-motion compensation, DROW wp-AUC*

**Built `SimpleTracker`** (`library/follow_the_drow/utils/tracking.py`), a minimal SORT-style tracker (constant-velocity alpha-beta filter, Hungarian assignment against a track's *predicted* position, confirmation via `min_hits`, miss-tolerant coasting up to `max_age`) — the mechanism that replaced an earlier naive-averaging design rejected on inspection (`memory/rejected-ideas.md`).

**Fixed a rotation-sign bug in `aligned_raw_scan()`** discovered while validating the tracker's ego-motion-compensation path on DROW, which has real per-frame odometry.
Fixing it took the DROW-native `SpaceTimeCNN` baseline from 15.0% to 18.5% wp-AUC on test — confirming ego-motion compensation is worth having correctly, not just in principle.

**What it means.**
DROW is the fairer testbed for validating this tracker's core assumptions, since it has real odometry to compensate with; FROG was treated as a static-sensor case by necessity at the time (before the entries above found real FROG odometry existed).

## `dtime=5` found as a real accuracy peak on FROG, not overfitting (2026-09-02 to 2026-09-04)

*keywords: dtime sweep, dtime=0, dtime=1, dtime=5, dtime=10, SpaceTimeCNN 80.7%*

**A proper `T=5` sweep over `dtime` in {0, 1, 5, 10} on FROG (SpaceTimeCNN, full retrain per point, zero odometry, evaluated on both train and test) found a real, non-overfitting peak at `dtime=5`**: 77.1% (dtime=0) -> 76.7% (dtime=1) -> **80.7%** (dtime=5) -> 78.0% (dtime=10), train ~ test at every point.

**What it means.**
The original `dtime=1` (~125ms) window was simply too narrow to capture real human motion at FROG's 40 Hz rate, not evidence that temporal fusion is structurally useless there — reframing the original "why doesn't our temporal architecture beat single-frame LFE-Peaks" question, which motivated re-checking that comparison specifically at `dtime=5` (see the full evaluation run below).
This sweep used zero (not real or estimated) odometry throughout and predates the FROG odometry work above — see `memory/performance-log.md` for the full table and its own retraction note.

## DROW zero-shot transfer beats native training and fine-tuning, for every proposed architecture (2026-08-13 to 2026-08-14)

*keywords: DROW-native, zero-shot transfer, fine-tuned, --init-weights, overfitting, annotation coverage*

**Zero-shot transfer of FROG-trained weights onto DROW wins outright over both training natively on DROW and fine-tuning from the FROG checkpoint, for all three proposed architectures** — SpaceTimeCNN 29.0% (zero-shot) vs. 15.0% (native) vs. 17.2% (fine-tuned); FullScanTCN 22.2% vs. 16.5% vs. 21.4%; TemporalUNet 22.1% vs. 19.4% vs. 12.7%.

**Root cause, checked directly rather than assumed**: DROW's train split has ~17.8x fewer total person-annotation instances than FROG's (6.1x fewer annotated frames, 2.9x fewer people per frame).
A DROW-native `spacetime_cnn` run shows textbook overfitting (val loss rising from epoch 2 onward while train loss keeps falling) under the same regularization that works on FROG — a data-scarcity problem relative to model capacity, not a broken model (a spot-check on real positive frames found it still produces confident, roughly-correct detections).

**What changed in the code.**
Added `--init-weights` to `train.py` (loads weights only, fresh optimiser/epoch count — unlike `--resume`, which also restores the optimiser state and would carry over the wrong learning rate).

**What it means, and what was decided.**
These architectures transfer to a differently-shaped LiDAR reasonably well without retraining, and can be retrained cheaply if desired, but at DROW's data scale retraining currently doesn't pay for itself.
Shrinking the architecture for DROW-scale data specifically has not been tried (`TODO.md`).

## Full evaluation run: two real fidelity bugs found and fixed in published baselines (2026-08-13)

*keywords: window_depth, THRESH_DIST, cutout normalization, LFEPPNDetector anchor count, LFE recall metric, _prec_rec_2d, torch-directml, ABI mismatch*

**DR-SPAAM's published weights measured 25.8% wp-AUC on DROW (paper: 69.6-72%+) — a missing cutout-normalization step, not an attention-mechanism bug.**
Found by cloning the official DR-SPAAM-Detector repo and diffing its real `scans_to_cutout()` source against this repo's `cutout()` (not just comparing config files): the official `dr_spaam.yaml` specifies `window_width=1.0, window_depth=0.5` (narrower/shallower than DROW's own `1.66`/`1.0`) and **divides the centred cutout by `window_depth`** — a step this repo's `cutout()` was missing entirely.
For DROW's own weights (`window_depth=1.0`) the missing division is a no-op, which is why `DrowDetector` looked unaffected throughout.

**What changed in the code.**
`cutout()` (`library/follow_the_drow/utils/drow_utils.py`) now divides by `thresh_dist` (backward-compatible no-op at DROW's default of 1.0); `DrSpaamDetector` gained `WIN_SZ=1.0`/`THRESH_DIST=0.5` class attributes matching the official config.
DROW wp-AUC after the fix: 68.1% (matching the paper's range, now correctly ranking above `DrowDetector`'s 66.5%); FROG (zero-shot transfer): 38.6% -> 67.2%.

**`LFEPPNDetector` had two independent bugs**: it hardcoded 31 depth anchors per sector, but the bundled ONNX model's real output shape has 30 — every evaluation crashed.
And `evaluate.py`'s LFE recall computation divided by its own found true positives rather than the total ground-truth count, structurally unable to penalize a detector for missing people outright — this inflated `LFEPeaksDetector`'s FROG number to 86.2%, ~21 points above its own paper's 64.9%.
Fixed by reading the anchor count from the model's own output shape, and routing LFE through the same `_prec_rec_2d` PR-curve code every other model already uses.
Post-fix: LFE-Peaks 74.6%, LFE-PPN 67.7% (close to the paper's 64.9%/66.5%).

**Environment issues found and fixed before any of the above could even run**: a stale editable `pip install -e ../library` pointing at an unrelated old checkout; a Python 3.14/cp312 ABI mismatch against the prebuilt C++ extension (resolved with a dedicated Python 3.12 venv, `.venv312`); the bundled DROW/DR-SPAAM/LFE weights and dataset were never actually downloaded in this environment; no GPU acceleration installed despite a usable AMD GPU (`torch==2.4.1` + `torch-directml` installed and verified).

**What it means, and what was decided.**
Every number in `memory/performance-log.md` sourced to this evaluation run reflects these fixes; the run is the baseline every later finding in this file builds on.
