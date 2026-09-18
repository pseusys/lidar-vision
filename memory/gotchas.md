# Gotchas

*keywords:* encoding, unicode, cp1252, DirectML, venv, python3, ABI mismatch, pip install -e, JRDB, torch_utils

Platform and environment traps that have cost real debugging time.

- **`UnicodeEncodeError` on any non-ASCII character printed to the console.**
  Windows' default console codepage is cp1252, not UTF-8; a `→`, `✓`, or similar in a `print()` crashes mid-run, sometimes after the expensive part of the operation already succeeded (see `AGENTS.md` I3) — use ASCII (`->`, plain words) in anything printed here.
- **`python3` resolves to the Windows Store's "install Python" stub on this machine, not a real interpreter.**
  `make venv` / `make test` invoke it and will not create a usable environment here — use `.venv/Scripts/python.exe` directly (see `AGENTS.md` I5); fixing the Makefile is tracked in `TODO.md`.
- **`ModuleNotFoundError: No module named 'follow_the_drow.cpp_binding'`.**
  The pybind11 C++ extension did not build — re-run `pip install -e ./library` (builds it via `Pybind11Extension` in `library/setup.py`) or `make build-lib` from the repo root.
- **A Python version newer than 3.12 breaks the prebuilt C++ extension with an ABI mismatch.**
  The bundled `.pyd`/`.so` is built against CPython 3.12's ABI; a 3.13/3.14 interpreter loads it and fails confusingly rather than refusing cleanly — always use a 3.12 interpreter (`.venv`) for this project.
- **One process gets ~10.7 GB of the RX 9060 XT's 16 GB under ROCm on Windows.**
  Measured 2026-09-14 by allocating 256 MB blocks until failure: 10.75 GiB succeeded, then `HIP out of memory` with the driver still reporting 5.02 GiB free.
  A killed process's VRAM is also released lazily, so a run started straight after one can fail far below that; size batches for ~8 GB and do not overlap GPU jobs.
  The machine's commit limit (16 GB RAM + 15 GB page file) is routinely ~26 GB used by other programs, so two torch processes plus a dataset load can fail with `_ArrayMemoryError` on a 167 MB array.
  **A tiny `_ArrayMemoryError` is the signature of commit pressure, not of a real memory requirement.** 2026-09-16: a step-2 run died on its *first* training batch, `clip_batch` -> `split.scans[index]`, unable to allocate **3.08 MiB** for a `(32, 35, 720)` float32 array, while committed memory was nowhere near the limit minutes later. It was self-inflicted -- the run was launched while two evaluation passes were still finishing, and a full `pytest` was then run beside it. Nothing was salvageable: zero evaluations had completed, so the arm had to be re-run from scratch. Do not start a training run until the GPU work before it has exited, and do not run the test suite against a live training run: `pytest` is CPU-only but still imports torch and costs commit.
- **Chain training runs with `&&`, never `;`.**
  The same 2026-09-16 chain used `;`, so the second run started even though the first had already crashed, and an hour of GPU went to an arm whose partner had produced nothing. `&&` stops the chain at the first failure, which is what every other chain in this project uses.
- **A clip shorter than the oldest temporal tap silently repeats its first frame.**
  Training and clip-based test scoring then read copies while streaming inference (`step`) reads the real frames, and the two paths disagree without any error. 2026-09-16: the 5-tap and `0 2 4` fine-convolution runs used 35 frames where they needed 37. `--clip-frames` now defaults to `clip_length(coarse_lags)` = `max(coarse_lags) + 1`; only an explicit value shorter than that repeats the bug.
- **The FROG loader still clamps missing returns to 10 m by default, and misses one encoding.**
  `FROG_Dataset(missing_return_m=10.0)` is the legacy default every baseline number was measured with: non-finite readings become a real-looking 10 m, so any statistic over "valid" beams silently includes them. It never touches train/val's finite **61.0 m** no-return encoding. The three-horizon trainer passes `missing_return_m=None` and sanitises readings in the network (`sanitize_ranges`). A new analysis that wants the truth should do the same, or read `h5["scans"]` directly. The 1.30 cm consecutive-scan disagreement of 2026-09-16 is biased low for this reason.
- **The median scan interval misreads both datasets' rates: FROG is 40 Hz, not 26.2 Hz; DROW is 12.7 Hz, not 10 Hz.**
  DROW's timestamps are rounded to 0.05 s, so its intervals are a mix of 0.05 s and 0.10 s, and the median lands on 0.10.
  FROG's scans are stamped in bunches: about 0.038 s, 0.038 s, then under 0.1 ms. No two consecutive scans are identical, and the mean over linked intervals is 25.0 ms in every file, matching the UTM-30LX's 25 ms per scan. The median lands on 0.038 s.
  Use scans over elapsed time per unbroken sequence (`utils/scan_anomalies.py` reports both).
  **Never divide by a stamped FROG interval:** about a quarter of them are under 1 ms. `manage_slots` does exactly that for slot velocity (TODO A50).
  Scripts still on the median or on `1/26.2` are listed in TODO A49.
- **The machine can go to sleep in the middle of a training run, with the plugged-in sleep timeout set to "never".**
  2026-09-15: step 1 stopped at 10:28 with no error in its log; the System log shows "shutdown transition" and "system initiated reboot from Sleeping (Idle)" at 10:29:54, and no out-of-memory or bugcheck record.
  `utils/train_three_horizon.py` now asks Windows to stay awake for as long as it runs (`SetThreadExecutionState`), which lasts only while the process lives and changes no power setting.
- **A finished ROCm process can hang at exit instead of terminating.**
  Seen 2026-09-15: a step-1 smoke run wrote its results, then sat for 2.8 hours burning CPU and holding 3.4 GB of commit memory, which starved every later process. A crash hangs the same way (seen the same day: a step-3b smoke run's traceback, then a live process). `utils/train_three_horizon.py` now always leaves through `os._exit` -- 0 on success, 1 after printing the traceback -- so a launcher sees the exit code; still check for leftover `python.exe` before a long run. **`os._exit` does not always prevent it:** 2026-09-15, a test-only evaluation script that ends with `os._exit` wrote its results at 23:09:37 and was still alive at 23:12:55, when the chain launcher's watchdog stopped it. Keep launchers judging completion from the log, and stopping a process that lingers after it.
- **A continued training run (`--init`) keeps its own best checkpoint, which starts empty.**
  `BestCheckpoint` saves the run's first defined validation as "best", even when it is worse than the weights the run started from. 2026-09-15: the step-2 continuation peaked at 73.0% val AP against the first run's 73.8% and would have handed step 3 worse weights.
  Compare the two runs' best validation before the next step starts, in the launcher, not in a script that polls alongside it: a guard that swapped the file raced step 3a, which had started loading the checkpoint 4 s earlier.
- **Ranking a multi-stage checkpoint on the end-to-end metric can select one whose sub-stage is worse.**
  `train_joint` offers `result["memory"]["ap"]` to `BestCheckpoint`, the whole detector's score, which never charges for stage 2 degrading underneath it. 2026-09-16: epoch 1.0 held stage 2 at its validation best (76.12%) with a joint AP of 78.37%, and epoch 1.5 won selection at 78.98% with stage 2 already down to 75.68%.
  Where a stage is trainable and its own quality is not implied by the end-to-end score, rank on it too, or at least record it every evaluation so the drift is visible.
- **A feedback path initialised at zero can stay at zero, so the run never tests the thing it exists to test.**
  Step 3b's memory prior enters as extra input channels of `aggregate`, zeroed at construction so that a loaded stage 2 is unchanged until they train. After 1.5 epochs at `lr` 1e-4 those columns had norm 0.0485 (max |w| 0.0071) against the shared columns' 7.08, and silencing them moved test AP by 0.00 points.
  Check such a path has actually left its initialisation before reading any result as evidence for or against feedback.
- **A NaN detection passes every distance gate, and can be scored as a true positive.**
  NumPy comparisons with NaN are False, so `cost > radius` rejects nothing and `_prec_rec_2d`'s `radius < distance` calls it in range. DROW's vote grid made such detections from tied maxima until 2026-09-15 (fixed in `votes_to_detections`, `tests/test_vote_decoding.py`); `SimpleTracker` filters them too. Anything new that matches detections should reject non-finite coordinates rather than trust the gate.
- **`.venv` has no DirectML, on purpose.**
  `torch-directml==0.2.5.dev240914` pins torch 2.4.1 and cannot share a venv with AMD's ROCm build (torch 2.9.1), so the one environment carries ROCm and its GPU appears as `torch.cuda`; `detect_device()` picks it up as CUDA/ROCm.
  Its CPU path is ~10x slower than torch 2.4.1's on stage 3's small per-step operations (`CHANGELOG.md` 2026-09-14), so CPU is no fallback for recurrent training.
- **`pip install -e ../library` can silently point at an unrelated old checkout.**
  If imports resolve to code that doesn't match what's on disk, `pip uninstall follow_the_drow` and reinstall from the current path.
- **`library/follow_the_drow/utils/torch_utils.py` is not where `detect_device()` lives.**
  That file is an old CUDA-only helper (`move()`, `init_module()`, credited to Lucas Eyer's `lbtoolbox`) and does not do CUDA->DirectML->CPU detection — the real `detect_device()` is in `utils/train.py`.
- **`LidarFrameDataset` caches every preprocessed frame, so a bigger split is a bigger RAM bill.**
  ~25 KB/frame in raw_scan mode: FROG's 108,356-frame training split is ~2.7 GB, and caching the 195,058-frame val split alongside it OOM'd a 17.1 GB machine outright (`CHANGELOG.md`, 2026-09-10).
  Val is never cached now.
  If you add a split or a dataset, check what a full pass over it costs *before* running — and check whether anything still consumes that pass at all.
  `--no-frame-cache` trades the RAM back for recomputation on every access.

- **`pip install onnx` silently upgrades numpy past this project's pin.**
  It pulls `ml_dtypes`, which requires `numpy>=2.0.0`, over `follow_the_drow`'s `numpy~=1.24`.
  pip reports the conflict as a warning and installs anyway.
  Reinstall the pin straight after (`pip install "numpy~=1.24"`) and re-run the tests; `ml_dtypes` then warns in the other direction and is harmless, because nothing here imports it.

- **A literal `%` in an argparse `help=` string crashes `--help` and nothing else.**
  argparse `%`-formats every help string when it renders, so `"76.3% vs 78.8%"` raises `TypeError: %o format: an integer is required` — at `--help` time only.
  Every other test and every real invocation keeps passing, so it hides indefinitely.
  Double it: `76.3%%`. `tests/test_train_eval_consistency.py::test_both_clis_can_print_help` now catches it.

- **JRDB requires manual registration** (jrdb.erc.monash.edu) **and is never auto-downloaded.**
  Absence of `library/follow_the_drow/include/JRDB-data/` is normal, not a broken install; every JRDB column in this project's own results is `n/a` for that reason, not a code limitation.
- **DirectML training crashes are not necessarily a code bug.**
  See `AGENTS.md` I4 for the three distinct crash modes hit training `Li2FormerDetector` — don't sink more than one workaround attempt into a DirectML-only crash before flagging it as an environment limitation.
- **`.venv` is not a self-contained interpreter — its `pyvenv.cfg` points `home` at `C:\Users\...\AppData\Local\Programs\Python\Python312`.**
  A process can end up running under that base path instead of `.venv\Scripts\python.exe` while still executing this project's code — observed three separate times now (a duplicate DR-SPAAM eval process; a duplicate `train.py --dtime 5` process, killed 2026-09-10; a third unrelated pair on a scratchpad script from a different session, same day), always as an exact command-line duplicate holding real CPU time under the base install's path — if `ps`/Task Manager shows two copies of the same script, check whether one is the base install before assuming it's unrelated, and kill both PIDs, not just the one matching the path you expected.
  Frequent enough now to treat as a standing property of how background processes get launched here, not a one-off fluke — still not root-caused to an exact trigger.
