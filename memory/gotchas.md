# Gotchas

*keywords:* encoding, unicode, cp1252, BOM, U+FEFF, utf-8-sig, mojibake, CRLF, DirectML, venv, python3, ABI mismatch, pip install -e, JRDB, torch_utils, set_num_threads, OMP_NUM_THREADS, spin-wait, CPU time vs wall time, docker, ROS, launch-docker-local, WSL

Platform and environment traps that have cost real debugging time.

- **`sed -i` from the Bash tool silently converts a CRLF file to bare LF, and the result is a whole-file diff.**
  Every markdown file in this repo is CRLF (`core.autocrlf=true`, no `*.md` rule in `.gitattributes`). On 2026-10-30
  two `sed -i` calls on `TODO.md` rewrote all 1,961 line endings to LF; `git diff --stat` then reported **3,918
  changed lines for a 6-line edit**, and `git diff --ignore-all-space` was what revealed the real change. MSYS `head`,
  `tail` and `cat` **do** preserve CRLF, so a head/tail/cat splice is safe.
  **`awk` is not, and this entry said it was.** On 2026-11-01, seven `awk '{print}'` splices -- the obvious
  tool for "insert these lines after line N" -- rewrote `TODO.md`, `CHANGELOG.md` and three `memory/` pages to LF,
  one file at a time over four hours, because MSYS awk reads *and writes* in text mode. `cp`-ing an awk output over
  the original carries it through. The sibling trap is in the next paragraph but one: the `grep` check says the file
  is fine, so nothing looks wrong until `git diff --stat` reports a four-figure line count for a ten-line edit.
  **Treat every MSYS text utility as LF-converting except `head`, `tail` and `cat`**, and verify with the byte count
  below after *any* of them, not only after `sed -i`.
  **Do not diagnose this with `grep -c $'\r$'`** -- MSYS grep reports a match on every line either way and will tell
  you the file is still CRLF. Count bytes instead:
  `.venv/Scripts/python.exe -c "import pathlib; b=pathlib.Path('F').read_bytes(); print(b.count(b'\r\n'), b.count(b'\n')-b.count(b'\r\n'))"`
  and repair with `b.replace(b'\n', b'\r\n')` after asserting there are no CRLFs left to double.
  **Prefer the `Edit` tool or a Python byte-level rewrite for in-place edits of tracked files.**
  **And "a Python rewrite" means *bytes*, not `read_text`/`write_text`:** 2026-10-31, the same trap again
  through a different door -- `path.read_text(encoding="utf-8")` applies universal newlines and hands back `\n`, so a
  following `write_text(..., newline="")` writes the whole file LF and `utils/train.py`'s **88-line** edit reported
  **4,478 changed lines**. Read with `read_bytes()`, decode, splice with `"\r\n"` in the search strings, and
  `write_bytes()` back.
  **Convert newlines in exactly one place.** The same day, a helper that took already-CRLF text and converted it again
  turned every line ending of a 77-line `CHANGELOG.md` entry into `\r\r\n`. It renders correctly and `git diff` shows
  nothing odd, but Python reading it with universal newlines sees a **blank line between every real one** -- which is how
  it was found: `verify_memory.py` reported 36 new "more than 1 consecutive blank line" warnings in a file whose own
  blank lines were all single. So when checking a file you rewrote, count `b'\r\r\n'` as well as bare `b'\n'`; a
  CRLF count alone says the file is fine.

- **Some of this repo's text files are not plain UTF-8, and two separate traps follow from it.**
  Found 2026-10-31 by an AST-based surface scan, which refused to parse four files and matched nothing in a fifth.
  * **Four Python sources start with a UTF-8 BOM** -- `library/takheliped/detectors/takheliped.py`,
    `utils/stage2_probe.py`, `tests/test_run_queue.py`, `tests/test_stage2_probe.py`. Python's *import* machinery strips
    it, so they run fine and nothing has ever complained; `ast.parse()` on text read as plain `utf-8` does **not**, and
    dies with `SyntaxError: invalid non-printable character U+FEFF`. Any tool here that reads source and parses it must
    use **`encoding="utf-8-sig"`**, which is correct for a BOM-less file too.
  * **`docs/RESEARCH.md` holds double-encoded punctuation.** Its em dashes are stored as the three characters
    U+00E2 U+20AC U+201D (the UTF-8 bytes of `-` re-encoded as if Latin-1), so a search string containing a real em
    dash matches **nothing** in that file while looking identical on screen -- a console will render both the same way.
    `README.md` and the `memory/` pages are clean. **Anchor edits to that file on ASCII-only substrings.** Repairing the
    mojibake itself is unclaimed work, deliberately not bundled into an unrelated item.

- **This development host cannot build or run the ROS side at all** (verified 2026-10-30 at the owner's prompting).
  There is **no `docker`** on `PATH`, no daemon, and **none inside WSL either**; WSL2/Ubuntu is installed but has **no ROS**
  (`/opt/ros` absent, no `roscore`). So `make launch-docker-local`, `make launch-docker-robot` and `make build-image`
  all fail here, and [`deployment.md`](deployment.md)'s "laptop only" next to the first of them means *a* laptop with
  Docker, **not this one**. The image is `ros:noetic-ros-base-focal` via `deploy/docker/Dockerfile`.
  **Consequence for planning:** anything whose value is realised on the robot cannot be verified in this repo's working
  environment, only reasoned about -- so say so in the report instead of implying a deployment benefit was demonstrated.
  Related: **there is no TAKHeLiPeD ROS node**. The only detector node is `deploy/takheliped/nodes/drow_detector.py`,
  loading `DrowDetector`, and `deploy/conf.env`'s `DROW_DETECTOR_THRESHOLD=0.80` is **DROW's** threshold -- no deployed
  node reads a TAKHeLiPeD score (`AGENTS.md` rule 5, `../TODO.md`'s NOT DEPLOYED section).

- **A finished ROCm process can hang unkillably, and `taskkill /F /T` will not clear it -- `Invoke-CimMethod` will.**
  The teardown hang is already known (a finished process lingers past `os._exit`, which is why scripts here call it at all),
  but 2026-10-12 produced the stubborn form: two completed runs sat in an uninterruptible driver wait, and `taskkill /F /T`
  answered **"ERROR: ... could not be terminated. Reason: There is no running instance of the task"** while Windows still
  listed them. `Stop-Process -Force` and `$p.Kill($true)` also failed. What worked:

      Get-CimInstance Win32_Process -Filter "ProcessId=<pid>" | Invoke-CimMethod -MethodName Terminate

  **Why it matters beyond tidiness:** each held **3.4 GB of commit** (982 MB resident), pushing the host to 24.5 GB of its
  29.85 GB commit limit -- and the next `pytest` run failed with `ImportError: DLL load failed while importing cv2: The
  paging file is too small for this operation to complete`, which looks like a broken OpenCV install and is not one. One
  test failed; clearing the two processes dropped commit to 16.9 GB and the same suite passed 865/865. **Check committed
  bytes before believing a resource-shaped test failure:**

      (Get-Counter '\Memory\Committed Bytes').CounterSamples.CookedValue / 1MB

- **A torch process draws far more CPU than `--threads` says, and it is not a bug.** At `torch.set_num_threads(4)` on this
  host (Ryzen 7 5700X, 8 cores / 16 logical) the process accumulates about **9.7 CPU-seconds per wall-second**: four threads
  are `Running` and doing the work, while **twelve more sit in a spin-wait pool at ~44% duty each** -- OpenMP sizes its pool
  to the logical core count when the process starts, and `set_num_threads` afterwards cannot shrink it.
  **It does not affect a latency measurement**, which is the reason not to chase it: DROW timed **319.55 ms** per frame with
  the spinning pool and **322.09 ms** with `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0`,
  a 0.8% difference inside run-to-run noise, because the spinners idle on cores the four workers are not using.
  Verified 2026-10-11 **after killing a 90-minute sweep to investigate it**, which is the cost this entry exists to
  spare. If the question is which thread count is fastest, sweep `--threads` and read the latency; the CPU/wall ratio answers
  a different question than it appears to.

- **`UnicodeEncodeError` on any non-ASCII character printed to the console.**
  Windows' default console codepage is cp1252, not UTF-8; a `→`, `✓`, or similar in a `print()` crashes mid-run, sometimes after the expensive part of the operation already succeeded (see `AGENTS.md` I3) — use ASCII (`->`, plain words) in anything printed here.
- **`python3` resolves to the Windows Store's "install Python" stub on this machine, not a real interpreter.**
  `make venv` / `make test` invoke it and will not create a usable environment here — use `.venv/Scripts/python.exe` directly (see `AGENTS.md` I5); fixing the Makefile is tracked in `TODO.md`.
- **`ModuleNotFoundError: No module named 'takheliped.cpp_binding'`.**
  The pybind11 C++ extension did not build — re-run `pip install -e ./library` (builds it via `Pybind11Extension` in `library/setup.py`) or `make build-lib` from the repo root.
- **A Python version newer than 3.12 breaks the prebuilt C++ extension with an ABI mismatch.**
  The bundled `.pyd`/`.so` is built against CPython 3.12's ABI; a 3.13/3.14 interpreter loads it and fails confusingly rather than refusing cleanly — always use a 3.12 interpreter (`.venv`) for this project.
- **One process gets ~10.7 GB of the RX 9060 XT's 16 GB under ROCm on Windows.**
  Measured 2026-09-14 by allocating 256 MB blocks until failure: 10.75 GiB succeeded, then `HIP out of memory` with the driver still reporting 5.02 GiB free.
  A killed process's VRAM is also released lazily, so a run started straight after one can fail far below that; size batches for ~8 GB and do not overlap GPU jobs.
  The machine's commit limit (16 GB RAM + 15 GB page file) is routinely ~26 GB used by other programs, so two torch processes plus a dataset load can fail with `_ArrayMemoryError` on a 167 MB array.
  **A tiny `_ArrayMemoryError` is the signature of commit pressure, not of a real memory requirement.** 2026-09-16: a step-2 run died on its *first* training batch, `clip_batch` -> `split.scans[index]`, unable to allocate **3.08 MiB** for a `(32, 35, 720)` float32 array, while committed memory was nowhere near the limit minutes later. It was self-inflicted -- the run was launched while two evaluation passes were still finishing, and a full `pytest` was then run beside it. Nothing was salvageable: zero evaluations had completed, so the arm had to be re-run from scratch. Do not start a training run until the GPU work before it has exited, and do not run the test suite against a live training run: `pytest` is CPU-only but still imports torch and costs commit.
  **The limit is commit, not RAM, and beside a live run there is none left: a second torch process cannot even `import torch`.** 2026-09-29, while a stage-3 run was training, a CPU-only inspection script (`CUDA_VISIBLE_DEVICES=""`) died at DLL load with `OSError: [WinError 1455] The paging file is too small ... loading c10.dll`. Measured at that moment: 15.9 GB physical with 5.4 GB free, but a **30.6 GB commit limit with 1.9 GB free**, and `S:\pagefile.sys` already at its configured 15000 MB maximum. Physical free memory therefore says nothing about whether a second process will start. So `machine_is_free.py` is not only about `pytest`: **nothing that imports torch runs beside a training job**, including a one-line `torch.load` of a checkpoint -- queue it for after the run. Raising the page file's maximum (or making it system-managed) would lift this; it is a machine setting, not a repo one.
  **Broken a third time on 2026-10-01, and the failure mode is worth knowing.** A one-line `python -c "import ObjectMemory; count parameters"` -- seconds of CPU, no GPU work -- was run beside a live stage-3 job to check a parameter count. The training process died 20 seconds later in `sklearn_auc` with `_ArrayMemoryError: Unable to allocate 1.93 MiB`, 45 minutes into a run, having just spent 1529 s rebuilding a cache. **The victim is whatever allocates next, not the process that took the memory**, so the traceback points somewhere innocent and reads like a bug in evaluation. `memory/scripts/machine_is_free.py` exists precisely to prevent this and takes one second to run; the rule is not "do not run pytest", it is **check before any torch process, however small**.
- **Never `tail -F` a file that a chain appends to: on Windows it locks the writer out, and stopping the watcher does not release it.**
  2026-10-02: a Monitor ran Git-bash `tail -n 0 -F progress.txt`; from then on every PowerShell `Add-Content` to that file failed with "being used by another process", so no chain could record a step, and chain B's wait for `CHAIN A DONE` would never have ended. `TaskStop` on the Monitor killed its shell but left `tail.exe` and `grep.exe` running and holding the file. Watch by polling instead -- `wc -l` then `sed -n` every 20 s, each opening the file for an instant -- and if a watcher is ever stopped, check for orphaned `tail`/`grep` before trusting that the file is free.
- **`os._exit` does not stop a finished ROCm process from hanging at exit.**
  2026-10-03: `tracking_baselines.py arms` wrote its last output at 05:31 and was still alive at 06:00, although its `__main__` ends in `os._exit`, as every script here does to dodge this hang. Its outputs were complete. A chain step should therefore not wait on the process alone: wait for the step's final log line, then stop the process if it has not exited two minutes later (that chain's `StepUntil`), so the chain moves on instead of idling for hours.
- **`evaluate()` on every annotated test frame spends ~20 minutes per checkpoint in scoring, not in the memory.**
  `score_detections` -> `_prec_rec_2d` scans every frame for every detection, so 50,088 frames x ~64 candidates (3.2M detections) is quadratic in practice; the memory replay itself takes about a minute. When only the frames are wanted, pass `frames_out` and `compute_metrics=False`, and score them with `tracking_baselines.PrecisionRecall`, which reproduces `score_detections`' AP exactly (a test pins it) in seconds.
- **A chain's wait pattern must not match lines left by an earlier attempt.**
  The same day, chain C waited for `CHAIN B DONE|CHAIN B STOPPED`, and `progress.txt` still held `CHAIN B STOPPED` from a failed first launch, so chain C started at once, beside a GPU run it was meant to follow. Give each launch a fresh progress file, or wait on a marker that only the current attempt can write.
- **Chain training runs with `&&`, never `;`.**
  The same 2026-09-16 chain used `;`, so the second run started even though the first had already crashed, and an hour of GPU went to an arm whose partner had produced nothing. `&&` stops the chain at the first failure, which is what every other chain in this project uses.
- **The FROG loader still clamps missing returns to 10 m by default, and misses one encoding.**
  `FROG_Dataset(missing_return_m=10.0)` is the legacy default every baseline number was measured with: non-finite readings become a real-looking 10 m, so any statistic over "valid" beams silently includes them. It never touches train/val's finite **61.0 m** no-return encoding. The three-horizon trainer passes `missing_return_m=None` and sanitises readings in the network (`sanitize_ranges`). A new analysis that wants the truth should do the same, or read `h5["scans"]` directly. The 1.30 cm consecutive-scan disagreement of 2026-09-16 is biased low for this reason.
- **The median scan interval misreads both datasets' rates: FROG is 40 Hz, not 26.2 Hz; DROW is 12.7 Hz, not 10 Hz.**
  DROW's timestamps are rounded to 0.05 s, so its intervals are a mix of 0.05 s and 0.10 s, and the median lands on 0.10.
  FROG's scans are stamped in bunches: about 0.038 s, 0.038 s, then under 0.1 ms. No two consecutive scans are identical, and the mean over linked intervals is 25.0 ms in every file, matching the UTM-30LX's 25 ms per scan. The median lands on 0.038 s.
  Use scans over elapsed time per unbroken sequence (`utils/scan_anomalies.py` reports both).
  **Never divide by a stamped FROG interval:** about a fifth of them are under 1 ms. `manage_slots` divides by its `dt`, and with stamped intervals it made matched slots' velocity median 4.0 m/s near people against 0.56 m/s measured over >= 1 s, and sent unmatched slots coasting at ~11 m/s. The stage-3 trainer now passes `frame_periods`, a running average of the intervals; anything new that needs a time step should use it.
  Scripts still on the median or on `1/26.2` are listed in `TODO.md`.
- **The GPU's state can go bad mid-run, and the giveaway is a number the model cannot influence.**
  2026-09-21, `a51_d1_sensor_s1`: at epoch 12 the *candidates'* validation AP fell from 72.85% to 3.40% while the training loss went to exactly 0.0000 and the rescore gate to +0.000. Those candidates are read from the cache, so no model state can change them -- the data on the device was corrupt. Six more epochs ran on garbage, then the process died with `HIP out of memory: tried to allocate 2.00 MiB` while reporting **14.87 GiB free**, which is itself a sign of a broken HIP context rather than real exhaustion. A sanity check afterwards (matmul against CPU, attention-shaped ops, `mem_get_info`) passed cleanly, so the fault is transient, not a dead card.
  **It was a GPU hang, found 2026-09-26:** Windows took a TDR live dump (`0x117`, `amdkmdag.sys`) at **23:06:26**, inside epoch 12's evaluation of that relaunch (started 22:19:22, ~3.9 min per epoch). A TDR resets the GPU engine, which destroys the process's device-side data, so everything after it was garbage.
  `train_memory` now stops at the first evaluation where the candidates' AP drifts by more than `CANDIDATE_AP_TOLERANCE` or the loss is zero or non-finite (`training_fault`). Reboot before the next long run, and do not read any result from a run that logged this.

- **An overflowing gradient norm is survivable; treating it as fatal is not.**
  2026-09-27: a guard stopped a healthy 5 h stage-3 run at its best epoch (val AP 80.88%) because `clip_grad_norm_` returned `inf`. The per-iteration diagnosis showed the loss at 0.814 and **no** non-finite output, parameter or gradient: one step's gradients were merely large enough that their squared sum overflowed float32. Torch then divides by that norm, so the update is zeroed and training continues by itself. `gradient_verdict` now skips such a step, logs it with the diagnosis, and stops only on a non-finite **loss** or more than `GRAD_SKIP_LIMIT` skips in one epoch.
  An interrupted run is not lost either: `utils/score_memory.py` scores its best checkpoint on the test split and writes the results file the run never reached.
  **A resumed run is a different draw, though.** 2026-09-29: an uninterrupted stage-3 rerun reproduces its seed exactly (46/46 evaluations), while a resumed one diverges at the first epoch after the resume point and ends 0.41 pp away in AP. `--resume` rescues the GPU time, not the run -- never report a resumed run as a reproduction of another, and never pair it seed-to-seed.

- **The GPU driver can bugcheck the whole machine under a long run.**
  2026-09-22 23:52: `0x7E SYSTEM_THREAD_EXCEPTION_NOT_HANDLED`, exception `0xc0000005` at `fffff803'3b8d4f50`, and Windows named the driver itself in event 1019: **`amdkmdag.sys`** (32.0.31041.1004, dated 2026-08-17). It killed a healthy 3.5 h stage-3 run at 42 epochs. It happened **again on 2026-09-25 22:29**, same bugcheck and same driver, killing a run at 3.9 h -- two in four days, both under sustained ROCm load, so treat it as a property of this machine rather than an incident.
  **WinDbg is installed** (Store package `Microsoft.WinDbg`, launcher `%LOCALAPPDATA%\Microsoft\WindowsApps\WinDbgX.exe`), but `C:\Windows\Minidump` is still unreadable without elevation, so the dumps from 09-22 and 09-25 remain unopened. Both dumps were analysed on 2026-09-26 (`results/windbg_09*.txt`) and they are **the same defect**: bugcheck `0x7E`, access violation, faulting instruction **`amdkmdag+0x74f50`** called from `amdkmdag+0x6bc452`, process `System`, bucket `AV_amdkmdag!unknown_function`, and an **identical failure hash `c753a2f9-94a6-448f-1d0e-e079b3442215`** both times. The stack holds only AMD frames -- nothing of ours -- and AMD ships no symbols, so `+0x74f50` is as precise as it gets. **No code change here can prevent it**; `--resume` caps each occurrence at one evaluation interval. Reproduce the analysis with `WinDbgX -z <dump> -c "!analyze -v; lmvm amdkmdag; q"` (WinDbg is installed as a Store package; the dumps need copying out of `C:\Windows\Minidump` from an elevated prompt first).
  **The full history (read 2026-09-26 from the System log and WER) shows a pattern, not a one-off.** Five `0x7E`/`c0000005` bugchecks, all on the `32.0.31xxx` driver branch that started with the 2026-05-08 install: 05-13, 06-29, then 09-01, 09-22 and 09-25 on 26.8.1 (31041.1004). The three on 26.8.1 fault at the **same offset** (`...4f50`, under a different KASLR base each boot), which is the signature of one deterministic code path, not of random memory corruption. TDR live dumps exist too, contrary to what this entry said before: `0x141` on 08-09 and 09-14, `0x117` on 07-27 and 09-21. The earlier `32.0.22xxx`/`23xxx` drivers (2025-12-22 to 2026-05-08, including 26.2.2 from 02-27 to 03-20) logged **no** bugcheck, though four unexpected shutdowns without a dump (03-20, 03-24, 04-02, 04-08) were never explained, and it never ran ROCm, which only arrived on 2026-09-14.
  **AMD requires exactly Adrenalin 26.2.2** (driver store `32.0.23027.2005`) for the ROCm 7.2.1 PyTorch wheels this repo pins ("must be installed", AMD's *Install PyTorch on Windows* page), so 26.8.1 was never a validated pairing. Reverting to 26.2.2 is queued in `TODO.md`.
  **The dumps, read 2026-09-26 with WinDbg (`cdb`, from the `Microsoft.WinDbg` winget package; the dumps were copied out of `C:\Windows\Minidump` by an elevated shell into `C:\Users\shvei\crashdumps`).** Both 09-22 and 09-25 are the *same* bug: a null-pointer read at `amdkmdag+0x74f50` (a four-byte getter, `mov eax,[rcx]`), called from `amdkmdag+0x6bc452`, on Windows' power-IRP worker thread: `nt!PopIrpWorker -> dxgkrnl!DpiDispatchPower -> DpiFdoHandleDevicePower -> DpiFdoSetAdapterPowerState -> DpiDxgkDdiSetPowerState -> amdkmdag`. So the driver crashes **while the GPU's power state is being changed**, not while computing. The machine is S3 with sleep and hibernate at *never*, but the display turns off after 5 min, the trainer only holds `ES_SYSTEM_REQUIRED`, and the driver has `PP_GPUPowerDownEnabled = 1` (BACO). On 09-25 the crash came 4 min 50 s after the last unlock. The two TDR dumps (`0x141`/`0x117`) blame the same point, `amdkmdag+0x1d0fb0`.
  **Display-off is not the trigger (tested 2026-09-26).** Ten minutes of ROCm matmul bursts with CPU-only gaps of 0.2-30 s, display off for ~9 of them (monitor timeout set to 1 min, no input for 616 s): the first kernel after a gap answered in 0.4-1.3 ms with the display off against 0.7-1.2 ms with it on, the GPU's reported D-state stayed D0 throughout (also in a 3-min control with no GPU process), a 256 MB reference tensor stayed intact, and nothing crashed. A chip in BACO would take tens of ms to wake. This agrees with the owner's experience that screen-off during training is routine. **What sent the power request is unknown:** the minidump holds only registers and stack (`!poaction`, `!irp` find nothing), because the machine wrote small dumps only (`CrashDumpEnabled = 3`; page file on `S:`, 5 GB fixed).
  **Kernel dumps are on since the 2026-09-26 05:58 boot** (`CrashDumpEnabled = 2`, `DedicatedDumpFile = C:\DedicatedDump.sys` 8 GB, `AlwaysKeepMemoryDump = 1`). After the next bugcheck, copy `C:\Windows\MEMORY.DMP` out from an elevated shell and run `!analyze -v`, then `!irp` on the power IRP and `!poaction` / `!popolicy`: that names the power request and the system's power action.
  **Next suspect: waking the display, not turning it off.** The owner was logging in to check progress both times, and the test above never woke the display under load. Since 09-14 the System log holds 401 display-wake or unlock transitions (Kernel-Power 566: 201 `SessionUnlock`, 195 `InputHid`) against 2 bugchecks, so if waking is the trigger it fires on well under 1% of wakes -- which is why screen use during training looks harmless. The monitor is an ASUS VG279Q3A on DisplayPort. Untested, and not to be tested with display cycling while a training run is live, since a hit would kill it.
  **It has not recurred since 2026-10-01, and the driver the pattern belonged to is no longer installed** (System log read 2026-11-03). The
  bugchecks are 2026-09-01, 09-22, 09-25 and **twice on 10-01** -- then nothing, 33 days, including the whole Li2Former programme of
  2026-11-01 to 11-03 (~30 h of ROCm training plus two CPU evaluation passes). This host now runs **`32.0.32015.7029`, dated 2026-10-18**,
  which is neither the `32.0.31041.1004` the five crashes sit on nor the `32.0.23027.2005` AMD names for the ROCm 7.2.1 wheels -- and those
  wheels demonstrably work on it, since they are what trained the port. **So do not read the five-crash pattern as a live risk on this
  driver, and do not read the absence as proof either**: the crashes stopped 17 days before the current driver arrived, so the two facts are
  not linked by anything measured. One event 41 on 2026-10-12 22:34 carries **no** accompanying bugcheck, so it is an unclean restart
  rather than this failure mode. Kernel dumps are still on; **the rule is to revert only if a `0x7E`/`c0000005` happens again**, and then with
  a dump in hand to confirm the power-state handler.
  Every trainer now takes `--resume` (`save_resume` / `load_resume`), which is the only defence available from inside the repo.

- **Running `pytest` beside a live training run kills the run, and it has now happened twice.**
  The rule is already below ("do not run the test suite against a live training run"), and on 2026-09-22 an agent broke it anyway: two `pytest` invocations during `a51_d1_sensor_s1` ended it after 3 h with `_ArrayMemoryError: Unable to allocate 64.0 MiB` inside `gather`, the documented commit-pressure signature. The tell was visible beforehand -- test collection took 199 s instead of 3 s. **Check for a running `python.exe` before any test run**, and treat a slow collection as a warning that something else is holding memory.

- **Stage 3 held all three candidate caches in RAM at once, and a 16 GB host cannot take it.**
  2026-09-21: `a51_d1_sensor_s1` died 3.5 h in with `RuntimeError: bad allocation` inside `loss.backward()` -- a *host* allocation failure, not GPU (5.3 GB of 16 GB used there), on a run whose seed-0 twin had finished. The FROG `official` caches are 614 MB (train), 998 MB (val) and 254 MB (test) on disk, and `train_memory` loaded all three before training although test is only scored at the end. It now loads test after the training loop and frees train and val first. The machine has 16 GB with a 5 GB page file (max 15 GB), so the margin is thin either way: check free memory before starting a long stage-3 run, and do not run anything else beside it.

- **`detect_device()` returns a string, not a `torch.device`.**
  So `device.type` raises `AttributeError`; ask `str(device).startswith("cuda")` instead, and note that DirectML returns a device object rather than either.
  2026-09-17: a `torch.cuda.synchronize()` guard written as `device.type == "cuda"` passed its unit test, which had handed it a real `torch.device`, and then crashed a smoke run *after* training finished. A launcher chain stopped there and the GPU sat idle for 19 hours.
  A unit test that constructs its own device does not cover this; the check is to run the path once against `detect_device()`'s own return value.

- **The machine can go to sleep in the middle of a training run, with the plugged-in sleep timeout set to "never".**
  2026-09-15: step 1 stopped at 10:28 with no error in its log; the System log shows "shutdown transition" and "system initiated reboot from Sleeping (Idle)" at 10:29:54, and no out-of-memory or bugcheck record.
  `utils/train_takheliped.py` now asks Windows to stay awake for as long as it runs (`SetThreadExecutionState`), which lasts only while the process lives and changes no power setting.
  **Re-read 2026-09-26: it was not sleep but a Windows Update restart.** Event 1074 at 10:29:09 shows `MoUsoCoreWorker.exe` initiating the restart on behalf of SYSTEM (update KB5129195 finished installing at 10:39); the "Sleeping (Idle)" wording in event 577 is misleading. `SetThreadExecutionState` does not stop that; the Windows Update active hours or a paused update do.
- **A chain launched with `schtasks /it` runs in a console window on the desktop, and input there ends it.**
  Twice in two days (2026-10-05 18:14, 2026-10-06 21:49) a chain and its trainer died together, with no traceback and no driver event, the task's result `0xC000013A` (`STATUS_CONTROL_C_EXIT`: Ctrl+C or the console closed), in the same second as a Kernel-Power 566 session transition with `Reason InputHid` -- keyboard or mouse input. The likely mechanism is a keystroke into, or a close of, the chain's visible PowerShell window; whether either was meant is not recorded. Nothing was lost either time, since every step resumes. Launch chains with `powershell.exe -NoProfile -WindowStyle Hidden ...` so no window can receive the input, and when a chain stops with this code, ask the owner before relaunching.
- **A finished ROCm process can hang at exit instead of terminating.**
  Seen 2026-09-15: a step-1 smoke run wrote its results, then sat for 2.8 hours burning CPU and holding 3.4 GB of commit memory, which starved every later process. A crash hangs the same way (seen the same day: a step-3b smoke run's traceback, then a live process). `utils/train_takheliped.py` now always leaves through `os._exit` -- 0 on success, 1 after printing the traceback -- so a launcher sees the exit code; still check for leftover `python.exe` before a long run. **`os._exit` does not always prevent it:** 2026-09-15, a test-only evaluation script that ends with `os._exit` wrote its results at 23:09:37 and was still alive at 23:12:55, when the chain launcher's watchdog stopped it. Keep launchers judging completion from the log, and stopping a process that lingers after it.
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
- **A ROCm run's first optimizer step can take 230-370 s with no output at all, and it looks exactly like a hang.**
  2026-11-01, Li2Former on ROCm: a `train.py` probe sat at `train: 0%| 0/27` for over ten minutes before the cause was
  found, and the process was burning **0.98 cores of a single thread** the whole time -- kernel compilation, not a stall.
  The cost is per **process** and per **batch shape**: batch 1 took 370 s and batch 2 another ~400 s in the same process.
  So a shape change mid-run is another silent multi-minute gap, and anything judging liveness on a clock will call it dead.
  Judge by the output file, and when in doubt measure the process's CPU rate -- a compiling process is pinned to ~1 core,
  a hung one is at 0. PyTorch also prints `Flash/Mem Efficient attention on Current AMD GPU is still experimental`
  (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` enables them), so attention runs the math path here: Li2Former's measured
  **259 ms/frame at batch 1 and 285 at batch 2** is ~2% of this card's fp32 peak, and batching does not help it.
- **tqdm's progress bar never reaches a redirected log unless Python is unbuffered.**
  Python's stderr is line-buffered, and tqdm writes `\r` without a newline, so a bar redirected to a file stays in the
  buffer: `2>&1 > log` gives a log frozen at `0%` while the job runs normally. Cost ten minutes of misdiagnosis on
  2026-11-01, on top of the compile above. Pass **`-u`** (as `run_queue.spawn_process` already does) and read the log with
  `tr '\r' '\n'`. A log without `-u` is not evidence of anything.
- **PyPI's `torchvision` installs against this torch and then cannot load.**
  `torchvision==0.24.1` from PyPI pairs with torch 2.9.1 by version but its compiled extension is built for CUDA, so the
  import dies with `RuntimeError: operator torchvision::nms does not exist` -- after installing cleanly. AMD ships a
  matching build at the same index as the torch wheel
  (`repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl`), and that one
  works. Install either with **`--no-deps`**, or pip may resolve torch itself and replace the ROCm build.
  It is needed because Li2Former's own `matcher.py` imports `box_area` at module scope, not by anything of ours.
- **`ruff check library/` never sees `takheliped/include/`, because ruff honours `.gitignore`.**
  That is why the downloaded GPL-3.0 Li2Former sources lint clean in CI and report 7 findings when pointed at directly.
  Convenient here -- those files must not be edited -- but it means a gitignored directory is unlinted, silently.
- **`.venv` has no DirectML, on purpose.**
  `torch-directml==0.2.5.dev240914` pins torch 2.4.1 and cannot share a venv with AMD's ROCm build (torch 2.9.1), so the one environment carries ROCm and its GPU appears as `torch.cuda`; `detect_device()` picks it up as CUDA/ROCm.
  Its CPU path is ~10x slower than torch 2.4.1's on stage 3's small per-step operations (`CHANGELOG.md` 2026-09-14), so CPU is no fallback for recurrent training.
- **`pip install -e ../library` can silently point at an unrelated old checkout.**
  If imports resolve to code that doesn't match what's on disk, `pip uninstall takheliped` and reinstall from the current path.
- **`library/takheliped/utils/torch_utils.py` is not where `detect_device()` lives.**
  That file is an old CUDA-only helper (`move()`, `init_module()`, credited to Lucas Eyer's `lbtoolbox`) and does not do CUDA->DirectML->CPU detection — the real `detect_device()` is in `utils/train.py`.
- **`LidarFrameDataset` caches every preprocessed frame, so a bigger split is a bigger RAM bill.**
  ~25 KB/frame in raw_scan mode: FROG's 108,356-frame training split is ~2.7 GB, and caching the 195,058-frame val split alongside it OOM'd a 17.1 GB machine outright (`CHANGELOG.md`, 2026-09-10).
  Val is never cached now.
  If you add a split or a dataset, check what a full pass over it costs *before* running — and check whether anything still consumes that pass at all.
  `--no-frame-cache` trades the RAM back for recomputation on every access.

- **`pip install onnx` silently upgrades numpy past this project's pin.**
  It pulls `ml_dtypes`, which requires `numpy>=2.0.0`, over `takheliped`'s `numpy~=1.24`.
  pip reports the conflict as a warning and installs anyway.
  Reinstall the pin straight after (`pip install "numpy~=1.24"`) and re-run the tests; `ml_dtypes` then warns in the other direction and is harmless, because nothing here imports it.

- **A literal `%` in an argparse `help=` string crashes `--help` and nothing else.**
  argparse `%`-formats every help string when it renders, so `"76.3% vs 78.8%"` raises `TypeError: %o format: an integer is required` — at `--help` time only.
  Every other test and every real invocation keeps passing, so it hides indefinitely.
  Double it: `76.3%%`. `tests/test_train_eval_consistency.py::test_both_clis_can_print_help` now catches it.

- **JRDB requires manual registration** (jrdb.erc.monash.edu) **and is never auto-downloaded.**
  Absence of `library/takheliped/include/JRDB-data/` is normal, not a broken install; every JRDB column in this project's own results is `n/a` for that reason, not a code limitation.
- **DirectML training crashes are not necessarily a code bug.**
  See `AGENTS.md` I4 for the three distinct crash modes hit training `Li2FormerDetector` — don't sink more than one workaround attempt into a DirectML-only crash before flagging it as an environment limitation.
- **`.venv` is not a self-contained interpreter — its `pyvenv.cfg` points `home` at `C:\Users\...\AppData\Local\Programs\Python\Python312`.**
  A process can end up running under that base path instead of `.venv\Scripts\python.exe` while still executing this project's code — observed three separate times now (a duplicate DR-SPAAM eval process; a duplicate `train.py --dtime 5` process, killed 2026-09-10; a third unrelated pair on a scratchpad script from a different session, same day), always as an exact command-line duplicate holding real CPU time under the base install's path — if `ps`/Task Manager shows two copies of the same script, check whether one is the base install before assuming it's unrelated, and kill both PIDs, not just the one matching the path you expected.
  Frequent enough now to treat as a standing property of how background processes get launched here, not a one-off fluke — still not root-caused to an exact trigger.
