# Gotchas

*keywords:* encoding, unicode, cp1252, DirectML, venv, python3, ABI mismatch, pip install -e, JRDB, torch_utils

Platform and environment traps that have cost real debugging time.

- **`UnicodeEncodeError` on any non-ASCII character printed to the console.**
  Windows' default console codepage is cp1252, not UTF-8; a `→`, `✓`, or similar in a `print()` crashes mid-run, sometimes after the expensive part of the operation already succeeded (see `AGENTS.md` I3) — use ASCII (`->`, plain words) in anything printed here.
- **`python3` resolves to the Windows Store's "install Python" stub on this machine, not a real interpreter.**
  `make venv` / `make test` invoke it and will not create a usable environment here — use `.venv312/Scripts/python.exe` directly (see `AGENTS.md` I5); fixing the Makefile is tracked in `TODO.md`.
- **`ModuleNotFoundError: No module named 'follow_the_drow.cpp_binding'`.**
  The pybind11 C++ extension did not build — re-run `pip install -e ./library` (builds it via `Pybind11Extension` in `library/setup.py`) or `make build-lib` from the repo root.
- **A Python version newer than 3.12 breaks the prebuilt C++ extension with an ABI mismatch.**
  The bundled `.pyd`/`.so` is built against CPython 3.12's ABI; a 3.13/3.14 interpreter loads it and fails confusingly rather than refusing cleanly — always use a 3.12 interpreter (`.venv312`) for this project.
- **`pip install -e ../library` can silently point at an unrelated old checkout.**
  If imports resolve to code that doesn't match what's on disk, `pip uninstall follow_the_drow` and reinstall from the current path.
- **`library/follow_the_drow/utils/torch_utils.py` is not where `detect_device()` lives.**
  That file is an old CUDA-only helper (`move()`, `init_module()`, credited to Lucas Eyer's `lbtoolbox`) and does not do CUDA->DirectML->CPU detection — the real `detect_device()` is in `utils/train.py`.
- **JRDB requires manual registration** (jrdb.erc.monash.edu) **and is never auto-downloaded.**
  Absence of `library/follow_the_drow/include/JRDB-data/` is normal, not a broken install; every JRDB column in this project's own results is `n/a` for that reason, not a code limitation.
- **DirectML training crashes are not necessarily a code bug.**
  See `AGENTS.md` I4 for the three distinct crash modes hit training `Li2FormerDetector` — don't sink more than one workaround attempt into a DirectML-only crash before flagging it as an environment limitation.
- **`.venv312` is not a self-contained interpreter — its `pyvenv.cfg` points `home` at `C:\Users\...\AppData\Local\Programs\Python\Python312`.**
  A process can end up running under that base path instead of `.venv312\Scripts\python.exe` while still executing this project's code (observed: a duplicate eval process, same script, different interpreter path, holding all the real CPU time) — if `ps`/Task Manager shows two copies of the same script, check whether one is the base install before assuming it's unrelated.
