# Coding guidelines

*keywords:* style, lint, ruff, shellcheck, hadolint, imports, type hints, magic numbers, line length, pybind11, CMake

How code is written here, per language.
Workflow rules — what to edit, when to abstract, what to re-run — are in [`dos-and-donts.md`](dos-and-donts.md); this file is about the code itself.

**None of the linters below are wired into CI yet** (`TODO.md` §A) — the rules are written as the intended standard, enforced today only by review.

## Everywhere

- **Generate nothing that is not used.**
  No functions, constants, variables, types or traits that nothing calls.
- **Comments describe the code, never how it came to be.**
  No notes about the generation process, no "updated to handle X", no changelog in a docstring.
  History goes in [`../CHANGELOG.md`](../CHANGELOG.md), reasoning in `memory/`.
- **One documentation comment per file, per type, per function.**
  Not per line, and it points at `memory/` for the why.
- **No backward-compatibility or migration code**, unless it is explicitly asked for.
  Delete the old path in the same commit as the new one.
- **Named constants instead of magic numbers.**
  A literal that appears in a comparison, or twice anywhere, is a constant that has not been named yet.
- **Imports at the top of the file, never inline.**
  Import individual names rather than modules, and refer to them directly rather than by a fully-qualified path.
- **Prefer importing over invoking.**
  When one file in this project needs another, import it; do not shell out to it, re-exec it, or read its output.
- **Prefer long lines to broken ones**, wherever the linter does not specifically enforce otherwise.
  The exception is markdown, which is one sentence per line — see [`dos-and-donts.md`](dos-and-donts.md).
- **Test code lives in its own tree** (`tests/`), mirroring the source tree where practical.
  Test-only helpers never sit in the sources they exercise.
- **Do not embed generated content in code.**
  Configuration, environment files and fixtures live as separate template files, not as string literals inside a program.

## Python

- **Type hints wherever the language allows them.**
- **`from module import name`, not `import module`**, so call sites read as the name and not the path.
  Standard-library imports follow the same rule, and go at the top with everything else.
- **Constants at module level, in `UPPER_SNAKE_CASE`**, above the first function.
- **GPU device selection always goes through `detect_device()`** (`utils/train.py`) — CUDA/ROCm, then DirectML, then CPU.
  Never hardcode `device="cpu"` in new code outside a one-off script that has an explicit reason to (see `commands.md`'s note on `evaluate.py`'s own CPU-only convention).
- **Trainable detectors stay CNN/TCN-only — no RNN, no attention.**
  This is the research's own scope boundary (`docs/RESEARCH.md` §2, constraint 1), not an oversight; don't reintroduce recurrence or attention into `spacetime_cnn`/`fullscan_tcn`/`temporal_unet` without checking with the owner first.
- **Intended linter: `ruff`.** Not yet configured — see `TODO.md`.

## C++

- **C++17, single namespace `follow_the_drow`** for all library code (`library/cpp_core/`).
- **pybind11 bindings (`binding.cpp`/`binding.hpp`) are excluded from the standalone C++ distribution** — `CMakeLists.txt` removes them from `SOURCES`/`HEADERS` before building the installable library, so the pure-C++ `FollowTheDrow` target never depends on pybind11 or Python.
- **The ROS build links against the compiled standalone library — it does not recompile a copy of it.**
  `deploy/follow_the_drow/CMakeLists.txt` does `find_package(FollowTheDrow CONFIG REQUIRED)` and links every node against it; `AlgorithmicDetector`'s actual logic lives only in `library/cpp_core/sources/`, and a change there reaches the robot on the next `make build-lib` + catkin rebuild with nothing to keep in sync by hand.
- **Intended linter: none chosen yet.** `clang-tidy` is the natural default if one is adopted — see `TODO.md`.

## Shell

- `deploy/docker/entrypoint.sh` is the only shell script in the repo, three lines: source ROS, `catkin_make`, exec the given command.
- **Intended linter: `shellcheck`.**
  Not yet configured (`.shellcheckrc` doesn't exist yet) — see "No config for a language nobody writes" below.

## Dockerfiles

- `deploy/docker/Dockerfile` has two targets: `basic` (ROS Noetic desktop-full + GDB + catkin workspace) and `lidar-vision` (adds this project's own package on top).
- **Intended linter: `hadolint`.**
  Not yet configured (`.hadolint.yaml` doesn't exist yet) — see "No config for a language nobody writes" below.

## Markdown

- **One sentence per line, no hard wrapping.**
- **Linted with `markdownlint` and `verify_memory.py`.**
  The full rule set and the reasoning behind it are in [`dos-and-donts.md`](dos-and-donts.md).

## GitHub Actions

Two workflows exist: [`../.github/workflows/build-lidar-vision-image.yml`](../.github/workflows/build-lidar-vision-image.yml) (builds/publishes the ROS Docker image, the Python library, and the C++ library, on every push touching `deploy/docker/**`, `library/**` or the `Makefile`), and [`../.github/workflows/lint.yaml`](../.github/workflows/lint.yaml) (lints this knowledge base, added alongside this documentation layout).

- **Run on every branch**, scoped with `paths` rather than a branch filter.
- **Declare `concurrency`** keyed on the workflow and the ref; cancel superseded runs on side branches, leave `main` alone.
- **`runs-on: ubuntu-latest`** for anything platform-independent.
- **Linted with `actionlint`**, configured in [`../.github/actionlint.yaml`](../.github/actionlint.yaml).

## Linters

Every language in the project gets a row.
A language with no row is a language with no enforcement, which means the rules above are advisory for it.

| Language | Tool | Config | Command | Status |
| --- | --- | --- | --- | --- |
| Python (`memory/scripts/`) | `ruff` | [`scripts/ruff.toml`](scripts/ruff.toml) | `ruff check --config memory/scripts/ruff.toml memory/scripts/` | enforced |
| Python (`library/`, `utils/`) | `ruff` | none yet | — | not yet enforced (`TODO.md`) |
| Markdown | `markdownlint`, `verify_memory.py` | [`../.markdownlint.jsonc`](../.markdownlint.jsonc) | `python memory/scripts/verify_memory.py` | enforced (CI); `--strict` (style warnings fail too) is available locally but not run in CI — `docs/RESEARCH.md` predates the one-sentence-per-line convention (`dos-and-donts.md`) |
| C++ | none chosen | — | — | not yet enforced (`TODO.md`) |
| Shell | `shellcheck` | none yet | — | not yet enforced (`TODO.md`) |
| Dockerfile | `hadolint` | none yet | — | not yet enforced (`TODO.md`) |
| GitHub Actions | `actionlint` | [`../.github/actionlint.yaml`](../.github/actionlint.yaml) | `actionlint` | enforced (CI) |

### No config for a language nobody writes

**A linter config is committed in the same commit as the first file it lints, and not before.**
`deploy/docker/entrypoint.sh` and `deploy/docker/Dockerfile` already exist without one — their configs (`.shellcheckrc`, `.hadolint.yaml`) and the CI jobs that run them are queued in `TODO.md` rather than added speculatively in this pass.

## Enforcing this — the worked example

[`scripts/`](scripts/) is linted by the rules above, and is the demonstration that they hold on real code rather than only in this document.

```bash
ruff check --config memory/scripts/ruff.toml memory/scripts/
```

The same command runs in CI, in [`../.github/workflows/lint.yaml`](../.github/workflows/lint.yaml).

**Adding a language's enforcement means adding all four things at once**: the linter table row above flips to "enforced", a config file lands, the command goes into [`commands.md`](commands.md), and a job goes into the lint workflow.
Three out of four is a rule nobody runs.
