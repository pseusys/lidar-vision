# Automation scripts

*keywords:* automation, scripts, verify_memory, repetitive, tooling, helper script

Recurring chores live in [`scripts/`](scripts/) as small Python programs, not in an agent's short-term memory.
This is about maintaining the knowledge base itself — the project's own research scripts (`utils/generate_frog_odometry.py`, `utils/nms_sweep.py`, `utils/error_breakdown.py`, and similar) are documented in [`commands.md`](commands.md) and [`detector-architectures.md`](detector-architectures.md) instead, since they belong to the research, not to working on the docs.

## The rule

**The second time you are about to do the same multi-step chore by hand, write a script instead.**
Not the first — you do not yet know which parts are stable.

Check [`scripts/`](scripts/) first; the chore may already be there.

## What belongs in `scripts/`

Chores that are **repetitive, mechanical, and verifiable**:

- checks that answer yes or no about the repo's own state (links, indexes, generated files being current)
- summaries and reports that are assembled the same way every time
- setup and teardown sequences that are easy to get subtly wrong

Not here: anything needing judgment, anything run once, and anything belonging to the project's research code rather than to working on it.

## The rules those scripts follow

**Python 3, standard library only.**
A script needing a dependency will not run on a fresh clone.

**One job per script, named for the job.**
`verify_memory.py`, not `utils.py`.

**A module docstring saying what it does, how to run it, and from where.**
That is the documentation; do not write a second copy elsewhere.

**Read-only by default.**
One that modifies files says so in its name, or requires a flag.

**Exit non-zero on failure**, so it works in a hook or in CI.

**Report what passed, not only what failed.**
A silent success is indistinguishable from a script that did nothing.

**Deleted when the chore stops existing.**

**Linted like everything else.**
[`scripts/ruff.toml`](scripts/ruff.toml) governs this directory:

```bash
ruff check --config memory/scripts/ruff.toml memory/scripts/
```

## The scripts that exist

| Script | What it does |
| --- | --- |
| [`scripts/verify_memory.py`](scripts/verify_memory.py) | Checks that internal links resolve, that the two indexes match the files on disk, and that the markdown house rules hold. Run it before committing a documentation change. |
| [`scripts/ruff.toml`](scripts/ruff.toml) | Not a script: the lint config for this directory. |
