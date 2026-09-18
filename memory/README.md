# Knowledge base

*keywords:* memory, knowledge base, index, growing memory, new doc

Everything that would otherwise inflate `AGENTS.md`, `CHANGELOG.md` or a code docstring.
Start with [`../AGENTS.md`](../AGENTS.md) for orientation; come here for depth.

## Rules and routing

| File | What's in it |
| --- | --- |
| [`keywords.md`](keywords.md) | Which document to read when, keyed by the words that appear in a request. |
| [`dos-and-donts.md`](dos-and-donts.md) | House rules: workflow, doc conventions, evidence standards, what to re-run after a change, what never gets committed. |
| [`coding-guidelines.md`](coding-guidelines.md) | How code is written here, per language, and the linter and config that enforce each set (mostly not-yet-enforced — see `TODO.md`). |
| [`gotchas.md`](gotchas.md) | Platform and environment traps that have cost real debugging time. |

## The system

| File | What's in it |
| --- | --- |
| [`detector-architectures.md`](detector-architectures.md) | How a scan becomes a detection, the three proposed architectures vs. the cutout baselines, `DETECTOR_REGISTRY` as source of truth, the tunables table. Full literature review and citations are in `../docs/RESEARCH.md`. |
| [`data-model.md`](data-model.md) | The DROW/FROG/JRDB file formats, FROG's session-splitting semantics, the real-vs-estimated odometry file formats, and what regenerating either invalidates. |
| [`dataset-properties.md`](dataset-properties.md) | What the *data* constrains, not what the files contain: how long people actually stay still, the ceiling that puts on any motion-based method, the frame-rate and density differences that make cross-dataset numbers incomparable, and what the false positives turn out to be. **Read before designing an architecture.** |
| [`noise-structure.md`](noise-structure.md) | What each dataset's range readings contain, what its sensor promises (sourced specs), what the loaders rewrite before a model sees it, and which denoising mechanism each kind of noise calls for. |
| [`deployment.md`](deployment.md) | The ROS/Docker pipeline on RobAIR: the one deploy command, the node reference table, and what silently isn't wired up yet. |
| [`commands.md`](commands.md) | Every routine invocation, with the flags actually used, in pipeline order. |

## Evaluation

| File | What's in it |
| --- | --- |
| [`interpreting-evaluation.md`](interpreting-evaluation.md) | How to discount an accuracy or speed number correctly — which regime it's from, what it doesn't correct for, what it's silently not comparable to. |
| [`performance-log.md`](performance-log.md) | The standing comparison: published SOTA against this project's own architectures, accuracy and speed, latest measurement only. Sweeps and history live in `CHANGELOG.md`. |
| [`informational-capacity-proxy.md`](informational-capacity-proxy.md) | A no-training AUC/mutual-information proxy for how much temporal signal a `(T, dtime)` configuration carries, with citations, and what it can't tell you. |
| [`static-detector-diagnosis.md`](static-detector-diagnosis.md) | Where a single-frame detector loses recall and to what: range-stratified error, the scale-vs-capacity answer from a no-training 1-NN probe over four window parameterisations, the sub-threshold miss budget, and the Cover & Hart bracket. **Read before proposing a static architecture.** |

## Tooling and operations

| File | What's in it |
| --- | --- |
| [`automation-scripts.md`](automation-scripts.md) | When to write a script instead of repeating a task, and the rules those scripts follow. |
| [`scripts/`](scripts/) | The scripts themselves. Start with `verify_memory.py`. |

## Decisions and history

| File | What's in it |
| --- | --- |
| [`rejected-ideas.md`](rejected-ideas.md) | Tested and rejected, with what would be needed to reopen. Check here before proposing. |
| [`changelog-archive/`](changelog-archive/) | Frozen `CHANGELOG-v*.md` from past releases. Grep here when a search of the live changelog turns up nothing. |

Step-by-step history for the current release is in [`../CHANGELOG.md`](../CHANGELOG.md).
Open work is in [`../TODO.md`](../TODO.md).

---

## Growing this knowledge base

The files above are the seed set plus what this project has already earned.
**Everything else is created when it is needed, not before.**

### When a fact earns a new file

A new file is justified when **all three** are true:

1. The fact is durable — it describes how things *are*, not what happened.
   What happened goes in `CHANGELOG.md`; what is open goes in `TODO.md`.
2. It has been re-derived twice, or cost more than an hour once.
3. It does not fit an existing file's scope.

### Files this project may still grow into

| Create | When | The question it must answer |
| --- | --- | --- |
| `parameters.md` | The tunables table in `detector-architectures.md` outgrows a single table. | What is this set to right now, and where is that value read? |
| `external-apis.md` | The FROG/DROW/DR-SPAAM data hosts (`robotics.upo.es`, GitHub release assets) produce more than a couple of surprises beyond the JRDB registration requirement already in `gotchas.md`. | What does this host do that its own listing does not say? |
| `regenerating-artifacts.md` | A committed file turns out to be build output rather than source (none is, as of this writing — everything generated is gitignored). | Which command rebuilds it, and what must match production for the result to be valid? |
| `experiments/<YYYY-MM>-<slug>.md` | Before running something with a specific predicted outcome, e.g. the deferred real-odometry `dtime` re-sweep. | What did I predict, and what will I do with each possible outcome? |

### Conventions every file here follows

- **A `*keywords:*` line** directly under the title, holding the symbols someone would grep.
- **Present tense only.**
  Dates and history belong in `CHANGELOG.md`.
- **Every rule carries its incident**, in a `*Because:*` clause.
- **Short.**
  One precise sentence beats three thorough ones.
- **One sentence per line**, and the rest of the markdown conventions in [`dos-and-donts.md`](dos-and-donts.md).
- **Rows in this index and in `keywords.md`**, added in the same commit as the file.

### When to split a file

Split when two of its sections answer questions asked by different people at different times.
Length alone is a weak signal; two audiences is a strong one.

### When to delete a file

If a file has been empty, or unopened, for a month, delete it and its two index rows.
Git has the history, and the criteria above will tell you if it is earned again.
