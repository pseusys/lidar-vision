# Keyword index — what to read, and when

*keywords:* keyword index, routing, triggers, when to read, what to read

A lookup from **the words that show up in a request** to **the documents that answer them**.

## How to use it

Scan the request for any trigger below, then read what it points to.
Match generously: triggers are stems and near-synonyms, not exact strings.
If two rows match, read both; if none do, read nothing extra and proceed.

| Trigger words | Read |
| --- | --- |
| new idea, proposal, why don't we, have we tried, suggestion, improvement, dtime, wide window, consensus filter | [`rejected-ideas.md`](rejected-ideas.md) |
| encoding, unicode, cp1252, path, shell, venv, python3, DirectML, ABI, import error, works on my machine, crash on startup, JRDB missing | [`gotchas.md`](gotchas.md) |
| how do I run, command, invocation, flags, rebuild, entrypoint, train, evaluate, render video, pytest | [`commands.md`](commands.md) |
| how does a detector work, architecture, cutout, full-scan, SpaceTimeCNN, FullScanTCN, TemporalUNet, DrowDetector, DrSpaamDetector, LFE, tracker, SimpleTracker, DETECTOR_REGISTRY | [`detector-architectures.md`](detector-architectures.md) |
| is this number real, accuracy, AUC, AP, benchmark, how good is it, can we trust, zero-shot, DROW-native, fine-tuned, false positive, false negative | [`interpreting-evaluation.md`](interpreting-evaluation.md) |
| dtime sweep, NMS sweep, threshold, tracker sweep, wp-AUC, ms/frame, compare architectures, PeTra, Li2Former numbers | [`performance-log.md`](performance-log.md) |
| dataset format, one row is, .h5, .csv, .odom2, .npz, session, sequence, det_id, DROW format, FROG format, JRDB format | [`data-model.md`](data-model.md) |
| deploy, release, RobAIR, ROS, Docker, launch-docker, conf.env, follow-me, production, it works locally but | [`deployment.md`](deployment.md) |
| workflow, commit, refactor, is this ok to change, review, evidence, blast radius, what not to commit | [`dos-and-donts.md`](dos-and-donts.md) |
| style, lint, formatting, type hints, imports, naming, ruff, shellcheck, hadolint, C++, pybind11 | [`coding-guidelines.md`](coding-guidelines.md) |
| again, every time, repetitive, by hand, automate, script this | [`automation-scripts.md`](automation-scripts.md) |
| ci, workflow, github actions, pipeline, build failed, red build | [`coding-guidelines.md`](coding-guidelines.md) |
| when did this change, who changed, regression, it used to, history, bug found and fixed | [`../CHANGELOG.md`](../CHANGELOG.md) and [`changelog-archive/`](changelog-archive/) |
| what's next, roadmap, open work, what's left, backlog | [`../TODO.md`](../TODO.md) |
