# Keyword index — what to read, and when

*keywords:* keyword index, routing, triggers, when to read, what to read

A lookup from **the words that show up in a request** to **the documents that answer them**.

## How to use it

Scan the request for any trigger below, then read what it points to.
Match generously: triggers are stems and near-synonyms, not exact strings.
If two rows match, read both; if none do, read nothing extra and proceed.

| Trigger words | Read |
| --- | --- |
| new idea, proposal, why don't we, have we tried, suggestion, improvement, dtime, wide window, consensus filter, merge the memory's output, NMS on slots | [`rejected-ideas.md`](rejected-ideas.md) |
| encoding, unicode, cp1252, path, shell, venv, python3, DirectML, ABI, import error, works on my machine, crash on startup, JRDB missing | [`gotchas.md`](gotchas.md) |
| long run, overnight, what's the progress, status report, monitor, watchdog, heartbeat, stale, hung, lingering process, relaunch, resume, background task, queue, launcher, run_queue | [`long-running-tasks.md`](long-running-tasks.md) |
| how do I run, command, invocation, flags, rebuild, entrypoint, train, evaluate, render video, pytest | [`commands.md`](commands.md) |
| how does a detector work, architecture, cutout, full-scan, TAKHeLiPeD, three-horizon, calibration network, object memory, SpaceTimeCNN, FullScanTCN, TemporalUNet, DrowDetector, DrSpaamDetector, LFE, tracker, SimpleTracker, DETECTOR_REGISTRY, BEAM_BATCH, input shape | [`detector-architectures.md`](detector-architectures.md) |
| is this number real, accuracy, AUC, AP, benchmark, how good is it, can we trust, zero-shot, DROW-native, fine-tuned, false positive, false negative, batch size changed the answer, eval-dtime, retracted, duplicate labels, double report | [`interpreting-evaluation.md`](interpreting-evaluation.md) |
| how good is it, state of the art, SOTA, wp-AUC, AP, ms/frame, compare architectures, published baseline, PeTra, Li2Former, LFE numbers, what is TAKHeLiPeD called, naming | [`performance-log.md`](performance-log.md) |
| dtime sweep, NMS sweep, threshold sweep, tracker sweep, what did this run score, past run | [`../CHANGELOG.md`](../CHANGELOG.md) |
| **the frozen architecture**, what the adopted model *is*, layer list, mermaid diagram, parametrisable vs fitted, proposed FROG parameters, why FROG was the development dataset, per-dataset parameter tables, the citable artifact | [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) |
| where are the trained weights, which checkpoint is the model, load the model, archived weights, release, pair stage 2 with stage 3, which seed goes with which, provenance, which commit produced this | [`../weights_archive/MANIFEST.md`](../weights_archive/MANIFEST.md), then the dated release's own manifest |
| tracking literature, related work, cite, citation, novelty, ByteTrack, OC-SORT, CBMOT, SORT baseline, track-before-detect, GOSPA, OSPA, HOTA, MOTA, identities, Deep Tracking, SPENCER, leg tracker | [`../docs/RESEARCH.md`](../docs/RESEARCH.md) §3.5 and §7.4 |
| why a published baseline has no row here, leg_detector, ROS people stack, Arras, Leigh, PeTra, not replicated, zero-shot baseline | [`../docs/RESEARCH.md`](../docs/RESEARCH.md) §2.2 |
| slots, object memory, stage 3, slot design, why this decision, ablation, room coordinates, value rule, fade time, association gate, coasting, rescoring, is it justified, learned slot state, feedback prior, re-injection, render_prior, oracle prior, feedback headroom, E10, E12, E13 | [`slot-design-evidence.md`](slot-design-evidence.md) |
| paper claims, claim ledger, pre-registration, what does the paper claim, tracking_baselines, beats a tracker, learned state vs rule, occlusion gain, first detection, end-to-end latency, tracking baseline, ByteTrack, CBMOT, rule scorer, E14, E15, E16, E20, E21, E22, GOSPA, bootstrap, where the gain sits, A58, A61, tracker re-run on the new recipe, --arms, summary, double report, duplicate labels | [`experiments/2026-09-tracking-claims.md`](experiments/2026-09-tracking-claims.md) |
| coarse convolution, coarse lags, no-temporal, default architecture, one frame per call, calibration horizon, do we need the coarse conv, e76aa94, restore the coarse convolution | [`rejected-ideas.md`](rejected-ideas.md) and `../TODO.md` A60 |
| informational capacity, mutual information, without training, zero-cost proxy, does this carry signal, plateau | [`informational-capacity-proxy.md`](informational-capacity-proxy.md) |
| scale vs capacity, distance normalisation, cutout, depth tunnel, range-stratified, why does it miss far people, sub-threshold, miss budget, Bayes error, 1-NN, Cover and Hart, why is stage 2 so good, why do we beat LFE, why do we beat DR-SPAAM, decoder, vote grid, nms_predicted_center, close pairs, label assignment, make_targets, label radius, weight averaging, SWA, stage2_probe, attribution ladder, F1, W, R, LFE's input, LFE's widths, DR-SPAAM's recipe, input_features, --select last | [`static-detector-diagnosis.md`](static-detector-diagnosis.md) and `../TODO.md` A59 |
| dataset format, one row is, .h5, .csv, .odom2, .npz, session, sequence, det_id, DROW format, FROG format, JRDB format, train/val/test split, guard band, leakage, recording pause, odometry quantization, duplicate labels, duplicate annotations, copied circles, unlinked annotations, de-duplicate | [`data-model.md`](data-model.md) |
| how long a window, time span, temporal horizon, receptive field, how fast do people move, stationary people, ceiling, why not just track, frame rate differs, people per frame, phantom persistence, is it a museum artifact, what should the architecture look like | [`dataset-properties.md`](dataset-properties.md) |
| noise, denoise, denoiser, missing return, NaN, infinity, sentinel, clamp, spike, jitter, outlier, sensor specs, rated range, laser model, UTM-30LX, S300, LMS500, scan anomalies | [`noise-structure.md`](noise-structure.md) |
| deploy, release, RobAIR, ROS, Docker, launch-docker, conf.env, follow-me, production, it works locally but | [`deployment.md`](deployment.md) |
| workflow, commit, refactor, is this ok to change, review, evidence, blast radius, what not to commit, overfitting, train vs val gap, early stopping | [`dos-and-donts.md`](dos-and-donts.md) |
| style, lint, formatting, type hints, imports, naming, ruff, shellcheck, hadolint, C++, pybind11 | [`coding-guidelines.md`](coding-guidelines.md) |
| again, every time, repetitive, by hand, automate, script this | [`automation-scripts.md`](automation-scripts.md) |
| ci, workflow, github actions, pipeline, build failed, red build | [`coding-guidelines.md`](coding-guidelines.md) |
| when did this change, who changed, regression, it used to, history, bug found and fixed | [`../CHANGELOG.md`](../CHANGELOG.md) and [`changelog-archive/`](changelog-archive/) |
| what's next, roadmap, open work, what's left, backlog | [`../TODO.md`](../TODO.md) |
