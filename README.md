# lidar-vision

A research project on **person detection from knee-height 2D LiDAR data**, of the kind carried by mobile service robots.
It benchmarks the published DROW and DR-SPAAM detectors, Li2Former, the ONNX LFE-Peaks/LFE-PPN baselines, and a classical rule-based detector against three novel **full-scan, non-recursive CNN/TCN architectures** proposed here, which drop the fixed-size "cutout" every cutout-based method relies on and process the whole LiDAR scan at once — the full pipeline (detection, tracking, a follow-me behaviour) also runs end-to-end on a real mobile robot (RobAIR) over ROS.

## How it works, in one paragraph

A raw scan (or `T` consecutive scans) is preprocessed either as a fixed-width per-beam polar cutout (the cutout-based baselines) or as the whole raw scan at once (this project's own architectures), optionally ego-motion-compensated with odometry; a shared per-beam head then produces a classification plus a 2D vote offset (or a heatmap), which is clustered into discrete person/wheelchair/walker detections; an optional SORT-style tracker turns that per-frame list into one persistent, ID-stable position, which the ROS deployment's follow-me behaviour consumes.
See [`AGENTS.md`](AGENTS.md) for the full pipeline breakdown and [`memory/detector-architectures.md`](memory/detector-architectures.md) for how each detector family differs.

## Where things stand

- **DROW-native training underperforms zero-shot FROG-to-DROW transfer, for every one of the three proposed architectures** — a data-scarcity artifact of DROW's ~17.8x smaller annotation count, not evidence the architectures don't transfer to DROW's sensor geometry.
- **False positives, not false negatives, are the dominant error on FROG for every model tested**, including the baselines.
- **Best measured result: `SpaceTimeCNN` at 80.7% wp-AUC on FROG** (`dtime=5`), ahead of every cutout baseline evaluated here — but this number predates a just-fixed real-odometry pipeline bug and needs re-verification before it's trusted as final (`TODO.md` A1).
- An independent, published-weights verification of DR-SPAAM is in progress, specifically so this project's own zero-shot-transfer numbers have something to check against beyond the original paper.

Full tables and the regime each number came from: [`memory/performance-log.md`](memory/performance-log.md) and [`memory/interpreting-evaluation.md`](memory/interpreting-evaluation.md).
The full research write-up — literature review, architecture rationale, citations — is [`docs/RESEARCH.md`](docs/RESEARCH.md).

## Setup

```bash
# Install the Python library (also downloads bundled weights + datasets, ~2 GB)
pip install ./library

# Research script dependencies
pip install -r utils/requirements.txt
```

No secrets to configure — this project has none (`AGENTS.md`).
GPU is auto-selected CUDA/ROCm -> DirectML -> CPU; see `memory/gotchas.md` before assuming DirectML training will just work.

## Useful commands

```bash
python evaluate.py --dataset drow --drow --drspaam           # evaluate published baselines, no training
python train.py --detector spacetime_cnn --dataset frog --epochs 30   # train one of this project's own architectures
python render_video.py --dataset frog                          # render a detection video
.venv/Scripts/python.exe -m pytest tests -q                 # run the test suite (63 tests, ~6s)
make launch-docker-local                                        # run the full ROS pipeline in Docker, laptop only
```

The complete, current reference — every flag, every phase — is [`memory/commands.md`](memory/commands.md).

## Documentation

- [`AGENTS.md`](AGENTS.md) — orientation, repo layout, and the working rules
- [`memory/`](memory/README.md) — the knowledge base: how things work now, the house rules, the environment traps, and what was tried and rejected
- [`CHANGELOG.md`](CHANGELOG.md) — what changed, when, and the evidence
- [`TODO.md`](TODO.md) — what's open
- [`docs/RESEARCH.md`](docs/RESEARCH.md) — the full research write-up, for a reader outside this repo's day-to-day work
- [`docs/ROS_IMAGE.md`](docs/ROS_IMAGE.md) — building and running the ROS/Docker deployment

## Deployment

The ROS pipeline reaches RobAIR via a manually-triggered Docker rebuild, not a CI auto-deploy: `make launch-docker-local` (laptop) or `make launch-docker-robot` (the real robot).
Only `DrowDetector` is currently wired into it — see [`memory/deployment.md`](memory/deployment.md) for what runs, what doesn't, and why.

## License

[MIT](LICENSE)
