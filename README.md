# lidar-vision

A research project on **person detection from knee-height 2D LiDAR data**, of the kind carried by mobile service robots. We benchmark the published [DROW](https://arxiv.org/abs/1804.02463) and [DR-SPAAM](https://arxiv.org/abs/2004.14079) detectors, [Li2Former](https://doi.org/10.1109/TIM.2024.3420353), the ONNX [LFE-Peaks / LFE-PPN](https://arxiv.org/abs/2306.08531) baselines, and a classical rule-based clustering detector against three novel **full-scan, non-recursive CNN/TCN architectures** proposed here, which drop the fixed-size "cutout" every prior cutout-based method relies on and process the whole LiDAR scan at once. The full pipeline — detection, single-target tracking, and a follow-me behaviour — is also deployed end-to-end on a real mobile robot (RobAIR) over ROS.

Along the way, we found that the DROW benchmark's own published scores are substantially depressed by sparse, gap-ridden ground-truth annotations in the original dataset, rather than by any real limitation of learned detectors; the denser FROG and JRDB datasets close most of that gap. The full research write-up — architecture rationale, prior-art comparison, and evaluation tables — lives in [docs/RESEARCH.md](docs/RESEARCH.md).

This repository contains four things: a Python/C++ **library** of eight detector implementations and three dataset loaders; a set of **ROS Noetic nodes** for real-time detection, tracking and visualisation, deployable in Docker on a laptop or on the robot itself ([docs/ROS_IMAGE.md](docs/ROS_IMAGE.md)); the **training/evaluation scripts** used to produce the research results; and a couple of **notebooks** for interactive exploration. For a full map of the codebase and development commands, see [AGENTS.md](AGENTS.md).

## Quick start

```bash
# Install the Python library (also downloads bundled weights + datasets, ~2 GB)
pip install ./library

# Evaluate the published DROW / DR-SPAAM baselines — no training required
cd utils && pip install -r requirements.txt
python evaluate.py --dataset drow --drow --drspaam

# Train one of this project's own architectures
python train.py --detector spacetime_cnn --dataset frog --epochs 30

# Render a detection video
python render_video.py --dataset frog

# Run the comparison notebooks
make redrow-detector-test

# Build the C++ detector library
make build-lib

# Run the full ROS pipeline in Docker (laptop-only, no robot needed)
make launch-docker-local
```

See [AGENTS.md](AGENTS.md) for the complete command/CLI reference, and [docs/RESEARCH.md](docs/RESEARCH.md) / [docs/ROS_IMAGE.md](docs/ROS_IMAGE.md) for the research and deployment documentation respectively.

## License

[MIT](LICENSE)
