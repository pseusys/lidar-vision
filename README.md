# lidar-vision — TAKHeLiPeD

A research project on **person detection from knee-height 2D LiDAR data**, of the kind carried by mobile service robots.
Its contribution is **TAKHeLiPeD**, a streaming detector that carries an explicit, learned **object memory** across seconds to a minute of scans, so that a chair the robot has watched for half a minute stops looking like a person.
It is benchmarked against the published DROW3, DR-SPAAM, Li2Former and LFE-Peaks/LFE-PPN detectors on FROG, and against a classical, non-learned reference point — `AlgorithmicDetector`, **prof. O. Aycard's** leg+chest clustering-and-tracking algorithm, which this repo ports and runs but did not devise.
The surrounding pipeline (detection, tracking, a follow-me behaviour) also runs end-to-end on a real mobile robot (RobAIR) over ROS.

## The name

**TAKHeLiPeD** — **T**(emporal) **A**(daptive) **K**(nee-)**He**(ight) **Li**(dar) **Pe**(rson) **D**(etector).
Pronounced *TA-KHé-Li-PeD*.

It names the detector that the older documents call "the three-horizon detector"; code identifiers (`three_horizon.py`, `train_three_horizon.py`, `checkpoints_three_horizon/`) keep the old spelling, so every checkpoint, cache and log path still matches.

## How it works, in one paragraph

`detections, state = model.step(scan, odometry, state)` — one raw scan in, a detection list and a fixed-size state out, at constant cost per frame.
A **calibration network** (1.18 M parameters, a 1D ConvNeXt U-Net over the whole scan with causal temporal convolutions at the bottleneck) turns 10 pose-invariant features per beam into a DROW-style vote grid, from which 64 candidates are decoded, sub-threshold ones included; an **object memory** (0.31 M parameters, 256 slots in room coordinates, cross-attention association plus a Δt-aware selective recurrence with decay times of 1-60 s) then rescores those candidates, coasts briefly-missed people, and can render its slots back into the calibration network.
The three temporal horizons the design is built on — calibration under a second, object permanence over 1-10 s, furniture-versus-person over 10-60 s — are argued in [`docs/PROPOSAL.md`](docs/PROPOSAL.md) and measured in [`docs/PAPER.md`](docs/PAPER.md).
For how this differs from the cutout-based baselines, see [`memory/detector-architectures.md`](memory/detector-architectures.md) and [§3 of the research write-up](docs/RESEARCH.md#3-previous-research-and-baselines).

## Where things stand

**FROG `official` test split, person class, AP at a 0.5 m match radius** — the published benchmark, every annotated frame of the test recording.

| Model | History used | AP @ 0.5 m | FP / frame |
| --- | --- | --- | --- |
| LFE-Peaks / LFE-PPN *(published)* | none | 65.6% / 69.2% | — |
| DROW3 *(published)* | one scan | 73.9% | — |
| DR-SPAAM `T=5` *(published)* | ~0.15 s | 75.6% | — |
| **TAKHeLiPeD, calibration network alone** | ~1 s | **77.7%** | 2.63 |
| **TAKHeLiPeD, + object memory** | **seconds to a minute** | **83.3%** | **1.09** |

Published rows are the papers' own figures at the same match radius; ours are measured here.
The memory row is the three-seed mean of the B0 baseline (82.96 / 83.17 / 83.88%, sd 0.47), sitting on the candidates in the row above it (77.66%; that backbone's own three-seed mean is 77.2%).
False positives are per frame at the 0.3 operating point.
**Speed**: the calibration network plus candidate decoding streams at **~10.4 ms/frame** and the memory adds **~4.4 ms**, one scan at a time on a Radeon RX 9060 XT under ROCm, against a sensor that delivers a scan every ~25 ms.

What the memory is, and is not:

- **The gain is the learned memory, not temporal filtering.** SORT on the same candidates *lowers* AP at every setting, and persistence maps, background subtraction and hindsight trail filters all remove fewer false positives per lost person than simply widening the merge radius.
- **Capacity is not the limit**: the memory uses a median of 24 of its 256 slots, 107 at most.
- **Reporting is.** Of the 26,996 people it does not report at 0.3, a live slot already sits on 95.5% — the head cannot yet separate a remembered person from a remembered phantom.
- **Still open**: precision at that operating point is 70%, so nearly a third of reported detections are false; and training both stages jointly with feedback has not yet beaten training the memory on a frozen backbone.

Full tables, the regime behind every number, and the caveats that stop two of them being compared: [`memory/performance-log.md`](memory/performance-log.md) and [`memory/interpreting-evaluation.md`](memory/interpreting-evaluation.md).
A short standalone summary of the result is [`docs/SHOWCASE.md`](docs/SHOWCASE.md).

## Documentation

The research write-up, [`docs/RESEARCH.md`](docs/RESEARCH.md), is the reference for everything below:

| Question | Section |
| --- | --- |
| What problem this is, and how it is scored | [§1 Problem statement](docs/RESEARCH.md#1-problem-statement) |
| Which detectors are replicated, which are novel, and how faithfully | [§2.1 Provenance](docs/RESEARCH.md#21-provenance-replicated-vs-novel) |
| How DROW, DR-SPAAM, Li2Former and LFE actually work | [§3 Previous research and baselines](docs/RESEARCH.md#3-previous-research-and-baselines) |
| Preprocessing: features, coordinates, odometry alignment, cutouts | [§4 Preprocessing](docs/RESEARCH.md#4-preprocessing) |
| Architectures that were tried and dropped, with the reasons | [§6 Design history](docs/RESEARCH.md#6-design-history-architectures-tried-and-dropped) |
| Where this sits in the literature | [§7 Related work](docs/RESEARCH.md#7-related-work) |

The rest:

- [`AGENTS.md`](AGENTS.md) — orientation, repo layout, and the working rules
- [`memory/`](memory/README.md) — the knowledge base: how things work now, the house rules, the environment traps, and what was tried and rejected
- [`docs/PROPOSAL.md`](docs/PROPOSAL.md) — TAKHeLiPeD's design document: the three horizons, the slot rules, the configuration and the build plan
- [`docs/PAPER.md`](docs/PAPER.md) — the measurements the design is argued from, and how to reproduce each one
- [`docs/SHOWCASE.md`](docs/SHOWCASE.md) — the result on one page
- [`docs/ROS_IMAGE.md`](docs/ROS_IMAGE.md) — building and running the ROS/Docker deployment
- [`CHANGELOG.md`](CHANGELOG.md) — what changed, when, and the evidence
- [`TODO.md`](TODO.md) — what's open

**On older documents.** Three earlier full-scan architectures (`SpaceTimeCNNDetector`, `FullScanTCNDetector`, `TemporalUNetDetector`) preceded TAKHeLiPeD and are still in the registry, but every accuracy number they carried is **retracted pending re-measurement** — they are documented in [§5](docs/RESEARCH.md#5-proposed-architectures) and [§6](docs/RESEARCH.md#6-design-history-architectures-tried-and-dropped) for provenance, not quoted as results.

## Setup

```bash
py -3.12 -m venv .venv                                                       # 3.12 explicitly; see memory/gotchas.md
.venv/Scripts/python.exe -m pip install --no-cache-dir -r requirements.txt   # library + datasets + bundled weights, ~2 GB
```

No secrets to configure — this project has none (`AGENTS.md`).
GPU is auto-selected CUDA/ROCm -> DirectML -> CPU; read [`memory/gotchas.md`](memory/gotchas.md) before assuming DirectML training will just work.

## Useful commands

Run the training and evaluation scripts from `utils/` — their relative paths (`../checkpoints`, `../results`) assume it.

```bash
python train_three_horizon.py step2                                  # TAKHeLiPeD stages 1-2, the calibration network
python train_three_horizon.py step3a --stage2 ../checkpoints_three_horizon/step2_calibration.best.pth   # the object memory on its candidates
python evaluate.py --dataset frog --lfe-peaks --lfe-ppn              # published baselines, no training
python render_video.py --dataset frog                                # render a detection video
```

```bash
.venv/Scripts/python.exe -m pytest tests -q   # from the repo root: 533 tests, ~60 s, CPU-only, no GPU or dataset needed
make launch-docker-local                      # the full ROS pipeline in Docker, laptop only
```

The complete, current reference — every flag, every phase — is [`memory/commands.md`](memory/commands.md).

## Deployment

The ROS pipeline reaches RobAIR via a manually-triggered Docker rebuild, not a CI auto-deploy: `make launch-docker-local` (laptop) or `make launch-docker-robot` (the real robot).
**Only `DrowDetector` is currently wired into it** — TAKHeLiPeD, the tracker and the real-odometry pipeline have never been integrated into a ROS node.
See [`memory/deployment.md`](memory/deployment.md) for what runs, what doesn't, and why.

## Credits

- **`AlgorithmicDetector`** — the classical leg+chest clustering-and-tracking detector in `library/cpp_core/` is **prof. O. Aycard's** algorithm and implementation (`sources/detector.cpp`, *"Person detector using 2 lidar data — Written by O. Aycard"*). It must be credited to him in anything published from this repository; only the pybind11 binding, the Python wrapper and the evaluation harness around it are this project's.
- **DROW3, DR-SPAAM, Li2Former, LFE-Peaks and LFE-PPN** are their authors' work, reproduced here from published weights or descriptions; each is credited with its papers in [§3 of the research write-up](docs/RESEARCH.md#3-previous-research-and-baselines).
- **TAKHeLiPeD, and the evaluation and training pipeline around all of the above**, are this project's own.

## License

[MIT](LICENSE)
