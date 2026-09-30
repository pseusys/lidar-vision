"""How much of stage 2's bottleneck does the memory's feedback prior actually carry? (`TODO.md` A51 E12)

Three E12 arms returned a null while their trained prior weights looked large -- up to 44% of the rest of the layer by rms. That
comparison is misleading, and this says why with a measurement rather than an argument. `aggregate` is a 1x1 convolution over
`cat([x, x.amax(), prior])`: **2*c2 = 256 body channels against 2 prior channels**, and `render_prior` marks one sector per
person, so the prior is non-zero in about 2% of them. A per-weight rms says nothing about a per-contribution share.

    python measure_prior_share.py --runs a51_e12b_oracle a51_e12a_norescore

Needs the machine to itself, like anything else that imports torch here (`memory/gotchas.md`).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import conv1d

from follow_the_drow.datasets.frog_dataset import frog_laser_angles
from follow_the_drow.detectors.three_horizon import beam_features
from train_three_horizon import NO_PEOPLE, calibration_network, load_frames, oracle_prior

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints_three_horizon"
SECTOR_DIVISOR = 8


def bottleneck(net, scan: np.ndarray) -> torch.Tensor:
    """Stage 2's encoder output for one scan, concatenated with its own maximum -- `aggregate`'s input bar the prior."""
    x = net.stem(beam_features(torch.from_numpy(scan[None]), net.angles, net.max_range_m))
    for block, down in zip(net.encoder, net.down):
        x = down(block(x))
    return torch.cat([x, x.amax(dim=-1, keepdim=True).expand_as(x)], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="step3b run directories under checkpoints_three_horizon/")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--coarse-lags", default="(0,)", help="the stage 2 these were trained on; A51 used the no-temporal one")
    args = parser.parse_args()

    split = load_frames("test")
    angles = frog_laser_angles(720)
    sectors = split.scans.shape[-1] // SECTOR_DIVISOR
    torch_angles = torch.from_numpy(np.asarray(angles, np.float32))

    for name in args.runs:
        state = torch.load(CHECKPOINTS / name / "step3b_joint.best.pth", map_location="cpu")
        net = calibration_network(dict(model=state["net"], args=dict(coarse_lags=args.coarse_lags)), angles, prior_channels=2).eval()
        body_w, prior_w = net.aggregate.weight[:, :-2].detach(), net.aggregate.weight[:, -2:].detach()

        body_terms, prior_terms, occupied = [], [], []
        with torch.no_grad():
            for i in range(min(args.frames, len(split.annotated))):
                f = int(split.annotated[i])
                prior = oracle_prior([split.people_xy[i] if len(split.people_xy[i]) else NO_PEOPLE], torch_angles, sectors, torch.device("cpu"))
                body_terms.append(conv1d(bottleneck(net, split.scans[f]), body_w).abs().mean().item())
                prior_terms.append(conv1d(prior, prior_w).abs().mean().item())
                occupied.append(float((prior[0, 0] > 0).float().mean()))

        body, carried = float(np.mean(body_terms)), float(np.mean(prior_terms))
        print(f"{name}")
        print(f"  |W_body . bottom|                : {body:.5f}")
        print(f"  |W_prior . prior|                : {carried:.5f}")
        print(f"  the prior's share of the sum     : {100 * carried / (body + carried):.3f}%")
        print(f"  sectors it is non-zero in        : {100 * float(np.mean(occupied)):.2f}%")
        print(f"  weight rms, prior against body   : {float(prior_w.pow(2).mean().sqrt() / body_w.pow(2).mean().sqrt()):.4f}"
              f"  (per weight, over {body_w.shape[1]} channels against {prior_w.shape[1]})")
        print()


if __name__ == "__main__":
    main()
