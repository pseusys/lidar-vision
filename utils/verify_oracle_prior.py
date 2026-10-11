"""Check E12's oracle prior against real recorded frames, not synthetic points (`AGENTS.md` rule 4).

A null result from E12b looks identical whether the feedback path is genuinely useless or the oracle was silently empty -- a
wrong frame or a disagreeing axis convention would both render a map of zeros. This answers which, on FROG's own annotations:

    python verify_oracle_prior.py --joint ../checkpoints_three_horizon/a51_e12b_oracle/step3b_joint.best.pth

It reports three things. **Coverage:** how many annotated frames draw a non-zero prior, and how many sectors they mark. **Bearing
agreement:** the sector each person lands in, recomputed from `people_polar`'s bearing independently of `render_prior`'s xy path,
against the sectors the prior actually marked -- a mismatch is the axis-convention bug. **Trained weight:** the size of
`aggregate.weight[:, 2*c2:]` after training, against the rest of the layer, which says whether the network moved the channels at
all. Needs the machine to itself: a second torch process cannot start beside a training job (`memory/gotchas.md`).
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))

from follow_the_drow.detectors.takheliped import render_prior
from train_three_horizon import NO_PEOPLE, load_frames, oracle_prior

SECTOR_DIVISOR = 8      # stage 2's bottleneck is three halvings below the beam grid


def bearing_sectors(polar: np.ndarray, angles: np.ndarray, sectors: int) -> np.ndarray:
    """Each person's sector, from its recorded bearing alone -- `render_prior`'s xy path is not involved, so the two disagree
    exactly when the xy convention is wrong."""
    width = (angles[1] - angles[0]) * len(angles) / sectors
    return np.floor((polar[:, 1] - angles[0]) / width).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joint", type=Path, help="a step3b checkpoint, to weigh its trained feedback channels")
    parser.add_argument("--split", default="test")
    parser.add_argument("--frames", type=int, default=400, help="annotated frames to check")
    args = parser.parse_args()

    split = load_frames(args.split)
    angles = np.linspace(-np.pi / 2, np.pi / 2, split.scans.shape[-1], endpoint=False, dtype=np.float32)
    angles = angles if split.scans.shape[-1] != 720 else _frog_angles()
    sectors = split.scans.shape[-1] // SECTOR_DIVISOR
    torch_angles = torch.from_numpy(np.asarray(angles, np.float32))

    chosen = np.arange(len(split.annotated))[: args.frames]
    empty, drawn, marked, people_total, agreed, outside = 0, 0, 0, 0, 0, 0
    for i in chosen:
        people = split.people_xy[i]
        if not len(people):
            empty += 1
            continue
        prior = oracle_prior([people], torch_angles, sectors, torch.device("cpu"))[0]
        hit = prior[0] > 0
        drawn += int(hit.any())
        marked += int(hit.sum())
        want = bearing_sectors(split.people_polar[i], angles, sectors)
        inside = (want >= 0) & (want < sectors)
        outside += int((~inside).sum())
        people_total += int(inside.sum())
        agreed += int(hit.numpy()[want[inside]].sum())

    checked = len(chosen) - empty
    print(f"{args.split}: {len(chosen)} annotated frames checked, {empty} of them holding no people")
    print(f"  frames drawing a non-zero prior : {drawn} of {checked} ({100 * drawn / max(checked, 1):.1f}%)")
    print(f"  sectors marked per such frame   : {marked / max(drawn, 1):.2f}")
    print(f"  people whose recorded bearing lands on a marked sector: {agreed} of {people_total} "
          f"({100 * agreed / max(people_total, 1):.1f}%)  [{outside} outside the field of view]")
    verdict = "the oracle is real" if drawn and agreed >= 0.99 * people_total else "THE ORACLE IS WRONG -- E12's null means nothing"
    print(f"  -> {verdict}")

    if args.joint:
        state = torch.load(args.joint, map_location="cpu")
        weight = (state["net"] if "net" in state else state)["aggregate.weight"]
        body, prior_channels = weight[:, :-2], weight[:, -2:]
        print(f"\n{args.joint.name}: aggregate.weight {tuple(weight.shape)}")
        print(f"  body rms           {body.pow(2).mean().sqrt():.6e}")
        print(f"  prior channels rms {prior_channels.pow(2).mean().sqrt():.6e}  "
              f"absmax {prior_channels.abs().max():.6e}  nonzero {int((prior_channels != 0).sum())}/{prior_channels.numel()}")
        print(f"  ratio prior/body   {float(prior_channels.pow(2).mean().sqrt() / body.pow(2).mean().sqrt()):.6e}")
        print("  -> the channels never moved" if prior_channels.abs().max() == 0 else
              "  -> the channels moved; whatever they learned is worth what E12 measured")
        # Under an oracle prior the two channels are identical -- both 1.0 wherever a person is -- so only their sum can act,
        # and the difference is a free direction the optimizer may park large weights in. Large weights with no measured effect
        # are explained if the sum is small; if it is not, the weights genuinely act and something else cancels them.
        total, difference = prior_channels[:, 0] + prior_channels[:, 1], prior_channels[:, 0] - prior_channels[:, 1]
        print(f"  channel sum rms    {total.pow(2).mean().sqrt():.6e}   (the only direction an identical-channel prior can act in)")
        print(f"  channel diff rms   {difference.pow(2).mean().sqrt():.6e}   (unidentifiable under an oracle prior)")
        print(f"  -> {'the acting direction is near zero, which is why silencing changes nothing' if total.pow(2).mean().sqrt() < 0.1 * difference.pow(2).mean().sqrt() else 'the acting direction is real, so the null is not a cancellation artefact'}")


def _frog_angles() -> np.ndarray:
    from follow_the_drow.datasets.frog_dataset import frog_laser_angles
    return frog_laser_angles(720)


if __name__ == "__main__":
    main()
