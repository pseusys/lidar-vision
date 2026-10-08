#!/usr/bin/env python3
"""What TAKHeLiPeD's stage-2 FROG score is made of (`TODO.md` A59): one network's per-beam outputs through four decoders, and
where its recall lies against LFE's at a matched false-positive rate.

Stage 2 with no temporal input at all scores ~77% AP on FROG `official` test, against DR-SPAAM's published 73.7% (`T=1`) and
LFE-PPN's 69.2%. Its decoder is shared with neither: DROW v2's vote grid makes a detection the mean of every beam that voted near
it, the official DR-SPAAM code keeps each detection's single best beam (`nms_predicted_center`), and LFE-Peaks reports the endpoint
of the beam where its probability peaks. Decoding the *same* probabilities and votes each way separates the decoder from
everything upstream of it (`memory/static-detector-diagnosis.md`).

    vote-grid      this project's decoder until 2026-10-06, `train_three_horizon.decode_votes`, beams at the range limit silenced
    vote-weighted  the same grid with each detection's voters weighted by probability, in position and in score
    nms            the official DR-SPAAM decoder: greedy over beams by probability, each keeping its own vote, 0.5 m apart
    nms-mean       the same, each detection at the probability-weighted mean of the votes it suppresses: stage 2's decoder since 2026-10-06
    nms-0.3        `nms` at LFE's 0.30 m, which separates the suppression radius from single-beam scoring
    peaks          LFE-Peaks' decoder: `find_peaks` on the per-beam probability, the peak beam's endpoint, merged at 0.30 m
    peaks-vote     the same peaks, each reported at its beam's vote rather than its endpoint: the centre regression alone

On test, the breakdown then scores the vote grid and both NMS decoders, LFE-Peaks and LFE-PPN at one false-positive rate per frame -- each LFE model's own
at its published 0.3 threshold -- and splits the recall by range and by the distance to the nearest other person, so that a gain
can be placed: far people (scale), close pairs (decoding), or everywhere (representation). People are counted once per frame
there, since FROG's test file repeats 2,254 circles (`memory/interpreting-evaluation.md`); the AP columns keep the official labels.
`--split val` scores the decoders' AP alone, which is where a decoder is chosen before test is read.

Usage (from utils/)
-----
    python stage2_probe.py --stage2 ../checkpoints_three_horizon/step2_static/step2_calibration.best.pth --stride 5
    python stage2_probe.py --stage2 <checkpoints> --split val --stride 25 --out ../results/stage2_probe_val.json
    python stage2_probe.py --stage2 <checkpoint> --stride 500 --out ../results/stage2_probe_smoke.json      # smoke run
"""
import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.datasets.frog_dataset import frog_laser_angles  # noqa: E402
from follow_the_drow.detectors import LFEPeaksDetector, LFEPPNDetector  # noqa: E402
from follow_the_drow.detectors.lfe_detector import _merge_nearby  # noqa: E402
from train_three_horizon import (  # noqa: E402
    MATCH_RADIUS_M, NMS_DIST_M, OPERATING_THRESHOLD, FrameSplit, beam_votes, calibration_network, candidate_targets, clip_batch, clip_length, decode_votes,
    detect_device, load_frames, nms_decode, score_detections, voting_beams,
)

LFE_MERGE_M = 0.30                  # `LFEPeaksDetector.merge_radius`
PEAK_RULE = dict(height=0.01, prominence=0.1, width=1, rel_height=0.5)    # `LFEPeaksDetector`'s `find_peaks` settings
LFE_PPN_FLOOR = 0.01                # LFE-PPN's score floor wherever a whole precision-recall curve is needed
EXTRA_RADIUS_M = 0.3
EVAL_BATCH = 64
RANGE_EDGES_M = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
CROWD_EDGES_M = (0.0, 0.6, 1.0, 2.0, math.inf)
BISECTION_STEPS = 12                # a threshold to 2.4e-4, far finer than the scores' own steps
BREAKDOWN_DECODERS = ("vote-grid", "nms", "nms-mean")   # the breakdown is the slow part; these are the decoders a choice is between

Detections = np.ndarray            # `[K, 3]`: score, x, y in the detection frame
Frame = Tuple[np.ndarray, np.ndarray, np.ndarray]


def peak_decode(scan: np.ndarray, angles: np.ndarray, prob: np.ndarray, votes: np.ndarray, at_vote: bool) -> Detections:
    """LFE-Peaks' decoder on per-beam probabilities: `find_peaks`, each peak at its beam's endpoint -- or, `at_vote`, at its beam's
    vote -- scored by the peak's probability, then merged greedily at LFE's 0.30 m."""
    peaks, _ = find_peaks(prob, **PEAK_RULE)
    offsets = votes[peaks] if at_vote else np.zeros((len(peaks), 2))
    xy = beam_votes(scan[peaks], angles[peaks], offsets)
    merged = _merge_nearby([(float(prob[p]), float(x), float(y)) for p, (x, y) in zip(peaks, xy, strict=True)], LFE_MERGE_M)
    return np.array(merged, dtype=np.float64).reshape(-1, 3)


def vote_grid_decode(scan: np.ndarray, angles: np.ndarray, prob: np.ndarray, votes: np.ndarray, weighted_avg: bool = False) -> Detections:
    return np.array(decode_votes(scan, angles, prob, votes, weighted_avg), dtype=np.float64).reshape(-1, 3)


DECODERS: Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Detections]] = {
    "vote-grid": vote_grid_decode,
    "vote-weighted": lambda scan, angles, prob, votes: vote_grid_decode(scan, angles, prob, votes, weighted_avg=True),
    "nms": lambda scan, angles, prob, votes: nms_decode(scan, angles, prob, votes, NMS_DIST_M),
    "nms-mean": lambda scan, angles, prob, votes: nms_decode(scan, angles, prob, votes, NMS_DIST_M, average=True),
    "nms-0.3": lambda scan, angles, prob, votes: nms_decode(scan, angles, prob, votes, LFE_MERGE_M),
    "peaks": lambda scan, angles, prob, votes: peak_decode(scan, angles, prob, votes, at_vote=False),
    "peaks-vote": lambda scan, angles, prob, votes: peak_decode(scan, angles, prob, votes, at_vote=True),
}


def unique_people(people: np.ndarray) -> np.ndarray:
    """A frame's annotated people `[G, 2]` with exact repeats counted once."""
    return np.unique(people.reshape(-1, 2), axis=0)


def nearest_other(people: np.ndarray) -> np.ndarray:
    """Per person `[G, 2]`, the distance to the nearest other one; infinite for a person alone."""
    dist = np.linalg.norm(people[:, None, :] - people[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1) if len(people) else np.zeros(0)


@torch.no_grad()
def stage2_outputs(net, split: FrameSplit, positions: np.ndarray, device) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per scored annotated frame: the ranges the network read, its per-beam probabilities with limit beams silenced, its votes."""
    net.eval()
    out = []
    for chunk in np.array_split(positions, math.ceil(len(positions) / EVAL_BATCH)):
        ranges, poses = clip_batch(split, split.annotated[chunk], clip_length(net.coarse.lags), device)
        logit, votes, _ = net(ranges, poses)
        prob, votes = torch.sigmoid(logit).cpu().numpy(), votes.cpu().numpy()
        for j, a in enumerate(chunk):
            scan, p = voting_beams(split.scans[split.annotated[a]], prob[j], net.max_range_m)
            out.append((scan, p, votes[j]))
    return out


def lfe_frames(detector, split: FrameSplit, positions: np.ndarray, angles: np.ndarray) -> List[Frame]:
    out = []
    for a in positions:
        dets = np.array(detector.detect(split.scans[split.annotated[a]], angles), dtype=np.float64).reshape(-1, 3)
        out.append((dets[:, 0], dets[:, 1:], split.people_xy[a]))
    return out


def ap_row(frames: List[Frame]) -> Dict[str, float]:
    main, tight = score_detections(frames), score_detections(frames, radius=EXTRA_RADIUS_M)
    return dict(ap=main["ap"], ap_03=tight["ap"], recall_03=main["recall"], fp_per_frame_03=main["fp_per_frame"])


def coverage(frames: List[Frame], threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Every distinct annotated person's range, nearest-other distance and whether a detection at `threshold` covers it (one to
    one, gated Hungarian at the match radius), with the false positives per frame that matching leaves."""
    ranges, crowd, covered, fp = [], [], [], 0
    for scores, xy, people in frames:
        people = unique_people(people)
        kept = scores >= threshold
        _, hit = candidate_targets(xy[kept], np.ones(int(kept.sum()), dtype=bool), people, MATCH_RADIUS_M)
        fp += int(kept.sum()) - int(hit.sum())
        ranges.append(np.linalg.norm(people, axis=1)), crowd.append(nearest_other(people)), covered.append(hit)
    return np.concatenate(ranges), np.concatenate(crowd), np.concatenate(covered), fp / len(frames)


def threshold_at(frames: List[Frame], fp_per_frame: float) -> float:
    """The lowest score threshold whose false positives per frame (`coverage`'s counting) stay within `fp_per_frame`: bisection,
    since adding a detection can only add an unmatched one."""
    lo, hi = 0.0, 1.0
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if coverage(frames, mid)[3] <= fp_per_frame else (mid, hi)
    return hi


def binned(values: np.ndarray, covered: np.ndarray, edges: Sequence[float]) -> List[Dict[str, float]]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = (values >= lo) & (values < hi)
        rows.append(dict(lo=lo, hi=hi, share=float(inside.mean()), recall=float(covered[inside].mean()) if inside.any() else float("nan")))
    return rows


def breakdown(frames: List[Frame], threshold: float) -> Dict[str, object]:
    ranges, crowd, covered, fp = coverage(frames, threshold)
    return dict(threshold=threshold, fp_per_frame=fp, recall=float(covered.mean()), people=int(len(covered)),
                by_range=binned(ranges, covered, RANGE_EDGES_M), by_crowding=binned(crowd, covered, CROWD_EDGES_M))


def print_breakdowns(title: str, named: Dict[str, Dict[str, object]]) -> None:
    print(f"\n{title}", flush=True)
    names = list(named)
    print("  " + " " * 14 + "".join(f"{n:>16}" for n in names))
    print("  " + f"{'threshold':<14}" + "".join(f"{named[n]['threshold']:>16.3f}" for n in names))
    print("  " + f"{'FP / frame':<14}" + "".join(f"{named[n]['fp_per_frame']:>16.3f}" for n in names))
    print("  " + f"{'recall':<14}" + "".join(f"{named[n]['recall']:>16.1%}" for n in names))
    for key, unit in (("by_range", "m"), ("by_crowding", "m to next")):
        print(f"  -- recall {key.replace('_', ' ')} ({unit}) --")
        for i, row in enumerate(named[names[0]][key]):
            label = f"{row['lo']:g}-{row['hi']:g} ({row['share']:.0%})"
            print("  " + f"{label:<14}" + "".join(f"{named[n][key][i]['recall']:>16.1%}" for n in names))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2", type=Path, nargs="+", required=True, help="stage 1-2 checkpoints to decode")
    ap.add_argument("--split", choices=("val", "test"), default="test", help="val scores the decoders' AP alone, for choosing one")
    ap.add_argument("--stride", type=int, default=5, help="score every Nth annotated frame")
    ap.add_argument("--out", type=Path, default=_HERE.parent / "results" / "stage2_probe.json")
    args = ap.parse_args()

    device, _ = detect_device()
    angles = frog_laser_angles(720)
    split = load_frames(args.split)
    positions = np.arange(0, len(split.annotated), args.stride)
    print(f"{args.split}: {len(positions)} of {len(split.annotated)} annotated frames (every {args.stride})", flush=True)

    lfe = {} if args.split == "val" else {"LFE-Peaks": lfe_frames(LFEPeaksDetector(), split, positions, angles),
                                          "LFE-PPN": lfe_frames(LFEPPNDetector(score_thresh=LFE_PPN_FLOOR), split, positions, angles)}
    results = dict(split=args.split, stride=args.stride, frames=len(positions), lfe={n: ap_row(f) for n, f in lfe.items()}, stage2={})
    for name, row in results["lfe"].items():
        print(f"  {name:<10} AP {row['ap']:.2%}  AP@0.3 {row['ap_03']:.2%}", flush=True)

    for path in args.stage2:
        net = calibration_network(torch.load(path, map_location="cpu", weights_only=False), angles).to(device)
        outputs = stage2_outputs(net, split, positions, device)
        people = [split.people_xy[a] for a in positions]
        decoded = {name: [(d[:, 0], d[:, 1:], g) for d, g in zip((decode(s, angles, p, v) for s, p, v in outputs), people, strict=True)] for name, decode in DECODERS.items()}
        run = dict(decoders={name: ap_row(frames) for name, frames in decoded.items()})
        print(f"\n=== {path.parent.name}/{path.name} ({sum(p.numel() for p in net.parameters()):,} parameters, coarse lags {net.coarse.lags}) ===", flush=True)
        for name, row in run["decoders"].items():
            print(f"  {name:<11} AP {row['ap']:.2%}  AP@0.3 {row['ap_03']:.2%}  at {OPERATING_THRESHOLD}: recall {row['recall_03']:.1%}  FP/frame {row['fp_per_frame_03']:.3f}", flush=True)
        run["matched"] = {}
        for reference, frames in lfe.items():
            own = breakdown(frames, OPERATING_THRESHOLD)
            run["matched"][reference] = {reference: own, **{name: breakdown(decoded[name], threshold_at(decoded[name], own["fp_per_frame"])) for name in BREAKDOWN_DECODERS}}
            print_breakdowns(f"at {reference}'s false positives per frame at its {OPERATING_THRESHOLD} threshold:", run["matched"][reference])
        results["stage2"][f"{path.parent.name}/{path.name}"] = run

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
        code = 0
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)
