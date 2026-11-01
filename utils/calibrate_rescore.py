#!/usr/bin/env python3
"""Fit a post-hoc temperature and bias on the memory's output logit, so a fixed threshold means something (`TODO.md` item 48).

A51 phase 1b measured the rescore head shifting the *whole* score distribution down -- a median -3.292 logits on people it
keeps against -1.497 on people it loses -- which leaves rank-based AP better while silently decalibrating any fixed
threshold. The repo's own conclusion was "the fix is calibration of the rescore head"; this is that fix, done post-hoc.

    (temperature, bias):   calibrated_logit = (logit + bias) / temperature

**Monotone for any positive temperature**, so AP, recall at matched false positives and every other rank-based number
cannot move; only the fixed-threshold columns do. That is what makes it safe to apply after the freeze rather than
before it. The transform is fitted on **val** and lands in the stage-3 checkpoint as a `calibration` key, which
`object_memory` reads back with an identity fallback -- see `tests/test_score_calibration.py` for both invariants.

    python calibrate_rescore.py --memory <step3a ckpt> --stage2 <step2 ckpt> --out <json> [--write]
    python calibrate_rescore.py --memory <step3a ckpt> --apply <json>

`--write` puts the fitted pair into the memory checkpoint; without it nothing is modified and the JSON is the only
output. `--apply` stores a pair from a JSON written earlier **without re-measuring**, which is the cheap way to do the
irreversible half: a fresh `--write` would redo both `score_detections` passes, about 55 minutes a seed. Either way the
report's gates are re-checked by `apply_calibration` before anything is saved.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from train_takheliped import (  # noqa: E402
    MATCH_RADIUS_M, OPERATING_THRESHOLD, PROBABILITY_EPS, detect_device, evaluate, evaluation_segments,
    load_split, object_memory, score_detections,
)

#: LBFGS is used because the objective is convex in the pair it actually optimises (see `fit_calibration`), so this
#: many iterations is about three times what any of the three freeze seeds needed to stop moving.
FIT_ITERATIONS = 200

#: How far AP may differ between the uncalibrated and calibrated columns before `apply_calibration` refuses. It is
#: **not** zero, and the reason is measured rather than assumed. The transform is monotone, so no ranking can change,
#: and rounding to nearest is monotone too, so float32 cannot *invert* two scores -- verified 2026-10-31 across all
#: three seeds and 6.4 M detections: **zero inversions**. What float32 can do is **collapse** distinct scores onto one
#: value: 24,812-26,010 adjacent pairs per seed, because every fitted temperature sharpened (0.71-0.79) and spread the
#: logits onto a coarser part of the sigmoid's grid. `_prec_rec_2d` accepts tied detections in mergesort order, so a
#: collapsed tie can reorder within its group and move AP by a hair: seeds 0 and 2 moved +-5.96e-8, exactly one
#: float32 ulp near 0.82, and seed 1 moved 0. This tolerance is two orders of magnitude below the 0.01 pp this repo
#: reports to, so it admits the rounding and nothing that could be read in a table.
AP_TOLERANCE = 1e-6


def to_logit(probability: np.ndarray) -> np.ndarray:
    """A probability as a logit, clamped exactly as `ObjectMemory._logit` clamps it, so the round trip is the model's own."""
    p = np.clip(probability, PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
    return np.log(p) - np.log1p(-p)


def to_probability(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logit))


def negative_log_likelihood(logit: np.ndarray, label: np.ndarray, temperature: float, bias: float) -> float:
    """Mean binary NLL of `(logit + bias) / temperature` against `label` -- a proper scoring rule, which is the point:
    it is minimised by the transform that makes the score mean what it says, not by the one that maximises AP."""
    z = torch.as_tensor((np.asarray(logit, dtype=np.float64) + bias) / temperature)
    y = torch.as_tensor(np.asarray(label, dtype=np.float64))
    return float(torch.nn.functional.binary_cross_entropy_with_logits(z, y))


def fit_calibration(logit: np.ndarray, label: np.ndarray, iterations: int = FIT_ITERATIONS) -> Tuple[float, float]:
    """The `(temperature, bias)` minimising `negative_log_likelihood`, with **temperature guaranteed positive**.

    Optimised as a two-parameter logistic regression on the logits, `sigmoid(exp(u) * logit + c)`, which is convex in
    `(exp(u), c)` and maps back as `temperature = exp(-u)` and `bias = c * temperature`. Two reasons for that
    parameterisation rather than fitting `(temperature, bias)` directly: it cannot return a **negative** slope, which
    would invert the ranking and silently destroy the one property this whole approach rests on, and it cannot return
    a zero temperature to divide by.
    """
    z = torch.as_tensor(np.asarray(logit, dtype=np.float64))
    y = torch.as_tensor(np.asarray(label, dtype=np.float64))
    if not (0 < float(y.sum()) < len(y)):
        raise ValueError(f"calibration needs both classes present, got {int(y.sum())} positives of {len(y)}; "
                         "a one-class fit drives the bias to infinity")
    u, c = torch.zeros(1, dtype=torch.float64, requires_grad=True), torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.LBFGS([u, c], max_iter=iterations, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(torch.exp(u) * z + c, y)
        loss.backward()
        return loss

    optimiser.step(closure)
    temperature = float(torch.exp(-u).item())
    return temperature, float(c.item()) * temperature


def greedy_labels(score: np.ndarray, xy: np.ndarray, people: np.ndarray,
                  radius: float = MATCH_RADIUS_M) -> np.ndarray:
    """Which detections the metric would count as true positives, assigned in **descending score order**.

    `_prec_rec_2d` accepts detections from the highest score down and re-runs a gated one-to-one assignment over
    everything accepted so far, so at every threshold a person belongs to the best-scoring detection that can reach
    it. `candidate_targets` would have been the obvious thing to reuse and is the wrong tool here: it minimises
    **distance** over all candidates at once and is score-blind, so it hands a person to a candidate 0.1 m away
    scoring 0.05 instead of one 0.3 m away scoring 0.9 -- labelling the detection the metric counts as the true
    positive a negative. That makes the label noise anti-correlated with the score, which is the one kind a
    temperature fit cannot survive: it would see confident detections "being wrong" and flatten the score to suit.
    `tests/test_score_calibration.py::TestLabelsFollowTheMetricsRule` pins both halves of that.
    """
    score, xy, people = np.asarray(score), np.asarray(xy), np.asarray(people)
    label = np.zeros(len(score), dtype=bool)
    if not len(people) or not len(score):
        return label
    taken = np.zeros(len(people), dtype=bool)
    for i in np.argsort(-score, kind="mergesort"):      # mergesort for determinism, as the metric itself does
        distance = np.linalg.norm(people - xy[i], axis=-1)
        distance[taken] = np.inf
        nearest = int(np.argmin(distance))
        if distance[nearest] <= radius:
            label[i] = taken[nearest] = True
    return label


def labels_for_frames(frames: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                      radius: float = MATCH_RADIUS_M) -> Tuple[np.ndarray, np.ndarray]:
    """Every detection on the replayed frames as `(logit, is_a_person)`, concatenated in frame order."""
    logit, label = [], []
    for score, xy, people in frames:
        if not len(np.asarray(score)):
            continue
        logit.append(to_logit(np.asarray(score))), label.append(greedy_labels(score, xy, people, radius))
    if not logit:
        raise ValueError("no detections on the replayed frames, so there is nothing to calibrate")
    return np.concatenate(logit), np.concatenate(label)


def check_labels_against_the_metric(frames: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], logit: np.ndarray,
                                    label: np.ndarray, metric_recall: float, threshold: float = OPERATING_THRESHOLD,
                                    tolerance: float = 0.02) -> Dict[str, float]:
    """How far the labels' recall sits from the recall `score_detections` reports, **both at the operating threshold**.

    `_prec_rec_2d` is deliberately left untouched -- every published number in this repo rests on it -- so the labels
    built here have to be *compared* against it rather than assumed equal, and `--write` refuses on a mismatch.

    The comparison is made at `threshold` because that is the only place the two are comparable. The first version of
    this check counted label positives over **every** candidate and compared them against the metric's recall at 0.3;
    on seed 0 that reported a 10.18 pp disagreement which was not a disagreement at all, just the recall thresholding
    gives up (94.92% against 84.75%). A gate whose refusal has to be explained away is worse than no gate.

    `metric_recall` is passed in rather than recomputed: `score_detections` over a whole val split is the expensive
    call in this script, and the caller already has it.
    """
    above = np.asarray(logit) >= to_logit(np.asarray(threshold, dtype=np.float64))
    people = sum(len(np.asarray(f[2])) for f in frames)
    matched = int(np.asarray(label)[above].sum())
    ours = matched / people if people else float("nan")
    return dict(people=people, threshold=threshold, matched=matched, label_recall=ours,
                metric_recall=metric_recall, difference=abs(ours - metric_recall),
                tolerance=tolerance, agrees=bool(abs(ours - metric_recall) <= tolerance))


def save_frames(path: Path, frames: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    """The replayed frames, so a change to the label rule or to the fit does not need the GPU or the checkpoints again.

    It does not make a re-fit quick: the replay is about 8 minutes of the hour and the two `score_detections` passes
    are the rest. What it buys is that a re-fit becomes a CPU-only job against a 25 MB file, reproducible by anyone
    without the 1 GB candidate cache.

    Flat arrays plus per-frame counts rather than a ragged object array: `np.load` of one needs `allow_pickle`, and a
    data file that executes code when it is read is not worth the few lines it saves.
    """
    np.savez_compressed(path, score=np.concatenate([f[0] for f in frames]).astype(np.float32),
                        xy=np.concatenate([f[1] for f in frames]).astype(np.float32),
                        detections=np.array([len(f[0]) for f in frames], np.int64),
                        people_xy=np.concatenate([f[2] for f in frames]).astype(np.float32),
                        people=np.array([len(f[2]) for f in frames], np.int64))


def load_frames(path: Path) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with np.load(path) as stored:
        score = np.split(stored["score"], np.cumsum(stored["detections"])[:-1])
        xy = np.split(stored["xy"], np.cumsum(stored["detections"])[:-1])
        people = np.split(stored["people_xy"], np.cumsum(stored["people"])[:-1])
    return list(zip(score, xy, people))


def columns(frames: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], temperature: float, bias: float) -> Dict[str, float]:
    """The fixed-threshold columns after calibration, and AP alongside them so it is visible that AP did not move."""
    moved = [(to_probability((to_logit(np.asarray(score)) + bias) / temperature), xy, people) for score, xy, people in frames]
    return score_detections(moved)


def _report(before: Dict[str, float], after: Dict[str, float], temperature: float, bias: float, agreement: Dict[str, float]) -> str:
    return ("\n".join([
        f"fitted temperature {temperature:.4f}  bias {bias:+.4f}",
        f"label check at {agreement['threshold']}: {agreement['label_recall']:.2%} against the metric's "
        f"{agreement['metric_recall']:.2%} (difference {agreement['difference']:.4f}, "
        f"tolerance {agreement['tolerance']})"
        f" -> {'agrees' if agreement['agrees'] else 'DISAGREES, do not write this'}",
        f"{'column':<22}{'uncalibrated':>14}{'calibrated':>14}",
        f"{'AP (must not move)':<22}{before['ap']:>13.4%}{after['ap']:>14.4%}",
        f"{'precision at 0.3':<22}{before['precision']:>13.2%}{after['precision']:>14.2%}",
        f"{'recall at 0.3':<22}{before['recall']:>13.2%}{after['recall']:>14.2%}",
        f"{'FP/frame at 0.3':<22}{before['fp_per_frame']:>13.3f}{after['fp_per_frame']:>14.3f}",
        f"{'FN/frame at 0.3':<22}{before['fn_per_frame']:>13.3f}{after['fn_per_frame']:>14.3f}",
    ]))


def apply_calibration(report: Path, memory: Path, tolerance: float = AP_TOLERANCE) -> Tuple[float, float]:
    """Store a pair this script already fitted into its stage-3 checkpoint, re-checking the report's own gates.

    This exists so that writing does not cost a re-measurement. `--write` on a fresh run would redo both
    `score_detections` passes -- about 55 minutes a seed, ~2.75 h for three -- to store two floats that are already
    sitting in the JSON. The frames cache does not help there: it skips the ~8 min replay, not the scoring.

    **The checks are re-run here rather than trusted.** A report is a file on disk that somebody may have read, moved
    or regenerated, and this is the irreversible half of the item -- it edits the three checkpoints the freeze is
    quoted from. The path check is the one that catches the realistic mistake: three seeds, three JSONs, and one
    mismatched argument writes seed 1's pair into seed 2's weights, which would be invisible afterwards.
    """
    fitted = json.loads(report.read_text(encoding="utf-8"))
    temperature, bias = float(fitted["temperature"]), float(fitted["bias"])
    if Path(fitted["memory"]).resolve() != memory.resolve():
        raise ValueError(f"{report} was fitted on a different checkpoint ({fitted['memory']}), refusing to write it "
                         f"into {memory}")
    if temperature <= 0.0:
        raise ValueError(f"refusing a non-positive temperature ({temperature}): it either divides by zero or inverts "
                         "the ranking, which is the property this whole approach rests on")
    if not fitted["label_check"]["agrees"]:
        raise ValueError(f"refusing to write: the labels disagree with the reported metric by "
                         f"{fitted['label_check']['difference']:.4f}, over its {fitted['label_check']['tolerance']} "
                         "tolerance, so the pair was fitted against something the metric does not share")
    moved = abs(fitted["calibrated"]["ap"] - fitted["uncalibrated"]["ap"])
    if moved > tolerance:
        raise ValueError(f"refusing to write: AP moved by {moved:.3e}, over the {tolerance:.0e} float32 tolerance -- "
                         "a monotone transform cannot do that, so the report is not measuring what it claims")
    checkpoint = torch.load(memory, map_location="cpu", weights_only=False)
    checkpoint["calibration"] = dict(temperature=temperature, bias=bias)
    torch.save(checkpoint, memory)
    return temperature, bias


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory", type=Path, required=True)
    ap.add_argument("--stage2", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--apply", type=Path, default=None,
                    help="store the pair from a JSON this script already wrote into --memory and do nothing else; "
                         "re-checks the report's gates first and never re-measures")
    ap.add_argument("--split", default="val", help="the split to fit on; it must not be test (item 48)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--limit-recordings", type=int, default=0)
    ap.add_argument("--frames", type=Path, default=None,
                    help="cache the replayed frames here and reuse them when the file already exists, so a re-fit "
                         "needs neither the GPU nor the checkpoints -- it skips the replay, not the scoring")
    ap.add_argument("--write", action="store_true", help="store the fitted pair in the memory checkpoint's `calibration` key")
    args = ap.parse_args()
    if args.apply is not None:
        temperature, bias = apply_calibration(args.apply, args.memory)
        print(f"wrote calibration (temperature {temperature:.4f}, bias {bias:+.4f}) into {args.memory}", flush=True)
        return
    if args.stage2 is None or args.out is None:
        raise SystemExit("--stage2 and --out are required unless --apply is given")
    if args.split == "test":
        raise SystemExit("refusing to fit on test: the transform would then be chosen on the split it is reported on")

    if args.frames is not None and args.frames.exists():
        frames = load_frames(args.frames)
        print(f"{args.memory.parent.name}: reusing {len(frames):,} cached frames from {args.frames}", flush=True)
    else:
        device, _ = detect_device()
        split = load_split(args.split, args.limit_recordings, args.stage2, device)
        checkpoint = torch.load(args.memory, map_location=device, weights_only=False)
        model = object_memory(checkpoint, split.data.features.shape[-1]).to(device)
        print(f"{args.memory.parent.name}: calibrating on {args.split}, "
              f"existing calibration {model.calibration}", flush=True)
        frames_out: Dict[str, list] = {}
        evaluate(model, split, device, args.stride, args.streams, evaluation_segments(split, 0, 0, 0),
                 frames_out=frames_out, compute_metrics=False)
        frames = frames_out["memory"]
        # A coasting report is not a rescored candidate: the transform cannot move its logit, so fitting on one and
        # then checking against a metric that counts it would compare two different sets. The frozen architecture has
        # no coasting head (item 24 removed it), so this is a guard against a future arm, not a code path.
        carried = [i for i, (score, _, _) in enumerate(frames) if frames_out["rescored"][i] != len(score)]
        if carried:
            raise SystemExit(f"{len(carried)} frames carry coasting reports, which this transform does not apply to; "
                             "calibrating against a metric that counts them would fit the wrong set")
        if args.frames is not None:
            save_frames(args.frames, frames)
            print(f"cached the replayed frames in {args.frames}", flush=True)

    logit, label = labels_for_frames(frames)
    print(f"{len(logit):,} detections on {len(frames):,} frames, {int(label.sum()):,} of them people", flush=True)

    temperature, bias = fit_calibration(logit, label)
    before = score_detections(list(frames))
    agreement = check_labels_against_the_metric(frames, logit, label, before["recall"])
    after = columns(frames, temperature, bias)
    print(_report(before, after, temperature, bias, agreement), flush=True)

    args.out.write_text(json.dumps(dict(memory=str(args.memory), stage2=str(args.stage2), split=args.split,
                                        temperature=temperature, bias=bias, detections=len(logit),
                                        positives=int(label.sum()), label_check=agreement,
                                        uncalibrated=before, calibrated=after,
                                        nll_before=negative_log_likelihood(logit, label, 1.0, 0.0),
                                        nll_after=negative_log_likelihood(logit, label, temperature, bias)), indent=1),
                        encoding="utf-8")
    print(f"wrote {args.out}", flush=True)

    if args.write:
        # Through `apply_calibration` rather than writing here, so the gates that guard a write live in exactly one
        # place and `--write` cannot drift from `--apply`.
        apply_calibration(args.out, args.memory)
        print(f"wrote calibration into {args.memory}", flush=True)


if __name__ == "__main__":
    try:
        main()
        code = 0
    except SystemExit as leaving:        # argparse's --help and its usage errors: pass their status through
        code = leaving.code if isinstance(leaving.code, int) else (0 if leaving.code is None else 1)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)                          # a finished ROCm process can hang past a normal exit (memory/gotchas.md)
