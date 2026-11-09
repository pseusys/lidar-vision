#!/usr/bin/env python3
"""Does the path every TAKHeLiPeD DROW number came through agree with `evaluate.py`'s, on the real DROW files?

Two checks, both on the recorded data rather than on fixtures (`AGENTS.md` rule 4):

1. **The frames.** For each split, `FrameSplit` -- what `train_takheliped.py` trains and scores on -- against the
   `DROW_Dataset` it is packed from, which is what `evaluate.py` and `train.py` read: the same annotated frames in the
   same order, each on its own scan, with the same people. The one intended difference is `DatasetSpec.real_people`,
   which drops labels past the sensor's rated range, so those are counted rather than called a mismatch.
2. **The score.** DROW3's published weights run once through `evaluate.py`'s own `eval_nn_model`, and the very
   detections behind its person AP are scored again by `score_detections` against `FrameSplit`'s test labels. The two
   must agree, and the first must be DROW3's reproduction through this repository, 66.5%.

A failure here means no TAKHeLiPeD DROW number can be set beside a baseline's until it is found.

    python check_drow_scoring.py                      # both checks, every split
    python check_drow_scoring.py --splits test        # one split at a time, for memory
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from json import dumps
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

from evaluate import _load_dataset, eval_nn_model
from takheliped.detectors import DrowDetector
from takheliped.utils.drow_utils import project_cartesian_from_polar
from train_takheliped import DROW, MATCH_RADIUS_M, FrameSplit, frames_from_dataset, score_detections

_HERE = Path(__file__).parent
SPLITS = ("train", "val", "test")
#: The two scorers share `_prec_rec_2d`; what may differ is the labels, the pairing and a float32 cast, so anything
#: past a rounding error is a finding.
AP_TOLERANCE = 1e-4
#: DROW3's published weights through `evaluate.py`, the figure every DROW comparison in this repository rests on.
DROW3_REPRODUCED_AP = 0.665
REPRODUCTION_TOLERANCE = 0.0005


def annotated_pairs(ds) -> List[Tuple[int, int]]:
    """(recording, annotation) for every annotated frame, in the order both `evaluate_auc` and `FrameSplit` walk them."""
    return [(s, d) for s in range(len(ds.det_id)) for d in range(len(ds.det_id[s]))]


def frame_differences(ds, split: FrameSplit) -> Dict[str, object]:
    """Where `split` and the dataset it was packed from disagree: frame count, scan, or people."""
    pairs = annotated_pairs(ds)
    wrong_scan, wrong_people, beyond_rated = [], [], 0
    for i, (s, d) in enumerate(pairs[:len(split.annotated)]):
        if not np.array_equal(split.scans[split.annotated[i]], np.asarray(ds.scans[s][ds.idet2iscan[s][d]], np.float32)):
            wrong_scan.append(i)
        labels = np.asarray(ds.det_wp[s][d], np.float64).reshape(-1, 2)
        kept = labels[labels[:, 0] <= DROW.rated_range_m]
        beyond_rated += len(labels) - len(kept)
        expected = np.array([project_cartesian_from_polar(r, phi) for r, phi in kept], np.float64).reshape(-1, 2)
        if split.people_xy[i].shape != expected.shape or not np.allclose(split.people_xy[i], expected):
            wrong_people.append(i)
    return dict(frames=len(pairs), frames_in_split=len(split.annotated), wrong_scan=len(wrong_scan),
                wrong_people=len(wrong_people), first_wrong=(wrong_scan + wrong_people)[:10], beyond_rated_range=beyond_rated)


def detections_by_frame(detections: Dict[str, object]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """`evaluate_auc`'s flat detections as one (scores, xy) pair per scored frame."""
    order = np.argsort(detections["frames"], kind="stable")
    frames = detections["frames"][order]
    cuts = np.searchsorted(frames, np.arange(detections["n_frames"] + 1))
    scores, xy = detections["scores"][order], np.stack([detections["x"], detections["y"]], axis=1)[order]
    return [(scores[a:b], xy[a:b]) for a, b in zip(cuts[:-1], cuts[1:], strict=True)]


def main() -> int:
    ap = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    ap.add_argument("--batch", type=int, default=16, help="evaluate.py's eval batch; numerically equivalent to 1")
    ap.add_argument("--out", type=Path, default=_HERE.parent / "results" / "check_drow_scoring.json")
    args = ap.parse_args()

    results, failures = {}, []
    for name in args.splits:
        ds, cfg = _load_dataset(SimpleNamespace(dataset="drow", split=name))
        split = frames_from_dataset(ds, DROW)
        check = frame_differences(ds, split)
        print(f"{name}: {check['frames']} annotated frames, {check['frames_in_split']} in FrameSplit, "
              f"{check['wrong_scan']} on another scan, {check['wrong_people']} with other people, "
              f"{check['beyond_rated_range']} labels past the rated range dropped", flush=True)
        if check["frames"] != check["frames_in_split"] or check["wrong_scan"] or check["wrong_people"]:
            failures.append(f"{name}: FrameSplit disagrees with DROW_Dataset ({check})")
        if name == "test":
            aucs = eval_nn_model("drow", DrowDetector.DEFAULT_WEIGHTS, ds, cfg, eval_r=MATCH_RADIUS_M, batch_size=args.batch,
                                 return_detections=True)
            if aucs["detections"]["pairs"] != annotated_pairs(ds):
                failures.append("test: evaluate_auc scored its frames in another order than FrameSplit packs them")
            frames = [(s, xy, people) for (s, xy), people in zip(detections_by_frame(aucs["detections"]), split.people_xy, strict=True)]
            ours = score_detections(frames)["ap"]
            check.update(evaluate_wp_ap=float(aucs["wp"]), score_detections_ap=float(ours))
            print(f"test: DROW3 person AP {aucs['wp']:.4%} through evaluate.py, {ours:.4%} through score_detections "
                  f"on FrameSplit's labels", flush=True)
            if abs(ours - aucs["wp"]) > AP_TOLERANCE:
                failures.append(f"test: the two scorers disagree, {aucs['wp']:.4%} against {ours:.4%}")
            if abs(aucs["wp"] - DROW3_REPRODUCED_AP) > REPRODUCTION_TOLERANCE:
                failures.append(f"test: DROW3 no longer reproduces at {DROW3_REPRODUCED_AP:.1%}: {aucs['wp']:.4%}")
        results[name] = check
        del ds, split

    results["failures"] = failures
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps(results, indent=1), encoding="utf-8")
    for line in failures:
        print(f"FAIL  {line}")
    print("PASS" if not failures else f"{len(failures)} failure(s)", f"-> {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
