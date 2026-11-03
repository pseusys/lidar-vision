#!/usr/bin/env python3
"""Does stage 2's coarse convolution earn anything end to end? Scores the six detectors -- E13's object memory on
the no-temporal and on the default (coarse) stage 2, three seeds each, both on the adopted recipe -- by the rule written down
before the runs:

    drop the convolution if the default's three-seed mean test AP is not above the no-temporal mean by more than 1.5 pp,
    and its first-second recall is not above by more than 2 pp; keep it otherwise

**First-second recall** needs no identities and no model. Annotations are linked into trajectories in world coordinates
(`motion_analysis.build_tracks`, 0.6 m, as the age strata are), and a person-observation is in its first second when its trajectory
began under a second earlier: the people a slot cannot have seen yet, where beam-level motion is the one cue a memory lacks. FROG's
copied labels are left out of it, since a copy would start a trajectory of its own and only a double report could match it. Recall,
there and overall, is read at one false-positive budget shared by all six: the no-temporal arm's mean false positives per frame at
0.3. AP keeps the official labels, every 5th annotated test frame, as every stage-3 number here.

Every run's test frames, and the trajectory ages, are saved under `checkpoints_three_horizon/a60/`, so re-scoring needs no GPU.

Usage (from utils/)
-----
    python coarse_e2e.py
"""
import argparse
import json
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from tracking_baselines import CHECKPOINTS, PrecisionRecall, load, save, stride_frames, subset  # noqa: E402

ARMS = {"static": "a60_e13_static_s{}", "default": "a60_e13_default_s{}"}
SEEDS = (0, 1, 2)
OUT_DIR = CHECKPOINTS / "a60"
TRACK_GATE_M = 0.6
FIRST_SECOND_S = 1.0
AP_BAR_PP = 1.5
FIRST_SECOND_BAR_PP = 2.0
OPERATING_THRESHOLD = 0.3
TIGHT_RADIUS_M = 0.3
STREAMS = 64

Person = Tuple[int, int]            # (frame, index into that frame's annotated people)


def trajectory_ages(people: Sequence[np.ndarray], frames: Sequence[int], times: Mapping[int, float], gate: float = TRACK_GATE_M) -> Dict[Person, float]:
    """Per annotated person-observation of one recording, seconds since its trajectory's first annotation. `people[i]` `[G, 2]` are
    frame `frames[i]`'s people in world coordinates; a row equal to an earlier row of its frame is a copied label and gets no age."""
    from motion_analysis import build_tracks

    kept = [[j for j in range(len(p)) if not (j and np.all(p[:j] == p[j], axis=1).any())] for p in people]
    unique = [np.asarray(p, np.float64).reshape(-1, 2)[k] for p, k in zip(people, kept, strict=True)]
    by_frame = dict(zip(frames, zip(unique, kept, strict=True), strict=True))
    ages = {}
    for track in build_tracks(unique, scan_index=list(frames), gate=gate):
        start = times[min(track)]
        for f, xy in track.items():
            points, index = by_frame[f]
            ages[(f, index[int(np.argmin(np.linalg.norm(points - xy[None], axis=1)))])] = float(times[f] - start)
    return ages


def first_second_recall(covered: Mapping[int, np.ndarray], ages: Mapping[Person, float], max_age_s: float = FIRST_SECOND_S) -> Tuple[int, int]:
    """(covered, counted) over the person-observations under `max_age_s` old whose frame was scored."""
    hits = counted = 0
    for (f, j), age in ages.items():
        if age < max_age_s and f in covered:
            counted += 1
            hits += int(covered[f][j])
    return hits, counted


def verdict(ap_gain_pp: float, first_second_gain_pp: float) -> str:
    """The decision rule, written before the runs."""
    return "keep" if ap_gain_pp > AP_BAR_PP or first_second_gain_pp > FIRST_SECOND_BAR_PP else "drop"


def run_split(run: Path, device):
    """A run's memory checkpoint and the test split of the stage-2 candidates it was trained on."""
    import torch
    from train_takheliped import load_split

    checkpoint = torch.load(run / "step3a_object_memory.best.pth", map_location=device, weights_only=False)
    return checkpoint, load_split("test", 0, Path(checkpoint["args"]["stage2"]), device)


def memory_frames(run: Path, device, streams: int):
    """A run's memory output on every annotated test frame, with its split."""
    from train_takheliped import evaluate, evaluation_segments, object_memory

    checkpoint, split = run_split(run, device)
    model = object_memory(checkpoint, split.data.features.shape[-1]).to(device)
    out = {}
    evaluate(model, split, device, 1, streams, evaluation_segments(split, 0, 0, 0), frames_out=out, compute_metrics=False)
    order = np.argsort(out["index"], kind="stable")
    return [out["memory"][k] for k in order], [out["index"][k] for k in order], split


def all_ages(split) -> Dict[Person, float]:
    """Every annotated test person-observation's trajectory age, recording by recording."""
    from motion_analysis import to_world

    d, every, ages = split.data, stride_frames(split, 1), {}
    for offset, n in zip(split.offsets, split.lengths, strict=True):
        here = [int(g) for g in every if offset <= g < offset + n]
        world = [to_world(d.gt_xy[g][d.gt_valid[g]].astype(np.float64), d.pose[g]) for g in here]
        ages.update(trajectory_ages(world, here, {g: float(d.time[g]) for g in here}))
    return ages


def main() -> None:
    from train_takheliped import MATCH_RADIUS_M, candidate_targets, detect_device

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--streams", type=int, default=STREAMS)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device, _ = detect_device()

    ages_path = OUT_DIR / "ages.pkl"
    ages = pickle.loads(ages_path.read_bytes()) if ages_path.exists() else None
    saved, found = {}, []
    for arm, pattern in ARMS.items():
        for seed in SEEDS:
            name = pattern.format(seed)
            path = OUT_DIR / f"{name}_test.pkl"
            if not path.exists():
                if not (CHECKPOINTS / name / "step3a_results.json").exists():
                    print(f"{name}: not trained yet, skipped", flush=True)
                    continue
                frames, index, split = memory_frames(CHECKPOINTS / name, device, args.streams)
                save(path, frames, index, run=name)
                if ages is None:
                    ages = dict(ages=all_ages(split), scored=stride_frames(split))
                    ages_path.write_bytes(pickle.dumps(ages))
                del split
            saved[(arm, seed)] = load(path)
            found.append(name)
    if not saved:
        raise SystemExit("no coarse-convolution run has finished training yet")
    if ages is None:                                 # every run's frames were saved before the ages were cached
        _, split = run_split(CHECKPOINTS / found[0], "cpu")
        ages = dict(ages=all_ages(split), scored=stride_frames(split))
        ages_path.write_bytes(pickle.dumps(ages))
        del split
    ages, scored = ages["ages"], ages["scored"]

    n5 = len(scored)
    rows = {}
    for key, s in saved.items():
        frames5 = subset(s["frames"], s["index"], scored)
        pr = PrecisionRecall(frames5)
        rows[key] = dict(pr=pr, ap=pr.ap(), ap_03=PrecisionRecall(frames5, radius=TIGHT_RADIUS_M).ap(), fp_at_03=pr.at_threshold(OPERATING_THRESHOLD, n5)["fp_per_frame"])
    statics = [r for (arm, _), r in rows.items() if arm == "static"] or list(rows.values())      # all runs, until a no-temporal one exists
    budget = float(np.mean([r["fp_at_03"] for r in statics]))
    for (arm, seed), r in rows.items():
        point = r.pop("pr").at_fp_budget(budget, n5)
        covered = {}
        for g, (scores, xy, gt) in zip(saved[(arm, seed)]["index"], saved[(arm, seed)]["frames"], strict=True):
            keep = np.asarray(scores) >= point["threshold"]
            covered[int(g)] = candidate_targets(np.asarray(xy)[keep], np.ones(int(keep.sum()), bool), np.asarray(gt, np.float64), MATCH_RADIUS_M)[1]
        hits, counted = first_second_recall(covered, ages)
        r.update(matched_fp_recall=point["recall"], matched_fp_threshold=point["threshold"], first_second_recall=hits / counted if counted else float("nan"), first_second_people=counted)

    means = {arm: {k: float(np.mean([r[k] for (a, _), r in rows.items() if a == arm])) for k in ("ap", "ap_03", "matched_fp_recall", "first_second_recall")}
             for arm in ARMS if any(a == arm for a, _ in rows)}
    result = dict(budget_fp_per_frame=budget, budget_from="static" if any(a == "static" for a, _ in rows) else "all runs", runs={f"{a}_s{s}": r for (a, s), r in rows.items()}, means=means)
    if len(means) == len(ARMS):
        gain = {k: 100 * (means["default"][k] - means["static"][k]) for k in means["static"]}
        result.update(gain_pp=gain, verdict=verdict(gain["ap"], gain["first_second_recall"]))
    for (arm, seed), r in sorted(rows.items()):
        print(f"{arm:8s} s{seed}  AP {r['ap']:.2%}  AP@0.3 {r['ap_03']:.2%}  recall at {budget:.3f} FP/frame {r['matched_fp_recall']:.2%}  "
              f"first-second recall {r['first_second_recall']:.2%} (n={r['first_second_people']})", flush=True)
    if "verdict" in result:
        print("default - static: " + "  ".join(f"{k} {v:+.2f} pp" for k, v in result["gain_pp"].items()) + f"  -> {result['verdict'].upper()}", flush=True)
    (OUT_DIR / "a60.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'a60.json'}", flush=True)


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
