#!/usr/bin/env python3
"""Which property of DROW makes locality matter? Our stage 2 against the cutout, scored per bin of each candidate.

Our stage 2 fits DROW's training recordings (63.6% train AP) and reaches its test recordings at 29.5%; DROW3's cutout,
retrained here on the same frames, scores 66.5% / 63.2%. A cutout sees one depth-normalised metre around each beam and
our network sees most of the scan, so the question is which property of DROW makes the wide view a liability. Three
candidates, each measured where it can separate the two models (`TODO.md` item 102):

1. **Scene novelty.** Each test frame's distance to the nearest *training* scan, as the mean absolute difference of
   coarse sector medians (the room's shape, which a person a few beams wide barely moves) over small rotations. The
   same distance for FROG says whether FROG's test ever leaves the familiar range at all.
2. **Look-alike objects.** Whether a wheelchair or a walker is within a metre of a person in the frame.
3. **Repeated backgrounds.** Train AP per kind of training recording -- driving, the lunch room, staged with the robot
   stationary, turning in place -- for both models: memorising shows as our stage 2 far above the cutout where the
   background never changes.

**What would name the property, written before the run.** A cause makes our deficit to the cutout grow along it while
the cutout stays roughly flat: our AP falling across the novelty quartiles with the cutout's level says the wide view
fails on unseen scenes; a deficit concentrated where look-alikes stand next to people says the wide view cannot tell
them apart; our train AP towering over the cutout's on the stationary recordings says repetition is what is learnt.

**The second pass, added after the first** found the deficit flat at ~30 pp in every novelty and look-alike bin, and
what looked like FROG's test scenes at distance zero from its training ones -- a ROCm `cdist` fault, which reading 5 is
what caught (`memory/gotchas.md`):

4. **Ranking within a frame against ranking across frames.** Pooled AP puts every frame's detections on one scale;
   `top_k_hits` asks only whether each frame's own best detections are its people, and AP over the frames with a person
   in them drops the scans with nobody there. Near the cutout within frames and far below it pooled says our scores do
   not compare between scans; AP per test recording reads the same at the scale of a whole recording.
5. **Whether FROG's zero is real.** The share of descriptor sectors at the clip, the distance again at the sensor's
   30 m, and the share of test scans found byte for byte in train.
6. **Whether 4's gap is DROW's.** The same readings for our FROG stage 2 on FROG test, every 5th annotated frame (its
   80.89%): pooled AP over within-frame hits at 0.75 or more says DROW's, 0.60 or less a trait of our model that FROG's
   test tolerates, written before the run (`TODO.md` row 102).

7. **What moves our score level** (`score-level`): per populated DROW test scan, the rank correlation of model-free
   scan statistics with each model's top false positive and mean true positive, and each statistic's quartiles on DROW
   train and FROG test. It found no statistic that moves ours and not the cutout's, and one pair that separates the
   splits: DROW test's returns are close (median 1.74 m against DROW train's 4.33 m).
8. **Geometry inside DROW test** (`geometry`): AP per bin of where a scan's median return range falls in DROW train's.
   Almost every test scan is in the bottom bin, so it could not reach the familiar regime.
9. **Geometry on DROW val** (`val-geometry`), which spans both regimes, with and without val's sister segments of train
   recordings, and per bin the readings no person-free scan moves. Prediction written before the run: geometry is the
   trigger if ours' deficit in the top bin is at most half the bottom bin's in both scopes.

    python drow_locality_property.py
    python drow_locality_property.py --readings drow-test frog-ranking --out ../results/drow_locality_ranking.json
    python drow_locality_property.py --readings val-geometry --out ../results/drow_val_geometry.json
"""

import re
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from json import dumps
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

from check_drow_scoring import detections_by_frame
from evaluate import _load_dataset, eval_nn_model
from takheliped.utils.drow_utils import project_cartesian_from_polar
from train import detect_device
from train_takheliped import (DROW, FROG, MATCH_RADIUS_M, EVAL_BATCH, FrameSplit, calibration_frames, calibration_network,
                              frames_from_dataset, load_frames, recording_bounds, score_detections)

_HERE = Path(__file__).parent
CHECKPOINTS = _HERE.parent / "checkpoints_three_horizon"
#: The headline stage 2's two seeds, and DROW3 retrained here on DR-SPAAM's recipe (`TODO.md` item 92).
OURS = {"ours_s0": CHECKPOINTS / "drow_stage2_s0" / "step2_calibration.best.pth",
        "ours_s1": CHECKPOINTS / "drow_headline_s1" / "step2_calibration.best.pth"}
CUTOUT = _HERE.parent / "checkpoints_drow" / "drow_t1_drspaam_recipe.pth"
#: Our FROG stage 2 whose train and test AP (93.66% / 80.89%) item 96 measured, and the stride that 80.89% is at.
OURS_FROG = CHECKPOINTS / "a69_no_aggregator_s0" / "step2_calibration.best.pth"
FROG_TEST_STRIDE = 5
READINGS = ("novelty", "drow-test", "drow-train", "frog-ranking", "score-level", "geometry", "val-geometry")
#: The rule for reading 7, written before its run (`TODO.md` row 102): a statistic moves our level at |rho| >= 0.3 for
#: both seeds with the cutout under half of ours, and is the property if DROW test's median also leaves the
#: interquartile range of both DROW train and FROG test.
LEVEL_RHO = 0.3
LEVELS = ("fp_top", "tp_mean")
#: Reading 8's bins: where a DROW test scan's median return range falls in DROW train's distribution.
GEOMETRY_PERCENTILES = (10, 25)
GEOMETRY_BINS = ("under DROW train's p10", "DROW train's p10-p25", "DROW train's p25 or above")
GEOMETRY_STATS = ("returns_within_10m", "median_range_m")
#: The descriptor: 2.5 deg sectors of ranges clipped at FROG's 10 m annotation limit, so both datasets are read alike.
SECTOR_DEG = 2.5
CLIP_M = 10.0
#: The second clip, at both sensors' rated range, so a scene beyond 10 m is told apart from another.
CLIP_WIDE_M = 30.0
#: Rotations tried when matching a scene, either way: 4 sectors, 10 deg.
MAX_SHIFT = 4
#: Every Nth scan of a training split goes into the bank of scenes seen; DROW at 12.7 Hz, FROG at 40 Hz.
BANK_STRIDE = {"drow": 5, "frog": 15}
#: Every Nth annotated FROG test frame is a query; FROG annotates every scan.
FROG_QUERY_STRIDE = 10
CONFUSER_M = 1.0
NOVELTY_BINS = ("novelty q1", "novelty q2", "novelty q3", "novelty q4")
CONFUSER_BINS = ("look-alike within 1 m of a person", "look-alike in frame, none near a person", "no look-alike")
RECORDING_KINDS = ("driving", "lunch", "stationary", "turning")


def scan_descriptor(scans: np.ndarray, spacing_deg: float, sector_deg: float = SECTOR_DEG, clip_m: float = CLIP_M) -> np.ndarray:
    """Scans `[F, N]` as the median range of each `sector_deg` sector, every reading clipped at `clip_m` and a missing
    one read as the clip: the room's coarse shape, which a person a few beams wide barely moves."""
    per = max(1, round(sector_deg / spacing_deg))
    n = scans.shape[1] // per * per
    r = np.asarray(scans[:, :n], np.float32)
    r = np.where(np.isfinite(r) & (r > 0), np.minimum(r, clip_m), clip_m)
    return np.median(r.reshape(len(r), n // per, per), axis=2)


def nearest_distance(query: np.ndarray, bank: np.ndarray, max_shift: int, device, chunk: int = 512) -> np.ndarray:
    """Each query descriptor's mean absolute difference, in metres, to its closest bank descriptor, over rotations of up
    to `max_shift` sectors either way, each compared on the sectors the two share.

    Explicit differences in `chunk` x `chunk` blocks, not `torch.cdist`: on this machine's ROCm build `cdist(p=1)`
    returns wrong distances, most of them zero, from a few thousand query rows on (`memory/gotchas.md`)."""
    q_all = torch.as_tensor(np.asarray(query, np.float32), device=device)
    b_all = torch.as_tensor(np.asarray(bank, np.float32), device=device)
    sectors = q_all.shape[1]
    best = torch.full((len(q_all),), float("inf"), device=device)
    for k in range(-max_shift, max_shift + 1):
        q_k = q_all[:, max(0, k):sectors + min(0, k)]
        b_k = b_all[:, max(0, -k):max(0, -k) + q_k.shape[1]]
        for qs in range(0, len(q_k), chunk):
            q = q_k[qs:qs + chunk, None, :]
            for bs in range(0, len(b_k), chunk):
                d = (q - b_k[None, bs:bs + chunk]).abs().mean(dim=2).min(dim=1).values
                best[qs:qs + chunk] = torch.minimum(best[qs:qs + chunk], d)
    return best.cpu().numpy().astype(np.float64)


def confuser_bins(people_xy: Sequence[np.ndarray], others_xy: Sequence[np.ndarray], radius: float = CONFUSER_M) -> np.ndarray:
    """Per frame: 0 if a wheelchair or walker stands within `radius` of a person, 1 if one is in the frame but near
    nobody, 2 if there is none."""
    out = np.full(len(people_xy), 2, np.int64)
    for i, (people, others) in enumerate(zip(people_xy, others_xy, strict=True)):
        if not len(others):
            continue
        near = len(people) and np.linalg.norm(np.asarray(people)[:, None] - np.asarray(others)[None], axis=-1).min() < radius
        out[i] = 0 if near else 1
    return out


def ap_by_bin(frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], bins: np.ndarray, names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """`score_detections` over each bin's frames alone, keyed by the bin's name."""
    return {name: score_detections([f for f, b in zip(frames, bins, strict=True) if b == i]) for i, name in enumerate(names)}


def top_k_hits(frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], radius: float = MATCH_RADIUS_M) -> float:
    """Of the people in frames that have any, the share found by their own frame's `k` best detections, `k` the frame's
    number of people, each detection taking the nearest person within `radius` that no better detection took: ranking
    inside the frame alone, which no per-scene score offset can move."""
    hits = people = 0
    for scores, xy, gt in frames:
        if not len(gt):
            continue
        gt = np.asarray(gt, np.float64)
        people += len(gt)
        free = np.ones(len(gt), bool)
        for j in np.argsort(-np.asarray(scores), kind="stable")[:len(gt)]:
            d = np.where(free, np.linalg.norm(gt - xy[j], axis=1), np.inf)
            i = int(d.argmin())
            if d[i] <= radius:
                free[i] = False
                hits += 1
    return hits / people if people else float("nan")


def ranking_summary(frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, float]:
    """Pooled AP over every frame, pooled AP over the frames with a person in them, and `top_k_hits`. What separates the
    first two is confident detections on scans with nobody there; what separates the last two is how scores compare
    between scans that do have people."""
    populated = [f for f in frames if len(f[2])]
    return dict(pooled_ap=score_detections(frames)["ap"], populated_ap=score_detections(populated)["ap"],
                top_k_hits=top_k_hits(frames), frames=len(frames), person_free=len(frames) - len(populated))


def scan_levels(frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], radius: float = MATCH_RADIUS_M) -> Tuple[np.ndarray, np.ndarray]:
    """Per scan, the mean score of its true positives and the highest score of its false positives (NaN where it has
    none), each detection in score order taking the nearest person within `radius` that no better detection took."""
    tp_mean, fp_top = np.full(len(frames), np.nan), np.full(len(frames), np.nan)
    for f, (scores, xy, gt) in enumerate(frames):
        gt = np.asarray(gt, np.float64).reshape(-1, 2)
        free, tps, fps = np.ones(len(gt), bool), [], []
        for j in np.argsort(-np.asarray(scores), kind="stable"):
            d = np.where(free, np.linalg.norm(gt - xy[j], axis=1), np.inf) if len(gt) else np.array([np.inf])
            i = int(d.argmin())
            if d[i] <= radius:
                free[i] = False
                tps.append(scores[j])
            else:
                fps.append(scores[j])
        if tps:
            tp_mean[f] = float(np.mean(tps))
        if fps:
            fp_top[f] = float(max(fps))
    return tp_mean, fp_top


def scan_statistics(scans: np.ndarray, people_xy: Sequence[np.ndarray], spacing_deg: float, near_m: float = CLIP_M,
                    jump_m: float = 0.5) -> Dict[str, np.ndarray]:
    """Per scan, model-free: the share of beams with a return within `near_m`, those returns' median range, range
    jumps over `jump_m` between neighbouring returns per 10 degrees of field of view (clutter), and its people's count
    and mean range."""
    r = np.asarray(scans, np.float64)
    near = np.isfinite(r) & (r > 0) & (r < near_m)
    with np.errstate(invalid="ignore"):                    # inf - inf between two missing returns; masked out below
        jumps = (np.abs(np.diff(r, axis=1)) > jump_m) & near[:, 1:] & near[:, :-1]
    median = np.array([np.median(row[m]) if m.any() else np.nan for row, m in zip(r, near, strict=True)])
    return dict(returns_within_10m=near.mean(axis=1), median_range_m=median,
                jumps_per_10deg=jumps.sum(axis=1) / (r.shape[1] * spacing_deg / 10.0),
                people=np.array([len(p) for p in people_xy], np.float64),
                people_range_m=np.array([np.hypot(*np.asarray(p).T).mean() if len(p) else np.nan for p in people_xy]))


def reference_bins(values: np.ndarray, reference: np.ndarray, percentiles: Sequence[float]) -> np.ndarray:
    """Each value's bin among `reference`'s `percentiles`: 0 below the first, then one bin per edge passed. An undefined
    value -- a scan with no return in range has no median range -- reads as the open end."""
    edges = np.nanpercentile(reference, percentiles)
    return np.digitize(np.where(np.isnan(values), np.inf, values), edges)


def sister_recordings(names: Sequence[str], train_names: Sequence[str]) -> np.ndarray:
    """Whether each recording is a lettered segment (`-a`, `-b`, ...) of a recording another segment of which is in
    `train_names`: the same place, minutes apart."""
    def base(n: str) -> str:
        return re.sub(r"-[a-z]$", "", n)
    train_bases = {base(n) for n in train_names}
    return np.array([bool(re.search(r"-[a-z]$", n)) and base(n) in train_bases for n in names])


def print_ranking(title: str, table: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{title}: pooled AP, the same over populated frames only, and each frame's own k best detections (k = its people)", flush=True)
    for m, r in table.items():
        print(f"  {m:<10} pooled AP {r['pooled_ap']:.2%}   populated only {r['populated_ap']:.2%}   top-k hits {r['top_k_hits']:.2%}   "
              f"pooled / top-k {r['pooled_ap'] / r['top_k_hits']:.2f}   ({r['person_free']} of {r['frames']} frames person-free)", flush=True)


def exact_twins(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Whether each query scan occurs byte for byte among the bank's scans."""
    seen = {row.tobytes() for row in np.ascontiguousarray(bank, np.float32)}
    return np.array([row.tobytes() in seen for row in np.ascontiguousarray(query, np.float32)])


def recording_type(name: str) -> str:
    """DROW's four kinds of recording, from the file name: `run` driving, `lunch` one room, `rot` turning in place, and
    the staged scenes (`n`, `r`, `s`, `s2w2p`) with the robot standing still."""
    prefix = Path(name).name.split("_")[0].split("-")[0]
    return {"run": "driving", "lunch": "lunch", "rot": "turning"}.get(prefix, "stationary")


def others_xy(split: FrameSplit) -> List[np.ndarray]:
    """Every annotated wheelchair and walker per annotated frame, in the detection frame."""
    return [np.array([project_cartesian_from_polar(r, phi) for q in t for r, phi in q], np.float64).reshape(-1, 2)
            for t in split.other_by_class]


def annotated_recordings(split: FrameSplit) -> np.ndarray:
    """The recording index of each annotated frame, in the order the loader read the recordings."""
    offsets, _ = recording_bounds(split.start)
    return np.searchsorted(offsets, split.annotated, side="right") - 1


def ours_frames(split: FrameSplit, device) -> Dict[str, list]:
    angles = DROW.angles()
    out = {}
    for name, path in OURS.items():
        net = calibration_network(torch.load(path, map_location="cpu"), angles).to(device)
        out[name] = calibration_frames(net, split, np.arange(len(split.annotated)), device, EVAL_BATCH)
    return out


def cutout_frames(ds, cfg, split: FrameSplit) -> list:
    """DROW3's detections through `evaluate.py`'s own pass, paired with `FrameSplit`'s labels (identical, item 93)."""
    aucs = eval_nn_model("drow", CUTOUT, ds, cfg, eval_r=MATCH_RADIUS_M, batch_size=16, return_detections=True)
    return [(s, xy, people) for (s, xy), people in zip(detections_by_frame(aucs["detections"]), split.people_xy, strict=True)]


def print_table(title: str, table: Dict[str, Dict[str, Dict[str, float]]], bins: Sequence[str]) -> None:
    print(f"\n{title}", flush=True)
    print(f"  {'bin':<42}" + "".join(f"{m:>10}" for m in table) + f"{'frames':>8}{'people':>8}")
    for b in bins:
        any_model = next(iter(table.values()))[b]
        print(f"  {b:<42}" + "".join(f"{table[m][b]['ap']:>10.2%}" for m in table) + f"{any_model['frames']:>8}{any_model['people']:>8}")


def score_level_reading(split: FrameSplit, played: Dict[str, list], novelty_m: np.ndarray) -> Dict[str, object]:
    """Reading 7: over populated DROW test scans, each statistic's rank correlation with each model's per-scan levels,
    and its distribution on DROW train, DROW test and FROG test; the verdict by `LEVEL_RHO`'s rule."""
    stats = scan_statistics(split.scans[split.annotated], split.people_xy, DROW.spacing_deg)
    populated = stats["people"] > 0
    test_stats = {**stats, "novelty_m": novelty_m}
    levels = {m: dict(zip(LEVELS, scan_levels(f)[::-1])) for m, f in played.items()}
    rho = {s: {m: {lv: float(spearmanr(v[populated], levels[m][lv][populated], nan_policy="omit")[0]) for lv in LEVELS}
               for m in played} for s, v in test_stats.items()}
    quartiles = {"drow_test": {s: np.nanpercentile(v[populated], [25, 50, 75]).tolist() for s, v in stats.items()}}
    for spec, name, split_name, stride in ((DROW, "drow_train", "train", 1), (FROG, "frog_test", "test", FROG_TEST_STRIDE)):
        other = load_frames(split_name, "official", spec)
        rows = np.arange(0, len(other.annotated), stride)
        s = scan_statistics(other.scans[other.annotated[rows]], [other.people_xy[a] for a in rows], spec.spacing_deg)
        quartiles[name] = {k: np.nanpercentile(v[s["people"] > 0], [25, 50, 75]).tolist() for k, v in s.items()}
        del other
    verdict = {}
    for s in test_stats:
        moves = [lv for lv in LEVELS
                 if min(abs(rho[s]["ours_s0"][lv]), abs(rho[s]["ours_s1"][lv])) >= LEVEL_RHO
                 and np.sign(rho[s]["ours_s0"][lv]) == np.sign(rho[s]["ours_s1"][lv])
                 and abs(rho[s]["cutout"][lv]) < 0.5 * min(abs(rho[s]["ours_s0"][lv]), abs(rho[s]["ours_s1"][lv]))]
        separates = s in stats and all(not quartiles[o][s][0] <= quartiles["drow_test"][s][1] <= quartiles[o][s][2]
                                       for o in ("drow_train", "frog_test"))
        verdict[s] = dict(moves_our_level=moves, separates_the_splits=bool(separates), is_the_property=bool(moves and separates))
    print("\nDROW test, populated scans: Spearman rho of each statistic with each model's level (fp = top false positive, "
          "tp = mean true positive)", flush=True)
    print(f"  {'statistic':<20}" + "".join(f"{m + ' ' + lv[:2]:>13}" for lv in LEVELS for m in played) + "   verdict", flush=True)
    for s in test_stats:
        v = verdict[s]
        print(f"  {s:<20}" + "".join(f"{rho[s][m][lv]:>13.2f}" for lv in LEVELS for m in played)
              + f"   moves {','.join(v['moves_our_level']) or '-'}; separates {v['separates_the_splits']}", flush=True)
    print("\nThe same statistics, populated scans, quartiles p25 / p50 / p75", flush=True)
    for s in stats:
        print(f"  {s:<20}" + "".join(f"   {o}: " + " / ".join(f"{x:.3g}" for x in quartiles[o][s]) for o in ("drow_train", "drow_test", "frog_test")),
              flush=True)
    return dict(rho=rho, quartiles=quartiles, verdict=verdict, populated_scans=int(populated.sum()))


def geometry_reading(split: FrameSplit, played: Dict[str, list]) -> Dict[str, object]:
    """Reading 8: DROW test scored per bin of where each scan's median return range falls in DROW train's, and the two
    geometry statistics' quartiles on every split that matters -- DROW train, val and test, FROG train and test."""
    train = load_frames("train", "official", DROW)
    reference = scan_statistics(train.scans[train.annotated], train.people_xy, DROW.spacing_deg)["median_range_m"]
    del train
    test_stats = scan_statistics(split.scans[split.annotated], split.people_xy, DROW.spacing_deg)
    bins = reference_bins(test_stats["median_range_m"], reference, GEOMETRY_PERCENTILES)
    by_bin = {m: ap_by_bin(f, bins, GEOMETRY_BINS) for m, f in played.items()}
    print_table("DROW test AP by where the scan's median return range falls in DROW train's", by_bin, GEOMETRY_BINS)
    quartiles = {}
    for spec, name, split_name, stride in ((DROW, "drow_train", "train", 1), (DROW, "drow_val", "val", 1), (DROW, "drow_test", "test", 1),
                                           (FROG, "frog_train", "train", 25), (FROG, "frog_test", "test", FROG_TEST_STRIDE)):
        other = load_frames(split_name, "official", spec)
        rows = np.arange(0, len(other.annotated), stride)
        s = scan_statistics(other.scans[other.annotated[rows]], [other.people_xy[a] for a in rows], spec.spacing_deg)
        quartiles[name] = {k: np.nanpercentile(s[k], [25, 50, 75]).tolist() for k in GEOMETRY_STATS}
        del other
    print("\nGeometry of every annotated scan, quartiles p25 / p50 / p75", flush=True)
    for k in GEOMETRY_STATS:
        print(f"  {k}", flush=True)
        for name, q in quartiles.items():
            print(f"    {name:<12}" + " / ".join(f"{x:.3g}" for x in q[k]), flush=True)
    edges = np.nanpercentile(reference, GEOMETRY_PERCENTILES).tolist()
    return dict(by_bin=by_bin, train_edges_m=edges, quartiles=quartiles,
                test_scans_per_bin=np.bincount(bins, minlength=len(GEOMETRY_BINS)).tolist())


def val_geometry_reading(device) -> Dict[str, object]:
    """Reading 9: reading 8's bins on DROW val, which spans both geometries where DROW test does not, scored over every
    val recording and over the ones with no sister segment in train alone (a familiar scene would flatter the wide view)."""
    train_ds = DROW.load("train", None, "official")
    train = frames_from_dataset(train_ds, DROW)
    reference = scan_statistics(train.scans[train.annotated], train.people_xy, DROW.spacing_deg)["median_range_m"]
    train_names = [Path(n).stem for n in train_ds.filenames]
    del train_ds, train
    ds, cfg = _load_dataset(SimpleNamespace(dataset="drow", split="val"))
    split = frames_from_dataset(ds, DROW)
    played = {**ours_frames(split, device), "cutout": cutout_frames(ds, cfg, split)}
    names = [Path(n).stem for n in ds.filenames]
    sister = sister_recordings(names, train_names)[annotated_recordings(split)]
    bins = reference_bins(scan_statistics(split.scans[split.annotated], split.people_xy, DROW.spacing_deg)["median_range_m"],
                          reference, GEOMETRY_PERCENTILES)
    out = dict(no_sister_recordings=[n for n, s in zip(names, sister_recordings(names, train_names), strict=True) if not s])
    for scope, keep in (("all_val", np.ones(len(bins), bool)), ("no_sister_in_train", ~sister)):
        idx = np.nonzero(keep)[0]
        table = {m: ap_by_bin([f[i] for i in idx], bins[idx], GEOMETRY_BINS) for m, f in played.items()}
        # The density check: the enclosed bins are sparse in people, which pooled AP punishes on its own, so each bin also
        # gets the two readings no person-free scan moves -- AP over populated scans, and the within-scan hits.
        ranking = {b: {m: ranking_summary([f[i] for i in idx[bins[idx] == k]]) for m, f in played.items()}
                   for k, b in enumerate(GEOMETRY_BINS)}
        out[scope] = dict(by_bin=table, by_bin_ranking=ranking, scans_per_bin=np.bincount(bins[idx], minlength=len(GEOMETRY_BINS)).tolist(),
                          whole={m: ranking_summary([f[i] for i in idx]) for m, f in played.items()})
        print_table(f"DROW val ({scope.replace('_', ' ')}) AP by where the scan's median return range falls in DROW train's",
                    table, GEOMETRY_BINS)
        for b in GEOMETRY_BINS:
            print_ranking(f"  {b}", ranking[b])
    print(f"\nval recordings with no sister segment in train: {', '.join(out['no_sister_recordings'])}", flush=True)
    return out


def main() -> int:
    ap = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=_HERE.parent / "results" / "drow_locality_property.json")
    ap.add_argument("--readings", nargs="+", choices=READINGS, default=list(READINGS),
                    help="which readings to run (default all); drow-test bins by DROW's novelty, so it computes that either "
                         "way, and drow-train is twenty minutes of the cutout on the CPU")
    args = ap.parse_args()
    device, _ = detect_device()
    results = {"novelty": {}}

    # 1. Scene novelty: test frames against the scenes their own training split shows, at both clips, with the share
    # of test scans recorded byte for byte in train (reading 5).
    novelty = {}
    drow_test = bool({"drow-test", "score-level", "geometry"} & set(args.readings))
    specs = ((DROW, "drow"), (FROG, "frog")) if "novelty" in args.readings else ((DROW, "drow"),) if drow_test else ()
    for spec, name in specs:
        train = load_frames("train", "official", spec)
        test = load_frames("test", "official", spec)
        rows = np.arange(len(test.annotated)) if name == "drow" else np.arange(0, len(test.annotated), FROG_QUERY_STRIDE)
        query = test.scans[test.annotated[rows]]
        entry = dict(frames=len(rows), exact_twins=float(exact_twins(query, train.scans).mean()))
        for clip in (CLIP_M, CLIP_WIDE_M):
            bank = scan_descriptor(train.scans[::BANK_STRIDE[name]], spec.spacing_deg, clip_m=clip)
            described = scan_descriptor(query, spec.spacing_deg, clip_m=clip)
            dist = nearest_distance(described, bank, MAX_SHIFT, device)
            if clip == CLIP_M:
                novelty[name] = dist
            q = np.percentile(dist, [10, 25, 50, 75, 90])
            entry[f"clip_{clip:g}m"] = dict(zip(("p10", "p25", "p50", "p75", "p90"), q.tolist()), sectors_at_clip=float((described >= clip).mean()),
                                            bank_sectors_at_clip=float((bank >= clip).mean()), at_zero=float((dist < 1e-3).mean()))
            print(f"{name} test, clip {clip:g} m, distance to the nearest training scene (m): p10 {q[0]:.3f}  p25 {q[1]:.3f}  median {q[2]:.3f}  "
                  f"p75 {q[3]:.3f}  p90 {q[4]:.3f}  ({len(rows)} frames; {(dist < 1e-3).mean():.1%} under 1 mm, "
                  f"{(described >= clip).mean():.1%} of test sectors and {(bank >= clip).mean():.1%} of train sectors at the clip)", flush=True)
        print(f"{name} test scans found byte for byte in train: {entry['exact_twins']:.2%}", flush=True)
        results["novelty"][name] = entry
        del train, test, bank

    if drow_test:
        # 2. Both models on DROW test, scored per novelty quartile and per look-alike bin.
        ds, cfg = _load_dataset(SimpleNamespace(dataset="drow", split="test"))
        split = frames_from_dataset(ds, DROW)
        played = {**ours_frames(split, device), "cutout": cutout_frames(ds, cfg, split)}
        quartile = np.digitize(novelty["drow"], np.percentile(novelty["drow"], [25, 50, 75]))
        look_alike = confuser_bins(split.people_xy, others_xy(split))
        results["test_by_novelty"] = {m: ap_by_bin(f, quartile, NOVELTY_BINS) for m, f in played.items()}
        results["test_by_look_alike"] = {m: ap_by_bin(f, look_alike, CONFUSER_BINS) for m, f in played.items()}
        results["novelty_quartile_edges_m"] = np.percentile(novelty["drow"], [25, 50, 75]).tolist()
        print_table("DROW test AP by scene novelty (q1 most familiar)", results["test_by_novelty"], NOVELTY_BINS)
        print_table("DROW test AP by look-alikes near people", results["test_by_look_alike"], CONFUSER_BINS)

        # 4. Ranking inside each frame against the pooled ranking, and AP per test recording.
        recordings = [Path(ds.filenames[r]).stem for r in range(len(ds.filenames))]
        results["test_ranking"] = {m: ranking_summary(f) for m, f in played.items()}
        results["test_by_recording"] = {m: ap_by_bin(f, annotated_recordings(split), recordings) for m, f in played.items()}
        print_ranking("DROW test", results["test_ranking"])
        print_table("DROW test AP per recording", results["test_by_recording"], recordings)
        if "score-level" in args.readings:
            results["score_level"] = score_level_reading(split, played, novelty["drow"])
        if "geometry" in args.readings:
            results["geometry"] = geometry_reading(split, played)
        del ds, split, played

    if "drow-train" in args.readings:
        # 3. Train AP per kind of training recording, for one seed of ours and the cutout.
        ds, cfg = _load_dataset(SimpleNamespace(dataset="drow", split="train"))
        split = frames_from_dataset(ds, DROW)
        kinds = np.array([RECORDING_KINDS.index(recording_type(ds.filenames[r])) for r in annotated_recordings(split)])
        net = calibration_network(torch.load(OURS["ours_s0"], map_location="cpu"), DROW.angles()).to(device)
        train_played = {"ours_s0": calibration_frames(net, split, np.arange(len(split.annotated)), device, EVAL_BATCH),
                        "cutout": cutout_frames(ds, cfg, split)}
        results["train_by_recording_kind"] = {m: ap_by_bin(f, kinds, RECORDING_KINDS) for m, f in train_played.items()}
        print_table("DROW train AP by kind of recording", results["train_by_recording_kind"], RECORDING_KINDS)
        del ds, split, train_played

    if "val-geometry" in args.readings:
        results["val_geometry"] = val_geometry_reading(device)

    if "frog-ranking" in args.readings:
        # 6. Our FROG stage 2, the same readings as 4, through the pass that scored its 80.89%.
        split = load_frames("test", "official", FROG)
        net = calibration_network(torch.load(OURS_FROG, map_location="cpu"), FROG.angles()).to(device)
        frames = calibration_frames(net, split, np.arange(0, len(split.annotated), FROG_TEST_STRIDE), device, EVAL_BATCH)
        results["frog_test_ranking"] = {"ours_frog": ranking_summary(frames)}
        print_ranking(f"FROG test, every {FROG_TEST_STRIDE}th annotated frame", results["frog_test_ranking"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
