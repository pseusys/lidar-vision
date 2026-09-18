#!/usr/bin/env python3
"""What do a dataset's raw range readings actually contain? (`TODO.md` A49, spec approved 2026-09-17.)

The answer picks the denoising mechanism, if any (`memory/noise-structure.md`), so the report is grouped by those categories:

    missing returns   non-finite, non-positive, or at the dataset's no-return value. How long they stay invalid (frames) and how
                      wide they are (beams) separates open space, which is long and contiguous, from dropouts, which flicker.
                      Whether they gather on annotated people decides whether healing from past frames could recover anything.
    spikes            A-B-A: the middle reading differs from both neighbours by more than `--jump-m` while the neighbours agree
                      within `--agree-m`. A spike filter is only worth building if these are common.
    jitter            consecutive-frame disagreement per range band, against the sensor's rated accuracy and the 0.5 m match
                      radius. Temporal averaging is only worth it if this is large.
    steps             A-B-B, the same thresholds: real motion, reported for scale and never to be filtered.

Scans are read raw: FROG from `h5["scans"]` before `FROG_Dataset`'s clamp to 10 m, DROW from its CSV files. Robot motion is not
compensated, so part of the jitter and a few steps are the robot's own movement (FROG moves 1.88 cm per frame at p95).
Files are cut into chunks of `CHUNK_FRAMES`, and runs and triples are not followed across a chunk boundary.
No torch, no GPU, one chunk in memory at a time; still, do not run it beside a live training run (`memory/gotchas.md`).

Usage
-----
    python scan_anomalies.py --dataset frog
    python scan_anomalies.py --dataset drow
    python scan_anomalies.py --dataset frog --limit-files 1      # smoke run
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import h5py
import numpy as np
from scipy.ndimage import median_filter

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))

import follow_the_drow  # noqa: E402
from follow_the_drow.datasets.frog_dataset import FROG_Dataset, frog_laser_angles  # noqa: E402
from follow_the_drow.utils.drow_utils import laser_angles  # noqa: E402
from follow_the_drow.utils.file_utils import DROW_DATA_PATH, FROG_DATA_PATH, JRDB_DATA_PATH  # noqa: E402

_LIBRARY = Path(follow_the_drow.__file__).parent

MATCH_RADIUS_M = 0.5
SPIKE_JUMP_M = 0.3
SPIKE_AGREE_M = 0.1
LINK_GAP_S = FROG_Dataset.SESSION_GAP_S         # consecutive scans further apart than this are not neighbours in time
CHUNK_FRAMES = 20_000
EXAMPLES = 5
TRIPLE = 3

HISTOGRAM_TOP_MM = 100_000                      # above the UTM-30LX's 60 m maximum and FROG's 61.0 m sentinel; the rest shares one overflow bin
PEAK_WINDOW_MM = 101
PEAK_RATIO = 20.0
PEAK_MIN_SHARE = 1e-4
PEAK_MIN_COUNT = 1_000
PEAKS_REPORTED = 10
JITTER_TOP_MM = 1_000

RANGE_BANDS_M = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, np.inf)
FRAME_RUN_BINS = (1, 2, 3, 6, 13, 26, 126, 1251)        # lowest length per bin: 1, 2, 3-5, 6-12, 13-25, 26-125, 126-1250, 1251+
BEAM_RUN_BINS = (1, 2, 3, 6, 21, 81)                     # lowest length per bin: 1, 2, 3-5, 6-20, 21-80, 81+
ANNOTATION_RANGES_M = (10.0, 15.0)


@dataclass(frozen=True)
class Sensor:
    model: str
    min_range_m: float
    max_range_m: float
    accuracy: str
    source: str


SENSORS: Dict[str, Sensor] = {
    "UTM-30LX": Sensor("Hokuyo UTM-30LX", 0.1, 30.0, "+-30 mm at 0.1-10 m, +-50 mm at 10-30 m", "https://www.hokuyo-aut.jp/search/single.php?serial=169"),
    "S300": Sensor("SICK S300", 0.0, 30.0, "not published", "https://www.sick.com/media/pdf/5/45/845/dataSheet_S30B-2011BA_1026820_en.pdf"),
}

# `no_return_m`: readings at or above it are the recording's own encoding of "no return". DROW's 29.96 m was measured
# 2026-09-17 (11.9 % of all readings, plus 29.98 and 29.99). FROG writes no return two ways, measured the same day: +inf in the
# test file and the three extras, exactly 61.0 m in `frog_11-36_12-43_train_val.h5` (7.46M readings, no inf), just past the
# UTM-30LX's 60 m maximum. +inf is invalid anyway; 60.5 m catches the 61.0.
DATASETS = {
    "frog": {"sensor": "UTM-30LX", "no_return_m": 60.5},
    "drow": {"sensor": "S300", "no_return_m": 29.95},
    "jrdb": {"sensor": None, "no_return_m": None},
}


@dataclass
class Segment:
    """Consecutive raw scans `[T, B]` from one file, their timestamps, and per frame the annotated objects `[K, 2]` as
    (range, angle) -- or `None` for a frame nobody annotated, which is unknown rather than empty."""
    name: str
    scans: np.ndarray
    times: np.ndarray
    people: Optional[List[Optional[np.ndarray]]]


def invalid_mask(scans: np.ndarray, no_return_m: Optional[float]) -> np.ndarray:
    """Non-finite, non-positive, or at or above the recording's no-return value."""
    with np.errstate(invalid="ignore"):
        invalid = ~np.isfinite(scans) | (scans <= 0)
        if no_return_m is not None:
            invalid |= scans >= no_return_m
    return invalid


def invalid_runs(mask: np.ndarray, axis: int) -> tuple:
    """Every run of True along `axis` of a 2-D mask, each line separately: `(line, start, length)` arrays."""
    lines = mask if axis == 1 else mask.T
    padded = np.zeros((lines.shape[0], lines.shape[1] + 2), dtype=np.int8)
    padded[:, 1:-1] = lines
    edges = np.diff(padded, axis=1)
    line, starts = np.nonzero(edges == 1)
    _, ends = np.nonzero(edges == -1)
    return line, starts, (ends - starts).astype(np.int64)


def run_lengths(mask: np.ndarray, axis: int) -> np.ndarray:
    """Lengths of every run of True along `axis` of a 2-D mask, each line separately."""
    return invalid_runs(mask, axis)[2]


def sentinel_peaks(histogram_mm: np.ndarray) -> List[Dict[str, float]]:
    """Exact values far more frequent than the values around them: `histogram_mm[i]` counts readings rounding to i mm, and the
    last bin is overflow. A peak holds at least `PEAK_MIN_SHARE` of all readings and `PEAK_RATIO` times its neighbourhood's median."""
    counts = histogram_mm[:-1].astype(np.float64)
    total = float(histogram_mm.sum())
    if total == 0:
        return []
    # Recordings round their ranges (DROW to 1 cm), leaving most millimetre bins empty: compare each value with its
    # neighbourhood at the recording's own resolution, the most common spacing between occupied bins.
    occupied = np.nonzero(counts)[0]
    spacing = np.diff(occupied)
    quantum = int(np.bincount(spacing).argmax()) if len(spacing) else 1
    background = np.zeros_like(counts)
    background[::quantum] = median_filter(counts[::quantum], size=max(PEAK_WINDOW_MM // quantum, 3) | 1, mode="constant")
    if quantum > 1:
        background[occupied] = np.maximum(background[occupied], background[occupied - occupied % quantum])
    background = np.maximum(background, 1.0)
    floor = max(PEAK_MIN_COUNT, PEAK_MIN_SHARE * total)
    idx = np.nonzero((counts >= floor) & (counts >= PEAK_RATIO * background))[0]
    peaks = [{"value_m": i / 1000.0, "count": int(counts[i]), "share": counts[i] / total, "times_neighbourhood": counts[i] / background[i]} for i in idx]
    return sorted(peaks, key=lambda p: -p["count"])


def triple_kinds(scans: np.ndarray, invalid: np.ndarray, linked: np.ndarray, jump_m: float, agree_m: float) -> tuple:
    """Classify each triple of consecutive frames `(t-1, t, t+1)`, per beam, as spike (A-B-A) or step (A-B-B).

    Returns `[T-2, B]` masks `(spike, step, valid)`. A triple counts only when all three readings are valid and both frame gaps
    are linked (`linked[t]` says frames t and t+1 are neighbours in time). A step that is only the return from the previous
    triple's spike is not a step."""
    a, b, c = scans[:-2], scans[1:-1], scans[2:]
    valid = ~(invalid[:-2] | invalid[1:-1] | invalid[2:]) & (linked[:-1] & linked[1:])[:, None]
    with np.errstate(invalid="ignore"):
        jumped = np.abs(b - a) > jump_m
        spike = valid & jumped & (np.abs(c - b) > jump_m) & (np.abs(c - a) <= agree_m)
        step = valid & jumped & (np.abs(c - b) <= agree_m)
    step[1:] &= ~spike[:-1]
    return spike, step, valid


def near_people_mask(people: Sequence[np.ndarray], angles: np.ndarray, radius_m: float) -> np.ndarray:
    """`[T, B]`: the beam passes within `radius_m` of an annotated object's centre."""
    near = np.zeros((len(people), len(angles)), dtype=bool)
    sizes = np.array([len(p) for p in people], dtype=np.int64)
    if not sizes.sum():
        return near
    objects = np.concatenate([p for p in people if len(p)]).astype(np.float64)
    r, phi = objects[:, 0], objects[:, 1]
    half = np.where(r <= radius_m, np.pi, np.arcsin(np.minimum(radius_m / np.maximum(r, radius_m), 1.0)))
    offset = np.abs(np.angle(np.exp(1j * (np.asarray(angles, dtype=np.float64)[None, :] - phi[:, None]))))
    hit = offset <= half[:, None]
    starts = np.concatenate([[0], np.cumsum(sizes[sizes > 0])[:-1]])
    near[sizes > 0] = np.logical_or.reduceat(hit, starts, axis=0)
    return near


def _binned(lengths: np.ndarray, edges: Sequence[int]) -> Dict[str, Dict[str, int]]:
    """Runs and readings per length bin; `edges` are each bin's lowest length, and the last bin is open."""
    bins = np.searchsorted(edges, lengths, side="right") - 1
    runs = np.bincount(bins, minlength=len(edges))
    readings = np.bincount(bins, weights=lengths, minlength=len(edges))
    out = {}
    for i, lower in enumerate(edges):
        if i == len(edges) - 1:
            label = f"{lower}+"
        elif edges[i + 1] - lower == 1:
            label = str(lower)
        else:
            label = f"{lower}-{edges[i + 1] - 1}"
        out[label] = {"runs": int(runs[i]), "readings": int(readings[i])}
    return out


def _band_labels() -> List[str]:
    return [f"{lo:g}-{hi:g} m" if np.isfinite(hi) else f"{lo:g}+ m" for lo, hi in zip(RANGE_BANDS_M[:-1], RANGE_BANDS_M[1:])]


def _band(values: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(RANGE_BANDS_M, values, side="right") - 1, 0, len(RANGE_BANDS_M) - 2)


def _percentile_mm(histogram: np.ndarray, q: float) -> Optional[float]:
    total = histogram.sum()
    if total == 0:
        return None
    return float(np.searchsorted(np.cumsum(histogram), q * total))


def _examples(segment: str, scans: np.ndarray, mask: np.ndarray, room: int) -> List[dict]:
    """The first `room` triples in `mask`; `mask[t]` is the triple centred on frame t + 1."""
    if room <= 0:
        return []
    return [{"segment": segment, "frame": int(t) + 1, "beam": int(b), "ranges_m": [round(float(v), 3) for v in scans[t:t + TRIPLE, b]]} for t, b in np.argwhere(mask)[:room]]


@dataclass
class ScanAnomalies:
    """Accumulates every category over segments added one at a time, so no dataset has to be in memory at once."""
    sensor: Sensor
    angles: np.ndarray
    no_return_m: Optional[float]
    jump_m: float = SPIKE_JUMP_M
    agree_m: float = SPIKE_AGREE_M
    radius_m: float = MATCH_RADIUS_M
    link_gap_s: float = LINK_GAP_S
    examples: int = EXAMPLES
    _counts: Dict[str, float] = field(default_factory=dict)
    _per_beam: Optional[np.ndarray] = None
    _histogram: np.ndarray = field(default_factory=lambda: np.zeros(HISTOGRAM_TOP_MM + 2, dtype=np.int64))
    _intervals: np.ndarray = field(default_factory=lambda: np.zeros(int(LINK_GAP_S * 1e5) + 2, dtype=np.int64))   # linked intervals, 10 us bins
    _frame_runs: List[np.ndarray] = field(default_factory=list)
    _beam_runs: List[np.ndarray] = field(default_factory=list)
    _jitter: np.ndarray = field(default_factory=lambda: np.zeros((len(RANGE_BANDS_M) - 1) * (JITTER_TOP_MM + 2), dtype=np.int64))
    _bands: Dict[str, np.ndarray] = field(default_factory=dict)
    _annotations: List[np.ndarray] = field(default_factory=list)
    _examples: Dict[str, list] = field(default_factory=lambda: {"spikes": [], "steps": [], "longest_invalid": [], "annotations_beyond_rated_range": []})

    def _add(self, key: str, value: float) -> None:
        self._counts[key] = self._counts.get(key, 0) + value

    def _band_add(self, key: str, values: np.ndarray, where: np.ndarray) -> None:
        counts = np.bincount(_band(values[where]), minlength=len(RANGE_BANDS_M) - 1)
        self._bands[key] = self._bands.get(key, np.zeros(len(RANGE_BANDS_M) - 1, dtype=np.int64)) + counts

    def add(self, segment: Segment) -> None:
        scans = segment.scans.astype(np.float32, copy=False)
        dt = np.diff(segment.times.astype(np.float64))
        linked = (dt > 0) & (dt <= self.link_gap_s)
        self._add("readings", scans.size)
        self._add("frames", scans.shape[0])
        self._add("linked_intervals", int(linked.sum()))
        self._add("linked_seconds", float(dt[linked].sum()))
        self._intervals += np.bincount(np.round(dt[linked] * 1e5).astype(np.int64), minlength=self._intervals.size)[:self._intervals.size]
        self._values(scans)

        invalid = invalid_mask(scans, self.no_return_m)
        self._invalid(segment.name, invalid, linked)
        self._jitter_add(scans, invalid, linked)
        known, near = self._people(segment, invalid)
        if scans.shape[0] >= TRIPLE:
            self._triples(segment.name, scans, invalid, linked, known, near)

    def _values(self, scans: np.ndarray) -> None:
        finite = np.isfinite(scans)
        with np.errstate(invalid="ignore"):
            positive = finite & (scans > 0)
            self._add("nan", int(np.isnan(scans).sum()))
            self._add("posinf", int(np.isposinf(scans).sum()))
            self._add("neginf", int(np.isneginf(scans).sum()))
            self._add("zero", int((scans == 0).sum()))
            self._add("negative", int((finite & (scans < 0)).sum()))
        kept = scans[positive]
        self._add("below_rated_range", int((kept < self.sensor.min_range_m).sum()))
        self._add("above_rated_range", int((kept > self.sensor.max_range_m).sum()))
        if len(kept):
            self._counts["largest_finite_m"] = max(self._counts.get("largest_finite_m", 0.0), float(kept.max()))
            self._counts["smallest_positive_m"] = min(self._counts.get("smallest_positive_m", np.inf), float(kept.min()))
            mm = np.minimum(np.round(kept.astype(np.float64) * 1000.0), HISTOGRAM_TOP_MM + 1).astype(np.int64)
            self._histogram += np.bincount(mm, minlength=HISTOGRAM_TOP_MM + 2)

    def _invalid(self, name: str, invalid: np.ndarray, linked: np.ndarray) -> None:
        self._add("invalid", int(invalid.sum()))
        per_beam = invalid.sum(axis=0)
        self._per_beam = per_beam if self._per_beam is None else self._per_beam + per_beam
        self._beam_runs.append(run_lengths(invalid, axis=1))
        breaks = np.concatenate([[0], np.nonzero(~linked)[0] + 1, [len(invalid)]])
        for start, end in zip(breaks[:-1], breaks[1:]):
            beam, first, length = invalid_runs(invalid[start:end], axis=0)
            self._frame_runs.append(length)
            longest = self._examples["longest_invalid"]
            if len(length) and (not longest or length.max() > longest[0]["frames"]):
                i = int(np.argmax(length))
                self._examples["longest_invalid"] = [{"segment": name, "beam": int(beam[i]), "frames": int(length[i]), "from_frame": int(start + first[i])}]

    def _jitter_add(self, scans: np.ndarray, invalid: np.ndarray, linked: np.ndarray) -> None:
        pair_valid = ~(invalid[:-1] | invalid[1:]) & linked[:, None]
        earlier, later = scans[:-1][pair_valid], scans[1:][pair_valid]
        change_mm = np.minimum(np.round(np.abs(later - earlier).astype(np.float64) * 1000.0), JITTER_TOP_MM + 1).astype(np.int64)
        self._jitter += np.bincount(_band(earlier) * (JITTER_TOP_MM + 2) + change_mm, minlength=self._jitter.size)

    def _people(self, segment: Segment, invalid: np.ndarray) -> tuple:
        """Invalid readings near annotated objects against elsewhere. Returns `(known, near)`: which frames are annotated, and
        the `[T, B]` near mask (False on unannotated frames) -- both `None` when no frame is annotated."""
        if segment.people is None:
            return None, None
        known = np.array([p is not None for p in segment.people])
        if not known.any():
            return None, None
        people = [p for p in segment.people if p is not None]
        near = np.zeros_like(invalid)
        near[known] = near_people_mask(people, self.angles, self.radius_m)
        inv, close = invalid[known], near[known]
        self._add("annotated_frames", int(known.sum()))
        self._add("near_readings", int(close.sum()))
        self._add("near_invalid", int((inv & close).sum()))
        self._add("elsewhere_readings", int((~close).sum()))
        self._add("elsewhere_invalid", int((inv & ~close).sum()))

        sizes = np.array([len(p) for p in people], dtype=np.int64)
        objects = np.concatenate([p for p in people if len(p)]).astype(np.float64) if sizes.sum() else np.zeros((0, 2))
        self._annotations.append(objects[:, 0])
        room = self.examples - len(self._examples["annotations_beyond_rated_range"])
        if room > 0 and len(objects):
            frame = np.repeat(np.nonzero(known)[0], sizes)
            far = np.nonzero(objects[:, 0] > self.sensor.max_range_m)[0][:room]
            self._examples["annotations_beyond_rated_range"] += [{"segment": segment.name, "frame": int(frame[i]), "range_m": round(float(objects[i, 0]), 2), "angle_rad": round(float(objects[i, 1]), 3)} for i in far]
        return known, near

    def _triples(self, name: str, scans: np.ndarray, invalid: np.ndarray, linked: np.ndarray, known: Optional[np.ndarray], near: Optional[np.ndarray]) -> None:
        spike, step, valid = triple_kinds(scans, invalid, linked, self.jump_m, self.agree_m)
        self._band_add("triples", scans[:-2], valid)
        self._band_add("spikes", scans[:-2], spike)
        self._band_add("steps", scans[:-2], step)
        self._add("spikes", int(spike.sum()))
        self._add("steps", int(step.sum()))
        self._add("triples", int(valid.sum()))
        self._examples["spikes"] += _examples(name, scans, spike, self.examples - len(self._examples["spikes"]))
        self._examples["steps"] += _examples(name, scans, step, self.examples - len(self._examples["steps"]))
        if known is None:
            return
        middle, close = known[1:-1], near[1:-1]
        self._add("near_triples", int((valid & close)[middle].sum()))
        self._add("near_spikes", int((spike & close)[middle].sum()))
        self._add("elsewhere_triples", int((valid & ~close)[middle].sum()))
        self._add("elsewhere_spikes", int((spike & ~close)[middle].sum()))

    def result(self) -> dict:
        c = self._counts
        frame_runs = np.concatenate(self._frame_runs) if self._frame_runs else np.zeros(0, dtype=np.int64)
        beam_runs = np.concatenate(self._beam_runs) if self._beam_runs else np.zeros(0, dtype=np.int64)
        per_beam = self._per_beam if self._per_beam is not None else np.zeros(len(self.angles))
        frames = max(c.get("frames", 0), 1)
        labels = _band_labels()

        def ratio(num: str, den: str) -> Optional[float]:
            return c.get(num, 0) / c[den] if c.get(den) else None

        jitter = {}
        for label, h in zip(labels, self._jitter.reshape(len(labels), JITTER_TOP_MM + 2)):
            pairs = int(h.sum())
            over = float(h[int(MATCH_RADIUS_M * 1000) + 1:].sum() / pairs) if pairs else None
            jitter[label] = {"pairs": pairs, "median_mm": _percentile_mm(h, 0.5), "p90_mm": _percentile_mm(h, 0.9), "p99_mm": _percentile_mm(h, 0.99), "share_over_match_radius": over}

        def banded(key: str) -> Dict[str, dict]:
            triples = self._bands.get("triples", np.zeros(len(labels), dtype=np.int64))
            counts = self._bands.get(key, np.zeros(len(labels), dtype=np.int64))
            return {label: {"count": int(n), "per_valid_triple": (float(n) / float(d)) if d else None} for label, n, d in zip(labels, counts, triples)}

        rate_hz = (c["linked_intervals"] / c["linked_seconds"]) if c.get("linked_seconds") else None
        timestamps = {"median_rate_hz": None, "share_under_quarter_period": None, "p10_interval_ms": None, "p90_interval_ms": None}
        if rate_hz:
            cumulative = np.cumsum(self._intervals)
            total = cumulative[-1]
            median_s = np.searchsorted(cumulative, 0.5 * total) / 1e5
            timestamps = {
                "median_rate_hz": 1.0 / median_s if median_s > 0 else None,
                "share_under_quarter_period": float(self._intervals[:int(0.25 / rate_hz * 1e5)].sum() / total),
                "p10_interval_ms": float(np.searchsorted(cumulative, 0.1 * total) / 100.0),
                "p90_interval_ms": float(np.searchsorted(cumulative, 0.9 * total) / 100.0),
            }
        temporal, spatial = _binned(frame_runs, FRAME_RUN_BINS), _binned(beam_runs, BEAM_RUN_BINS)
        annotations = np.concatenate(self._annotations) if self._annotations else np.zeros(0)
        values = {k: int(c.get(k, 0)) for k in ("nan", "posinf", "neginf", "zero", "negative", "below_rated_range", "above_rated_range")}
        values |= {"largest_finite_m": c.get("largest_finite_m"), "smallest_positive_m": c.get("smallest_positive_m"), "frequent_values": sentinel_peaks(self._histogram)[:PEAKS_REPORTED]}
        return {
            "sensor": asdict(self.sensor),
            "no_return_m": self.no_return_m,
            "thresholds": {"jump_m": self.jump_m, "agree_m": self.agree_m, "radius_m": self.radius_m, "link_gap_s": self.link_gap_s},
            "readings": int(c.get("readings", 0)),
            "frames": int(c.get("frames", 0)),
            "rate_hz": rate_hz,
            "timestamps": timestamps,
            "values": values,
            "invalid": {
                "count": int(c.get("invalid", 0)),
                "share": ratio("invalid", "readings"),
                "per_beam_share": {"min": float(per_beam.min() / frames), "median": float(np.median(per_beam) / frames), "max": float(per_beam.max() / frames), "worst_beam": int(np.argmax(per_beam))},
                "temporal_runs": {k: v["runs"] for k, v in temporal.items()},
                "temporal_runs_readings": {k: v["readings"] for k, v in temporal.items()},
                "beam_runs": {k: v["runs"] for k, v in spatial.items()},
                "beam_runs_readings": {k: v["readings"] for k, v in spatial.items()},
                "annotated_frames": int(c.get("annotated_frames", 0)),
                "share_near_people": ratio("near_invalid", "near_readings"),
                "share_elsewhere": ratio("elsewhere_invalid", "elsewhere_readings"),
            },
            "spikes": {
                "count": int(c.get("spikes", 0)), "valid_triples": int(c.get("triples", 0)), "per_valid_triple": ratio("spikes", "triples"), "by_range": banded("spikes"),
                "near_people": int(c.get("near_spikes", 0)), "per_triple_near_people": ratio("near_spikes", "near_triples"), "per_triple_elsewhere": ratio("elsewhere_spikes", "elsewhere_triples"),
            },
            "steps": {"count": int(c.get("steps", 0)), "per_valid_triple": ratio("steps", "triples"), "by_range": banded("steps")},
            "jitter": jitter,
            "annotations": {
                "count": int(len(annotations)), "beyond_m": {f"{r:g}": int((annotations > r).sum()) for r in ANNOTATION_RANGES_M},
                "beyond_rated_range": int((annotations > self.sensor.max_range_m).sum()), "largest_m": float(annotations.max()) if len(annotations) else None,
            },
            "examples": self._examples,
        }


def _pct(x: Optional[float], digits: int = 2) -> str:
    return "n/a" if x is None else f"{100.0 * x:.{digits}f}%"


def format_report(name: str, r: dict) -> str:
    """The printed report, grouped by the denoising mechanism each category would call for. ASCII only (`AGENTS.md` I3)."""
    rate, s, t = r["rate_hz"], r["sensor"], r["thresholds"]
    header = f"=== {name}: {r['frames']:,} frames, {r['readings']:,} readings" + (f", {rate:.2f} Hz over linked intervals ===" if rate else " ===")
    lines = [
        header,
        f"sensor: {s['model']}, rated {s['min_range_m']:g}-{s['max_range_m']:g} m, accuracy {s['accuracy']}",
        f"no-return value: {r['no_return_m'] if r['no_return_m'] is not None else 'none (only non-finite and non-positive readings are invalid)'}",
    ]
    ts = r["timestamps"]
    if rate:
        lines.append(f"timestamps: mean rate {rate:.2f} Hz, median-interval rate {ts['median_rate_hz']:.2f} Hz; intervals p10 {ts['p10_interval_ms']:.2f} ms, p90 {ts['p90_interval_ms']:.2f} ms; "
                     f"{_pct(ts['share_under_quarter_period'])} under a quarter of the mean period (bunched stamps: never divide by them)")
    lines += ["", "--- Missing returns and sentinels -> validity masking or encoding, not denoising ---"]
    v = r["values"]
    lines.append(f"  NaN {v['nan']:,}  +inf {v['posinf']:,}  -inf {v['neginf']:,}  zero {v['zero']:,}  negative {v['negative']:,}")
    lines.append(f"  finite positive: below rated range {v['below_rated_range']:,}, above it {v['above_rated_range']:,}; smallest {v['smallest_positive_m']} m, largest {v['largest_finite_m']} m")
    lines += [f"  frequent value {p['value_m']:.3f} m: {p['count']:,} readings ({_pct(p['share'])}), {p['times_neighbourhood']:.0f}x its neighbourhood" for p in v["frequent_values"]]
    inv = r["invalid"]
    beams = inv["per_beam_share"]
    lines.append(f"  invalid: {inv['count']:,} ({_pct(inv['share'])}); per beam min {_pct(beams['min'])}, median {_pct(beams['median'])}, max {_pct(beams['max'])} (beam {beams['worst_beam']})")
    total = max(inv["count"], 1)
    lines.append("  how long a beam stays invalid, in frames" + (f" (1 frame = {1000.0 / rate:.0f} ms)" if rate else "") + ": runs, share of invalid readings")
    lines += [f"    {k:>9}: {inv['temporal_runs'][k]:>12,}  {_pct(inv['temporal_runs_readings'][k] / total)}" for k in inv["temporal_runs"]]
    lines.append("  how wide an invalid stretch is within one scan, in beams: runs, share of invalid readings")
    lines += [f"    {k:>9}: {inv['beam_runs'][k]:>12,}  {_pct(inv['beam_runs_readings'][k] / total)}" for k in inv["beam_runs"]]
    lines.append(f"  on {inv['annotated_frames']:,} annotated frames: invalid on {_pct(inv['share_near_people'])} of beams passing within {t['radius_m']:g} m of an annotated object, {_pct(inv['share_elsewhere'])} elsewhere")
    lines += [f"  longest invalid run: {e['frames']:,} frames on beam {e['beam']} in {e['segment']} from frame {e['from_frame']}" for e in r["examples"]["longest_invalid"]]

    sp = r["spikes"]
    lines += ["", f"--- Single-frame spikes (A-B-A: jump over {t['jump_m']:g} m both ways, ends agree within {t['agree_m']:g} m) -> spike filter ---"]
    lines.append(f"  {sp['count']:,} of {sp['valid_triples']:,} valid triples ({_pct(sp['per_valid_triple'], 4)}); near annotated objects {_pct(sp['per_triple_near_people'], 4)}, elsewhere {_pct(sp['per_triple_elsewhere'], 4)}")
    lines += [f"    {k:>8}: {b['count']:>10,}  {_pct(b['per_valid_triple'], 4)}" for k, b in sp["by_range"].items()]
    lines += [f"  example: {e['segment']} frame {e['frame']} beam {e['beam']}: {e['ranges_m']}" for e in r["examples"]["spikes"]]

    lines += ["", f"--- Jitter (consecutive valid readings on one beam) -> temporal averaging, if large next to the {t['radius_m']:g} m match radius ---"]
    lines += [f"    {k:>8}: {j['pairs']:>12,} pairs  median {j['median_mm']} mm  p90 {j['p90_mm']} mm  p99 {j['p99_mm']} mm  over {t['radius_m']:g} m {_pct(j['share_over_match_radius'])}" for k, j in r["jitter"].items()]

    st = r["steps"]
    lines += ["", "--- Steps (A-B-B) -> real motion, never filtered; for scale ---", f"  {st['count']:,} ({_pct(st['per_valid_triple'], 4)} of valid triples)"]
    lines += [f"    {k:>8}: {b['count']:>10,}  {_pct(b['per_valid_triple'], 4)}" for k, b in st["by_range"].items()]
    lines += [f"  example: {e['segment']} frame {e['frame']} beam {e['beam']}: {e['ranges_m']}" for e in r["examples"]["steps"]]

    a = r["annotations"]
    beyond = ", ".join(f"{k} m: {n:,}" for k, n in a["beyond_m"].items())
    lines += ["", "--- Annotations ---", f"  {a['count']:,} annotated objects; beyond {beyond}; beyond the rated range: {a['beyond_rated_range']:,}; largest {a['largest_m']} m"]
    lines += [f"  beyond rated range: {e['segment']} frame {e['frame']} at {e['range_m']} m" for e in r["examples"]["annotations_beyond_rated_range"]]
    return "\n".join(lines)


def frog_segments(limit_files: Optional[int]) -> Iterator[Segment]:
    """Every FROG file, raw `h5["scans"]` before the loader's clamp, in chunks; annotations from the circles' Cartesian columns
    (every FROG frame is annotated, so an empty frame is known to be empty)."""
    for path in sorted((_LIBRARY / FROG_DATA_PATH).glob("*.h5"))[:limit_files]:
        with h5py.File(path, "r") as h5:
            times = h5["timestamps"][:].astype(np.float64)
            circles = h5["circles"][:].astype(np.float64)
            circle_idx = h5["circle_idx"][:].astype(np.int64)
            circle_num = h5["circle_num"][:].astype(np.int64)
            for start in range(0, len(times), CHUNK_FRAMES):
                end = min(start + CHUNK_FRAMES, len(times))
                people = []
                for i in range(start, end):
                    objects = circles[circle_idx[i]:circle_idx[i] + circle_num[i]]
                    people.append(np.stack([np.hypot(objects[:, 0], objects[:, 1]), np.arctan2(objects[:, 1], objects[:, 0])], axis=1))
                yield Segment(f"{path.name}[{start}:{end}]", h5["scans"][start:end].astype(np.float32), times[start:end], people)


def _drow_annotations(csv: Path) -> Dict[int, list]:
    """Scan id -> every annotated object of any class, as (range, angle)."""
    out: Dict[int, list] = {}
    for suffix in (".wp", ".wc", ".wa"):
        path = csv.with_suffix(suffix)
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                scan_id, objects = line.split(",", 1)
                out.setdefault(int(scan_id), []).extend(json.loads(objects))
    return out


def drow_segments(limit_files: Optional[int]) -> Iterator[Segment]:
    """Every DROW CSV (train, val, test) in chunks; every object class counts as annotated, and unannotated scans are unknown."""
    for path in sorted((_LIBRARY / DROW_DATA_PATH).glob("*/*.csv"))[:limit_files]:
        table = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
        annotations = _drow_annotations(path)
        for start in range(0, len(table), CHUNK_FRAMES):
            part = table[start:start + CHUNK_FRAMES]
            ids = part[:, 0].astype(np.int64)
            people = [np.asarray(annotations[i], dtype=np.float64).reshape(-1, 2) if i in annotations else None for i in ids]
            yield Segment(f"{path.parent.name}/{path.name}[{start}:{start + len(part)}]", part[:, 2:].astype(np.float32), part[:, 1], people)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--no-return-m", type=float, default=None, help="override the dataset's no-return value: readings at or above it are invalid")
    ap.add_argument("--jump-m", type=float, default=SPIKE_JUMP_M, help="a spike's middle reading differs from both neighbours by more than this")
    ap.add_argument("--agree-m", type=float, default=SPIKE_AGREE_M, help="while the neighbours agree within this")
    ap.add_argument("--limit-files", type=int, default=None, help="read only the first N files (smoke run)")
    ap.add_argument("--out", type=Path, default=None, help="JSON output (default: results/scan_anomalies_<dataset>.json)")
    args = ap.parse_args()

    if args.dataset == "jrdb":
        raise SystemExit(f"JRDB is not on disk ({_LIBRARY / JRDB_DATA_PATH}): it needs manual registration (memory/gotchas.md), and its sensors are not in SENSORS yet.")
    config = DATASETS[args.dataset]
    no_return_m = args.no_return_m if args.no_return_m is not None else config["no_return_m"]
    angles = frog_laser_angles(720) if args.dataset == "frog" else laser_angles(450)
    segments = frog_segments(args.limit_files) if args.dataset == "frog" else drow_segments(args.limit_files)
    checker = ScanAnomalies(SENSORS[config["sensor"]], np.asarray(angles, dtype=np.float64), no_return_m, jump_m=args.jump_m, agree_m=args.agree_m)
    for segment in segments:
        checker.add(segment)
        print(f"  read {segment.name}: {len(segment.times):,} frames", flush=True)

    result = checker.result()
    out = args.out or _HERE.parent / "results" / f"scan_anomalies_{args.dataset}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("\n" + format_report(args.dataset, result), flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
