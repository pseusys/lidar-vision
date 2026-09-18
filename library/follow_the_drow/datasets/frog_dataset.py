"""
FROG dataset loader.

Reference paper:
  "FROG: A new people detection dataset for knee-high 2D range finders"
  arXiv:2306.08531  /  Frontiers in Robotics and AI, 2025
  Universidad Pablo de Olavide Service Robotics Lab

Dataset home:
  https://robotics.upo.es/datasets/frog/laser2d_people/

The FROG dataset uses a Hokuyo UTM-30LX 2D LiDAR (720 beams, 180° FoV, 0.25°
resolution, 40 Hz, mounted at 35 cm height).  Data is stored in HDF5 files.
Every scan in the dataset is annotated (unlike DROW where only 5% of scans have
labels), with annotations for pedestrians only (no wheelchair / walker split).

This class mirrors the DROW_Dataset interface so it can be used as a drop-in
replacement in verify_dataset.py, compare_detectors.py, etc.

Coordinate conventions
----------------------
FROG uses a standard robotics frame: X = forward, Y = left, angles are CCW from
the X-axis.  Annotation (x, y) values are converted to (r, phi) pairs compatible
with the existing ``project_cartesian_from_polar(r, phi)`` helper, which uses the same convention
(phi = 0 forward, phi > 0 left).

Laser angles
------------
Beam 0 is at -90° (rightmost), beam 719 is at ~+89.75° (leftmost):
  angle_i = -π/2 + i * π/720   for i = 0 … 719

Scan-to-annotation mapping
--------------------------
Every scan is annotated.  The HDF5 file stores:
  circle_idx[i]  – index of the first annotation in ``circles`` for scan i
  circle_num[i]  – number of annotations belonging to scan i

Modes and splits
----------------
``mode`` picks a *partition* of the six recordings; ``split`` picks one part of
it. Every mode assigns whole recordings, so no window, no guard band and no
near-duplicate frame can ever bridge two splits (`_MODES`).

* ``official`` (the default) -- the published benchmark. ``train`` is the
  paper's own ``split == 0`` half of ``frog_11-36_12-43_train_val.h5``
  (108,356 frames); ``test`` is ``frog_16-41_test.h5``, the recording the
  paper's Table 4 reports on; ``val`` is the three further published
  recordings nothing else touches. Populated frames only -- which is what both
  official files already are, so it is a no-op on them.
* ``transferred`` -- the *same* ``train`` and ``val`` as ``official``, so one
  checkpoint serves both, and a ``test`` split of the raw recordings'
  **person-free** frames. Measures the thing the benchmark structurally
  cannot: what a detector does when there is nobody there.
* ``balanced`` -- our own partition. One populated-only recording plus one raw
  recording per split, so ``train``, ``val`` and ``test`` all see crowded,
  sparse and empty scenes in near-identical proportion.

Frames outside the requested split are *not* included in ``det_id`` but ARE
kept in ``scans`` / ``odoms``, so ``get_scan()`` can still return unbroken
temporal windows through them. A whole *recording* belonging to another split
is the exception: it is not loaded at all, so nothing can draw history across
a split boundary.
"""

from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple, Union
import re
import shutil
import ssl
import urllib.request

import numpy as np
from numpy import array

from ..utils.file_utils import FROG_DATA_PATH
from ..utils.generic_utils import Logging

_DATASET_PATH = Path(__file__).parent.parent / FROG_DATA_PATH

# Official download base URL (file names appended at runtime)
_BASE_URL = "https://robotics.upo.es/datasets/frog/laser2d_people/data"

# Known HDF5 files in the FROG release
_H5_FILES: Dict[str, str] = {
    "test":       "frog_16-41_test.h5",
    "train_val":  "frog_11-36_12-43_train_val.h5",
    "extra_1031": "frog_10-31.h5",
    "extra_1457": "frog_14-57.h5",
    "extra_1553": "frog_15-53.h5",
}


# --- Loading modes -----------------------------------------------------------
#
# One dataset, three partitions (TODO.md A30). A mode decides, for each split,
# *which recordings* it reads and *which of their frames* get scored -- and it
# is the only thing that decides either, so a partition is one table entry
# rather than a scatter of conditionals.
#
# This table is a whitelist for a reason. It used to be a hint for the
# downloader alone, while loading globbed every .h5 in the directory -- so
# `split="test"` returned all 120,396 frames of
# frog_11-36_12-43_train_val.h5, because frog_16-41_test.h5 was never on disk
# and the glob simply picked up whatever was. Every FROG "test" number measured
# before 2026-09-10 is train-set accuracy as a result (TODO.md A11,
# CHANGELOG.md). The symmetric half of the same bug is why an exclusion list
# would not do: the test file carries no `split` field, so a glob-based
# `split="train"` would pull all of it into training through the "no split
# field -> every frame is a detection" branch.
#
# **Recordings, not files, are the unit.** frog_11-36_12-43_train_val.h5
# bundles two recordings ~24.6 h apart and `balanced` sends them to different
# splits, so a spec entry is `(file_key, recording_index)`, with `None` meaning
# "every recording in this file". Recording boundaries come from
# `_split_at_gaps()` at `RECORDING_GAP_S`.
#
# Frame rules decide which frames of an included recording land in `det_id`,
# and so get scored. Frames failing the rule stay in `scans`/`odoms`, so a
# temporal window still draws unbroken history through them:
#
#   "paper_train"  the FROG paper's own `split == 0` half, reproducing their
#                  training set exactly (108,356 frames). Their `split == 1`
#                  half is a per-frame random holdout sitting ~38 ms from
#                  training frames -- useless as a holdout and not theirs to
#                  train on, so it is dropped (TODO.md A12).
#   "populated"    frames carrying at least one annotated person. The
#                  composition both official files already have (train_val and
#                  test are 100% populated), so a no-op there and a protocol
#                  match on the raw extras, which are only ~62-65% populated.
#                  Getting this wrong cost a val wp-AUC of 50.8% against a
#                  train probe of 85.7% -- a +35pp gap that was pure
#                  composition artifact.
#   "empty"        frames carrying none. Every detection is a false positive by
#                  construction, which is exactly the point (TODO.md A31).
#   "all"          every frame, in the proportion the sensor produced them.
_RecordingSpec = Tuple[Tuple[str, Optional[int]], ...]

_MODES: Dict[str, Dict[str, Tuple[_RecordingSpec, str]]] = {
    # The published benchmark, and the default: nothing moves under a caller
    # that does not ask for a mode.
    "official": {
        "train": ((("train_val", None),), "paper_train"),
        "val":   ((("extra_1031", None), ("extra_1457", None),
                   ("extra_1553", None)), "populated"),
        "test":  ((("test", None),), "populated"),
    },
    # Same model, different question. `train` and `val` are *identical* to
    # official's, deliberately: a transferred number has to be the official
    # checkpoint measured somewhere else, not a different experiment run under
    # a different name (TODO.md A32b). Only `test` moves.
    #
    # Its test frames come from the same three recordings official uses for
    # `val`, disjointly -- populated frames there, person-free frames here.
    # Nothing trains on either, but model *selection* has seen the venue, which
    # can only make the detector look better on it. That is the conservative
    # direction for a claim that false positives will be high, so it is
    # recorded rather than engineered around.
    "transferred": {
        "train": ((("train_val", None),), "paper_train"),
        "val":   ((("extra_1031", None), ("extra_1457", None),
                   ("extra_1553", None)), "populated"),
        "test":  ((("extra_1031", None), ("extra_1457", None),
                   ("extra_1553", None)), "empty"),
    },
    # Our own partition. Each split gets one populated-only recording plus one
    # raw recording, whole -- see _BALANCED_COMPOSITION for why this pairing
    # and not one of the other five.
    "balanced": {
        "train": ((("train_val", 1), ("extra_1457", None)), "all"),
        "val":   ((("train_val", 0), ("extra_1031", None)), "all"),
        "test":  ((("test", None), ("extra_1553", None)), "all"),
    },
}

DEFAULT_MODE = "official"

# `balanced`, measured on the real files rather than assumed:
#
# | split | recordings    | frames  | empty | people / populated frame |
# | ----- | ------------- | ------- | ----- | ------------------------ |
# | train | 12-43 + 14-57 | 133,367 | 20.6% | 3.70 |
# | val   | 11-36 + 10-31 | 121,329 | 19.8% | 3.52 |
# | test  | 16-41 + 15-53 | 110,846 | 19.1% | 3.20 |
#
# **Whole recordings, not blocks.** Cutting each recording into patches and
# dealing them round-robin would let every split see all six environments,
# which is more diversity -- but adjacent blocks of one recording share venue,
# lighting and the same people in the same clothes, so it trades the property
# this project spent a session buying back (TODO.md A11-A14) for a diversity
# gain it cannot verify. Cross-recording splits also measure the kind of
# generalization that actually matters here.
#
# **Why this pairing.** Three populated-only recordings (11-36, 12-43, 16-41,
# all 0% empty) and three raw ones (10-31 37.4%, 14-57 39.2%, 15-53 34.8%) pair
# up six ways. This one minimises the spread of empty frames across splits --
# 20.6 / 19.8 / 19.1 %, against up to 17.0-22.9 % for the alternatives. The
# residual prior shift, 3.70 people per populated frame in train against 3.20
# in test, is *smaller* than `official`'s own (3.93 -> 3.07).
#
# **Why 16-41 + 15-53 is the test split, specifically.** Two reasons, both
# load-bearing:
#
#  1. The published static baselines (LFE-Peaks, LFE-PPN, DROW) were trained by
#     their authors on FROG's official train set, i.e. on 11-36 and 12-43.
#     Neither appears here, so `balanced`'s test is genuinely held out for them
#     and A32's numbers are honest. `balanced`'s **val** does contain 11-36 --
#     so never report a published baseline's number on it.
#  2. It contains frog_16-41 whole, so the official test recording stays
#     extractable as a subset of the same evaluation pass. One run yields both
#     the balanced number and one directly comparable to every published one.
_BALANCED_COMPOSITION = {
    "train": {"recordings": ("12-43", "14-57"), "frames": 133_367,
              "empty_frac": 0.206, "people_per_populated_frame": 3.70},
    "val":   {"recordings": ("11-36", "10-31"), "frames": 121_329,
              "empty_frac": 0.198, "people_per_populated_frame": 3.52},
    "test":  {"recordings": ("16-41", "15-53"), "frames": 110_846,
              "empty_frac": 0.191, "people_per_populated_frame": 3.20},
}

# Frames closer than this to a *sequence* start cannot fill their temporal
# window, so get_scan() clamps and repeats the earliest available scan -- a
# padded, motion-free history paired with a correct label. Harmless for a
# single-frame detector, aimed squarely at what this project studies.
#
# A **fixed** width, deliberately, never `(T-1) * dtime`: sizing it from dtime
# would give every point of a dtime sweep a different frame set, which is the
# same trap the train/val guard band fell into. 40 covers T=5, dtime=10; a
# wider dtime means raising this and re-running everything at the new value.
#
# Measured cost of dropping them: train -1.81%, val -0.00% (three unbroken
# recordings, no pauses), test -3.75% (47 fragments, median 688 scans).
RUNUP_SCANS = 40

# Test keeps them. The FROG paper scores every populated scan, so trimming
# 3.75% of the test frames would break comparability with every published
# number in order to fix a smaller contaminant. Report the clean subset
# separately when the temporal analysis needs it (TODO.md A21).
#
# `balanced`'s test inherits the 0 for the same reason: it contains frog_16-41
# whole so that the official-test subset stays extractable from the same pass,
# and that only works if the frames match.
_SPLIT_RUNUP = {"train": RUNUP_SCANS, "val": RUNUP_SCANS, "test": 0}

# `transferred`'s test is the one evaluation split with no published number to
# stay comparable with -- it reads the raw extras, which nobody has scored. It
# takes the standard warm-up instead, because its metric is a false-positive
# count and a padded (frozen-history) window is precisely the degenerate input
# that would distort one.
_MODE_RUNUP_OVERRIDE = {("transferred", "test"): RUNUP_SCANS}

# The FROG authors publish real per-session odometry as separate files named
# after each individual recording's start time (frog_<HH-MM>_odom.npz), same
# data/ directory as the .h5 files -- NOT one-per-.h5-file, since
# frog_11-36_12-43_train_val.h5 itself bundles two sessions (11:36 and
# 12:43) into one file. _SESSION_TOKEN_RE pulls those HH-MM tokens out of an
# .h5 filename, in the order they appear (which matches chronological order
# for every known filename), to build the matching odom filename(s).
_SESSION_TOKEN_RE = re.compile(r"\d{2}-\d{2}")


def _session_odom_filenames(h5_filename: str) -> List[str]:
    tokens = _SESSION_TOKEN_RE.findall(h5_filename)
    return [f"frog_{t}_odom.npz" for t in tokens]


def _split_at_gaps(timestamps: np.ndarray, gap_s: float) -> List[Tuple[int, int]]:
    """Split a timestamp array into [start, end) index ranges at every gap
    longer than *gap_s*.

    Used at two scales, with two different thresholds and two different jobs:

    * ``RECORDING_GAP_S`` (300 s) separates the genuine recordings a single
      .h5 bundles -- frog_11-36_12-43_train_val.h5 holds two, ~24.6 hours
      apart. Recordings are the unit the train/val carve works on, so a val
      block never straddles one.
    * ``SESSION_GAP_S`` (0.5 s) separates *recording pauses* within one
      recording -- 60 of them across the two training recordings, up to 74.3 s
      long, together 15-20% of each recording's wall time. Sessions are the
      unit ``get_scan()`` windows are clamped to, so a nominal 0.38 s window
      can no longer silently span 74 seconds of a coffee break (TODO.md A13).
    """
    if len(timestamps) == 0:
        return []
    gap_idx = np.where(np.diff(timestamps) > gap_s)[0]
    starts = [0] + [int(i) + 1 for i in gap_idx]
    ends = [int(i) + 1 for i in gap_idx] + [len(timestamps)]
    return list(zip(starts, ends))


def _recordings_per_file(
    rec_spec: _RecordingSpec,
) -> Dict[str, Optional[FrozenSet[int]]]:
    """Collapse a mode's ``(file_key, recording_index)`` spec into
    ``{file_key: recording indices or None}``, preserving the order the spec
    lists the files in.

    ``None`` means "every recording in this file" and wins over any explicit
    index for the same file, so a spec cannot accidentally narrow a file it
    also asked for whole.
    """
    out: Dict[str, Optional[FrozenSet[int]]] = {}
    for key, rec in rec_spec:
        if key not in _H5_FILES:
            raise ValueError(f"Unknown FROG file key {key!r} in mode table.")
        if rec is None:
            out[key] = None
        elif key not in out:
            out[key] = frozenset({rec})
        elif out[key] is not None:
            out[key] = out[key] | {rec}
    return out


def _ssl_open(url: str):
    """
    Open *url* over HTTPS, working around the SSL certificate issues that are
    common on Windows (Python does not use the Windows certificate store by
    default).

    Resolution order:
      1. ``certifi`` CA bundle — install with ``pip install certifi`` (recommended).
      2. Default system SSL context — works on Linux / macOS out of the box.
      3. Unverified context — last resort, prints a clear warning.
    """
    # 1. certifi
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(url, context=ctx)
    except ImportError:
        pass

    # 2. Default SSL
    try:
        return urllib.request.urlopen(url)
    except urllib.error.URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise

    # 3. Unverified — last resort
    print(
        "\n  WARNING: SSL certificate verification failed and certifi is not installed.\n"
        "  Falling back to unverified HTTPS — install certifi to suppress this:\n"
        "    pip install certifi\n"
    )
    ctx = ssl._create_unverified_context()
    return urllib.request.urlopen(url, context=ctx)


def _download_file(url: str, dest: Path) -> None:
    """Download *url* → *dest* with a progress bar, SSL-safe on Windows."""
    print(f"  Downloading {dest.name} …", flush=True)
    with _ssl_open(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(1 << 20)  # 1 MB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    mb_done = downloaded / 1e6
                    mb_total = total / 1e6
                    print(f"\r    {pct:3d}%  {mb_done:.1f} / {mb_total:.1f} MB",
                          end="", flush=True)
    print(f"\r  Saved -> {dest}" + " " * 30)


LEGACY_MISSING_RETURN_M = 10.0


def encode_missing_returns(scans: np.ndarray, missing_return_m: Optional[float]) -> np.ndarray:
    """Non-finite readings rewritten to ``missing_return_m``, or the raw readings when it is ``None``.

    The rewrite is legacy preprocessing that every baseline number in this repo was measured with (10 m, the annotation limit),
    and it is incomplete: ``frog_11-36_12-43_train_val.h5`` writes no return as a finite 61.0 m, which it never touches
    (`memory/noise-structure.md`). The three-horizon detector passes ``None`` and sanitises readings itself.
    """
    if missing_return_m is None:
        return scans
    return np.where(np.isfinite(scans), scans, missing_return_m).astype(scans.dtype, copy=False)


def frog_laser_angles(n: int = 720) -> np.ndarray:
    """
    Return the beam angles (radians) for the FROG Hokuyo UTM-30LX scanner.

    180° FoV, 720 beams, 0.25° angular resolution.
    Angle 0 is forward (FROG X-axis / robot forward direction).
    Positive angles are counterclockwise (left side of the robot).

    Beam layout:
      beam 0   → -π/2  (rightmost,  -90°)
      beam 359 →  0    (forward,     0°)
      beam 719 → +π/2 - π/720  (leftmost, ≈ +89.75°)
    """
    return np.linspace(-np.pi / 2, np.pi / 2, n, endpoint=False, dtype=np.float32)


class FROG_Dataset(Logging):
    """
    Loader for the FROG 2D LiDAR person-detection dataset.

    Parameters
    ----------
    datapath : Path or str
        Directory containing the FROG .h5 files.
        Defaults to ``library/follow_the_drow/include/FROG-data/``.
    mode : str
        Which *partition* of the six recordings to use — ``"official"`` (the
        published benchmark, the default), ``"transferred"`` (official's train
        and val, person-free test) or ``"balanced"`` (every split sees crowded,
        sparse and empty scenes). See `_MODES` and the module docstring.
    split : str
        Which part of that partition to read, and so which scans land in
        ``det_id`` / ``det_wp``: ``"train"``, ``"val"`` or ``"test"``.
        Every split is a disjoint set of whole recordings in every mode, so no
        window can bridge two of them.
    time_frame_size : int
        Number of consecutive scans returned by ``get_scan()``.
    verbose : bool
        Print loading progress messages.

    Public attributes (same interface as DROW_Dataset)
    --------------------------------------------------
    scan_id      : ndarray[object]   each element is ndarray[uint32] shape (N_seq,)
    scan_time    : ndarray[object]   each element is ndarray[float64] shape (N_seq,) -- float32 is not enough precision at Unix-epoch magnitude (~1.4e9): consecutive raw frames collapsed to identical values, quantized in ~128s steps
    scans        : ndarray[object]   each element is ndarray[float32] shape (N_seq, 720)
    det_id       : ndarray[object]   each element is ndarray[uint32]  shape (D_seq,)
    det_wc       : ndarray[object]   always empty lists  (FROG has no wheelchair class)
    det_wa       : ndarray[object]   always empty lists  (FROG has no walker class)
    det_wp       : ndarray[object]   each element is list of lists of (r, phi) tuples
    odoms        : ndarray[object]   each element is structured ndarray, dtype (eq,t,xya)
    idet2iscan   : list[dict]        det_index → scan_array_index mapping per sequence
    time_frame   : int               time_frame_size passed at construction
    """

    TIME_FRAME     = 5
    LASER_MEASURES = 720
    LASER_MIN_ANGLE = -np.pi / 2   # rightmost beam, -90°
    LASER_MAX_ANGLE =  np.pi / 2   # approx leftmost beam, +90°
    LASER_INCREMENT =  np.pi / 720  # 0.25° per beam

    # Gap thresholds -- see _split_at_gaps() for what each one separates.
    RECORDING_GAP_S = 300.0
    SESSION_GAP_S   = 0.5

    def __init__(
        self,
        datapath: Union[Path, str] = _DATASET_PATH,
        split: str = "test",
        mode: str = DEFAULT_MODE,
        time_frame_size: int = TIME_FRAME,
        verbose: bool = True,
        auto_download: bool = True,
        session_gap_s: float = SESSION_GAP_S,
        runup_scans: int = None,
        missing_return_m: Optional[float] = LEGACY_MISSING_RETURN_M,
    ):
        Logging.__init__(self, verbose)
        self.time_frame = time_frame_size

        if mode not in _MODES:
            raise ValueError(
                f"Unknown FROG mode {mode!r}. Choose one of {sorted(_MODES)}."
            )
        if split not in _MODES[mode]:
            raise ValueError(
                f"Unknown FROG split {split!r} for mode {mode!r}. "
                f"Choose one of {sorted(_MODES[mode])}."
            )
        self.mode = mode
        self.split = split
        rec_spec, frame_rule = _MODES[mode][split]
        wanted_recordings = _recordings_per_file(rec_spec)

        if runup_scans is None:
            runup = _MODE_RUNUP_OVERRIDE.get((mode, split), _SPLIT_RUNUP[split])
        else:
            runup = runup_scans

        datapath = Path(datapath)
        file_keys = list(wanted_recordings)
        h5_files = [datapath / _H5_FILES[k] for k in file_keys]
        missing = [p for p in h5_files if not p.exists()]
        if missing:
            if auto_download:
                self._print(
                    f"FROG {mode}/{split} needs {[p.name for p in missing]}, "
                    f"not found in {datapath}. Downloading …"
                )
                self.download(datapath, which=file_keys)
                missing = [p for p in h5_files if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"FROG {mode}/{split} requires {[p.name for p in missing]}, "
                    f"missing from {datapath}.\n"
                    f"Run FROG_Dataset.download('{datapath}') to fetch every "
                    f"published file, or set 'datapath' to the directory "
                    f"containing them."
                )

        seq_data = [session for key, f in zip(file_keys, h5_files)
                    for session in self._load_h5(
                        f,
                        recordings_wanted=wanted_recordings[key],
                        frame_rule=frame_rule,
                        session_gap_s=session_gap_s,
                        runup=runup,
                        missing_return_m=missing_return_m)]
        n_seq = len(seq_data)

        # np.array([ndarray, ...], dtype=object) unpacks into inner dimensions
        # when all arrays share a shape.  Fill an empty object array instead.
        def _obj(items):
            out = np.empty(n_seq, dtype=object)
            for i, v in enumerate(items):
                out[i] = v
            return out

        self.scan_id   = _obj([d[0] for d in seq_data])
        self.scan_time = _obj([d[1] for d in seq_data])
        self.scans     = _obj([d[2] for d in seq_data])
        self.det_id    = _obj([d[3] for d in seq_data])
        self.det_wc    = _obj([d[4] for d in seq_data])
        self.det_wa    = _obj([d[5] for d in seq_data])
        self.det_wp    = _obj([d[6] for d in seq_data])
        self.odoms     = _obj([d[7] for d in seq_data])

        # det_id[seq][i] is directly the scan-array index for detection i.
        self.idet2iscan: List[Dict[int, int]] = [
            {i: int(did[i]) for i in range(len(did))}
            for did in self.det_id
        ]

        self._print(
            f"FROG dataset loaded: {n_seq} sequence(s) from {len(h5_files)} file(s), "
            f"mode='{mode}', split='{split}', frames={sum(len(s) for s in self.scans)}, "
            f"scored={sum(len(d) for d in self.det_id)}, time_frame={time_frame_size}"
        )

    # ------------------------------------------------------------------
    # Core interface (identical to DROW_Dataset)
    # ------------------------------------------------------------------

    def get_scan(
        self, sequence_id: int, scan_id: int, time_window: int, dtime: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a (scans, odoms) window of `time_window` frames ending at
        scan_id, spaced `dtime` raw scans apart (dtime=1: consecutive frames,
        the original behaviour).

        dtime > 1 spreads the same number of frames over more real time —
        e.g. dtime=40 on FROG's 40Hz data spans ~4s instead of ~125ms, closer
        to the ~500ms "half gait cycle" the T=5 window was originally sized
        for at DROW's ~10Hz rate (see docs/RESEARCH.md Section 5.4/8.1.1).

        If there is not enough history, indices are clamped to 0, which
        repeats the earliest available scan/odom (same effect the old
        pad-with-first-scan logic had, generalised via fancy indexing).
        """
        idx = np.clip(scan_id - dtime * np.arange(time_window - 1, -1, -1), 0, scan_id)
        return self.scans[sequence_id][idx], self.odoms[sequence_id][idx]

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    @classmethod
    def _load_h5(
        cls, path: Path,
        recordings_wanted: Optional[FrozenSet[int]] = None,
        frame_rule: str = "all",
        session_gap_s: float = SESSION_GAP_S, runup: int = 0,
        missing_return_m: Optional[float] = LEGACY_MISSING_RETURN_M,
    ) -> List[Tuple]:
        """
        Load one HDF5 file and return a list of per-*session* 8-tuples.

        Three separate structures live in one file, and they are not the same
        thing:

        **Recordings** (>``RECORDING_GAP_S``, 300 s apart) are the physical
        sessions the FROG authors recorded. frog_11-36_12-43_train_val.h5 holds
        two, ~24.6 hours apart -- its own name says so. Recordings own their
        odometry file, and the train/val carve runs inside one recording so a
        validation block never straddles a recording boundary.

        **Sessions** (>``session_gap_s``, 0.5 s apart) are the contiguous runs
        of scans within a recording, split at every recording *pause*. There
        are 60 such pauses across the two training recordings -- up to 74.3 s
        long, together 15-20% of each recording's wall time. They matter
        because ``get_scan()`` indexes by scan number, not by time: before
        this, a nominal "0.38 s" T=5/dtime=10 window that happened to straddle
        a pause really spanned 74 seconds of two unrelated scenes, and 2.4% of
        dtime=10 windows did (scaling linearly with dtime, so it biased the
        sweep against exactly the wide windows it was measuring). Sequences are
        the unit ``get_scan()`` clamps to, so a session boundary is a hard wall
        (TODO.md A13).

        **Splits** own whole recordings, and ``recordings_wanted`` is how: a
        recording this split does not own is skipped entirely rather than
        merely unscored, so no window can draw history across a split boundary.
        ``None`` takes every recording in the file. Within a kept recording,
        ``frame_rule`` decides which frames are *scored* (`_MODES` documents the
        four rules); frames it rejects stay in ``scans``/``odoms`` and are only
        left out of ``det_id``, so a window still draws unbroken history
        through them.
        """
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required to load FROG data.  "
                "Install it with:  pip install h5py"
            ) from exc

        with h5py.File(path, "r") as h5:
            scans      = h5["scans"][:].astype(np.float32)       # (N, 720)
            timestamps = h5["timestamps"][:]                       # (N,)
            circles    = h5["circles"][:].astype(np.float32)      # (M, 6)
            circle_idx = h5["circle_idx"][:].astype(np.int64)     # (N,)
            circle_num = h5["circle_num"][:].astype(np.int64)     # (N,)
            split_field = h5["split"][:] if "split" in h5 else None

        scans = encode_missing_returns(scans, missing_return_m)

        N = len(scans)
        wp_per_scan = cls._build_wp_per_scan(circles, circle_idx, circle_num, N)

        recordings = _split_at_gaps(timestamps, cls.RECORDING_GAP_S)

        if frame_rule == "paper_train":
            # Reproduces the paper's own training set exactly. Their
            # `split == 1` half is the interleaved holdout -- useless as a
            # holdout (it scores identically to training data) and not theirs
            # to train on either, so it is simply dropped.
            if split_field is None:
                raise ValueError(
                    f"{path.name} carries no 'split' field, so it cannot supply "
                    f"a 'paper_train' split; only the train_val file can."
                )
            det_mask = split_field == 0
        elif frame_rule == "populated":
            # A protocol match, not a convenience: a no-op on the two official
            # files, which are 100% populated, and on the raw extras it drops
            # the ~35-39% of frames that offer only false positives to a
            # precision-recall metric. Conditioned on being populated the
            # extras average 3.13-3.36 people/frame against test's 3.07.
            det_mask = circle_num > 0
        elif frame_rule == "empty":
            # The complement, and the whole point of `transferred`: every
            # detection here is a false positive by construction, so wp-AUC is
            # undefined and the metric is a false-positive rate (TODO.md A31).
            det_mask = circle_num == 0
        elif frame_rule == "all":
            det_mask = np.ones(N, dtype=bool)
        else:
            raise ValueError(
                f"Unknown FROG frame rule {frame_rule!r}. "
                f"Choose one of ('paper_train', 'populated', 'empty', 'all')."
            )

        # --- odometry, one file per *recording* ------------------------------
        # Not per session: with recording pauses now splitting sessions, one
        # recording contains dozens of them, and they all share the recording's
        # single published odometry file.
        real_odom_names = _session_odom_filenames(path.name)
        use_real_per_recording = len(real_odom_names) == len(recordings)
        shared_odom_path = path.with_name(path.stem + "_odom.npz")

        # --- sessions --------------------------------------------------------
        sessions = []
        for start, end in _split_at_gaps(timestamps, session_gap_s):
            n_s = end - start
            timestamps_s = timestamps[start:end]
            det_mask_s = det_mask[start:end]
            wp_per_scan_s = wp_per_scan[start:end]

            det_id_s = np.where(det_mask_s)[0]
            det_id_s = det_id_s[det_id_s >= runup].astype(np.uint32)
            det_wp_s = [wp_per_scan_s[int(i)] for i in det_id_s]
            det_wc_s = [[] for _ in det_id_s]   # FROG has no wheelchair class
            det_wa_s = [[] for _ in det_id_s]   # FROG has no walker class

            rec_i = next(i for i, (rs, re_) in enumerate(recordings)
                         if rs <= start < re_)
            if recordings_wanted is not None and rec_i not in recordings_wanted:
                continue

            odom_path = shared_odom_path
            if use_real_per_recording:
                candidate = path.with_name(real_odom_names[rec_i])
                if candidate.exists():
                    odom_path = candidate
            odoms_s = cls._load_odom(odom_path, timestamps_s)

            sessions.append((
                np.arange(n_s, dtype=np.uint32),
                timestamps_s.astype(np.float64),
                scans[start:end],
                det_id_s,
                array(det_wc_s, dtype=object),
                array(det_wa_s, dtype=object),
                array(det_wp_s, dtype=object),
                odoms_s,
            ))
        return sessions

    @staticmethod
    def _build_wp_per_scan(
        circles: np.ndarray,
        circle_idx: np.ndarray,
        circle_num: np.ndarray,
        N: int,
    ) -> List[List[Tuple[float, float]]]:
        """
        Convert FROG circle annotations to per-scan lists of (r, phi) pairs.

        FROG annotation columns (circles array):
          0 – X  Cartesian (forward, metres)
          1 – Y  Cartesian (left,    metres)
          2 – bounding-circle radius (metres)
          3 – polar angle   (radians, CCW from X = forward)
          4 – polar distance (metres)
          5 – angular radius (radians)

        We derive (r, phi) from the unambiguous Cartesian columns so that the
        result is directly compatible with ``project_cartesian_from_polar(r, phi)`` used
        throughout the codebase (phi = 0 forward, phi > 0 left).
        """
        wp_per_scan: List[List[Tuple[float, float]]] = []
        for i in range(N):
            count = int(circle_num[i])
            if count == 0:
                wp_per_scan.append([])
                continue
            start = int(circle_idx[i])
            anns  = circles[start: start + count]   # (count, 6)
            x_fwd = anns[:, 0]                       # forward component
            y_lft = anns[:, 1]                       # left component
            r     = np.sqrt(x_fwd ** 2 + y_lft ** 2)
            phi   = np.arctan2(y_lft, x_fwd)        # CCW from forward
            wp_per_scan.append(
                [(float(r[k]), float(phi[k])) for k in range(count)]
            )
        return wp_per_scan

    @staticmethod
    def _load_odom(odom_path: Path, timestamps: np.ndarray) -> np.ndarray:
        """
        Load per-scan odometry from an adjacent .npz file, interpolating it to
        the scan timestamps.

        Expected .npz keys:
          ``ts``   – (M,) UTC timestamps (same units as scan timestamps)
          ``data`` – (M, 3) columns: x, y, heading_angle (radians)

        Raises rather than falling back. This used to be
        ``_load_or_fake_odom()``, which wrapped the whole load in
        ``except Exception: pass`` and substituted all-zero odometry on any
        failure -- a missing file, a corrupt npz, a renamed key -- silently.
        Zero odometry is indistinguishable from a perfectly stationary robot,
        so that fallback turned "ego-motion compensation is broken" into
        "ego-motion compensation looks like a no-op", and no downstream number
        preserves the difference. Every conclusion about ``--align-scans`` and
        about ``dtime`` rests on the odometry being real (TODO.md A14).
        """
        if not odom_path.exists():
            raise FileNotFoundError(
                f"No FROG odometry at {odom_path}. Real per-recording odometry "
                f"is required; run FROG_Dataset.download() to fetch it."
            )
        dtype = np.dtype([
            ("eq",  np.uint32),
            ("t",   np.float64),
            ("xya", np.float32, 3),
        ])
        npz       = np.load(odom_path)
        odom_ts   = npz["ts"].astype(float)
        odom_data = npz["data"].astype(float)  # (M, 3)
        # Heading is interpolated unwrapped and wrapped back, so a turn across +-pi takes the short way.
        heading        = np.interp(timestamps, odom_ts, np.unwrap(odom_data[:, 2]))
        odoms          = np.zeros(len(timestamps), dtype=dtype)
        odoms["t"]     = timestamps.astype(np.float64)
        odoms["xya"]   = np.stack(
            [np.interp(timestamps, odom_ts, odom_data[:, 0]),
             np.interp(timestamps, odom_ts, odom_data[:, 1]),
             np.arctan2(np.sin(heading), np.cos(heading))],
            axis=1).astype(np.float32)
        return odoms

    # ------------------------------------------------------------------
    # Download helper
    # ------------------------------------------------------------------

    @classmethod
    def download(
        cls,
        datapath: Union[Path, str] = _DATASET_PATH,
        which: Union[str, List[str]] = "all",
    ) -> None:
        """
        Download FROG .h5 files, and the odometry that goes with them, from
        the official server.

        Parameters
        ----------
        datapath : Path or str
            Destination directory (created if it does not exist).
        which : str or list of str
            ``"all"`` (the default) fetches every published file; otherwise one
            key from _H5_FILES, or a list of them.

        Defaults to ``"all"`` because the missing test file is exactly how
        every FROG "test" number ended up being measured on the training data:
        the loader globbed whatever happened to be on disk, and only
        frog_11-36_12-43_train_val.h5 ever was (TODO.md A11). Auto-download
        from the constructor still asks for only the split it needs, so a
        training run does not silently pull ~1.4 GB it will not open.

        Odometry is no longer best-effort. ``_load_odom()`` raises without it,
        so a failure here has to be loud rather than quietly degrading every
        downstream ego-motion result (TODO.md A14).

        Example
        -------
        >>> FROG_Dataset.download()                    # everything
        >>> FROG_Dataset.download(which="train_val")   # one file and its odometry
        """
        datapath = Path(datapath)
        datapath.mkdir(parents=True, exist_ok=True)

        if which == "all":
            keys = list(_H5_FILES)
        elif isinstance(which, str):
            keys = [which]
        else:
            keys = list(which)
        unknown = [k for k in keys if k not in _H5_FILES]
        if unknown:
            raise ValueError(
                f"Unknown 'which' value(s): {unknown}.  "
                f"Choose from {list(_H5_FILES.keys()) + ['all']}."
            )

        for fname in [_H5_FILES[k] for k in keys]:
            dest = datapath / fname
            if dest.exists():
                print(f"  Already present: {dest}")
            else:
                _download_file(f"{_BASE_URL}/{fname}", dest)

            # Real per-recording odometry, published separately from the scan
            # data -- see _session_odom_filenames()'s docstring.
            for odom_fname in _session_odom_filenames(fname):
                odom_dest = datapath / odom_fname
                if odom_dest.exists():
                    print(f"  Already present: {odom_dest}")
                    continue
                _download_file(f"{_BASE_URL}/{odom_fname}", odom_dest)
