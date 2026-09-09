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

The ``split`` field (present in train/val files) marks scans as training (0) or
validation (1).  Scans that belong to a different split are *not* included in
``det_id`` but ARE kept in ``scans`` / ``odoms`` so that ``get_scan()`` can
return unbroken temporal windows.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
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


# Maps the split= constructor argument to the _H5_FILES key that must be present.
_SPLIT_TO_WHICH: Dict[Optional[str], str] = {
    "test":  "test",
    "train": "train_val",
    "val":   "train_val",
    None:    "test",
}

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
    split : str or None
        Which scans to include in ``det_id`` / ``det_wp``:
          - ``"test"``    – all scans in the test file(s)
          - ``"train"``   – scans labelled split==0 in train/val files
          - ``"val"``     – scans labelled split==1 in train/val files
          - ``None``      – all scans in every .h5 file found
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

    def __init__(
        self,
        datapath: Union[Path, str] = _DATASET_PATH,
        split: Optional[str] = "test",
        time_frame_size: int = TIME_FRAME,
        verbose: bool = True,
        auto_download: bool = True,
    ):
        Logging.__init__(self, verbose)
        self.time_frame = time_frame_size

        datapath = Path(datapath)
        h5_files = sorted(datapath.glob("*.h5"))
        if not h5_files:
            if auto_download:
                which = _SPLIT_TO_WHICH.get(split, "test")
                self._print(
                    f"No FROG .h5 files found in {datapath}. "
                    f"Downloading '{which}' …"
                )
                self.download(datapath, which=which)
                h5_files = sorted(datapath.glob("*.h5"))
            if not h5_files:
                raise FileNotFoundError(
                    f"No FROG .h5 files found in {datapath}.\n"
                    f"Run FROG_Dataset.download('{datapath}') to fetch the data,\n"
                    f"or set 'datapath' to the directory containing your .h5 files."
                )

        seq_data = [session for f in h5_files
                   for session in self._load_h5(f, split=split)]
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
            f"split='{split}', time_frame={time_frame_size}"
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
        cls, path: Path, split: Optional[str], session_gap_s: float = 300.0
    ) -> List[Tuple]:
        """
        Load one HDF5 file and return a list of per-*session* 8-tuples (not
        always exactly one).

        Some FROG files bundle more than one physical recording -- found
        empirically (not assumed) from the real timestamps in
        frog_11-36_12-43_train_val.h5, whose name already says as much: two
        of the six sessions in the paper's own Table 1 (starting 11:36 and
        12:43) are concatenated into one file, with a ~24.6-hour real gap
        between them (plus 61 much smaller sub-90s recording pauses that are
        *not* session boundaries, just brief pauses within one session).
        get_scan()'s temporal window has no way to know about a boundary
        like that -- it will happily pull "history" from before a session
        break into a window for an annotated frame shortly after it, feeding
        the network raw scans that jump between two unrelated recordings
        (different rooms, possibly different days), independent of whatever
        odometry correction is or isn't applied on top. Splitting at gaps
        this large into separate *sequences* (mirroring DROW's own
        multi-sequence structure) fixes this the same way get_scan()'s
        existing clamp-at-sequence-start already protects the very start of
        a recording: a window can never reach past a sequence boundary.

        session_gap_s=300s (5 min) is chosen to cleanly isolate only the one
        genuine ~24.6-hour session boundary actually found in this dataset,
        well above the largest ordinary within-session pause (~90s) --
        those smaller pauses are left as ordinary (if temporally sparse)
        in-sequence frames; see aligned_raw_scan()'s odometry-side handling
        of small gaps (odometry_estimation.integrate_trajectory's max_dt)
        for that separate, narrower concern.
        """
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required to load FROG data.  "
                "Install it with:  pip install h5py"
            ) from exc

        with h5py.File(path, "r") as h5:
            # Core scan data
            scans      = h5["scans"][:].astype(np.float32)       # (N, 720)
            timestamps = h5["timestamps"][:]                       # (N,)
            circles    = h5["circles"][:].astype(np.float32)      # (M, 6)
            circle_idx = h5["circle_idx"][:].astype(np.int64)     # (N,)
            circle_num = h5["circle_num"][:].astype(np.int64)     # (N,)

            # Build the detection mask (which scans count as labelled frames)
            if split in ("train", "val") and "split" in h5:
                split_field = h5["split"][:]
                split_val   = 0 if split == "train" else 1
                det_mask    = split_field == split_val
            else:
                # test files have no 'split' field; treat every scan as a det.
                det_mask = np.ones(len(scans), dtype=bool)

        # Clamp infinite / NaN range readings to a safe max (10 m)
        scans = np.where(np.isfinite(scans), scans, 10.0)

        N = len(scans)

        # Per-scan annotation lists -> converted to (r, phi) for project_cartesian_from_polar compat.
        wp_per_scan = cls._build_wp_per_scan(circles, circle_idx, circle_num, N)

        # Split into sessions at any gap exceeding session_gap_s -- see
        # docstring. bounds are [start, end) index ranges into the full N
        # arrays above; a file with no large gap yields exactly one session
        # spanning the whole file (the common case, unchanged behaviour).
        gap_idx = np.where(np.diff(timestamps) > session_gap_s)[0]
        starts = [0] + [int(i) + 1 for i in gap_idx]
        ends = [int(i) + 1 for i in gap_idx] + [N]
        n_sessions = len(starts)

        # Real per-session odometry (see _session_odom_filenames()) is
        # preferred when present on disk; it's published by the FROG
        # authors per individual recording, matching 1:1 with the sessions
        # just split out above (both derived from the same underlying
        # per-session structure -- verified independently: the real files'
        # own timestamp gap between sessions matches this file's detected
        # scan-timestamp gap to within 3 seconds). Falls back to the
        # existing single shared-file convention (this project's own
        # estimated pseudo-odometry, or fake/zero) per session when the
        # matching real file isn't there, or when the token count parsed
        # from the filename doesn't line up with the sessions actually
        # found (an unexpected filename shape -- safer to fall back for
        # every session than guess a wrong pairing).
        real_odom_names = _session_odom_filenames(path.name)
        use_real_per_session = len(real_odom_names) == n_sessions
        shared_odom_path = path.with_name(path.stem + "_odom.npz")

        sessions = []
        for session_i, (start, end) in enumerate(zip(starts, ends)):
            n_s = end - start
            scan_id_s = np.arange(n_s, dtype=np.uint32)
            timestamps_s = timestamps[start:end]
            scans_s = scans[start:end]
            det_mask_s = det_mask[start:end]
            wp_per_scan_s = wp_per_scan[start:end]

            det_id_s = np.where(det_mask_s)[0].astype(np.uint32)
            det_wp_s = [wp_per_scan_s[int(i)] for i in det_id_s]
            det_wc_s = [[] for _ in det_id_s]   # FROG has no wheelchair class
            det_wa_s = [[] for _ in det_id_s]   # FROG has no walker class

            odom_path = shared_odom_path
            if use_real_per_session:
                candidate = path.with_name(real_odom_names[session_i])
                if candidate.exists():
                    odom_path = candidate
            odoms_s = cls._load_or_fake_odom(odom_path, timestamps_s)

            sessions.append((
                scan_id_s,
                timestamps_s.astype(np.float64),
                scans_s,
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
    def _load_or_fake_odom(
        odom_path: Path, timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Load per-scan odometry from an adjacent .npz file if it exists,
        interpolating to the scan timestamps.  Falls back to zero-motion
        odometry so the class works without odometry data.

        Expected .npz keys:
          ``ts``   – (M,) UTC timestamps (same units as scan timestamps)
          ``data`` – (M, 3) columns: x, y, heading_angle (radians)
        """
        dtype = np.dtype([
            ("eq",  np.uint32),
            ("t",   np.float64),
            ("xya", np.float32, 3),
        ])
        N = len(timestamps)

        if odom_path.exists():
            try:
                npz       = np.load(odom_path)
                odom_ts   = npz["ts"].astype(float)
                odom_data = npz["data"].astype(float)  # (M, 3)
                x  = np.interp(timestamps, odom_ts, odom_data[:, 0])
                y  = np.interp(timestamps, odom_ts, odom_data[:, 1])
                th = np.interp(timestamps, odom_ts, odom_data[:, 2])
                odoms          = np.zeros(N, dtype=dtype)
                odoms["t"]     = timestamps.astype(np.float64)
                odoms["xya"]   = np.stack([x, y, th], axis=1).astype(np.float32)
                return odoms
            except Exception:
                pass   # fall through to zero-motion fallback

        # Zero-motion fallback — the detector still runs, just without
        # motion compensation between time-window frames.
        odoms        = np.zeros(N, dtype=dtype)
        odoms["t"]   = timestamps.astype(np.float64)
        return odoms

    # ------------------------------------------------------------------
    # Download helper
    # ------------------------------------------------------------------

    @classmethod
    def download(
        cls,
        datapath: Union[Path, str] = _DATASET_PATH,
        which: str = "test",
    ) -> None:
        """
        Download FROG .h5 files from the official server.

        Parameters
        ----------
        datapath : Path or str
            Destination directory (created if it does not exist).
        which : str
            One of the keys in _H5_FILES (e.g. ``"test"``, ``"train_val"``)
            or ``"all"`` to fetch every available file.

        Example
        -------
        >>> FROG_Dataset.download()          # download test set to default path
        >>> FROG_Dataset.download(which="all")
        """
        datapath = Path(datapath)
        datapath.mkdir(parents=True, exist_ok=True)

        if which == "all":
            files = list(_H5_FILES.values())
        elif which in _H5_FILES:
            files = [_H5_FILES[which]]
        else:
            raise ValueError(
                f"Unknown 'which' value: {which!r}.  "
                f"Choose from {list(_H5_FILES.keys()) + ['all']}."
            )

        for fname in files:
            dest = datapath / fname
            if dest.exists():
                print(f"  Already present: {dest}")
            else:
                url = f"{_BASE_URL}/{fname}"
                _download_file(url, dest)

            # Real per-session odometry, published separately from the scan
            # data (see _session_odom_filenames()'s docstring) -- best
            # effort: without it, FROG_Dataset still works, just falling
            # back to estimated/zero odometry (see _load_or_fake_odom()).
            for odom_fname in _session_odom_filenames(fname):
                odom_dest = datapath / odom_fname
                if odom_dest.exists():
                    print(f"  Already present: {odom_dest}")
                    continue
                try:
                    _download_file(f"{_BASE_URL}/{odom_fname}", odom_dest)
                except Exception as exc:
                    print(f"  [WARN] Could not download {odom_fname}: {exc}\n"
                          f"        FROG_Dataset will fall back to estimated/"
                          f"zero odometry for the affected session(s).")
