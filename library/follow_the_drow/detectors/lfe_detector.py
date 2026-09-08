"""
LFE-Peaks and LFE-PPN detectors from the FROG benchmark (ONNX inference).

Both detectors are inference-only wrappers around published ONNX weights from:
  Amodeo, Pérez-Higueras, Merino, Caballero
  "FROG: A New People Detection Dataset for Knee-High 2D Range Finders"
  Frontiers in Robotics and AI, 2025 — arXiv:2306.08531

Weights are downloaded automatically during ``pip install ./library``.
Download URLs:
  LFE-Peaks : https://robotics.upo.es/~famozur/onnx/LFE-Peaks.onnx
  LFE-PPN   : https://robotics.upo.es/~famozur/onnx/LFE-PPN.onnx

Architecture
------------
Both detectors share an LFE backbone: a 1-D U-Net FCN operating directly on
the raw laser scan vector (720 beams for FROG, 450 for DROW).  They differ in
the output head:

  LFE-Peaks  : per-beam sigmoid probability → scipy find_peaks post-processing
  LFE-PPN    : 1-D region proposal network on sector-anchor grid → NMS decoding

Key constraints
---------------
- Single scan only (no temporal history).
- Trained on 720-beam FROG scans; inference on 450-beam DROW scans requires
  zero-padding to 720 beams (accuracy may degrade).
- Both models return detections as class-agnostic (pedestrian) confidence only.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..utils.file_utils import LFE_PEAKS_WEIGHTS_PATH, LFE_PPN_WEIGHTS_PATH

_LFE_PEAKS_PATH = Path(__file__).parent.parent / LFE_PEAKS_WEIGHTS_PATH
_LFE_PPN_PATH   = Path(__file__).parent.parent / LFE_PPN_WEIGHTS_PATH

# Physical constants matching the FROG training configuration
_SCAN_NEAR       = 0.2    # metres — minimum range clipped
_SCAN_FAR        = 10.0   # metres — maximum range clipped
_PERSON_RADIUS   = 0.4    # metres — assumed person half-width
_TRAINED_N_BEAMS = 720    # beam count the ONNX models were trained on


def _normalize_scan(scan: np.ndarray) -> np.ndarray:
    """
    Normalise raw range values to [0, 1].

    Closer obstacles → higher value (1.0 at SCAN_NEAR, 0.0 at SCAN_FAR).
    Missing / max-range readings are mapped to 0.0.
    """
    clipped = np.clip(scan, _SCAN_NEAR, _SCAN_FAR)
    return (1.0 - clipped / _SCAN_FAR).astype(np.float32)


# LFE's trained angular convention: 720 beams, -90 deg to +90 deg, endpoint
# excluded (matches follow_the_drow.datasets.frog_dataset.frog_laser_angles).
# Defined locally rather than imported to avoid a detectors -> datasets
# import for one formula.
_TRAINED_ANGLES = np.linspace(-np.pi / 2, np.pi / 2, _TRAINED_N_BEAMS,
                              endpoint=False, dtype=np.float64)


def _resample_to_trained(scan: np.ndarray, angles: np.ndarray):
    """
    Resample a scan onto LFE's trained 720-beam, -90..+90 deg angular grid.

    Cross-dataset transfer previously zero-padded a shorter scan (e.g.
    DROW's 450 beams) into the first N of 720 slots and left the rest as
    padding. That's wrong on two independent counts, found by inspection
    and confirmed empirically (a padding-*value* fix alone barely moved
    DROW's numbers):

    1. The padded slots carried the wrong "no data" value (see the fixed
       bug in git history / RESEARCH.md) -- a real but secondary issue.
    2. More fundamentally, beam *index* correspondence is not beam *angle*
       correspondence. DROW's 450 beams span +-112.25 deg at ~0.5 deg/beam
       -- a wider FoV at coarser resolution than FROG's 720 beams at
       +-90 deg / ~0.25 deg/beam. Copying DROW's beam i into slot i treats
       "beam index" as if it meant the same real-world angle in both
       datasets, which it doesn't -- e.g. DROW's beam at true angle +45 deg
       lands in a slot the network was trained to associate with roughly
       -11.5 deg. Every learned position-dependent behaviour (the sector
       grid in particular) is then applied to content that's egregiously
       misaligned with what was actually observed at that angle.

    DROW's FoV fully contains FROG's (+-112.25 deg >= +-90 deg), so no
    padding is needed at all for DROW -> FROG-grid transfer: every one of
    the 720 target angles has a real DROW measurement to interpolate from.
    (A dataset with a *narrower* FoV than FROG would still need to pad the
    uncovered edge — none of DROW/FROG/JRDB do.)

    Returns (resampled_scan, resampled_angles) -- resampled_angles is
    always _TRAINED_ANGLES; returned alongside for a uniform call signature
    with the pre-resampling code.
    """
    if len(scan) == _TRAINED_N_BEAMS and np.allclose(angles, _TRAINED_ANGLES, atol=1e-3):
        return scan, angles
    # np.interp requires ascending xp; DROW/FROG/JRDB angle arrays already
    # are (linspace from min to max), but sort defensively rather than
    # assume every future caller's angle convention agrees.
    order = np.argsort(angles)
    resampled = np.interp(_TRAINED_ANGLES, angles[order], scan[order])
    return resampled.astype(scan.dtype), _TRAINED_ANGLES


def _load_onnx(path: Path):
    """Load an ONNX InferenceSession (CPU provider)."""
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required for LFE detectors. "
            "Install it with: pip install onnxruntime"
        ) from e

    if not path.exists():
        raise FileNotFoundError(
            f"LFE ONNX weights not found at {path}.\n"
            "Re-install the library to re-download them:\n"
            "  pip install ./library"
        )
    return ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )


def _merge_nearby(
    detections: List[Tuple[float, float, float]],
    radius:     float,
) -> List[Tuple[float, float, float]]:
    """
    Greedy NMS-style merge of nearby centroids into single detections.

    Matches the FROG paper's post-processing description (Amodeo et al.,
    Sec. 5.2.4): "Centroids that are close together are interpreted as legs
    or part of legs, and merged together into final person detections using
    a NMS-like process." Highest-confidence centroids are kept first; any
    remaining centroid within `radius` of an already-kept one is dropped.
    """
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: -d[0])
    keep = []
    for det in dets:
        _, x, y = det
        if all(
            np.sqrt((x - kx) ** 2 + (y - ky) ** 2) > radius
            for _, kx, ky in keep
        ):
            keep.append(det)
    return keep


# ---------------------------------------------------------------------------
# LFE-Peaks
# ---------------------------------------------------------------------------

class LFEPeaksDetector:
    """
    LFE-Peaks: 1-D U-Net FCN + scipy find_peaks + centroid-merge post-processing.

    Returns class-agnostic person detections from a single raw scan.

    Parameters
    ----------
    onnx_path : path to the LFE-Peaks ONNX file (default: bundled weights)
    peak_height      : minimum peak height threshold (default 0.01)
    peak_prominence  : minimum prominence for find_peaks (default 0.1)
    peak_width       : minimum width in samples (default 1)
    merge_radius     : centroids closer than this (metres) are merged into a
                        single detection, keeping the higher-confidence one
                        (default: `_PERSON_RADIUS` — the paper describes this
                        step but does not give an exact radius)
    """

    DEFAULT_WEIGHTS = _LFE_PEAKS_PATH

    def __init__(
        self,
        onnx_path:       Path  = _LFE_PEAKS_PATH,
        peak_height:     float = 0.01,
        peak_prominence: float = 0.1,
        peak_width:      int   = 1,
        merge_radius:    float = _PERSON_RADIUS,
    ):
        self._session        = _load_onnx(onnx_path)
        self._input_name     = self._session.get_inputs()[0].name
        self._output_name    = self._session.get_outputs()[0].name
        self._peak_height    = peak_height
        self._peak_prominence = peak_prominence
        self._peak_width     = peak_width
        self._merge_radius   = merge_radius

    def detect(
        self,
        scan:   np.ndarray,
        angles: np.ndarray,
    ) -> List[Tuple[float, float, float]]:
        """
        Detect people in a single raw scan.

        Parameters
        ----------
        scan   : (N,) float32  raw range measurements in metres
        angles : (N,) float64  beam angles in radians

        Returns
        -------
        list of (confidence, x, y) in robot-frame Cartesian coordinates
        """
        from scipy.signal import find_peaks

        scan_p, angles_p = _resample_to_trained(scan, angles)
        norm   = _normalize_scan(scan_p)

        # ONNX models expect (batch, steps, channels) — Keras Conv1D layout
        inp = norm.reshape(1, _TRAINED_N_BEAMS, 1)
        out = self._session.run(
            [self._output_name], {self._input_name: inp}
        )[0]                                        # (1, N, 1) or (1, N)
        prob = out.squeeze()                        # (_TRAINED_N_BEAMS,) — every
        # position is real (resampled) data now, no crop needed

        peaks, _ = find_peaks(
            prob,
            height     = self._peak_height,
            prominence = self._peak_prominence,
            width      = self._peak_width,
            rel_height = 0.5,
        )

        detections = []
        for pk in peaks:
            score = float(prob[pk])
            r     = float(scan_p[pk])
            phi   = float(angles_p[pk])
            x     = r * -np.sin(phi)
            y     = r *  np.cos(phi)
            detections.append((score, x, y))

        return _merge_nearby(detections, self._merge_radius)


# ---------------------------------------------------------------------------
# LFE-PPN
# ---------------------------------------------------------------------------

class LFEPPNDetector:
    """
    LFE-PPN: LFE backbone + 1-D polar region proposal network.

    Divides the FoV into N/6 angular sectors with M range anchors each,
    predicts objectness + distance/angle offsets per anchor, then decodes and
    applies NMS.

    Parameters
    ----------
    onnx_path     : path to the LFE-PPN ONNX file (default: bundled weights)
    score_thresh  : minimum objectness score to keep a proposal (default 0.3)
    nms_radius    : minimum distance (m) between surviving detections (default 0.8)
    """

    DEFAULT_WEIGHTS = _LFE_PPN_PATH

    # Anchor parameters matching the FROG training configuration
    _SECTOR_STRIDE       = 6     # one anchor sector every 6 beams
    _N_ANCHORS_FALLBACK  = 31    # design estimate: (FAR-NEAR)/(0.8*RADIUS) — only
                                  # used if the ONNX model's output shape isn't
                                  # static; the bundled published model actually
                                  # has 30, not 31 (confirmed via its output
                                  # shape ['unk','unk', 30, 3]) — trust the model
                                  # over the formula.

    def __init__(
        self,
        onnx_path:    Path  = _LFE_PPN_PATH,
        score_thresh: float = 0.3,
        nms_radius:   float = 0.8,
    ):
        self._session      = _load_onnx(onnx_path)
        self._input_name   = self._session.get_inputs()[0].name
        self._output_name  = self._session.get_outputs()[0].name
        self._score_thresh = score_thresh
        self._nms_radius   = nms_radius

        out_shape   = self._session.get_outputs()[0].shape
        n_anchors   = out_shape[2] if isinstance(out_shape[2], int) else self._N_ANCHORS_FALLBACK
        self._n_anchors     = n_anchors
        self._depth_spacing = (_SCAN_FAR - _SCAN_NEAR) / (n_anchors - 1)

        # Pre-compute anchor depths
        self._anchor_depths = np.linspace(
            _SCAN_NEAR, _SCAN_FAR, self._n_anchors, dtype=np.float32
        )

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def detect(
        self,
        scan:   np.ndarray,
        angles: np.ndarray,
    ) -> List[Tuple[float, float, float]]:
        """
        Detect people in a single raw scan via anchor grid decoding.

        Parameters
        ----------
        scan   : (N,) float32  raw range measurements in metres
        angles : (N,) float64  beam angles in radians

        Returns
        -------
        list of (confidence, x, y) in robot-frame Cartesian coordinates
        """
        scan_p, angles_p = _resample_to_trained(scan, angles)
        norm   = _normalize_scan(scan_p)

        inp = norm.reshape(1, _TRAINED_N_BEAMS, 1)
        out = self._session.run(
            [self._output_name], {self._input_name: inp}
        )[0]                        # (1, N_sectors, M, 3) or (N_sectors, M, 3)

        while out.ndim > 3:
            out = out.squeeze(0)    # (N_sectors, M, 3)

        n_sectors = out.shape[0]

        # Sector centre angles: take the middle beam of each sector. Every
        # index is valid now (angles_p always has _TRAINED_N_BEAMS real
        # entries, resampled — not the original, shorter-for-DROW array),
        # so this no longer needs the defensive n_orig clip that used to
        # collapse every out-of-range sector onto one repeated angle.
        sector_beam_indices = np.arange(n_sectors) * self._SECTOR_STRIDE + self._SECTOR_STRIDE // 2
        sector_beam_indices = np.clip(sector_beam_indices, 0, _TRAINED_N_BEAMS - 1)
        sector_angles = angles_p[sector_beam_indices]

        objectness = self._sigmoid(out[:, :, 0])  # (N_sectors, M)
        d_offset   = out[:, :, 1]                  # distance offset
        l_offset   = out[:, :, 2]                  # arc offset (normalised)

        # Decode all anchors above threshold
        detections = []
        for s in range(n_sectors):
            for m in range(self._n_anchors):
                score = float(objectness[s, m])
                if score < self._score_thresh:
                    continue
                anchor_d   = float(self._anchor_depths[m])
                final_d    = anchor_d + float(d_offset[s, m])
                # Arc offset normalised by depth spacing; convert to angle offset
                phi_offset = (float(l_offset[s, m]) * self._depth_spacing
                              / max(anchor_d, 0.1))
                final_phi  = float(sector_angles[s]) + phi_offset

                x = final_d * -np.sin(final_phi)
                y = final_d *  np.cos(final_phi)
                detections.append((score, x, y))

        return _merge_nearby(detections, self._nms_radius)
