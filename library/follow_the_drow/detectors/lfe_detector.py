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
_PERSON_RADIUS   = 0.4    # metres — assumed person half-width. Confirmed against
                          # FROG's own annotations: 151,611 of 153,655 circles in
                          # the test recording carry exactly this radius. Kept for
                          # reference; it is deliberately NOT LFEPeaksDetector's
                          # merge radius any more — see that class's docstring.
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
    slope:      float = 0.0,
) -> List[Tuple[float, float, float]]:
    """
    Greedy NMS-style merge of nearby centroids into single detections.

    Matches the FROG paper's post-processing description (Amodeo et al.,
    Sec. 5.2.4): "Centroids that are close together are interpreted as legs
    or part of legs, and merged together into final person detections using
    a NMS-like process." Highest-confidence centroids are kept first; any
    remaining centroid within the radius of an already-kept one is dropped.

    `slope` makes that radius range-dependent: `r(d) = radius + slope * d`,
    floored at zero and evaluated at the *mean* range of the pair being
    compared. `slope = 0.0` is exactly the old single-scalar behaviour and is
    the default, because every published number in this repository was measured
    under it.

    **Why it exists, and why it is not the default** (`memory/rejected-ideas.md`,
    TODO.md A38/A40). The hypothesis was that one scalar fails in both
    directions at once: too large for adjacent people, too small for duplicates,
    because a detector's localisation error grows with range while the
    separation between two people does not. Swept on both detectors and at both
    association distances, it **loses to a constant radius of the same average
    size** -- and the recall it was meant to recover turns out not to be there:
    recall moves only 1.8 pp across radii from 0.20 m to 0.50 m, while
    duplicates fall 18-fold. The radius is a precision lever, not a recall one.
    Kept, defaulting to 0.0, because it costs nothing and the sweep may be worth
    repeating on a detector with different localisation behaviour.

    The mean-range convention is symmetric and, in practice, arbitrary: two
    centroids close enough to merge are necessarily at nearly the same range.
    """
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: -d[0])
    keep, keep_r = [], []
    for det in dets:
        _, x, y = det
        r_self = float(np.sqrt(x * x + y * y))
        if all(
            np.sqrt((x - kx) ** 2 + (y - ky) ** 2)
            > max(0.0, radius + slope * 0.5 * (r_self + kr))
            for (_, kx, ky), kr in zip(keep, keep_r)
        ):
            keep.append(det)
            keep_r.append(r_self)
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
                        single detection, keeping the higher-confidence one.
                        Default **0.30 m**, calibrated against the published AP
                        rather than derived, because the paper describes the
                        step without giving a number.

                        Do NOT set this to the person *diameter* (0.8 m). That
                        rule belongs to LFE-PPN, whose NMS deduplicates whole
                        *person proposals*; this merge does a different job --
                        the paper's words are "centroids that are close
                        together are interpreted as **legs or part of legs**,
                        and merged together into final person detections", so
                        its natural scale is the gap between one person's legs.
                        At 0.8 m it over-merges genuinely distinct people and
                        costs ~8pp.

                        Measured on frog_16-41, every 5th frame, against the
                        paper's AP@0.5 65.6 / AP@0.3 63.2:

                          0.25  63.7 / 62.0    (-1.9 / -1.2)
                          0.30  65.1 / 63.4    (-0.5 / +0.2)  <- default
                          0.35  66.7 / 64.9    (+1.1 / +1.7)
                          0.40  67.7 / 65.7    (+2.1 / +2.5)  <- was the default
                          0.80  59.3 / 57.5    (-6.3 / -5.7)

                        The old 0.40 (`_PERSON_RADIUS`) was the whole of the
                        "our LFE-Peaks overshoots its paper" discrepancy
                        (TODO.md A32d).
    """

    DEFAULT_WEIGHTS = _LFE_PEAKS_PATH

    def __init__(
        self,
        onnx_path:       Path  = _LFE_PEAKS_PATH,
        peak_height:     float = 0.01,
        peak_prominence: float = 0.1,
        peak_width:      int   = 1,
        merge_radius:    float = 0.30,
        merge_radius_slope: float = 0.0,
    ):
        self._session        = _load_onnx(onnx_path)
        self._input_name     = self._session.get_inputs()[0].name
        self._output_name    = self._session.get_outputs()[0].name
        self._peak_height    = peak_height
        self._peak_prominence = peak_prominence
        self._peak_width     = peak_width
        self._merge_radius   = merge_radius
        self._merge_radius_slope = merge_radius_slope

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

        return _merge_nearby(detections, self._merge_radius,
                             self._merge_radius_slope)


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
    score_thresh  : minimum objectness score to keep a proposal (default 0.3).
                    Compared against the graph's own sigmoid output, so it is a
                    probability -- see the note in `detect()`.
    nms_radius    : minimum distance (m) between surviving detections.
                    Default **0.30 m**, calibrated against the published AP at
                    *both* association distances the FROG benchmark reports:
                    69.3 / 63.9 against their 69.2 / 62.5.

                    Not the 0.8 m the paper's own sentence implies ("a given
                    hyperparameter, which usually matches the most common ground
                    truth circle diameter" -- the most common annotated radius is
                    0.400 m). At 0.8 m this scores 65.8 / 58.6, ~3.5pp low on
                    both. `LFEPeaksDetector` independently calibrates to the same
                    0.30 m, and two separately post-processed models agreeing on
                    one radius is better evidence than either alone: the paper's
                    sentence appears to describe design intent rather than the
                    value used.
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
        nms_radius:   float = 0.30,
        nms_radius_slope: float = 0.0,
    ):
        self._session      = _load_onnx(onnx_path)
        self._input_name   = self._session.get_inputs()[0].name
        self._output_name  = self._session.get_outputs()[0].name
        self._score_thresh = score_thresh
        self._nms_radius   = nms_radius
        self._nms_radius_slope = nms_radius_slope

        out_shape   = self._session.get_outputs()[0].shape
        n_anchors   = out_shape[2] if isinstance(out_shape[2], int) else self._N_ANCHORS_FALLBACK
        self._n_anchors     = n_anchors
        self._depth_spacing = (_SCAN_FAR - _SCAN_NEAR) / (n_anchors - 1)

        # Pre-compute anchor depths
        self._anchor_depths = np.linspace(
            _SCAN_NEAR, _SCAN_FAR, self._n_anchors, dtype=np.float32
        )

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

        # Channel 0 is ALREADY a probability: the exported graph ends with
        # `Slice -> Sigmoid -> Concat`, applying the sigmoid to the objectness
        # channel and concatenating the two raw offset channels after it
        # (verified by reading lfe_ppn.onnx's node list, not inferred from the
        # value range). This decoder used to apply a second sigmoid, which
        # squashed every score into [0.5, 0.731] and broke three things at once:
        #
        #  * `score_thresh` became inert -- 0.01 and 0.3 returned bit-identical
        #    output, because all 3,600 anchors cleared any threshold below 0.5;
        #  * all 3,600 proposals entered the greedy NMS every frame instead of
        #    the ~61 that genuinely score above 0.3, leaving 182 "detections"
        #    per frame on scenes holding ~3 people;
        #  * inference cost 224 ms/frame, ~200x LFE-Peaks, which is what made
        #    LFE-PPN look unusable for repeated evaluation (TODO.md A5).
        #
        # AUC survived it better than the rest, since a sigmoid is monotonic and
        # preserves detection ordering -- which is exactly why the bug hid: an
        # earlier measured 67.7% sat plausibly close to the published 66.5%.
        # Any *threshold* read off that run means nothing, and so does any
        # false-positive rate at one (TODO.md A31).
        objectness = out[:, :, 0]                  # (N_sectors, M), already in [0, 1]
        d_offset   = out[:, :, 1]                  # distance offset
        l_offset   = out[:, :, 2]                  # arc offset (normalised)

        # **Both offset channels are normalised by their own anchor cell's
        # extent**, which is the ordinary convention for anchor-based
        # regression and is what this decoder used to get wrong. The paper does
        # not state it, so it was established by scoring the 16 plausible
        # conventions against LFE-PPN's published 66.5% AP on the official test
        # recording (2,525 frames):
        #
        #   depth offset in raw metres     50.5%   <- what this code did
        #   depth offset x anchor spacing  65.8-68.5%, for every angle convention
        #
        # The depth axis separates by ~15pp and does so under all four angular
        # conventions tried, so it is a clean finding rather than a fit: it was
        # a choice between two named conventions, not a tuned constant.
        #
        # **The arc channel was re-identified on 2026-09-10 and this comment
        # block records a mistake worth not repeating.** The first calibration
        # scored candidates at an association distance of 0.5 m, called the
        # angular axis "weakly identified" (67.4% with no arc offset at all
        # against 68.1% with a sector-width one) and picked the sector-width
        # form. That yardstick ranks the candidates *backwards*: a sector is
        # 1.5 deg wide, so the whole correction is under 10 cm at conversational
        # range and a 0.5 m gate cannot see it.
        #
        # At d = 0.3 m -- which the FROG benchmark also reports -- the same
        # candidates span 28.7-58.6%, and the metric form below wins:
        #
        #   l x spacing / depth   (this)      58.6 / 65.8   <- also the original
        #   l x 2 sector widths               56.4 / 66.6
        #   l x 1 sector width                52.5 / 68.1   <- briefly the default
        #   no arc offset at all              28.7 / 67.4
        #
        # So the depth-scaling fix above was a real +15pp, and the angular
        # change made alongside it was a regression that AP@0.5 could not show.
        # **Calibrate anything positional at the tightest association distance
        # available.**
        #
        # Anchor depths stay `linspace(NEAR, FAR, 30)`, spacing 9.8/29 -- which
        # also settles the open question in docs/RESEARCH.md about the paper's
        # `-1` term: the endpoint-inclusive form scores 68.1% against the
        # step form's 67.5%.
        detections = []
        for s in range(n_sectors):
            for m in range(self._n_anchors):
                score = float(objectness[s, m])
                if score < self._score_thresh:
                    continue
                anchor_d   = float(self._anchor_depths[m])
                final_d    = anchor_d + float(d_offset[s, m]) * self._depth_spacing
                # Arc offset: a length in metres, normalised by the anchor
                # spacing, converted to an angle by dividing by the decoded
                # depth. NOT an angle scaled by the sector width -- see below.
                phi_offset = (float(l_offset[s, m]) * self._depth_spacing
                              / max(final_d, 0.2))
                final_phi  = float(sector_angles[s]) + phi_offset

                x = final_d * -np.sin(final_phi)
                y = final_d *  np.cos(final_phi)
                detections.append((score, x, y))

        return _merge_nearby(detections, self._nms_radius,
                             self._nms_radius_slope)
