"""
Minimal SORT-style tracker for late-fusion temporal consensus over a
detector's own per-frame (score, x, y) outputs.

Motivation and design rationale: see docs/RESEARCH.md Section 8's consensus
subsection. In short — an earlier version of this idea (_temporal_consensus_filter
in utils/evaluate.py) confirmed detections by nearest-neighbor matching
against a rolling cache of *raw* historical positions, then averaged their
scores. That has two structural problems:

1. A moving person is not "near the average of where they were at T, T-1,
   T-2..." — they're wherever they are *now*. Matching against stale raw
   positions with a fixed radius either misses fast-moving people (radius
   too small) or spuriously merges distinct nearby people (radius too big).
2. It can only suppress false positives (require agreement before reporting)
   — it has no mechanism to recover false negatives (a real person the
   detector momentarily missed).

This module fixes both by tracking a constant-velocity state per person
(position + velocity, exponentially-smoothed — a light "alpha-beta filter",
not a full Kalman filter with covariance propagation, to keep this cheap and
easy to reason about for a pilot) and using *predicted* position for data
association each frame, not raw historical position. A track only needs to
be re-detected within a gate of its prediction to keep updating, which
handles motion; and a confirmed track that goes briefly undetected keeps
reporting its predicted (coasted) position for a few frames instead of
disappearing, which recovers transient false negatives.

Ego-motion compensation (optional, `ego_dtheta` on `step()`): a moving or
rotating robot makes stationary objects appear to move too, which a tracker
with no notion of the robot's own motion can't tell apart from real person
motion. FROG ships no odometry at all, so it's used there under an explicit
"assume the sensor is static" caveat (`ego_dtheta` simply omitted). DROW
carries real per-frame odometry, so predicted track positions there are
counter-rotated by the robot's own heading change before matching — the
same rotation-only approximation (no translation correction) already used
throughout this codebase for temporal alignment (`aligned_raw_scan()`,
`cutout()`'s odometry correction), for consistency rather than inventing a
second convention. See docs/IDEAS_BACKLOG.md item 3.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class _Track:
    __slots__ = ("id", "x", "y", "vx", "vy", "score", "hits",
                "age", "time_since_update")

    def __init__(self, x: float, y: float, score: float, track_id: int):
        self.id = track_id
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0
        self.score = score
        self.hits = 1                  # total number of frames matched to a real detection
        self.age = 0                   # frames since creation
        self.time_since_update = 0     # consecutive frames since the last real match (0 = matched this frame)

    def predict(self, dt: float = 1.0, ego_dtheta: float = 0.0,
               ego_dxy: Tuple[float, float] = (0.0, 0.0)) -> None:
        """
        ego_dtheta : robot heading change since the last predict(), radians.
        ego_dxy    : robot's own translation since the last predict(), already
                     rotated into the *current* frame and expressed in this
                     module's (x, y) convention (see SimpleTracker.step()'s
                     docstring for the exact formula — do not pass raw
                     odometry x/y here, the axis convention differs).
        """
        if ego_dtheta != 0.0:
            # Counter-rotate the track's stored position AND velocity vector
            # by the robot's own heading change, bringing an observation
            # made in the *old* robot frame into the *current* one before
            # extrapolating the person's own motion. Matches this codebase's
            # existing rotation-only (x=-r*sin(phi), y=r*cos(phi)) convention.
            c, s = np.cos(-ego_dtheta), np.sin(-ego_dtheta)
            x, y = self.x, self.y
            self.x, self.y = c * x - s * y, s * x + c * y
            vx, vy = self.vx, self.vy
            self.vx, self.vy = c * vx - s * vy, s * vx + c * vy
        # The robot moving forward makes a stationary point appear to shift
        # backward by the same amount in the robot's own frame — subtract,
        # not add. Only position needs this (translation is a one-time
        # offset, not a per-step rotation of the velocity vector itself, so
        # velocity is untouched here — see the derivation this was checked
        # against in the tracker's git history / RESEARCH.md).
        dxt, dyt = ego_dxy
        self.x -= dxt
        self.y -= dyt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.age += 1
        self.time_since_update += 1

    def update(self, x: float, y: float, score: float,
              dt: float = 1.0, vel_alpha: float = 0.5) -> None:
        """Fold in a real detection matched to this track this frame."""
        if dt > 0:
            obs_vx, obs_vy = (x - self.x) / dt, (y - self.y) / dt
            self.vx = vel_alpha * obs_vx + (1.0 - vel_alpha) * self.vx
            self.vy = vel_alpha * obs_vy + (1.0 - vel_alpha) * self.vy
        self.x, self.y = x, y
        self.score = score
        self.hits += 1
        self.time_since_update = 0


def odom_xya_delta_to_tracker_frame(dx_world: float, dy_world: float,
                                    theta_new: float) -> Tuple[float, float]:
    """
    Convert a raw world-frame odometry displacement (dx_world, dy_world) —
    e.g. `odoms[t]["xya"][:2] - odoms[t-1]["xya"][:2]` — into the robot's
    *own* translation this step, expressed in this module's detection-frame
    (x, y) convention (x=-r*sin(phi), y=r*cos(phi), phi=0 forward,
    phi>0 left — same as everywhere else in this codebase).

    Verified empirically against real DROW odometry (not assumed): the raw
    "xya" field is in the standard robotics world-frame convention (x=
    forward at theta=0, y=left at theta=0, theta CCW-positive) — rotating
    (dx_world, dy_world) by -theta gives a signal that's ~97% positive and
    ~100x larger on the forward axis than the lateral one, exactly the
    "robot mostly drives forward" signature a wheeled/diff-drive base
    should produce. That "forward" axis is *not* this module's y axis by
    label — it has to be remapped: odometry's local-forward -> this
    module's y, odometry's local-left -> this module's -x.

    Pass the result as `ego_dxy` to SimpleTracker.step() / _Track.predict().
    """
    local_fwd  =  np.cos(theta_new) * dx_world + np.sin(theta_new) * dy_world
    local_left = -np.sin(theta_new) * dx_world + np.cos(theta_new) * dy_world
    return -local_left, local_fwd


class SimpleTracker:
    """
    Per-sequence SORT-style tracker — predict, gate+match via Hungarian
    assignment, update, confirm, coast, drop.

    Call `step(detections)` once per frame with that frame's raw
    (score, x, y) detections; get back the list of (score, x, y) for
    confirmed tracks (which may include a coasted position for a track
    that wasn't freshly matched this frame — that's the false-negative
    recovery mechanism).

    Parameters
    ----------
    match_radius : metres — gate on the Hungarian assignment; a detection
                   further than this from a track's *predicted* position
                   cannot be matched to it.
    min_hits     : a track must accumulate this many real matches before it's
                   reported at all — the false-positive suppression knob
                   (higher = stricter, more latency before first report).
    max_age      : a track survives this many consecutive missed frames
                   (coasting on prediction) before being dropped — the
                   false-negative recovery window (higher = more tolerant of
                   gaps, but risks coasting a track that's actually gone).
    vel_alpha    : smoothing factor for the velocity estimate, in (0, 1] —
                   1.0 tracks the latest observed displacement exactly (noisy);
                   lower values smooth more but react to real speed changes slower.
    coast_decay  : per-miss score decay applied when reporting a coasted
                   (not freshly matched) track — reflects growing uncertainty
                   the longer a track goes without confirmation. 1.0 = no decay.
    dt           : time step per `step()` call, in whatever units velocity
                   should be expressed in — 1.0 (frames) is fine as long as
                   frame spacing is roughly constant, which holds here.
    """

    def __init__(self, match_radius: float = 0.5, min_hits: int = 3,
                max_age: int = 3, vel_alpha: float = 0.5,
                coast_decay: float = 0.8, dt: float = 1.0):
        self.match_radius = match_radius
        self.min_hits = min_hits
        self.max_age = max_age
        self.vel_alpha = vel_alpha
        self.coast_decay = coast_decay
        self.dt = dt
        self.tracks: List[_Track] = []
        self._next_id = 0

    def step(self, detections: List[Tuple[float, float, float]],
            ego_dtheta: float = 0.0,
            ego_dxy: Tuple[float, float] = (0.0, 0.0)
            ) -> List[Tuple[float, float, float]]:
        """
        ego_dtheta : robot heading change (radians) since the previous
                     `step()` call, e.g. odoms[t]["xya"][2] - odoms[t-1]["xya"][2].
                     0.0 (default) assumes a static sensor -- which FROG is
                     NOT: 1.88 cm and up to 0.4 deg per frame on the test
                     recording. The claim that FROG odometry is all-zero, which
                     this docstring used to make, described `_load_or_fake_odom()`
                     substituting zeros on failure, not the robot (TODO.md A14).
        ego_dxy    : robot's own translation since the previous `step()`,
                     already converted to this module's (x, y) convention —
                     see odom_xya_delta_to_tracker_frame(). (0.0, 0.0)
                     default assumes a static sensor.
        """
        # Vote-decoded detections can occasionally carry a NaN score/position
        # (degenerate vote cluster) — drop them before they reach cdist/
        # linear_sum_assignment, which otherwise fail ("matrix contains
        # invalid numeric entries") since a NaN cost isn't excluded by
        # `cost > match_radius` (NaN comparisons are always False in NumPy,
        # so the gate silently fails open instead of rejecting the match).
        detections = [d for d in detections if all(np.isfinite(v) for v in d)]

        for t in self.tracks:
            t.predict(self.dt, ego_dtheta, ego_dxy)

        matched_dets = set()
        if self.tracks and detections:
            track_pos = np.array([[t.x, t.y] for t in self.tracks])
            det_pos   = np.array([[x, y] for _, x, y in detections])
            cost = cdist(track_pos, det_pos)
            gated = np.where(cost > self.match_radius, 1e6, cost)
            rows, cols = linear_sum_assignment(gated)
            for r, c in zip(rows, cols):
                if gated[r, c] < 1e6:
                    score, x, y = detections[c]
                    self.tracks[r].update(x, y, score, self.dt, self.vel_alpha)
                    matched_dets.add(c)

        for i, (score, x, y) in enumerate(detections):
            if i not in matched_dets:
                self.tracks.append(_Track(x, y, score, self._next_id))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        out = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                score = t.score * (self.coast_decay ** t.time_since_update)
                out.append((score, t.x, t.y))
        return out


# ---------------------------------------------------------------------------
# SORT, faithfully — for citation
# ---------------------------------------------------------------------------

class _KalmanPoint:
    """Constant-velocity Kalman filter over a 2-D point, state [x, y, vx, vy].

    The point-detection analogue of SORT's 7-state box filter, with its noise
    scaling carried across term by term: SORT gives the unobservable initial
    velocities high variance (`P[4:,4:] *= 1000`), inflates the whole covariance
    (`P *= 10`) and damps the velocity process noise (`Q[4:,4:] *= 0.01`). The
    box-specific terms it also scales (`R[2:,2:] *= 10`, on the scale and aspect
    *measurements*) have no counterpart here, because a point has neither.

    Written out rather than pulled from `filterpy` (which SORT uses) to avoid a
    dependency for twenty lines of linear algebra.
    """

    __slots__ = ("x", "P", "F", "H", "Q", "R")

    def __init__(self, x0: float, y0: float):
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=np.float64)
        self.F = np.array([[1.0, 0, 1.0, 0],
                           [0, 1.0, 0, 1.0],
                           [0, 0, 1.0, 0],
                           [0, 0, 0, 1.0]])
        self.H = np.array([[1.0, 0, 0, 0],
                           [0, 1.0, 0, 0]])
        self.P = np.eye(4)
        self.P[2:, 2:] *= 1000.0
        self.P *= 10.0
        self.Q = np.eye(4)
        self.Q[2:, 2:] *= 0.01
        self.R = np.eye(2)

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z_x: float, z_y: float) -> None:
        z = np.array([z_x, z_y], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def apply_ego(self, ego_dtheta: float, ego_dxy: Tuple[float, float]) -> None:
        """Bring the state from the previous robot frame into the current one.

        SORT has no equivalent — it assumes a static camera. This is a necessary
        addition rather than a deviation: FROG's robot translates 1.88 cm and
        rotates up to 0.4 deg per frame, and 0.4 deg/frame over a track's
        lifetime is a third of a metre at 8 m range.
        """
        if ego_dtheta != 0.0:
            c, s = np.cos(-ego_dtheta), np.sin(-ego_dtheta)
            rot = np.array([[c, -s], [s, c]])
            self.x[:2] = rot @ self.x[:2]
            self.x[2:] = rot @ self.x[2:]
        self.x[0] -= ego_dxy[0]
        self.x[1] -= ego_dxy[1]


class _SortTrack:
    __slots__ = ("kf", "id", "score", "hits", "hit_streak", "age", "time_since_update")

    def __init__(self, x: float, y: float, score: float, track_id: int):
        self.kf = _KalmanPoint(x, y)
        self.id = track_id
        self.score = score
        # SORT starts both counters at zero: the creating detection is not a
        # "hit", so a track needs `min_hits` *further* consecutive matches
        # before it is reported.
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 0

    def predict(self, ego_dtheta: float = 0.0,
                ego_dxy: Tuple[float, float] = (0.0, 0.0)) -> None:
        self.kf.apply_ego(ego_dtheta, ego_dxy)
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

    def update(self, x: float, y: float, score: float) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = score
        self.kf.update(x, y)

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.kf.x[0]), float(self.kf.x[1])


class SortTracker:
    """SORT (Bewley et al., *ICIP* 2016), for point detections.

    Follows the reference implementation's track management exactly — the part
    that decides what gets reported, and therefore the part a false-positive
    count depends on:

    * a track is created from an unmatched detection with `hits = hit_streak = 0`;
    * `predict()` zeroes `hit_streak` whenever the previous frame was a miss,
      so confirmation needs `min_hits` **consecutive** matches, not `min_hits`
      matches spread over any number of frames;
    * a track is reported only when `time_since_update < 1` — i.e. it was
      matched *this* frame. **Coasted tracks are not emitted.**
    * the `frame_count <= min_hits` grace period at the start of a sequence is
      preserved;
    * a track is dropped once `time_since_update > max_age`.

    Two deviations, both forced by the setting rather than chosen:

    1. **Association on Euclidean distance, not IoU.** These detections are
       points on the ground plane; there are no boxes. Synthesising equal-radius
       circles would not change anything, since circle IoU is a monotone
       function of centre distance — it is distance gating with a relabelled
       threshold.
    2. **Ego-motion compensation** (`step()`'s `ego_dtheta` / `ego_dxy`). SORT
       assumes a static camera; this robot moves.

    `SimpleTracker` in this module is a *different* algorithm — fixed-gain
    velocity, confirmation on total rather than consecutive hits, and coasted
    tracks reported for false-negative recovery. Useful for deployment, but not
    SORT, and not what a paper should cite as SORT.
    """

    def __init__(self, match_radius: float = 0.5, min_hits: int = 3,
                 max_age: int = 1):
        self.match_radius = match_radius
        self.min_hits = min_hits
        self.max_age = max_age
        self.tracks: List[_SortTrack] = []
        self._next_id = 0
        self.frame_count = 0

    def step(self, detections: List[Tuple[float, float, float]],
             ego_dtheta: float = 0.0,
             ego_dxy: Tuple[float, float] = (0.0, 0.0)
             ) -> List[Tuple[float, float, float]]:
        self.frame_count += 1
        detections = [d for d in detections if all(np.isfinite(v) for v in d)]

        for t in self.tracks:
            t.predict(ego_dtheta, ego_dxy)

        matched_dets = set()
        if self.tracks and detections:
            track_pos = np.array([t.position for t in self.tracks])
            det_pos = np.array([[x, y] for _, x, y in detections])
            cost = cdist(track_pos, det_pos)
            gated = np.where(cost > self.match_radius, 1e6, cost)
            rows, cols = linear_sum_assignment(gated)
            for r, c in zip(rows, cols):
                if gated[r, c] < 1e6:
                    score, x, y = detections[c]
                    self.tracks[r].update(x, y, score)
                    matched_dets.add(c)

        for i, (score, x, y) in enumerate(detections):
            if i not in matched_dets:
                self.tracks.append(_SortTrack(x, y, score, self._next_id))
                self._next_id += 1

        out = []
        for t in self.tracks:
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits
                                            or self.frame_count <= self.min_hits):
                x, y = t.position
                out.append((t.score, x, y))
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.max_age]
        return out


# ---------------------------------------------------------------------------
# ByteTrack with CBMOT's track confidence -- the strongest post-hoc baseline
# ---------------------------------------------------------------------------

CBMOT_RULES = ("addition", "maximum", "multiplication", "parallel_addition")


def cbmot_update(track_score: float, det_score: float, rule: str = "multiplication") -> float:
    """A matched track's new confidence from its own and the detection's (Benbarka, Schroeder & Zell, *IROS* 2021, eq. 4-7).

    `addition` is the rule CBMOT argues against, as over-confident; the other three never fall below either input (its eq. 3).
    """
    c, s = track_score, det_score
    if rule == "addition":
        return c + s
    if rule == "maximum":
        return max(c, s)
    if rule == "multiplication":
        return 1.0 - (1.0 - c) * (1.0 - s)
    if rule == "parallel_addition":
        denominator = (1.0 - c) + (1.0 - s)
        return 1.0 if denominator <= 0.0 else 1.0 - (1.0 - c) * (1.0 - s) / denominator
    raise ValueError(f"unknown CBMOT rule {rule!r}; expected one of {CBMOT_RULES}")


class _ByteTrack(_SortTrack):
    __slots__ = ("track_score", "activated")

    def __init__(self, x: float, y: float, score: float, track_id: int, activated: bool):
        super().__init__(x, y, score, track_id)
        self.track_score = score
        self.activated = activated


def _gated_matches(tracks: List[_ByteTrack], track_idx: List[int], det_pos: np.ndarray, det_idx: List[int],
                   radius: float) -> List[Tuple[int, int]]:
    """Hungarian assignment between the listed tracks and detections, pairs further apart than `radius` refused, exactly as
    `SortTracker` gates it: matched `(track index, detection index)` pairs."""
    if not track_idx or not det_idx:
        return []
    cost = cdist(np.array([tracks[t].position for t in track_idx]), det_pos[det_idx])
    gated = np.where(cost > radius, 1e6, cost)
    rows, cols = linear_sum_assignment(gated)
    return [(track_idx[r], det_idx[c]) for r, c in zip(rows, cols) if gated[r, c] < 1e6]


class ByteTracker:
    """ByteTrack (Zhang et al., *ECCV* 2022) for point detections, carrying CBMOT's track confidence (`TODO.md` A56 E14).

    The association is ByteTrack's (`two_stage=True`): detections at or above `high_thresh` are matched against every confirmed
    track, tracked or lost; those between `low_thresh` and `high_thresh` against the confirmed tracks that were tracked last frame
    and are still unmatched; unconfirmed tracks against the high detections left over. Only an unmatched high detection at or above
    `new_track_thresh` starts a track, unconfirmed unless it is the first frame; a track is confirmed once `hit_streak >= min_hits`
    (ByteTrack's own rule is `min_hits = 1`), an unconfirmed track that misses is dropped, and a confirmed one is lost until it is
    matched again or `max_age` frames pass. Kalman filter, ego-motion and gating are `SortTracker`'s, Euclidean on points.

    `two_stage=False` is one association of every detection at or above `low_thresh` against every track, with births and
    reporting as in SORT; with both thresholds at zero it reproduces `SortTracker` frame for frame, which a test pins.

    **Scores.** Every track keeps CBMOT's confidence: it starts at the creating detection's score, falls by `score_decay_per_s`
    for every second of every frame, and on a match becomes `cbmot_update(confidence, detection score, score_rule)`; a track under
    `delete_below` is deleted (CBMOT's confidence-based termination). `score_mode` chooses what a reported track carries: the
    matched detection's score (`"detection"`, ByteTrack's own output) or the confidence (`"track"`).

    `step()` returns what the tracker reports -- confirmed tracks matched this frame, plus confirmed tracks lost for at most
    `coast_frames` frames -- at their filtered positions. `rescored` then holds every input detection at its own position, scored by
    its track when that track is reported and by `unmatched_scale` times its own score otherwise, plus the same coasting reports:
    the form in which a tracker *rescores* candidates for an AP comparison rather than filtering them.
    """

    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.1, new_track_thresh: Optional[float] = None,
                 match_radius: float = 0.5, low_match_radius: Optional[float] = None, max_age: int = 30, min_hits: int = 1,
                 two_stage: bool = True, score_mode: str = "detection", score_rule: str = "multiplication",
                 score_decay_per_s: float = 0.0, delete_below: Optional[float] = None, coast_frames: int = 0,
                 unmatched_scale: float = 1.0):
        if score_mode not in ("detection", "track"):
            raise ValueError(f"score_mode must be 'detection' or 'track', not {score_mode!r}")
        cbmot_update(0.5, 0.5, score_rule)
        self.high_thresh, self.low_thresh = high_thresh, low_thresh
        self.new_track_thresh = high_thresh if new_track_thresh is None else new_track_thresh
        self.match_radius = match_radius
        self.low_match_radius = match_radius if low_match_radius is None else low_match_radius
        self.max_age, self.min_hits, self.two_stage = max_age, min_hits, two_stage
        self.score_mode, self.score_rule = score_mode, score_rule
        self.score_decay_per_s, self.delete_below = score_decay_per_s, delete_below
        self.coast_frames, self.unmatched_scale = coast_frames, unmatched_scale
        self.tracks: List[_ByteTrack] = []
        self.rescored: List[Tuple[float, float, float]] = []
        self._next_id = 0
        self.frame_count = 0

    def _associate(self, scores: List[float], det_pos: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Matched `(track, detection)` pairs, and the detections that start tracks."""
        if not self.two_stage:
            use = [i for i, s in enumerate(scores) if s >= self.low_thresh]
            matches = _gated_matches(self.tracks, list(range(len(self.tracks))), det_pos, use, self.match_radius)
            claimed = {d for _, d in matches}
            return matches, [d for d in use if d not in claimed and scores[d] >= self.new_track_thresh]
        high = [i for i, s in enumerate(scores) if s >= self.high_thresh]
        low = [i for i, s in enumerate(scores) if self.low_thresh <= s < self.high_thresh]
        confirmed = [i for i, t in enumerate(self.tracks) if t.activated]
        unconfirmed = [i for i, t in enumerate(self.tracks) if not t.activated]
        first = _gated_matches(self.tracks, confirmed, det_pos, high, self.match_radius)
        taken = {t for t, _ in first}
        tracked_left = [i for i in confirmed if i not in taken and self.tracks[i].time_since_update == 1]
        second = _gated_matches(self.tracks, tracked_left, det_pos, low, self.low_match_radius)
        claimed = {d for _, d in first + second}
        high_left = [d for d in high if d not in claimed]
        third = _gated_matches(self.tracks, unconfirmed, det_pos, high_left, self.match_radius)
        claimed |= {d for _, d in third}
        return first + second + third, [d for d in high_left if d not in claimed and scores[d] >= self.new_track_thresh]

    def _reported(self, t: _ByteTrack) -> bool:
        if t.time_since_update != 0:
            return False
        return t.activated if self.two_stage else (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)

    def _coasting(self, t: _ByteTrack) -> bool:
        if not 1 <= t.time_since_update <= self.coast_frames:
            return False
        return t.activated if self.two_stage else t.hits >= self.min_hits

    def _score(self, t: _ByteTrack) -> float:
        return t.track_score if self.score_mode == "track" else t.score

    def step(self, detections: List[Tuple[float, float, float]], ego_dtheta: float = 0.0,
             ego_dxy: Tuple[float, float] = (0.0, 0.0), dt: float = 0.025) -> List[Tuple[float, float, float]]:
        self.frame_count += 1
        detections = [d for d in detections if all(np.isfinite(v) for v in d)]
        for t in self.tracks:
            t.predict(ego_dtheta, ego_dxy)
            t.track_score -= self.score_decay_per_s * dt

        scores = [s for s, _, _ in detections]
        det_pos = np.array([[x, y] for _, x, y in detections], dtype=np.float64).reshape(-1, 2)
        matches, born = self._associate(scores, det_pos)
        track_of = {}
        for ti, di in matches:
            t = self.tracks[ti]
            score, x, y = detections[di]
            t.update(x, y, score)
            t.track_score = cbmot_update(t.track_score, score, self.score_rule)
            t.activated = t.activated or t.hit_streak >= self.min_hits
            track_of[di] = t
        for di in born:
            score, x, y = detections[di]
            t = _ByteTrack(x, y, score, self._next_id, activated=self.frame_count == 1)
            self._next_id += 1
            self.tracks.append(t)
            track_of[di] = t
        if self.delete_below is not None:
            self.tracks = [t for t in self.tracks if t.track_score >= self.delete_below]
            track_of = {d: t for d, t in track_of.items() if t.track_score >= self.delete_below}

        out = [(self._score(t), *t.position) for t in self.tracks if self._reported(t)]
        coasted = [(self._score(t), *t.position) for t in self.tracks if self._coasting(t)]
        self.rescored = [(self._score(track_of[i]), x, y) if i in track_of and self._reported(track_of[i]) else (s * self.unmatched_scale, x, y)
                         for i, (s, x, y) in enumerate(detections)] + coasted
        if self.two_stage:
            self.tracks = [t for t in self.tracks if (t.activated or t.time_since_update == 0) and t.time_since_update <= self.max_age]
        else:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return out + coasted
