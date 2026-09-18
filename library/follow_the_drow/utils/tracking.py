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
