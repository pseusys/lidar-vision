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
                     0.0 (default) assumes a static sensor, e.g. FROG.
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
