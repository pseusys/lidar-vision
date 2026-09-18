"""
Pseudo-odometry estimation for datasets with no real odometry.

**Not currently reached for FROG.** The FROG authors do publish real
per-recording odometry (``frog_<HH-MM>_odom.npz``), all six sessions of it are
downloaded by ``FROG_Dataset.download()``, and ``_load_odom()`` now *raises*
rather than falling back if any of it is missing -- a silent zero-odometry
fallback is indistinguishable from a stationary robot and made every
ego-motion conclusion unfalsifiable (see frog_dataset.py, TODO.md A14).
This module remains for a dataset that genuinely ships none, and as the
source of the ``<stem>_odom.npz`` sidecar ``_load_odom()`` reads when no
per-recording file exists.

It estimates odometry via classical 2-D LiDAR scan matching
(trimmed ICP): for each consecutive pair of raw scans, find the rigid
transform that best aligns the *static* structure between them (walls,
furniture, ...), rejecting points that don't fit that transform (typically
people, since they're what's moving) as outliers -- then integrate the
recovered per-frame deltas into a running (x, y, theta) trajectory.

Only short-horizon relative motion (frame-to-frame, ~25ms at FROG's 40Hz) is
needed here, not a globally-consistent trajectory -- there's no loop closure
or drift correction, matching the "short horizon, not SLAM-grade" tractability
argument in docs/IDEAS_BACKLOG.md item 4. Convention: local Cartesian points
use FROG's own (forward, left) axes (frog_laser_angles(): angle 0 = forward,
positive = counterclockwise/left) -- the same axis labels the world-frame
(x, y, theta) trajectory uses, so a scan's own local frame and the world frame
share axis directions and differ only by the accumulated heading theta.
"""
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def local_cartesian_points(scan: np.ndarray, angles: np.ndarray,
                           max_range: float = 9.9) -> np.ndarray:
    """
    Convert one raw range scan to local (forward, left) Cartesian points,
    dropping beams at/beyond max_range (treated as "no return" rather than
    real geometry -- FROG clamps invalid readings to 10.0m, see
    FROG_Dataset._load_h5()).
    """
    valid = scan < max_range
    r = scan[valid]
    phi = angles[valid]
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1).astype(np.float64)


def _kabsch_2d(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed-form optimal 2-D rigid transform (R, t) minimising
    sum ||R @ source_i + t - target_i||^2 for corresponding point pairs
    (Kabsch/orthogonal Procrustes algorithm, restricted to proper rotations).
    """
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_c = source - src_mean
    tgt_c = target - tgt_mean
    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    t = tgt_mean - R @ src_mean
    return R, t


def icp_2d(source: np.ndarray, target: np.ndarray, max_iters: int = 15,
          inlier_thresh: float = 0.15, min_inliers: int = 10,
          init_R: Optional[np.ndarray] = None, init_t: Optional[np.ndarray] = None
          ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Trimmed point-to-point ICP: iteratively find each source point's nearest
    neighbour in target under the current transform estimate, drop pairs
    farther apart than inlier_thresh (rejects points that don't belong to
    static structure -- e.g. a person having moved between frames -- since
    those won't have a good match under the transform that aligns everything
    else), then re-solve the transform (_kabsch_2d) from the surviving pairs.

    Returns (R, t, n_inliers) from the final iteration. If fewer than
    min_inliers pairs survive at any point, returns the best result found so
    far rather than continuing to fit a transform to noise.
    """
    if len(source) < min_inliers or len(target) < min_inliers:
        return np.eye(2), np.zeros(2), 0

    R = init_R if init_R is not None else np.eye(2)
    t = init_t if init_t is not None else np.zeros(2)
    tree = cKDTree(target)
    best = (R, t, 0)

    for _ in range(max_iters):
        transformed = source @ R.T + t
        dist, idx = tree.query(transformed)
        inliers = dist < inlier_thresh
        n_inliers = int(inliers.sum())
        if n_inliers < min_inliers:
            break
        R, t = _kabsch_2d(source[inliers], target[idx[inliers]])
        if n_inliers >= best[2]:
            best = (R, t, n_inliers)

    return best


def icp_rotation_angle(R: np.ndarray) -> float:
    """Rotation angle (radians, CCW+) of a 2x2 proper rotation matrix."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def estimate_pairwise_delta(scan_prev: np.ndarray, scan_curr: np.ndarray,
                            angles: np.ndarray, theta_curr_guess: float = 0.0,
                            **icp_kwargs) -> Tuple[float, float, float, int]:
    """
    Estimate one frame-to-frame ego-motion delta via ICP between two
    consecutive raw scans.

    Derivation (local axes = forward/left, world axes = same labels, related
    by the accumulated heading theta -- see module docstring): for a static
    point with local coordinates p_prev (at the previous pose) and p_curr (at
    the current pose), if the robot's world-frame pose changes from
    (X,Y,theta) to (X+dX, Y+dY, theta+dtheta) between the two scans, then

        p_curr = Rot(-dtheta) @ p_prev + Rot(-(theta+dtheta)) @ (-dX, -dY)

    ICP recovers (R, t) such that p_curr ~= R @ p_prev + t for the static
    (inlier) points, i.e. R = Rot(-dtheta) and t = -Rot(-(theta+dtheta)) @ (dX, dY).
    Solving: dtheta = -angle(R), then (dX, dY) = -Rot(theta+dtheta) @ t, using
    the running heading estimate theta_curr_guess = theta (this frame's own
    heading before applying dtheta -- i.e. this function's own caller has to
    pass the *previous* frame's known theta, matching how a real integrator
    only ever knows the pose *before* the step it's about to take, and this
    is verified end-to-end against a known synthetic trajectory in
    tests/test_odometry_estimation.py, not just derived on paper).

    Returns (dtheta, dX_world, dY_world, n_inliers).
    """
    src = local_cartesian_points(scan_prev, angles)
    tgt = local_cartesian_points(scan_curr, angles)
    R, t, n_inliers = icp_2d(src, tgt, **icp_kwargs)
    dtheta = -icp_rotation_angle(R)
    theta_next = theta_curr_guess + dtheta
    c, s = np.cos(theta_next), np.sin(theta_next)
    rot_theta_next = np.array([[c, -s], [s, c]])
    d_world = -rot_theta_next @ t
    return float(dtheta), float(d_world[0]), float(d_world[1]), n_inliers


def integrate_trajectory(scans: np.ndarray, angles: np.ndarray,
                         timestamps: Optional[np.ndarray] = None,
                         max_dt: float = 0.1,
                         max_step_dist: float = 0.2, max_step_dtheta: float = np.radians(10),
                         **icp_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run estimate_pairwise_delta() over every consecutive scan pair in a
    sequence and dead-reckon-integrate the deltas into a (N, 3) world-frame
    (x, y, theta) trajectory, starting at the origin with theta=0 (an
    arbitrary reference -- only relative motion within a temporal window
    is ever used downstream, by aligned_raw_scan()/SimpleTracker, so a global
    offset/rotation of the whole trajectory has no effect).

    timestamps / max_dt: recorded FROG sessions are not always one
    contiguous 40Hz stream -- found empirically (not assumed) via the actual
    file's own timestamps: 61 small recording pauses (0.1-90s) plus one
    ~24.6-hour gap between two concatenated sessions. estimate_pairwise_delta()
    assumes "consecutive array entries are ~25ms apart", which is false at
    these boundaries -- comparing scans from before/after a multi-second (or
    multi-hour) gap is registering two potentially *unrelated* scenes, not
    estimating real short-horizon motion. This is a stricter, more reliable
    check than max_step_dist/max_step_dtheta below: those only catch a
    result that *looks* implausible, but ICP can return a small-looking,
    confidently-wrong transform when matching unrelated scenes (verified:
    31 of the 62 real timestamp gaps in the FROG data produced a plausible-
    looking small delta and slipped straight past the magnitude clamp,
    including the 24.6-hour session boundary itself). When timestamps is
    given, any consecutive pair more than max_dt apart in real time is
    forced to zero relative motion unconditionally, regardless of what ICP
    returns for it -- not just when the result happens to look wrong.
    timestamps=None (e.g. for the synthetic tests, which have no real
    timestamps) skips this check entirely.

    max_step_dist / max_step_dtheta: sanity clamp on a single frame's
    estimate, independent of the timestamp check above. Found empirically
    necessary: in feature-poor regions (e.g. a large open space where most
    beams exceed the valid range and get filtered out, leaving too few
    points for reliable registration), ICP can return a confidently-wrong
    transform with just-barely-enough inliers to avoid the min_inliers
    fallback -- and because this is dead-reckoning (no loop closure), a
    single such frame permanently offsets every pose after it. A per-frame
    implausible-velocity check (default bounds: ~0.2m or ~10deg between
    consecutive 40Hz-ish frames, well above real per-step motion but far
    below a registration failure's typical jump) is a standard robustness
    measure in scan-matching odometry; a rejected step falls back to zero
    relative motion ("coast") for that one frame rather than accepting the
    bad estimate.

    Returns (trajectory (N,3) float32, n_inliers (N,) int -- n_inliers[0] is
    undefined/0 since there's no previous frame for the first scan; use it to
    flag frames whose estimate is likely unreliable, e.g. n_inliers too low
    or a rejected/clamped/timestamp-gapped step, which reports n_inliers=0
    as well).
    """
    n = len(scans)
    traj = np.zeros((n, 3), dtype=np.float32)
    inlier_counts = np.zeros(n, dtype=np.int32)
    x, y, theta = 0.0, 0.0, 0.0
    for i in range(1, n):
        if timestamps is not None and (timestamps[i] - timestamps[i - 1]) > max_dt:
            dtheta, dx, dy, n_inliers = 0.0, 0.0, 0.0, 0
        else:
            dtheta, dx, dy, n_inliers = estimate_pairwise_delta(
                scans[i - 1], scans[i], angles, theta_curr_guess=theta, **icp_kwargs)
            step_dist = float(np.hypot(dx, dy))
            if step_dist > max_step_dist or abs(dtheta) > max_step_dtheta:
                dtheta, dx, dy, n_inliers = 0.0, 0.0, 0.0, 0
        theta += dtheta
        x += dx
        y += dy
        traj[i] = (x, y, theta)
        inlier_counts[i] = n_inliers
    return traj, inlier_counts
