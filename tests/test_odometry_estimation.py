"""
Unit tests for library/follow_the_drow/utils/odometry_estimation.py.

The end-to-end trajectory test is the important one here, in the same spirit
as tests/test_geometry.py: this module's derivation (relating an ICP-recovered
local transform to a world-frame odometry delta) was worked out on paper, and
this session has repeatedly found sign errors in exactly that kind of
derivation once actually run. Nothing here is trusted until a known synthetic
ground-truth trajectory comes back out correctly.
"""
import numpy as np

from follow_the_drow.datasets.frog_dataset import frog_laser_angles
from follow_the_drow.utils.odometry_estimation import (
    _kabsch_2d, icp_2d, icp_rotation_angle, estimate_pairwise_delta,
    integrate_trajectory, local_cartesian_points,
)

ANGLES = frog_laser_angles(720)
FAR = 10.0


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _synthesize_scan(pose_xytheta, world_points, angles=ANGLES, n_beams=720):
    """Render a synthetic FROG-style scan of `world_points` as seen from a
    given (x, y, theta) robot pose -- bins each point to its nearest beam,
    keeping the closest point per beam; everything else stays at FAR."""
    x, y, theta = pose_xytheta
    local = (world_points - np.array([x, y])) @ _rot(-theta).T  # world -> local
    r = np.linalg.norm(local, axis=1)
    phi = np.arctan2(local[:, 1], local[:, 0])
    scan = np.full(n_beams, FAR, dtype=np.float32)
    for ri, phii in zip(r, phi):
        if not (angles[0] <= phii <= angles[-1]):
            continue
        beam = int(np.argmin(np.abs(angles - phii)))
        if ri < scan[beam]:
            scan[beam] = ri
    return scan


class TestKabsch2D:
    def test_recovers_known_rotation_and_translation_exactly(self):
        rng = np.random.RandomState(0)
        source = rng.uniform(-5, 5, size=(20, 2))
        theta, t_true = 0.3, np.array([1.5, -0.7])
        target = source @ _rot(theta).T + t_true
        R, t = _kabsch_2d(source, target)
        assert abs(icp_rotation_angle(R) - theta) < 1e-6
        np.testing.assert_allclose(t, t_true, atol=1e-6)


class TestIcp2D:
    # ICP with an identity initial guess only converges when the true motion
    # is small relative to the point spacing -- exactly FROG's real regime
    # (consecutive raw frames are 25ms apart at 40Hz, so per-step motion is
    # tiny), so tests use realistic per-step magnitudes rather than arbitrary
    # large ones (which naive nearest-neighbour correspondence can't be
    # expected to resolve, on a scattered point cloud with no fixed structure).
    def test_recovers_transform_without_known_correspondences(self):
        rng = np.random.RandomState(1)
        source = rng.uniform(-5, 5, size=(100, 2))
        theta, t_true = -0.015, np.array([0.03, 0.02])
        target = source @ _rot(theta).T + t_true
        R, t, n_inliers = icp_2d(source, target, inlier_thresh=0.15)
        assert n_inliers >= 95
        assert abs(icp_rotation_angle(R) - theta) < 1e-3
        np.testing.assert_allclose(t, t_true, atol=1e-3)

    def test_robust_to_a_minority_of_moving_outlier_points(self):
        rng = np.random.RandomState(2)
        static_source = rng.uniform(-5, 5, size=(100, 2))
        theta, t_true = 0.012, np.array([0.04, -0.01])
        static_target = static_source @ _rot(theta).T + t_true
        # add outliers: points that do NOT follow the rigid transform
        outlier_source = rng.uniform(-5, 5, size=(25, 2))
        outlier_target = rng.uniform(-5, 5, size=(25, 2))  # unrelated positions
        source = np.concatenate([static_source, outlier_source])
        target = np.concatenate([static_target, outlier_target])
        R, t, n_inliers = icp_2d(source, target, inlier_thresh=0.15)
        assert n_inliers <= 110  # outliers mostly excluded, not all 125 counted
        assert abs(icp_rotation_angle(R) - theta) < 1e-2
        np.testing.assert_allclose(t, t_true, atol=1e-2)


class TestLocalCartesianPoints:
    def test_drops_max_range_beams(self):
        scan = np.full(720, FAR, dtype=np.float32)
        scan[360] = 3.0  # forward beam, real return
        pts = local_cartesian_points(scan, ANGLES, max_range=9.9)
        assert len(pts) == 1
        np.testing.assert_allclose(pts[0], [3.0, 0.0], atol=1e-3)


class TestEndToEndTrajectory:
    """The critical check: feed a known robot trajectory through synthetic
    scans and confirm the recovered trajectory matches it.

    Per-step motion magnitudes are picked to match FROG's actual regime
    (consecutive raw frames are 25ms apart at 40Hz, so real per-step motion
    is small) and to stay above the beam-quantisation noise floor (720 beams
    over 180 deg -> ~0.25 deg/beam, ~3.5cm lateral quantisation at 8m range)
    without exceeding the point-cloud's own nearest-neighbour spacing, which
    is what identity-initial-guess ICP can be expected to resolve -- see the
    TestIcp2D comment above for why arbitrary large motions aren't a fair
    (or relevant) test of this module's actual use case.
    """

    def _landmarks(self, seed=0, n=120):
        rng = np.random.RandomState(seed)
        r = rng.uniform(1.5, 8.0, n)
        phi = rng.uniform(-1.3, 1.3, n)
        return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)

    def test_pure_forward_translation(self):
        landmarks = self._landmarks()
        poses = [(0.0, 0.0, 0.0), (0.06, 0.0, 0.0), (0.12, 0.0, 0.0)]
        scans = np.stack([_synthesize_scan(p, landmarks) for p in poses])
        traj, inliers = integrate_trajectory(scans, ANGLES, inlier_thresh=0.15)
        np.testing.assert_allclose(traj[1], [0.06, 0.0, 0.0], atol=0.02)
        np.testing.assert_allclose(traj[2], [0.12, 0.0, 0.0], atol=0.02)
        assert inliers[1] > 30 and inliers[2] > 30

    def test_pure_rotation_in_place(self):
        landmarks = self._landmarks()
        poses = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.015), (0.0, 0.0, 0.03)]
        scans = np.stack([_synthesize_scan(p, landmarks) for p in poses])
        traj, inliers = integrate_trajectory(scans, ANGLES, inlier_thresh=0.15)
        np.testing.assert_allclose(traj[1], [0.0, 0.0, 0.015], atol=0.01)
        np.testing.assert_allclose(traj[2], [0.0, 0.0, 0.03], atol=0.01)

    def test_combined_forward_and_turn(self):
        landmarks = self._landmarks()
        poses = [(0.0, 0.0, 0.0), (0.05, 0.01, 0.02), (0.10, 0.03, 0.04)]
        scans = np.stack([_synthesize_scan(p, landmarks) for p in poses])
        traj, inliers = integrate_trajectory(scans, ANGLES, inlier_thresh=0.15)
        for est, gt in zip(traj[1:], poses[1:]):
            np.testing.assert_allclose(est, gt, atol=0.02)

    def test_sideways_motion(self):
        # Sideways (left/right) drift is the harder case for range-only
        # matching (less range change per beam than head-on motion) -- still
        # must recover the right sign, not just the right rough magnitude.
        landmarks = self._landmarks()
        poses = [(0.0, 0.0, 0.0), (0.0, 0.06, 0.0), (0.0, 0.12, 0.0)]
        scans = np.stack([_synthesize_scan(p, landmarks) for p in poses])
        traj, inliers = integrate_trajectory(scans, ANGLES, inlier_thresh=0.15)
        np.testing.assert_allclose(traj[1], [0.0, 0.06, 0.0], atol=0.02)
        np.testing.assert_allclose(traj[2], [0.0, 0.12, 0.0], atol=0.02)

    def test_static_robot_yields_near_zero_drift(self):
        landmarks = self._landmarks()
        poses = [(0.0, 0.0, 0.0)] * 4
        scans = np.stack([_synthesize_scan(p, landmarks) for p in poses])
        traj, _ = integrate_trajectory(scans, ANGLES, inlier_thresh=0.1)
        np.testing.assert_allclose(traj, 0.0, atol=1e-3)

    def test_implausible_single_step_is_clamped_to_zero_motion(self):
        # A feature-poor scan (almost no valid returns) between two otherwise
        # normal frames must not inject a wild, permanent jump into the rest
        # of the dead-reckoned trajectory -- this is the real failure mode
        # found on actual FROG data (a large open-space region) that
        # motivated adding the clamp at all.
        landmarks = self._landmarks()
        poses = [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.10, 0.0, 0.0)]
        scans = [_synthesize_scan(p, landmarks) for p in poses]
        sparse_scan = np.full(720, FAR, dtype=np.float32)
        sparse_scan[350:370] = 8.0  # a handful of ambiguous far returns only
        scans_with_glitch = np.stack([scans[0], sparse_scan, scans[2]])
        traj, inliers = integrate_trajectory(
            scans_with_glitch, ANGLES, inlier_thresh=0.15,
            max_step_dist=0.2, max_step_dtheta=np.radians(10))
        # frame 1 (the glitch) must not show a wild jump
        step1 = np.hypot(traj[1, 0] - 0.0, traj[1, 1] - 0.0)
        assert step1 < 0.2
        assert inliers[1] == 0  # clamp reports it as a rejected/untrusted step

    def test_recording_gap_forces_zero_motion_even_when_icp_looks_fine(self):
        # The real bug found on actual FROG data: a multi-second (or
        # multi-hour) recording pause between two otherwise-normal scans can
        # produce a small, *plausible-looking* ICP delta purely by chance
        # (matching two unrelated views of the scene), which the magnitude
        # clamp above cannot catch since nothing about the *result* looks
        # wrong. Two scenes that are genuinely unrelated (here: landmarks
        # completely repositioned, simulating "the scene changed during the
        # gap") must still be forced to zero motion once a timestamp gap is
        # present, regardless of what ICP itself returns for that pair.
        landmarks_a = self._landmarks(seed=0)
        landmarks_b = self._landmarks(seed=99)  # a different, unrelated scene
        scans = np.stack([
            _synthesize_scan((0.0, 0.0, 0.0), landmarks_a),
            _synthesize_scan((0.0, 0.0, 0.0), landmarks_b),  # same pose label, different scene
            _synthesize_scan((0.05, 0.0, 0.0), landmarks_b),
        ])
        timestamps = np.array([0.0, 50.0, 50.025])  # a 50s gap before frame 1
        traj, inliers = integrate_trajectory(
            scans, ANGLES, timestamps=timestamps, max_dt=0.1, inlier_thresh=0.15)
        assert traj[1, 0] == 0.0 and traj[1, 1] == 0.0 and traj[1, 2] == 0.0
        assert inliers[1] == 0
        # the *next* (normal-gap) transition must still work normally
        np.testing.assert_allclose(traj[2, :2] - traj[1, :2], [0.05, 0.0], atol=0.02)

    def test_no_timestamps_skips_the_gap_check_entirely(self):
        landmarks = self._landmarks()
        poses = [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0)]
        scans = np.stack([_synthesize_scan(p, landmarks) for p in poses])
        traj, _ = integrate_trajectory(scans, ANGLES, timestamps=None, inlier_thresh=0.15)
        np.testing.assert_allclose(traj[1, :2], [0.05, 0.0], atol=0.02)
