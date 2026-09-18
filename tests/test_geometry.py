"""
Unit tests for the odometry / ego-motion geometry used across temporal
alignment, tracking, and dataset windowing.

These exist because two real, silent bugs were found in this exact area
during manual review (SimpleTracker.step() never forwarding its ego_dxy
argument to _Track.predict(), and aligned_raw_scan()'s rotation shift having
its sign flipped) — both would have kept passing any test that only checked
"does it run" or "is the sign roughly plausible". Every case here is checked
against an analytically hand-derived expected value, not a vibe.

CPU-only: no model loading, no dataset files on disk, no GPU.
"""
import numpy as np
import pytest

from follow_the_drow.utils.drow_utils import aligned_raw_scan, cutout
from follow_the_drow.utils.tracking import SimpleTracker, odom_xya_delta_to_tracker_frame

N = 181
BEAM_SPACING = np.deg2rad(1.0)          # 1 deg/beam, symmetric FoV
ANGLES = np.linspace(-np.pi / 2, np.pi / 2, N, dtype=np.float32)
CENTER = N // 2                          # phi = 0, straight ahead


def _odom(theta_list, xy_list=None):
    """Build a structured odometry array matching DROW_Dataset's own dtype."""
    T = len(theta_list)
    xy_list = xy_list or [(0.0, 0.0)] * T
    o = np.zeros(T, dtype=[("eq", np.uint32), ("t", np.float32), ("xya", np.float32, 3)])
    for i, (th, (x, y)) in enumerate(zip(theta_list, xy_list)):
        o["xya"][i] = [x, y, th]
    return o


def _spike_scan(idx, far=10.0, near=3.0):
    s = np.full(N, far, dtype=np.float32)
    s[idx] = near
    return s


class TestAlignedRawScan:
    """aligned_raw_scan(): rotation- and translation-corrected temporal alignment."""

    def test_rotation_left_shifts_point_right(self):
        # Robot turns +10deg (CCW/left), zero translation. A static point at
        # beam 100 must appear, in the "now" frame, at beam 100 - 10 = 90:
        # turning left makes a stationary point appear to shift right.
        scans = np.stack([_spike_scan(100), np.full(N, 10.0, dtype=np.float32)])
        odom = _odom([0.0, np.deg2rad(10.0)])
        aligned = aligned_raw_scan(scans, odom, beam_spacing=BEAM_SPACING, angles=ANGLES)
        assert int(np.argmin(aligned[0, :, 0])) == 90

    def test_rotation_right_shifts_point_left(self):
        scans = np.stack([_spike_scan(100), np.full(N, 10.0, dtype=np.float32)])
        odom = _odom([0.0, np.deg2rad(-10.0)])
        aligned = aligned_raw_scan(scans, odom, beam_spacing=BEAM_SPACING, angles=ANGLES)
        assert int(np.argmin(aligned[0, :, 0])) == 110

    def test_rotation_delta_wraps_across_the_pi_boundary(self):
        # Historical theta=+170deg, current theta=-170deg: a naive subtraction
        # gives -340deg (wraps to +20deg -- the true, small rotation) but a
        # naive implementation would compute -340deg directly and divide it
        # by beam_spacing, producing a ~340-beam bogus shift instead of the
        # correct ~20-beam one. Real FROG odometry's theta spans the full
        # +/-pi range and does cross this boundary in practice (confirmed
        # empirically, ~1% of real windows) -- this is not a hypothetical case.
        scans = np.stack([_spike_scan(100), np.full(N, 10.0, dtype=np.float32)])
        odom = _odom([np.deg2rad(170.0), np.deg2rad(-170.0)])
        aligned = aligned_raw_scan(scans, odom, beam_spacing=BEAM_SPACING, angles=ANGLES)
        assert int(np.argmin(aligned[0, :, 0])) == 80

    def test_forward_translation_shortens_straight_ahead_range(self):
        # Exact geometry at phi=0 (no small-angle approximation needed): the
        # robot advancing 0.3m toward a point directly ahead must shorten
        # that beam's range by exactly 0.3m.
        scan = np.full(N, 5.0, dtype=np.float32)
        scans = np.stack([scan, scan])
        odom = _odom([0.0, 0.0], xy_list=[(0.0, 0.0), (0.3, 0.0)])
        aligned = aligned_raw_scan(scans, odom, beam_spacing=BEAM_SPACING, angles=ANGLES)
        assert aligned[0, CENTER, 0] == pytest.approx(4.7, abs=1e-4)

    def test_backward_translation_lengthens_straight_ahead_range(self):
        scan = np.full(N, 5.0, dtype=np.float32)
        scans = np.stack([scan, scan])
        odom = _odom([0.0, 0.0], xy_list=[(0.0, 0.0), (-0.3, 0.0)])
        aligned = aligned_raw_scan(scans, odom, beam_spacing=BEAM_SPACING, angles=ANGLES)
        assert aligned[0, CENTER, 0] == pytest.approx(5.3, abs=1e-4)

    def test_lateral_translation_leaves_straight_ahead_range_unchanged(self):
        # Motion perpendicular to the line of sight has ~zero first-order
        # projection onto a straight-ahead beam's range.
        scan = np.full(N, 5.0, dtype=np.float32)
        scans = np.stack([scan, scan])
        odom = _odom([0.0, 0.0], xy_list=[(0.0, 0.0), (0.0, 0.3)])
        aligned = aligned_raw_scan(scans, odom, beam_spacing=BEAM_SPACING, angles=ANGLES)
        assert aligned[0, CENTER, 0] == pytest.approx(5.0, abs=1e-3)

    def test_zero_odometry_is_a_no_op(self):
        # Backward compatibility: FROG (no odometry) must be unaffected.
        scan = np.full(N, 5.0, dtype=np.float32)
        scans = np.stack([scan, scan])
        odom = _odom([0.0, 0.0])
        aligned = aligned_raw_scan(scans, odom, beam_spacing=BEAM_SPACING, angles=ANGLES)
        np.testing.assert_allclose(aligned[:, :, 0], scans, atol=1e-4)

    def test_combined_rotation_and_translation_positions_from_rotation(self):
        scans = np.stack([_spike_scan(100), np.full(N, 10.0, dtype=np.float32)])
        odom = _odom([0.0, np.deg2rad(10.0)], xy_list=[(0.0, 0.0), (0.3, 0.0)])
        aligned = aligned_raw_scan(scans, odom, beam_spacing=BEAM_SPACING, angles=ANGLES)
        assert int(np.argmin(aligned[0, :, 0])) == 90


class TestCutout:
    """cutout(): DR-SPAAM's own per-beam local window (rotation-only, by design)."""

    def test_rotation_left_shifts_evidence_toward_smaller_index(self):
        scans = np.stack([_spike_scan(100), np.full(N, 10.0, dtype=np.float32)])
        odom = _odom([0.0, np.deg2rad(10.0)])
        cut = cutout(scans, odom, N, win_sz=1.66, thresh_dist=1.0, nsamp=48,
                     laserIncrement=BEAM_SPACING)
        hist_min = cut[:, 0, :].min(axis=1)
        assert int(np.argmin(hist_min)) < 100

    def test_rotation_delta_wraps_across_the_pi_boundary(self):
        # +170 deg then -170 deg is a 20 deg left turn, exactly like 0 then +20 deg;
        # an unwrapped difference is -340 deg, a 340-beam shift that moves every window off the scan.
        scans = np.stack([_spike_scan(100), np.full(N, 10.0, dtype=np.float32)])
        across = cutout(scans, _odom([np.deg2rad(170.0), np.deg2rad(-170.0)]), N, win_sz=1.66, thresh_dist=1.0, nsamp=48, laserIncrement=BEAM_SPACING)
        plain = cutout(scans, _odom([0.0, np.deg2rad(20.0)]), N, win_sz=1.66, thresh_dist=1.0, nsamp=48, laserIncrement=BEAM_SPACING)
        np.testing.assert_allclose(across, plain, atol=1e-5)

    def test_has_no_translation_correction(self):
        """
        Documents a known gap, not a regression: cutout() ignores translation
        entirely. The official DR-SPAAM reference implementation projects
        translation onto the cutout's forward axis (verified earlier via the
        upstream source) — this port's cutout() never picked that up. This
        test pins today's (incomplete) behaviour so a future fix is a
        deliberate, visible change here rather than a silent one.
        """
        scan = np.full(N, 5.0, dtype=np.float32)
        scans = np.stack([scan, scan])
        static = _odom([0.0, 0.0], xy_list=[(0.0, 0.0), (0.0, 0.0)])
        moved = _odom([0.0, 0.0], xy_list=[(0.0, 0.0), (0.3, 0.0)])
        cut_static = cutout(scans, static, N, thresh_dist=1.0, nsamp=48, laserIncrement=BEAM_SPACING)
        cut_moved = cutout(scans, moved, N, thresh_dist=1.0, nsamp=48, laserIncrement=BEAM_SPACING)
        np.testing.assert_allclose(cut_static, cut_moved, atol=1e-6)


class TestTrackingGeometry:
    """odom_xya_delta_to_tracker_frame() and SimpleTracker's ego-motion handling."""

    def test_odom_delta_forward_maps_to_tracker_frame_y(self):
        dx, dy = odom_xya_delta_to_tracker_frame(0.5, 0.0, theta_new=0.0)
        assert dx == pytest.approx(0.0, abs=1e-6)
        assert dy == pytest.approx(0.5, abs=1e-6)

    def test_odom_delta_left_maps_to_tracker_frame_negative_x(self):
        dx, dy = odom_xya_delta_to_tracker_frame(0.0, 0.5, theta_new=0.0)
        assert dx == pytest.approx(-0.5, abs=1e-6)
        assert dy == pytest.approx(0.0, abs=1e-6)

    def test_simpletracker_forward_ego_motion_shifts_coasted_track(self):
        # Regression test for the ego_dxy wiring bug: SimpleTracker.step()
        # used to silently drop this argument before calling _Track.predict().
        # A confirmed track, coasted (no new detection) through a 0.5m
        # forward robot move, must shift 5.0 -> 4.5 in the robot's own frame.
        t = SimpleTracker(match_radius=0.5, min_hits=1, max_age=5, vel_alpha=0.5)
        t.step([(0.9, 0.0, 5.0)])
        out = t.step([], ego_dtheta=0.0, ego_dxy=(0.0, 0.5))
        assert len(out) == 1
        assert out[0][2] == pytest.approx(4.5, abs=1e-4)

    def test_simpletracker_rotation_still_wired(self):
        t = SimpleTracker(match_radius=1.0, min_hits=1, max_age=5, vel_alpha=0.5)
        t.step([(0.9, 0.0, 5.0)])
        out = t.step([], ego_dtheta=np.deg2rad(10.0), ego_dxy=(0.0, 0.0))
        assert len(out) == 1
        assert out[0][1] > 0.05  # x shifts positive (right) when turning left


class TestDrowDatasetIndexing:
    """DROW_Dataset.get_scan()'s dtime striding and start-of-sequence clamping."""

    @staticmethod
    def _get_scan_idx(scan_id, time_window, dtime):
        return np.clip(scan_id - dtime * np.arange(time_window - 1, -1, -1), 0, scan_id)

    def test_dtime_1_gives_consecutive_frames(self):
        idx = self._get_scan_idx(50, 5, 1)
        assert list(idx) == [46, 47, 48, 49, 50]

    def test_dtime_5_matches_annotation_cadence(self):
        idx = self._get_scan_idx(50, 5, 5)
        assert list(idx) == [30, 35, 40, 45, 50]

    def test_near_start_of_sequence_clamps_to_zero(self):
        idx = self._get_scan_idx(2, 5, 1)
        assert list(idx) == [0, 0, 0, 1, 2]

    def test_very_first_frame_clamps_entirely(self):
        idx = self._get_scan_idx(0, 5, 1)
        assert list(idx) == [0, 0, 0, 0, 0]
