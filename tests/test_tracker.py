"""
Correctness audit for `SimpleTracker`, before any paper cites a number it
produced (owner's request, 2026-09-10).

Two separate questions, and they need separating:

**Is it correct?** Does it do what its docstring says — predict, gate, match,
confirm after `min_hits`, coast for `max_age`, drop — under motion, under
ego-motion, and with more than one target?

**Is it citeable, and as what?** It is *not* SORT. SORT (Bewley et al., ICIP
2016) uses a Kalman filter with covariance propagation and associates on
bounding-box IoU. This associates point detections on Euclidean distance and
carries velocity in an alpha-beta filter, and unlike SORT it *reports* coasted
tracks. The honest description for a write-up is "a SORT-style tracker adapted
to point detections: Hungarian assignment (Kuhn 1955) on Euclidean distance
rather than IoU, an alpha-beta filter rather than a Kalman filter, and coasted
tracks reported for up to `max_age` frames". These tests pin each of those
deviations so the description cannot drift from the code.
"""
import numpy as np
import pytest

from follow_the_drow.utils.tracking import (
    SimpleTracker, odom_xya_delta_to_tracker_frame,
)


def _feed(tracker, points, n):
    """Run `n` frames of the same detection list; return the last output."""
    out = []
    for _ in range(n):
        out = tracker.step([(0.9, x, y) for x, y in points])
    return out


class TestConfirmation:
    def test_a_track_is_silent_until_min_hits_then_reported(self):
        tr = SimpleTracker(min_hits=3, max_age=3)
        assert tr.step([(0.9, 1.0, 2.0)]) == []            # hits=1
        assert tr.step([(0.9, 1.0, 2.0)]) == []            # hits=2
        out = tr.step([(0.9, 1.0, 2.0)])                    # hits=3
        assert len(out) == 1
        assert out[0][1:] == pytest.approx((1.0, 2.0))

    def test_min_hits_1_reports_on_the_creating_frame(self):
        tr = SimpleTracker(min_hits=1)
        assert len(tr.step([(0.9, 1.0, 2.0)])) == 1

    def test_hits_accumulate_non_consecutively(self):
        """`hits` counts matched frames over the track's whole life, not a run.

        This is why `min_hits` is a weak filter against a phantom that flickers:
        as long as it never misses more than `max_age` in a row, it still
        confirms — just later. Load-bearing for reading the paper's tracking
        table, so it is pinned rather than left implicit.
        """
        tr = SimpleTracker(min_hits=3, max_age=3)
        tr.step([(0.9, 1.0, 2.0)])          # hit 1
        tr.step([])                          # miss, coasting
        tr.step([(0.9, 1.0, 2.0)])          # hit 2
        tr.step([])                          # miss
        out = tr.step([(0.9, 1.0, 2.0)])    # hit 3 -> confirmed
        assert len(out) == 1


class TestCoasting:
    def test_a_confirmed_track_survives_exactly_max_age_misses(self):
        tr = SimpleTracker(min_hits=2, max_age=3)
        _feed(tr, [(1.0, 2.0)], 2)
        for expected_frame in range(1, 4):          # 3 coasted frames
            assert len(tr.step([])) == 1, expected_frame
        assert tr.step([]) == []                     # 4th miss -> dropped

    def test_coasted_scores_decay(self):
        tr = SimpleTracker(min_hits=2, max_age=5, coast_decay=0.5)
        _feed(tr, [(1.0, 2.0)], 2)
        assert tr.step([])[0][0] == pytest.approx(0.9 * 0.5)
        assert tr.step([])[0][0] == pytest.approx(0.9 * 0.25)

    def test_coasted_tracks_are_reported_unlike_in_sort(self):
        """The deviation from SORT that matters most for false-positive counts:
        SORT emits only tracks updated in the current frame, so a coasted track
        contributes nothing. Here it is emitted, which is the false-negative
        recovery mechanism -- and also why a permissive `min_hits` can *raise*
        the false-positive count above the untracked detector's."""
        tr = SimpleTracker(min_hits=2, max_age=3)
        _feed(tr, [(1.0, 2.0)], 2)
        out = tr.step([])                     # nothing detected at all
        assert len(out) == 1, "a coasted track must still be reported"


class TestMotion:
    def test_a_moving_target_stays_one_track(self):
        tr = SimpleTracker(match_radius=0.5, min_hits=2, max_age=3, vel_alpha=1.0)
        out = None
        for i in range(6):
            out = tr.step([(0.9, 0.0, 2.0 + 0.3 * i)])
        assert len(tr.tracks) == 1, "constant-velocity motion must not spawn tracks"
        assert out[0][2] == pytest.approx(2.0 + 0.3 * 5, abs=1e-6)

    def test_prediction_not_last_position_is_what_the_gate_tests(self):
        """A target that vanishes for a frame reappears 0.6 m from its last
        *observation* -- outside a 0.4 m gate -- but 0 m from its prediction."""
        tr = SimpleTracker(match_radius=0.4, min_hits=2, max_age=3, vel_alpha=1.0)
        for i in range(4):
            tr.step([(0.9, 0.0, 2.0 + 0.3 * i)])      # bootstraps vy = 0.3
        tr.step([])                                     # missed frame, coasts
        out = tr.step([(0.9, 0.0, 2.0 + 0.3 * 5)])     # 0.6 m from last observation
        assert len(tr.tracks) == 1, "prediction should have carried the track across"
        assert len(out) == 1

    def test_velocity_cannot_bootstrap_when_motion_exceeds_the_gate(self):
        """A real limitation, not a bug, and SORT shares it: a new track starts
        at zero velocity, so it can only earn a velocity estimate from a second
        match -- which the gate refuses if the target moves further than
        `match_radius` per frame. The track then fragments, one per frame.

        Harmless at FROG's rate: a person at 1.4 m/s covers 5.3 cm per frame at
        26.2 Hz, against a 0.5 m gate -- about a tenth of it. Worth stating in a
        write-up precisely *because* the gate is so much larger than the motion:
        association here is close to "was something detected near here last
        frame", which is why the tracker's behaviour tracks the phantom
        persistence measurement so closely."""
        tr = SimpleTracker(match_radius=0.2, min_hits=2, max_age=3, vel_alpha=1.0)
        for i in range(4):
            tr.step([(0.9, 0.0, 2.0 + 0.3 * i)])       # 0.3 m/frame > 0.2 m gate
        assert len(tr.tracks) == 4, "each frame seeds its own track"

    def test_two_separated_targets_stay_two_tracks(self):
        tr = SimpleTracker(match_radius=0.5, min_hits=2, max_age=3)
        out = _feed(tr, [(0.0, 2.0), (3.0, 2.0)], 3)
        assert len(out) == 2
        assert len(tr.tracks) == 2

    def test_a_detection_beyond_the_gate_starts_a_new_track(self):
        tr = SimpleTracker(match_radius=0.5, min_hits=1, max_age=3)
        tr.step([(0.9, 0.0, 2.0)])
        tr.step([(0.9, 0.0, 5.0)])           # 3 m away, far outside the gate
        assert len(tr.tracks) == 2


class TestEgoMotion:
    def test_robot_rotation_keeps_a_world_stationary_point_matched(self):
        """Without ego compensation the robot's own turn looks like the target
        moving, and a tight gate loses the track."""
        dtheta = np.deg2rad(5.0)
        r, phi = 4.0, 0.0
        tr = SimpleTracker(match_radius=0.2, min_hits=2, max_age=1)
        for k in range(5):
            # The point is world-stationary, so in the robot frame it appears
            # to rotate by -k*dtheta.
            p = phi - k * dtheta
            tr.step([(0.9, r * -np.sin(p), r * np.cos(p))], ego_dtheta=dtheta)
        assert len(tr.tracks) == 1, "ego-compensated rotation must not break the track"

    def test_without_compensation_the_same_rotation_breaks_it(self):
        dtheta = np.deg2rad(5.0)
        r, phi = 4.0, 0.0
        tr = SimpleTracker(match_radius=0.2, min_hits=2, max_age=1, vel_alpha=0.0)
        for k in range(5):
            p = phi - k * dtheta
            tr.step([(0.9, r * -np.sin(p), r * np.cos(p))])   # no ego_dtheta
        assert len(tr.tracks) > 1, (
            "control: 5 deg/frame at 4 m is 35 cm, well outside a 20 cm gate, so "
            "the uncompensated tracker must fragment -- otherwise the test above "
            "proves nothing")

    def test_translation_moves_a_stationary_point_backwards(self):
        """Robot drives forward -> a world-stationary point approaches it."""
        tr = SimpleTracker(match_radius=1.0, min_hits=1, max_age=3, vel_alpha=0.0)
        tr.step([(0.9, 0.0, 5.0)])
        before = (tr.tracks[0].x, tr.tracks[0].y)
        tr.step([], ego_dxy=(0.0, 0.5))       # 0.5 m forward
        assert tr.tracks[0].y == pytest.approx(before[1] - 0.5, abs=1e-6)

    def test_odom_conversion_maps_forward_motion_to_the_y_axis(self):
        """Odometry's local-forward is this module's +y, local-left its -x."""
        dx, dy = odom_xya_delta_to_tracker_frame(1.0, 0.0, 0.0)   # driving +x world, facing +x
        assert dy == pytest.approx(1.0)
        assert dx == pytest.approx(0.0)
        dx, dy = odom_xya_delta_to_tracker_frame(0.0, 1.0, 0.0)   # sliding left
        assert dx == pytest.approx(-1.0)
        assert dy == pytest.approx(0.0)


class TestRobustness:
    def test_non_finite_detections_are_dropped_not_crashed_on(self):
        tr = SimpleTracker(min_hits=1)
        out = tr.step([(0.9, np.nan, 2.0), (0.8, 1.0, 2.0), (np.inf, 3.0, 4.0)])
        assert len(out) == 1
        assert out[0][1:] == pytest.approx((1.0, 2.0))

    def test_empty_input_is_fine(self):
        tr = SimpleTracker()
        assert tr.step([]) == []

    def test_assignment_is_one_to_one(self):
        """Two detections inside one track's gate must not both bind to it."""
        tr = SimpleTracker(match_radius=1.0, min_hits=1, max_age=3)
        tr.step([(0.9, 0.0, 2.0)])
        tr.step([(0.9, 0.0, 2.0), (0.9, 0.3, 2.0)])
        assert len(tr.tracks) == 2, "the unmatched detection must seed its own track"


class TestSortTrackerIsFaithfulToSort:
    """`SortTracker` exists so the paper can cite SORT without a caveat list.

    These tests pin the four track-management rules that decide *what gets
    reported*, and therefore what a false-positive count measures. Each is
    checked against the reference implementation's behaviour, and each is a
    place `SimpleTracker` differs.
    """

    def test_the_creating_detection_is_not_a_hit(self):
        """SORT starts `hits` and `hit_streak` at 0, so `min_hits` counts
        matches *after* creation. `SimpleTracker` starts at 1 and reports one
        frame sooner."""
        from follow_the_drow.utils.tracking import SortTracker
        tr = SortTracker(min_hits=3, max_age=1)
        for _ in range(3):
            tr.step([(0.9, 1.0, 2.0)])                 # creation + 2 matches
        assert tr.tracks[0].hits == 2
        out = tr.step([(0.9, 1.0, 2.0)])               # 3rd match -> confirmed
        assert len(out) == 1

    def test_confirmation_needs_consecutive_hits(self):
        """The deviation that matters most for this paper. SORT zeroes
        `hit_streak` on any missed frame, so a phantom that flickers never
        confirms; `SimpleTracker`'s total-hit rule lets it confirm eventually."""
        from follow_the_drow.utils.tracking import SortTracker
        tr = SortTracker(min_hits=3, max_age=3)
        for _ in range(10):
            tr.step([(0.9, 1.0, 2.0)])                 # hit
            out = tr.step([])                           # miss -> streak resets
        assert out == [], "an every-other-frame detection must never confirm"

        steady = SortTracker(min_hits=3, max_age=3)
        for _ in range(6):
            out = steady.step([(0.9, 1.0, 2.0)])
        assert len(out) == 1, "control: the same detector, seen every frame, confirms"

    def test_coasted_tracks_are_not_reported(self):
        from follow_the_drow.utils.tracking import SortTracker
        tr = SortTracker(min_hits=2, max_age=5)
        for _ in range(4):
            tr.step([(0.9, 1.0, 2.0)])
        assert tr.step([]) == [], "SORT emits only tracks matched this frame"
        assert len(tr.tracks) == 1, "but the track itself survives to be re-matched"

    def test_the_early_frame_grace_period_is_preserved(self):
        """SORT reports unconfirmed tracks while `frame_count <= min_hits`, so a
        sequence does not open with `min_hits` frames of guaranteed silence."""
        from follow_the_drow.utils.tracking import SortTracker
        tr = SortTracker(min_hits=3, max_age=1)
        assert len(tr.step([(0.9, 1.0, 2.0)])) == 1

    def test_a_track_is_dropped_after_max_age_misses(self):
        from follow_the_drow.utils.tracking import SortTracker
        tr = SortTracker(min_hits=1, max_age=2)
        for _ in range(4):
            tr.step([(0.9, 1.0, 2.0)])
        tr.step([]); tr.step([])
        assert len(tr.tracks) == 1
        tr.step([])
        assert tr.tracks == []

    def test_the_kalman_filter_estimates_velocity(self):
        from follow_the_drow.utils.tracking import SortTracker
        tr = SortTracker(match_radius=1.0, min_hits=2, max_age=3)
        for i in range(12):
            tr.step([(0.9, 0.0, 2.0 + 0.3 * i)])
        assert len(tr.tracks) == 1
        assert tr.tracks[0].kf.x[3] == pytest.approx(0.3, abs=0.05), \
            "vy should converge on the true 0.3 m/frame"

    def test_ego_rotation_is_compensated(self):
        from follow_the_drow.utils.tracking import SortTracker
        dtheta = np.deg2rad(5.0)
        r = 4.0
        tr = SortTracker(match_radius=0.2, min_hits=2, max_age=1)
        for k in range(6):
            p = -k * dtheta
            tr.step([(0.9, r * -np.sin(p), r * np.cos(p))], ego_dtheta=dtheta)
        assert len(tr.tracks) == 1

    def test_assignment_stays_one_to_one(self):
        from follow_the_drow.utils.tracking import SortTracker
        tr = SortTracker(match_radius=1.0, min_hits=1, max_age=3)
        tr.step([(0.9, 0.0, 2.0)])
        tr.step([(0.9, 0.0, 2.0), (0.9, 0.3, 2.0)])
        assert len(tr.tracks) == 2
