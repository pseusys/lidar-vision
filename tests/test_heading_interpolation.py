"""The two heading interpolations, and the difference between them.

`FROG_Dataset._load_odom` interpolated heading straight through 0 at every +-180 degree crossing until 2026-09-14: 54
single-frame jumps above 2 degrees on the test recording, up to 144 degrees. Fixing it moved the displacement oracle from 4.20
to 2.58 FP per FN, far more than the ~0.14% of affected frames should move anything, and nobody has explained why.
`utils/heading_ab.py` answers that by re-projecting the same detections through both pose series, so both interpolations have
to be exactly what the loader did before and after.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from heading_ab import naive_heading, unwrapped_heading

PI = np.pi


class TestHeadingInterpolation:
    def test_the_two_agree_where_no_crossing_happens(self):
        odom_t, heading = np.array([0.0, 1.0, 2.0]), np.array([0.1, 0.2, 0.3])
        at = np.array([0.5, 1.5])

        assert naive_heading(at, odom_t, heading) == pytest.approx(unwrapped_heading(at, odom_t, heading))

    def test_the_fixed_one_takes_the_short_way_across_the_crossing(self):
        """From +170 to -170 degrees is a 20 degree turn, and the midpoint is 180, not 0."""
        odom_t = np.array([0.0, 1.0])
        heading = np.radians([170.0, -170.0])

        midpoint = unwrapped_heading(np.array([0.5]), odom_t, heading)[0]

        assert abs(abs(midpoint) - PI) < 1e-9

    def test_the_old_one_swings_through_zero_instead(self):
        odom_t = np.array([0.0, 1.0])
        heading = np.radians([170.0, -170.0])

        midpoint = naive_heading(np.array([0.5]), odom_t, heading)[0]

        assert abs(midpoint) < 1e-9                                  # the robot appears to spin 170 degrees and back

    def test_the_old_one_is_wrong_by_up_to_pi_near_a_crossing(self):
        odom_t = np.array([0.0, 1.0])
        heading = np.radians([179.0, -179.0])
        at = np.linspace(0.0, 1.0, 11)

        error = np.abs(np.angle(np.exp(1j * (naive_heading(at, odom_t, heading) - unwrapped_heading(at, odom_t, heading)))))

        assert error.max() > PI - 0.05

    def test_both_return_angles_inside_one_turn(self):
        odom_t = np.array([0.0, 1.0, 2.0, 3.0])
        heading = np.radians([170.0, -170.0, 150.0, -150.0])
        at = np.linspace(0.0, 3.0, 50)

        for series in (naive_heading(at, odom_t, heading), unwrapped_heading(at, odom_t, heading)):
            assert np.all(np.abs(series) <= PI + 1e-12)

    def test_a_sample_outside_the_odometry_span_clamps_rather_than_extrapolating(self):
        odom_t, heading = np.array([1.0, 2.0]), np.array([0.5, 0.7])

        assert unwrapped_heading(np.array([0.0, 3.0]), odom_t, heading) == pytest.approx([0.5, 0.7])
