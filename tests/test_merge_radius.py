"""`_merge_nearby`'s range-aware radius (TODO.md A38).

The radius that decides whether two centroids are one person was a single
scalar in metres.  Tier 0 (`memory/static-detector-diagnosis.md`) measured it
failing in both directions at the same value: at 0.30 m LFE-Peaks loses ~5 pp of
recall to close pairs merged into one detection, while LFE-PPN emits 1.609
duplicates per frame.  A detector's localisation error grows with range and the
separation between two people does not, so the radius is now
``r(d) = merge_radius + merge_radius_slope * d``.

``slope = 0`` must be bit-identical to the old behaviour -- that is the first
test here, and the one that matters most, because every published number in this
repository was measured under it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))

from follow_the_drow.detectors.lfe_detector import _merge_nearby


def _at(score, x, y):
    return (score, x, y)


class TestSlopeZeroIsTheOldBehaviour:
    """The regression guard. Every number in the repo was measured at slope 0."""

    def test_empty_input(self):
        assert _merge_nearby([], 0.30) == []
        assert _merge_nearby([], 0.30, slope=0.0) == []

    def test_keeps_the_higher_score_and_drops_the_neighbour(self):
        dets = [_at(0.4, 2.0, 0.0), _at(0.9, 2.1, 0.0)]
        kept = _merge_nearby(dets, 0.30)
        assert kept == [_at(0.9, 2.1, 0.0)]

    def test_keeps_both_when_further_apart_than_the_radius(self):
        dets = [_at(0.9, 2.0, 0.0), _at(0.4, 2.4, 0.0)]
        assert len(_merge_nearby(dets, 0.30)) == 2

    def test_explicit_zero_slope_matches_the_default(self):
        dets = [_at(0.9, 5.0, 0.0), _at(0.8, 5.2, 0.0), _at(0.7, 5.9, 0.0),
                _at(0.6, 1.0, 1.0), _at(0.5, 1.1, 1.0)]
        assert _merge_nearby(dets, 0.30) == _merge_nearby(dets, 0.30, slope=0.0)

    @pytest.mark.parametrize("radius", [0.25, 0.30, 0.40, 0.80])
    def test_zero_slope_is_unchanged_at_every_calibrated_radius(self, radius):
        """0.25/0.30/0.40/0.80 are the four radii this project has published a
        number under; none of them may move."""
        dets = [_at(0.95, 3.0, 0.0), _at(0.90, 3.35, 0.0), _at(0.85, 3.9, 0.0),
                _at(0.80, 8.0, 0.0), _at(0.75, 8.5, 0.0)]
        out = _merge_nearby(dets, radius, slope=0.0)
        expected = _merge_nearby(dets, radius)
        assert out == expected


class TestTheRadiusGrowsWithRange:
    """The point of the change: tight where the scan is dense, loose where it
    is sparse."""

    def test_same_separation_merges_far_and_survives_near(self):
        """0.35 m apart: two people at 2 m, one person's two legs at 8 m.

        With r(d) = 0.20 + 0.03 d the radius is 0.26 m at 2 m and 0.44 m at 8 m,
        so the identical geometry resolves near and merges far. A single scalar
        cannot do this, which is the whole argument.
        """
        near = [_at(0.9, 2.0, 0.0), _at(0.8, 2.35, 0.0)]
        far = [_at(0.9, 8.0, 0.0), _at(0.8, 8.35, 0.0)]
        assert len(_merge_nearby(near, 0.20, slope=0.03)) == 2
        assert len(_merge_nearby(far, 0.20, slope=0.03)) == 1

    @pytest.mark.parametrize("sep,expected", [(0.39, 1), (0.41, 2)])
    def test_the_threshold_is_a_plus_b_times_mean_range(self, sep, expected):
        """Radius is evaluated at the mean of the pair's ranges -- symmetric,
        and the two are always at similar range anyway when they are close
        enough to merge at all.

        Both detections are placed on the same range arc, so the mean range is
        exactly `d` and the threshold is exactly `a + b*d` with no slack to
        argue about. Separating them along x instead would put the second one
        further out, raising the radius by the very term under test.
        """
        import math
        a, b, d = 0.10, 0.05, 6.0                 # r(6 m) = 0.40 m exactly
        theta = 2 * math.asin(sep / (2 * d))      # chord of length `sep`
        pair = [_at(0.9, d, 0.0),
                _at(0.8, d * math.cos(theta), d * math.sin(theta))]
        assert len(_merge_nearby(pair, a, slope=b)) == expected

    def test_range_is_euclidean_not_just_x(self):
        """A detection off to the side is at its own range, not its x."""
        a, b = 0.0, 0.10             # r(d) = d / 10
        # (3, 4) is at range 5 -> radius 0.50 m.
        pair = [_at(0.9, 3.0, 4.0), _at(0.8, 3.0, 4.45)]
        assert len(_merge_nearby(pair, a, slope=b)) == 1
        apart = [_at(0.9, 3.0, 4.0), _at(0.8, 3.0, 4.55)]
        assert len(_merge_nearby(apart, a, slope=b)) == 2

    def test_score_ordering_still_decides_which_survives(self):
        dets = [_at(0.3, 9.0, 0.0), _at(0.99, 9.2, 0.0), _at(0.5, 9.1, 0.0)]
        kept = _merge_nearby(dets, 0.20, slope=0.05)
        assert len(kept) == 1 and kept[0][0] == pytest.approx(0.99)

    def test_negative_slope_is_allowed_and_tightens_with_range(self):
        """Not expected to help, but the sweep must be able to try it rather
        than silently clamp -- a clamped parameter looks like a measured
        optimum."""
        far = [_at(0.9, 9.0, 0.0), _at(0.8, 9.25, 0.0)]
        assert len(_merge_nearby(far, 0.50, slope=-0.03)) == 2   # r = 0.23 m
        near = [_at(0.9, 1.0, 0.0), _at(0.8, 1.25, 0.0)]
        assert len(_merge_nearby(near, 0.50, slope=-0.03)) == 1  # r = 0.47 m

    def test_radius_never_goes_negative(self):
        """A steep negative slope must floor at zero rather than start merging
        nothing-or-everything by sign accident."""
        dets = [_at(0.9, 20.0, 0.0), _at(0.8, 20.001, 0.0)]
        kept = _merge_nearby(dets, 0.30, slope=-1.0)
        assert len(kept) == 2


class TestDetectorsExposeIt:
    """Both detectors must take the slope, and default to the old behaviour."""

    def test_peaks_defaults_to_zero_slope(self):
        from follow_the_drow.detectors.lfe_detector import LFEPeaksDetector
        import inspect
        sig = inspect.signature(LFEPeaksDetector.__init__)
        assert sig.parameters["merge_radius_slope"].default == 0.0

    def test_ppn_defaults_to_zero_slope(self):
        from follow_the_drow.detectors.lfe_detector import LFEPPNDetector
        import inspect
        sig = inspect.signature(LFEPPNDetector.__init__)
        assert sig.parameters["nms_radius_slope"].default == 0.0
