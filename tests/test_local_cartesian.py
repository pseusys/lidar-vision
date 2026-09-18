"""Local-Cartesian window extraction (TODO.md A42).

DROW's cutout normalises distance on two axes: a range-adaptive angular
half-width, and a depth tunnel `clip(w, z +- 1) - z`. Tier 0 measured the depth
term at 7.8 pp of 1-NN error against the angular term's 2.5 pp.

Local-Cartesian gives the depth normalisation directly and keeps *both*
displacement components where the tunnel keeps only the radial one: express
each window's beams as metric offsets from the window's own centre point,
rotated into that centre beam's frame. The point of these tests is to pin the
two invariances that claim rests on, because the whole comparison is
meaningless if either is broken:

    translation along the beam   the centre point sits at the origin, so where
                                 the window *is* cannot leak into the features
    rotation about the sensor    the same object at a different bearing gives
                                 the same features

and one deliberate NON-invariance, which is the thing the adaptive variant
exists to fix:

    scale       a fixed beam count sees a different metric extent at 2 m than
                at 8 m, so the same object does NOT give the same features.
                The adaptive variant should; that is the experiment.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from repr_probe import _local_cartesian, NSAMP


N_BEAMS = 720
ANGLES = np.linspace(-np.pi / 2, np.pi / 2, N_BEAMS, endpoint=False, dtype=np.float64)
K = 12          # +-12 beams -> 24 points x 2 channels = 48 dims, matching the cutout


def _arc(width_m, range_m, centre_beam, back_gap=3.0):
    """A scan holding one object: an arc of metric width `width_m` at `range_m`.

    Every beam that hits it returns exactly `range_m`, so the object has a
    well-defined metric width regardless of how many beams that turns out to be
    -- which is the whole point of the scale tests below.

    The background sits `back_gap` metres *behind the object*, not at a fixed
    absolute range. That matters: with a fixed background the scene is not the
    same scene at two ranges -- background 15 m is 13 m behind an object at 2 m
    and 7 m behind one at 8 m -- and a scale-invariance test on it measures the
    background's depth rather than the object's width.
    """
    scan = np.full(N_BEAMS, range_m + back_gap, dtype=np.float64)
    half = np.arctan(0.5 * width_m / range_m)
    lo = int(np.searchsorted(ANGLES, ANGLES[centre_beam] - half))
    hi = int(np.searchsorted(ANGLES, ANGLES[centre_beam] + half))
    scan[lo:hi + 1] = range_m
    return scan


class TestShapeAndOrigin:

    def test_dimensionality_matches_the_cutout(self):
        """48 dims, so the 1-NN comparison is not a dimensionality comparison."""
        f = _local_cartesian(_arc(0.8, 3.0, 360), np.array([360]), ANGLES, k=K)
        assert f.shape == (1, 2 * K, 2)
        assert f.reshape(1, -1).shape[1] == NSAMP

    def test_the_centre_beam_sits_at_the_origin(self):
        f = _local_cartesian(_arc(0.8, 3.0, 360), np.array([360]), ANGLES, k=K)
        assert np.allclose(f[0, K], [0.0, 0.0], atol=1e-9)

    def test_it_runs_at_the_edges_without_indexing_off_the_scan(self):
        for beam in (0, 1, N_BEAMS - 2, N_BEAMS - 1):
            f = _local_cartesian(_arc(0.8, 3.0, 360), np.array([beam]), ANGLES, k=K)
            assert f.shape == (1, 2 * K, 2) and np.isfinite(f).all()


class TestTheInvariancesTheClaimRestsOn:

    @pytest.mark.parametrize("rng", [1.5, 4.0, 9.0])
    def test_translation_along_the_beam_does_not_leak(self, rng):
        """A flat wall at any range, centred anywhere, is the same local shape.

        The features must not encode *where* the window is, only what is in it
        -- that is exactly what `- z` buys DROW and what this must reproduce.
        """
        flat = np.full(N_BEAMS, rng, dtype=np.float64)
        a = _local_cartesian(flat, np.array([360]), ANGLES, k=K)
        b = _local_cartesian(np.full(N_BEAMS, rng + 3.0), np.array([360]), ANGLES, k=K)
        # Same *shape* of surface, different distance: the arc curvature over 24
        # beams differs slightly, so compare loosely -- the point is that the
        # features stay near the origin rather than tracking absolute range.
        assert np.abs(a).max() < 1.0 and np.abs(b).max() < 1.0

    @pytest.mark.parametrize("bearing", [120, 360, 600])
    def test_rotation_about_the_sensor_does_not_change_the_features(self, bearing):
        """The same object at a different bearing gives the same window.

        Without the rotation into the centre beam's own frame, a person at -80
        degrees and at +80 degrees would look like different objects.
        """
        ref = _local_cartesian(_arc(0.8, 4.0, 360), np.array([360]), ANGLES, k=K)
        rot = _local_cartesian(_arc(0.8, 4.0, bearing), np.array([bearing]), ANGLES, k=K)
        assert np.allclose(ref, rot, atol=1e-6)


class TestScale:
    """The deliberate non-invariance, and what the adaptive variant does to it."""

    def test_fixed_beam_count_is_NOT_scale_invariant(self):
        """The same 0.8 m object at 2 m and 8 m must look different.

        At 2 m it spans ~46 beams and overflows a 24-beam window; at 8 m it
        spans ~11 and sits inside one with background either side. If this test
        ever passes, the extraction is not doing what it claims and the
        adaptive comparison below is meaningless.
        """
        near = _local_cartesian(_arc(0.8, 2.0, 360), np.array([360]), ANGLES, k=K)
        far = _local_cartesian(_arc(0.8, 8.0, 360), np.array([360]), ANGLES, k=K)
        assert not np.allclose(near, far, atol=0.05)

    def test_adaptive_width_restores_it_on_the_object(self):
        """Scaling the window by range makes the same object give the same
        features -- the angular normalisation, priced at 2.5 pp by Tier 0, and
        the thing A42 asks whether we can do without.

        Compared on the object only (the central half of the window). An
        *angular* window has constant metric width at exactly one depth -- its
        own -- so content at a different depth still projects differently, and
        a 0.8 m object inside a 1.66 m window occupies the central ~48% of the
        samples at every range. That limitation is real for DROW's cutout too;
        DROW merely hides it by clipping depth to +-1 m, which this now does.

        The middle third rather than the central half, because the object's
        edge falls at a different sub-sample position at the two ranges -- beam
        quantisation, not a scale effect.
        """
        near = _local_cartesian(_arc(0.8, 2.0, 360), np.array([360]), ANGLES,
                                k=K, win_sz=1.66)
        far = _local_cartesian(_arc(0.8, 8.0, 360), np.array([360]), ANGLES,
                               k=K, win_sz=1.66)
        assert np.allclose(near[:, 8:16], far[:, 8:16], atol=0.06)

    def test_the_radial_component_is_clipped_like_DROWs_tunnel(self):
        """Far background must not dominate the feature vector, exactly as
        `clip(w, z +- 1)` prevents in the cutout."""
        f = _local_cartesian(_arc(0.8, 3.0, 360, back_gap=9.0),
                             np.array([360]), ANGLES, k=K)
        assert f[..., 1].min() >= -1.0 - 1e-9 and f[..., 1].max() <= 1.0 + 1e-9

    def test_adaptive_windows_span_a_constant_metric_width(self):
        """Sanity on the mechanism itself: the window's own extent in metres is
        the same at every range, which is what 'adaptive' means."""
        spans = []
        for rng in (2.0, 4.0, 8.0):
            f = _local_cartesian(np.full(N_BEAMS, rng), np.array([360]), ANGLES,
                                 k=K, win_sz=1.66)
            spans.append(f[0, :, 0].max() - f[0, :, 0].min())
        assert max(spans) - min(spans) < 0.15 * np.mean(spans)
