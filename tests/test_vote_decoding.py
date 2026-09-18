"""
DROW's vote grid (`votes_to_detections`): per-beam votes -> detections.

These exist because a real, silent bug was found here (2026-09-15): two neighbouring grid cells whose blurred votes tie exactly are
both local maxima, every vote goes to the nearer one, and the other became a detection averaged over no votes -- NaN position and
NaN probability. A NaN detection passes every distance gate (NaN comparisons are False), so `_prec_rec_2d` could count it as a
true positive, and it crashed step 3a's candidate matching.

CPU-only: no model loading, no dataset files on disk, no GPU.
"""
import numpy as np

from follow_the_drow.utils.drow_utils import votes_to_detections

# `utils/train_three_horizon.py`'s VOTE_DECODING
SETTINGS = dict(blur_sigma=2.0, blur_win=11, bin_size=0.1, vote_collect_radius=0.5, min_thresh=1e-3)

# Found by a random search: the blurred grid ties at bins (142, 105) and (143, 104), and the only vote near them is nearer the second.
TIED_X = np.array([-0.8575, -1.3792, -1.0528, -0.7165, -1.5498, -0.3678, -0.332, -0.4872])
TIED_Y = np.array([5.8933, 5.9216, 6.2022, 5.4168, 4.8366, 5.3797, 6.5449, 6.612])


def _decode(x, y, prob):
    probas = np.zeros((1, len(x), 4), np.float32)
    probas[0, :, 3] = prob
    return votes_to_detections(x[None], y[None], probas, **SETTINGS)[0]


class TestTiedMaxima:

    def test_every_detection_is_finite(self):
        for x, y, probs in _decode(TIED_X, TIED_Y, 0.5):
            assert np.isfinite(x) and np.isfinite(y) and np.isfinite(probs).all()

    def test_only_the_peak_without_votes_is_dropped(self):
        found = sorted((round(float(x), 3), round(float(y), 3)) for x, y, _ in _decode(TIED_X, TIED_Y, 0.5))
        assert found == [(-1.55, 4.837), (-1.216, 6.062), (-0.858, 5.893), (-0.542, 5.398), (-0.487, 6.612), (-0.332, 6.545)]


class TestOrdinaryVotes:

    def test_a_tight_cluster_decodes_to_its_mean_with_its_mean_probability(self):
        x, y = np.array([1.0, 1.02, 0.98]), np.array([3.0, 3.01, 2.99])
        detections = _decode(x, y, np.array([0.4, 0.6, 0.8]))
        assert len(detections) == 1
        dx, dy, probs = detections[0]
        assert abs(dx - 1.0) < 1e-6 and abs(dy - 3.0) < 1e-6
        assert abs(probs[3] - 0.6) < 1e-6
