"""The decoders `utils/stage2_probe.py` swaps under one stage-2 network, and the per-person breakdown it reads at a matched
false-positive rate (`TODO.md` A59).

The decoders are ports, so what is pinned is the behaviour each source defines: the official DR-SPAAM repository's
`nms_predicted_center` keeps a detection's best beam and suppresses every vote within `min_dist` of it; LFE-Peaks' decoder puts a
person at the endpoint of the beam where the probability peaks and merges peaks closer than its radius. The breakdown must count
each annotated person once even where FROG's test file repeats a circle (`memory/data-model.md`), and must describe crowding by
the distance to the nearest *other* person.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from stage2_probe import nearest_other, peak_decode, unique_people
from train_takheliped import beam_votes, nms_decode

N_BEAMS = 720
ANGLES = np.linspace(-np.pi / 2, np.pi / 2, N_BEAMS)
WALL_M = 8.0


def _scan():
    return np.full(N_BEAMS, WALL_M)


def _votes_to(scan, angles, target_xy, beams):
    """Votes `[N, 2]` sending `beams` to `target_xy` exactly, in the (across, along - range) convention `make_targets` trains."""
    votes = np.zeros((len(scan), 2))
    r_gt, phi_gt = np.hypot(*target_xy), np.arctan2(-target_xy[0], target_xy[1])
    dphi = phi_gt - angles[beams]
    votes[beams, 0] = r_gt * np.sin(dphi)
    votes[beams, 1] = r_gt * np.cos(dphi) - scan[beams]
    return votes


class TestBeamVotes:

    def test_a_zero_vote_lands_on_the_beam_endpoint(self):
        xy = beam_votes(_scan(), ANGLES, np.zeros((N_BEAMS, 2)))
        assert np.allclose(np.hypot(xy[:, 0], xy[:, 1]), WALL_M)

    def test_a_trained_vote_lands_on_its_target(self):
        target = np.array([0.5, 4.0])
        beams = np.arange(355, 365)
        xy = beam_votes(_scan(), ANGLES, _votes_to(_scan(), ANGLES, target, beams))
        assert np.allclose(xy[beams], target, atol=1e-9)


class TestNmsDecode:

    def test_votes_within_min_dist_become_one_detection_scored_by_the_best_beam(self):
        scan, prob = _scan(), np.zeros(N_BEAMS)
        beams = np.arange(355, 365)
        prob[beams] = np.linspace(0.2, 0.9, len(beams))
        dets = nms_decode(scan, ANGLES, prob, _votes_to(scan, ANGLES, np.array([0.5, 4.0]), beams), min_dist=0.5)
        assert len(dets) == 1
        assert dets[0, 0] == np.float64(0.9)
        assert np.allclose(dets[0, 1:], [0.5, 4.0])

    def test_votes_farther_apart_than_min_dist_are_kept_apart(self):
        scan, prob = _scan(), np.zeros(N_BEAMS)
        left, right = np.arange(300, 305), np.arange(420, 425)
        prob[left], prob[right] = 0.8, 0.6
        votes = _votes_to(scan, ANGLES, np.array([1.0, 4.0]), left) + _votes_to(scan, ANGLES, np.array([-1.0, 4.0]), right)
        dets = nms_decode(scan, ANGLES, prob, votes, min_dist=0.5)
        assert sorted(dets[:, 0].tolist()) == [0.6, 0.8]

    def test_averaging_moves_the_detection_to_the_weighted_mean_of_the_votes_it_suppressed(self):
        scan, prob = _scan(), np.zeros(N_BEAMS)
        prob[[350, 351]] = [0.9, 0.3]
        votes = _votes_to(scan, ANGLES, np.array([0.0, 4.0]), np.array([350])) + _votes_to(scan, ANGLES, np.array([0.4, 4.0]), np.array([351]))
        dets = nms_decode(scan, ANGLES, prob, votes, min_dist=0.5, average=True)
        assert dets[:, 0].tolist() == [0.9]
        assert np.allclose(dets[0, 1:], [0.1, 4.0])

    def test_beams_below_the_vote_floor_cast_no_detection(self):
        dets = nms_decode(_scan(), ANGLES, np.zeros(N_BEAMS), np.zeros((N_BEAMS, 2)), min_dist=0.5)
        assert dets.shape == (0, 3)


class TestPeakDecode:

    def _two_bumps(self):
        prob = np.zeros(N_BEAMS)
        beams = np.arange(N_BEAMS)
        for centre in (300, 420):
            prob += 0.9 * np.exp(-0.5 * ((beams - centre) / 3.0) ** 2)
        return prob

    def test_each_peak_is_reported_at_its_beam_endpoint(self):
        dets = peak_decode(_scan(), ANGLES, self._two_bumps(), np.zeros((N_BEAMS, 2)), at_vote=False)
        endpoints = beam_votes(_scan(), ANGLES, np.zeros((N_BEAMS, 2)))[[300, 420]]
        assert len(dets) == 2
        assert np.allclose(sorted(dets[:, 1].tolist()), sorted(endpoints[:, 0].tolist()))

    def test_at_vote_reports_the_peak_beam_vote_instead(self):
        scan = _scan()
        votes = _votes_to(scan, ANGLES, np.array([1.0, 4.0]), np.arange(295, 306)) + _votes_to(scan, ANGLES, np.array([-1.0, 4.0]), np.arange(415, 426))
        dets = peak_decode(scan, ANGLES, self._two_bumps(), votes, at_vote=True)
        assert np.allclose(sorted(dets[:, 1].tolist()), [-1.0, 1.0])

    def test_peaks_closer_than_the_merge_radius_merge_into_the_stronger(self):
        prob = np.zeros(N_BEAMS)
        prob[[358, 362]] = [0.9, 0.5]                     # four beams apart, ~0.14 m at 8 m
        dets = peak_decode(_scan(), ANGLES, prob, np.zeros((N_BEAMS, 2)), at_vote=False)
        assert dets[:, 0].tolist() == [0.9]


class TestPeople:

    def test_a_repeated_circle_counts_once(self):
        people = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
        assert unique_people(people).tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_crowding_is_the_distance_to_the_nearest_other_person(self):
        people = np.array([[0.0, 2.0], [0.0, 2.5], [3.0, 2.0]])
        assert np.allclose(nearest_other(people), [0.5, 0.5, 3.0])

    def test_a_person_alone_has_no_neighbour(self):
        assert np.isinf(nearest_other(np.array([[0.0, 2.0]]))).all()
