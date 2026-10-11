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


def _reference_nms(scan, angles, prob, votes, min_dist, average):
    """The decoder as written before 2026-10-11, transcribed: one Python iteration per voter, alive or not.

    Kept here as the thing the faster loop must agree with exactly. It is the definition of `nms-mean` that every number
    in `memory/performance-log.md` was produced under, so a restructuring that merely *approximates* it would silently
    re-point the whole stage-3 comparison at a different decoder.
    """
    from train_takheliped import VOTE_FLOOR
    voters = np.nonzero(prob > VOTE_FLOOR)[0]
    voters = voters[np.argsort(-prob[voters], kind="stable")]
    p, xy = prob[voters], beam_votes(scan[voters], angles[voters], votes[voters])
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    alive = np.ones(len(voters), dtype=bool)
    kept, centres = [], []
    for i in range(len(voters)):
        if alive[i]:
            members = alive & (dist[i] < min_dist)
            kept.append(i)
            centres.append(np.average(xy[members], axis=0, weights=p[members]) if average else xy[i])
            alive[members] = False
    return np.column_stack([p[kept], np.array(centres).reshape(-1, 2)])


class TestGreedyOrderIsTheDefinition:
    """What an incorrect speed-up of the decode loop would break, pinned before the loop was touched (item 30).

    The decode runs 314 Python iterations per frame to keep 22 detections, so only the survivors need iterating -- but
    `members` must keep being drawn from the beams still **alive**, not from every beam within the radius. The two differ
    whenever three votes chain: A suppresses B, and C, too far from A to be suppressed, must then average over itself
    alone, because B has already been consumed by A.
    """

    @staticmethod
    def _chain():
        scan, prob = _scan(), np.zeros(N_BEAMS)
        beams = np.array([350, 351, 352])
        prob[beams] = [0.9, 0.8, 0.7]
        votes = (_votes_to(scan, ANGLES, np.array([0.0, 4.0]), beams[:1])
                 + _votes_to(scan, ANGLES, np.array([0.4, 4.0]), beams[1:2])
                 + _votes_to(scan, ANGLES, np.array([0.8, 4.0]), beams[2:]))
        return scan, prob, votes

    def test_a_consumed_vote_is_not_averaged_into_a_later_detection(self):
        scan, prob, votes = self._chain()

        dets = nms_decode(scan, ANGLES, prob, votes, min_dist=0.5, average=True)

        assert dets[:, 0].tolist() == [0.9, 0.7]
        assert np.allclose(dets[0, 1], (0.0 * 0.9 + 0.4 * 0.8) / 1.7)   # A averaged with B
        assert np.allclose(dets[1, 1], 0.8)                              # C alone: B was already taken

    def test_the_chain_matches_the_reference(self):
        scan, prob, votes = self._chain()

        assert np.array_equal(nms_decode(scan, ANGLES, prob, votes, 0.5, average=True),
                              _reference_nms(scan, ANGLES, prob, votes, 0.5, average=True))

    def test_randomised_frames_match_the_reference_exactly_in_both_modes(self):
        """Bit-exact, not approximate: the candidate caches on disk were decoded by the reference."""
        rng = np.random.default_rng(20261011)
        for _ in range(40):
            scan = rng.uniform(1.0, WALL_M, N_BEAMS)
            prob = np.where(rng.random(N_BEAMS) < 0.4, rng.random(N_BEAMS), 0.0)
            votes = rng.normal(0.0, 0.3, (N_BEAMS, 2))
            for average in (False, True):
                assert np.array_equal(nms_decode(scan, ANGLES, prob, votes, 0.5, average=average),
                                      _reference_nms(scan, ANGLES, prob, votes, 0.5, average=average))

    def test_a_single_voter_survives_on_its_own(self):
        scan, prob = _scan(), np.zeros(N_BEAMS)
        prob[400] = 0.42

        dets = nms_decode(scan, ANGLES, prob, np.zeros((N_BEAMS, 2)), 0.5, average=True)

        assert dets.shape == (1, 3) and dets[0, 0] == 0.42

    def test_every_vote_in_one_cluster_collapses_to_one_detection(self):
        scan, prob = _scan(), np.zeros(N_BEAMS)
        beams = np.arange(340, 380)
        prob[beams] = np.linspace(0.1, 0.99, len(beams))

        dets = nms_decode(scan, ANGLES, prob, _votes_to(scan, ANGLES, np.array([0.0, 4.0]), beams), 0.5, average=True)

        assert len(dets) == 1
