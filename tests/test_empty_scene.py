"""What the detector does where nobody is: the counting behind `utils/empty_scene.py`.

On FROG `transferred`'s person-free frames every detection is a false positive by construction, so wp-AUC is undefined and the
measurement is a count (`memory/interpreting-evaluation.md`). The published static detectors emit 1.39 (LFE-Peaks) and 1.94
(LFE-PPN) per frame there and 88.3% / 77.6% of those recur in the next frame within 0.3 m, which is what makes them furniture
rather than noise; this module's numbers have to be the same quantities, or the comparison is not one.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from empty_scene import coasting_share, frame_period, phantom_counts, since_person, time_since_person

RADIUS_M = 0.3


def phantom_counts_of(frames, **kwargs):
    """`phantom_counts` on frames that are adjacent in the recording, which is the common case in these tests."""
    return phantom_counts(frames, _adjacent(frames), **kwargs)


def _frames(*positions):
    """One (scores, xy) pair per frame, each detection given as (x, y, score); the frames are adjacent in the recording."""
    out = []
    for frame in positions:
        points = np.asarray(frame, dtype=np.float64).reshape(-1, 3)
        out.append((points[:, 2], points[:, :2]))
    return out


def _adjacent(frames):
    """Packed indices for frames that really do follow one another."""
    return np.arange(len(frames))


class TestPhantomCounts:
    def test_detections_per_frame_is_the_mean_over_frames(self):
        counts = phantom_counts_of(_frames([[0, 1, 0.9], [0, 2, 0.9]], [[0, 1, 0.9]], []), threshold=0.3, radius=RADIUS_M, seconds=1.0)

        assert counts["per_frame"] == 1.0
        assert counts["frames"] == 3
        assert counts["detections"] == 3

    def test_a_detection_under_the_threshold_is_not_counted(self):
        counts = phantom_counts_of(_frames([[0, 1, 0.9], [0, 2, 0.2]]), threshold=0.3, radius=RADIUS_M, seconds=1.0)

        assert counts["detections"] == 1

    def test_the_share_of_frames_holding_at_least_one_detection(self):
        counts = phantom_counts_of(_frames([[0, 1, 0.9]], [], [[0, 2, 0.9]], []), threshold=0.3, radius=RADIUS_M, seconds=1.0)

        assert counts["frames_with_any"] == 0.5

    def test_per_second_divides_by_the_recordings_own_elapsed_time(self):
        counts = phantom_counts_of(_frames([[0, 1, 0.9]], [[0, 1, 0.9]]), threshold=0.3, radius=RADIUS_M, seconds=0.5)

        assert counts["per_second"] == 4.0

    def test_a_detection_in_the_same_place_next_frame_recurs(self):
        counts = phantom_counts_of(_frames([[0, 1, 0.9]], [[0.1, 1.05, 0.9]]), threshold=0.3, radius=RADIUS_M, seconds=1.0)

        assert counts["recurs"] == 1.0

    def test_a_detection_that_moves_further_than_the_radius_does_not_recur(self):
        counts = phantom_counts_of(_frames([[0, 1, 0.9]], [[0, 3, 0.9]]), threshold=0.3, radius=RADIUS_M, seconds=1.0)

        assert counts["recurs"] == 0.0

    def test_recurrence_looks_backwards_so_the_first_frame_is_not_counted(self):
        counts = phantom_counts_of(_frames([[0, 1, 0.9]], [[0, 1, 0.9]], [[0, 1, 0.9]]), threshold=0.3, radius=RADIUS_M, seconds=1.0)

        assert counts["recur_of"] == 2

    def test_a_frame_with_nothing_breaks_the_chain_without_crashing(self):
        counts = phantom_counts_of(_frames([[0, 1, 0.9]], [], [[0, 1, 0.9]]), threshold=0.3, radius=RADIUS_M, seconds=1.0)

        assert counts["recurs"] == 0.0
        assert counts["recur_of"] == 1

    def test_nothing_anywhere_reports_zeros_rather_than_dividing_by_zero(self):
        counts = phantom_counts_of(_frames([], []), threshold=0.3, radius=RADIUS_M, seconds=1.0)

        assert counts["per_frame"] == 0.0 and counts["frames_with_any"] == 0.0
        assert np.isnan(counts["recurs"])

    def test_no_frames_at_all_reports_zeros(self):
        counts = phantom_counts([], np.zeros(0, dtype=int), threshold=0.3, radius=RADIUS_M, seconds=0.0)

        assert counts["frames"] == 0 and counts["per_frame"] == 0.0 and counts["per_second"] == 0.0


class TestAdjacency:
    """A person-free frame's neighbour in the *recording* is what "recurs next frame" means. The scored frames are interleaved
    with populated ones the split drops, so two consecutive scored frames can be minutes apart and pairing them would invent a
    recurrence -- the same trap `memory/interpreting-evaluation.md` records for per-second rates."""

    def test_frames_that_are_not_adjacent_in_the_recording_are_not_paired(self):
        frames = _frames([[0, 1, 0.9]], [[0, 1, 0.9]])

        counts = phantom_counts(frames, np.array([10, 400]), threshold=0.3, radius=RADIUS_M, seconds=1.0)

        assert counts["recur_of"] == 0
        assert np.isnan(counts["recurs"])

    def test_only_the_adjacent_pairs_count(self):
        frames = _frames([[0, 1, 0.9]], [[0, 1, 0.9]], [[0, 1, 0.9]])

        counts = phantom_counts(frames, np.array([5, 6, 99]), threshold=0.3, radius=RADIUS_M, seconds=1.0)

        assert counts["recur_of"] == 1 and counts["recurs"] == 1.0

    def test_the_frames_need_not_arrive_in_order(self):
        """`evaluate` plays several recordings at once and appends one entry per stream per step, so its frames arrive
        interleaved: without sorting, two entries that follow one another in the list are from different recordings."""
        frames = _frames([[0, 1, 0.9]], [[9, 9, 0.9]], [[0.1, 1.0, 0.9]], [[9, 9, 0.9]])

        interleaved = phantom_counts(frames, np.array([0, 500, 1, 501]), threshold=0.3, radius=RADIUS_M, seconds=1.0)
        ordered = phantom_counts([frames[0], frames[2], frames[1], frames[3]], np.array([0, 1, 500, 501]),
                                 threshold=0.3, radius=RADIUS_M, seconds=1.0)

        assert interleaved["recur_of"] == 2 and interleaved["recurs"] == 1.0
        assert interleaved == ordered

    def test_the_detections_themselves_are_still_all_counted(self):
        frames = _frames([[0, 1, 0.9]], [[0, 2, 0.9]])

        counts = phantom_counts(frames, np.array([0, 500]), threshold=0.3, radius=RADIUS_M, seconds=1.0)

        assert counts["detections"] == 2 and counts["per_frame"] == 1.0


class TestSincePerson:
    """A detection on a person-free frame can be a *stale memory of a real person*, which a static detector cannot produce: the
    three raw recordings interleave populated and person-free frames, and the slots fade over 10 s. Counting without this split
    would charge the memory for remembering someone who walked out of view two seconds ago."""

    def test_the_count_rises_with_each_frame_since_a_person_was_in_view(self):
        populated = np.array([True, False, False, False])
        time = np.array([0.0, 0.5, 1.0, 1.5])

        assert since_person(populated, time).tolist() == [0.0, 0.5, 1.0, 1.5]

    def test_a_populated_frame_resets_it(self):
        populated = np.array([True, False, True, False])
        time = np.array([0.0, 1.0, 2.0, 3.0])

        assert since_person(populated, time).tolist() == [0.0, 1.0, 0.0, 1.0]

    def test_before_anyone_has_appeared_it_is_infinite(self):
        populated = np.array([False, False, True])
        time = np.array([0.0, 1.0, 2.0])

        since = since_person(populated, time)
        assert np.isinf(since[:2]).all() and since[2] == 0.0

    def test_a_recording_with_nobody_in_it_is_infinite_throughout(self):
        since = since_person(np.zeros(3, dtype=bool), np.arange(3.0))

        assert np.isinf(since).all()

    def test_one_recordings_people_never_count_for_the_next(self):
        split = SimpleNamespace(offsets=np.array([0, 3]), lengths=np.array([3, 3]),
                                data=SimpleNamespace(time=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])))
        populated = np.array([True, False, False, False, False, False])

        since = time_since_person(split, populated)

        assert since[:3].tolist() == [0.0, 1.0, 2.0]
        assert np.isinf(since[3:]).all()


class TestFramePeriod:
    """Detections per second must be `frames * median frame period`, never the span between scored frames: the person-free
    frames are interleaved with populated ones the split drops, so consecutive scored frames can be minutes apart and a span
    would bill the detector for time it was never asked about (`memory/interpreting-evaluation.md`)."""

    def test_the_period_comes_from_the_recording_not_from_the_gaps_between_scored_frames(self):
        time = np.array([0.0, 0.04, 0.08, 0.12, 60.0, 60.04])         # a pause in the middle, as a dropped populated stretch
        split = SimpleNamespace(offsets=np.array([0]), lengths=np.array([6]), data=SimpleNamespace(time=time))

        assert frame_period(split) == pytest.approx(0.04)

    def test_each_recording_contributes_its_own_intervals(self):
        time = np.array([0.0, 0.1, 0.2, 0.0, 0.1, 0.2])
        split = SimpleNamespace(offsets=np.array([0, 3]), lengths=np.array([3, 3]), data=SimpleNamespace(time=time))

        assert frame_period(split) == pytest.approx(0.1)


class TestWhereThePhantomsComeFrom:
    """a memory frame's detections are rescored candidates first, then coasting reports, and the two mean
    different things.

    The memory cuts person-free false positives 72%, and the person-free work all reads the survivors as **rescored
    candidates** -- a scoring question. If they are instead **coasting reports**, the memory is inventing detections with
    no candidate under them, and the creation-rule work is aimed at a population that is not there. Coasting
    fires ~0.001 times per scored frame on the populated split, which is why that reading is plausible, but it has never
    been measured where nobody is.
    """

    @staticmethod
    def _frames_and_split(*frames):
        """Each frame as (detections, n_rescored_candidates); detections are (x, y, score) with candidates listed first."""
        rows, counts = [], []
        for detections, n in frames:
            points = np.asarray(detections, dtype=np.float64).reshape(-1, 3)
            rows.append((points[:, 2], points[:, :2]))
            counts.append(n)
        return rows, np.asarray(counts)

    def test_with_no_coasting_every_phantom_is_a_rescored_candidate(self):
        rows, counts = self._frames_and_split(([[0, 1, 0.9], [0, 2, 0.9]], 2), ([[0, 1, 0.9]], 1))

        share = coasting_share(rows, counts, threshold=0.3)

        assert share["from_candidates"] == 3
        assert share["from_coasting"] == 0
        assert share["coasting_share"] == 0.0

    def test_a_coasting_report_is_counted_as_one(self):
        """The frame holds one rescored candidate and one slot the memory reported with nothing under it."""
        rows, counts = self._frames_and_split(([[0, 1, 0.9], [5, 5, 0.8]], 1),)

        share = coasting_share(rows, counts, threshold=0.3)

        assert share["from_candidates"] == 1
        assert share["from_coasting"] == 1
        assert share["coasting_share"] == 0.5

    def test_the_threshold_applies_to_each_population_separately(self):
        rows, counts = self._frames_and_split(([[0, 1, 0.9], [5, 5, 0.1]], 1), ([[0, 1, 0.1], [5, 5, 0.9]], 1))

        share = coasting_share(rows, counts, threshold=0.3)

        assert share["from_candidates"] == 1        # the 0.9 candidate; the 0.1 one is under
        assert share["from_coasting"] == 1          # the 0.9 coasting report; the 0.1 one is under

    def test_the_split_is_exhaustive_so_it_can_be_trusted(self):
        """Whatever the threshold, the two populations must add up to what `phantom_counts` counts in total."""
        rows, counts = self._frames_and_split(([[0, 1, 0.9], [5, 5, 0.4], [2, 2, 0.2]], 2), ([], 0), ([[1, 1, 0.5]], 0))

        for threshold in (0.1, 0.3, 0.45, 0.95):
            share = coasting_share(rows, counts, threshold=threshold)
            total = phantom_counts(rows, _adjacent(rows), threshold, RADIUS_M, 1.0)

            assert share["from_candidates"] + share["from_coasting"] == total["detections"]

    def test_a_frame_with_nothing_is_not_a_division_by_zero(self):
        rows, counts = self._frames_and_split(([], 0), ([], 0))

        share = coasting_share(rows, counts, threshold=0.3)

        assert share["from_coasting"] == 0
        assert share["coasting_share"] == 0.0
