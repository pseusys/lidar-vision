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

from inspect import signature

from takheliped.detectors.takheliped import ObjectMemory

from empty_scene import (
    PEOPLE_FROM, build_parser, calibration_override, coasting_share, counted_frames, frame_period, load_person_free,
    main, phantom_counts, populated_frames, since_person, time_since_person,
)

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


class TestWhichPartitionItCounts:
    """The script was written for one partition and hardcoded it, which made the comparison it exists to serve impossible.

    `transferred` is the only partition whose **test** split is person-free, and its train and val splits are *identical to
    `official`'s* -- so nothing trained on it has ever seen a room with nobody in it. `balanced` is the one partition with
    person-free frames in **every** split, which makes the pair the experiment: the same script, the same recording's empty
    frames, one detector that trained on empty rooms and one that did not.

    Two things have to hold for that comparison to mean anything. The partition must reach the **loader** -- a job that
    forgets it trains or scores somewhere else and succeeds anyway. And on a partition that interleaves them, only the
    **person-free** frames may be counted: a detection on a populated frame can be a person, and counting it as a phantom
    would hide the effect under the thing being measured.
    """

    @staticmethod
    def _split(populated, recordings=1):
        """A packed split whose labels say which frames hold a person, as a partition with frame rule `all` does."""
        populated = np.asarray(populated, dtype=bool)
        n = len(populated) // recordings
        return SimpleNamespace(
            offsets=np.arange(recordings) * n, lengths=np.full(recordings, n),
            data=SimpleNamespace(gt_valid=populated[:, None], time=np.arange(len(populated)) * 0.025,
                                 annotated=np.ones(len(populated), dtype=bool)))

    def test_the_partition_and_the_split_are_both_flags(self):
        flags = {action.dest: action for action in build_parser()._actions}

        assert tuple(flags["frog_mode"].choices) == ("official", "transferred", "balanced")
        assert tuple(flags["split"].choices) == ("train", "val", "test")

    def test_the_defaults_are_what_every_published_row_was_measured_on(self):
        """`transferred` test, 72,533 person-free frames: changing that default would silently restate published numbers."""
        flags = {action.dest: action for action in build_parser()._actions}

        assert flags["frog_mode"].default == "transferred"
        assert flags["split"].default == "test"

    def test_the_flag_reaches_the_loader(self, monkeypatch):
        """The failure this guards is silent: a mode that does not arrive loads `official` and reports a clean run."""
        seen = {}

        def fake_load_split(name, limit, stage2, device, mode):
            seen.update(name=name, limit=limit, stage2=stage2, device=device, mode=mode)
            return self._split([True, False, False])

        monkeypatch.setattr("empty_scene.load_split", fake_load_split)

        load_person_free("val", "balanced", 2, "ckpt", "cpu")

        assert seen["mode"] == "balanced"
        assert seen["name"] == "val" and seen["limit"] == 2

    def test_the_loader_also_hands_back_who_was_in_view(self, monkeypatch):
        """The stale-memory split needs it, and it has to describe the partition just loaded rather than another one."""
        monkeypatch.setattr("empty_scene.load_split",
                            lambda *a, **kw: self._split([True, False, False, True]))

        split, populated = load_person_free("test", "balanced", 0, "ckpt", "cpu")

        assert populated.tolist() == [True, False, False, True]
        assert len(populated) == int(split.lengths.sum())

    def test_main_reads_the_partition_off_the_arguments(self):
        """Asserted on the reference rather than by running it: `main` loads checkpoints and replays a dataset."""
        assert "frog_mode" in main.__code__.co_names
        assert "split" in main.__code__.co_names


class TestWhereThePeopleAreRead:
    def test_a_split_that_carries_its_own_people_is_read_from_its_labels(self):
        """`balanced`'s frame rule is `all`, so its own annotations say which frames hold someone."""
        split = TestWhichPartitionItCounts._split([True, False, False, True])

        assert populated_frames(split, "balanced", "test", 0).tolist() == [True, False, False, True]

    def test_only_the_person_free_only_split_needs_a_sibling(self):
        """`transferred` test carries no annotations at all -- every frame holds nobody -- so when a person was last in
        view cannot be read from it, and comes from the populated split over the same three recordings."""
        assert PEOPLE_FROM == {("transferred", "test"): "val"}

    def test_the_sibling_is_read_at_the_same_partition(self, monkeypatch):
        asked = {}

        class FakeDataset:
            def __init__(self, split, mode, **kwargs):
                asked.update(split=split, mode=mode)
                self.scans = [np.zeros((4, 720))]
                self.det_id = [[1, 3]]

        monkeypatch.setattr("empty_scene.FROG_Dataset", FakeDataset)
        split = TestWhichPartitionItCounts._split([False, False, False, False])

        mask = populated_frames(split, "transferred", "test", 0)

        assert asked == {"split": "val", "mode": "transferred"}
        assert mask.tolist() == [False, True, False, True]


class TestOnlyThePersonFreeFramesAreCounted:
    def test_a_populated_frame_is_not_counted_as_a_phantom(self):
        """On `balanced` the scored frames interleave both, and a detection on a populated one can be a person."""
        index = np.array([0, 1, 2, 3])
        populated = np.array([True, False, True, False])

        assert counted_frames(index, populated, "balanced", "test").tolist() == [False, True, False, True]

    def test_frames_are_selected_by_their_packed_position_not_their_order(self):
        """`evaluate` plays recordings as parallel streams, so the scored frames arrive interleaved."""
        index = np.array([7, 2, 5])
        populated = np.zeros(8, dtype=bool)
        populated[[2, 7]] = True

        assert counted_frames(index, populated, "balanced", "val").tolist() == [False, False, True]

    def test_a_person_free_split_keeps_every_frame_it_scored(self):
        index = np.arange(5)

        assert counted_frames(index, np.zeros(5, dtype=bool), "transferred", "test").all()

    def test_a_partition_with_no_person_free_frame_is_refused_rather_than_reported_as_zero(self):
        """`official` is 100% populated in every split. Counting there would print 0.000 phantoms per frame and read as
        a result, when the truth is that the question was not asked."""
        with pytest.raises(SystemExit, match="official"):
            counted_frames(np.arange(3), np.ones(3, dtype=bool), "official", "test")


class TestComparingACalibratedArmWithAnUncalibratedOne:
    """Every row this script prints is read at a fixed threshold, and the two arms of the person-free experiment are
    not on the same scale: `calibrate_rescore.py` fitted a temperature and bias into the `official` runs' stage 3 and
    the `balanced` run has none. How much that matters was measured when the pair was written in -- the same
    checkpoints' phantom count fell 11.8% at threshold 0.3 and rose 8x at 0.9 -- so comparing across it would price
    the calibration and call it the training data. Stage 2 was never calibrated, so only the memory rows need this.
    """

    def test_the_calibration_can_be_ignored_so_both_arms_read_on_one_scale(self):
        assert calibration_override(True) == {"calibration": (1.0, 0.0)}

    def test_what_it_substitutes_is_the_identity_that_an_uncalibrated_checkpoint_already_gets(self):
        """`(logit + bias) / temperature` with (1.0, 0.0) is the logit, and it is `ObjectMemory`'s own default --
        which is what makes this equal to the uncalibrated arm rather than merely close to it."""
        assert signature(ObjectMemory).parameters["calibration"].default == (1.0, 0.0)

    def test_it_is_off_by_default_so_a_checkpoint_scores_as_it_was_published(self):
        assert calibration_override(False) == {}
        assert {action.dest: action for action in build_parser()._actions}["no_calibration"].default is False

    def test_main_passes_the_override_rather_than_editing_the_checkpoint(self):
        """The alternative -- stripping the key from the checkpoint -- would destroy a fitted pair that cost 55
        minutes a seed, for a comparison that needs it only in memory."""
        assert "calibration_override" in main.__code__.co_names
