"""What a slot creation/confirmation rule would actually remove.

The memory cuts person-free false positives from 2.622 per frame to 0.740, and **what it keeps is furniture**: its surviving
phantoms recur in the next frame 92.2% of the time against the candidates' 84.9% and real people's 99.8%. The creation-rule work proposes a
higher creation threshold or confirmation over several frames, on the reading that this is a creation-rule question.

The reading deserves a test rather than an argument, because a confirmation rule filters **young** slots and furniture is the
opposite of young: a chair sits in the room for the whole recording, so its slot is among the oldest and highest-hit-rate in
the memory. If the slots backing the surviving phantoms are old, no creation or confirmation rule can reach them, and the only
thing such a rule would buy is a delay on genuinely new people. This module measures which it is.

The phantom audit is what makes the question well-posed: 99.997% of those phantoms are **rescored candidates** rather than coasting
reports, so each one has a slot matched to it whose age and hit rate can be read.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from slot_confirmation import PHANTOM, PERSON, confirmation_curve, populations, slot_age_frames


class TestSlotAgeInFrames:
    def test_seconds_become_frames_at_the_sensors_own_period(self):
        """FROG delivers a scan every 25 ms, so a slot one second old has been matched across 40 frames."""
        assert slot_age_frames(np.array([1.0]), period_s=0.025)[0] == pytest.approx(40.0)

    def test_a_slot_born_this_frame_is_zero_frames_old(self):
        assert slot_age_frames(np.array([0.0]), period_s=0.025)[0] == 0.0

    def test_the_period_must_be_positive_or_the_conversion_is_meaningless(self):
        with pytest.raises(ValueError):
            slot_age_frames(np.array([1.0]), period_s=0.0)


class TestConfirmationCurve:
    @staticmethod
    def _detections(*rows):
        """Each detection as (slot age in frames, PHANTOM or PERSON)."""
        ages = np.array([age for age, _ in rows], dtype=np.float64)
        population = np.array([pop for _, pop in rows], dtype=np.int8)
        return ages, population

    def test_requiring_one_frame_removes_nothing(self):
        ages, population = self._detections((0.0, PHANTOM), (5.0, PERSON))

        row = confirmation_curve(ages, population, k=1)

        assert row["phantoms_removed"] == 0
        assert row["people_removed"] == 0

    def test_a_rule_removes_exactly_the_slots_with_fewer_than_k_frames_of_evidence(self):
        """`k` counts the creation frame, so a slot of age `k - 1` has seen `k` frames and may report. That convention is
        what makes `delay_frames` equal `k - 1`: a rule of k = 1 is free, and the boundary sits at age k - 1."""
        ages, population = self._detections((0.0, PHANTOM), (1.0, PHANTOM), (2.0, PHANTOM), (9.0, PHANTOM))

        row = confirmation_curve(ages, population, k=3)

        assert row["phantoms_removed"] == 2        # ages 0 and 1 have 1 and 2 frames of evidence
        assert row["phantoms_kept"] == 2           # age 2 has exactly 3, and age 9 more than enough

    def test_the_two_populations_are_counted_separately(self):
        ages, population = self._detections((0.0, PHANTOM), (0.0, PERSON), (50.0, PHANTOM), (50.0, PERSON))

        row = confirmation_curve(ages, population, k=4)

        assert (row["phantoms_removed"], row["people_removed"]) == (1, 1)
        assert (row["phantoms_kept"], row["people_kept"]) == (1, 1)

    def test_the_shares_are_of_each_population_not_of_the_whole(self):
        ages, population = self._detections((0.0, PHANTOM), (99.0, PHANTOM), (99.0, PHANTOM), (0.0, PERSON))

        row = confirmation_curve(ages, population, k=2)

        assert row["phantom_share_removed"] == pytest.approx(1 / 3)
        assert row["person_share_removed"] == pytest.approx(1.0)

    def test_the_delay_is_k_minus_one_frames_because_the_first_frame_is_free(self):
        ages, population = self._detections((7.0, PERSON),)

        assert confirmation_curve(ages, population, k=1)["delay_frames"] == 0
        assert confirmation_curve(ages, population, k=5)["delay_frames"] == 4

    def test_a_rule_that_removes_more_people_than_phantoms_is_reported_not_hidden(self):
        """The outcome this measurement exists to catch: furniture is old, so a youth filter can hit people harder."""
        ages, population = self._detections((500.0, PHANTOM), (500.0, PHANTOM), (1.0, PERSON))

        row = confirmation_curve(ages, population, k=10)

        assert row["phantom_share_removed"] == 0.0
        assert row["person_share_removed"] == 1.0

    def test_no_detections_reports_zeros_rather_than_dividing_by_zero(self):
        row = confirmation_curve(np.zeros(0), np.zeros(0, dtype=np.int8), k=3)

        assert row["phantoms_removed"] == 0
        assert row["phantom_share_removed"] == 0.0
        assert row["person_share_removed"] == 0.0


class TestLabellingTheTwoPopulations:
    """Which reported detections are phantoms and which are people, and what backs each one.

    A detection on a **person-free** frame is a phantom by construction -- that split holds no annotated people at all,
    which is why it exists. On a populated frame a detection near an annotated person is that person; a populated frame's
    *other* false positives are deliberately left out rather than pooled with the empty-room ones, because `docs/PAPER.md`
    argues precisely that the two hide each other when a ratio mixes them.
    """

    @staticmethod
    def _frames(*rows):
        """Each frame as (scores, xy, gt, backing slot ages)."""
        memory = [(np.array(s, dtype=float), np.array(xy, dtype=float).reshape(-1, 2), np.array(gt, dtype=float).reshape(-1, 2))
                  for s, xy, gt, _ in rows]
        return dict(memory=memory, slot_age=[np.array(age, dtype=float) for *_, age in rows])

    def test_every_detection_on_a_person_free_frame_is_a_phantom(self):
        frames = self._frames(([0.9, 0.8], [[0, 1], [0, 2]], [], [30.0, 40.0]))

        ages, labels = populations(frames, gt_radius_m=0.5, threshold=0.3)

        assert list(labels) == [PHANTOM, PHANTOM]
        assert sorted(ages) == [30.0, 40.0]

    def test_a_detection_near_an_annotated_person_is_that_person(self):
        frames = self._frames(([0.9], [[0, 4]], [[0, 4.1]], [12.0]))

        ages, labels = populations(frames, gt_radius_m=0.5, threshold=0.3)

        assert list(labels) == [PERSON]
        assert list(ages) == [12.0]

    def test_a_false_positive_on_a_populated_frame_is_left_out_not_pooled(self):
        """It is a third population -- an error beside real people -- and mixing it in would answer another question."""
        frames = self._frames(([0.9, 0.9], [[0, 4], [9, 9]], [[0, 4.0]], [12.0, 99.0]))

        ages, labels = populations(frames, gt_radius_m=0.5, threshold=0.3)

        assert list(labels) == [PERSON]
        assert list(ages) == [12.0]

    def test_detections_under_the_threshold_are_not_reported_at_all(self):
        frames = self._frames(([0.9, 0.1], [[0, 1], [0, 2]], [], [30.0, 40.0]))

        ages, _ = populations(frames, gt_radius_m=0.5, threshold=0.3)

        assert list(ages) == [30.0]

    def test_a_detection_with_no_backing_slot_is_dropped_rather_than_counted_as_age_zero(self):
        """A coasting report has no candidate under it, so it has no matched slot -- and age 0 would read as 'brand new',
        which is the opposite of what a long-lived coasting slot is."""
        frames = self._frames(([0.9, 0.9], [[0, 1], [0, 2]], [], [np.nan, 40.0]))

        ages, labels = populations(frames, gt_radius_m=0.5, threshold=0.3)

        assert list(ages) == [40.0]
        assert len(labels) == 1

    def test_no_frames_gives_empty_arrays_rather_than_an_error(self):
        ages, labels = populations(dict(memory=[], slot_age=[]), gt_radius_m=0.5, threshold=0.3)

        assert len(ages) == 0 and len(labels) == 0
