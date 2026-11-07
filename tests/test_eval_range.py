"""Item 82's 10 m column: scoring one replay against a restricted label set, without re-running the detector.

Decision 1 of 2026-11-04 asked for the 29.99 m figure to carry a 10 m-restricted column beside it, and the
reason it is computed this way matters. **The memory must run on every candidate it was trained to see.**
Dropping far candidates before `evaluate` plays them would change what competes for slots -- a different run,
not a restricted score of the same one -- so the restriction is applied to the frames `evaluate` hands back
and `score_detections` is called again on those. One replay, several scorings.

**Two readings, both reported, because the decision does not say which.** Truncating the *labels* alone is what
"ground truth truncated at 10 m" literally asks for, and it charges the detector for every detection it makes
further out. Truncating the detections too is what the 10 m-trained baselines effectively do, since a 10 m model
cannot place a vote past its own limit. They answer different questions and the gap between them is the cost of
the 29.99 m choice, so the column is only readable with both.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import train_takheliped as tak  # noqa: E402


def frame(scores, xy, gt):
    """One frame as `evaluate`'s `frames_out` builds it: (scores `[D]`, detection xy `[D, 2]`, people `[G, 2]`)."""
    return (np.asarray(scores, np.float32), np.asarray(xy, np.float32).reshape(-1, 2),
            np.asarray(gt, np.float64).reshape(-1, 2))


#: Straight ahead, so a position's y is its range: 4 m and 16 m from the sensor.
NEAR, FAR = (0.0, 4.0), (0.0, 16.0)


class TestTheLabelsBeyondTheLimitAreDropped:
    def test_a_person_past_the_limit_leaves_the_scored_frame(self):
        out = tak.restrict_to_range([frame([0.9], [NEAR], [NEAR, FAR])], 10.0)
        assert out[0][2].tolist() == [list(NEAR)]

    def test_a_person_within_it_stays(self):
        out = tak.restrict_to_range([frame([0.9], [NEAR], [NEAR])], 10.0)
        assert out[0][2].tolist() == [list(NEAR)]

    def test_the_limit_itself_is_inside(self):
        """`<=`, matching `real_people`'s own rated-range filter, so one convention covers both."""
        out = tak.restrict_to_range([frame([0.9], [NEAR], [(0.0, 10.0)])], 10.0)
        assert len(out[0][2]) == 1

    def test_a_frame_left_with_nothing_annotated_is_still_a_frame(self):
        """It is the denominator of false positives per frame, so dropping it would deflate that rate."""
        out = tak.restrict_to_range([frame([0.9], [FAR], [FAR])], 10.0)
        assert len(out) == 1 and len(out[0][2]) == 0


class TestWhatHappensToTheDetections:
    def test_by_default_a_far_detection_is_kept_and_so_is_charged_as_a_false_positive(self):
        out = tak.restrict_to_range([frame([0.9, 0.8], [NEAR, FAR], [NEAR])], 10.0)
        assert out[0][0].tolist() == pytest.approx([0.9, 0.8])

    def test_asked_to_restrict_them_it_drops_the_far_one_with_its_score(self):
        out = tak.restrict_to_range([frame([0.9, 0.8], [NEAR, FAR], [NEAR])], 10.0, detections=True)
        assert out[0][0].tolist() == pytest.approx([0.9])
        assert out[0][1].tolist() == [list(NEAR)]

    def test_scores_and_positions_stay_paired_through_the_cut(self):
        out = tak.restrict_to_range([frame([0.1, 0.9, 0.5], [FAR, NEAR, FAR], [NEAR])], 10.0, detections=True)
        assert out[0][0].tolist() == pytest.approx([0.9]) and out[0][1].tolist() == [list(NEAR)]


class TestItIsAScoringPassAndNotARun:
    def test_the_frames_handed_in_are_not_modified(self):
        """`evaluate`'s frames are scored again at the full limit in the same pass, so a mutation would corrupt that."""
        frames = [frame([0.9, 0.8], [NEAR, FAR], [NEAR, FAR])]
        tak.restrict_to_range(frames, 10.0, detections=True)
        assert len(frames[0][0]) == 2 and len(frames[0][2]) == 2

    def test_a_limit_nothing_reaches_leaves_the_score_exactly_where_it_was(self):
        frames = [frame([0.9, 0.2], [NEAR, (0.0, 6.0)], [NEAR]), frame([0.7], [(1.0, 2.0)], [(1.0, 2.0)])]
        before = tak.score_detections(frames)
        after = tak.score_detections(tak.restrict_to_range(frames, 10.0, detections=True))
        assert after["ap"] == pytest.approx(before["ap"])
        assert after["people"] == before["people"] and after["frames"] == before["frames"]

    def test_restricting_both_ends_can_only_help_a_detector_that_reaches_too_far(self):
        """The whole point of the column: a far false positive is charged under one reading and not the other.

        Read off precision and false positives per frame rather than AP, because AP is **undefined** on a frame
        this small -- `_safe_auc` needs two points on the curve to integrate and returns NaN below that. The
        test below carries the AP claim on enough frames to have a curve.
        """
        frames = [frame([0.95, 0.9], [NEAR, FAR], [NEAR])]
        labels_only = tak.score_detections(tak.restrict_to_range(frames, 10.0))
        both = tak.score_detections(tak.restrict_to_range(frames, 10.0, detections=True))
        assert labels_only["fp_per_frame"] == pytest.approx(1.0) and labels_only["precision"] == pytest.approx(0.5)
        assert both["fp_per_frame"] == pytest.approx(0.0) and both["precision"] == pytest.approx(1.0)
        assert both["recall"] == pytest.approx(labels_only["recall"])

    def test_and_the_ap_moves_the_same_way_once_there_is_a_curve_to_integrate(self):
        far_and_confident = [frame([0.9, 0.8], [NEAR, FAR], [NEAR]),
                             frame([0.7, 0.6], [(1.0, 3.0), (2.0, 15.0)], [(1.0, 3.0)]),
                             frame([0.5, 0.4], [(0.0, 5.0), (0.0, 20.0)], [(0.0, 5.0)])]
        labels_only = tak.score_detections(tak.restrict_to_range(far_and_confident, 10.0))
        both = tak.score_detections(tak.restrict_to_range(far_and_confident, 10.0, detections=True))
        assert both["ap"] > labels_only["ap"]


class TestTheFlagReachesTheScoringCommand:
    def test_score_takes_an_evaluation_limit_that_defaults_to_off(self):
        args = tak.build_parser().parse_args(["score", "--stage2", "a.pth", "--memory", "b.pth"])
        assert args.eval_max_range_m == 0.0

    def test_it_is_a_scoring_flag_and_not_a_training_one(self):
        """Stage 2's `--max-range-m` shapes the network; this one only narrows what a finished run is scored on."""
        args = tak.build_parser().parse_args(["score", "--stage2", "a.pth", "--memory", "b.pth",
                                              "--eval-max-range-m", "10"])
        assert args.eval_max_range_m == 10.0
