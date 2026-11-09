"""Stage 2's restricted scoring column, so a model trained at one range limit can be read against another.

Item 82 built this for stage 3: a detector trained at DROW's 29.99 m is scored again with everything past 10 m
removed, under both readings of what "restricted" means -- labels alone, and labels with the detections that
cannot exist in a 10 m model. Stage 2 had no such column, which left the opposite arm unreadable: a chain
*trained* at 10 m on DROW scores every label including the 7.3% past its own limit, and is charged for people it
cannot represent.

The two assertions that matter are the ones that keep the existing numbers: an unset flag changes nothing, and
the restricted pass re-scores the frames the run already produced rather than running the network again.
"""
import sys
from pathlib import Path

import numpy as np

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import torch  # noqa: E402
from takheliped.detectors.takheliped import CalibrationNetwork  # noqa: E402
from train_takheliped import (FrameSplit, build_parser, calibration_frames, evaluate_calibration,  # noqa: E402
                              restricted_columns, score_detections)


def frames():
    """Two frames: a person at 4 m found, a person at 20 m found, and one false positive at 25 m."""
    near = (np.array([0.9]), np.array([[0.0, 4.0]]), np.array([[0.0, 4.0]]))
    far = (np.array([0.8, 0.7]), np.array([[0.0, 20.0], [0.0, 25.0]]), np.array([[0.0, 20.0]]))
    return [near, far]


class TestTheFlagIsThereAndOffByDefault:
    def test_stage_two_takes_the_same_flag_stage_three_does(self):
        assert build_parser().parse_args(["step2"]).eval_max_range_m == 0.0
        assert build_parser().parse_args(["step2", "--eval-max-range-m", "10"]).eval_max_range_m == 10.0

    def test_the_scoring_pass_item_82_built_still_has_it(self):
        """`score` is where the whole detector's restricted column lives; stage 2 now has the same flag name,
        so the two reports are read the same way rather than through two spellings."""
        assert build_parser().parse_args(["score", "--stage2", "x.pth", "--memory", "y.pth",
                                          "--eval-max-range-m", "10"]).eval_max_range_m == 10.0


class TestTheRestrictedColumnsTakeWhateverSidesTheyAreGiven:
    def test_a_single_sided_report_is_allowed_because_stage_two_has_one_side(self):
        columns = restricted_columns({"stage2": frames()}, 10.0)
        assert set(columns) == {"within_10m_labels", "within_10m_labels_and_detections"}
        assert set(columns["within_10m_labels"]) == {"stage2"}

    def test_stage_threes_two_sides_are_unchanged(self):
        played = {"candidates": frames(), "memory": frames()}
        columns = restricted_columns(played, 10.0)
        assert list(columns["within_10m_labels"]) == ["candidates", "memory"]

    def test_the_far_person_leaves_and_the_far_false_positive_can_stay_or_go(self):
        columns = restricted_columns({"stage2": frames()}, 10.0)
        labels = columns["within_10m_labels"]["stage2"]
        both = columns["within_10m_labels_and_detections"]["stage2"]
        assert labels["people"] == both["people"] == 1, "only the 4 m person is in range"
        assert labels["fp_per_frame"] > both["fp_per_frame"], "dropping the detections drops the far ones too"

    def test_an_unrestricted_score_is_what_it_always_was(self):
        """The unrestricted row must not move when the flag exists but is unset."""
        assert score_detections(frames())["people"] == 2


class TestTheScoringPassStillRunsWithoutGradients:
    """Found by a smoke run, not by this suite: splitting `evaluate_calibration` left its `@torch.no_grad()` on
    the wrapper while the loop that calls `.numpy()` moved into `calibration_frames`, which raised on the test
    pass of **every** stage-2 run. The suite only ever reached the decorated wrapper."""

    def _split(self):
        scans = np.full((2, 48), 3.0, dtype=np.float32)
        people = [np.array([[3.0, 0.0]])] * 2
        xy = [np.array([[0.0, 3.0]])] * 2
        return FrameSplit(scans, np.zeros((2, 3), np.float32), np.zeros(2, np.int64), np.arange(2), people, xy,
                          [()] * 2, np.arange(2) / 40.0)

    def _net(self):
        angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
        return CalibrationNetwork(torch.from_numpy(angles), channels=(8, 16, 32), blocks=(1, 1, 1),
                                  bottleneck_blocks=1)

    def test_the_frames_come_back_outside_a_no_grad_block(self):
        frames = calibration_frames(self._net().train(), self._split(), np.arange(2), "cpu", 2)
        assert len(frames) == 2

    def test_the_wrapper_and_the_frames_agree(self):
        net, split = self._net(), self._split()
        assert evaluate_calibration(net, split, np.arange(2), "cpu", 2) == score_detections(
            calibration_frames(net, split, np.arange(2), "cpu", 2))
