"""Weight averaging of one stage-2 run's periodic checkpoints (`utils/average_checkpoints.py`), part of stage 2's recipe since the
owner's call of 2026-10-06 (`TODO.md` A59): the average of the epochs in a window, the window chosen on val, saved as a checkpoint
that loads wherever `step2_calibration.best.pth` does."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from average_checkpoints import WINDOWS, average_window, window_members
from follow_the_drow.detectors.takheliped import CalibrationNetwork
from train_takheliped import PeriodicCheckpoint, calibration_network


def _run(directory, epochs, args=None):
    """A run directory holding one periodic checkpoint per epoch, each a small network whose every weight equals its epoch."""
    angles = np.linspace(-np.pi / 2, np.pi / 2, 48, endpoint=False).astype(np.float32)
    keeper = PeriodicCheckpoint(directory, "step2_calibration", 0.25)
    for epoch in epochs:
        net = CalibrationNetwork(torch.from_numpy(angles), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1)
        state = {k: torch.full_like(v, epoch) if v.is_floating_point() else v for k, v in net.state_dict().items()}
        keeper.due = epoch
        keeper.offer(epoch, {"model": state, "epoch": epoch, "args": args or {}})
    return angles


class TestWindows:

    def test_a_window_holds_the_epochs_inside_it_and_ends_inclusive(self):
        assert window_members([0.25, 0.5, 0.75, 1.0, 1.25, 2.0], (0.5, 1.0)) == [0.5, 0.75, 1.0]

    def test_the_windows_are_the_ones_the_recipe_was_measured_with(self):
        # the six windows that chose 82.20% (TODO.md A59); changing them changes the recipe
        assert WINDOWS == ((0.5, 1.0), (0.75, 1.25), (0.5, 1.5), (1.0, 2.0), (0.5, 2.0), (0.25, 2.25))


class TestAverageWindow:

    def test_the_average_is_the_mean_of_the_members_weights(self, tmp_path):
        _run(tmp_path, [0.25, 0.5, 0.75, 1.0])
        checkpoint = average_window(tmp_path, (0.5, 1.0))
        weight = next(v for k, v in checkpoint["model"].items() if k.endswith("weight"))
        assert torch.allclose(weight, torch.full_like(weight, 0.75))
        assert checkpoint["averaged_epochs"] == [0.5, 0.75, 1.0]

    def test_it_loads_as_the_network_it_was_trained_as(self, tmp_path):
        angles = _run(tmp_path, [0.5, 1.0], args={"max_range_m": "15.0"})
        checkpoint = average_window(tmp_path, (0.5, 1.0))
        net = calibration_network(checkpoint, angles, channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1)
        assert net.max_range_m == 15.0
