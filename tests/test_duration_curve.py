"""Selecting a run's periodic checkpoints (`TODO.md` A50 phase 1 item 4).

`duration_curve.checkpoints` turns a run directory into the `(epoch, path)` pairs to score. Two things make it worth pinning:
the epoch is parsed back out of the filename `PeriodicCheckpoint` wrote, so the two must agree on the format; and a run that
stopped early simply has fewer checkpoints, which must shorten the curve rather than raise an error -- the dropout run stopped
at epoch 3.75 and so has no epoch-4.00 checkpoint, while the control does.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from duration_curve import checkpoints
from train_three_horizon import PeriodicCheckpoint


def _run(directory, epochs):
    """A run directory holding one periodic checkpoint per epoch, named exactly as `PeriodicCheckpoint` names them."""
    keeper = PeriodicCheckpoint(directory, "step2_calibration", 0.25)
    for epoch in epochs:
        keeper.due = epoch          # save at exactly these epochs, without replaying a whole training run
        keeper.offer(epoch, {"model": {}, "epoch": epoch})
    return directory


class TestCheckpoints:

    def test_it_reads_back_the_epochs_periodic_checkpoint_wrote(self, tmp_path):
        _run(tmp_path, [0.25, 1.0, 2.0])
        assert [epoch for epoch, _ in checkpoints(tmp_path, [])] == [0.25, 1.0, 2.0]

    def test_it_keeps_only_the_wanted_epochs_in_order(self, tmp_path):
        _run(tmp_path, [0.25, 0.5, 1.0, 2.0])
        assert [epoch for epoch, _ in checkpoints(tmp_path, [2.0, 0.25])] == [0.25, 2.0]

    def test_an_epoch_a_run_never_reached_is_absent_rather_than_an_error(self, tmp_path):
        # the dropout run stopped at 3.75, so it has no epoch-4.00 checkpoint while the control does
        _run(tmp_path, [1.0, 2.0, 3.0])
        assert [epoch for epoch, _ in checkpoints(tmp_path, [1.0, 2.0, 3.0, 4.0])] == [1.0, 2.0, 3.0]

    def test_an_empty_directory_yields_no_curve_rather_than_an_error(self, tmp_path):
        assert checkpoints(tmp_path, []) == []

    def test_the_best_checkpoint_is_not_mistaken_for_a_periodic_one(self, tmp_path):
        _run(tmp_path, [1.0])
        (tmp_path / "step2_calibration.best.pth").write_bytes(b"")
        assert [epoch for epoch, _ in checkpoints(tmp_path, [])] == [1.0]
