"""Where `utils/slot_experiments.py` writes its results (`TODO.md` A51).

The path used to be derived from the experiment alone, so re-running a sweep on a second checkpoint silently overwrote the first
run's numbers: on 2026-10-02 the `odometry` and `cost` modes destroyed E2's and E4's JSON, which survived only in the logs. The
caller must be able to name the file.
"""
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from slot_experiments import EXPERIMENTS, output_path


class TestOutputPath:
    def test_each_experiment_has_its_own_file_by_default(self, tmp_path):
        paths = {output_path(Namespace(experiment=name, out_dir=tmp_path, out=None)) for name in EXPERIMENTS}

        assert len(paths) == len(EXPERIMENTS)

    def test_the_default_keeps_the_name_the_recorded_results_were_written_under(self, tmp_path):
        assert output_path(Namespace(experiment="odometry", out_dir=tmp_path, out=None)) == tmp_path / "a51_odometry.json"

    def test_an_explicit_out_is_used_as_given(self, tmp_path):
        chosen = tmp_path / "elsewhere" / "s3_drift.json"

        assert output_path(Namespace(experiment="odometry", out_dir=tmp_path, out=chosen)) == chosen

    @pytest.mark.parametrize("name", list(EXPERIMENTS))
    def test_every_experiment_is_reachable_by_name(self, name):
        assert callable(EXPERIMENTS[name])


class TestTheSlotCapacityReplay:
    """`TODO.md` item 50: 256 slots is the last number in the design nobody has priced.

    It is a **replay**, like `sigma` and `cost`: `object_memory(..., capacity=k)` rebuilds a trained memory at another
    slot count with its weights untouched, so the sweep costs minutes instead of an arm apiece. Read a flat curve
    weakly -- the weights adapted to 256 -- but read a *cliff* strongly, because a capacity the model genuinely needs
    cannot be taken away without one.
    """

    def test_the_experiment_is_registered(self):
        assert "slots" in EXPERIMENTS

    def test_the_capacities_bracket_the_adopted_count(self):
        from slot_experiments import CAPACITIES

        assert 256 in CAPACITIES, "the adopted count has to be in the sweep or there is nothing to compare against"
        assert min(CAPACITIES) < 256 < max(CAPACITIES), "a sweep that only shrinks cannot show 256 is too small"
        assert CAPACITIES == tuple(sorted(CAPACITIES)), "report them in order"
