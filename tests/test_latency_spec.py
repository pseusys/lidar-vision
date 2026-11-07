"""The CPU latency sweep has to time the network on the data it was trained for.

Second instance of the defect `tests/test_averaging_spec.py` guards in the averaging script, and the one that was
about to bite: `run_latency` hardcoded `frog_laser_angles(720)` and `FROG_Dataset(split="test", mode="official")`,
so timing a 0.5 deg checkpoint would have fed a 360-beam network 720 beams. An aggregator-free calibration network
is fully convolutional over beams, so that does not raise -- it silently reports the time to process twice the
beams the checkpoint was trained on, which is the opposite of the saving the sweep exists to measure.

`ms/frame` on the CPU is the number this project is judged on (`AGENTS.md`), so a sweep that cannot be told which
sensor it is timing cannot answer the question it is asked. These tests pin the flags and the absence of a literal
grid; what the timings come out as is not something a CPU-only test can assert.
"""
import sys
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import tracking_baselines as tb  # noqa: E402
from train_takheliped import DROW, FROG, dataset_spec  # noqa: E402

def _run_latency_source() -> str:
    """`run_latency`'s body alone, so the sibling functions' deliberate FROG literals do not satisfy these tests."""
    source = Path(tb.__file__).read_text(encoding="utf-8")
    start = source.index("def run_latency(")
    return source[start:source.index("\ndef ", start + 1)]


LATENCY = _run_latency_source()


def parse(*argv):
    return tb.build_parser().parse_args(["latency", *argv])


class TestTheSweepCanBeToldWhichSensorItIsTiming:
    @pytest.mark.parametrize("flag,value,attribute", [("--dataset", "drow", "dataset"),
                                                      ("--frog-mode", "balanced", "frog_mode"),
                                                      ("--beam-stride", "2", "beam_stride")])
    def test_the_flag_reaches_the_parsed_arguments(self, flag, value, attribute):
        assert str(getattr(parse(flag, value), attribute)) == value

    def test_the_dataset_is_resolved_by_the_trainers_own_resolver(self):
        """A second copy of this resolution is how the two drift apart again. The import stays lazy because this
        module keeps torch out of module scope for its worker pools, so the call site is what can be checked."""
        assert "dataset_spec(args)" in LATENCY

    def test_a_decimated_checkpoint_is_timed_at_its_own_spacing(self):
        spec = dataset_spec(parse("--beam-stride", "2"))
        assert spec.beams == FROG.source_beams // 2 and spec.spacing_deg == 0.5

    def test_a_drow_checkpoint_is_timed_on_drow(self):
        spec = dataset_spec(parse("--dataset", "drow"))
        assert spec.name == "drow" and spec.beams == DROW.beams

    def test_frog_official_at_full_resolution_stays_the_default(self):
        """Every CPU figure on record was measured there; the defaults must reproduce them."""
        spec = dataset_spec(parse())
        assert spec.name == "frog" and spec.decimate == 1 and spec.beams == 720
        assert parse().frog_mode == "official"


class TestNoGridIsHardcodedWhereAFlagHasToReachIt:
    def test_run_latency_builds_its_grid_from_the_spec(self):
        assert "frog_laser_angles(720)" not in LATENCY and "spec.angles()" in LATENCY

    def test_run_latency_loads_the_split_through_the_spec_rather_than_naming_frog(self):
        assert "FROG_Dataset(" not in LATENCY and 'spec.load("test"' in LATENCY

    def test_the_scans_are_thinned_so_the_timed_input_has_the_beams_the_network_expects(self):
        """The grid following the flag is half the fix; the scans have to be decimated to match it."""
        assert "spec.thin(" in LATENCY

    def test_the_memorys_bearing_grid_follows_the_same_spec(self):
        """`laser_angles` maps a slot's bearing to a sector, and its default is FROG's full 720-beam grid."""
        assert "laser_angles(device, spec)" in LATENCY

    def test_the_baseline_tables_are_left_alone_because_frog_is_what_they_compare_against(self):
        """`run_arms` and `run_report` score published LFE detectors against ours on FROG. A dataset flag there
        would be meaningless, so their literal grids are correct and deliberately not parameterised."""
        source = Path(tb.__file__).read_text(encoding="utf-8")
        assert source.count("frog_laser_angles(720)") == 2
