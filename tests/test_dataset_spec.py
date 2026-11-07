"""The trainer's per-dataset parameter table, and what carries across datasets and what does not.

`train_takheliped.py` named `FROG_Dataset` and `frog_laser_angles(720)` in six places and derived its sector
count from a literal 720, so "which dataset" was not a thing a run could state -- it was a thing the file
assumed. The table here is the owner's asked-for per-dataset parameter block: beams, field of view, spacing,
rate, annotated share, people per annotated frame, classes and range limit, printed at the head of a run and
stored in the checkpoint, so a result can be read without knowing the command that made it.

Two things are pinned beyond the values. **FROG stays the default**, so every command and every number measured
before the flag existed is reproduced by the unchanged command; and **the candidate cache keeps its names**,
because 37 GB of built candidates are keyed by that filename and a changed key silently rebuilds all of them.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import evaluate as ev  # noqa: E402
import train_takheliped as tak  # noqa: E402
from takheliped.datasets.frog_dataset import frog_laser_angles  # noqa: E402


class TestTheTableSaysWhatTheDatasetIs:
    def test_both_datasets_this_project_trains_on_are_registered(self):
        assert set(tak.DATASETS) == {"frog", "drow"}

    def test_frog_is_the_default_so_an_unchanged_command_is_an_unchanged_run(self):
        assert tak.DEFAULT_DATASET == "frog"
        assert tak.DATASETS[tak.DEFAULT_DATASET] is tak.FROG

    @pytest.mark.parametrize("field,frog,drow", [
        ("beams", 720, 450),
        ("fov_deg", 180.0, 224.5),
        ("spacing_deg", 0.25, 0.5),
        ("rate_hz", 40.01, 12.70),
        ("max_range_m", 10.0, 29.99),
        ("annotated_share", 1.0, 0.05),
        ("people_per_annotated_frame", 3.06, 0.81),
        ("sectors", 90, 57),
    ])
    def test_the_measured_values_are_the_ones_dataset_properties_records(self, field, frog, drow):
        """Pinned against `memory/dataset-properties.md`, so a value cannot drift away from the page quoting it."""
        assert getattr(tak.DATASETS["frog"], field) == pytest.approx(frog)
        assert getattr(tak.DATASETS["drow"], field) == pytest.approx(drow)

    def test_drow_carries_its_three_classes_and_frog_its_one(self):
        assert tak.DATASETS["frog"].classes == ("pedestrian",)
        assert tak.DATASETS["drow"].classes == ("pedestrian", "wheelchair", "walker")

    def test_the_time_step_window_stays_in_frames_and_is_stated_per_dataset(self):
        """Measured on the real files: 32 intervals hold the dt error to 0.5-0.9% on DROW, where one costs 36%.

        Restating it in seconds would be `docs/PROPOSAL.md` principle 9 applied to the wrong quantity -- what is
        being averaged out is a stamp defect counted in frames, not an interval of real time, and FROG's 0.8 s
        equivalent is 10 DROW frames, which triples the error to 1.7%.
        """
        assert tak.DATASETS["frog"].dt_average_frames == 32
        assert tak.DATASETS["drow"].dt_average_frames == 32

    def test_the_printed_table_names_every_row_and_is_pure_ascii(self):
        for spec in tak.DATASETS.values():
            table = tak.parameter_table(spec)
            assert table.isascii(), table
            for row in ("beams", "field of view", "spacing", "rate", "annotated", "people", "classes", "range"):
                assert row in table.lower(), (spec.name, row)


class TestTheBeamGrid:
    def test_each_dataset_builds_its_own_angles_at_its_own_count(self):
        for name, spec in tak.DATASETS.items():
            angles = spec.angles()
            assert len(angles) == spec.beams, name
            assert np.rad2deg(angles[1] - angles[0]) == pytest.approx(spec.spacing_deg, abs=1e-4), name

    def test_frogs_angles_are_the_ones_every_frog_run_was_trained_on(self):
        assert np.array_equal(tak.DATASETS["frog"].angles(), frog_laser_angles(720))


class TestDecimatingFrogToDrowsSpacing:
    """Item 73's chain: the owner's common grid is DROW's 0.5 deg, met by halving FROG rather than interpolating DROW."""

    def test_it_halves_the_beams_and_doubles_the_spacing(self):
        half = tak.DATASETS["frog"].decimated(2)
        assert (half.beams, half.spacing_deg, half.sectors) == (360, 0.5, 45)

    def test_it_keeps_the_field_of_view_and_the_rate_it_decimates_nothing_about(self):
        half = tak.DATASETS["frog"].decimated(2)
        assert (half.fov_deg, half.rate_hz, half.max_range_m) == (180.0, 40.01, 10.0)

    def test_its_angles_are_every_second_beam_of_the_real_grid_not_a_resampling(self):
        """Taking every second real beam invents nothing; `docs/PROPOSAL.md` section 5.3's resampler would."""
        assert np.array_equal(tak.DATASETS["frog"].decimated(2).angles(), frog_laser_angles(720)[::2])

    def test_not_decimating_is_the_dataset_itself(self):
        assert tak.DATASETS["frog"].decimated(1) == tak.DATASETS["frog"]


class TestTheCandidateCacheKey:
    def test_frogs_existing_cache_names_are_unchanged(self):
        """37 GB of built candidates live under these names; a changed key rebuilds every one of them in silence."""
        assert tak.candidate_cache_path("train", 0, None).name == "lfe-peaks_official_train_f0.01_p64_s90.pkl"
        assert tak.candidate_cache_path("test", 2, None, "transferred").name == "lfe-peaks_transferred_test_f0.01_p64_s90_first2.pkl"

    def test_drow_takes_the_partition_slot_frog_uses_for_its_mode(self):
        """DROW has no partition of its own, so its name fills that slot -- and its 57 sectors key it apart anyway."""
        name = tak.candidate_cache_path("train", 0, None, spec=tak.DATASETS["drow"]).name
        assert name == "lfe-peaks_drow_train_f0.01_p64_s57.pkl"

    def test_a_decimated_chain_cannot_read_the_full_rate_chains_candidates(self):
        half = tak.candidate_cache_path("train", 0, None, spec=tak.DATASETS["frog"].decimated(2)).name
        assert half == "lfe-peaks_official-d2_train_f0.01_p64_s45.pkl"
        assert half != tak.candidate_cache_path("train", 0, None).name


class TestCorruptLabelsAreNotTargets:
    """DROW carries four annotations at 316-716 m -- in a care facility, with a 30 m sensor.

    `results/scan_anomalies_drow.json` found them and they are all in train. A vote regressed towards one is a
    target the beam it sits on cannot reach, and a classification target on a beam that reads 29.96 m of empty
    space. They are corrupt readings of the annotation tool, not distant people.

    The rule is the **sensor's** rated range, not the run's `--max-range-m`, and that distinction is the point:
    DROW's 3,165 labels between 10 m and 18 m are real people the owner chose to keep by training at 29.99 m,
    while these four are not people at all.
    """

    def test_a_label_past_the_sensors_rated_range_is_dropped(self):
        polar = np.array([[2.0, 0.1], [715.84, -0.883], [17.7, 0.2], [315.86, -0.23]])
        kept = tak.DATASETS["drow"].real_people(polar)
        assert kept.tolist() == [[2.0, 0.1], [17.7, 0.2]]

    def test_labels_the_owner_chose_to_keep_are_kept(self):
        """Past 10 m but inside the sensor: the 7.3% of DROW's labels a 10 m model cannot represent at all."""
        polar = np.array([[12.0, 0.0], [17.73, 0.0], [29.5, 0.0]])
        assert len(tak.DATASETS["drow"].real_people(polar)) == 3

    def test_it_is_a_measured_no_op_on_frog(self):
        """FROG's labels stop at exactly 10.00 m, its annotation limit, in all three splits -- 0 past it."""
        polar = np.array([[10.0, 0.0], [9.999, 0.5]])
        assert len(tak.DATASETS["frog"].real_people(polar)) == 2
        assert tak.DATASETS["frog"].rated_range_m > tak.DATASETS["frog"].max_range_m

    def test_an_empty_label_list_survives_the_filter(self):
        """A frame can be annotated and hold nobody; that is a negative, not a missing annotation."""
        assert tak.DATASETS["drow"].real_people(np.zeros((0, 2))).shape == (0, 2)


class TestEvaluateHonoursTheDrowSplit:
    """`--dataset drow --split train` silently scored the test set, and the help text said so out loud.

    It mattered for nothing while every DROW number quoted here was a test number. It matters now: the DROW
    plan reads a val split to select on and a train split to check an overfit against, and a flag that is
    accepted and ignored is how a model gets selected on the frames it is about to be scored on.
    """

    def _args(self, split):
        return SimpleNamespace(dataset="drow", split=split, eval_time_frame=5, eval_stride=1)

    def test_the_split_reaches_the_loader(self, monkeypatch):
        asked = {}

        class Fake:
            def __init__(self, **kwargs):
                asked.update(kwargs)
                self.scan_id, self.det_id = [[0]], [[0]]

        monkeypatch.setitem(sys.modules, "takheliped.datasets",
                            SimpleNamespace(DROW_Dataset=Fake, FROG_Dataset=None, JRDB_Dataset=None,
                                            frog_laser_angles=None, jrdb_laser_angles=None))
        for split in ("train", "val", "test"):
            asked.clear()
            ev._load_dataset(self._args(split))
            assert asked.get("split") == split, (split, asked)

    def test_the_help_no_longer_claims_drow_is_test_only(self):
        help_text = ev.build_parser().format_help() if hasattr(ev, "build_parser") else ""
        if help_text:
            assert "DROW always uses test" not in help_text
        source = (Path(ev.__file__)).read_text(encoding="utf-8")
        assert "DROW always uses test" not in source


class TestLabelsTheRangeLimitCannotRepresent:
    """`--max-range-m` still defaults to 10 m, and on DROW that is 8% of the labels the run can never hit.

    `sanitize_ranges` maps every reading at or past the limit onto it, so a person at 12 m is, to the model, a
    beam reporting empty space at 10 m. The target is still placed, so the loss asks for a detection the input
    cannot support -- the one way a forgotten flag costs a run without ever raising.

    It warns rather than refuses, because item 74 runs exactly this configuration on purpose: a FROG model
    carries FROG's 10 m onto DROW, and the restricted-label column is what it is scored against.
    """

    def test_a_dataset_inside_the_limit_says_nothing(self):
        assert tak.unlearnable_labels([np.array([[2.0, 0.0], [9.9, 0.1]])], 10.0) is None

    def test_labels_past_the_limit_are_counted_and_named(self):
        people = [np.array([[2.0, 0.0], [12.0, 0.1]]), np.array([[17.7, 0.0]]), np.zeros((0, 2))]
        note = tak.unlearnable_labels(people, 10.0)
        assert note is not None and note.isascii()
        assert "2 of 3" in note and "66.7%" in note and "10 m" in note and "17.7" in note

    def test_a_label_exactly_at_the_limit_counts_as_unreachable(self):
        """`sanitize_ranges` maps `>= max_range_m` onto the limit, so the boundary is on the wrong side of it."""
        assert tak.unlearnable_labels([np.array([[10.0, 0.0]])], 10.0) is not None

    def test_it_survives_a_split_with_no_annotations(self):
        assert tak.unlearnable_labels([], 10.0) is None
