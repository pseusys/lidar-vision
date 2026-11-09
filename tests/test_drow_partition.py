"""DROW repartitioned by scan geometry: the solver that builds the partitions, the loader that reads them, and the
committed tables themselves.

DROW's official test recordings are enclosed and its training recordings open (`memory/dataset-properties.md`), and
our stage 2 breaks on enclosed scans it never trained on. `geometry-balanced` gives every split DROW's overall share of
enclosed scans; `geometry-shifted` is the control, built the same way and keeping the official test, so the two differ
in geometry alone. Both move as few annotated scans away from the official split as their constraints allow, and keep
the official composition: the staged, lunch and turning recordings in train, val and test driving runs.
"""
import itertools
import sys
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from drow_geometry_partition import Group, place, solve_partition  # noqa: E402
from drow_locality_property import recording_type  # noqa: E402
from takheliped.datasets.drow_dataset import partition_files  # noqa: E402
from takheliped.datasets.drow_partitions import DROW_PARTITIONS, SUMMARY  # noqa: E402
from train_takheliped import DROW, FROG  # noqa: E402

SPLITS = ("train", "val", "test")


class TestAPlace:
    @pytest.mark.parametrize("stem,where", [("run_2015-11-25-10-34-20-d.bag", "run_2015-11-25-10-34-20"),
                                            ("run_t_2015-11-26-13-47-42.bag", "run_t_2015-11-26-13-47-42"),
                                            ("lunch_2015-11-26-12-04-23.bag", "lunch_2015-11-26-12-04-23")])
    def test_a_segment_s_place_is_its_recording_as_the_files_name_it(self, stem, where):
        """The stems are `<name>.bag`, as DROW's `<name>.bag.csv` files give them."""
        assert place(stem) == where

    def test_the_tables_have_sister_segments_to_group(self):
        """Guards the test below against passing on nothing: DROW has places of several segments."""
        stems = [s for split in SPLITS for s in DROW_PARTITIONS["official"][split]]
        assert len({place(s) for s in stems}) < len(stems)


class TestTheSolver:
    @staticmethod
    def brute_force(groups, allowed, targets, size_tol, share, share_tol):
        """Every assignment, kept if it meets the constraints, the cheapest returned: the reference `milp` must match."""
        best = None
        for choice in itertools.product(*(allowed[g.name] for g in groups)):
            sizes = {s: sum(g.annotated for g, c in zip(groups, choice) if c == s) for s in SPLITS}
            enclosed = {s: sum(g.enclosed for g, c in zip(groups, choice) if c == s) for s in SPLITS}
            if any(abs(sizes[s] - targets[s]) > size_tol * targets[s] for s in SPLITS):
                continue
            if share is not None and any(abs(enclosed[s] - share * sizes[s]) > share_tol * sizes[s] for s in SPLITS):
                continue
            cost = sum(g.annotated - g.official.get(c, 0) for g, c in zip(groups, choice))
            if best is None or cost < best:
                best = cost
        return best

    @staticmethod
    def toy(shut_b_official=None):
        """A small DROW: open groups in train, half-enclosed ones in test, one group split across train and val."""
        groups = [Group("open_a", "driving", 100, 0, {"train": 100}), Group("open_b", "driving", 100, 10, {"train": 100}),
                  Group("open_c", "driving", 50, 0, {"val": 50}), Group("open_d", "driving", 50, 5, {"train": 50}),
                  Group("shut_a", "driving", 50, 25, {"test": 50}),
                  Group("shut_b", "driving", 100, 60, shut_b_official or {"train": 60, "val": 40}),
                  Group("shut_c", "driving", 50, 30, {"test": 50}), Group("staged", "stationary", 100, 0, {"train": 100})]
        allowed = {g.name: ("train",) if g.kind != "driving" else SPLITS for g in groups}
        return groups, allowed, {"train": 400, "val": 100, "test": 100}

    def test_it_moves_nothing_when_the_official_split_already_fits(self):
        groups, allowed, _ = self.toy({"train": 100})
        targets = {"train": 450, "val": 50, "test": 100}
        assignment = solve_partition(groups, allowed, targets, size_tolerance=0.05, enclosed_share=None, share_tolerance=0.0)
        assert assignment == {g.name: max(g.official, key=g.official.get) for g in groups}

    def test_it_meets_the_share_at_the_fewest_scans_moved(self):
        groups, allowed, targets = self.toy()
        share = sum(g.enclosed for g in groups) / sum(g.annotated for g in groups)
        assignment = solve_partition(groups, allowed, targets, size_tolerance=0.1, enclosed_share=share, share_tolerance=0.1)
        sizes = {s: sum(g.annotated for g in groups if assignment[g.name] == s) for s in SPLITS}
        enclosed = {s: sum(g.enclosed for g in groups if assignment[g.name] == s) for s in SPLITS}
        assert all(abs(enclosed[s] - share * sizes[s]) <= 0.1 * sizes[s] + 1e-9 for s in SPLITS)
        moved = sum(g.annotated - g.official.get(assignment[g.name], 0) for g in groups)
        reference = self.brute_force(groups, allowed, targets, 0.1, share, 0.1)
        assert reference is not None and moved == reference

    def test_a_group_goes_only_where_it_is_allowed(self):
        groups, allowed, targets = self.toy()
        share = sum(g.enclosed for g in groups) / sum(g.annotated for g in groups)
        assignment = solve_partition(groups, allowed, targets, size_tolerance=0.1, enclosed_share=share, share_tolerance=0.1)
        assert assignment["staged"] == "train"

    def test_an_impossible_partition_is_refused(self):
        groups, allowed, targets = self.toy()
        with pytest.raises(ValueError):
            solve_partition(groups, allowed, targets, size_tolerance=0.0, enclosed_share=0.5, share_tolerance=0.0)


class TestTheLoader:
    @staticmethod
    def tree(root: Path) -> Path:
        for split, stems in (("train", ("run_x-b", "lunch_y")), ("val", ("run_x-a",)), ("test", ("run_z",))):
            (root / split).mkdir(parents=True)
            for stem in stems:
                (root / split / f"{stem}.csv").write_text("")
        return root

    def test_the_official_partition_is_each_split_s_own_directory(self, tmp_path):
        root = self.tree(tmp_path)
        assert sorted(Path(f).name for f in partition_files(root, "train", "official")) == ["lunch_y", "run_x-b"]

    def test_a_partition_finds_its_recordings_in_whichever_directory_holds_them(self, tmp_path):
        root = self.tree(tmp_path)
        table = {"mine": {"train": ["run_z", "lunch_y"], "val": [], "test": ["run_x-a", "run_x-b"]}}
        files = partition_files(root, "test", "mine", partitions=table)
        assert [Path(f).parent.name + "/" + Path(f).name for f in files] == ["val/run_x-a", "train/run_x-b"]
        assert [Path(f).name for f in partition_files(root, "train", "mine", partitions=table)] == ["run_z", "lunch_y"]

    def test_a_recording_the_table_names_and_the_disk_lacks_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            partition_files(self.tree(tmp_path), "train", "mine", partitions={"mine": {"train": ["gone"], "val": [], "test": []}})

    def test_an_unknown_partition_is_an_error(self, tmp_path):
        with pytest.raises(ValueError):
            partition_files(self.tree(tmp_path), "train", "nonsense")


class TestTheCommittedTables:
    @pytest.mark.parametrize("mode", ["geometry-balanced", "geometry-shifted"])
    def test_every_recording_is_in_exactly_one_split_and_none_is_lost(self, mode):
        table = DROW_PARTITIONS[mode]
        stems = [s for split in SPLITS for s in table[split]]
        assert len(stems) == len(set(stems))
        assert set(stems) == {s for split in SPLITS for s in DROW_PARTITIONS["official"][split]}

    @pytest.mark.parametrize("mode", ["geometry-balanced", "geometry-shifted"])
    def test_no_place_is_in_two_splits(self, mode):
        """Sister segments of one recording stay together -- the official val's leak is not carried over."""
        home = {}
        for split in SPLITS:
            for stem in DROW_PARTITIONS[mode][split]:
                assert home.setdefault(place(stem), split) == split, stem

    @pytest.mark.parametrize("mode", ["geometry-balanced", "geometry-shifted"])
    def test_the_official_composition_is_kept(self, mode):
        table = DROW_PARTITIONS[mode]
        assert all(s in table["train"] for s in DROW_PARTITIONS["official"]["train"] if recording_type(s) != "driving")
        assert all(recording_type(s) == "driving" for split in ("val", "test") for s in table[split])

    def test_the_control_keeps_the_official_test(self):
        assert sorted(DROW_PARTITIONS["geometry-shifted"]["test"]) == sorted(DROW_PARTITIONS["official"]["test"])

    @pytest.mark.parametrize("mode", ["geometry-balanced", "geometry-shifted"])
    def test_every_split_is_within_five_percent_of_the_official_size(self, mode):
        for split in SPLITS:
            official = SUMMARY["official"][split]["annotated"]
            assert abs(SUMMARY[mode][split]["annotated"] - official) <= 0.05 * official

    def test_the_balanced_splits_carry_the_overall_enclosed_share(self):
        overall = sum(SUMMARY["official"][s]["enclosed"] for s in SPLITS) / sum(SUMMARY["official"][s]["annotated"] for s in SPLITS)
        for split in SPLITS:
            row = SUMMARY["geometry-balanced"][split]
            assert abs(row["enclosed"] / row["annotated"] - overall) <= 0.02 + 1e-9

    def test_the_control_stays_shifted(self):
        rows = SUMMARY["geometry-shifted"]
        assert rows["test"]["enclosed"] / rows["test"]["annotated"] > 0.6 > 0.2 > rows["train"]["enclosed"] / rows["train"]["annotated"]


class TestEachDatasetRefusesTheOther_sPartitions:
    def test_drow_refuses_a_frog_partition(self):
        with pytest.raises(ValueError):
            DROW.load("train", None, "transferred")

    def test_frog_refuses_a_drow_partition(self):
        with pytest.raises(ValueError):
            FROG.load("train", None, "geometry-balanced")
