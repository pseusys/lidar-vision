"""
Unit tests for FROG_Dataset's split selection, train/val carving and odometry
strictness -- the four data-side bugs recorded as TODO.md A11-A14.

Each test here corresponds to a failure that was found on the *real* dataset
first (TODO.md A11-A14 carry the measured numbers), then reproduced
synthetically so it can be checked without a 360 MB download. Per
memory/dos-and-donts.md, a synthetic pass alone does not close any of these
items -- the real-data check is recorded in the TODO entry.
"""
import h5py
import numpy as np
import pytest

from follow_the_drow.datasets.frog_dataset import (
    FROG_Dataset, _split_at_gaps, _MODES, _recordings_per_file,
)


DT = 1.0 / 26.2       # FROG's real median inter-scan interval
N_BEAMS = 720


def _write_h5(path, n, *, with_split_field=True, recording_gap_after=None,
              pause_after=(), pause_s=5.0, n_ann=1, marker=5.0, empty_every=0):
    """A FROG-format .h5 file.

    recording_gap_after : index after which a >300s (recording) gap is inserted
    pause_after         : indices after which a `pause_s` recording pause is inserted
    """
    ts = np.arange(n, dtype=np.float64) * DT
    for i in sorted(pause_after):
        ts[i + 1:] += pause_s
    if recording_gap_after is not None:
        ts[recording_gap_after + 1:] += 100_000.0

    # One annotation per scan, 2 m straight ahead -- except for `empty_every`th
    # frame, which carries none. The official FROG files are 100% populated;
    # the three raw extras are ~62%, which is the difference that matters.
    circle_num = np.full(n, n_ann, dtype=np.int64)
    if empty_every:
        circle_num[::empty_every] = 0
    circle_idx = np.concatenate(([0], np.cumsum(circle_num)[:-1])).astype(np.int64)
    circles = np.zeros((int(circle_num.sum()), 6), dtype=np.float32)
    circles[:, 0] = 2.0          # x forward
    circles[:, 1] = 0.0          # y left

    with h5py.File(path, "w") as h5:
        h5.create_dataset("scans", data=np.full((n, N_BEAMS), marker, dtype=np.float32))
        h5.create_dataset("timestamps", data=ts)
        h5.create_dataset("circles", data=circles)
        h5.create_dataset("circle_idx", data=circle_idx)
        h5.create_dataset("circle_num", data=circle_num)
        if with_split_field:
            # FROG's own interleaved per-frame holdout: every 10th frame is
            # their val half. We honour split==0 to reproduce their training
            # set exactly, and never use their val half -- see A12.
            split = np.zeros(n, dtype=np.uint8)
            split[::10] = 1
            h5.create_dataset("split", data=split)


def _write_odom(path, n, x_val=0.0):
    ts = np.arange(n, dtype=np.float64) * DT
    data = np.zeros((n, 3), dtype=np.float32)
    data[:, 0] = x_val
    np.savez(path, ts=ts, data=data)


@pytest.fixture
def frog_dir(tmp_path):
    """Every published FROG file plus its odometry -- the configuration in
    which the A11 glob bug does maximum damage, and the one the real
    cross-recording split needs."""
    _write_h5(tmp_path / "frog_11-36_12-43_train_val.h5", 4000,
              recording_gap_after=1999, pause_after=(500, 2500), pause_s=5.0)
    _write_h5(tmp_path / "frog_16-41_test.h5", 1000, with_split_field=False,
              marker=9.0)
    for i, name in enumerate(("frog_10-31.h5", "frog_14-57.h5", "frog_15-53.h5")):
        _write_h5(tmp_path / name, 600, with_split_field=False, marker=2.0 + i,
                  empty_every=3)      # raw recordings: a third of frames empty
    for token in ("11-36", "12-43", "16-41", "10-31", "14-57", "15-53"):
        _write_odom(tmp_path / f"frog_{token}_odom.npz", 4000, x_val=1.0)
    return tmp_path


# ---------------------------------------------------------------------------
# A11 -- split must select files, not glob whatever is on disk
# ---------------------------------------------------------------------------

class TestSplitSelectsFiles:
    def test_test_split_does_not_load_the_train_val_file(self, frog_dir):
        ds = FROG_Dataset(datapath=frog_dir, split="test", auto_download=False,
                          verbose=False)
        # The test file has 1000 scans; the train_val file has 4000. Loading
        # both (the old glob behaviour) would give 5000.
        assert sum(len(s) for s in ds.scans) == 1000

    def test_train_split_does_not_swallow_the_test_file(self, frog_dir):
        """The trap in the obvious partial fix: the test file has no `split`
        field, so the old `det_mask = ones` branch pulled all of it into train."""
        ds = FROG_Dataset(datapath=frog_dir, split="train", auto_download=False,
                          verbose=False)
        assert sum(len(s) for s in ds.scans) == 4000

    def test_each_split_reads_its_own_files_only(self, frog_dir):
        """The headline A11 regression: `test` returned the training file's
        every frame, so train + val == test exactly."""
        counts, scans = {}, {}
        for split in ("train", "val", "test"):
            ds = FROG_Dataset(datapath=frog_dir, split=split, auto_download=False,
                              verbose=False)
            counts[split] = sum(len(d) for d in ds.det_id)
            scans[split] = sum(len(s) for s in ds.scans)
        assert scans == {"train": 4000, "val": 1800, "test": 1000}
        # test keeps every populated frame -- the published protocol
        assert counts["test"] == 1000
        # train and val additionally drop RUNUP_SCANS at each sequence start,
        # so they sit just below the populated counts (1200 val, 3600 train)
        assert counts["val"] == 1122
        assert counts["train"] == 3456

    def test_unknown_split_is_rejected(self, frog_dir):
        with pytest.raises(ValueError):
            FROG_Dataset(datapath=frog_dir, split="trian", auto_download=False,
                         verbose=False)


# ---------------------------------------------------------------------------
# A12 -- val is a held-out RECORDING, and train matches the paper exactly
# ---------------------------------------------------------------------------

class TestCrossRecordingVal:
    """Val is three whole recordings the model never trains on, not a carve-out
    of the training recordings.

    The first attempt at fixing A12 held out contiguous blocks *inside* the
    training recordings, with guard bands. That works, but it measures
    same-recording generalization -- which systematically overestimates the
    cross-recording performance the test set actually reports, so early
    stopping selects for the wrong thing. FROG publishes three further fully
    annotated recordings that nothing else uses, so a cross-recording val costs
    nothing and lets training keep the paper's full 108,356 frames.
    """

    def test_val_reads_only_the_extra_recordings(self, frog_dir):
        va = FROG_Dataset(datapath=frog_dir, split="val", auto_download=False,
                          verbose=False)
        assert sum(len(s) for s in va.scans) == 3 * 600

    def test_train_reads_only_the_train_val_file(self, frog_dir):
        tr = FROG_Dataset(datapath=frog_dir, split="train", auto_download=False,
                          verbose=False)
        assert sum(len(s) for s in tr.scans) == 4000

    def test_train_honours_frogs_own_split_field_for_paper_parity(self, frog_dir):
        """Training must reproduce the paper's training set exactly -- its
        `split == 0` frames -- not the whole file. The extra 10% is the
        authors' own val half; using it would beat them partly on data."""
        tr = FROG_Dataset(datapath=frog_dir, split="train", auto_download=False,
                          verbose=False)
        n = sum(len(d) for d in tr.det_id)
        # 3600 populated split==0 frames, less RUNUP_SCANS per sequence start
        assert n == 3456, f"expected split==0 minus the run-up, got {n}"
        with_runup = FROG_Dataset(datapath=frog_dir, split="train", runup_scans=0,
                                  auto_download=False, verbose=False)
        assert sum(len(d) for d in with_runup.det_id) == 3600

    def test_no_recording_appears_in_two_splits(self, frog_dir):
        """The property the whole design exists for: a val frame and a train
        frame can never come from the same recording, so no window, no guard
        band and no near-duplicate can bridge them."""
        seen = {}
        for split in ("train", "val", "test"):
            ds = FROG_Dataset(datapath=frog_dir, split=split, auto_download=False,
                              verbose=False)
            for seq in range(len(ds.scans)):
                # first scan of a sequence identifies its recording uniquely here
                key = float(ds.scans[seq][0][0])
                seen.setdefault(key, set()).add(split)
        overlaps = {k: v for k, v in seen.items() if len(v) > 1}
        assert not overlaps, f"a recording is shared across splits: {overlaps}"

    def test_val_annotates_every_populated_frame(self, frog_dir):
        """The extras carry no `split` field, so every *populated* frame counts
        -- the same branch the test recording takes, where all of them are."""
        va = FROG_Dataset(datapath=frog_dir, split="val", auto_download=False,
                          verbose=False)
        assert sum(len(d) for d in va.det_id) == 1122
        # every kept frame is populated; the shortfall from 1200 is the run-up
        for seq in range(len(va.det_id)):
            for det_idx in range(len(va.det_id[seq])):
                assert len(va.det_wp[seq][det_idx]) > 0

    def test_val_is_independent_of_time_frame(self, frog_dir):
        a = FROG_Dataset(datapath=frog_dir, split="val", time_frame_size=5,
                         auto_download=False, verbose=False)
        b = FROG_Dataset(datapath=frog_dir, split="val", time_frame_size=10,
                         auto_download=False, verbose=False)
        assert [len(d) for d in a.det_id] == [len(d) for d in b.det_id]


# ---------------------------------------------------------------------------
# A13 -- recording pauses become sequence boundaries
# ---------------------------------------------------------------------------

class TestSessionSplitting:
    def test_split_at_gaps_finds_every_pause(self):
        ts = np.arange(100, dtype=np.float64) * DT
        ts[50:] += 5.0
        bounds = _split_at_gaps(ts, 0.5)
        assert bounds == [(0, 50), (50, 100)]

    def test_no_gap_yields_one_bound(self):
        ts = np.arange(100, dtype=np.float64) * DT
        assert _split_at_gaps(ts, 0.5) == [(0, 100)]

    def test_pauses_split_sequences(self, frog_dir):
        """4000 scans, one recording gap and two 5 s pauses -> 4 sequences."""
        ds = FROG_Dataset(datapath=frog_dir, split="train", auto_download=False,
                          verbose=False)
        assert len(ds.scans) == 4

    def test_no_window_spans_a_pause(self, frog_dir):
        """The A13 failure: a nominal 0.38 s window that really spans 74 s."""
        ds = FROG_Dataset(datapath=frog_dir, split="train", time_frame_size=5,
                          auto_download=False, verbose=False)
        for seq in range(len(ds.scans)):
            t = ds.scan_time[seq]
            if len(t) > 1:
                assert np.diff(t).max() < 0.5

    def test_default_session_gap_is_sub_second(self):
        assert FROG_Dataset.SESSION_GAP_S <= 1.0


# ---------------------------------------------------------------------------
# A14 -- odometry is required, never silently faked
# ---------------------------------------------------------------------------

class TestOdometryIsRequired:
    def test_missing_odometry_raises(self, tmp_path):
        _write_h5(tmp_path / "frog_11-36_12-43_train_val.h5", 500)
        with pytest.raises(FileNotFoundError):
            FROG_Dataset(datapath=tmp_path, split="train", auto_download=False,
                         verbose=False)

    def test_corrupt_odometry_raises_rather_than_falling_back(self, tmp_path):
        _write_h5(tmp_path / "frog_11-36_12-43_train_val.h5", 500)
        (tmp_path / "frog_11-36_odom.npz").write_bytes(b"not an npz")
        (tmp_path / "frog_12-43_odom.npz").write_bytes(b"not an npz")
        with pytest.raises(Exception) as exc:
            FROG_Dataset(datapath=tmp_path, split="train", auto_download=False,
                         verbose=False)
        assert not isinstance(exc.value, AssertionError)

    def test_real_odometry_is_used_when_present(self, frog_dir):
        ds = FROG_Dataset(datapath=frog_dir, split="train", auto_download=False,
                          verbose=False)
        assert np.allclose(ds.odoms[0]["xya"][:, 0], 1.0)

    def test_heading_is_interpolated_the_short_way_across_plus_minus_pi(self, tmp_path):
        """Published FROG heading is 10 Hz and steps 180 -> -179 deg when the robot
        turns 1 deg across the wrap; a straight-line interpolation of the raw
        numbers sweeps through 0 and put 140-deg single-frame jumps into the
        real test recording (CHANGELOG.md 2026-09-14)."""
        _write_h5(tmp_path / "frog_16-41_test.h5", 500, with_split_field=False)
        odom_ts = np.arange(200, dtype=np.float64) * 0.1
        heading = np.where(np.arange(200) % 2 == 0, np.radians(179.0), np.radians(-179.0))
        data = np.stack([odom_ts * 0.5, np.zeros(200), heading], axis=1).astype(np.float32)
        np.savez(tmp_path / "frog_16-41_odom.npz", ts=odom_ts, data=data)
        ds = FROG_Dataset(datapath=tmp_path, split="test", auto_download=False,
                          verbose=False)
        h = ds.odoms[0]["xya"][:, 2].astype(np.float64)
        assert np.cos(h).max() < np.cos(np.radians(179.0)) + 1e-6   # every scan within 1 deg of 180
        assert np.all(np.abs(h) <= np.pi + 1e-6)                    # and still wrapped
        t = ds.odoms[0]["t"]
        assert np.allclose(ds.odoms[0]["xya"][:, 0], 0.5 * t, atol=1e-4)  # position untouched


# ---------------------------------------------------------------------------
# Recording boundaries and per-sequence bookkeeping
# (carried over from the superseded tests/test_frog_session_splitting.py)
# ---------------------------------------------------------------------------

class TestRecordingBoundaries:
    def test_det_id_is_rebased_within_each_sequence(self, frog_dir):
        ds = FROG_Dataset(datapath=frog_dir, split="train", auto_download=False,
                          verbose=False)
        for seq in range(len(ds.scans)):
            if len(ds.det_id[seq]):
                assert ds.det_id[seq].max() < len(ds.scans[seq])
                assert ds.det_id[seq].min() >= 0

    def test_get_scan_clamps_at_the_sequence_start(self, frog_dir):
        """Near the start of a later sequence a T=5/dtime=10 window must clamp
        to that sequence's own index 0, never reach back into the previous
        recording."""
        ds = FROG_Dataset(datapath=frog_dir, split="train", auto_download=False,
                          verbose=False)
        _, odoms_hist = ds.get_scan(1, 2, time_window=5, dtime=10)
        assert len(odoms_hist) == 5

    def test_no_scan_is_lost_across_the_split(self, frog_dir):
        ds = FROG_Dataset(datapath=frog_dir, split="train", auto_download=False,
                          verbose=False)
        assert sum(len(s) for s in ds.scans) == 4000

    def test_recordings_split_even_without_a_pause(self, tmp_path):
        _write_h5(tmp_path / "frog_11-36_12-43_train_val.h5", 400,
                  recording_gap_after=199)
        for token in ("11-36", "12-43"):
            _write_odom(tmp_path / f"frog_{token}_odom.npz", 400)
        ds = FROG_Dataset(datapath=tmp_path, split="train", auto_download=False,
                          verbose=False)
        assert len(ds.scans) == 2
        assert [len(s) for s in ds.scans] == [200, 200]


class TestSessionOdomFilenames:
    def test_two_session_filename_yields_two_odom_names_in_order(self):
        from follow_the_drow.datasets.frog_dataset import _session_odom_filenames
        assert _session_odom_filenames("frog_11-36_12-43_train_val.h5") == [
            "frog_11-36_odom.npz", "frog_12-43_odom.npz",
        ]

    def test_single_session_filename_yields_one_odom_name(self):
        from follow_the_drow.datasets.frog_dataset import _session_odom_filenames
        assert _session_odom_filenames("frog_16-41_test.h5") == ["frog_16-41_odom.npz"]
        assert _session_odom_filenames("frog_10-31.h5") == ["frog_10-31_odom.npz"]

    def test_no_token_yields_empty_list(self):
        from follow_the_drow.datasets.frog_dataset import _session_odom_filenames
        assert _session_odom_filenames("frog_notoken.h5") == []


# ---------------------------------------------------------------------------
# Val must match the benchmark's own frame composition
# ---------------------------------------------------------------------------

class TestHeldOutSplitComposition:
    """Both official FROG files are 100% frames-containing-people; the three
    extra recordings are ~62%, because they are raw and uncurated.

    Leaving the empty ~38% in val puts pure-false-positive-opportunity frames
    into a precision-recall metric that neither train nor test contains, and it
    is devastating: val wp-AUC read 50.8% against a train probe of 85.7% on the
    first run that tried it, a +35pp gap that is a composition artifact, not
    generalization failure. Conditioned on containing people the extras average
    3.13-3.36 people/frame against test's 3.07 -- the scenes match, only the
    empty stretches differ.

    The rule is therefore "a held-out split annotates frames containing at
    least one person", and the evidence it is the right rule is that it is a
    no-op on both official files.
    """

    def test_val_drops_frames_with_no_people(self, frog_dir):
        va = FROG_Dataset(datapath=frog_dir, split="val", auto_download=False,
                          verbose=False)
        for seq in range(len(va.det_id)):
            for det_idx in range(len(va.det_id[seq])):
                assert len(va.det_wp[seq][det_idx]) > 0

    def test_the_rule_is_a_no_op_on_the_official_test_file(self, frog_dir):
        """Every frame of the real test recording already contains people, so
        this must not remove a single one -- that is what makes it a protocol
        match rather than a convenience."""
        te = FROG_Dataset(datapath=frog_dir, split="test", auto_download=False,
                          verbose=False)
        assert sum(len(d) for d in te.det_id) == sum(len(s) for s in te.scans)

    def test_populated_val_is_smaller_but_not_empty(self, frog_dir):
        va = FROG_Dataset(datapath=frog_dir, split="val", auto_download=False,
                          verbose=False)
        n_ann = sum(len(d) for d in va.det_id)
        n_scans = sum(len(s) for s in va.scans)
        assert 0 < n_ann < n_scans


# ---------------------------------------------------------------------------
# A30 -- three loading modes over one dataset
# ---------------------------------------------------------------------------

def _recording_ids(ds):
    """Which physical recording each loaded sequence came from, identified by
    its timestamp decade.

    The synthetic train_val file puts its second recording 100,000 s after the
    first (as the real one sits ~24.6 h after), so the start timestamp
    separates them where scan counts and marker values cannot.
    """
    return {(round(float(t[0]) // 1000), round(float(sc[0, 0])))
            for t, sc in zip(ds.scan_time, ds.scans)}


class TestModeSelection:
    def test_official_is_the_default(self, frog_dir):
        """Adding modes must not move a single frame for a caller that does not
        ask for one -- every FROG number on record was measured this way."""
        for split in ("train", "val", "test"):
            implicit = FROG_Dataset(datapath=frog_dir, split=split,
                                    auto_download=False, verbose=False)
            explicit = FROG_Dataset(datapath=frog_dir, split=split,
                                    mode="official", auto_download=False,
                                    verbose=False)
            assert [len(d) for d in implicit.det_id] == [len(d) for d in explicit.det_id]
            assert [len(s) for s in implicit.scans] == [len(s) for s in explicit.scans]

    def test_unknown_mode_is_rejected(self, frog_dir):
        with pytest.raises(ValueError, match="Unknown FROG mode"):
            FROG_Dataset(datapath=frog_dir, split="test", mode="blanced",
                         auto_download=False, verbose=False)

    def test_every_mode_defines_every_split(self):
        for mode, splits in _MODES.items():
            assert set(splits) == {"train", "val", "test"}, mode

    def test_recordings_per_file_lets_whole_file_win(self):
        """`None` means every recording; a spec asking for a file both whole and
        by index must not end up narrowed to the index."""
        assert _recordings_per_file(
            (("test", None), ("test", 0))) == {"test": None}
        assert _recordings_per_file(
            (("test", 0), ("test", None))) == {"test": None}
        assert _recordings_per_file(
            (("train_val", 0), ("train_val", 1))) == {"train_val": frozenset({0, 1})}


class TestTransferredMode:
    """`transferred` is the *same experiment* as `official` with a different
    question asked of it, so its training data must be identical bit for bit.
    A32b's whole premise is that the checkpoint is shared; if these ever drift
    apart, the comparison silently becomes two experiments."""

    @pytest.mark.parametrize("split", ["train", "val"])
    def test_train_and_val_are_identical_to_official(self, frog_dir, split):
        off = FROG_Dataset(datapath=frog_dir, split=split, mode="official",
                           auto_download=False, verbose=False)
        tra = FROG_Dataset(datapath=frog_dir, split=split, mode="transferred",
                           auto_download=False, verbose=False)
        assert len(off.scans) == len(tra.scans)
        for a, b in zip(off.det_id, tra.det_id):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(off.scans, tra.scans):
            np.testing.assert_array_equal(a, b)

    def test_test_split_scores_only_person_free_frames(self, frog_dir):
        te = FROG_Dataset(datapath=frog_dir, split="test", mode="transferred",
                          auto_download=False, verbose=False)
        assert sum(len(d) for d in te.det_id) > 0
        for seq in range(len(te.det_id)):
            for det_idx in range(len(te.det_id[seq])):
                assert len(te.det_wp[seq][det_idx]) == 0

    def test_it_is_the_exact_complement_of_officials_val(self, frog_dir):
        """Same three recordings, disjoint frames, nothing dropped between
        them -- so a claim about one cannot be quietly about a different subset
        of the other."""
        va = FROG_Dataset(datapath=frog_dir, split="val", mode="official",
                          auto_download=False, verbose=False)
        te = FROG_Dataset(datapath=frog_dir, split="test", mode="transferred",
                          auto_download=False, verbose=False)
        assert [len(s) for s in va.scans] == [len(s) for s in te.scans]
        for populated, empty, scans in zip(va.det_id, te.det_id, va.scans):
            assert set(populated).isdisjoint(set(empty))
            # every frame past the run-up is in exactly one of the two
            assert len(populated) + len(empty) == len(scans) - 40

    def test_test_split_takes_the_standard_runup(self, frog_dir):
        """Unlike official's test it has no published number to stay comparable
        with, and a padded window is exactly the degenerate input a
        false-positive count must not be fed."""
        te = FROG_Dataset(datapath=frog_dir, split="test", mode="transferred",
                          auto_download=False, verbose=False)
        assert all(int(i) >= 40 for d in te.det_id for i in d)


class TestBalancedMode:
    def test_every_split_sees_both_empty_and_populated_frames(self, frog_dir):
        """The property the mode is named for. Without it there is nothing to
        distinguish `balanced` from `official` with extra frames."""
        for split in ("train", "val", "test"):
            ds = FROG_Dataset(datapath=frog_dir, split=split, mode="balanced",
                              auto_download=False, verbose=False)
            populated = sum(1 for seq in ds.det_wp for wp in seq if len(wp) > 0)
            empty = sum(1 for seq in ds.det_wp for wp in seq if len(wp) == 0)
            assert populated > 0, split
            assert empty > 0, split

    def test_no_recording_appears_in_two_splits(self, frog_dir):
        seen = {}
        for split in ("train", "val", "test"):
            ds = FROG_Dataset(datapath=frog_dir, split=split, mode="balanced",
                              auto_download=False, verbose=False)
            seen[split] = _recording_ids(ds)
        assert seen["train"].isdisjoint(seen["val"])
        assert seen["train"].isdisjoint(seen["test"])
        assert seen["val"].isdisjoint(seen["test"])

    def test_all_six_recordings_are_used_exactly_once(self, frog_dir):
        used = []
        for split in ("train", "val", "test"):
            ds = FROG_Dataset(datapath=frog_dir, split=split, mode="balanced",
                              auto_download=False, verbose=False)
            used += list(_recording_ids(ds))
        assert len(used) == 6
        assert len(set(used)) == 6

    def test_test_split_holds_out_the_published_baselines_training_data(self):
        """LFE-Peaks, LFE-PPN and DROW were trained by their authors on FROG's
        official train set, i.e. on both recordings of the train_val file.
        Either of them in `balanced`'s test would make A32's numbers a score on
        their own training data -- the exact class of bug A11 records."""
        test_spec, _ = _MODES["balanced"]["test"]
        assert not any(key == "train_val" for key, _ in test_spec)

    def test_a_recording_from_another_split_is_not_even_loaded(self, frog_dir):
        """Not merely unscored: absent. An unscored frame still supplies
        temporal history to its neighbours, which would let a training window
        read across a split boundary."""
        tr = FROG_Dataset(datapath=frog_dir, split="train", mode="balanced",
                          auto_download=False, verbose=False)
        va = FROG_Dataset(datapath=frog_dir, split="val", mode="balanced",
                          auto_download=False, verbose=False)
        # train takes train_val's second recording, val its first; the
        # synthetic file puts them 100,000 s apart, and marks both with 5.0
        # so only the timestamp can tell them apart.
        def train_val_starts(ds):
            return [float(t[0]) for sc, t in zip(ds.scans, ds.scan_time)
                    if abs(float(sc[0, 0]) - 5.0) < 1e-6]

        assert train_val_starts(tr) and all(x >= 100_000 for x in train_val_starts(tr))
        assert train_val_starts(va) and all(x < 100_000 for x in train_val_starts(va))

    def test_it_ignores_the_papers_own_interleaved_split_field(self, frog_dir):
        """`balanced` is our partition, so the paper's per-frame holdout has no
        standing in it -- keeping it would drop a tenth of the frames for a
        reason that no longer applies."""
        tr = FROG_Dataset(datapath=frog_dir, split="train", mode="balanced",
                          auto_download=False, verbose=False)
        train_val_frames = sum(
            len(s) for s, t in zip(tr.scans, tr.scan_time)
            if float(t[0]) >= 100_000)
        scored = sum(
            len(d) for d, t in zip(tr.det_id, tr.scan_time)
            if float(t[0]) >= 100_000)
        # only the run-up is dropped, not the paper's every-10th-frame half
        assert scored > 0.95 * train_val_frames
