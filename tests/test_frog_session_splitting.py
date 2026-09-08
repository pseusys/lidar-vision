"""
Unit tests for FROG_Dataset's session-splitting logic.

Found empirically (not assumed): frog_11-36_12-43_train_val.h5 bundles two
separate recordings with a ~24.6-hour real gap between them -- get_scan()'s
temporal window has no way to know about a boundary like that unless it's
represented as a separate *sequence*, the same way DROW_Dataset already
splits its own multi-sequence recordings. These tests build a small
synthetic .h5 file with a controlled gap and check the split directly,
rather than trusting it only against the one real file that happens to be
on disk.
"""
import h5py
import numpy as np
import pytest

from follow_the_drow.datasets.frog_dataset import FROG_Dataset, _session_odom_filenames


def _write_fake_h5(path, n, gap_after=None, gap_s=1000.0, dt=0.025, n_beams=720):
    """A minimal FROG-format h5 file: flat scans, no annotations, evenly
    spaced timestamps except for one optional large gap after `gap_after`."""
    ts = np.arange(n, dtype=np.float64) * dt
    if gap_after is not None:
        ts[gap_after + 1:] += gap_s
    with h5py.File(path, "w") as h5:
        h5.create_dataset("scans", data=np.full((n, n_beams), 5.0, dtype=np.float32))
        h5.create_dataset("timestamps", data=ts)
        h5.create_dataset("circles", data=np.zeros((0, 6), dtype=np.float32))
        h5.create_dataset("circle_idx", data=np.zeros(n, dtype=np.int64))
        h5.create_dataset("circle_num", data=np.zeros(n, dtype=np.int64))
        h5.create_dataset("split", data=np.zeros(n, dtype=np.uint8))  # all "train"


class TestFrogSessionSplitting:
    def test_no_gap_yields_one_session(self, tmp_path):
        h5_path = tmp_path / "frog_fake.h5"
        _write_fake_h5(h5_path, n=200, gap_after=None)
        ds = FROG_Dataset(datapath=tmp_path, split="train", auto_download=False)
        assert len(ds.scans) == 1
        assert len(ds.scans[0]) == 200

    def test_large_gap_splits_into_two_sessions(self, tmp_path):
        h5_path = tmp_path / "frog_fake.h5"
        _write_fake_h5(h5_path, n=200, gap_after=120, gap_s=1000.0)
        ds = FROG_Dataset(datapath=tmp_path, split="train", auto_download=False)
        assert len(ds.scans) == 2
        assert len(ds.scans[0]) == 121   # frames 0..120 inclusive
        assert len(ds.scans[1]) == 79    # frames 121..199

    def test_small_gap_does_not_split(self, tmp_path):
        # A within-session pause (well under the 300s default threshold)
        # must NOT be treated as a new sequence.
        h5_path = tmp_path / "frog_fake.h5"
        _write_fake_h5(h5_path, n=200, gap_after=120, gap_s=60.0)
        ds = FROG_Dataset(datapath=tmp_path, split="train", auto_download=False)
        assert len(ds.scans) == 1
        assert len(ds.scans[0]) == 200

    def test_det_id_rebased_within_each_session(self, tmp_path):
        h5_path = tmp_path / "frog_fake.h5"
        _write_fake_h5(h5_path, n=200, gap_after=120, gap_s=1000.0)
        ds = FROG_Dataset(datapath=tmp_path, split="train", auto_download=False)
        for seq in range(len(ds.scans)):
            assert ds.det_id[seq].max() < len(ds.scans[seq])
            assert ds.det_id[seq].min() >= 0

    def test_get_scan_does_not_cross_the_session_boundary(self, tmp_path):
        h5_path = tmp_path / "frog_fake.h5"
        _write_fake_h5(h5_path, n=200, gap_after=120, gap_s=1000.0)
        ds = FROG_Dataset(datapath=tmp_path, split="train", auto_download=False)
        # near the very start of session 2 (index 2), a T=5/dtime=10 window
        # must clamp to session 2's own start (index 0), never reach index
        # -18 which would only make sense if it reached back into session 1.
        _, odoms_hist = ds.get_scan(1, 2, time_window=5, dtime=10)
        assert len(odoms_hist) == 5  # still returns a full window (clamped)

    def test_no_data_lost_across_the_split(self, tmp_path):
        h5_path = tmp_path / "frog_fake.h5"
        _write_fake_h5(h5_path, n=200, gap_after=120, gap_s=1000.0)
        ds = FROG_Dataset(datapath=tmp_path, split="train", auto_download=False)
        assert sum(len(s) for s in ds.scans) == 200


class TestSessionOdomFilenames:
    def test_two_session_filename_yields_two_odom_names_in_order(self):
        assert _session_odom_filenames("frog_11-36_12-43_train_val.h5") == [
            "frog_11-36_odom.npz", "frog_12-43_odom.npz",
        ]

    def test_single_session_filename_yields_one_odom_name(self):
        assert _session_odom_filenames("frog_16-41_test.h5") == ["frog_16-41_odom.npz"]
        assert _session_odom_filenames("frog_10-31.h5") == ["frog_10-31_odom.npz"]

    def test_no_token_yields_empty_list(self):
        assert _session_odom_filenames("frog_notoken.h5") == []


class TestRealOdometryPreference:
    def test_real_per_session_file_is_preferred_when_present(self, tmp_path):
        h5_path = tmp_path / "frog_11-36_12-43_train_val.h5"
        _write_fake_h5(h5_path, n=200, gap_after=120, gap_s=1000.0)
        # write distinctive, easily-recognised "real" odometry for each session
        for token, x_val in (("11-36", 111.0), ("12-43", 222.0)):
            ts = np.linspace(0, 20, 50)
            data = np.zeros((50, 3), dtype=np.float32)
            data[:, 0] = x_val
            np.savez(tmp_path / f"frog_{token}_odom.npz", ts=ts, data=data)
        ds = FROG_Dataset(datapath=tmp_path, split="train", auto_download=False)
        assert len(ds.scans) == 2
        # session 0's odometry should come from the 111.0 file, session 1's from 222.0
        assert np.allclose(ds.odoms[0]["xya"][:, 0], 111.0, atol=1.0)
        assert np.allclose(ds.odoms[1]["xya"][:, 0], 222.0, atol=1.0)

    def test_falls_back_to_shared_file_when_real_odom_missing(self, tmp_path):
        h5_path = tmp_path / "frog_11-36_12-43_train_val.h5"
        _write_fake_h5(h5_path, n=200, gap_after=120, gap_s=1000.0)
        # no frog_11-36_odom.npz / frog_12-43_odom.npz written -> must not crash,
        # falls back to fake (zero) odometry via the existing mechanism
        ds = FROG_Dataset(datapath=tmp_path, split="train", auto_download=False)
        assert len(ds.scans) == 2
        assert (ds.odoms[0]["xya"] == 0).all()
        assert (ds.odoms[1]["xya"] == 0).all()
