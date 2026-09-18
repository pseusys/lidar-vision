"""The FROG loader's missing-return clamp is opt-in (`TODO.md` A49, owner's call 2026-09-17).

The three-horizon detector reads raw scans and sanitises them itself (`three_horizon.sanitize_ranges`); the baselines keep the
legacy clamp they were always run with. Pinned: the default still clamps, so no baseline number moves; `None` passes every
reading through untouched, including train/val's finite 61.0 m no-return encoding (`memory/noise-structure.md`).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))

from follow_the_drow.datasets.frog_dataset import LEGACY_MISSING_RETURN_M, encode_missing_returns


class TestEncodeMissingReturns:

    def test_the_default_is_the_legacy_ten_metre_clamp_of_non_finite_readings(self):
        scans = np.array([[np.inf, np.nan, -np.inf, 61.0, 3.0]], dtype=np.float32)
        assert LEGACY_MISSING_RETURN_M == 10.0
        assert encode_missing_returns(scans, LEGACY_MISSING_RETURN_M).tolist() == [[10.0, 10.0, 10.0, 61.0, 3.0]]

    def test_none_leaves_the_raw_readings(self):
        scans = np.array([[np.inf, np.nan, 61.0, 3.0]], dtype=np.float32)
        out = encode_missing_returns(scans, None)
        assert np.isposinf(out[0, 0]) and np.isnan(out[0, 1]) and out[0, 2:].tolist() == [61.0, 3.0]
