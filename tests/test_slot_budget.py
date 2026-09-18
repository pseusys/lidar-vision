"""Slot budget for the object memory (TODO.md A43, owner's question 3.9).

Stage 3 holds a fixed number of slots K. A slot is spawned from an unmatched
proposal and retired only after `retire` frames without a match, so at any
frame the slots in use are every track that has started and has not yet been
unmatched for longer than `retire`. The owner's rule: overshoot rather than
undershoot, so the measurement must never *under*-count.

Two pieces, both pure:

    associate   gated Hungarian association with a gap tolerance measured
                in scan indices, not in list entries -- DROW annotates every 5th
                scan, so "3 missed entries" is 15 scans there and 3 on FROG
    occupancy   per-frame count of slots in use, each track occupying
                [first, last + retire], clipped to the sequence
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from slot_budget import associate, occupancy


def _frames(points_per_frame):
    return [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in points_per_frame]


class TestOccupancy:

    def test_a_track_occupies_its_life_plus_the_retire_window(self):
        c = occupancy([(10, 20)], n_frames=40, retire=5)
        assert c[9] == 0 and c[10] == 1 and c[25] == 1 and c[26] == 0

    def test_zero_retire_is_the_track_alone(self):
        c = occupancy([(3, 3)], n_frames=10, retire=0)
        assert c.tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    def test_the_tail_is_clipped_to_the_sequence(self):
        c = occupancy([(8, 9)], n_frames=10, retire=100)
        assert len(c) == 10 and c[8] == 1 and c[9] == 1

    def test_overlapping_tracks_add(self):
        c = occupancy([(0, 5), (3, 8)], n_frames=10, retire=0)
        assert c.tolist() == [1, 1, 1, 2, 2, 2, 1, 1, 1, 0]

    def test_no_tracks_is_all_zero(self):
        assert occupancy([], n_frames=4, retire=3).tolist() == [0, 0, 0, 0]


class TestAssociate:

    def test_a_static_point_is_one_track(self):
        spans = associate(_frames([[(1, 1)]] * 5), scan_index=range(5),
                          gate=0.5, max_gap=0)
        assert spans == [(0, 4)]

    def test_a_gap_within_tolerance_keeps_one_track(self):
        pts = _frames([[(1, 1)], [(1, 1)], [], [], [], [(1, 1)]])
        assert associate(pts, scan_index=range(6), gate=0.5, max_gap=3) == [(0, 5)]

    def test_a_gap_beyond_tolerance_spawns_a_second_track(self):
        pts = _frames([[(1, 1)], [(1, 1)], [], [], [], [(1, 1)]])
        assert sorted(associate(pts, scan_index=range(6), gate=0.5,
                                max_gap=2)) == [(0, 1), (5, 5)]

    def test_the_gap_is_measured_in_scans_not_entries(self):
        """DROW-shaped: annotations every 5th scan, so consecutive entries have
        4 unobserved scans between them. `max_gap` counts those scans, the same
        unit as FROG's: a tolerance of 3 must break the track and 4 must not."""
        pts = _frames([[(1, 1)], [(1, 1)]])
        assert len(associate(pts, scan_index=[0, 5], gate=0.5, max_gap=3)) == 2
        assert associate(pts, scan_index=[0, 5], gate=0.5, max_gap=4) == [(0, 5)]

    def test_a_point_beyond_the_gate_spawns_a_new_track(self):
        pts = _frames([[(0, 0)], [(3, 0)]])
        assert sorted(associate(pts, scan_index=range(2), gate=0.5,
                                max_gap=10)) == [(0, 0), (1, 1)]

    def test_two_separated_points_are_two_tracks(self):
        pts = _frames([[(0, 0), (5, 5)]] * 3)
        assert sorted(associate(pts, scan_index=range(3), gate=0.5,
                                max_gap=0)) == [(0, 2), (0, 2)]

    def test_association_maximises_matches_like_the_evaluation_does(self):
        """Track A at 0.0, track B at 0.45. New points at 0.40 and 0.85.
        Nearest-first would hand B the 0.40 point (0.05 away) and orphan A,
        spawning a spurious third track. Gated Hungarian -- the assignment the
        evaluation itself uses -- matches A to 0.40 and B to 0.85, both inside
        the gate, and keeps two tracks."""
        pts = _frames([[(0.0, 0), (0.45, 0)], [(0.40, 0), (0.85, 0)]])
        assert sorted(associate(pts, scan_index=range(2), gate=0.5,
                                max_gap=0)) == [(0, 1), (0, 1)]

    def test_empty_input_has_no_tracks(self):
        assert associate([], scan_index=[], gate=0.5, max_gap=3) == []
