"""World-frame persistence accumulator (TODO.md A41).

Role C: a phantom is a chair, and a chair stays where it is. The oracle says
89.7% of LFE-Peaks' phantom *detections* and 93.9% of LFE-PPN's sit on trails
long enough to be rejectable at a 30 s horizon, while only 3.6% of real person
trajectories last that long -- so evidence accumulated per world location
separates the two classes, if it is accumulated in the right frame.

`PersistenceMap` is the classical form of that: an exponentially-forgetting
per-cell counter in world coordinates, O(1) per detection per frame. Two knobs
and no training -- `decay` sets the horizon, and the caller's threshold sets how
much evidence is enough.

The three properties worth pinning are the ones that would silently flatter the
result if broken: the horizon must actually be the horizon, the state must be
*world*-indexed rather than carried along with the robot, and a query must not
see the detection it is being asked to judge.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from persistence_map import PersistenceMap, horizon_to_decay


class TestAccumulation:

    def test_an_unseen_cell_is_zero(self):
        m = PersistenceMap(cell=0.25, decay=0.99)
        assert m.query(3.0, 4.0, frame=0) == 0.0

    def test_one_observation_reads_back_as_one(self):
        m = PersistenceMap(cell=0.25, decay=0.99)
        m.add(3.0, 4.0, frame=0)
        assert m.query(3.0, 4.0, frame=0) == pytest.approx(1.0)

    def test_evidence_decays_with_elapsed_frames_not_with_accesses(self):
        """Querying repeatedly must not age the cell -- only time may."""
        m = PersistenceMap(cell=0.25, decay=0.9)
        m.add(0.0, 0.0, frame=0)
        for _ in range(5):
            assert m.query(0.0, 0.0, frame=10) == pytest.approx(0.9 ** 10)

    def test_a_permanent_object_saturates_at_the_horizon_length(self):
        """Adding every frame converges to 1/(1-decay), which is the number of
        frames the horizon holds -- so the threshold is readable in frames."""
        decay = 0.99
        m = PersistenceMap(cell=0.25, decay=decay)
        for f in range(2000):
            m.add(0.0, 0.0, frame=f)
        assert m.query(0.0, 0.0, frame=2000) == pytest.approx(1 / (1 - decay), rel=0.02)

    def test_query_does_not_mutate(self):
        """Causality at the call site depends on this: a detection is judged by
        what was there *before* it, so query must be read-only."""
        m = PersistenceMap(cell=0.25, decay=0.99)
        m.add(0.0, 0.0, frame=0)
        before = m.query(0.0, 0.0, frame=5)
        m.query(0.0, 0.0, frame=5)
        assert m.query(0.0, 0.0, frame=5) == pytest.approx(before)


class TestItIsIndexedByTheWorld:

    def test_points_in_one_cell_share_state(self):
        m = PersistenceMap(cell=0.50, decay=0.99)
        m.add(1.00, 1.00, frame=0)
        assert m.query(1.20, 1.20, frame=0) == pytest.approx(1.0)

    def test_points_in_different_cells_do_not(self):
        m = PersistenceMap(cell=0.25, decay=0.99)
        m.add(1.0, 1.0, frame=0)
        assert m.query(3.0, 3.0, frame=0) == 0.0

    def test_a_static_world_point_accumulates_while_the_robot_moves(self):
        """The whole reason the state is world-indexed. Over 786 frames the
        robot travels ~15 m, so a sensor-indexed state would be pointing
        somewhere else entirely; a world-indexed one keeps the address."""
        m = PersistenceMap(cell=0.25, decay=0.999)
        for f in range(400):
            m.add(7.0, -2.0, frame=f)       # same world point, any robot pose
        assert m.query(7.0, -2.0, frame=400) > 300

    def test_a_moving_point_does_not_accumulate(self):
        """A person walking at 1 m/s leaves 0.25 m cells faster than evidence
        builds, which is exactly the asymmetry role C exploits."""
        m = PersistenceMap(cell=0.25, decay=0.999)
        for f in range(400):
            m.add(0.0, f * 0.04, frame=f)   # ~1 m/s at 26.2 Hz
        vals = [m.query(0.0, f * 0.04, frame=400) for f in range(0, 400, 50)]
        assert max(vals) < 20


class TestHorizonConversion:

    @pytest.mark.parametrize("seconds", [1.0, 10.0, 30.0, 60.0])
    def test_decay_round_trips_to_the_requested_horizon(self, seconds):
        """`decay` is unreadable as a number; horizons are what the oracle
        reports, so the knob is specified in seconds and converted."""
        hz = 26.2
        d = horizon_to_decay(seconds, hz)
        assert 1 / (1 - d) == pytest.approx(seconds * hz, rel=1e-6)

    def test_a_longer_horizon_forgets_more_slowly(self):
        hz = 26.2
        assert horizon_to_decay(30.0, hz) > horizon_to_decay(3.0, hz)
