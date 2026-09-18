"""Background model from raw scan returns (TODO.md A44).

A41 counted *detections* per world cell and failed: people revisit locations, so
detection frequency conflates queues and doorways with furniture (0.57 FP per
FN). A45 then showed that filtering *trails* has a low ceiling too, and that the
ceiling is association-limited.

A44 uses neither. It accumulates **raw scan returns**, because a chair reflects
a beam whether or not anything detects it, and asks a different question: *is
this detection standing on a surface that was already here?*

The discriminating quantity is **hit rate, not hit count** -- what fraction of
the frames spanning a cell's life actually returned from it:

    a chair                 seen whenever observed      rate ~1, span long
    someone standing 5 s    seen whenever observed      rate ~1, span SHORT
    a busy corridor cell    many people, each briefly   rate LOW, span long

so background is `rate >= min_rate AND span >= min_span`, and each of the three
cases above is excluded by a different one of those two conditions. A count
alone cannot separate them, which is exactly how A41 failed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from persistence_map import BackgroundMap


class TestTheThreeCasesItMustSeparate:

    def test_a_chair_is_background(self):
        """Returned from on every frame across a long span."""
        m = BackgroundMap(cell=0.25)
        for f in range(2000):
            m.observe([(4.0, 1.0)], f)
        assert m.is_background(4.0, 1.0, min_rate=0.6, min_span=500)

    def test_someone_standing_briefly_is_not(self):
        """Rate is 1.0 -- identical to the chair -- and only the span
        separates them. This is the case a hit *count* cannot exclude."""
        m = BackgroundMap(cell=0.25)
        for f in range(131):                      # 5 s at 26.2 Hz
            m.observe([(4.0, 1.0)], f)
        assert not m.is_background(4.0, 1.0, min_rate=0.6, min_span=500)

    def test_a_busy_corridor_cell_is_not(self):
        """Span is long -- identical to the chair -- and only the rate
        separates them. This is the case A41 got wrong."""
        m = BackgroundMap(cell=0.25)
        for f in range(2000):
            if f % 40 < 2:                        # brief visits, many people
                m.observe([(4.0, 1.0)], f)
            else:
                m.observe([(9.0, 9.0)], f)        # keep the clock running
        assert not m.is_background(4.0, 1.0, min_rate=0.6, min_span=500)


class TestTheAccounting:

    def test_an_unseen_cell_is_not_background(self):
        m = BackgroundMap(cell=0.25)
        m.observe([(1.0, 1.0)], 0)
        assert not m.is_background(8.0, 8.0, min_rate=0.0, min_span=0)

    def test_rate_is_hits_over_the_span_they_cover(self):
        m = BackgroundMap(cell=0.25)
        for f in range(0, 100, 2):                # every other frame
            m.observe([(2.0, 0.0)], f)
        assert m.rate(2.0, 0.0) == pytest.approx(50 / 99, abs=0.02)

    def test_span_is_first_to_last_inclusive(self):
        m = BackgroundMap(cell=0.25)
        m.observe([(2.0, 0.0)], 10)
        m.observe([(2.0, 0.0)], 19)
        assert m.span(2.0, 0.0) == 10

    def test_a_single_observation_has_span_one_and_rate_one(self):
        """Rate alone would call this background; the span condition is what
        stops a one-frame return from being furniture."""
        m = BackgroundMap(cell=0.25)
        m.observe([(2.0, 0.0)], 7)
        assert m.rate(2.0, 0.0) == pytest.approx(1.0)
        assert m.span(2.0, 0.0) == 1
        assert not m.is_background(2.0, 0.0, min_rate=0.6, min_span=500)

    def test_points_sharing_a_cell_share_state(self):
        m = BackgroundMap(cell=0.50)
        for f in range(1000):
            m.observe([(1.00, 1.00)], f)
        assert m.is_background(1.20, 1.20, min_rate=0.6, min_span=500)

    def test_many_returns_in_one_frame_count_once_per_cell(self):
        """A person subtends many beams at short range; without this a nearby
        object would look denser in time than a distant one purely because it
        is closer, which is range bias, not persistence."""
        m = BackgroundMap(cell=0.50)
        m.observe([(2.0, 0.0), (2.01, 0.0), (2.02, 0.0)], 0)
        m.observe([(2.0, 0.0)], 1)
        assert m.rate(2.0, 0.0) == pytest.approx(1.0)
        assert m.span(2.0, 0.0) == 2

    def test_neighbourhood_query_takes_the_strongest_nearby_cell(self):
        """A detection's centroid need not land in the same cell as the
        surface it stands on, so the query looks at the cells around it."""
        m = BackgroundMap(cell=0.25)
        for f in range(2000):
            m.observe([(4.0, 1.0)], f)
        assert m.is_background(4.2, 1.0, min_rate=0.6, min_span=500, radius=0.4)
        assert not m.is_background(6.0, 1.0, min_rate=0.6, min_span=500, radius=0.4)
