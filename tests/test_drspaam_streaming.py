"""DR-SPAAM in the streaming form its authors published (`TODO.md` A47, §A item 6).

The official detector is **stateful**: `_SpatialAttentionMemory` keeps its template between calls and `reset()` clears it, so
what it reports on a frame depends on every earlier frame of the sequence. Ours re-encodes a `T`-scan window from scratch on
every call and initialises the template from the oldest scan in that window, which makes every DR-SPAAM number in this
repository the *windowed* form rather than the published algorithm.

The equivalence that makes the streaming path trustworthy is exact rather than statistical: **playing frames 0..T-1 one at a
time must give what the windowed forward gives for the window 0..T-1**, because both evaluate the same recursion. The state is
passed in and returned rather than hidden on the module, which is the convention `ObjectMemory.step` already uses here.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))

from follow_the_drow.detectors.architectures import DrSpaamDetector

BEAMS = 6
SAMPLES = 56
SCANS = 4
TOLERANCE = 1e-5


@pytest.fixture
def detector():
    torch.manual_seed(0)
    return DrSpaamDetector(num_scans=SCANS, num_pts=SAMPLES, pedestrian_only=True).eval()


@pytest.fixture
def window():
    torch.manual_seed(1)
    return torch.randn(BEAMS, SCANS, SAMPLES)


class TestStreamingMatchesTheWindow:
    def test_the_last_streamed_frame_equals_the_windowed_forward(self, detector, window):
        with torch.no_grad():
            expected_logits, expected_votes = detector(window)
            template = None
            for t in range(SCANS):
                logits, votes, template = detector.stream(window[:, t, :], template)

        assert torch.allclose(logits, expected_logits, atol=TOLERANCE)
        assert torch.allclose(votes, expected_votes, atol=TOLERANCE)

    def test_every_streamed_frame_equals_the_windowed_forward_over_the_prefix(self, detector, window):
        """Frame k has seen scans 0..k, which is what a window of those scans encodes."""
        with torch.no_grad():
            template = None
            for t in range(SCANS):
                logits, votes, template = detector.stream(window[:, t, :], template)
                expected_logits, expected_votes = detector(window[:, : t + 1, :])
                assert torch.allclose(logits, expected_logits, atol=TOLERANCE), t
                assert torch.allclose(votes, expected_votes, atol=TOLERANCE), t

    def test_the_first_frame_needs_no_state(self, detector, window):
        with torch.no_grad():
            logits, votes, _ = detector.stream(window[:, 0, :], None)
            expected_logits, expected_votes = detector(window[:, :1, :])

        assert torch.allclose(logits, expected_logits, atol=TOLERANCE)
        assert torch.allclose(votes, expected_votes, atol=TOLERANCE)

    def test_dropping_the_state_is_what_a_sequence_boundary_does(self, detector, window):
        """Two sequences played through one detector must not leak into each other."""
        with torch.no_grad():
            first, _, carried = detector.stream(window[:, 0, :], None)
            for t in range(1, SCANS):
                _, _, carried = detector.stream(window[:, t, :], carried)
            after_reset, _, _ = detector.stream(window[:, 0, :], None)

        assert torch.allclose(first, after_reset, atol=TOLERANCE)

    def test_the_returned_state_carries_no_graph(self, detector, window):
        """The template is state, not a path for gradients: the official detector detaches its own recurrence too."""
        _, _, template = detector.stream(window[:, 0, :], None)

        assert not template.requires_grad

    def test_the_output_keeps_one_row_per_beam(self, detector, window):
        with torch.no_grad():
            logits, votes, template = detector.stream(window[:, 0, :], None)

        assert logits.shape == (BEAMS, 1)
        assert votes.shape == (BEAMS, 2)
        assert template.shape[:2] == (1, BEAMS)
