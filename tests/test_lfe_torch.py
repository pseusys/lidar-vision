"""LFE-Peaks' own architecture in PyTorch, so it can be trained here (`TODO.md` §A item 7, A59 phase 3b).

LFE-Peaks scores 65.1% on FROG where our single-frame stage 2 scores 82.2%, and A59's ladder has ruled out the easy
explanations: not LFE's input features, not its size, not DR-SPAAM's training protocol. What is left is the network's form and
the training recipe, and those two can only be separated by training **LFE's architecture** the way we train ours. Their
published recipe never shuffles its training frames and early-stops on a validation loss computed over frames drawn from the
same recordings it trains on, so the 65% may be the training rather than the design.

What is pinned here is that the port **is** their network: every convolution of `lfe_peaks.onnx` in order, with its shape, its
kernel and its grouping, and the same number of parameter values. The weights are published, so the structure is not a matter
of opinion -- it is read off the graph. Matching their released weights' *outputs* is a separate verification against an
external artefact rather than a test of our code, and is reported with the result.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))

from follow_the_drow.detectors.lfe_torch import LFEPeaksNet, onnx_convolutions

GRAPH = Path(__file__).parent.parent / "library" / "follow_the_drow" / "include" / "lfe_peaks.onnx"
BEAMS = 720
ONNX_PARAMETER_VALUES = 53258       # every initialiser in lfe_peaks.onnx, batch-norm folded as the graph stores it
ONNX_CONVOLUTIONS = 29


@pytest.fixture
def net():
    torch.manual_seed(0)
    return LFEPeaksNet().eval()


class TestTheArchitectureIsLFEs:
    def test_the_convolutions_match_the_published_graph_in_order(self, net):
        """Shape, kernel and grouping of every convolution, against the graph the authors released."""
        if not GRAPH.exists():
            pytest.skip(f"{GRAPH.name} is not installed")

        assert [(tuple(m.weight.shape), m.kernel_size[0], m.groups) for m in net.convolutions()] == onnx_convolutions(GRAPH)

    def test_there_are_as_many_convolutions_as_the_graph_has(self, net):
        assert len(list(net.convolutions())) == ONNX_CONVOLUTIONS

    def test_the_parameter_count_matches_the_graphs_initialisers(self, net):
        """Batch norm is folded, as the graph stores it, so the two counts are comparable at all."""
        assert sum(p.numel() for p in net.parameters()) == ONNX_PARAMETER_VALUES

    def test_the_depthwise_convolutions_really_are_depthwise(self, net):
        depthwise = [m for m in net.convolutions() if m.groups > 1]

        assert depthwise, "LFE is depthwise-separable; a port with no grouped convolution is not LFE"
        for conv in depthwise:
            assert conv.groups == conv.in_channels
            assert conv.weight.shape[1] == 1


class TestItRunsOnAScan:
    def test_one_probability_per_beam(self, net):
        with torch.no_grad():
            out = net(torch.rand(2, BEAMS))

        assert out.shape == (2, BEAMS)

    def test_the_output_is_a_probability(self, net):
        with torch.no_grad():
            out = net(torch.rand(4, BEAMS) * 10.0)

        assert torch.all((out >= 0.0) & (out <= 1.0))

    def test_it_accepts_any_beam_count_so_drow_scans_fit(self, net):
        with torch.no_grad():
            assert net(torch.rand(1, 450)).shape == (1, 450)

    def test_a_scan_is_processed_independently_of_the_others_in_the_batch(self, net):
        """Their global-max branches pool over beams, which must not pool over the batch."""
        scans = torch.rand(3, BEAMS)

        with torch.no_grad():
            batched = net(scans)
            alone = torch.cat([net(scans[i:i + 1]) for i in range(len(scans))])

        assert torch.allclose(batched, alone, atol=1e-6)
