"""
A frame's prediction must not depend on which other frames share its minibatch
-- TODO.md A18.

The bug these lock down: `SpaceTimeCNNDetector.forward()` reshaped a
`(B*N_beams, T, 1)` minibatch into a single `(1, 1, B*N_beams, T)` image, so a
batch of 4 scans became one 2880-beam "scan". GroupNorm and the SE gate then
pooled statistics across every frame in the batch, and the beam-axis
convolutions (receptive field 177 beams) convolved across the seam between one
frame's leftmost beams and the next frame's rightmost.

Measured before the fix, on the real val split at dtime=10, beam-level person
AUC rose monotonically with batch size -- 0.9644 (batch 1), 0.9696 (4), 0.9765
(16), 0.9810 (64) -- because evaluation batches are consecutive frames, so a
bigger batch leaked more temporal context. `evaluate.py` defaults to 16,
`train.py` used 4, and the robot runs `forward_one()` at batch 1.
"""
import numpy as np
import pytest
import torch
from follow_the_drow.detectors.full_scan import (
    FullScanTCNDetector,
    SpaceTimeCNNDetector,
    TemporalUNetDetector,
)

N_BEAMS = 128     # small but > the 177-beam RF's half-width matters only at the seam
T = 5


def _frames(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(N_BEAMS, T, 1, generator=g) * 8.0 for _ in range(n)]


def _build(cls, head="drow"):
    torch.manual_seed(0)
    if cls is SpaceTimeCNNDetector:
        net = cls(n_time=T, channels=96, n_spatial_stages=3, dropout=0.5, head=head)
    elif cls is FullScanTCNDetector:
        net = cls(n_time=T, dropout=0.5, head=head)
    else:
        net = cls(n_time=T, dropout=0.5, head=head)
    return net.eval()


ALL_MODELS = [SpaceTimeCNNDetector, FullScanTCNDetector, TemporalUNetDetector]


@pytest.mark.parametrize("cls", ALL_MODELS)
class TestBatchIndependence:
    def test_declares_beam_batch(self, cls):
        """Every full-scan model takes (B, N, T, C); the DataLoader must not
        flatten the batch into the beam axis on its behalf."""
        assert getattr(cls, "BEAM_BATCH", False) is True

    @pytest.mark.parametrize("batch", [2, 4, 16])
    def test_first_frame_output_is_unchanged_by_its_batch_mates(self, cls, batch):
        net = _build(cls)
        frames = _frames(batch)
        with torch.no_grad():
            solo = net(frames[0].unsqueeze(0))[0]
            batched = net(torch.stack(frames, dim=0))[0][:N_BEAMS]
        assert torch.allclose(solo, batched, atol=1e-5), (
            f"{cls.__name__}: max diff {(solo - batched).abs().max():.6f} at batch={batch}"
        )

    def test_every_frame_in_the_batch_matches_its_solo_run(self, cls):
        net = _build(cls)
        frames = _frames(8)
        with torch.no_grad():
            batched = net(torch.stack(frames, dim=0))[0]
            for i, f in enumerate(frames):
                solo = net(f.unsqueeze(0))[0]
                got = batched[i * N_BEAMS:(i + 1) * N_BEAMS]
                assert torch.allclose(solo, got, atol=1e-5), f"{cls.__name__} frame {i}"

    def test_batch_order_does_not_matter(self, cls):
        net = _build(cls)
        frames = _frames(4)
        with torch.no_grad():
            fwd = net(torch.stack(frames, dim=0))[0][:N_BEAMS]
            rev = net(torch.stack(frames[::-1], dim=0))[0][-N_BEAMS:]
        assert torch.allclose(fwd, rev, atol=1e-5)

    def test_seam_beams_are_not_special(self, cls):
        """Before the fix the last ~90 beams (adjacent to the next frame in the
        concatenation) moved 5x more than the rest."""
        net = _build(cls)
        frames = _frames(4)
        with torch.no_grad():
            solo = net(frames[0].unsqueeze(0))[0]
            batched = net(torch.stack(frames, dim=0))[0][:N_BEAMS]
        d = (solo - batched).abs().max(dim=1).values
        edge = d[-N_BEAMS // 8:].mean()
        middle = d[N_BEAMS // 3: 2 * N_BEAMS // 3].mean()
        assert edge <= middle + 1e-5

    def test_forward_accepts_an_unbatched_frame(self, cls):
        """forward_one() and the ROS node still hand over a bare (N, T, 1)."""
        net = _build(cls)
        with torch.no_grad():
            out = net(_frames(1)[0])[0]
        assert out.shape[0] == N_BEAMS

    def test_forward_one_matches_a_batch_of_one(self, cls):
        net = _build(cls)
        f = _frames(1)[0]
        confs, _ = net.forward_one(f.numpy())
        with torch.no_grad():
            logits = net(f.unsqueeze(0))[0]
            expected = torch.softmax(logits, dim=-1).numpy()
        assert np.allclose(confs, expected, atol=1e-5)


@pytest.mark.parametrize("cls", [SpaceTimeCNNDetector, FullScanTCNDetector])
def test_heatmap_head_is_also_batch_independent(cls):
    net = _build(cls, head="heatmap")
    frames = _frames(4)
    with torch.no_grad():
        (solo,) = net(frames[0].unsqueeze(0))
        (batched,) = net(torch.stack(frames, dim=0))
    assert torch.allclose(solo, batched[:N_BEAMS], atol=1e-5)
