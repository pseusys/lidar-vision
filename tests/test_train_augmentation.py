"""
Unit tests for training-time data augmentation in utils/train.py.

Same motivation as tests/test_geometry.py: this session has repeatedly found
real sign/index bugs in beam-index geometry, so any new transform of that
kind gets a hand-derived-ground-truth check before being trusted in a real
training run -- not just "the training loss went down".
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from train import (  # noqa: E402
    _shift_beams, _diff_encode, _global_normalize, _local_normalize,
    _jitter_range, _freeze_backbone, make_targets,
)
from follow_the_drow.detectors import SpaceTimeCNNDetector  # noqa: E402


class TestShiftBeams:
    def test_zero_shift_is_identity(self):
        t = torch.arange(10)
        assert torch.equal(_shift_beams(t, 0), t)

    def test_positive_shift_moves_content_to_higher_index(self):
        # new[i] = old[i - shift]: content at index 0 should appear at
        # index `shift` after a positive shift.
        t = torch.tensor([9, 1, 2, 3, 4, 5, 6, 7, 8, 0])
        out = _shift_beams(t, 3)
        # out[3] should equal in[0] = 9
        assert out[3].item() == 9
        # edge (indices 0..2) clamped to in[0]
        assert torch.equal(out[:3], torch.tensor([9, 9, 9]))

    def test_negative_shift_moves_content_to_lower_index(self):
        t = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        out = _shift_beams(t, -3)
        assert out[0].item() == 3          # in[3] moved to out[0]
        assert torch.equal(out[-3:], torch.tensor([9, 9, 9]))  # edge clamp

    def test_works_on_2d_tensor_shifting_only_axis_0(self):
        t = torch.arange(20).reshape(10, 2)
        out = _shift_beams(t, 2)
        assert torch.equal(out[2], t[0])
        assert out.shape == t.shape


class TestBeamShiftLabelConsistency:
    """The property that actually matters for training correctness: after a
    shift, the label/vote tensors built from make_targets() must still line
    up with where the corresponding scan feature ends up -- verified by
    shifting make_targets()'s own output, not by re-deriving the geometry."""

    def test_shifted_labels_track_a_shifted_spike(self):
        N = 181
        angles = np.linspace(-np.pi / 2, np.pi / 2, N, dtype=np.float32)
        scan = np.full(N, 10.0, dtype=np.float32)
        spike_idx = 90  # angle ~= 0, straight ahead
        scan[spike_idx] = 3.0
        r_gt, phi_gt = 3.0, float(angles[spike_idx])

        labels, votes = make_targets(scan, angles, {3: [(r_gt, phi_gt)]},
                                     vote_collect_radius=0.3)
        assert labels[spike_idx] == 3
        original_positive = int(np.argmax(labels == 3))

        shift = 15
        labels_t = torch.from_numpy(labels)
        shifted = _shift_beams(labels_t, shift)
        # the positive label must now sit at spike_idx + shift
        assert int(shifted[spike_idx + shift]) == 3
        assert int((shifted == 3).nonzero()[0]) == original_positive + shift


class TestDiffEncode:
    def test_last_slot_stays_the_raw_current_frame(self):
        r = np.arange(5 * 4).reshape(5, 4, 1).astype(np.float32)  # (T=5, N=4, 1)
        out = _diff_encode(r)
        np.testing.assert_array_equal(out[-1], r[-1])

    def test_earlier_slots_are_consecutive_differences(self):
        r = np.array([
            [[1.0], [2.0], [3.0], [4.0]],
            [[2.0], [2.0], [5.0], [4.0]],
            [[10.0], [2.0], [5.0], [0.0]],
        ], dtype=np.float32)  # (T=3, N=4, 1)
        out = _diff_encode(r)
        np.testing.assert_allclose(out[0], r[1] - r[0])
        np.testing.assert_allclose(out[1], r[2] - r[1])
        np.testing.assert_allclose(out[2], r[2])  # raw current frame, unchanged

    def test_static_background_nets_to_zero(self):
        # A beam whose range never changes across the window must read 0.0
        # in every diff channel -- the whole point of this encoding.
        r = np.full((5, 1, 1), 7.5, dtype=np.float32)
        out = _diff_encode(r)
        np.testing.assert_allclose(out[:-1], 0.0)
        np.testing.assert_allclose(out[-1], 7.5)

    def test_shape_preserved(self):
        r = np.zeros((5, 450, 1), dtype=np.float32)
        assert _diff_encode(r).shape == r.shape

    def test_single_frame_window_is_a_no_op(self):
        r = np.array([[[3.0]]], dtype=np.float32)  # T=1
        np.testing.assert_array_equal(_diff_encode(r), r)


class TestGlobalNormalize:
    def test_subtracts_current_frame_mean_from_every_slice(self):
        r = np.array([
            [[1.0], [2.0], [3.0]],   # older frame
            [[4.0], [6.0], [8.0]],   # current frame, mean=6.0
        ], dtype=np.float32)
        out = _global_normalize(r)
        np.testing.assert_allclose(out[1], [[-2.0], [0.0], [2.0]])  # current, re-centered
        np.testing.assert_allclose(out[0], [[-5.0], [-4.0], [-3.0]])  # same constant subtracted

    def test_relative_frame_to_frame_difference_is_unaffected(self):
        # Subtracting one constant everywhere must not change the *shape* of
        # motion between frames -- that's the property diff-encoding relies on.
        r = np.random.RandomState(0).rand(4, 20, 1).astype(np.float32) * 10
        out = _global_normalize(r)
        np.testing.assert_allclose(out[1:] - out[:-1], r[1:] - r[:-1], atol=1e-4)

    def test_zero_mean_current_frame_is_a_no_op(self):
        r = np.array([[[-1.0], [0.0], [1.0]]], dtype=np.float32)
        np.testing.assert_allclose(_global_normalize(r), r)


class TestLocalNormalize:
    def test_shape_preserved_odd_and_even_window(self):
        r = np.random.RandomState(0).rand(3, 50, 1).astype(np.float32)
        assert _local_normalize(r, window=21).shape == r.shape
        assert _local_normalize(r, window=20).shape == r.shape

    def test_uniform_frame_normalizes_to_zero_everywhere(self):
        # A perfectly flat current frame -- local mean equals the constant
        # everywhere (edge-replicate padding keeps it flat at the boundary
        # too), so every beam should net to ~0.
        r = np.full((2, 40, 1), 5.0, dtype=np.float32)
        out = _local_normalize(r, window=11)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_far_from_a_local_spike_reads_near_zero(self):
        # A single spike in an otherwise-flat current frame: beams well
        # outside the local window shouldn't "see" it, so they stay ~0.
        n = 100
        current = np.full(n, 5.0, dtype=np.float32)
        current[50] = 20.0
        r = np.stack([current], axis=0)[:, :, None].astype(np.float32)  # (1, N, 1)
        out = _local_normalize(r, window=11)
        assert abs(float(out[0, 10, 0])) < 1e-4   # far from beam 50
        assert abs(float(out[0, 50, 0])) > 1.0     # at the spike itself

    def test_differs_from_global_normalize_on_a_non_uniform_frame(self):
        # The whole point of local vs. global: different beams get different
        # reference values when the scene isn't flat.
        n = 60
        current = np.linspace(2.0, 10.0, n, dtype=np.float32)  # a ramp, not flat
        r = current[None, :, None]
        local_out = _local_normalize(r, window=9)
        global_out = _global_normalize(r)
        assert not np.allclose(local_out, global_out)


class TestJitterRange:
    def test_zero_magnitude_is_a_no_op(self):
        x = torch.rand(20, 5, 1)
        out = _jitter_range(x, scale_eps=0.0, offset_delta=0.0)
        assert torch.equal(out, x)

    def test_scale_jitter_stays_within_bounds(self):
        random.seed(0)
        x = torch.full((10, 1, 1), 5.0)
        for _ in range(200):
            out = _jitter_range(x, scale_eps=0.1, offset_delta=0.0)
            assert torch.all(out >= 5.0 * 0.9 - 1e-4)
            assert torch.all(out <= 5.0 * 1.1 + 1e-4)

    def test_offset_jitter_stays_within_bounds(self):
        random.seed(0)
        x = torch.full((10, 1, 1), 5.0)
        for _ in range(200):
            out = _jitter_range(x, scale_eps=0.0, offset_delta=0.2)
            assert torch.all(out >= 5.0 - 0.2 - 1e-4)
            assert torch.all(out <= 5.0 + 0.2 + 1e-4)

    def test_applies_uniformly_across_the_whole_tensor(self):
        # A single scalar scale+offset per call, not per-beam noise -- so
        # relative shape (e.g. a spike) must be preserved, only rescaled.
        random.seed(1)
        x = torch.tensor([1.0, 2.0, 4.0]).reshape(3, 1, 1)
        out = _jitter_range(x, scale_eps=0.2, offset_delta=0.1)
        ratios = (out[1:] - out[:-1]).flatten()
        # differences must still be proportional to the original differences
        # (same scale factor applied everywhere), not independently perturbed
        assert torch.allclose(ratios[1] / ratios[0], torch.tensor(2.0), atol=1e-3)


class TestFreezeBackbone:
    def test_only_head_params_stay_trainable(self):
        net = SpaceTimeCNNDetector(n_time=5, channels=8, n_spatial_stages=1)
        _freeze_backbone(net)
        for name, p in net.named_parameters():
            if name.startswith("head_"):
                assert p.requires_grad, f"{name} should stay trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_head_params_are_a_small_minority(self):
        # Sanity check that freezing actually does something meaningful --
        # if "head" params were most of the network this test would be moot.
        net = SpaceTimeCNNDetector(n_time=5, channels=32, n_spatial_stages=2)
        total = sum(p.numel() for p in net.parameters())
        head = sum(p.numel() for n, p in net.named_parameters() if n.startswith("head_"))
        assert head < total * 0.1
