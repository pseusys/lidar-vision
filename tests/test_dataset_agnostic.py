"""
Unit tests for the dataset-agnostic half of stage 2: a beam count the U-Net's
three halvings do not divide, and the sector count the memory's prior is built
at.

The failure these exist for is not hypothetical. FROG's 720 beams and a FROG
scan decimated to 0.5 deg (360) both divide by 8; **DROW's 450 does not**, and
the network does not merely lose precision there -- it raises, because
`Conv1d(kernel 2, stride 2)` floors 225 to 112 on the way down while
`interpolate(scale_factor=2)` doubles 112 to 224 on the way up, and the skip
connection it is concatenated with is 225 wide.

The guarantee that keeps every FROG number intact is the first test below: on a
beam count that already divides, the padding is zero and the tensor that reaches
the U-Net is the one that always reached it.
"""
import pytest
import torch

from takheliped.detectors.takheliped import (
    BEAM_DIVISOR, CalibrationNetwork, beam_pad, free_range_profile, pad_beams, padded_beam_count, render_prior,
    sector_count, visibility_feature,
)
from takheliped.datasets.frog_dataset import frog_laser_angles
from takheliped.utils.drow_utils import laser_angles as drow_laser_angles


FROG_BEAMS = 720
DROW_BEAMS = 450
DECIMATED_FROG_BEAMS = 360


def _sector_of_each_beam(angles, sectors):
    """Which sector a slot standing in the middle of each beam's angular cell is drawn into, read out of `render_prior`.

    One slot per batch row, so the row's one-hot prior names the sector rather than merging with its neighbours.
    The middle of the cell rather than the beam's own bearing, because a sector boundary falls exactly on a beam:
    reconstructing that bearing through sin/cos and back through `atan2` lands a half-ulp either side of it, which
    decides the `floor` and would make this test measure float reconstruction rather than the geometry.
    """
    n = len(angles)
    middle = angles + (angles[1] - angles[0]) / 2
    xy = torch.stack([-torch.sin(middle), torch.cos(middle)], dim=-1).reshape(n, 1, 2) * 3.0
    ones = torch.ones(n, 1)
    prior = render_prior(xy, ones, ones, torch.ones(n, 1, dtype=torch.bool), angles, sectors)
    assert torch.equal(prior[:, 0].amax(dim=-1), ones[:, 0]), "a beam's own bearing read as outside the field of view"
    return prior[:, 0].argmax(dim=-1)


def _net(n_beams, angles, **kwargs):
    """A small stage 2 on `n_beams`: the widths are irrelevant to shape bookkeeping and keep the test fast."""
    torch.manual_seed(0)
    return CalibrationNetwork(torch.from_numpy(angles), channels=(4, 8, 16), blocks=(1, 1, 1),
                              bottleneck_blocks=1, **kwargs)


class TestABeamCountTheHalvingsDoNotDivide:
    @pytest.mark.parametrize("n", [FROG_BEAMS, DECIMATED_FROG_BEAMS, 8, 456])
    def test_a_grid_that_already_divides_is_passed_through_untouched(self, n):
        """The regression guard for every FROG number: no padding means the identical tensor, not an equal copy."""
        assert padded_beam_count(n) == n
        x = torch.randn(2, 3, n)
        assert pad_beams(x, n) is x

    @pytest.mark.parametrize("n,expected", [(DROW_BEAMS, 456), (449, 456), (451, 456), (7, 8), (1, 8)])
    def test_a_grid_that_does_not_divide_rounds_up_to_the_next_multiple(self, n, expected):
        assert padded_beam_count(n) == expected
        assert expected % BEAM_DIVISOR == 0

    def test_the_padding_is_split_evenly_so_a_mirrored_scan_stays_mirrored(self):
        """`--mirror` reflects the scan about its centre, so uneven padding would make the augmentation asymmetric."""
        x = torch.randn(1, 1, DROW_BEAMS)
        padded = pad_beams(x, DROW_BEAMS)
        assert padded.shape[-1] == 456
        left = (456 - DROW_BEAMS) // 2
        assert torch.equal(padded[..., left:left + DROW_BEAMS], x)

    def test_a_drow_scan_gets_one_logit_and_one_vote_per_real_beam(self):
        """The red test: 450 beams in, 450 beams out, nothing of the padding visible to the caller."""
        net = _net(DROW_BEAMS, drow_laser_angles(DROW_BEAMS))
        logit, votes, features = net(torch.rand(2, DROW_BEAMS) * 10.0)
        assert logit.shape == (2, DROW_BEAMS)
        assert votes.shape == (2, DROW_BEAMS, 2)
        assert features.shape[0] == 2 and features.shape[-1] == DROW_BEAMS

    @pytest.mark.parametrize("n,angles_fn", [(FROG_BEAMS, frog_laser_angles),
                                             (DECIMATED_FROG_BEAMS, frog_laser_angles),
                                             (DROW_BEAMS, drow_laser_angles)])
    def test_every_grid_this_project_trains_on_runs(self, n, angles_fn):
        net = _net(n, angles_fn(n))
        logit, votes, _ = net(torch.rand(1, n) * 10.0)
        assert logit.shape == (1, n) and votes.shape == (1, n, 2)


class TestWhereTheSectorCountComesFrom:
    @pytest.mark.parametrize("n,expected", [(FROG_BEAMS, 90), (DECIMATED_FROG_BEAMS, 45), (DROW_BEAMS, 57)])
    def test_sectors_are_the_padded_beam_count_over_the_divisor(self, n, expected):
        """FROG's 90 is what every stored candidate cache is keyed on, so it must not move."""
        assert sector_count(n) == expected

    def test_the_memory_prior_is_read_at_the_padded_sector_count(self):
        """Stage 3 draws its prior per sector; on DROW that is 57, and the network has to accept exactly that."""
        net = _net(DROW_BEAMS, drow_laser_angles(DROW_BEAMS), prior_channels=2)
        prior = torch.zeros(2, 2, sector_count(DROW_BEAMS))
        logit, _, _ = net(torch.rand(2, DROW_BEAMS) * 10.0, prior)
        assert logit.shape == (2, DROW_BEAMS)

    def test_a_prior_at_the_unpadded_count_is_refused_rather_than_broadcast(self):
        """56 sectors is what 450 // 8 gives, and silently accepting it would misalign every slot's bearing."""
        net = _net(DROW_BEAMS, drow_laser_angles(DROW_BEAMS), prior_channels=2)
        with pytest.raises(RuntimeError):
            net(torch.rand(2, DROW_BEAMS) * 10.0, torch.zeros(2, 2, DROW_BEAMS // BEAM_DIVISOR))


class TestThePaddingIsNotReadAsAPerson:
    def test_the_padded_beams_carry_no_return(self):
        """Padding with zeros would place a surface at the sensor's own origin, which is a person-shaped artefact.

        The pad is replicate-on-features, so what the U-Net sees at the ends is the end beam repeated -- the same
        thing every convolution in the network already does at its boundary.
        """
        x = torch.full((1, 1, DROW_BEAMS), 3.0)
        x[..., 0] = 7.0
        x[..., -1] = 9.0
        padded = pad_beams(x, DROW_BEAMS)
        assert torch.equal(padded[..., :3], torch.full((1, 1, 3), 7.0))
        assert torch.equal(padded[..., -3:], torch.full((1, 1, 3), 9.0))

    def test_the_crop_returns_the_beams_the_padding_pushed_aside(self, monkeypatch):
        """The crop offset, pinned exactly rather than inferred from how the output moves.

        An earlier version of this test perturbed one stretch of beams and asserted that distant beams did not
        move. That premise is wrong by design: the whole-scan max aggregator is *documented* as the one path by
        which a return at one end of the field of view reaches the other end's logits, so every beam is global.
        Here `_decode` is replaced by one returning each beam's own index, which says where the crop lands and
        nothing else.
        """
        net = _net(DROW_BEAMS, drow_laser_angles(DROW_BEAMS))
        width = padded_beam_count(DROW_BEAMS)

        def ramp(bottom, skips):
            index = torch.arange(width, dtype=torch.float32).expand(1, width)
            return index, index.unsqueeze(-1).expand(1, width, 2), index.unsqueeze(1)

        monkeypatch.setattr(net, "_decode", ramp)
        logit, votes, features = net(torch.rand(1, DROW_BEAMS) * 10.0)
        left = (width - DROW_BEAMS) // 2
        expected = torch.arange(left, left + DROW_BEAMS, dtype=torch.float32)
        assert torch.equal(logit[0], expected)
        assert torch.equal(votes[0, :, 0], expected)
        assert torch.equal(features[0, 0], expected)


class TestSectorGeometryOnThePaddedGrid:
    """The padding reaches past the U-Net: the sector profile and the memory's prior are built on the padded axis too.

    `free_range_profile` reshapes the beam axis into sectors, and `render_prior`/`visibility_feature` derive a
    sector's angular width from the beam count. On 450 beams and 57 sectors the reshape does not divide and the
    width is 7.89 beams where stage 2's bottleneck sector is 8 -- so a slot's bearing would be drawn into a
    neighbouring sector from the one the convolution put its evidence in.
    """

    def test_a_drow_scan_gets_one_free_range_value_per_bottleneck_sector(self):
        sectors = sector_count(DROW_BEAMS)
        ranges = torch.arange(1, DROW_BEAMS + 1, dtype=torch.float32)[None] * 0.05
        profile = free_range_profile(ranges, sectors, max_range_m=29.99)
        assert profile.shape == (1, sectors)
        assert torch.equal(profile[0], pad_beams(ranges, DROW_BEAMS).reshape(-1, BEAM_DIVISOR).amin(dim=-1))

    def test_the_padding_never_invents_an_obstacle_nearer_than_the_scan_holds(self):
        """A zero pad would put a surface at the sensor's own origin in the end sectors, which reads as an occluder."""
        sectors = sector_count(DROW_BEAMS)
        ranges = torch.full((1, DROW_BEAMS), 20.0)
        ranges[0, 0], ranges[0, -1] = 5.0, 6.0
        profile = free_range_profile(ranges, sectors, max_range_m=29.99)[0]
        assert profile[0].item() == pytest.approx(5.0) and profile[-1].item() == pytest.approx(6.0)
        assert profile[1:-1].tolist() == pytest.approx([20.0] * (sectors - 2))

    @pytest.mark.parametrize("n,angles_fn", [(FROG_BEAMS, frog_laser_angles),
                                             (DECIMATED_FROG_BEAMS, frog_laser_angles),
                                             (DROW_BEAMS, drow_laser_angles)])
    def test_a_slot_is_drawn_into_the_sector_its_beam_feeds(self, n, angles_fn):
        """The invariant that ties the prior to the convolution: sector = (beam + left pad) // 8, every grid.

        On FROG the left pad is 0, so this is the formula the code has always computed and the regression guard for
        every stored stage-3 run; on DROW it is 3, and ignoring it misplaces the prior by up to half a sector.
        """
        angles = torch.from_numpy(angles_fn(n))
        left, _ = beam_pad(n)
        expected = (torch.arange(n) + left) // BEAM_DIVISOR
        assert torch.equal(_sector_of_each_beam(angles, sector_count(n)), expected)

    def test_visibility_reads_the_sector_the_slot_points_into_on_a_drow_grid(self):
        """The occlusion evidence and the free-range profile have to agree about which sector is which."""
        angles = torch.from_numpy(drow_laser_angles(DROW_BEAMS))
        sectors = sector_count(DROW_BEAMS)
        free = torch.full((1, sectors), 29.99)
        free[0, 0] = 1.0
        bearing = angles[0]
        slot = torch.tensor([[[-torch.sin(bearing) * 5.0, torch.cos(bearing) * 5.0]]])
        assert visibility_feature(slot, free, angles, sectors, scale=2.0)[0, 0].item() < 0
        free[0, 0] = 29.99
        assert visibility_feature(slot, free, angles, sectors, scale=2.0)[0, 0].item() > 0
