"""The decision head as a swappable part: `binary` unchanged, `multiclass` able to say which class a beam is.

Stage 2 has always emitted one person logit plus a two-channel vote, while `DatasetSpec` has recorded three
classes and printed all three in every parameter table. On DROW that silence costs something real -- 56.4% of
its annotated test objects are wheelchairs or walkers, trained as background -- so the head becomes a named
component the dataset selects, the way `--block` and `--vote-frame` already are.

Two things are pinned hardest here, because the whole FROG record depends on them:

* `binary` is **bit-identical** to the head that produced every number on file -- same channel count, same
  state-dict keys, same 976,707 parameters at 720 beams, same 2-D logit shape.
* a one-class `sigmoid` head is **numerically identical** to `binary`, not merely close, so the multi-class
  path is a strict generalisation and a binary checkpoint can be carried into it unchanged.

A `softmax` head is not a generalisation of `binary` -- it adds a background column and the denominator changes
-- so loading a binary checkpoint into one is refused rather than padded.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from takheliped.detectors.takheliped import (CLASS_NORMS, HEADS, CalibrationNetwork,  # noqa: E402
                                             class_channels, load_calibration_weights, person_column)
from train import compute_loss, soft_class_target  # noqa: E402
from train_takheliped import DROW, FROG, class_columns, network_shape  # noqa: E402


def _angles(n: int) -> torch.Tensor:
    return torch.from_numpy(np.linspace(-np.pi / 2, np.pi / 2, n, dtype=np.float32))


def net(**kwargs) -> CalibrationNetwork:
    """A small network, so these tests run on the CPU in milliseconds."""
    return CalibrationNetwork(_angles(48), channels=(8, 16, 32), blocks=(1, 1, 1), bottleneck_blocks=1, **kwargs).eval()


def scan(beams: int = 48, batch: int = 2) -> torch.Tensor:
    return torch.full((batch, beams), 4.0)


class TestTheHeadIsANamedPartLikeTheBlockAndTheVoteFrame:
    def test_both_heads_and_both_normalisations_are_named(self):
        assert HEADS == ("binary", "multiclass")
        assert CLASS_NORMS == ("softmax", "sigmoid")

    @pytest.mark.parametrize("bad", ["person", "", "Binary"])
    def test_an_unknown_head_is_refused_at_construction(self, bad):
        with pytest.raises(ValueError, match="head"):
            net(head=bad)

    def test_an_unknown_normalisation_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="class_norm"):
            net(head="multiclass", class_norm="logistic", classes=3)

    @pytest.mark.parametrize("head,norm,classes,columns", [
        ("binary", "softmax", 1, 1),
        ("binary", "softmax", 3, 1),          # the class count cannot widen a binary head
        ("multiclass", "sigmoid", 1, 1),
        ("multiclass", "sigmoid", 3, 3),
        ("multiclass", "softmax", 1, 2),      # one class plus background
        ("multiclass", "softmax", 3, 4),
    ])
    def test_the_column_count_is_the_head_the_norm_and_the_class_count(self, head, norm, classes, columns):
        assert class_channels(head, norm, classes) == columns
        assert net(head=head, class_norm=norm, classes=classes).head.out_channels == columns + 2

    @pytest.mark.parametrize("head,norm,column", [
        ("binary", "softmax", 0), ("multiclass", "sigmoid", 0), ("multiclass", "softmax", 1)])
    def test_the_person_is_the_first_class_column_after_any_background_one(self, head, norm, column):
        """`DatasetSpec.classes` lists the person class first, so the person's column does not vary by dataset."""
        assert person_column(head, norm) == column
        assert net(head=head, class_norm=norm, classes=3).person_column == column


class TestBinaryIsBitIdenticalToTheHeadThatProducedEveryNumberOnFile:
    def test_the_default_is_binary_and_emits_three_channels(self):
        assert net().head.out_channels == 3
        assert net().head_kind == "binary"

    def test_the_parameter_count_of_the_adopted_network_is_unchanged(self):
        """The same guard `test_takheliped.py` carries, repeated here because this file is what could break it."""
        assert sum(p.numel() for p in CalibrationNetwork(_angles(720)).parameters()) == 976_707

    def test_the_state_dict_keys_are_unchanged_so_every_checkpoint_still_loads(self):
        keys = set(net().state_dict())
        assert {"head.weight", "head.bias"} <= keys
        assert not [k for k in keys if "class" in k or "decision" in k]

    def test_the_logit_is_still_two_dimensional_so_nothing_downstream_changes(self):
        logit, votes, _ = net()(scan())
        assert logit.shape == (2, 48) and votes.shape == (2, 48, 2)

    def test_a_multiclass_head_returns_a_column_per_class_instead(self):
        logit, votes, _ = net(head="multiclass", class_norm="sigmoid", classes=3)(scan())
        assert logit.shape == (2, 48, 3) and votes.shape == (2, 48, 2)

    def test_person_logit_reads_either_shape(self):
        binary = net()
        multi = net(head="multiclass", class_norm="softmax", classes=3)
        assert binary.person_logit(binary(scan())[0]).shape == (2, 48)
        assert multi.person_logit(multi(scan())[0]).shape == (2, 48)


class TestOneClassSigmoidIsTheBinaryHeadAndNotMerelyCloseToIt:
    def _pair(self):
        source = net()
        target = net(head="multiclass", class_norm="sigmoid", classes=1)
        load_calibration_weights(target, source.state_dict())
        return source, target

    def test_the_person_logits_are_identical(self):
        source, target = self._pair()
        with torch.no_grad():
            mine, theirs = source(scan())[0], target(scan())[0]
        assert torch.equal(mine, target.person_logit(theirs))

    def test_the_votes_are_identical_because_the_head_shares_them(self):
        source, target = self._pair()
        with torch.no_grad():
            assert torch.equal(source(scan())[1], target(scan())[1])

    def test_a_three_class_sigmoid_head_takes_the_binary_weights_with_the_extra_columns_silent(self):
        """`prior_channels`' rule applied to the head: a widened network computes what the checkpoint did."""
        source = net()
        target = net(head="multiclass", class_norm="sigmoid", classes=3)
        load_calibration_weights(target, source.state_dict())
        with torch.no_grad():
            assert torch.equal(source(scan())[0], target.person_logit(target(scan())[0]))
            extra = target(scan())[0][..., 1:]
        assert torch.equal(extra, torch.zeros_like(extra)), "an untrained class must start silent, not noisy"

    def test_a_softmax_head_refuses_a_binary_checkpoint_rather_than_padding_it(self):
        """The denominator changes, so padded weights would not compute what the checkpoint did."""
        with pytest.raises(ValueError, match="softmax"):
            load_calibration_weights(net(head="multiclass", class_norm="softmax", classes=3), net().state_dict())

    def test_a_multiclass_checkpoint_is_refused_by_a_binary_network(self):
        source = net(head="multiclass", class_norm="sigmoid", classes=3)
        with pytest.raises((ValueError, RuntimeError)):
            load_calibration_weights(net(), source.state_dict())


class TestTheLossTakesTheNormalisationAndOneClassReproducesTheBinaryTerm:
    def _batch(self, columns: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        logits = torch.randn(64, columns, generator=g)
        votes = torch.randn(64, 2, generator=g)
        labels = torch.randint(0, 2, (64,), generator=g) * 1  # column 1 is the person class
        targets = torch.randn(64, 2, generator=g)
        return logits, votes, labels, targets

    def test_one_column_is_the_binary_path_whatever_the_normalisation_says(self):
        logits, votes, labels, targets = self._batch(1)
        binary = compute_loss(logits, votes, labels, targets)[0]
        for norm in CLASS_NORMS:
            assert torch.equal(compute_loss(logits, votes, labels, targets, class_norm=norm)[0], binary)

    def test_a_sigmoid_head_at_one_class_is_numerically_the_binary_loss(self):
        """The head emits one column for one class, so this is the same assertion from the loss's side."""
        logits, votes, labels, targets = self._batch(1)
        assert torch.equal(compute_loss(logits, votes, labels, targets, class_norm="sigmoid")[0],
                           compute_loss(logits, votes, labels, targets)[0])

    def test_the_softmax_path_is_what_it_was_so_the_default_cannot_drift(self):
        logits, votes, labels, targets = self._batch(4)
        assert torch.equal(compute_loss(logits, votes, labels, targets, class_norm="softmax")[0],
                           compute_loss(logits, votes, labels, targets)[0])

    def test_sigmoid_scores_each_class_independently_against_a_one_hot_target(self):
        """A beam labelled class 2 is positive in column 2 and negative in the others, so raising column 2's
        logit must lower the loss while raising column 1's must raise it."""
        labels = torch.tensor([2, 2, 0, 0])
        votes, targets = torch.zeros(4, 2), torch.zeros(4, 2)
        base = torch.zeros(4, 3)
        loss = lambda z: compute_loss(z, votes, labels, targets, class_norm="sigmoid")[0]  # noqa: E731
        right, wrong = base.clone(), base.clone()
        right[:2, 1] = 3.0     # column index 1 is class 2 (column 0 is class 1)
        wrong[:2, 0] = 3.0
        assert loss(right) < loss(base) < loss(wrong)

    def test_sigmoid_needs_a_column_per_class_and_says_so(self):
        labels = torch.tensor([3, 0])
        with pytest.raises(ValueError, match="class"):
            compute_loss(torch.zeros(2, 2), torch.zeros(2, 2), labels, torch.zeros(2, 2), class_norm="sigmoid")

    def test_the_soft_person_target_survives_a_multiclass_head(self):
        """The adopted recipe's target is a metric Gaussian on the person class, so a head that cannot take one
        would differ from the headline in two things at once and price neither."""
        logits, votes, labels, targets = self._batch(3)
        soft = torch.rand(64)
        for norm in CLASS_NORMS:
            value = compute_loss(logits, votes, labels, targets, soft_labels=soft, class_norm=norm)[0]
            assert torch.isfinite(value)

    def test_a_sigmoid_head_at_one_class_reproduces_the_soft_binary_target_exactly(self):
        """The strict-special-case rule has to hold for the adopted recipe too, not just for a hard label."""
        logits, votes, labels, targets = self._batch(1)
        soft = torch.rand(64)
        assert torch.equal(compute_loss(logits, votes, labels, targets, soft_labels=soft, class_norm="sigmoid")[0],
                           compute_loss(logits, votes, labels, targets, soft_labels=soft)[0])

    def test_the_sigmoid_branchs_pos_weight_is_counted_from_the_hard_labels(self):
        """`compute_loss` promises that `pos_weight` under a soft target is still a count of positive *beams*,
        "because it is a count of positive beams and taking it from a soft target would move the class balance
        as well as the target". The sigmoid branch summed its positives *after* the person column had been
        overwritten with the Gaussian, so that column's weight was its Gaussian mass -- the hard label marks a
        person's whole extent at 1 while the Gaussian decays, so the weight came out larger and the arm fired
        harder. That made `heads_sigmoid`'s +2.65 pp on DROW a head change and a class-balance change at once
        (2026-11-07)."""
        logits, votes, _, targets = self._batch(3)
        g = torch.Generator().manual_seed(7)
        labels = torch.randint(0, 4, (64,), generator=g)
        soft = torch.rand(64, generator=g)
        hard = torch.nn.functional.one_hot(labels, 4)[..., 1:].to(logits.dtype)
        scored = hard.clone()
        scored[:, 0] = soft                                     # the person column, as the branch writes it
        counts = hard.sum(0)                                    # the beams, not the Gaussian's mass
        weight = torch.where(counts > 0, (hard.shape[0] - counts) / counts.clamp(min=1.0),
                             torch.ones_like(counts))
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, scored, pos_weight=weight)
        assert compute_loss(logits, votes, labels, targets, soft_labels=soft,
                            class_norm="sigmoid")[1] == pytest.approx(expected.item())

    def test_the_soft_target_lands_on_the_person_column_and_not_on_a_wheelchairs(self):
        """A beam near a person is a soft person positive whatever else it is labelled; raising the person
        column's logit where the Gaussian is high must cost less than raising it where the Gaussian is zero."""
        labels = torch.zeros(4, dtype=torch.long)
        votes, targets = torch.zeros(4, 2), torch.zeros(4, 2)
        soft = torch.tensor([1.0, 1.0, 0.0, 0.0])
        confident = torch.zeros(4, 3)
        confident[:2, 0] = 4.0          # sigmoid column 0 is the person class
        wrong = torch.zeros(4, 3)
        wrong[2:, 0] = 4.0
        loss = lambda z: compute_loss(z, votes, labels, targets, soft_labels=soft, class_norm="sigmoid")[0]  # noqa: E731
        assert loss(confident) < loss(wrong)

    def test_a_softmax_heads_soft_target_is_a_distribution_over_its_columns(self):
        """Background takes the complement, which is what the binary soft target means written as a row."""
        labels = torch.ones(2, dtype=torch.long)
        soft = torch.tensor([0.25, 1.0])
        rows = soft_class_target(labels, soft, columns=4)
        assert rows.shape == (2, 4)
        assert torch.allclose(rows.sum(-1), torch.ones(2))
        assert torch.allclose(rows[0], torch.tensor([0.75, 0.25, 0.0, 0.0]))
        assert torch.allclose(rows[1], torch.tensor([0.0, 1.0, 0.0, 0.0]))

    def test_a_softmax_row_for_another_class_keeps_that_class(self):
        rows = soft_class_target(torch.tensor([2]), torch.tensor([0.0]), columns=4)
        assert torch.allclose(rows[0], torch.tensor([0.0, 0.0, 1.0, 0.0]))

    def test_the_ignore_mask_still_works_under_both_normalisations(self):
        logits, votes, labels, targets = self._batch(3)
        ignore = torch.zeros(64, dtype=torch.bool)
        ignore[:8] = True
        for norm in CLASS_NORMS:
            kept = compute_loss(logits, votes, labels, targets, class_norm=norm, ignore=ignore)[0]
            assert torch.isfinite(kept) and not torch.equal(
                kept, compute_loss(logits, votes, labels, targets, class_norm=norm)[0])


class TestTheColumnsFollowTheSpecsOwnClassOrder:
    def test_each_spec_names_the_label_id_of_every_class_it_lists(self):
        """`make_targets` has used DROW's own ids since the baselines: 1 wheelchair, 2 walker, 3 person."""
        assert FROG.classes == ("pedestrian",) and FROG.class_ids == (3,)
        assert DROW.classes == ("pedestrian", "wheelchair", "walker") and DROW.class_ids == (3, 1, 2)

    def test_a_label_becomes_the_column_of_its_class_and_background_stays_zero(self):
        labels = np.array([0, 3, 1, 2, 0])
        assert class_columns(labels, DROW.class_ids).tolist() == [0, 1, 2, 3, 0]

    def test_a_one_class_spec_maps_its_only_class_to_column_one(self):
        assert class_columns(np.array([0, 3, 3]), FROG.class_ids).tolist() == [0, 1, 1]

    def test_an_id_the_spec_does_not_list_is_dropped_rather_than_mislabelled(self):
        """A dataset whose `other_detections` are not all passed must not have them land on another class."""
        assert class_columns(np.array([3, 1]), FROG.class_ids).tolist() == [1, 0]


class TestTheCheckpointRecordsWhichHeadItIs:
    @pytest.mark.parametrize("stored,expected", [
        ({}, ("binary", "softmax", 1)),
        ({"head": "multiclass", "class_norm": "sigmoid", "classes": "3"}, ("multiclass", "sigmoid", 3)),
        ({"head": "None", "class_norm": "None", "classes": "None"}, ("binary", "softmax", 1)),
    ])
    def test_the_three_shape_keys_round_trip_through_network_shape(self, stored, expected):
        """A multi-class checkpoint rebuilt as binary would score nonsense, so these are shape keys, not options."""
        shape = network_shape(stored)
        assert (shape["head"], shape["class_norm"], shape["classes"]) == expected

    def test_a_checkpoint_from_before_the_flags_rebuilds_as_the_head_it_trained_with(self):
        small = {"channels": (8, 16, 32), "blocks": (1, 1, 1), "bottleneck_blocks": 1}
        built = CalibrationNetwork(_angles(48), **{**network_shape({"max_range_m": "10.0"}), **small})
        assert built.head_kind == "binary" and built.head.out_channels == 3
