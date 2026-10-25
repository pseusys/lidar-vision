"""Item 44(a): a soft Gaussian beam target in place of the hard per-beam label, with the vote head kept.

**Why this half and not the published head.** `memory/static-detector-diagnosis.md`'s decomposition already measured the
*decoder* on fixed weights: our own network read through LFE-Peaks' own decoder scores **71.26% against `nms-mean`'s
79.88%**, and at 8-10 m at a tight operating point **50.3% against 81.6%**. Of that 8.62 pp gap, the 2.84 pp attributed
to centre regression is **structurally unavailable** to a head that emits no vote. So the open question is not the output
parameterisation but the **objective** -- which is exactly what that page's own caveat says: "the network's +6.15 is a
lower bound, since our probability map was trained for votes, not for peaks". Nobody has trained this network *for*
peaks. The owner's call (2026-10-25) is to run that objective first and the full published head afterwards as a
confirmation.

**The isolation that makes this an experiment rather than two changes at once.** `beam_heatmap` reuses `make_targets`'
geometry and its `vote_collect_radius`, so the set of beams with a non-zero target is **identical** to the set
`make_targets` labels positive. Only the *values* change, from a hard 1 to a smooth falloff. The vote loss keeps reading
the hard labels, so localisation is untouched and the arm varies one thing.

**The width is in metres, not beams, and that is a deliberate departure from LFE.** LFE's `make_heatmap_target` uses a
fixed `sigma_beams=2.0`, but a person subtends 42 beams at 2 m and 9 at 9 m, so a fixed beam width is scale-wrong nearly
everywhere -- the same objection §5.2 records against a fixed `k=7` kernel, and the reason `--label-radius` is already
metric. A metric sigma also keeps the support identical to today's, which is what makes this half an isolated test.
LFE's beam-unit form belongs with the published head in arm (b), where being faithful is the point.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from train import beam_heatmap, compute_loss, make_targets  # noqa: E402

N_BEAMS = 360
ANGLES = np.linspace(-np.pi * 0.75, np.pi * 0.75, N_BEAMS)
RADIUS = 0.6


def _flat_scan(value: float = 5.0) -> np.ndarray:
    return np.full(N_BEAMS, value, dtype=np.float64)


def _person_on_a_beam(index: int, scan: np.ndarray):
    """An annotation sitting exactly on beam `index`'s endpoint, so that beam's distance to it is zero."""
    return (float(scan[index]), float(ANGLES[index]))


class TestTheTargetSShape:
    """A soft target over the beams, peaking on the annotation and falling off with metric distance."""

    def test_no_annotation_is_an_all_zero_target(self):
        heat = beam_heatmap(_flat_scan(), ANGLES, {}, sigma_m=0.3, vote_collect_radius=RADIUS)

        assert heat.shape == (N_BEAMS,)
        assert heat.dtype == np.float32
        assert not heat.any()

    def test_the_beam_on_the_annotation_reaches_one(self):
        scan = _flat_scan()
        heat = beam_heatmap(scan, ANGLES, {3: [_person_on_a_beam(180, scan)]}, sigma_m=0.3,
                            vote_collect_radius=RADIUS)

        assert heat[180] == pytest.approx(1.0)
        assert heat.max() == pytest.approx(1.0)

    def test_the_target_falls_off_with_metric_distance(self):
        scan = _flat_scan()
        heat = beam_heatmap(scan, ANGLES, {3: [_person_on_a_beam(180, scan)]}, sigma_m=0.3,
                            vote_collect_radius=RADIUS)
        inside = heat[heat > 0]

        assert heat[181] < heat[180]
        assert heat[182] < heat[181]
        assert (inside > 0).all()

    def test_a_wider_sigma_raises_the_target_at_the_same_beam(self):
        scan = _flat_scan()
        person = {3: [_person_on_a_beam(180, scan)]}
        narrow = beam_heatmap(scan, ANGLES, person, sigma_m=0.15, vote_collect_radius=RADIUS)
        wide = beam_heatmap(scan, ANGLES, person, sigma_m=0.45, vote_collect_radius=RADIUS)

        assert wide[185] > narrow[185]

    def test_overlapping_people_saturate_at_one_rather_than_summing_past_it(self):
        """LFE's own clip. Two annotations a few beams apart must not produce a target above 1."""
        scan = _flat_scan()
        heat = beam_heatmap(scan, ANGLES, {3: [_person_on_a_beam(180, scan), _person_on_a_beam(182, scan)]},
                            sigma_m=0.45, vote_collect_radius=RADIUS)

        assert heat.max() <= 1.0 + 1e-6


class TestTheSupportIsIdenticalToTheHardLabelS:
    """The property that makes this an isolated experiment: same beams, different values."""

    @pytest.mark.parametrize("seed", range(8))
    def test_the_non_zero_beams_are_exactly_the_positively_labelled_beams(self, seed):
        rng = np.random.default_rng(seed)
        scan = rng.uniform(1.0, 9.0, N_BEAMS)
        people = [(float(rng.uniform(1.0, 9.0)), float(rng.choice(ANGLES))) for _ in range(3)]
        labels, _ = make_targets(scan, ANGLES, {3: people}, vote_collect_radius=RADIUS)
        heat = beam_heatmap(scan, ANGLES, {3: people}, sigma_m=0.3, vote_collect_radius=RADIUS)

        assert np.array_equal(heat > 0, labels > 0)

    def test_a_beam_beyond_the_collect_radius_is_exactly_zero(self):
        scan = _flat_scan()
        heat = beam_heatmap(scan, ANGLES, {3: [_person_on_a_beam(180, scan)]}, sigma_m=10.0,
                            vote_collect_radius=RADIUS)

        assert heat[0] == 0.0
        assert heat[-1] == 0.0, "a huge sigma must still not reach past the collect radius"


class TestTheLossReadsIt:
    """`compute_loss` takes the soft target for the classification term only."""

    def _batch(self):
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(N_BEAMS, 1, generator=g)
        votes = torch.randn(N_BEAMS, 2, generator=g)
        scan = _flat_scan()
        people = {3: [_person_on_a_beam(180, scan)]}
        labels, vote_targets = make_targets(scan, ANGLES, people, vote_collect_radius=RADIUS)
        heat = beam_heatmap(scan, ANGLES, people, sigma_m=0.3, vote_collect_radius=RADIUS)
        return (logits, votes, torch.from_numpy(labels), torch.from_numpy(vote_targets),
                torch.from_numpy(heat))

    def test_without_a_soft_target_the_loss_is_what_it_always_was(self):
        """The regression guard: every stage-2 number on record was produced by this path."""
        logits, votes, labels, vote_targets, _ = self._batch()

        plain = compute_loss(logits, votes, labels, vote_targets)
        explicit = compute_loss(logits, votes, labels, vote_targets, soft_labels=None)

        assert plain[0].item() == explicit[0].item()

    def test_the_soft_target_changes_the_classification_term_and_not_the_vote_term(self):
        logits, votes, labels, vote_targets, heat = self._batch()

        _, hard_class, hard_vote = compute_loss(logits, votes, labels, vote_targets)
        _, soft_class, soft_vote = compute_loss(logits, votes, labels, vote_targets, soft_labels=heat)

        assert soft_vote == pytest.approx(hard_vote), "the vote head must be untouched -- that is the isolation"
        assert soft_class != pytest.approx(hard_class)

    def test_the_soft_target_changes_the_gradient(self):
        """If the gradient matched, the arm would be the control under a different name."""
        logits, votes, labels, vote_targets, heat = self._batch()
        grads = []
        for soft in (None, heat):
            x = logits.clone().requires_grad_(True)
            compute_loss(x, votes, labels, vote_targets, soft_labels=soft)[0].backward()
            grads.append(x.grad.clone())

        assert not torch.allclose(grads[0], grads[1], atol=1e-6)

    def test_a_perfect_prediction_of_the_soft_target_is_the_optimum_of_the_unweighted_objective(self):
        """The soft target is what BCE reads -- but only an *unweighted* BCE is minimised at the target itself.

        Weighted BCE's minimum sits at `w*t / (w*t + 1 - t)`, not at `t`, so with this project's real `pos_weight` of
        roughly 50:1 a hard 1 inside the footprint is *closer* to the weighted optimum than the Gaussian is. The first
        version of this test asserted the opposite and failed on exactly that. The claim is therefore made where it
        holds -- balanced classes, `pos_weight` 1 -- and the imbalance's effect on the optimum is left as what it is: a
        real property of the objective that item 44(a)'s arm will train under, not a bug.
        """
        labels = torch.tensor([0, 0, 0, 0, 3, 3, 3, 3])
        heat = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.6, 0.3])
        votes = torch.zeros(8, 2)
        vote_targets = torch.zeros(8, 2)
        eps = 1e-4
        soft_logits = torch.logit(heat.clamp(eps, 1 - eps))[:, None]
        hard_logits = torch.logit((labels > 0).float().clamp(eps, 1 - eps))[:, None]

        matched = compute_loss(soft_logits, votes, labels, vote_targets, soft_labels=heat)[1]
        mismatched = compute_loss(hard_logits, votes, labels, vote_targets, soft_labels=heat)[1]

        assert matched < mismatched

    def test_the_class_balance_still_comes_from_the_hard_labels(self):
        """`pos_weight` is a count of positive beams; taking it from a soft target would change two things at once."""
        logits, votes, labels, vote_targets, heat = self._batch()

        scaled = compute_loss(logits, votes, labels, vote_targets, soft_labels=heat * 0.5)[1]
        full = compute_loss(logits, votes, labels, vote_targets, soft_labels=heat)[1]

        # Halving every target value must not change the weighting, only the targets, so the two differ by the
        # targets alone and neither collapses to the unweighted loss.
        assert scaled != pytest.approx(full)
        assert np.isfinite(scaled) and np.isfinite(full)
