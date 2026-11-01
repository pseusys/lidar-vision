"""Item 48: a post-hoc temperature and bias on the memory's output logit, so a fixed threshold means something.

**Why calibration and not retraining.** A51 phase 1b measured the rescore head shifting the *whole* score distribution
down -- a median -3.292 logits on people it keeps against -1.497 on people it loses -- which leaves rank-based AP
better while silently decalibrating any fixed threshold. At 0.22 the memory strictly dominates stage 2 (135,520 people
against 134,196, at 1.767 false positives per frame against 1.910), so the capability is already trained in and only
the operating point hides it. The repo's own conclusion was "the fix is calibration of the rescore head"; nothing was
ever built.

Done post-hoc the transform is **monotone**, and that is the whole safety argument: AP, recall at matched false
positives and every other rank-based number cannot move, so this cannot disturb the frozen headline (86.29%, item 47).
Implemented as a training loss it would change the weights and would have had to precede the freeze.

Three invariants carry the design, and each has a test here that would fail without it:

    the default is the identity     `(temperature, bias) = (1.0, 0.0)`, applied unconditionally, so a checkpoint
                                    written before this key loads and scores **bit-identically**. This is the rule
                                    `network_shape` and `object_memory` already follow, and the reason `--aggregator`
                                    still exists: a shape that defaults the wrong way makes an older checkpoint
                                    refuse its own weights.
    only the candidate logit        a silent coasting head is pinned at `SILENT_COAST_LOGIT`, not learned. Calibrating
                                    that too would let a bias term **un-silence a head the architecture removed**
                                    (item 24), inventing reports from slots. So the transform lives in `_rescored`.
    ranking is preserved exactly    asserted numerically through `score_detections`, not argued from monotonicity.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "library"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from takheliped.detectors.takheliped import SILENT_COAST_LOGIT, ObjectMemory, SlotRules  # noqa: E402

from calibrate_rescore import (  # noqa: E402
    AP_TOLERANCE, apply_calibration, check_labels_against_the_metric, fit_calibration, greedy_labels,
    labels_for_frames, negative_log_likelihood, to_logit, to_probability,
)
from train_takheliped import object_memory, score_detections  # noqa: E402

DIM, HEADS, CANDIDATES, CAPACITY = 32, 4, 6, 16


def _memory(**kwargs) -> ObjectMemory:
    """The frozen architecture's shape, small: no exchange, no learned match, minGRU, no coasting head."""
    torch.manual_seed(0)
    return ObjectMemory(SlotRules(capacity=CAPACITY), dim=DIM, heads=HEADS, recurrence="mingru",
                        exchange=False, learned_match=False, **kwargs)


def _inputs(batch: int = 2, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(batch, CANDIDATES, 2, generator=g) * 6 - 3,
            torch.rand(batch, CANDIDATES, generator=g),
            torch.ones(batch, CANDIDATES, dtype=torch.bool),
            torch.zeros(batch, 3))


def _play(model: ObjectMemory, steps: int = 3, batch: int = 2):
    """Several steps, so the carried state is non-trivial by the time the logit is read."""
    xy, score, valid, pose = _inputs(batch)
    slots = model.initial_state(batch)
    out = None
    for _ in range(steps):
        out = model.step(xy, score, valid, pose, torch.full((batch,), 0.025), slots)
        slots = out.slots
    return out


def _loud(model: ObjectMemory) -> ObjectMemory:
    """Open the gate, so the head's correction is actually present in the logit and the test is not reading `_logit` alone."""
    with torch.no_grad():
        model.rescore_gate.fill_(1.0)
        for p in model.rescore[-1].parameters():
            p.normal_(0.0, 0.5)
    return model


class TestTheDefaultIsTheIdentity:
    def test_the_constructor_default_is_one_and_zero(self):
        assert _memory().calibration == (1.0, 0.0)

    def test_passing_the_identity_explicitly_is_bit_identical_to_not_passing_it(self):
        a = _play(_loud(_memory()))
        b = _play(_loud(_memory(calibration=(1.0, 0.0))))
        assert torch.equal(a.candidate_logit, b.candidate_logit)

    def test_a_checkpoint_without_the_key_loads_as_the_identity(self):
        model = _memory()
        checkpoint = {"model": model.state_dict(),
                      "args": {"dim": str(DIM), "heads": str(HEADS), "recurrence": "mingru",
                               "exchange": "False", "learned_match": "False", "slots": str(CAPACITY)}}
        assert object_memory(checkpoint, 0).calibration == (1.0, 0.0)

    def test_a_checkpoint_carrying_the_key_restores_it(self):
        model = _memory()
        checkpoint = {"model": model.state_dict(),
                      "calibration": {"temperature": 2.5, "bias": -0.75},
                      "args": {"dim": str(DIM), "heads": str(HEADS), "recurrence": "mingru",
                               "exchange": "False", "learned_match": "False", "slots": str(CAPACITY)}}
        assert object_memory(checkpoint, 0).calibration == (2.5, -0.75)


class TestTheTransform:
    def test_the_bias_shifts_and_the_temperature_divides(self):
        plain = _play(_loud(_memory())).candidate_logit
        warmed = _play(_loud(_memory(calibration=(2.0, -1.5)))).candidate_logit
        torch.testing.assert_close(warmed, (plain - 1.5) / 2.0)

    def test_a_non_positive_temperature_is_refused(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="temperature"):
                _memory(calibration=(bad, 0.0))

    def test_the_order_of_candidates_is_unchanged(self):
        plain = _play(_loud(_memory())).candidate_logit
        for temperature, bias in ((0.5, 0.0), (3.0, 2.0), (1.0, -4.0), (0.1, 7.5)):
            moved = _play(_loud(_memory(calibration=(temperature, bias)))).candidate_logit
            assert torch.equal(plain.argsort(dim=-1), moved.argsort(dim=-1))


class TestOnlyTheCandidateLogit:
    def test_a_silent_coasting_head_stays_exactly_silent(self):
        """Without this the bias would un-silence a head item 24 removed, inventing slot reports from nothing."""
        out = _play(_memory(calibration=(0.25, 6.0)))
        assert torch.equal(out.slot_logit, torch.full_like(out.slot_logit, SILENT_COAST_LOGIT))
        assert not out.coasting.any()


class TestRankingSurvivesTheMetric:
    """The monotonicity argument is what makes item 48 post-hoc-safe, so it is asserted through the metric itself."""

    @staticmethod
    def _frames(seed: int = 3, frames: int = 40):
        g = np.random.default_rng(seed)
        out = []
        for _ in range(frames):
            people = g.uniform(-5, 5, size=(g.integers(1, 4), 2)).astype(np.float32)
            noise = g.uniform(-5, 5, size=(g.integers(2, 7), 2)).astype(np.float32)
            xy = np.concatenate([people + g.normal(0, 0.15, people.shape).astype(np.float32), noise])
            # People span the 0.3 threshold deliberately: with every person above it, recall at 0.3 saturates at 1.0
            # and `test_the_fixed_threshold_columns_do_move` cannot observe the thing it exists to observe.
            score = np.concatenate([g.uniform(0.15, 0.95, len(people)), g.uniform(0.01, 0.6, len(noise))]).astype(np.float32)
            out.append((score, xy, people))
        return out

    @staticmethod
    def _recalibrate(frames, temperature: float, bias: float):
        moved = []
        for score, xy, people in frames:
            logit = np.log(np.clip(score, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(score, 1e-6, 1 - 1e-6))
            moved.append((1.0 / (1.0 + np.exp(-(logit + bias) / temperature)), xy, people))
        return moved

    def test_ap_is_unchanged_by_any_calibration(self):
        frames = self._frames()
        base = score_detections(frames)["ap"]
        for temperature, bias in ((0.5, 0.0), (2.0, -1.0), (1.0, 3.0), (4.0, 2.5)):
            assert score_detections(self._recalibrate(frames, temperature, bias))["ap"] == pytest.approx(base, abs=1e-6)

    def test_the_fixed_threshold_columns_do_move(self):
        """The point of the item: AP is pinned, the fixed-threshold operating point is not."""
        frames = self._frames()
        base = score_detections(frames)
        shifted = score_detections(self._recalibrate(frames, 1.0, 3.0))
        assert shifted["recall"] > base["recall"]
        assert shifted["fp_per_frame"] > base["fp_per_frame"]


class TestLabelsFollowTheMetricsRule:
    """`_prec_rec_2d` accepts detections from the highest score down, so at every threshold a person belongs to the
    best-scoring detection that can reach it. `_gated_assignment` minimises distance instead and is score-blind, which
    labels the detection the metric counts as the true positive a negative whenever a nearer, weaker one exists. That
    noise is anti-correlated with the score -- the one kind a temperature fit cannot survive -- so the labels follow
    the metric's rule, and `test_the_label_recall_equals_the_metrics_own` is what holds the two together."""

    def test_the_person_goes_to_the_best_scoring_detection_not_the_nearest(self):
        """The defect this rule exists to avoid: a weak detection sitting closer must not steal the label."""
        score = np.array([0.05, 0.9], np.float32)
        xy = np.array([[0.1, 0.0], [0.3, 0.0]], np.float32)
        people = np.array([[0.0, 0.0]], np.float32)
        assert greedy_labels(score, xy, people).tolist() == [False, True]

    def test_one_person_is_one_positive_however_many_detections_reach_it(self):
        score = np.array([0.9, 0.8, 0.7], np.float32)
        xy = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], np.float32)
        assert greedy_labels(score, xy, np.array([[0.0, 0.0]], np.float32)).sum() == 1

    def test_a_detection_outside_the_radius_is_never_a_positive(self):
        score = np.array([0.99], np.float32)
        xy = np.array([[0.6, 0.0]], np.float32)
        assert not greedy_labels(score, xy, np.array([[0.0, 0.0]], np.float32)).any()

    def test_two_people_take_the_two_best_detections_that_reach_them(self):
        score = np.array([0.4, 0.95, 0.6], np.float32)
        xy = np.array([[0.0, 0.1], [0.0, 0.2], [3.0, 0.1]], np.float32)
        people = np.array([[0.0, 0.0], [3.0, 0.0]], np.float32)
        assert greedy_labels(score, xy, people).tolist() == [False, True, True]

    def test_no_people_means_no_positives(self):
        score = np.array([0.9, 0.8], np.float32)
        xy = np.array([[0.0, 0.0], [1.0, 0.0]], np.float32)
        assert not greedy_labels(score, xy, np.zeros((0, 2), np.float32)).any()


class TestTheLabelCrossCheck:
    """The labels are built here and the reported metric comes from `_prec_rec_2d`, which is deliberately left
    untouched. The two therefore have to be *compared* rather than assumed equal, and `--write` refuses on a mismatch.

    The comparison is made **at the operating threshold**, which is the whole point: the first version of this check
    compared the label positives over every candidate against the metric's recall at 0.3 and refused seed 0 over a
    10.18 pp gap that was nothing but the recall thresholding gives up."""

    @staticmethod
    def _frames(seed: int = 7, frames: int = 60):
        g = np.random.default_rng(seed)
        out = []
        for _ in range(frames):
            people = g.uniform(-5, 5, size=(g.integers(1, 4), 2)).astype(np.float32)
            noise = g.uniform(-5, 5, size=(g.integers(2, 7), 2)).astype(np.float32)
            xy = np.concatenate([people + g.normal(0, 0.2, people.shape).astype(np.float32), noise])
            score = np.concatenate([g.uniform(0.1, 0.95, len(people)), g.uniform(0.01, 0.6, len(noise))]).astype(np.float32)
            out.append((score, xy, people))
        return out

    def test_the_label_recall_equals_the_metrics_own(self):
        """The real check: the same frames, scored both ways, have to agree at the operating threshold."""
        frames = self._frames()
        logit, label = labels_for_frames(frames)
        reported = score_detections(frames)
        check = check_labels_against_the_metric(frames, logit, label, reported["recall"])
        assert check["label_recall"] == pytest.approx(reported["recall"], abs=0.02)
        assert check["agrees"]

    def test_a_material_disagreement_is_flagged_rather_than_averaged_away(self):
        frames = self._frames()
        logit, label = labels_for_frames(frames)
        check = check_labels_against_the_metric(frames, logit, label, 0.2)
        assert not check["agrees"]
        assert check["difference"] > 0.02

    def test_only_detections_above_the_threshold_count_towards_the_check(self):
        """A positive scoring below 0.3 is a person the metric does not have at 0.3 either, so it must not be counted."""
        frames = [(np.array([0.9, 0.05], np.float32), np.array([[0.0, 0.0], [3.0, 0.0]], np.float32),
                   np.array([[0.0, 0.0], [3.0, 0.0]], np.float32))]
        logit, label = labels_for_frames(frames)
        assert label.tolist() == [True, True]
        assert check_labels_against_the_metric(frames, logit, label, 0.5)["label_recall"] == pytest.approx(0.5)


class TestFittingOnLabels:
    @staticmethod
    def _sample(n: int = 4000, temperature: float = 2.5, bias: float = -1.25, seed: int = 5):
        """Labels drawn from a *calibrated* logit, then miscalibrated by a known (T, b) -- so the fit has a right answer."""
        g = np.random.default_rng(seed)
        true_logit = g.normal(0.0, 2.5, n)
        labels = g.uniform(size=n) < 1.0 / (1.0 + np.exp(-true_logit))
        return (true_logit * temperature - bias * 1.0, labels)

    def test_the_fit_recovers_a_known_miscalibration(self):
        observed, labels = self._sample(temperature=2.5, bias=-1.25)
        temperature, bias = fit_calibration(observed, labels)
        assert temperature == pytest.approx(2.5, rel=0.12)
        assert bias == pytest.approx(-1.25, abs=0.2)

    def test_the_fit_lowers_the_negative_log_likelihood(self):
        observed, labels = self._sample()
        before = negative_log_likelihood(observed, labels, 1.0, 0.0)
        temperature, bias = fit_calibration(observed, labels)
        assert negative_log_likelihood(observed, labels, temperature, bias) < before

    def test_the_temperature_is_always_positive(self):
        for seed in range(4):
            observed, labels = self._sample(seed=seed)
            assert fit_calibration(observed, labels)[0] > 0.0

    def test_already_calibrated_logits_fit_near_the_identity(self):
        observed, labels = self._sample(temperature=1.0, bias=0.0)
        temperature, bias = fit_calibration(observed, labels)
        assert temperature == pytest.approx(1.0, rel=0.12)
        assert bias == pytest.approx(0.0, abs=0.2)

    def test_one_class_only_is_refused_rather_than_fitted(self):
        """All-positive or all-negative labels drive the bias to infinity; a silent `inf` in a checkpoint is worse."""
        logit = np.linspace(-3, 3, 200)
        for labels in (np.ones(200, bool), np.zeros(200, bool)):
            with pytest.raises(ValueError, match="both classes"):
                fit_calibration(logit, labels)

class TestTheFloat32RoundTripNeverInverts:
    """Why the gate asks for rank preservation and a float32 tolerance rather than bit equality.

    Measured 2026-10-31 on all three seeds, 6.4 M detections: **zero inversions**, and 24,812-26,010 adjacent pairs
    per seed that were distinct in float32 collapsing into ties. Sharpening (every fitted temperature was 0.71-0.79)
    spreads the logits onto a coarser part of the sigmoid's float32 grid, so distinct scores can land on the same
    float32 value -- and `_prec_rec_2d` accepts tied detections in mergesort order, so AP can shift by a tie-order
    effect. That is the whole of the +-5.96e-8 (one float32 ulp near 0.82) seen on seeds 0 and 2, and seed 1 showed
    exactly 0. Rounding to nearest is monotone, which is why a *reordering* is impossible and a tie is not.
    """

    @staticmethod
    def _scores(seed: int, n: int = 200_000):
        return np.random.default_rng(seed).uniform(0.0, 1.0, n).astype(np.float32)

    def test_no_calibration_ever_inverts_two_scores_in_float32(self):
        score = self._scores(13)
        order = np.argsort(score, kind="mergesort")
        for temperature, bias in ((0.7082, 0.0807), (0.7917, -0.1792), (0.7424, 0.1580), (0.25, 3.0), (4.0, -2.0)):
            moved = to_probability((to_logit(score) + bias) / temperature).astype(np.float32)
            assert np.diff(moved[order].astype(np.float64)).min() >= 0.0

    def test_sharpening_does_collapse_distinct_scores_into_ties(self):
        """The benign half, asserted so the tolerance is not mistaken for slack: ties are real and expected."""
        score = self._scores(17)
        moved = to_probability((to_logit(score) + 0.0807) / 0.7082).astype(np.float32)
        assert len(np.unique(moved)) < len(np.unique(score))


class TestApplyingAFittedPair:
    """`--apply` exists because re-running the fit to store two floats costs ~55 min of scoring per seed. It is the
    irreversible half of item 48, so it re-checks the report's own gates rather than trusting that they were read."""

    @staticmethod
    def _checkpoint(tmp_path: Path) -> Path:
        model = _memory()
        path = tmp_path / "step3a_object_memory.best.pth"
        torch.save({"model": model.state_dict(),
                    "args": {"dim": str(DIM), "heads": str(HEADS), "recurrence": "mingru",
                             "exchange": "False", "learned_match": "False", "slots": str(CAPACITY)}}, path)
        return path

    @staticmethod
    def _report(tmp_path: Path, memory: Path, *, temperature: float = 0.7917, bias: float = -0.1792,
                agrees: bool = True, ap_delta: float = 0.0) -> Path:
        path = tmp_path / "calibration.json"
        ap = 0.8180959224700928
        path.write_text(json.dumps(dict(memory=str(memory), split="val", temperature=temperature, bias=bias,
                                        label_check=dict(agrees=agrees, label_recall=0.8464, metric_recall=0.8475,
                                                         difference=0.0010, tolerance=0.02),
                                        uncalibrated=dict(ap=ap), calibrated=dict(ap=ap + ap_delta))),
                        encoding="utf-8")
        return path

    def test_it_writes_the_pair_into_the_checkpoint_and_returns_it(self, tmp_path):
        memory = self._checkpoint(tmp_path)
        assert apply_calibration(self._report(tmp_path, memory), memory) == (0.7917, -0.1792)
        stored = torch.load(memory, map_location="cpu", weights_only=False)["calibration"]
        assert (stored["temperature"], stored["bias"]) == (0.7917, -0.1792)

    def test_the_written_key_is_what_object_memory_reads_back(self, tmp_path):
        """The round trip that matters: a written checkpoint has to load with the pair in place."""
        memory = self._checkpoint(tmp_path)
        apply_calibration(self._report(tmp_path, memory), memory)
        restored = torch.load(memory, map_location="cpu", weights_only=False)
        assert object_memory(restored, 0).calibration == (0.7917, -0.1792)

    def test_one_float32_ulp_of_ap_movement_is_accepted(self):
        """What all three seeds actually produced; refusing it would refuse the measurement itself."""
        assert AP_TOLERANCE >= 5.96e-08

    def test_it_refuses_when_the_label_check_disagrees(self, tmp_path):
        memory = self._checkpoint(tmp_path)
        with pytest.raises(ValueError, match="labels"):
            apply_calibration(self._report(tmp_path, memory, agrees=False), memory)

    def test_it_refuses_when_ap_moved_beyond_float32_rounding(self, tmp_path):
        memory = self._checkpoint(tmp_path)
        with pytest.raises(ValueError, match="AP"):
            apply_calibration(self._report(tmp_path, memory, ap_delta=1e-3), memory)

    def test_it_refuses_a_report_fitted_on_a_different_checkpoint(self, tmp_path):
        """The mistake this catches: three seeds, three JSONs, one mismatched path writes seed 1's pair into seed 2."""
        memory = self._checkpoint(tmp_path)
        other = tmp_path / "somewhere_else" / "step3a_object_memory.best.pth"
        with pytest.raises(ValueError, match="different checkpoint"):
            apply_calibration(self._report(tmp_path, other), memory)

    def test_it_refuses_a_non_positive_temperature(self, tmp_path):
        memory = self._checkpoint(tmp_path)
        with pytest.raises(ValueError, match="temperature"):
            apply_calibration(self._report(tmp_path, memory, temperature=0.0), memory)

    def test_it_leaves_the_checkpoint_untouched_when_it_refuses(self, tmp_path):
        memory = self._checkpoint(tmp_path)
        before = memory.read_bytes()
        with pytest.raises(ValueError):
            apply_calibration(self._report(tmp_path, memory, agrees=False), memory)
        assert memory.read_bytes() == before
