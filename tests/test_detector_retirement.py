"""`FullScanTCNDetector` and `TemporalUNetDetector` are retired, `SpaceTimeCNNDetector` is not.

Three of the project's first full-scan architectures lived in one module. Two of them influenced no decision and nothing has
trained on them since 2026-09; the third is the one whose history ablation reorganised the whole queue, so the owner's
refinement was to keep it -- and keeping it is what makes that decision reproducible from this tree rather than merely
recorded.

What this file holds down is the *shape* of that retirement, which is the part that could go wrong quietly:

* the two names are gone from every surface a caller can reach -- the registry, the package, and the scripts' `--detector`
  choices, skip-lists, checkpoint maps and help text -- so no script offers a detector that cannot be built;
* the four blocks the survivor reaches are **still there**, `_SEBlock` only through `_MultiScaleBlock`, which is why
  reachability is asserted rather than assumed;
* the survivor still answers the raw-scan contract on both heads.

The surface scan reads identifiers and non-docstring literals out of the AST, so **docstrings and comments are deliberately
exempt**: the two detectors' measurements stay in the record (`memory/performance-log.md`, and
`utils/ablation_zero_history.py`'s own rationale), and a file is free to say what was removed and why. What it may not do is
*offer* them.
"""
import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from takheliped.detectors import DETECTOR_REGISTRY
from takheliped.detectors import full_scan
from takheliped.detectors.full_scan import (
    SpaceTimeCNNDetector,
    _JointConvBlock,
    _MultiScaleBlock,
    _SEBlock,
    _gn,
)

REPO = Path(__file__).parent.parent

RETIRED_KEYS = ("fullscan_tcn", "temporal_unet")

RETIRED_DEFINITIONS = (
    "FullScanTCNDetector",
    "TemporalUNetDetector",
    "DilatedScanBackbone",
    "_TemporalTCNBlock",
    "_UNetEncoderBlock",
    "_UNetDecoderBlock",
)

# Every spelling a caller could reach them by: registry key, CLI flag, class or label.
RETIRED_TOKENS = RETIRED_DEFINITIONS + RETIRED_KEYS + (
    "fullscan-tcn", "temporal-unet", "FullScanTCN", "TemporalUNet",
)

# This file names them on purpose; `full_scan.py` is checked by attribute instead of by text.
SCANNED = sorted(
    path for path in [*(REPO / "utils").glob("*.py"), *(REPO / "tests").glob("*.py"),
                      *(REPO / "library" / "takheliped").rglob("*.py")]
    if path.name != Path(__file__).name
)


def reachable_names(path: Path) -> list[str]:
    """Identifiers and non-docstring string literals -- the file's executable surface, prose excluded."""
    # utf-8-sig, not utf-8: four files in this tree start with a BOM, which `ast.parse` rejects outright
    # ("invalid non-printable character U+FEFF") although Python imports them happily -- see `memory/gotchas.md`.
    tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
    prose = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)
    }
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.alias):
            names.extend(part for part in (node.name, node.asname) if part)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            names.append(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose:
            names.append(node.value)
    return names


class TestTheRegistryNoLongerOffersThem:
    @pytest.mark.parametrize("key", RETIRED_KEYS)
    def test_the_key_is_gone(self, key):
        assert key not in DETECTOR_REGISTRY

    def test_spacetime_cnn_stays(self):
        assert DETECTOR_REGISTRY["spacetime_cnn"] is SpaceTimeCNNDetector

    def test_every_registered_detector_can_still_be_named(self):
        """A key pointing at nothing is the failure this retirement could introduce."""
        for key, cls in DETECTOR_REGISTRY.items():
            assert cls is not None, key
            assert isinstance(cls, type), key


class TestTheDefinitionsAreGone:
    @pytest.mark.parametrize("name", RETIRED_DEFINITIONS)
    def test_the_module_does_not_define_it(self, name):
        assert not hasattr(full_scan, name)

    @pytest.mark.parametrize("name", ("FullScanTCNDetector", "TemporalUNetDetector"))
    def test_the_package_does_not_export_it(self, name):
        import takheliped.detectors as detectors
        assert not hasattr(detectors, name)


class TestTheSharedBlocksSurvive:
    """Four definitions stay because the survivor reaches them -- one of them only indirectly."""

    def test_the_blocks_are_still_defined(self):
        assert all(callable(block) for block in (_gn, _SEBlock, _JointConvBlock, _MultiScaleBlock))

    def test_the_survivor_reaches_the_joint_and_multiscale_blocks_directly(self):
        net = SpaceTimeCNNDetector(n_time=3, channels=8, n_spatial_stages=1)
        assert isinstance(net.joint1, _JointConvBlock)
        assert all(isinstance(stage, _MultiScaleBlock) for stage in net.spatial_stages)
        assert all(isinstance(stage, _MultiScaleBlock) for stage in net.temporal_stages)

    def test_the_squeeze_excite_block_is_reached_only_through_the_multiscale_block(self):
        net = SpaceTimeCNNDetector(n_time=3, channels=8, n_spatial_stages=1)
        holders = {type(parent).__name__
                   for parent in net.modules()
                   for child in parent.children() if isinstance(child, _SEBlock)}
        assert holders == {"_MultiScaleBlock"}

    def test_group_norm_is_reached(self):
        net = SpaceTimeCNNDetector(n_time=3, channels=8, n_spatial_stages=1)
        assert any(isinstance(module, torch.nn.GroupNorm) for module in net.modules())


class TestTheSurvivorStillAnswersTheRawScanContract:
    def test_the_drow_head_returns_logits_and_votes(self):
        net = SpaceTimeCNNDetector(n_time=3, channels=8, n_spatial_stages=1, head="drow")
        logits, votes = net(torch.zeros(2, 16, 3, 1))
        assert logits.shape == (32, 4)
        assert votes.shape == (32, 2)

    def test_the_heatmap_head_returns_one_channel(self):
        net = SpaceTimeCNNDetector(n_time=3, channels=8, n_spatial_stages=1, head="heatmap")
        (heatmap,) = net(torch.zeros(2, 16, 3, 1))
        assert heatmap.shape == (32, 1)

    def test_a_bare_single_scan_is_treated_as_one_frame(self):
        net = SpaceTimeCNNDetector(n_time=3, channels=8, n_spatial_stages=1)
        logits, _ = net(torch.zeros(16, 3, 1))
        assert logits.shape == (16, 4)

    def test_forward_one_is_numpy_in_numpy_out(self):
        net = SpaceTimeCNNDetector(n_time=3, channels=8, n_spatial_stages=1)
        confs, votes = net.forward_one(np.zeros((16, 3, 1), dtype=np.float32))
        assert isinstance(confs, np.ndarray) and isinstance(votes, np.ndarray)
        assert confs.shape == (16, 4) and votes.shape == (16, 2)


class TestNoScriptStillOffersThem:
    """The nine-script sweep, made checkable instead of asserted."""

    def test_the_scan_covers_the_scripts_this_item_touches(self):
        scanned = {path.name for path in SCANNED}
        assert {"train.py", "train_all.py", "evaluate.py", "nms_sweep.py", "render_video.py",
                "informational_capacity.py", "error_breakdown.py", "finetune_on_drow.py",
                "cpu_latency.py", "ablation_zero_history.py"} <= scanned

    def test_no_reachable_name_mentions_a_retired_detector(self):
        """One test over every file rather than one per file: ~130 ids for a single invariant is noise in the suite count."""
        offenders = {
            str(path.relative_to(REPO)): sorted({token for name in reachable_names(path)
                                                 for token in RETIRED_TOKENS if token in name})
            for path in SCANNED
        }
        offenders = {path: tokens for path, tokens in offenders.items() if tokens}
        assert not offenders, "still reaching a retired detector:\n" + "\n".join(
            f"  {path}: {', '.join(tokens)}" for path, tokens in sorted(offenders.items()))
