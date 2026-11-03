"""
Li2Former, as its authors released it, behind this repository's detector interface.

Yang et al., "Li2Former: Omni-Dimension Aggregation Transformer for Person Detection in 2-D Range
Data", IEEE TIM 73, 2024 (DOI 10.1109/TIM.2024.3420353).  Their implementation lives at
github.com/ai-winter/Li2Former-v1 and is **downloaded, never vendored**: it is GPL-3.0 and this
package is MIT, so `library/setup.py` fetches it at a pinned commit into the gitignored
`takheliped/include/li2former_upstream/` tree beside the datasets and published weights.  Only this
adapter is ours.

Why this exists at all, when `li2former.py` already carries the name: that class is a conv backbone
and one temporal encoder layer with no cross-beam path, written from the paper's prose before the
authors' code was found.  It is not the paper's model, so a number measured from it could not go in
the Li2Former row -- see `TODO.md` item 31 and `memory/detector-architectures.md`.

What the authors' model does that ours did not
----------------------------------------------
  MODE=TEMPORAL        their published setting: attend over the T scans of each beam
  BatchFormer          an auxiliary encoder attending across the batch, **training only**, which
                       concatenates its output to the original and so doubles the batch.  The
                       criterion therefore sees duplicated targets -- `duplicates_targets_in_training`
  alpha pooling        learnable softmax-weighted sum of average and max pooling over the T tokens
  adaptive focus       the regression loss is weighted by a * exp(ln(1/3) * (loss/r - 1)^2), which
                       suppresses both trivial and hopeless regressions.  This is the difference
                       between the paper's 76.4% (Li2Former-A) and its 74.8%
  their cutout         1.0 m wide, 0.5 m deep, area-sampled, no odometry correction -- a different
                       window from this repository's DROW `cutout()` (1.66 m / 1.0 m), hence a
                       separate INPUT_MODE

Kept as released, deliberately
------------------------------
The regression head ends in `nn.ReLU`, so it cannot predict a negative `dx`, while their own
`global2local` produces signed `dx` and their own left-right flip augmentation negates it.  That is
their code, not a transcription error, and this adapter does not fix it: the point is to measure
what they published.  Removing it is queued as a one-line ablation.

Input / output
--------------
  INPUT_MODE  = "cutout_li2former"   their `scans2cutout`, not this repository's `cutout()`
  N_SAMP      = 64                   points per cutout
  BEAM_BATCH  = True                 all N beams of one scan stay together
  logits      : (B*N, 1)             binary person logit -- (2*B*N, 1) while training
  votes       : (B*N, 2)             (dx, dy) offsets in window space, same convention as ours
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# Transcribed from their `src/pedestrian_tracker/scripts/cfgs/ros_li2former.yaml`, which is the only
# configuration they publish -- the repository is their ROS deployment and carries no training code,
# so the recipe (epochs, optimiser, schedule) is not theirs and has to be this project's own.
#
# `MODE` is the one value NOT taken from that file, and deliberately.  Their yaml says `TEMPORAL`,
# which is their robot deployment setting: it runs the 1-layer temporal encoder alone and never calls
# the 3-layer spatial encoder, so it has no cross-beam path and leaves ~22M of the 29.47M parameters
# without a gradient (measured: 88 parameter tensors, `tests/test_li2former_port.py`).  The paper's ODA
# module "jointly encode[s] temporal and spatial information" (abstract, IEEE TIM 2024
# 10.1109/TIM.2024.3420353), so the published number cannot come from that branch.  `PARALLEL` runs
# both encoders and fuses them, and is the only mode that uses `fusion` at all.
#
# `NUM_CTS` is the second value that is not theirs, for the same reason and with no choice in it: 90 is
# how many cutouts their robot's lidar yields, and a beam count is a property of the sensor, not of the
# model.  DROW scans carry 450 beams, JRDB 541, FROG 720.  It reaches exactly two places upstream --
# `buildBackbone(num_cts=...)`, where `ConvBackbone` accepts it and never uses it, and
# `PositionalEncodingSine(d_model, max_len=num_cts)`, whose `forward` returns `pe[:, :x.size(1), :]`
# and so comes back *short* rather than raising when a scan is wider than the cap.  Under their
# `TEMPORAL` mode the sequence is the T scans, so 90 was never reached; the omni-dimension modes attend
# over the beams, where it is.  Their encoding is closed-form in the position, so a longer buffer is
# bit-identical to theirs at every position either can reach -- this raises the ceiling and changes no
# value and no parameter (the buffer is registered, not learned).
LI2FORMER_MODEL_KWARGS = {
    "NUM_CTS": 720,
    "NUM_PTS": 64,
    "BACKBONE_TYPE": "ConvBackbone",
    "MAX_NUM_PTS": 1.0e6,
    "MODE": "PARALLEL",
    "TEMPORAL_ENCODER": {
        "D_MODEL": 512, "NHEAD": 8, "NUM_LAYERS": 1, "DIM_FFN": 2048,
        "DROPOUT": 0.1, "ACTIVATION": "relu", "NORMAL_BEFORE": True,
    },
    "SPATIAL_ENCODER": {
        "D_MODEL": 512, "NHEAD": 8, "NUM_LAYERS": 3, "DIM_FFN": 2048,
        "DROPOUT": 0.1, "ACTIVATION": "relu", "NORMAL_BEFORE": True,
    },
}

# `DYNAMIC_REG` is the adaptive focus mechanism; `EOS_WEIGHT` and the `cost_*` weights belong to the
# box/matcher path their published loss types do not use, and are carried only so their criterion
# constructs exactly as it does upstream.
LI2FORMER_LOSS_KWARGS = {
    "TYPE": ["labels", "reg"],
    "LOSS_WEIGHT": {"loss_labels": 1.0, "loss_reg": 3.0, "loss_giou": 2.0},
    "EOS_WEIGHT": 0.1,
    "COST_WEIGHT": {"cost_labels": 1.0, "cost_boxes": 5.0, "cost_giou": 2.0},
    "DYNAMIC_REG": {"a": 1.5, "r": 0.35, "delta": 0.333333, "alpha": 1.5},
}

# Their `CUTOUT_KWARGS`, as `scans2cutout` names them.
LI2FORMER_CUTOUT_KWARGS = {
    "stride": 1,
    "centered": True,
    "fixed": True,
    "window_width": 1.0,
    "window_depth": 0.5,
    "num_cutout_pts": 64,
    "padding_val": 29.99,
    "area_mode": True,
}


def _upstream():
    """Their package, imported on construction rather than on import.

    `library/setup.py` downloads it, so a checkout whose `include/` predates item 31 has everything
    except this one tree -- and importing it at module scope would take the whole detector package
    down with it. `DrowDetector` reads its weights from `include/` the same way, at construction.
    """
    try:
        from ..include.li2former_upstream.models.li2former.li2former import Li2Former
        from ..include.li2former_upstream.utils.scan_utils import scans2cutout
    except ImportError as exc:
        raise ImportError(
            "Li2Former's own implementation is not in takheliped/include/li2former_upstream. "
            "It is downloaded, not vendored (GPL-3.0); re-run `pip install ./library`."
        ) from exc
    return Li2Former, scans2cutout


class OfficialLi2FormerDetector(nn.Module):
    """Li2Former's released network, reached the way every detector here is reached."""

    INPUT_MODE = "cutout_li2former"
    N_SAMP     = 64
    BEAM_BATCH = True
    CUTOUT_KWARGS = LI2FORMER_CUTOUT_KWARGS
    # Their head emits one logit, so something must say what it predicts. The published DROW figures
    # in this lineage come from `pedestrian_only=True` weights and the published metric is person AP,
    # so wheelchair and walker beams are background -- not "any non-background beam", which is what
    # `compute_loss`'s binary mode counts. Inferred from the regime, since their loader is not in the
    # released repository. On FROG the two rules agree: it annotates no wheelchairs or walkers.
    POSITIVE_CLASS = 3

    def __init__(self) -> None:
        super().__init__()
        Li2Former, self._scans2cutout = _upstream()
        self.net = Li2Former(loss_kwargs=LI2FORMER_LOSS_KWARGS,
                             model_kwargs=LI2FORMER_MODEL_KWARGS)
        # Their `__init__` hardcodes `add_bf=True`, but their `SERIAL`/`PARALLEL` branches never reshape
        # the doubled batch and carry the comment "Keep add_bf option False" -- so in the released code
        # the auxiliary stream and the omni-dimension modes are mutually exclusive, and running the
        # paper's mode means turning it off.  `Encoder` takes `add_bf` as a constructor argument that
        # `Li2Former` does not forward, so it is cleared here instead.  The dead `batch_layers` go with
        # it: no branch reads them, and leaving them would write 12.6M untrained parameters into every
        # checkpoint.  The BatchFormer is training-only, so the inference path is identical either way.
        for encoder in (self.net.t_encoder, self.net.s_encoder):
            encoder.add_bf = False
            del encoder.batch_layers
        # Their own criterion, because the adaptive focus mechanism is part of what is being measured.
        self.criterion = self.net.criterion
        # True only while their BatchFormer is on, when a training step predicts twice per beam.
        self.duplicates_targets_in_training = bool(
            self.net.t_encoder.add_bf or self.net.s_encoder.add_bf)

    def preprocess(self, scans: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Their cutout, for (T, N_beams) ranges and (N_beams,) angles -> (N_beams, T, N_SAMP)."""
        return self._scans2cutout(scans, angles, **self.CUTOUT_KWARGS)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, N, T, P)  or  (N, T, P) for a single scan

        Returns
        -------
        logits : (B*N, 1)   binary person logit -- (2*B*N, 1) in training, see the class docstring
        votes  : (B*N, 2)   vote offsets (dx, dy)
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        pred_cls, pred_reg = self.net(x)
        return pred_cls.reshape(-1, 1), pred_reg.reshape(-1, 2)

    def forward_one(self, xb: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Single-scan inference with 4-class output expansion, as `Li2FormerDetector.forward_one`."""
        self.eval()
        with torch.no_grad():
            logits, votes = self(xb)

        confs = torch.zeros(logits.shape[0], 4)
        confs[:, 3] = torch.sigmoid(logits)[:, 0]
        return confs.cpu().numpy(), votes.cpu().numpy()
