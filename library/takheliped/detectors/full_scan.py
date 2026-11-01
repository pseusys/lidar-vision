"""
The full-scan, non-recursive lidar-based person detector.

It processes the *entire* scan at once, using the full angular context
available across all N_beams, in contrast to the per-beam cutout approach
used by DROW and DR-SPAAM. Two design choices, made deliberately for
efficiency and explainability (see RESEARCH.md):

* Non-recursive only: every block is a CNN or TCN layer (or pooling/norm/
  linear "glue"). No RNN/GRU/LSTM, no attention — all are fully
  feed-forward and parallel across beams.
* Raw input, no preprocessing: input is the raw range scan, one channel.
  Prepared via drow_utils.raw_scan() (unaligned) or
  drow_utils.aligned_raw_scan() (rotation-corrected via odometry, default).

SpaceTimeCNNDetector shares the same input/output contract as DrowDetector
(default --head drow):
  forward(x)       → (logits (B*N_beams, 4), votes (B*N_beams, 2))
  forward_one(xb)  → (confs, votes)   numpy in / numpy out

With --head heatmap:
  forward(x)       → ((heatmap (B*N_beams, 1),),)
  forward_one(xb)  → (sigmoid_heatmap, zeros_votes)

It sets BEAM_BATCH=True (like DrSpaamDetector), so the DataLoader passes
(B, N_beams, T, 1) rather than a (B*N_beams, T, 1) flattened batch. It
convolves along the beam axis, so flattening splices B frames into one long
scan and lets normalization statistics, SE gates and the convolutions
themselves leak between frames that happen to share a minibatch — which made
measured accuracy a function of batch size (TODO.md A18).

Input shape: (B, N_beams, T, 1); a bare (N_beams, T, 1) single scan is also
accepted and treated as B=1.
Channel 0 is the raw range measurement, optionally rotation-aligned.

Detectors
---------
  SpaceTimeCNNDetector  — 2D dilated Conv over (N_beams, T) grid, collapses T

Receptive field (SpaceTimeCNNDetector)
--------------------------------------
Three dilated Conv2d layers with beam-axis dilations 1,2,4 and kernel_size=3
give a total beam receptive field of 15 beams (±3.75° at 0.5°/beam).

Two retired siblings
--------------------
This module held three architectures until TODO.md item 32 (2026-10-31).
FullScanTCNDetector (causal temporal TCN over T + a dilated 1D CNN over beams,
on a shared DilatedScanBackbone) and TemporalUNetDetector (a 1D U-Net with T
input channels, heatmap head by default) were retired: nothing had trained on
either since 2026-09 and neither influenced a project decision. SpaceTimeCNN
stayed because the 2026-09-10 history ablation that reorganised the whole
queue was measured on it, and keeping the class is what keeps that
reproducible. Their own measurements are kept, marked removed, in
memory/performance-log.md, with the commit where the code last existed; four
blocks they shared with this detector — _gn, _SEBlock, _JointConvBlock and
_MultiScaleBlock — stayed with it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SpaceTimeCNNDetector building blocks
# ---------------------------------------------------------------------------

def _gn(num_channels: int) -> nn.GroupNorm:
    """GroupNorm selecting the largest power-of-2 divisor of num_channels."""
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018).

    Global-averages the spatial map, passes through a 2-layer bottleneck MLP,
    and re-weights each channel with a sigmoid gate.  Near-zero parameter
    overhead; consistently adds ~1 % accuracy in image CNNs.

    Input / output: (B, C, H, W) — same shape.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(2, 3))                       # (B, C) global avg-pool
        w = self.fc(w)                               # (B, C) per-channel gate
        return x * w.unsqueeze(-1).unsqueeze(-1)


class _JointConvBlock(nn.Module):
    """Dilated 2-D conv block over both beam and time axes simultaneously.

    Captures joint space-time patterns (e.g. a range-decrease at a specific
    angular position over consecutive frames) that purely factorised
    beam-then-time towers cannot represent.

    For k=3 and dilation d the same-padding is d on each axis.

    Input / output: (1, C, N_beams, T) — same shape.
    """

    def __init__(self, C_in: int, C_out: int,
                 dilation: tuple = (1, 1), dropout: float = 0.1):
        super().__init__()
        pad            = (dilation[0], dilation[1])   # same-padding for k=3
        self.conv      = nn.Conv2d(C_in, C_out, (3, 3),
                                   padding=pad, dilation=dilation, bias=False)
        self.norm      = _gn(C_out)
        self.skip      = (nn.Conv2d(C_in, C_out, 1, bias=False)
                          if C_in != C_out else nn.Identity())
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm(self.conv(x)))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        return out + self.skip(x)


class _MultiScaleBlock(nn.Module):
    """Inception-style multi-scale conv block for one axis (beam or time).

    Runs parallel branches with different kernel sizes along the specified
    axis, concatenates the outputs and projects back to C_out via a 1×1 conv,
    then applies a Squeeze-and-Excitation gate and a residual skip.

    axis='beam'  — kernels are (k, 1) with beam-axis dilation; the T dimension
                   passes through unmodified so subsequent temporal stages see
                   the full T.
    axis='time'  — kernels are (1, k); beam axis is unchanged.

    Stacking two of these blocks for the same axis builds a compositional
    hierarchy: Stage A learns velocity-level patterns; Stage B (applied to
    Stage A's output) learns acceleration / periodicity — patterns that no
    single wide kernel can represent alone.

    Input / output: (1, C, N_beams, T) — same shape.
    """

    def __init__(self, C_in: int, C_out: int, kernel_sizes: tuple,
                 axis: str, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        assert axis in ('beam', 'time')
        n        = len(kernel_sizes)
        C_branch = C_out // n            # channels per parallel branch

        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            if axis == 'beam':
                pad  = ((k - 1) // 2 * dilation, 0)
                kern = (k, 1)
                dil  = (dilation, 1)
            else:                        # time
                pad  = (0, (k - 1) // 2)
                kern = (1, k)
                dil  = (1, 1)
            self.branches.append(nn.Sequential(
                nn.Conv2d(C_in, C_branch, kern,
                          padding=pad, dilation=dil, bias=False),
                _gn(C_branch),
                nn.ReLU(inplace=True),
            ))

        self.project = nn.Sequential(
            nn.Conv2d(C_branch * n, C_out, 1, bias=False),
            _gn(C_out),
        )
        self.se        = _SEBlock(C_out)
        self.skip      = (nn.Conv2d(C_in, C_out, 1, bias=False)
                          if C_in != C_out else nn.Identity())
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([b(x) for b in self.branches], dim=1)
        out = F.relu(self.project(out))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = self.se(out)
        return out + self.skip(x)


# ---------------------------------------------------------------------------
# SpaceTimeCNNDetector
# ---------------------------------------------------------------------------

class SpaceTimeCNNDetector(nn.Module):
    """
    Full-scan detector: multi-scale 2-D conv over the (N_beams × T) space-time grid.

    Architecture
    ------------
    Input (N_beams, T, 1) — raw range only, unaligned across T (see
    drow_utils.raw_scan()) — is reshaped to (1, 1, N_beams, T) and processed
    as a 1-channel 2-D "image" where height = beams and width = time frames.

    Stem
      A single joint Conv2d(3×3) to lift the 1 input channel to C.

    Joint blocks (2×, dilations 1 and 2 on both axes)
      Dilated 2-D convolutions that learn *co-occurring* beam+time patterns —
      e.g. "range is falling specifically at beam 200 over three frames" — which
      purely factorised spatial-then-temporal towers cannot represent.

    Spatial stages (n_spatial_stages = 1–4, default 3)
      Stacked _MultiScaleBlock along the beam axis.  Each stage runs parallel
      inception branches with kernel sizes (3, 5, 9) at an exponentially
      growing beam dilation (1 → 4 → 16 → 64).
        • narrow branches (k=3):  person features at far range (few beams)
        • wide branches (k=9):    person features at close range (many beams)
      Stacking builds hierarchy — Stage A detects leg-scale clusters, Stage B
      composes them into person-scale features, and so on.

    Temporal stages (2×, kernel sizes 3, 5, 7)
      Stacked _MultiScaleBlock along the time axis.  Stage A sees the raw
      temporal signal at three scales (velocity-like); Stage B operates on
      Stage A's output to learn acceleration / gait-periodicity patterns.

    Collapse
      Mean over the T axis — handles any T without architecture changes.

    Beam receptive field (widest branch k=9 at each spatial stage)
    ---------------------------------------------------------------
    pre-spatial (stem + 2 joint blocks):  9 beams
    + Spatial A  (d=1):                  17 beams
    + Spatial B  (d=4):                  49 beams
    + Spatial C  (d=16):                177 beams   ← default (n_spatial_stages=3)
    + Spatial D  (d=64):                689 beams   ← optional (n_spatial_stages=4)

    At 0.5°/beam: 3 stages → ±44°; 4 stages → covers full 720-beam FROG scan.

    DirectML-compatible (no GRU).
    """

    INPUT_MODE = "raw_scan"
    # (B, N_beams, T, 1), not a (B*N_beams, T, 1) flattened batch: the beam
    # axis is a spatial axis this model convolves along, so folding the batch
    # into it makes a batch of B scans into one B*N_beams-beam "scan". See
    # forward().
    BEAM_BATCH = True

    _SPATIAL_DILATIONS = (1, 4, 16, 64)
    _SPATIAL_KS        = (3, 5, 9)
    _TEMPORAL_KS       = (3, 5, 7)

    def __init__(self, n_time: int = 5, channels: int = 96,
                 n_spatial_stages: int = 3, dropout: float = 0.1,
                 head: str = "drow"):
        super().__init__()
        assert 1 <= n_spatial_stages <= 4, "n_spatial_stages must be 1–4"
        assert head in ("drow", "heatmap"), f"head must be 'drow' or 'heatmap', got {head!r}"
        self.n_time           = n_time
        self.channels         = channels
        self.n_spatial_stages = n_spatial_stages
        self.head_type        = head
        C = channels

        # ── Stem ────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, (3, 3), padding=(1, 1), bias=False),
            _gn(C),
            nn.ReLU(inplace=True),
        )

        # ── Joint blocks ─────────────────────────────────────────────────────
        self.joint1 = _JointConvBlock(C, C, dilation=(1, 1), dropout=dropout)
        self.joint2 = _JointConvBlock(C, C, dilation=(2, 2), dropout=dropout)

        # ── Spatial multi-scale stages ───────────────────────────────────────
        self.spatial_stages = nn.ModuleList([
            _MultiScaleBlock(C, C, self._SPATIAL_KS, axis='beam',
                             dilation=d, dropout=dropout)
            for d in self._SPATIAL_DILATIONS[:n_spatial_stages]
        ])

        # ── Temporal multi-scale stages ──────────────────────────────────────
        self.temporal_stages = nn.ModuleList([
            _MultiScaleBlock(C, C, self._TEMPORAL_KS, axis='time', dropout=dropout),
            _MultiScaleBlock(C, C, self._TEMPORAL_KS, axis='time', dropout=dropout),
        ])

        if head == "heatmap":
            self.head_heatmap = nn.Linear(C, 1)
        else:
            self.head_logits = nn.Linear(C, 4)
            self.head_votes  = nn.Linear(C, 2)

    def forward(self, x: torch.Tensor):
        """
        x : (B, N_beams, T, 1)  from DataLoader / evaluate_auc (BEAM_BATCH=True)
            (N_beams, T, 1)     a bare single scan (forward_one, ROS node)

        Returns ((B*N, 1),) for heatmap, or ((B*N, 4), (B*N, 2)) for drow head.

        The batch axis is kept separate all the way through. It used to be
        folded into the beam axis -- `x.permute(2, 0, 1).unsqueeze(0)` on a
        `(B*N_beams, T, 1)` input, making a batch of 4 scans into one
        2880-beam "scan". Three things then leaked between frames that share a
        minibatch: GroupNorm's statistics, `_SEBlock`'s global channel gate,
        and the beam-axis convolutions themselves (receptive field 177 beams)
        convolving across the seam between one frame's leftmost beams and the
        next frame's rightmost. Because evaluation batches are consecutive
        frames in time, a bigger batch leaked more temporal context and scored
        higher: measured on the real val split at dtime=10, beam-level person
        AUC ran 0.9644 (batch 1) -> 0.9696 (4) -> 0.9765 (16) -> 0.9810 (64),
        while the robot itself runs `forward_one()` at batch 1 (TODO.md A18).
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)                 # (N, T, 1) → (1, N, T, 1)
        B, N = x.shape[:2]
        x = x.permute(0, 3, 1, 2)              # → (B, 1, N_beams, T)

        x = self.stem(x)
        x = self.joint1(x)
        x = self.joint2(x)

        for stage in self.spatial_stages:
            x = stage(x)

        for stage in self.temporal_stages:
            x = stage(x)

        x = x.mean(dim=-1)                     # (B, C, N_beams) — collapse T
        x = x.permute(0, 2, 1).reshape(B * N, -1)   # (B*N, C)
        if self.head_type == "heatmap":
            return (self.head_heatmap(x),)     # ((B*N, 1),)
        return self.head_logits(x), self.head_votes(x)

    @torch.no_grad()
    def forward_one(self, xb) -> tuple:
        """Numpy in, numpy out.  xb: (N_beams, T, 1)

        Heatmap head: returns (sigmoid(heatmap), zeros) where heatmap is
        (N_beams, 1).  For proper peak-based person detection use
        evaluate_auc() or call forward() + scipy.signal.find_peaks directly.
        """
        import numpy as np
        self.eval()
        x = torch.from_numpy(np.asarray(xb, dtype=np.float32))
        if self.head_type == "heatmap":
            (h,) = self(x)
            return torch.sigmoid(h).cpu().numpy(), np.zeros((h.shape[0], 2), dtype=np.float32)
        logits, votes = self(x)
        return F.softmax(logits, dim=-1).cpu().numpy(), votes.cpu().numpy()

    def save_weights(self, path, dataset: str = ""):
        torch.save({
            "type":             "SpaceTimeCNNDetector",
            "n_time":           self.n_time,
            "channels":         self.channels,
            "n_spatial_stages": self.n_spatial_stages,
            "head":             self.head_type,
            "model":            self.state_dict(),
            "dataset":          dataset,
        }, path)

    @classmethod
    def load(cls, path, map_location: str = "cpu") -> "SpaceTimeCNNDetector":
        ckpt     = torch.load(path, map_location=map_location)
        channels = ckpt.get("channels", ckpt.get("out_channels", 96))
        model    = cls(
            n_time           = ckpt.get("n_time", 5),
            channels         = channels,
            n_spatial_stages = ckpt.get("n_spatial_stages", 3),
            head             = ckpt.get("head", "drow"),
        )
        model.load_state_dict(ckpt["model"])
        return model
