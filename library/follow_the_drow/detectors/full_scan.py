"""
Full-scan, non-recursive lidar-based person detectors.

These detectors process the *entire* scan at once, using the full angular
context available across all N_beams, in contrast to the per-beam cutout
approach used by DROW and DR-SPAAM. All share two design choices, made
deliberately for efficiency and explainability (see RESEARCH.md):

* Non-recursive only: every block is a CNN or TCN layer (or pooling/norm/
  linear "glue"). No RNN/GRU/LSTM, no attention — all are fully
  feed-forward and parallel across beams.
* Raw input, no preprocessing: input is the raw range scan, one channel.
  Prepared via drow_utils.raw_scan() (unaligned) or
  drow_utils.aligned_raw_scan() (rotation-corrected via odometry, default).

SpaceTimeCNNDetector and FullScanTCNDetector share the same input/output
contract as DrowDetector (default --head drow):
  forward(x)       → (logits (N_beams, 4), votes (N_beams, 2))
  forward_one(xb)  → (confs, votes)   numpy in / numpy out

With --head heatmap (all full-scan detectors):
  forward(x)       → ((heatmap (N_beams, 1),),)
  forward_one(xb)  → (sigmoid_heatmap, zeros_votes)

TemporalUNetDetector has BEAM_BATCH=True (like DrSpaamDetector) — the
DataLoader passes (B, N_beams, T, 1) rather than (B*N_beams, T, 1) because
the U-Net pools over the beam axis and cannot split beams across the batch.
Its default head is "heatmap" to match LFE-Peaks.

Input shape: (N_beams, T, 1) for A/B; (B, N_beams, T, 1) for C.
Channel 0 is the raw range measurement, optionally rotation-aligned.

Detectors
---------
  SpaceTimeCNNDetector  — 2D dilated Conv over (N_beams, T) grid, collapses T
  FullScanTCNDetector   — Causal temporal TCN over T + dilated 1D CNN over beams
  TemporalUNetDetector  — 1D U-Net with T input channels (Architecture C)

Receptive field (DilatedScanBackbone, used by FullScanTCNDetector)
-------------------------------------------------------------------
Four dilated Conv1d layers with dilation=1,2,4,8 and kernel_size=3 give a
total receptive field of 31 beams.  At DROW's 0.5°/beam resolution this covers
±7.5°.  At 3 m range that corresponds to ~0.8 m lateral extent — sufficient
to cover a standing person (shoulder width ~0.5 m).

Receptive field (SpaceTimeCNNDetector)
--------------------------------------
Three dilated Conv2d layers with beam-axis dilations 1,2,4 and kernel_size=3
give a total beam receptive field of 15 beams (±3.75° at 0.5°/beam).

Receptive field (TemporalUNetDetector)
--------------------------------------
Three MaxPool1d(2) stages compress the beam axis to N//8; the bottleneck
Conv1d(k=3) there has an effective RF of 8*3=24 beams in original beam space.
Skip connections restore full spatial resolution without limiting the global RF.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared dilated backbone
# ---------------------------------------------------------------------------

class DilatedScanBackbone(nn.Module):
    """
    Dilated 1D CNN applied over the beam (angular) dimension.

    Processes the full scan with an exponentially growing receptive field
    (dilation = 1, 2, 4, 8).  Each timestep is treated independently as a
    batch element.

    Input:  (N_beams, T, in_channels)
    Output: (N_beams, T, out_channels)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout_p   = dropout
        mid              = out_channels

        self.conv1 = nn.Conv1d(in_channels, mid, 3, padding=1,  dilation=1)
        self.bn1   = nn.BatchNorm1d(mid)
        self.conv2 = nn.Conv1d(mid,         mid, 3, padding=2,  dilation=2)
        self.bn2   = nn.BatchNorm1d(mid)
        self.conv3 = nn.Conv1d(mid,         mid, 3, padding=4,  dilation=4)
        self.bn3   = nn.BatchNorm1d(mid)
        self.conv4 = nn.Conv1d(mid,         mid, 3, padding=8,  dilation=8)
        self.bn4   = nn.BatchNorm1d(mid)
        # 1×1 projection for the end-to-end residual skip connection
        self.skip  = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N_beams, T, in_channels)
        N, T, C = x.shape
        x_t = x.permute(1, 2, 0)                           # (T, in_channels, N_beams)
        res = self.skip(x_t)                                # (T, out_channels, N_beams)

        x_t = F.relu(self.bn1(self.conv1(x_t)))
        x_t = F.dropout(x_t, p=self.dropout_p, training=self.training)
        x_t = F.relu(self.bn2(self.conv2(x_t)))
        x_t = F.dropout(x_t, p=self.dropout_p, training=self.training)
        x_t = F.relu(self.bn3(self.conv3(x_t)))
        x_t = F.dropout(x_t, p=self.dropout_p, training=self.training)
        x_t = F.relu(self.bn4(self.conv4(x_t)) + res)      # residual before final activation

        return x_t.permute(2, 0, 1)                        # (N_beams, T, out_channels)


# ---------------------------------------------------------------------------
# Temporal TCN building block (for FullScanTCNDetector)
# ---------------------------------------------------------------------------

class _TemporalTCNBlock(nn.Module):
    """
    Causal dilated 1-D conv block over the time axis — one TCN layer.

    Every beam is processed independently (T is the conv's spatial axis,
    N_beams is the batch axis), so this is fully parallel across beams —
    no recurrence.  Causal "chomp" padding (pad left by (k-1)*dilation, drop
    the trailing dilation*(k-1) positions of the symmetric-padded conv output)
    means output position t depends only on input positions <= t, mirroring
    an RNN's left-to-right dependency without sequential recurrence.

    Input / output: (N_beams, C, T) — same T, channels may change.
    """

    def __init__(self, C_in: int, C_out: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self._chomp = 2 * dilation                # (kernel_size - 1) * dilation, kernel_size=3
        self.conv   = nn.Conv1d(C_in, C_out, 3, padding=self._chomp,
                                dilation=dilation, bias=False)
        self.norm   = _gn(C_out)
        self.skip   = (nn.Conv1d(C_in, C_out, 1, bias=False)
                      if C_in != C_out else nn.Identity())
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)[:, :, :-self._chomp]    # drop the non-causal (future-looking) tail
        out = F.relu(self.norm(out))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        return out + self.skip(x)


# ---------------------------------------------------------------------------
# Full-scan detectors
# ---------------------------------------------------------------------------

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
        """x : (N_beams, T, 1)"""
        x = x.permute(2, 0, 1).unsqueeze(0)   # → (1, 1, N_beams, T)

        x = self.stem(x)
        x = self.joint1(x)
        x = self.joint2(x)

        for stage in self.spatial_stages:
            x = stage(x)

        for stage in self.temporal_stages:
            x = stage(x)

        x = x.mean(dim=-1)                     # (1, C, N_beams) — collapse T
        x = x.squeeze(0).permute(1, 0)         # (N_beams, C)
        if self.head_type == "heatmap":
            return (self.head_heatmap(x),)     # ((N_beams, 1),)
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


class FullScanTCNDetector(nn.Module):
    """
    Full-scan detector: causal temporal TCN (per beam) + dilated spatial CNN.

    Architecture
    ------------
      Per-beam causal TCN over T   (2 layers, dilation 1, 2 — full T=5 receptive
                                     field — collapses T to a single per-beam
                                     "now" summary, analogous to a GRU's final
                                     hidden state but fully feed-forward)
        → DilatedScanBackbone        (shared spatial block, mixes across beams)
        → detection heads

    Inverts SpaceTimeCNNDetector's fusion order: there, beam and time are
    mixed jointly from the first layer; here, each beam's own 5-frame history
    is contextualised by the TCN *first*, then DilatedScanBackbone mixes
    across beams — e.g. a cluster of beams that are all approaching together
    is visible to the spatial step as such, which a joint-from-layer-1 design
    has to learn implicitly instead.

    Cheaper than a per-timestep spatial CNN would be: the spatial backbone
    runs once per scan (T collapsed to 1 beforehand) instead of once per
    timestep.

    Fully feed-forward (no recurrence): trains and infers via convolutions only.
    """

    INPUT_MODE = "raw_scan"

    def __init__(self, n_time: int = 5, tcn_channels: int = 64,
                 backbone_channels: int = 64, dropout: float = 0.1,
                 head: str = "drow"):
        super().__init__()
        assert head in ("drow", "heatmap"), f"head must be 'drow' or 'heatmap', got {head!r}"
        self.n_time        = n_time
        self.head_type     = head
        self.temporal_tcn  = nn.ModuleList([
            _TemporalTCNBlock(1,            tcn_channels, dilation=1, dropout=dropout),
            _TemporalTCNBlock(tcn_channels, tcn_channels, dilation=2, dropout=dropout),
        ])
        self.backbone       = DilatedScanBackbone(tcn_channels, backbone_channels, dropout)
        if head == "heatmap":
            self.head_heatmap = nn.Linear(backbone_channels, 1)
        else:
            self.head_logits  = nn.Linear(backbone_channels, 4)
            self.head_votes   = nn.Linear(backbone_channels, 2)

    def forward(self, x: torch.Tensor):
        """x : (N_beams, T, 1)  single scan  |  (B*N_beams, T, 1)  flattened batch"""
        x_t = x.permute(0, 2, 1)                            # (rows, 1, T) — conv1d over T per beam
        for block in self.temporal_tcn:
            x_t = block(x_t)                                # (rows, tcn_channels, T)
        beam_feat = x_t[:, :, -1]                           # (rows, tcn_channels) — causal "now" summary

        spatial_in  = beam_feat.unsqueeze(1)                # (rows, 1, tcn_channels) — single pseudo-timestep
        spatial_out = self.backbone(spatial_in)             # (rows, 1, backbone_channels)
        pooled      = spatial_out.squeeze(1)                # (rows, backbone_channels)
        if self.head_type == "heatmap":
            return (self.head_heatmap(pooled),)             # ((rows, 1),)
        return self.head_logits(pooled), self.head_votes(pooled)

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
            "type":              "FullScanTCNDetector",
            "n_time":            self.n_time,
            "tcn_channels":      self.temporal_tcn[0].conv.out_channels,
            "backbone_channels": self.backbone.conv4.out_channels,
            "head":              self.head_type,
            "model":             self.state_dict(),
            "dataset":           dataset,
        }, path)

    @classmethod
    def load(cls, path, map_location: str = "cpu") -> "FullScanTCNDetector":
        ckpt  = torch.load(path, map_location=map_location)
        model = cls(
            n_time=ckpt.get("n_time", 5),
            tcn_channels=ckpt.get("tcn_channels", 64),
            backbone_channels=ckpt.get("backbone_channels", 64),
            head=ckpt.get("head", "drow"),
        )
        model.load_state_dict(ckpt["model"])
        return model


# ---------------------------------------------------------------------------
# TemporalUNetDetector building blocks
# ---------------------------------------------------------------------------

class _UNetEncoderBlock(nn.Module):
    """Two Conv1d + GroupNorm + ReLU encoder (or bottleneck) block."""

    def __init__(self, C_in: int, C_out: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(C_in,  C_out, 3, padding=1, bias=False), _gn(C_out), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(C_out, C_out, 3, padding=1, bias=False), _gn(C_out), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UNetDecoderBlock(nn.Module):
    """Upsample → cat(skip) → two Conv1d decoder block.

    Parameters
    ----------
    C_up   : channels of the upsampled feature map
    C_skip : channels of the skip connection from the encoder
    C_out  : output channels after the two convolutions
    """

    def __init__(self, C_up: int, C_skip: int, C_out: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(C_up + C_skip, C_out, 3, padding=1, bias=False), _gn(C_out), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(C_out,         C_out, 3, padding=1, bias=False), _gn(C_out), nn.ReLU(inplace=True),
        )

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        # Upsample to exactly match the skip's spatial size (handles non-power-of-2 N).
        x_up = F.interpolate(x_up, size=x_skip.shape[-1], mode='linear', align_corners=False)
        return self.block(torch.cat([x_up, x_skip], dim=1))


# ---------------------------------------------------------------------------
# TemporalUNetDetector (Architecture C)
# ---------------------------------------------------------------------------

class TemporalUNetDetector(nn.Module):
    """
    Architecture C: multi-frame 1-D U-Net with T range scans as input channels.

    Extends LFE-Peaks/LFE-PPN to multi-frame input by replacing the
    single-channel stem Conv1d(1, C) with Conv1d(T, C), treating T raw
    range scans as T parallel input channels to a 1-D U-Net that operates
    over the beam axis.  Structural differences from LFE:

      * T input channels (vs LFE's 1) — the key temporal extension
      * GroupNorm throughout (vs LFE's Keras BatchNorm, which degrades at B=1)
      * Trained from scratch (vs LFE's published ONNX weights)
      * Raw range input (vs LFE's (1-r/10) normalization; raw is fine with GN)

    This model pools over the beam axis, so individual beams from the same
    scan cannot be separated into the PyTorch batch dimension.  BEAM_BATCH=True
    tells the training loop to pass (B, N_beams, T, 1) as a unit.

    Architecture (default unet_channels=32, 3 encoder stages)
    ----------------------------------------------------------
    Encoder:
      E1: Conv1d(T→C,   k=3)×2  + GN + ReLU    at  N_beams          (skip_1)
      MaxPool1d(2)
      E2: Conv1d(C→2C,  k=3)×2  + GN + ReLU    at  N//2             (skip_2)
      MaxPool1d(2)
      E3: Conv1d(2C→4C, k=3)×2  + GN + ReLU    at  N//4             (skip_3)
      MaxPool1d(2)
    Bottleneck:
      B:  Conv1d(4C→8C, k=3)×2  + GN + ReLU    at  N//8
    Decoder:
      D3: Upsample → cat(B, skip_3) → Conv1d(12C→4C, k=3)×2         at  N//4
      D2: Upsample → cat(D3, skip_2) → Conv1d(6C→2C,  k=3)×2        at  N//2
      D1: Upsample → cat(D2, skip_1) → Conv1d(3C→C,   k=3)×2        at  N
    Head:
      heatmap (default): Linear(C, 1) — per-beam logit; sigmoid gives probability
      drow:              Linear(C, 4) + Linear(C, 2) — logits + vote offsets
    """

    BEAM_BATCH = True        # training loop must pass (B, N, T, 1), not (B*N, T, 1)
    INPUT_MODE = "raw_scan"

    def __init__(self, n_time: int = 5, unet_channels: int = 32,
                 dropout: float = 0.1, head: str = "heatmap"):
        super().__init__()
        assert head in ("drow", "heatmap"), f"head must be 'drow' or 'heatmap', got {head!r}"
        self.n_time        = n_time
        self.unet_channels = unet_channels
        self.head_type     = head
        C = unet_channels

        self.enc1       = _UNetEncoderBlock(n_time, C,   dropout)
        self.enc2       = _UNetEncoderBlock(C,      C*2, dropout)
        self.enc3       = _UNetEncoderBlock(C*2,    C*4, dropout)
        self.pool       = nn.MaxPool1d(2)
        self.bottleneck = _UNetEncoderBlock(C*4,    C*8, dropout)

        self.dec3 = _UNetDecoderBlock(C*8, C*4, C*4, dropout)
        self.dec2 = _UNetDecoderBlock(C*4, C*2, C*2, dropout)
        self.dec1 = _UNetDecoderBlock(C*2, C,   C,   dropout)

        if head == "heatmap":
            self.head_heatmap = nn.Linear(C, 1)
        else:
            self.head_logits  = nn.Linear(C, 4)
            self.head_votes   = nn.Linear(C, 2)

    def forward(self, x: torch.Tensor):
        """
        x : (B, N_beams, T, 1)  from DataLoader / evaluate_auc (BEAM_BATCH=True)
            (1, N_beams, T, 1)  from forward_one (after unsqueeze)

        Returns (B*N, 1) for heatmap, or ((B*N, 4), (B*N, 2)) for drow head.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)                             # (N, T, 1) → (1, N, T, 1)
        B, N, T, _ = x.shape

        h = x.squeeze(-1).permute(0, 2, 1)                # (B, T, N) — T as channels

        e1 = self.enc1(h)                                  # (B, C,  N)
        e2 = self.enc2(self.pool(e1))                      # (B, 2C, N//2)
        e3 = self.enc3(self.pool(e2))                      # (B, 4C, N//4)
        b  = self.bottleneck(self.pool(e3))                # (B, 8C, N//8)

        d3 = self.dec3(b,  e3)                             # (B, 4C, N//4)
        d2 = self.dec2(d3, e2)                             # (B, 2C, N//2)
        d1 = self.dec1(d2, e1)                             # (B, C,  N)

        out = d1.permute(0, 2, 1).reshape(B * N, -1)      # (B*N, C)
        if self.head_type == "heatmap":
            return (self.head_heatmap(out),)               # ((B*N, 1),)
        return self.head_logits(out), self.head_votes(out)

    @torch.no_grad()
    def forward_one(self, xb) -> tuple:
        """Numpy in, numpy out.  xb: (N_beams, T, 1)"""
        import numpy as np
        self.eval()
        x = torch.from_numpy(np.asarray(xb, dtype=np.float32)).unsqueeze(0)  # (1, N, T, 1)
        if self.head_type == "heatmap":
            (h,) = self(x)                                 # (N, 1) since B=1
            return torch.sigmoid(h).cpu().numpy(), np.zeros((h.shape[0], 2), dtype=np.float32)
        logits, votes = self(x)
        return F.softmax(logits, dim=-1).cpu().numpy(), votes.cpu().numpy()

    def save_weights(self, path, dataset: str = ""):
        torch.save({
            "type":          "TemporalUNetDetector",
            "n_time":        self.n_time,
            "unet_channels": self.unet_channels,
            "head":          self.head_type,
            "model":         self.state_dict(),
            "dataset":       dataset,
        }, path)

    @classmethod
    def load(cls, path, map_location: str = "cpu") -> "TemporalUNetDetector":
        ckpt  = torch.load(path, map_location=map_location)
        model = cls(
            n_time        = ckpt.get("n_time", 5),
            unet_channels = ckpt.get("unet_channels", 32),
            head          = ckpt.get("head", "heatmap"),
        )
        model.load_state_dict(ckpt["model"])
        return model
