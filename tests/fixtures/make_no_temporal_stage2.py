"""Writes `no_temporal_stage2.pt`: a small no-temporal stage 2 (`coarse_lags=(0,)`) built by the code at commit e76aa94, the last with
the coarse temporal convolution, with its state dict, inputs and outputs. `tests/test_three_horizon.py` loads that state dict into
today's single-frame network and requires the same outputs, which is what keeps every stage-3 result's stage 2 loading identically
. Regenerating it needs that commit's code: `git worktree add ../lv-e76aa94 e76aa94`, then run this from there."""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "library"))
from takheliped.detectors.takheliped import CalibrationNetwork  # noqa: E402

N_BEAMS, FRAMES, SECTORS = 48, 3, 6

torch.manual_seed(0)
angles = torch.from_numpy(np.linspace(-np.pi / 2, np.pi / 2, N_BEAMS, endpoint=False).astype(np.float32))
ranges = torch.rand(FRAMES, N_BEAMS, generator=torch.Generator().manual_seed(1)) * 6 + 0.5
ranges[0, 5:9] = float("inf")
ranges[1, 20] = 61.0
prior = torch.rand(FRAMES, 2, SECTORS, generator=torch.Generator().manual_seed(2))
out = {"angles": angles, "ranges": ranges, "prior": prior}
for name, channels in (("plain", 0), ("with_prior", 2)):
    torch.manual_seed(3)
    net = CalibrationNetwork(angles, channels=(4, 8, 16), blocks=(1, 1, 1), bottleneck_blocks=1, expansion=2, coarse_lags=(0,), prior_channels=channels).eval()
    if channels:
        with torch.no_grad():
            net.aggregate.weight[:, -channels:].normal_()                  # a trained prior, not the silent one it starts as
    with torch.no_grad():
        logit, votes, features = net(ranges[:, None], torch.zeros(FRAMES, 1, 3), prior[:, None] if channels else None)
    out[name] = {"state": net.state_dict(), "logit": logit, "votes": votes, "features": features}
torch.save(out, HERE / "no_temporal_stage2.pt")
print("wrote", HERE / "no_temporal_stage2.pt")
