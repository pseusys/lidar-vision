"""Is Li2Former's vote head dead, or is our adapter starving it?

The DROW leg's `loss_reg` is bit-stable across epochs in both train (0.1816) and val (0.1410),
which can only happen if `pred_reg` on foreground beams is not moving. Two candidates:

  A. their terminal `nn.ReLU()` on `reg_head` has gone dead -- every pre-activation negative,
     so the output is pinned at 0 and the gradient through it is exactly 0;
  B. a defect in this repository's adapter, so the head never receives a gradient at all.

They are distinguishable without the GPU and without a dataset: A leaves a live gradient on the
trunk and a dead one on `reg_head`, with the pre-ReLU activations negative; B leaves `reg_head`
grads at zero while the pre-activations straddle zero.

It is kept because arm B of the ReLU ablation needs the same check in reverse -- proof that the head
stays alive -- and because a claim about a published repository should be reproducible on demand.

CPU only, no dataset, 8 beams, ~100 MB, so it is safe to run beside a training job:
  .venv/Scripts/python.exe -I utils/diagnose_vote_head.py                                   # fresh network
  .venv/Scripts/python.exe -I utils/diagnose_vote_head.py checkpoints_drow/li2former_official.best.pth
"""
import sys

import torch

from takheliped.detectors import OfficialLi2FormerDetector


def report(tag, net, x, target_reg, fg):
    pre = {}
    # The layer feeding the terminal ReLU, so we see what the ReLU is clamping.
    net.net.reg_head[2].register_forward_hook(lambda m, i, o: pre.__setitem__("z", o.detach()))

    net.zero_grad(set_to_none=True)
    pred_cls, pred_reg = net(x)
    if pred_reg.shape[0] == 2 * target_reg.shape[0]:   # their BatchFormer doubled the batch
        target_reg, fg = torch.cat([target_reg, target_reg]), torch.cat([fg, fg])
    loss = torch.nn.functional.mse_loss(pred_reg[fg], target_reg[fg])
    loss.backward()

    z = pre["z"].reshape(-1, 2)
    out = pred_reg.detach()
    grad = net.net.reg_head[2].weight.grad
    trunk = net.net.cls_head.weight.grad

    print(f"[{tag}]")
    print(f"  pre-ReLU z:        min {z.min():+.4f}  max {z.max():+.4f}  frac>0 {(z > 0).float().mean():.4f}")
    print(f"    channel dx:      min {z[:, 0].min():+.4f}  max {z[:, 0].max():+.4f}  frac>0 {(z[:, 0] > 0).float().mean():.4f}")
    print(f"    channel dy:      min {z[:, 1].min():+.4f}  max {z[:, 1].max():+.4f}  frac>0 {(z[:, 1] > 0).float().mean():.4f}")
    print(f"  pred_reg:          min {out.min():+.4f}  max {out.max():+.4f}  frac==0 {(out == 0).float().mean():.4f}")
    print(f"  grad reg_head[2]:  norm {0.0 if grad is None else grad.norm():.3e}")
    print(f"  grad cls_head:     norm {0.0 if trunk is None else trunk.norm():.3e}   <- trunk, for contrast")
    verdict = "B: adapter starves the head" if (grad is None or grad.norm() == 0) and (z > 0).any() \
        else "A: terminal ReLU is dead" if not (z > 0).any() \
        else "neither -- the head is alive and learning"
    print(f"  => {verdict}")
    print()


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    net = OfficialLi2FormerDetector()
    n_beams, t, n_pts = 8, net.net.num_scans if hasattr(net.net, "num_scans") else 5, net.N_SAMP
    # their forward wants [B, CT, T, P] -- the adapter adds B, so (beams, scans, points per cutout)
    x = torch.randn(n_beams, t, n_pts).abs() * 0.1
    target_reg = torch.randn(n_beams, 2) * 0.3          # signed, as their global2local produces
    fg = torch.zeros(n_beams, dtype=torch.bool)
    fg[2:6] = True

    net.train()
    report("fresh weights, train mode", net, x, target_reg, fg)

    if len(sys.argv) > 1:
        state = torch.load(sys.argv[1], map_location="cpu")
        net.load_state_dict(state["model"] if "model" in state else state)
        net.train()
        report(f"trained: {sys.argv[1]}", net, x, target_reg, fg)


if __name__ == "__main__":
    main()
