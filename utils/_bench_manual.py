"""Manual CPU inference timing for AlgorithmicDetector and LFE-Peaks/LFE-PPN
— these aren't torch.nn.Module so evaluate.py --bench's harness (built around
_build_bench_models) doesn't cover them. Times each at all three dataset beam
counts (450/720/541) since AlgorithmicDetector and LFE both scale with N_beams.

One-off script per docs/EVALUATION_PLAN.md Phase 3's "Optional manual timing"
snippets, extended to cover all three beam counts for a full §8.2 table.
"""
import time
import numpy as np

from follow_the_drow.detectors import AlgorithmicDetector, LFEPeaksDetector, LFEPPNDetector

N_WARMUP = 5
N_ITERS  = 100


def bench_algorithmic(n_beams: int) -> float:
    algo = AlgorithmicDetector(verbose=False)
    scan = np.random.uniform(0.2, 10.0, n_beams).astype("float32")
    odom_xya = np.zeros(3, dtype="float32")
    scans_hist = [scan] * algo.time_frame
    odoms_hist = [odom_xya] * algo.time_frame
    for _ in range(N_WARMUP):
        algo.forward_one(scans_hist[-1], odoms_hist[-1])
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        algo.forward_one(scans_hist[-1], odoms_hist[-1])
    return (time.perf_counter() - t0) / N_ITERS * 1000


def bench_lfe(cls, n_beams: int) -> float:
    det = cls()
    scan = np.random.uniform(0.2, 10.0, n_beams).astype("float32")
    angles = np.linspace(-np.pi / 2, np.pi / 2, n_beams)
    for _ in range(N_WARMUP):
        det.detect(scan, angles)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        det.detect(scan, angles)
    return (time.perf_counter() - t0) / N_ITERS * 1000


if __name__ == "__main__":
    print(f"Config: warmup={N_WARMUP}, iters={N_ITERS}, device=CPU\n")
    for label, n_beams in [("DROW (450 beams)", 450), ("FROG (720 beams)", 720), ("JRDB (541 beams)", 541)]:
        print(f"=== {label} ===")
        try:
            ms = bench_algorithmic(n_beams)
            print(f"  AlgorithmicDetector  {ms:6.2f} ms/frame")
        except Exception as exc:
            print(f"  AlgorithmicDetector  ERROR: {exc}")
        try:
            ms = bench_lfe(LFEPeaksDetector, n_beams)
            print(f"  LFEPeaksDetector     {ms:6.2f} ms/frame")
        except Exception as exc:
            print(f"  LFEPeaksDetector     ERROR: {exc}")
        try:
            ms = bench_lfe(LFEPPNDetector, n_beams)
            print(f"  LFEPPNDetector       {ms:6.2f} ms/frame")
        except Exception as exc:
            print(f"  LFEPPNDetector       ERROR: {exc}")
        print()
