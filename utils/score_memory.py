#!/usr/bin/env python3
"""Score a saved stage-3 checkpoint on the FROG `official` test split, without retraining it.

A run that is interrupted after its best checkpoint was written still has a usable model; its `step3a_results.json` is what is
missing. This produces the same test block a finished run prints, so an interrupted arm still yields a number.

    python score_memory.py --memory <checkpoint> --stage2 <stage 2 checkpoint> --out <json>
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from train_takheliped import (  # noqa: E402
    EXTRA_TEST_RADII_M, detect_device, evaluate, evaluation_segments, load_split, memory_ms_per_frame, object_memory,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory", type=Path, required=True)
    ap.add_argument("--stage2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--limit-recordings", type=int, default=0)
    args = ap.parse_args()

    device, _ = detect_device()
    split = load_split("test", args.limit_recordings, args.stage2, device)
    checkpoint = torch.load(args.memory, map_location=device, weights_only=False)
    model = object_memory(checkpoint, split.data.features.shape[-1]).to(device)
    print(f"{args.memory.parent.name}: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)

    ms = memory_ms_per_frame(model, split, device)
    r = evaluate(model, split, device, args.stride, args.streams, evaluation_segments(split, 0, 0, 0), extra_radii=EXTRA_TEST_RADII_M)
    c, m = r["candidates"], r["memory"]
    print(f"\ntest, every {args.stride} annotated frame: candidates AP {c['ap']:.2%} (0.3 m {c['ap_0.3m']:.2%}) "
          f"FP {c['fp_per_frame']:.3f} FN {c['fn_per_frame']:.3f}"
          f"  |  memory AP {m['ap']:.2%} (0.3 m {m['ap_0.3m']:.2%}) P {m['precision']:.1%} R {m['recall']:.1%} "
          f"FP {m['fp_per_frame']:.3f} FN {m['fn_per_frame']:.3f}"
          f"\n  memory at its own threshold {m['own_threshold']:.3f}: R {m['own_recall']:.1%} FP {m['own_fp_per_frame']:.3f}"
          f"  |  rescoring shift person {m['rescore_shift_person']:+.3f} other {m['rescore_shift_other']:+.3f}"
          f"  |  coasting {m['coast_per_frame']:.3f} per frame, precision {m['coast_precision']:.1%}"
          f"  |  {ms:.2f} ms per frame", flush=True)
    args.out.write_text(json.dumps(dict(memory=str(args.memory), memory_ms_per_frame=ms,
                                        test={f"stride_{args.stride}": r}), indent=1), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
        code = 0
    except SystemExit as leaving:        # argparse's --help and its usage errors: pass their status through
        code = leaving.code if isinstance(leaving.code, int) else (0 if leaving.code is None else 1)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)                          # a finished ROCm process can hang past a normal exit (memory/gotchas.md)
