#!/usr/bin/env python3
"""Is there anything only the object memory can do, and where does the shrinking-gain trend have to stop?

The memory's advantage over its own candidates falls as stage 2 improves -- **+5.68 pp** of AP on candidates scoring 77.66%,
**+3.68** on 82.2%, **+2.76** when stage 2 also carries temporal convolutions. A reader who sees that will extrapolate it to
zero, and the paper has to say where the extrapolation stops being true rather than wait to be asked.

It stops at whatever a single-frame detector cannot do however good it gets. **A person whose returns are blocked is not in the
scan at all**, so no static detector finds them; a *visible* person our stage 2 misses is a person a better stage 2 would find.
The occlusion strata measure precisely that split, so the memory's recall gain decomposes into

    substitutable   the gain on people who are visible or partly visible -- what a stronger static detector could also take
    irreplaceable   the gain on people who are mostly occluded -- bounded above by that stratum's share of the population

This reads the baselines' three reports and the nine stage-3 runs; it measures nothing itself and needs no GPU.

Usage (from utils/)
-----
    python memory_headroom.py
    python memory_headroom.py --out ../results/memory_headroom.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).parent
CHECKPOINTS = _HERE.parent / "checkpoints_three_horizon"
ARM = "E13:a60_e13_static_s{seed}"
SEEDS = (0, 1, 2)
IRREPLACEABLE = ("occluded",)       # the stratum for a person whose own returns are blocked in that frame
RUNS = (("B0, the no-temporal backbone", "a51_b0_short_s{seed}"),
        ("the adopted recipe", "a60_e13_static_s{seed}"),
        ("the adopted recipe plus stage 2's coarse convolution", "a60_e13_default_s{seed}"))


def decompose(strata: Dict[str, Dict[str, float]], irreplaceable: Sequence[str]) -> Dict[str, float]:
    """The memory's recall gain, split by whether a perfect single-frame detector could have reached those people.

    `strata` maps a stratum to `people`, `candidates` and `memory` (the last two being recalls). Each stratum contributes its
    share of the population times its own gain, so the parts sum to the whole.
    """
    unknown = [name for name in irreplaceable if name not in strata]
    if unknown:
        raise ValueError(f"no stratum named {', '.join(unknown)}; the strata are {', '.join(strata)}")
    total = sum(s["people"] for s in strata.values())
    parts = {name: (s["people"] / total) * (s["memory"] - s["candidates"]) for name, s in strata.items()}
    hidden = sum(value for name, value in parts.items() if name in irreplaceable)
    gain = sum(parts.values())
    return dict(gain=gain, irreplaceable=hidden, substitutable=gain - hidden,
                irreplaceable_share=hidden / gain if gain else float("nan"),
                per_stratum={name: dict(share=strata[name]["people"] / total,
                                        gain=strata[name]["memory"] - strata[name]["candidates"],
                                        contribution=value) for name, value in parts.items()})


def static_ceiling(strata: Dict[str, Dict[str, float]], irreplaceable: Sequence[str]) -> float:
    """The recall a flawless single-frame detector would reach: every person with returns of their own in that scan."""
    total = sum(s["people"] for s in strata.values())
    return sum(s["people"] for name, s in strata.items() if name not in irreplaceable) / total


def trend_crosses_zero_at(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Where a straight line through `(candidate AP, memory gain)` reaches zero gain, or `None` if it never falls."""
    if len(points) < 2:
        return None
    (x0, y0), (x1, y1) = min(points), max(points)
    if x1 == x0 or y1 >= y0:
        return None
    return x0 + y0 * (x1 - x0) / (y0 - y1)


def gain_trend() -> List[Dict[str, float]]:
    """Each recipe's candidate AP and the memory's gain over it, from the stage-3 results already on disk."""
    rows = []
    for label, pattern in RUNS:
        pairs = []
        for seed in SEEDS:
            path = CHECKPOINTS / pattern.format(seed=seed) / "step3a_results.json"
            if not path.exists():
                continue
            block = json.loads(path.read_text())["test"]["stride_5"]
            pairs.append((block["candidates"]["ap"] * 100, block["memory"]["ap"] * 100))
        if pairs:
            rows.append(dict(recipe=label, seeds=len(pairs),
                             candidates=sum(c for c, _ in pairs) / len(pairs),
                             memory=sum(m for _, m in pairs) / len(pairs),
                             gain=sum(m - c for c, m in pairs) / len(pairs)))
    return rows


def pooled_strata() -> Dict[str, Dict[str, float]]:
    """The occlusion strata, pooled over three seeds: people, the candidates' recall, and the memory's."""
    pooled: Dict[str, Dict[str, float]] = {}
    for seed in SEEDS:
        report = json.loads((CHECKPOINTS / "a61" / f"s{seed}" / "report.json").read_text())
        for name, bodies in report["strata"]["occlusion"].items():
            entry = pooled.setdefault(name, dict(people=0.0, candidates=0.0, memory=0.0, seeds=0))
            entry["people"] = bodies["candidates"]["people"]
            entry["candidates"] += bodies["candidates"]["recall"]
            entry["memory"] += bodies[ARM.format(seed=seed)]["recall"]
            entry["seeds"] += 1
    for entry in pooled.values():
        entry["candidates"] /= entry["seeds"]
        entry["memory"] /= entry["seeds"]
    return pooled


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=_HERE.parent / "results" / "memory_headroom.json")
    args = ap.parse_args()

    trend = gain_trend()
    print("The memory's gain against its candidates' quality, from the runs on disk:\n")
    print(f"  {'recipe':<52} {'seeds':>5} {'candidates':>11} {'memory':>8} {'gain':>7}")
    for row in trend:
        print(f"  {row['recipe']:<52} {row['seeds']:>5} {row['candidates']:>11.2f} {row['memory']:>8.2f} {row['gain']:>+7.2f}")
    crossing = trend_crosses_zero_at([(row["candidates"], row["gain"]) for row in trend if "coarse" not in row["recipe"]])
    print(f"\n  A straight line through the two no-temporal points reaches zero gain at candidates of "
          f"{crossing:.1f}% AP." if crossing else "\n  The gain does not fall across these points.")

    strata = pooled_strata()
    split = decompose(strata, IRREPLACEABLE)
    ceiling = static_ceiling(strata, IRREPLACEABLE)
    print(f"\nWhat a flawless single-frame detector could reach: {ceiling:.2%} of people have returns of their own.\n")
    print(f"  {'stratum':<10} {'people':>8} {'share':>7} {'candidates':>11} {'memory':>8} {'gain':>7} {'contributes':>12}")
    for name, body in strata.items():
        per = split["per_stratum"][name]
        print(f"  {name:<10} {int(body['people']):>8} {per['share']:>6.1%} {body['candidates']:>10.1%} "
              f"{body['memory']:>7.1%} {per['gain']:>+6.1%} {per['contribution']:>+11.2%}")
    print(f"\n  recall gain {split['gain']:+.2%}: {split['substitutable']:+.2%} on people a better static detector could "
          f"also find, {split['irreplaceable']:+.2%} on people it could not "
          f"({split['irreplaceable_share']:.1%} of the gain)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(trend=trend, crossing=crossing, static_ceiling=ceiling, strata=strata, split=split),
                                   indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
