"""Judge every A51 arm against a multi-seed control, from the runs' `step3a_results.json` files alone.

Standard library only and no torch, so this runs beside a live training job -- see `memory/gotchas.md` on the commit limit.

    python judge_arms.py                          # the fresh control (a51_b0f_s*), every arm
    python judge_arms.py --control a51_b0_short   # the older short control, to see how much the control itself moved
    python judge_arms.py --json out.json          # the same table as data

An arm "clears" a metric when its distance from the control mean exceeds that metric's three-sigma bar. Two sets of bars are
printed: the control's own (three times the sd of its seeds) and A51 E9's published ones, which came from the short control and
are what `TODO.md` and `memory/slot-design-evidence.md` have judged against so far. They disagree by design -- a three-seed sd is
a noisy estimate -- so an arm that clears one and not the other is reported as borderline rather than as a result.
"""

import argparse
import json
import statistics as st
from pathlib import Path

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints_three_horizon"

# The metrics a slot-design arm is judged on, as `(label, key, decimals, scale)`. `own_recall` is recall at the candidates'
# false-positive rate, which is the operating point A51 reports; `ap` is the person-class AUC at 0.5 m that leads any answer.
METRICS = (
    ("AP", "ap", 2, 100.0),
    ("AP@0.3", "ap_0.3m", 2, 100.0),
    ("recall@FP", "own_recall", 2, 100.0),
    ("coastP", "coast_precision", 2, 100.0),
    ("coast/f", "coast_per_frame", 3, 1.0),
)

# E9's three-sigma bars (`memory/slot-design-evidence.md`), keyed as above; AP@0.3 is E9's 1.29 and coast/f was never barred.
E9_BARS = {"ap": 1.45, "ap_0.3m": 1.29, "own_recall": 0.86, "coast_precision": 8.19}

# Every arm, with the flag that defines it. The comment on each is what the arm is meant to isolate.
ARMS = (
    ("T1 no gate", "a51_t1_s0", "--no-rescore-gate"),
    ("T2 plateau", "a51_t2_s0", "--schedule plateau"),
    ("O1 bounded", "a51_o1_bounded_s0c", "--no-rescore-gate --rescore-limit 3"),
    ("O2 coast x3", "a51_o2_coast3_s0", "--coast-weight 3"),
    ("D1 sensor frame", "a51_d1_sensor_s0", "--no-room-frame"),
    ("D2 latest value", "a51_d2_latest_s0", "--latest-value"),
    ("D3 fixed decay", "a51_d3_fixeddecay_s0", "--fixed-decay-s 8"),
    ("E6 no exchange", "a51_e6_noexchange_s0", "--no-exchange"),
    ("E7 no match", "a51_e7_nomatch_s0", "--no-learned-match"),
    ("E6+E7 recurrent", "a51_e67_recurrent_s0", "--no-exchange --no-learned-match"),
    ("E10 no state", "a51_e10_norecurrent_s0", "--no-recurrent"),
)

# Every arm was confirmed 2026-09-30 to share the control's settings -- `step3a` on `step2_static`, `chunk_s [1.0, 10.0]`,
# `test_strides [5]`, `max_hours 24`, 256 slots -- read from each checkpoint's own stored `args` rather than from a launcher
# script, which is the durable source: `torch.load(<run>/step3a_object_memory.best.pth)["args"]`.
UNVERIFIED_CLI: set = set()


def read(name):
    """One run's test metrics, plus its measured milliseconds per frame; None when the run has not written its results yet."""
    path = CHECKPOINTS / name / "step3a_results.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    row = dict(blob["test"]["stride_5"]["memory"])
    row["ms_per_frame"] = blob.get("memory_ms_per_frame")
    row["candidate_ap"] = blob["test"]["stride_5"]["candidates"]["ap"]
    return row


def values(row):
    return {key: row[key] * scale for _, key, _, scale in METRICS}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control", default="a51_b0f", help="control run prefix; seeds are <prefix>_s0, _s1, _s2")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--json", type=Path, help="also write the table here")
    args = parser.parse_args()

    seeds = [(s, read(f"{args.control}_s{s}")) for s in range(args.seeds)]
    present = [(s, r) for s, r in seeds if r is not None]
    missing = [s for s, r in seeds if r is None]
    if len(present) < 2:
        raise SystemExit(f"{args.control}: only {len(present)} seed(s) have results; nothing to judge against")

    per_seed = [values(r) for _, r in present]
    mean = {key: st.mean(v[key] for v in per_seed) for _, key, _, _ in METRICS}
    sd = {key: st.stdev(v[key] for v in per_seed) for _, key, _, _ in METRICS}
    bars = {key: 3 * sd[key] for key in sd}

    print(f"control {args.control}: seeds {[s for s, _ in present]}"
          + (f"  MISSING {missing} -- the mean and every bar below will move when they land" if missing else ""))
    candidate_aps = {round(r["candidate_ap"] * 100, 2) for _, r in present}
    print(f"candidates AP across those seeds: {sorted(candidate_aps)}"
          + ("  (identical, as they must be -- they are read from the cache)" if len(candidate_aps) == 1 else
             "  ** NOT IDENTICAL: the candidates come from the cache, so a difference means a corrupted run **"))
    print()

    head = f"{'':<18}" + "".join(f"{label:>11}" for label, *_ in METRICS) + f"{'ms/frame':>11}"
    print(head)
    for s, row in present:
        v = values(row)
        ms = row["ms_per_frame"]
        print(f"{'seed ' + str(s):<18}" + "".join(f"{v[key]:>11.{d}f}" for _, key, d, _ in METRICS)
              + (f"{ms:>11.2f}" if ms else f"{'-':>11}"))
    print(f"{'mean':<18}" + "".join(f"{mean[key]:>11.{d}f}" for _, key, d, _ in METRICS))
    print(f"{'sd':<18}" + "".join(f"{sd[key]:>11.{d}f}" for _, key, d, _ in METRICS))
    print(f"{'3 sigma (own)':<18}" + "".join(f"{bars[key]:>11.{d}f}" for _, key, d, _ in METRICS))
    print(f"{'3 sigma (E9)':<18}" + "".join(f"{E9_BARS.get(key, float('nan')):>11.{d}f}" for _, key, d, _ in METRICS))
    print()

    rows = []
    print(f"{'arm':<18}" + "".join(f"{label:>11}" for label, *_ in METRICS) + "   verdict per metric (own bar / E9 bar)")
    for label, name, flag in ARMS:
        row = read(name)
        if row is None:
            print(f"{label:<18}{'not yet run':>11}")
            continue
        v = values(row)
        deltas = {key: v[key] - mean[key] for _, key, _, _ in METRICS}
        verdicts = []
        for _, key, _, _ in METRICS:
            own = abs(deltas[key]) > bars[key] if bars[key] > 0 else False
            e9 = abs(deltas[key]) > E9_BARS[key] if key in E9_BARS else None
            verdicts.append("--" if e9 is None else "clears" if own and e9 else "ns" if not own and not e9 else "borderline")
        mark = " *" if name in UNVERIFIED_CLI else ""
        print(f"{label + mark:<18}" + "".join(f"{v[key]:>11.{d}f}" for _, key, d, _ in METRICS))
        print(f"{'  delta':<18}" + "".join(f"{deltas[key]:>+11.{d}f}" for _, key, d, _ in METRICS)
              + "   " + " ".join(f"{label}:{verdict}" for (label, *_), verdict in zip(METRICS, verdicts)))
        rows.append(dict(arm=label, run=name, flag=flag, values=v, deltas=deltas, verdicts=verdicts,
                         ms_per_frame=row["ms_per_frame"], cli_verified=name not in UNVERIFIED_CLI))

    if any(name in UNVERIFIED_CLI for _, name, _ in ARMS):
        print("\n* the launcher for this arm is not on disk any more, so it was not confirmed to share the control's settings")

    if args.json:
        args.json.write_text(json.dumps(dict(control=args.control, seeds=[s for s, _ in present], missing=missing,
                                             mean=mean, sd=sd, bars=bars, e9_bars=E9_BARS, arms=rows), indent=1),
                             encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
