#!/usr/bin/env python3
"""Verify the `balanced` partition against the **real recorded FROG files**, which rule 4 demands.

`tests/test_frog_splits.py` pins the partition's logic -- disjoint recordings, all six used once, every split
seeing both empty and populated frames, the published baselines' training data held out of test -- but it does
it against synthetic HDF5 files written into a `tmp_path`. That is the right way to test the loader, and it is
not evidence about the dataset: this repository has twice shipped dataset logic that passed clean synthetic
tests and was only caught by looking at the real recordings.

So this reads the actual files and checks the three things a synthetic fixture cannot:

1. **The composition is what `_BALANCED_COMPOSITION` claims** -- which recordings, how many frames, what share
   of them empty, how many people per populated frame. Those numbers are quoted in `docs/` and in `memory/`,
   and they were measured once, by hand, in 2026-10. Nothing has re-checked them since.
2. **Empty frames really are in train and val**, not only in test. That is the whole reason the partition
   exists: `official` and `transferred` have no person-free *training* frames at all, so the detector has never
   seen a room with nobody in it.
3. **It differs from `official`** on every split, by recording, so a number from one is never a number from the
   other.

    .venv/Scripts/python.exe -I utils/check_balanced_partition.py
    .venv/Scripts/python.exe -I utils/check_balanced_partition.py --splits val

CPU only, no model, no training. It reads annotations rather than scans where it can, but the loader brings a
recording in whole, so expect a few minutes and a few GB per split; `--splits` runs them one at a time.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "library"))

from takheliped.datasets.frog_dataset import _BALANCED_COMPOSITION, FROG_Dataset  # noqa: E402

TOLERANCE = {"empty_frac": 0.005, "people_per_populated_frame": 0.05}


def composition(dataset) -> dict:
    """What the loaded split actually contains, in the same terms `_BALANCED_COMPOSITION` states.

    `frames` counts **scans**, which is what the table counts, and it is not the same as the number of frames
    that can be scored: `train` and `val` hand out detections only from frame 40 of each sequence, so a
    temporal model's first judged frame has a full window behind it. `test` takes no such run-up, because
    `official`'s test must stay extractable from it frame for frame, and dropping 40 frames from each of its 48
    sequences would break exactly that. Both numbers are reported, since the gap between them is the run-up and
    a reader comparing against the table needs to know which one it is.
    """
    empty = populated = people = 0
    for sequence in dataset.det_wp:
        for frame in sequence:
            if len(frame):
                populated += 1
                people += len(frame)
            else:
                empty += 1
    annotated = empty + populated
    frames = sum(len(stamps) for stamps in dataset.scan_time)
    return {"frames": frames, "annotated": annotated, "runup": frames - annotated,
            "sequences": len(dataset.det_wp),
            "empty_frac": empty / annotated if annotated else 0.0,
            "people_per_populated_frame": people / populated if populated else 0.0,
            "empty": empty, "populated": populated}


def frame_times(dataset) -> set:
    """Every loaded frame, identified by its scan timestamp in milliseconds.

    Stronger than naming recordings, and it needs nothing from the partition table that is under test: two
    splits hold the same data exactly when they hold the same timestamps. The sensor stamps at ~40 Hz, so a
    millisecond key separates consecutive frames with room to spare, and 130k of them cost a few MB.
    """
    return {int(round(float(t) * 1000)) for stamps in dataset.scan_time for t in stamps}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=("train", "val", "test"))
    args = ap.parse_args()

    failures, overlaps = [], {}
    for split in args.splits:
        print(f"\n== {split} ==", flush=True)
        balanced = FROG_Dataset(split=split, mode="balanced", verbose=False)
        official = FROG_Dataset(split=split, mode="official", verbose=False)
        got, want = composition(balanced), _BALANCED_COMPOSITION[split]

        print(f"  scans                      {got['frames']:,} (claimed {want['frames']:,})")
        print(f"  scorable                   {got['annotated']:,} over {got['sequences']} sequences, "
              f"{got['runup']:,} held back as run-up ({got['runup'] // max(got['sequences'], 1)} per sequence)")
        print(f"  empty                      {got['empty']:,} = {got['empty_frac']:.1%} "
              f"(claimed {want['empty_frac']:.1%})")
        print(f"  people / populated frame   {got['people_per_populated_frame']:.2f} "
              f"(claimed {want['people_per_populated_frame']:.2f})")

        if got["frames"] != want["frames"]:
            failures.append(f"{split}: {got['frames']:,} frames, the table claims {want['frames']:,}")
        for key, slack in TOLERANCE.items():
            if abs(got[key] - want[key]) > slack:
                failures.append(f"{split}: {key} is {got[key]:.4f}, the table claims {want[key]:.4f}")

        # 2. The reason the partition exists. Test may be person-free-heavy on any partition; train and val
        #    are the splits `official` cannot supply, so an absence here would make the mode pointless.
        if split in ("train", "val") and not got["empty"]:
            failures.append(f"{split}: no person-free frames at all, which is the whole point of this mode")
        if not got["populated"]:
            failures.append(f"{split}: no populated frames at all")

        # 3. Different data from `official`'s split of the same name, or the two modes' numbers are the same
        #    number reported twice. Measured on the frames themselves, and the overlap is reported rather than
        #    only its absence, because two of the overlaps are load-bearing claims in their own right.
        here, there = frame_times(balanced), frame_times(official)
        shared = here & there
        print(f"  frames shared with official's {split}: {len(shared):,} "
              f"= {len(shared) / len(here):.1%} of this split, {len(shared) / len(there):.1%} of official's")
        if here == there:
            failures.append(f"{split}: frame-for-frame identical to official's {split}")
        if split == "test" and not there <= here:
            # The loader's stated reason for this test pairing: one evaluation pass yields the balanced number
            # and a directly comparable official-test number, because 16-41 is in here whole.
            failures.append(f"test: official's test is NOT a subset, so it cannot be extracted from one pass "
                            f"({len(there - here):,} of its {len(there):,} frames are missing)")
        overlaps[split] = here
        del balanced, official

    # 4. The trap the loader warns about, measured rather than trusted: `balanced`'s val holds 11-36, which is
    #    half of `official`'s train, so validating there while training on `official` scores a model on its own
    #    training data. This confirms the overlap exists, so the warning is never quietly dropped as stale.
    if "val" in overlaps:
        official_train = frame_times(FROG_Dataset(split="train", mode="official", verbose=False))
        leak = overlaps["val"] & official_train
        print(f"\n== the documented trap ==\n  balanced val frames also in official TRAIN: {len(leak):,} "
              f"= {len(leak) / len(overlaps['val']):.1%} of balanced's val")
        if not leak:
            failures.append("balanced's val no longer overlaps official's train, so the loader's warning "
                            "against mixing partitions is stale and should be rewritten, not deleted")

    print()
    for line in failures:
        print(f"FAIL  {line}")
    print(f"{'FAILED' if failures else 'OK'}: {len(failures)} problem(s) over {len(args.splits)} split(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
