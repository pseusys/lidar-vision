"""Range-stratified error: is the static gap SCALE or CAPACITY?  (PROPOSAL Tier 0)

The governing hypothesis is that DROW's advantage over LFE comes from the
cutout's distance normalisation rather than from its parameter count.  The
obvious test -- compare per-range recall against DROW3's -- is not available:
DROW3's 73.9% on FROG is *published*, and the bundled DROW weights here are
DROW-trained, so we cannot produce a comparable per-bin curve without a
training run.

So the test is restructured to need only one detector, by controlling for
information content.  Recall falls with range for *every* detector, because a
distant person subtends fewer beams.  The scale hypothesis predicts something
sharper: that LFE loses recall with range *beyond* what the drop in beams-on-
target explains, because a full-scan FCN has to learn the scale invariance a
cutout gets for free.

    recall is a clean function of beams-on-target, no residual range term
        -> LFE is already effectively scale-invariant -> the gap is CAPACITY
    recall still falls with range at FIXED beams-on-target
        -> a scale-handling deficiency -> the gap is SCALE

The same pass also answers the sub-threshold question (also Tier 0): at every
missed person, what did the network's per-beam probability map actually say?
Near threshold means a decoder/operating-point failure; ~0 means the
representation never saw them.

Usage
-----
    python range_probe.py                       # both detectors, official test
    python range_probe.py --mode balanced
    python range_probe.py --stride 5            # quick look
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.datasets.frog_dataset import FROG_Dataset, frog_laser_angles
from follow_the_drow.detectors import LFEPeaksDetector, LFEPPNDetector
from follow_the_drow.detectors.lfe_detector import _normalize_scan, _SCAN_FAR

PERSON_R = 0.4          # FROG's own annotation radius (151,611 of 153,655 circles)


def _detector(name):
    return LFEPeaksDetector() if name == "lfe-peaks" else LFEPPNDetector(score_thresh=0.01)


def _prob_map(det, scan):
    """LFE-Peaks' raw per-beam probability, before find_peaks and merging.

    Replicates the three lines of LFEPeaksDetector.detect() that precede
    peak-finding; the class does not expose them.  FROG is already on the
    trained 720-beam grid, so the resample inside detect() is a no-op and the
    beam indices here line up with `scan` one-for-one.
    """
    inp = _normalize_scan(scan).reshape(1, 720, 1)
    return det._session.run([det._output_name], {det._input_name: inp})[0].squeeze()


def collect(name, mode, stride):
    """One detector pass -> one record per annotated person."""
    ds = FROG_Dataset(split="test", mode=mode, auto_download=False, verbose=False)
    det = _detector(name)
    angles = frog_laser_angles(720).astype(np.float64)
    sin_a, cos_a = -np.sin(angles), np.cos(angles)
    peaks = (name == "lfe-peaks")

    rows = []
    for seq in range(len(ds.det_id)):
        for det_idx in range(0, len(ds.det_id[seq]), stride):
            iscan = int(ds.idet2iscan[seq][det_idx])
            wp = np.asarray(ds.det_wp[seq][det_idx], dtype=np.float64).reshape(-1, 2)
            if not len(wp):
                continue
            rng, phi = wp[:, 0], wp[:, 1]
            gx, gy = rng * -np.sin(phi), rng * np.cos(phi)

            scan = np.asarray(ds.scans[seq][iscan], dtype=np.float64)
            bx, by = scan * sin_a, scan * cos_a          # every beam's endpoint
            prob = _prob_map(det, scan.astype(np.float32)) if peaks else None

            d = np.asarray(det.detect(scan.astype(np.float32), angles),
                           dtype=np.float64).reshape(-1, 3)
            score = np.full(len(wp), np.nan)             # NaN = never matched
            nn = np.full(len(wp), np.inf)
            if len(d):
                cost = np.hypot(gx[:, None] - d[None, :, 1], gy[:, None] - d[None, :, 2])
                nn = cost.min(axis=1)
                gated = np.where(cost > 0.5, 1e6, cost)
                for r, c in zip(*linear_sum_assignment(gated)):
                    if gated[r, c] < 1e6:
                        score[r] = d[c, 0]

            for j in range(len(wp)):
                on = np.hypot(bx - gx[j], by - gy[j]) <= PERSON_R
                rows.append((seq, iscan, rng[j], phi[j], int(on.sum()),
                             float(prob[on].max()) if (peaks and on.any()) else np.nan,
                             score[j], nn[j], len(d)))
    return np.array(rows, dtype=np.float64)


# ------------------------------------------------------------------- reports

COLS = dict(seq=0, frame=1, rng=2, phi=3, beams=4, prob=5, score=6, nn=7, ndet=8)
RANGE_BINS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 30]
BEAM_BINS = [0, 1, 2, 4, 6, 9, 14, 21, 32, 48, 10000]


def _col(a, k):
    return a[:, COLS[k]]


def _bin_table(a, key, edges, tau, unit):
    print("\n  {:>12} {:>9} {:>7} {:>8} {:>10} {:>10}".format(
        "bin", "people", "share", "recall", "med beams", "med score"))
    v = _col(a, key)
    hit = _col(a, "score") >= tau
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v >= lo) & (v < hi)
        if m.sum() == 0:
            continue
        s = _col(a, "score")[m]
        med = np.nanmedian(s) if np.isfinite(s).any() else float("nan")
        print("  {:>12} {:>9d} {:>6.1%} {:>7.1%} {:>10.0f} {:>10.2f}".format(
            "{}-{}{}".format(lo, hi, unit), int(m.sum()), m.mean(), hit[m].mean(),
            np.median(_col(a, "beams")[m]), med))


def _residual(a, tau):
    """Recall at FIXED beams-on-target, split by range. The scale signature."""
    print("\n  Recall at fixed beams-on-target, split by range.")
    print("  Columns agreeing within a row => LFE is already scale-invariant (capacity).")
    print("  Recall falling left-to-right within a row => a scale deficit.")
    rng, beams = _col(a, "rng"), _col(a, "beams")
    hit = _col(a, "score") >= tau
    cuts = [(0, 3), (3, 5), (5, 7), (7, 10), (10, 30)]
    print("\n  {:>10} ".format("beams") +
          " ".join("{:>13}".format("{}-{}m".format(lo, hi)) for lo, hi in cuts))
    for lo, hi in zip(BEAM_BINS[:-1], BEAM_BINS[1:]):
        mb = (beams >= lo) & (beams < hi)
        if mb.sum() < 50:
            continue
        cells = []
        for rlo, rhi in cuts:
            m = mb & (rng >= rlo) & (rng < rhi)
            cells.append("{:>6.1%}({:>5d})".format(hit[m].mean(), int(m.sum()))
                         if m.sum() >= 30 else "{:>13}".format("-"))
        print("  {:>10} ".format("{}-{}".format(lo, hi)) + " ".join(cells))


def _subthreshold(a, tau):
    """At a missed person, what did the per-beam probability map say?"""
    prob = _col(a, "prob")
    ok = np.isfinite(prob)
    if not ok.any():
        return
    a, prob = a[ok], prob[ok]
    hit = _col(a, "score") >= tau
    miss = ~hit
    print("\n  Peak of the raw per-beam probability map on the person's own beams.")
    print("  {:>22} {:>8} ".format("group", "n") +
          " ".join("{:>7}".format(q) for q in ("p10", "p25", "med", "p75", "p90")))
    for label, m in (("detected", hit), ("MISSED", miss)):
        q = np.percentile(prob[m], [10, 25, 50, 75, 90]) if m.sum() else [np.nan] * 5
        print("  {:>22} {:>8d} ".format(label, int(m.sum())) +
              " ".join("{:>7.3f}".format(x) for x in q))
    print("\n  Missed people by what the map said (peak_height=0.01, prominence=0.1).")
    print("  `stolen` = a detection DID land within the 0.5 m association radius,")
    print("  but the Hungarian assignment gave it to a different person.")
    nn = _col(a, "nn")
    print("  {:>22} {:>8} {:>7} {:>7} {:>7} {:>8} {:>8}".format(
        "map said", "n", "share", "beams", "range", "med nn", "stolen"))
    edges = [0.0, 0.01, 0.1, 0.3, 0.5, 1.01]
    names = ["~0 (<0.01)", "0.01-0.1", "0.1-0.3", "0.3-0.5", ">0.5"]
    for (lo, hi), nm in zip(zip(edges[:-1], edges[1:]), names):
        m = miss & (prob >= lo) & (prob < hi)
        if not m.sum():
            continue
        print("  {:>22} {:>8d} {:>7.1%} {:>7.0f} {:>6.1f}m {:>8.2f} {:>8.1%}".format(
            nm, int(m.sum()), m.sum() / max(miss.sum(), 1),
            np.median(_col(a, "beams")[m]), np.median(_col(a, "rng")[m]),
            np.median(nn[m]), (nn[m] <= 0.5).mean()))

    print("\n  Miss budget, as a share of ALL annotated people:")
    rep = miss & (prob < 0.1)
    bord = miss & (prob >= 0.1) & (prob < 0.3)
    dec = miss & (prob >= 0.3)
    n = len(a)
    for label, m in (("representation (map < 0.1)", rep),
                     ("borderline (0.1-0.3)", bord),
                     ("decoder (map >= 0.3, peak lost)", dec)):
        print("  {:>34} {:>8d} {:>7.2f} pp of recall".format(
            label, int(m.sum()), 100.0 * m.sum() / n))


def report(name, a, tau):
    n = len(a)
    hit = _col(a, "score") >= tau
    far = _col(a, "rng") > _SCAN_FAR
    print("\n" + "=" * 72)
    print("=== {} @ score >= {} ===".format(name, tau))
    print("=" * 72)
    print("{} annotated person-observations, recall {:.1%}".format(n, hit.mean()))
    print("beyond LFE's {:.0f} m input clip: {:.1%} of people, recall there {:.1%}"
          .format(_SCAN_FAR, far.mean(), hit[far].mean() if far.any() else float("nan")))
    print("\n--- how big a person IS, in beams " + "-" * 38)
    print("  subtense = 2*atan(0.4/d) / 0.25deg, what a 0.8 m person spans;")
    print("  fill = beams actually returning from the person / subtense.")
    print("  {:>12} {:>10} {:>10} {:>7}".format("bin", "med beams", "subtense", "fill"))
    inc = np.pi / 720.0
    for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
        m = (_col(a, "rng") >= lo) & (_col(a, "rng") < hi)
        if m.sum() == 0:
            continue
        d = np.median(_col(a, "rng")[m])
        sub = 2 * np.arctan(0.4 / d) / inc
        mb = np.median(_col(a, "beams")[m])
        print("  {:>12} {:>10.0f} {:>10.1f} {:>6.0%}".format(
            "{}-{}m".format(lo, hi), mb, sub, mb / sub))

    print("\n--- recall vs RANGE " + "-" * 52)
    _bin_table(a, "rng", RANGE_BINS, tau, "m")
    print("\n--- recall vs BEAMS-ON-TARGET (the information control) " + "-" * 16)
    _bin_table(a, "beams", BEAM_BINS, tau, "")
    print("\n--- the residual " + "-" * 55)
    _residual(a, tau)
    if np.isfinite(_col(a, "prob")).any():
        print("\n--- sub-threshold evidence at misses " + "-" * 35)
        _subthreshold(a, tau)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", choices=["lfe-peaks", "lfe-ppn", "both"], default="both")
    ap.add_argument("--mode", default="official",
                    choices=["official", "transferred", "balanced"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--cache", type=Path, default=_HERE / ".range_cache")
    args = ap.parse_args()

    args.cache.mkdir(exist_ok=True)
    names = ["lfe-peaks", "lfe-ppn"] if args.detector == "both" else [args.detector]
    for name in names:
        f = args.cache / "{}_{}_s{}.pkl".format(name, args.mode, args.stride)
        if f.exists():
            a = pickle.loads(f.read_bytes())
            print("[cached] {}".format(f.name))
        else:
            a = collect(name, args.mode, args.stride)
            f.write_bytes(pickle.dumps(a))
        report("{} / FROG {} test".format(name, args.mode), a, args.thresh)


if __name__ == "__main__":
    main()
