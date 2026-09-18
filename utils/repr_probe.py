"""Does distance normalisation make the person/background decision easier?
   (PROPOSAL Tier 0, the mechanistic half)

`range_probe.py` shows *where* LFE loses recall.  It cannot say *why*, because
recall at fixed beams-on-target still mixes model capacity with representation.
This probe removes the model.

DROW's cutout normalises distance twice:

    angular   the window half-width is atan(0.5 * win_sz / z) beams, so it
              always spans 1.66 m of arc, then resamples to 48 points
    depth     values are clipped to z +- 1 m and centred on z

LFE's full scan does neither: a fixed beam grid, globally normalised by
1 - r/10.  So build the 2x2 and classify each window with 1-NN, which has
essentially no capacity of its own -- whatever separates the four numbers is
the *representation*, not the model.

    the adaptive-angular rows beat the fixed ones, and by more at long range
        -> distance normalisation is what DROW is buying  -> SCALE
    the four rows agree
        -> the representations carry the same information -> CAPACITY

1-NN also brackets the Bayes error of the best row (Cover & Hart, IEEE Trans.
Inf. Theory 13(1), 1967):  E_Bayes <= E_1NN <= 2 E_Bayes (1 - E_Bayes), so
E_Bayes >= (1 - sqrt(1 - 2 E_1NN)) / 2.

Scope, stated so it is not overclaimed: this does NOT simulate LFE, whose
U-Net receptive field is far wider than any window here.  It tests the one
narrow claim the governing hypothesis rests on -- that normalising a window by
measured range makes person-vs-background easier, and increasingly so with
range.

Usage
-----
    python repr_probe.py
    python repr_probe.py --train-stride 40 --test-stride 20
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.datasets.frog_dataset import FROG_Dataset, frog_laser_angles

PERSON_R = 0.4
NEG_R = 1.0             # a negative beam must be this far from every person
NSAMP = 48              # DROW's cutout length
WIN_SZ = 1.66           # DROW's cutout width in metres
INC = np.pi / 720.0     # FROG beam spacing, radians
UNK = 29.99


def _windows(scan, beams, hw):
    """Resample a window of half-width hw[k] beams around each beams[k]."""
    n = len(scan)
    t = np.linspace(0.0, 1.0, NSAMP, dtype=np.float64)
    frac = (beams - hw)[:, None] + (2 * hw)[:, None] * t[None, :]
    lo = np.floor(frac).astype(int)
    alpha = frac - lo
    oob_lo, oob_hi = (lo < 0) | (lo >= n), (lo + 1 < 0) | (lo + 1 >= n)
    v = (scan[np.clip(lo, 0, n - 1)] * (1 - alpha)
         + scan[np.clip(lo + 1, 0, n - 1)] * alpha)
    v[oob_lo & ~oob_hi] = scan[np.clip(lo + 1, 0, n - 1)][oob_lo & ~oob_hi]
    v[~oob_lo & oob_hi] = scan[np.clip(lo, 0, n - 1)][~oob_lo & oob_hi]
    v[oob_lo & oob_hi] = UNK
    return v


def _local_cartesian(scan, beams, angles, k=12, win_sz=None, hw_fixed=None):
    """Metric offsets from the window's own centre point, in that beam's frame.

    The owner's proposal (TODO.md A42): drop polar preprocessing and hand the
    model Cartesian coordinates from the start.  For each sampled beam, take
    its neighbours, convert to (x, y), subtract the centre beam's own endpoint
    and rotate so that endpoint's bearing points along +y.

    Against DROW's `cutout()` this is the *same* normalisation on the axis Tier
    0 found dominant -- subtracting the local reference, worth 7.8 pp of 1-NN
    error -- while keeping **both** displacement components where the 1D depth
    tunnel keeps only the radial one.  It also deletes the fiddliest part of
    that function: no range-dependent half-width, no bilinear resample, no
    out-of-bounds branches.

    Three window choices, and the third exists because the first two are not a
    fair comparison against `cutout()`:

      neither          `2k` raw neighbouring beams. Window extent shrinks with
                       range, and `k` sets extent and dimensionality at once.
      `hw_fixed`       a constant half-width in *beams*, resampled to `2k`
                       points. This is the honest counterpart of the probe's
                       `fixed` cutout row, which uses a 41-beam half-width --
                       extent and dimensionality are now separate knobs.
      `win_sz`         a range-adaptive half-width spanning `win_sz` metres,
                       resampled to `2k` points: DROW's angular normalisation,
                       the 2.5 pp this asks whether we can do without.

    Returns (N, 2k, 2).
    """
    beams = np.asarray(beams, dtype=np.float64)
    n = len(scan)
    # Indexing needs integers; the adaptive branch's window arithmetic needs
    # the float form, so keep both rather than rounding once and losing the
    # sub-beam position.
    bi = np.clip(np.rint(beams).astype(int), 0, n - 1)
    px, py = scan * -np.sin(angles), scan * np.cos(angles)
    phi = angles[bi]

    if win_sz is None and hw_fixed is None:
        idx = np.clip(bi[:, None] + np.arange(-k, k)[None, :], 0, n - 1)
        wx, wy = px[idx], py[idx]
    else:
        if win_sz is not None:
            z = np.clip(scan[bi], 0.2, None)
            hw = np.arctan(0.5 * win_sz / z) / INC
        else:
            hw = np.full(len(bi), float(hw_fixed))
        t = np.linspace(0.0, 1.0, 2 * k, dtype=np.float64)
        frac = (beams - hw)[:, None] + (2 * hw)[:, None] * t[None, :]
        lo = np.clip(np.floor(frac).astype(int), 0, n - 1)
        hi = np.clip(lo + 1, 0, n - 1)
        a = frac - np.floor(frac)
        wx = px[lo] * (1 - a) + px[hi] * a
        wy = py[lo] * (1 - a) + py[hi] * a

    cx, cy = px[bi], py[bi]
    dx, dy = wx - cx[:, None], wy - cy[:, None]
    # Rotate by -phi so the centre beam points along +y: R maps
    # (-sin phi, cos phi) -> (0, 1). Without this a person at -80 deg and at
    # +80 deg are different objects to a nearest-neighbour classifier.
    s, c = np.sin(phi)[:, None], np.cos(phi)[:, None]
    rx, ry = c * dx + s * dy, -s * dx + c * dy
    # Clip the radial component to +-1 m. This is the exact analogue of DROW's
    # depth tunnel (`clip(w, z +- 1)`), and it is here to keep the comparison
    # fair rather than to help: without it the two representations differ in
    # what they do with far background as well as in coordinate system, and the
    # probe could not attribute the result to either. The lateral component
    # needs no clip -- the window's angular extent already bounds it.
    return np.stack([rx, np.clip(ry, -1.0, 1.0)], axis=-1)


def _depth_centred(w, z):
    """DROW: clip to a +-1 m tunnel around the beam's own range, then centre."""
    return np.clip(w, z[:, None] - 1.0, z[:, None] + 1.0) - z[:, None]


def _depth_lfe(w):
    """LFE: one global normalisation, 1 - clip(r, 0.2, 10) / 10."""
    return 1.0 - np.clip(w, 0.2, 10.0) / 10.0


def sample(split, mode, stride, rng_seed, hw_fixed):
    """-> dict of representation -> (N, NSAMP), plus labels and ranges."""
    ds = FROG_Dataset(split=split, mode=mode, auto_download=False, verbose=False)
    angles = frog_laser_angles(720).astype(np.float64)
    sin_a, cos_a = -np.sin(angles), np.cos(angles)
    rs = np.random.default_rng(rng_seed)

    scans, beams, labels = [], [], []
    for seq in range(len(ds.det_id)):
        for det_idx in range(0, len(ds.det_id[seq]), stride):
            iscan = int(ds.idet2iscan[seq][det_idx])
            wp = np.asarray(ds.det_wp[seq][det_idx], dtype=np.float64).reshape(-1, 2)
            if not len(wp):
                continue
            gx, gy = wp[:, 0] * -np.sin(wp[:, 1]), wp[:, 0] * np.cos(wp[:, 1])
            scan = np.asarray(ds.scans[seq][iscan], dtype=np.float64)
            bx, by = scan * sin_a, scan * cos_a
            dist = np.hypot(bx[:, None] - gx[None, :], by[:, None] - gy[None, :])

            pos = []
            for j in range(len(wp)):
                k = int(np.argmin(dist[:, j]))
                if dist[k, j] <= PERSON_R:
                    pos.append(k)
            if not pos:
                continue
            free = np.nonzero((dist.min(axis=1) > NEG_R) & (scan < 10.0))[0]
            if len(free) < len(pos):
                continue
            neg = rs.choice(free, size=len(pos), replace=False)
            for k in pos:
                scans.append(scan); beams.append(k); labels.append(1)
            for k in neg:
                scans.append(scan); beams.append(int(k)); labels.append(0)

    beams = np.asarray(beams, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    z = np.asarray([s[int(b)] for s, b in zip(scans, beams)], dtype=np.float64)

    hw_adapt = np.arctan(0.5 * WIN_SZ / np.clip(z, 0.2, None)) / INC
    out = {}
    for aname, hw in (("adaptive", hw_adapt), ("fixed", np.full_like(z, hw_fixed))):
        w = np.stack([_windows(s, np.array([b]), np.array([h]))[0]
                      for s, b, h in zip(scans, beams, hw)])
        out[(aname, "centred")] = _depth_centred(w, z).astype(np.float32)
        out[(aname, "lfe")] = _depth_lfe(w).astype(np.float32)

    # A42: the same two angular choices, but metric offsets from the window's
    # own centre point instead of a 1D range tunnel. 24 points x 2 channels =
    # 48 dims, matching the cutout exactly so this is not a dimensionality
    # comparison.
    for aname, kw in (("adaptive", {"win_sz": WIN_SZ}),
                      ("fixed", {"hw_fixed": hw_fixed})):
        xy = np.stack([_local_cartesian(s, np.array([b]), angles,
                                        k=NSAMP // 4, **kw)[0]
                       for s, b in zip(scans, beams)])
        out[(aname, "cartesian")] = xy.reshape(len(xy), -1).astype(np.float32)

    return out, labels, z, float(np.median(hw_adapt))


def knn_err(ref, ref_y, qry, qry_y, chunk=2048):
    """Brute-force 1-NN error, BLAS-backed. -> per-query correctness (bool)."""
    rn = (ref * ref).sum(1)
    ok = np.empty(len(qry), dtype=bool)
    for i in range(0, len(qry), chunk):
        q = qry[i:i + chunk]
        d = rn[None, :] - 2.0 * (q @ ref.T)          # + |q|^2, constant per row
        ok[i:i + chunk] = ref_y[d.argmin(1)] == qry_y[i:i + chunk]
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="official")
    ap.add_argument("--train-stride", type=int, default=60)
    ap.add_argument("--test-stride", type=int, default=30)
    ap.add_argument("--cache", type=Path, default=_HERE / ".repr_cache")
    args = ap.parse_args()
    args.cache.mkdir(exist_ok=True)

    # The version tag is part of the key on purpose: adding a representation
    # silently invalidates every cached sample, and this session already lost
    # time to a stale artifact that nothing flagged (CHANGELOG 2026-09-11).
    f = args.cache / "v3_s{}_{}_{}.pkl".format(args.train_stride, args.test_stride, args.mode)
    if f.exists():
        ref, ref_y, ref_z, qry, qry_y, qry_z, hw_fixed = pickle.loads(f.read_bytes())
        print("[cached] {}".format(f.name))
    else:
        # One pass over test first, purely to fix the fixed-window width at the
        # median of what the adaptive one uses -- so neither side is handed a
        # systematically wider window than the other.
        qry, qry_y, qry_z, hw_med = sample("test", args.mode, args.test_stride, 0, 32.0)
        hw_fixed = round(hw_med)
        qry, qry_y, qry_z, _ = sample("test", args.mode, args.test_stride, 0, hw_fixed)
        ref, ref_y, ref_z, _ = sample("train", args.mode, args.train_stride, 1, hw_fixed)
        f.write_bytes(pickle.dumps((ref, ref_y, ref_z, qry, qry_y, qry_z, hw_fixed)))

    print("\nreference {} windows from FROG {} train (stride {})"
          .format(len(ref_y), args.mode, args.train_stride))
    print("query     {} windows from FROG {} test  (stride {})"
          .format(len(qry_y), args.mode, args.test_stride))
    print("classes balanced by construction: {:.1%} positive".format(qry_y.mean()))
    print("fixed window half-width {} beams ({:.1f} deg), the median of what the"
          .format(hw_fixed, 2 * hw_fixed * 0.25))
    print("adaptive window uses -- so neither side gets a wider window on average.")

    bins = [0, 2, 3, 4, 5, 6, 8, 10]
    hdr = "\n{:>10} {:>8} {:>9} {:>9} {:>8} ".format(
        "angular", "depth", "1-NN err", "E_Bayes>=", "miss-p")
    print(hdr + " ".join("{:>9}".format("{}-{}m".format(lo, hi))
                         for lo, hi in zip(bins[:-1], bins[1:])))
    print("{:>10} {:>8} {:>9} {:>9} {:>8} ".format("", "", "", "", "")
          + " ".join("{:>9}".format("n={}".format(int(((qry_z >= lo) & (qry_z < hi)).sum())))
                     for lo, hi in zip(bins[:-1], bins[1:])))
    results = {}
    for aname in ("adaptive", "fixed"):
        for dname in ("centred", "lfe", "cartesian"):
            ok = knn_err(ref[(aname, dname)], ref_y, qry[(aname, dname)], qry_y)
            e = 1.0 - ok.mean()
            lower = (1 - np.sqrt(max(0.0, 1 - 2 * e))) / 2
            missp = 1 - ok[qry_y == 1].mean()      # positives called background
            cells = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (qry_z >= lo) & (qry_z < hi)
                cells.append("{:>9.1%}".format(1 - ok[m].mean()) if m.sum() >= 100
                             else "{:>9}".format("-"))
            results[(aname, dname)] = 1 - ok
            print("{:>10} {:>8} {:>9.2%} {:>9.2%} {:>8.1%} ".format(
                aname, dname, e, lower, missp) + " ".join(cells))

    print("\nThe range-adaptive window's advantage, per bin (fixed err - adaptive err):")
    print("{:>10} ".format("depth") + " ".join(
        "{:>9}".format("{}-{}m".format(lo, hi)) for lo, hi in zip(bins[:-1], bins[1:])))
    for dname in ("centred", "lfe", "cartesian"):
        cells = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (qry_z >= lo) & (qry_z < hi)
            gain = results[("fixed", dname)][m].mean() - results[("adaptive", dname)][m].mean()
            cells.append("{:>+8.1f}pp".format(100 * gain) if m.sum() >= 100
                         else "{:>9}".format("-"))
        print("{:>10} ".format(dname) + " ".join(cells))

    print("\nThe two `adaptive` rows are DROW's parameterisation; the two `fixed`")
    print("rows drop the range-adaptive width. `lfe` depth is LFE's own global")
    print("1 - r/10; `centred` is DROW's per-beam tunnel. Bayes bound from")
    print("Cover & Hart (1967), valid asymptotically and for the best row only.")


if __name__ == "__main__":
    main()
