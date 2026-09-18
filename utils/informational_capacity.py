"""
No-training proxy for how much decodable person-vs-background signal a given
(time_frame, dtime) configuration carries, before spending any epochs
training a real detector to find out.

Motivation
----------
A cheap SpaceTimeCNN screening run (utils/train.py --subsample 0.2 --epochs 8)
ranked dtime=15 below dtime=1 -- but that run's own per-epoch val_loss showed
dtime=15 still improving faster early on before overfitting the small
screening subsample, while dtime=1 never converged in the budget at all
(docs: CHANGELOG.md 2026-09-09, TODO.md A1). That confounds "this dtime has
less useful information" with "this run hadn't converged yet". This script
sidesteps training entirely and asks the input-side question directly: does
the raw temporal signal at each (T, dtime) actually separate person beams
from background beams, at all, before any learning happens?

Method
------
For each real annotated FROG frame, build the T-frame `aligned_raw_scan()`
window (ego-motion-corrected, same function used for real training) and
compute one scalar feature per beam: the per-beam standard deviation across
the T aligned frames. This collapses to (a monotone transform of) the
simple two-frame |diff| for T=2, so results here are consistent with the
original T=2, dtime-only sweep. Each beam is labelled "person" if it falls
within a few beams of a real annotated person's angular position in the
window's current (last) frame, else "background".

Two measures of how informative that one feature is for the person/
background label, chosen to be complementary rather than redundant:

1. AUC of the feature alone as a person-vs-background classifier.
   Equivalent to the probability a randomly drawn person-beam's feature value
   exceeds a randomly drawn background-beam's (the Mann-Whitney U / Wilcoxon
   rank-sum statistic) -- Hanley & McNeil, "The Meaning and Use of the Area
   Under a Receiver Operating Characteristic (ROC) Curve", Radiology 143(1),
   1982. This is the same quantity as this project's own wp-AUC (both are
   trapezoidal ROC-AUC), just computed on one raw feature instead of a
   trained model's output, and the same idea used throughout applied
   statistics/bioinformatics for cheap univariate marker screening before
   fitting anything -- see e.g. Pepe et al., "Combining diagnostic test
   results to increase accuracy", Biostatistics 4(3), 2003, and Guyon &
   Elisseeff, "An Introduction to Variable and Feature Selection", JMLR 3,
   2003, for the general "filter method" framing this falls under (score
   each feature's own discriminative power before any model is trained, as
   opposed to a "wrapper method" that needs a trained model to score).

2. Mutual information I(feature; label), in bits, between the continuous
   feature and the binary label -- an actual Shannon (1948) information
   quantity, not an analogy. Estimated via `sklearn.feature_selection.
   mutual_info_classif`, a k-nearest-neighbour estimator for exactly the
   continuous-feature/discrete-label case here -- Ross, "Mutual Information
   between Discrete and Continuous Data Sets", PLoS ONE 9(2), 2014.

Neither measure is an upper bound on what a trained network could extract:
a real detector combines many beams' features nonlinearly (spatial context,
learned thresholds), which this univariate proxy cannot see at all -- it can
only show whether *some* linear-rank-order signal exists in the raw temporal
channel, and how that changes across the (T, dtime) grid. Absence of a trend
here would be strong evidence against a configuration; presence of a trend
is necessary but not sufficient evidence for it.

The grid samples `dtime` adaptively per `T`, targeting a fixed list of total
window spans (`(T-1) * dtime * dt_native`) rather than reusing one `dtime`
list across every `T` -- a fixed list undersamples the peak for large `T`
(where the peak `dtime` is small and a coarse grid can straddle it without
ever landing near it) and oversamples it for small `T`. Confirmed necessary
empirically: an earlier fixed-grid pass (T in {2,3,5,10}, dtime in
{1,5,10,15,20,25,35,50}) found every row's peak clustering around a
1.1-1.7s span, but had only one or two points below each T's own peak,
which is not enough to confirm the *rise* side of the curve, only the
decline -- see performance-log.md for that pass's own numbers, kept as the
first (span-oblivious) evidence for the same conclusion this finer grid
checks properly on both sides.

Run from utils/ with the project venv:
    ../.venv/Scripts/python.exe informational_capacity.py
"""
import argparse
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, "..")
sys.path.insert(0, "../library")

from follow_the_drow.datasets.frog_dataset import FROG_Dataset, frog_laser_angles
from follow_the_drow.utils.drow_utils import aligned_raw_scan

N_BEAMS = 720
PERSON_HALFWIDTH_BEAMS = 4     # +/- beams around a person's angular position
FRAME_STRIDE = 15              # sample every 15th annotated frame for AUC
MI_SUBSAMPLE = 20_000          # random (feature, label) pairs for the MI estimate
RNG_SEED = 0
DT_NATIVE_S = 0.03815          # FROG's measured native rate, ~26.2 Hz

T_VALUES = (2, 3, 5, 10, 15, 20, 25)
# Target total window spans (seconds), sampled densely on both sides of the
# ~1.1-1.7s peak found by the earlier fixed-grid pass, converted to an
# integer dtime per T below rather than reused as one dtime list for every T.
TARGET_SPANS_S = (0.15, 0.3, 0.5, 0.75, 1.0, 1.2, 1.4, 1.6, 1.9, 2.3, 2.8, 3.5, 4.5, 6.0, 8.0)


def dtimes_for_T(T):
    dtimes = {1}
    for span in TARGET_SPANS_S:
        dtimes.add(max(1, round(span / ((T - 1) * DT_NATIVE_S))))
    return sorted(dtimes)


def per_beam_feature_and_label(ds, angles, T, dtime):
    all_feat, all_label = [], []
    for seq in range(len(ds.scans)):
        det_id = ds.det_id[seq]
        det_wp = ds.det_wp[seq]
        for i in range(0, len(det_id), FRAME_STRIDE):
            scan_id = int(det_id[i])
            if scan_id < dtime * (T - 1):
                continue
            scans_hist, odoms_hist = ds.get_scan(seq, scan_id, T, dtime=dtime)
            # FROG's own 0.25 deg/beam spacing, not aligned_raw_scan()'s default
            # of DROW's 0.5 deg. Omitting it halved the proxy's rotation
            # correction relative to what training applies through cfg.laser_inc
            # -- a train/proxy inconsistency on the exact axis the proxy is used
            # to choose (TODO.md A19).
            aligned = aligned_raw_scan(scans_hist, odoms_hist,
                                       FROG_Dataset.LASER_INCREMENT,
                                       angles=angles)[:, :, 0]  # (T, N)
            feat = aligned.std(axis=0)  # (N,) -- generalizes |diff| at T=2

            label = np.zeros(N_BEAMS, dtype=bool)
            for (r, phi) in det_wp[i]:
                beam = int(np.argmin(np.abs(angles - phi)))
                lo, hi = max(0, beam - PERSON_HALFWIDTH_BEAMS), min(N_BEAMS, beam + PERSON_HALFWIDTH_BEAMS + 1)
                label[lo:hi] = True

            all_feat.append(feat)
            all_label.append(label)
    return np.concatenate(all_feat), np.concatenate(all_label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frog-mode", default="official",
                        choices=["official", "transferred", "balanced"],
                        help="FROG partition whose train split to measure "
                             "(default: official). 'transferred' has the same "
                             "train split as 'official'; 'balanced' does not, so "
                             "a capacity grid computed under one does not carry "
                             "over to the other.")
    args = parser.parse_args()

    # The train split, and only the train split: choosing a (T, dtime) by
    # peeking at val or test would leak the very thing the sweep then measures.
    ds = FROG_Dataset(split="train", mode=args.frog_mode, auto_download=False)
    angles = frog_laser_angles(N_BEAMS)
    rng = np.random.RandomState(RNG_SEED)

    print(f"{'T':>3} {'dtime':>6} {'span(s)':>8} {'n_beams':>10} {'n_person':>9}  "
          f"{'med|feat|person':>16} {'med|feat|bg':>12}  {'AUC':>6} {'MI(bits)':>9}")
    for T in T_VALUES:
        for dtime in dtimes_for_T(T):
            span = (T - 1) * dtime * DT_NATIVE_S
            feat, label = per_beam_feature_and_label(ds, angles, T, dtime)
            n_person = int(label.sum())
            if n_person == 0 or n_person == len(label):
                continue
            auc = roc_auc_score(label, feat)

            idx = rng.choice(len(feat), size=min(MI_SUBSAMPLE, len(feat)), replace=False)
            mi_nats = mutual_info_classif(
                feat[idx].reshape(-1, 1), label[idx], discrete_features=False, random_state=RNG_SEED
            )[0]
            mi_bits = mi_nats / np.log(2)

            print(f"{T:>3} {dtime:>6} {span:8.2f} {len(feat):>10d} {n_person:>9d}  "
                  f"{np.median(feat[label])*100:14.2f}cm {np.median(feat[~label])*100:10.2f}cm  "
                  f"{auc:.4f} {mi_bits:9.4f}", flush=True)


if __name__ == "__main__":
    main()
