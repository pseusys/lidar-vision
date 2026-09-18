"""How many slots does stage 3's object memory need, and what evicts what? (TODO.md A43)

Two measurements for `docs/PROPOSAL.md` §6.

**Occupancy** (default). Stage 3 holds a fixed number of slots K. A slot is
spawned from an unmatched detection and retired only after `retire` scans
without a match, so the slots in use at any scan are every track that has
started and has not yet gone unmatched for longer than `retire`. Three sources
bracket the answer:

    annotated people       a floor: only people, association near-perfect
    LFE-Peaks detections   people + phantoms + fragments, at threshold 0.3
    LFE-PPN detections     the same, from the detector with ~2.5x the phantoms

Tracks are formed by gated Hungarian association against each track's last
position, with no motion model, so detection rows over-count -- the intended
direction under the owner's rule to overshoot K rather than undershoot it.

**Priority memory** (`--memory`). The owner's eviction rule of 2026-09-14:
every slot carries one value, its *accumulated* certainty; any candidate above
a low floor may spawn; when memory is full a candidate replaces the weakest
slot only if clearly more certain. Replayed on consecutive frames of
sub-threshold candidates, against two references -- unlimited capacity, and an
oracle that never evicts a slot holding an annotated person while another slot
is evictable. Reports how often people's slots are evicted and how many missed
people still have a slot near them.

FROG occupancy reuses `persistence_map.py`'s detection cache; the memory replay
writes its own (`.slot_cache/`), because it needs candidates far below 0.3.

Usage
-----
    python slot_budget.py
    python slot_budget.py --retire-s 0 5 10 --no-drow
    python slot_budget.py --memory --detector lfe-peaks
"""
import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "library"))
sys.path.insert(0, str(_HERE))

from follow_the_drow.utils.drow_utils import project_cartesian_from_polar

from motion_analysis import to_world


def associate(per_frame_world, scan_index, gate, max_gap):
    """Tracks as (first_scan, last_scan) spans.

    Each scan's points are assigned to live tracks by gated Hungarian on the
    distance to each track's last position -- the assignment the evaluation
    uses -- so a closer claim by one track cannot orphan another. A track stays
    live while it has gone at most `max_gap` consecutive scans unobserved; the
    count is in scan indices, so sparsely annotated DROW and fully annotated
    FROG mean the same thing by it.
    """
    spans, last_xy, live = [], [], []
    for pts, f in zip(per_frame_world, scan_index):
        f = int(f)
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        live = [k for k in live if f - spans[k][1] - 1 <= max_gap]
        used = np.zeros(len(pts), dtype=bool)
        if live and len(pts):
            prev = np.stack([last_xy[k] for k in live])
            cost = np.linalg.norm(prev[:, None, :] - pts[None, :, :], axis=-1)
            gated = np.where(cost > gate, 1e6, cost)
            for i, j in zip(*linear_sum_assignment(gated)):
                if gated[i, j] < 1e6:
                    k = live[i]
                    spans[k][1] = f
                    last_xy[k] = pts[j]
                    used[j] = True
        for j in np.nonzero(~used)[0]:
            spans.append([f, f])
            last_xy.append(pts[j])
            live.append(len(spans) - 1)
    return [tuple(s) for s in spans]


def occupancy(spans, n_frames, retire):
    """Slots in use per scan: each track holds one over [first, last + retire]."""
    diff = np.zeros(n_frames + 1, dtype=np.int64)
    for first, last in spans:
        diff[first] += 1
        diff[min(last + retire, n_frames - 1) + 1] -= 1
    return np.cumsum(diff[:-1])


# ---------------------------------------------------------------- priority memory

class PriorityMemory:
    """Rule-based stand-in for stage 3's slot management (`PROPOSAL.md` §6.3).

    One value per slot, its accumulated certainty:

        match      v <- v + gain * (1 - v) * score
        unmatched  v <- v * exp(-dt / tau),  tau = retire_s / ln(1 / floor)
        retire     v < floor

    so a slot at full certainty fades to the floor in exactly `retire_s`.
    Spawning fills free slots strongest-first; when full, the strongest
    remaining candidate replaces the weakest evictable slot only if
    `score > value + margin`, and a slot matched or spawned this step is never
    evictable. `oracle=True` evicts slots holding an annotated person last.
    `capacity=None` is unlimited.
    """

    def __init__(self, capacity, floor, margin, retire_s, gain=0.1, gate=0.6,
                 oracle=False):
        self.capacity = capacity
        self.floor = floor
        self.margin = margin
        self.tau = retire_s / math.log(1.0 / floor)
        self.gain = gain
        self.gate = gate
        self.oracle = oracle
        self.positions = np.empty((0, 2))
        self.values = np.empty(0)
        self.persons = np.empty(0, dtype=bool)

    @property
    def n_alive(self):
        return len(self.values)

    def step(self, xy, scores, dt, is_person=None):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        lab = (np.zeros(len(s), dtype=bool) if is_person is None
               else np.asarray(is_person, dtype=bool).reshape(-1))
        keep = s >= self.floor
        xy, s, lab = xy[keep], s[keep], lab[keep]

        pos, v, per = self.positions, self.values, self.persons
        matched = np.zeros(len(v), dtype=bool)
        claimed = np.zeros(len(s), dtype=bool)
        if len(v) and len(s):
            cost = np.linalg.norm(pos[:, None, :] - xy[None, :, :], axis=-1)
            rows = np.nonzero((cost <= self.gate).any(axis=1))[0]
            if len(rows):
                gated = np.where(cost[rows] > self.gate, 1e6, cost[rows])
                for i, j in zip(*linear_sum_assignment(gated)):
                    if gated[i, j] < 1e6:
                        k = rows[i]
                        matched[k] = claimed[j] = True
                        pos[k] = xy[j]
                        v[k] += self.gain * (1.0 - v[k]) * s[j]
                        per[k] = lab[j]
        v[~matched] *= math.exp(-dt / self.tau)

        alive = v >= self.floor
        retired = int((~alive).sum())
        pos, v, per, matched = pos[alive], v[alive], per[alive], matched[alive]

        cand = np.nonzero(~claimed)[0]
        cand = cand[np.argsort(-s[cand], kind="stable")]
        free = len(cand) if self.capacity is None else max(self.capacity - len(v), 0)
        take, rest = cand[:free], cand[free:]
        protected = np.concatenate([matched, np.ones(len(take), dtype=bool)])
        pos = np.concatenate([pos, xy[take]])
        v = np.concatenate([v, s[take]])
        per = np.concatenate([per, lab[take]])

        evicted = evicted_person = 0
        if len(rest):
            idx = np.nonzero(~protected)[0]
            order = (np.lexsort((v[idx], per[idx])) if self.oracle
                     else np.argsort(v[idx], kind="stable"))
            for c, k in zip(rest, idx[order]):
                if s[c] <= v[k] + self.margin:
                    break
                evicted += 1
                evicted_person += int(per[k])
                pos[k], v[k], per[k] = xy[c], s[c], lab[c]

        self.positions, self.values, self.persons = pos, v, per
        return dict(spawned=len(take), evicted=evicted,
                    evicted_person=evicted_person, retired=retired)


def _covered(slot_xy, gt_xy, gate):
    """Per annotated person: does a slot sit within `gate`, one slot per person?"""
    slot_xy = np.asarray(slot_xy, dtype=np.float64).reshape(-1, 2)
    gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    hit = np.zeros(len(gt_xy), dtype=bool)
    if len(slot_xy) and len(gt_xy):
        cost = np.linalg.norm(gt_xy[:, None, :] - slot_xy[None, :, :], axis=-1)
        cols = np.nonzero((cost <= gate).any(axis=0))[0]
        if len(cols):
            gated = np.where(cost[:, cols] > gate, 1e6, cost[:, cols])
            for i, j in zip(*linear_sum_assignment(gated)):
                if gated[i, j] < 1e6:
                    hit[i] = True
    return hit


def coverage(slot_xy, gt_xy, gate):
    return int(_covered(slot_xy, gt_xy, gate).sum())


def _assigned(det_xy, gt_xy, gate):
    """(detection matched?, person matched?) under gated Hungarian."""
    d_hit = np.zeros(len(det_xy), dtype=bool)
    g_hit = np.zeros(len(gt_xy), dtype=bool)
    if len(det_xy) and len(gt_xy):
        cost = np.linalg.norm(det_xy[:, None, :] - gt_xy[None, :, :], axis=-1)
        gated = np.where(cost > gate, 1e6, cost)
        for i, j in zip(*linear_sum_assignment(gated)):
            if gated[i, j] < 1e6:
                d_hit[i] = g_hit[j] = True
    return d_hit, g_hit


# ---------------------------------------------------------------- data

def frog_sequences(cache_dir):
    """[(label, [(scan_index, world_xy)] per sequence, n_scans per sequence)]."""
    from follow_the_drow.datasets.frog_dataset import FROG_Dataset

    ds = FROG_Dataset(split="test", mode="official", auto_download=False,
                      verbose=False)
    hz = 1.0 / float(np.median(np.diff(np.asarray(ds.scan_time[0], dtype=float))))
    n_scans = [len(ds.scans[s]) for s in range(len(ds.scans))]
    out, people = [], None
    for name, label in (("lfe-peaks", "LFE-Peaks detections"),
                        ("lfe-ppn", "LFE-PPN detections")):
        f = cache_dir / "{}_official_s1_0.3.pkl".format(name)
        if not f.exists():
            sys.exit("missing {} -- run `python persistence_map.py` first".format(f))
        seqs = pickle.loads(f.read_bytes())
        out.append((label, [[(i, dw) for i, _, dw, _ in fr] for fr in seqs]))
        if people is None:
            people = [[(i, to_world(gt, ds.odoms[s][i]["xya"]) if len(gt)
                        else np.empty((0, 2)))
                       for i, _, _, gt in fr] for s, fr in enumerate(seqs)]
    return [("annotated people", people)] + out, n_scans, hz


def drow_sequences():
    from follow_the_drow.datasets import DROW_Dataset

    ds = DROW_Dataset(verbose=False)
    hz = 1.0 / float(np.median(np.diff(np.asarray(ds.scan_time[0], dtype=float))))
    people = []
    for s in range(len(ds.det_id)):
        frames = []
        for d in range(len(ds.det_id[s])):
            i = int(ds.idet2iscan[s][d])
            rp = list(ds.det_wp[s][d]) + list(ds.det_wc[s][d]) + list(ds.det_wa[s][d])
            xy = np.array([project_cartesian_from_polar(r, p) for r, p in rp],
                          dtype=np.float64).reshape(-1, 2)
            frames.append((i, to_world(xy, ds.odoms[s][i]["xya"]) if len(xy)
                           else np.empty((0, 2))))
        people.append(frames)
    n_scans = [len(ds.scans[s]) for s in range(len(ds.scans))]
    return [("annotated people + aids", people)], n_scans, hz


def collect_candidates(name, cache_dir, floor, top_p):
    """Consecutive frames of `official` test: (time, scores, world xy, world GT).

    The detector runs with its own floor lowered to `floor`, then the top
    `top_p` candidates by score are kept -- stage 2's proposal budget.
    """
    f = cache_dir / "{}_official_f{}_p{}.pkl".format(name, floor, top_p)
    if f.exists():
        print("[cached] {}".format(f.name), flush=True)
        return pickle.loads(f.read_bytes())

    from follow_the_drow.datasets.frog_dataset import FROG_Dataset, frog_laser_angles
    from follow_the_drow.detectors import LFEPeaksDetector, LFEPPNDetector

    det = (LFEPeaksDetector(peak_height=floor) if name == "lfe-peaks"
           else LFEPPNDetector(score_thresh=floor))
    ds = FROG_Dataset(split="test", mode="official", auto_download=False,
                      verbose=False)
    angles = frog_laser_angles(720)
    seqs = []
    for s in range(len(ds.det_id)):
        frames = []
        for d in range(len(ds.det_id[s])):
            i = int(ds.idet2iscan[s][d])
            xya = ds.odoms[s][i]["xya"]
            c = np.asarray(det.detect(ds.scans[s][i], angles),
                           dtype=np.float64).reshape(-1, 3)
            c = c[np.argsort(-c[:, 0], kind="stable")[:top_p]]
            gt = np.array([project_cartesian_from_polar(r, p)
                           for r, p in ds.det_wp[s][d]],
                          dtype=np.float64).reshape(-1, 2)
            frames.append((float(ds.scan_time[s][i]), c[:, 0].copy(),
                           to_world(c[:, 1:3], xya) if len(c) else np.empty((0, 2)),
                           to_world(gt, xya) if len(gt) else np.empty((0, 2))))
        seqs.append(frames)
        print("  {} sequence {}: {} frames".format(name, s, len(frames)), flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(seqs))
    return seqs


def replay(seqs, capacity, floor, margin, retire_s, oracle, thresh=0.3, assoc=0.5):
    """Run the memory over every frame and score it against annotations."""
    acc = dict(frames=0, people=0, missed=0, covered=0, covered_missed=0,
               spawned=0, evicted=0, evicted_person=0)
    occ = []
    for frames in seqs:
        mem = PriorityMemory(capacity, floor, margin, retire_s, oracle=oracle)
        prev = None
        for t, scores, xy, gt in frames:
            dt = 1.0 / 26.2 if prev is None else max(t - prev, 1e-3)
            prev = t
            is_person, _ = _assigned(xy, gt, assoc)
            strong = scores >= thresh
            _, found = _assigned(xy[strong], gt, assoc)
            out = mem.step(xy, scores, dt, is_person=is_person)
            hit = _covered(mem.positions, gt, assoc)
            acc["frames"] += 1
            acc["people"] += len(gt)
            acc["missed"] += int((~found).sum())
            acc["covered"] += int(hit.sum())
            acc["covered_missed"] += int((hit & ~found).sum())
            for k in ("spawned", "evicted", "evicted_person"):
                acc[k] += out[k]
            occ.append(mem.n_alive)
    acc["occupancy"] = np.asarray(occ)
    return acc


def report_memory(name, rows):
    print("\n{}  (official test, every frame; cover = a slot within 0.5 m)".format(name),
          flush=True)
    print("  {:<24} {:>6} {:>6} {:>6} {:>8} {:>8} {:>10} {:>9} {:>10} {:>10}".format(
        "memory", "slots", "p99", "max", "spawn/f", "evict/f", "person-ev",
        "covered", "miss cov", "miss cov"), flush=True)
    print("  {:<24} {:>6} {:>6} {:>6} {:>8} {:>8} {:>10} {:>9} {:>10} {:>10}".format(
        "", "mean", "", "", "", "", "per 1k f", "people", "of missed", "pp of all"),
        flush=True)
    for label, a in rows:
        o, f = a["occupancy"], a["frames"]
        print("  {:<24} {:>6.1f} {:>6.0f} {:>6d} {:>8.2f} {:>8.3f} {:>10.2f} {:>8.1%} "
              "{:>9.1%} {:>9.2f}".format(
                  label, o.mean(), np.percentile(o, 99), int(o.max()),
                  a["spawned"] / f, a["evicted"] / f,
                  1000.0 * a["evicted_person"] / f,
                  a["covered"] / a["people"],
                  a["covered_missed"] / max(a["missed"], 1),
                  100.0 * a["covered_missed"] / a["people"]), flush=True)
    print("  people {}, missed at 0.3: {} ({:.1%})".format(
        rows[0][1]["people"], rows[0][1]["missed"],
        rows[0][1]["missed"] / rows[0][1]["people"]), flush=True)


def memory_main(args):
    for name in args.detector:
        seqs = collect_candidates(name, args.slot_cache, args.floor, args.top_p)
        rows = []
        for cap in args.capacity:
            capacity = None if cap <= 0 else cap
            margins = args.margin if capacity is not None else [0.0]
            modes = (False, True) if capacity is not None else (False,)
            for margin in margins:
                for oracle in modes:
                    label = "{} m={:g} {}".format(
                        "K=inf" if capacity is None else "K={}".format(capacity),
                        margin, "oracle" if oracle else "rule")
                    rows.append((label, replay(seqs, capacity, args.floor, margin,
                                               args.retire, oracle)))
                    print("  done: {} {}".format(name, label), flush=True)
        report_memory(name, rows)


# ---------------------------------------------------------------- report

def report(title, sources, n_scans, hz, gate, retire_s):
    print("\n{}  ({:.1f} Hz, gate {:.2f} m)".format(title, hz, gate), flush=True)
    print("  {:<26} {:>9} {:>7} {:>7} {:>7} {:>7} {:>7}".format(
        "source", "retire s", "mean", "p50", "p99", "p99.9", "max"), flush=True)
    for label, seqs in sources:
        present = np.concatenate([[len(xy) for _, xy in fr] for fr in seqs if fr])
        print("  {:<26} {:>9} {:>7.2f} {:>7.0f} {:>7.0f} {:>7.0f} {:>7d}".format(
            label, "in frame", present.mean(), np.percentile(present, 50),
            np.percentile(present, 99), np.percentile(present, 99.9),
            int(present.max())), flush=True)
        for r_s in retire_s:
            retire = int(round(r_s * hz))
            counts = []
            for s, fr in enumerate(seqs):
                if not fr:
                    continue
                spans = associate([xy for _, xy in fr], [i for i, _ in fr],
                                  gate=gate, max_gap=retire)
                c = occupancy(spans, n_scans[s], retire)
                counts.append(c[[i for i, _ in fr]])     # observed scans only
            c = np.concatenate(counts)
            print("  {:<26} {:>9g} {:>7.2f} {:>7.0f} {:>7.0f} {:>7.0f} {:>7d}".format(
                "", r_s, c.mean(), np.percentile(c, 50), np.percentile(c, 99),
                np.percentile(c, 99.9), int(c.max())), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--retire-s", type=float, nargs="+", default=[0, 1, 5, 10, 30])
    ap.add_argument("--gate", type=float, default=0.6,
                    help="FROG association gate in metres")
    ap.add_argument("--drow-gate", type=float, default=1.25,
                    help="DROW gate: annotations are 0.5 s apart, so a walker "
                         "at 2.5 m/s covers 1.25 m between them")
    ap.add_argument("--no-drow", action="store_true")
    ap.add_argument("--cache", type=Path, default=_HERE / ".persist_cache")
    ap.add_argument("--memory", action="store_true",
                    help="replay the value-ranked slot memory instead")
    ap.add_argument("--detector", nargs="+", default=["lfe-peaks", "lfe-ppn"],
                    choices=["lfe-peaks", "lfe-ppn"])
    ap.add_argument("--capacity", type=int, nargs="+", default=[256, 0],
                    help="slot counts; 0 means unlimited")
    ap.add_argument("--margin", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    ap.add_argument("--floor", type=float, default=0.01)
    ap.add_argument("--top-p", type=int, default=64)
    ap.add_argument("--retire", type=float, default=10.0,
                    help="seconds for a slot at full certainty to fade out")
    ap.add_argument("--slot-cache", type=Path, default=_HERE / ".slot_cache")
    args = ap.parse_args()

    if args.memory:
        memory_main(args)
        return
    sources, n_scans, hz = frog_sequences(args.cache)
    report("FROG official test (frog_16-41, every frame)", sources, n_scans, hz,
           args.gate, args.retire_s)
    if not args.no_drow:
        sources, n_scans, hz = drow_sequences()
        report("DROW test", sources, n_scans, hz, args.drow_gate, args.retire_s)


if __name__ == "__main__":
    main()
