# Data model — dataset and odometry file formats

*keywords:* DROW, FROG, JRDB, odom.npz, session splitting, sequence, h5, csv, det_id, timestamps, gap

What one record is, in what units, and what a rebuild invalidates.

## DROW (`DROW_Dataset`)

Per-sequence flat files: `<seq>.bag.csv` (index, timestamp, 450-beam scan), `<seq>.bag.odom2` (index, timestamp, x/y/angle real odometry), `<seq>.bag.wa`/`.wc`/`.wp` (index, per-class detection arrays: walker, wheelchair, person).
Annotations are **sparse** — only ~5.2% of scans carry ground truth, at a regular ~1-in-5 cadence (median gap 0.4 s).
`LidarFrameDataset`'s sample index enumerates only annotated frames; the raw-scan temporal window still draws from the full, dense, unannotated-included array, so an annotated frame's history is real consecutive sensor data, not backfilled.

## FROG (`FROG_Dataset`)

One HDF5 file per split-group holds `scans` (N x 720 beams), `timestamps` (float64 seconds), `circles`/`circle_idx`/`circle_num` (dense, every-frame person annotations), and `split` (train/val/test per frame).
Every scan is annotated — no sparse-label problem the way DROW has one.
`FROG_Dataset.scan_time` and the odometry struct's `"t"` field store this same timestamp as `float64`, matching the source `.h5` — **not** `float32`, which loses all sub-minute precision at Unix-epoch magnitude (~1.4e9); nothing in the training/eval pipeline reads either field directly, so this only matters if you do.

**A single `.h5` file can bundle more than one real recording session.**
`frog_11-36_12-43_train_val.h5` contains two of the FROG paper's six recorded sessions (11:36 and 12:43 starts) with a ~24.6-hour real gap between them, plus 61 much smaller (0.1-90s) within-session pauses.
`_load_h5()` splits on any gap exceeding `session_gap_s=300.0` into a separate *sequence*, mirroring DROW's own multi-sequence structure — chosen specifically to isolate the one genuine cross-session boundary without splitting on the 61 sub-90s pauses.
`get_scan()`'s existing clamp-at-sequence-start logic then protects every session boundary the same way it already protects the very start of a recording: a temporal window never reaches across it.
This matters for both odometry integration and plain raw-scan stacking — a window built with zero odometry can still pull history across an unrelated recording if the file is treated as one sequence.

## FROG odometry — real official files, preferred over the ICP fallback

The FROG authors publish real, official per-session odometry as `frog_<HH-MM>_odom.npz` (`ts` + `data` arrays: X, Y, heading, ~10 Hz — sparser than scan rate, aligned via `np.interp`) at `https://robotics.upo.es/datasets/frog/laser2d_people/data/`.
`FROG_Dataset.download()` fetches these alongside each `.h5` file (best-effort; falls back to the estimated/zero path with a warning if a fetch fails, never blocks the `.h5` download itself).
`_session_odom_filenames()` parses `\d{2}-\d{2}` session tokens out of the `.h5` filename to match each session to its own real odometry file; `_load_h5()` prefers a matching real file when present on disk, and falls back to `_load_or_fake_odom()` (zero odometry, or an ICP-estimated sidecar if one has been generated) otherwise.
As of this writing only `frog_11-36_odom.npz`/`frog_12-43_odom.npz` (the two sessions inside `train_val.h5`) are fetched into the real dataset directory; the four sessions belonging to other splits have not been (`TODO.md`).

**Rebuilding the ICP pseudo-odometry sidecar** (`utils/generate_frog_odometry.py`, only relevant when a session has no real file) writes `<h5-stem>_odom.npz` next to the source `.h5`.
It computes real timestamp gaps itself and forces zero relative motion across any gap exceeding `max_dt` (default 0.1s) *before* asking ICP for an estimate at all — ICP's own magnitude clamp cannot be trusted alone, since a small, plausible-looking delta can come from matching two genuinely unrelated scenes across a gap.
Any training run that used a since-regenerated sidecar, or used the pre-real-odometry fallback for a session that now has a real file, has stale results and should be re-evaluated (`performance-log.md`'s dtime sweep is the current example).

## JRDB (`JRDB_Dataset`)

DROW-compatible format, 541 beams, 270 deg FoV.
Requires free manual registration at jrdb.erc.monash.edu and manual extraction into `library/follow_the_drow/include/JRDB-data/` — never auto-downloaded, so its absence is normal (`gotchas.md`).

## `LiveDataset` (ROS deployment only)

No file format — a `collections.deque(maxlen=time_frame)` sliding window fed by `push_measure()` from live ROS topics.
Exists only for real-time inference; see `deployment.md`.
