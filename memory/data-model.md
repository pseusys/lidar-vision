# Data model — dataset and odometry file formats

*keywords:* DROW, FROG, JRDB, odom.npz, session splitting, sequence, h5, csv, det_id, timestamps, gap

What one record is, in what units, and what a rebuild invalidates.

## DROW (`DROW_Dataset`)

Per-sequence flat files: `<seq>.bag.csv` (index, timestamp, 450-beam scan), `<seq>.bag.odom2` (index, timestamp, x/y/angle real odometry), `<seq>.bag.wa`/`.wc`/`.wp` (index, per-class detection arrays: walker, wheelchair, person).
Annotations are **sparse** — only ~5.2% of scans carry ground truth, at a regular ~1-in-5 cadence (median gap 0.4 s).
`LidarFrameDataset`'s sample index enumerates only annotated frames; the raw-scan temporal window still draws from the full, dense, unannotated-included array, so an annotated frame's history is real consecutive sensor data, not backfilled.

## FROG (`FROG_Dataset`)

One HDF5 file per split-group holds `scans` (N x 720 beams), `timestamps` (float64 seconds), `circles`/`circle_idx`/`circle_num` (dense, every-frame person annotations), and `split` (a per-frame train/val flag this project deliberately ignores — see below).
Every scan is annotated — no sparse-label problem the way DROW has one.
`FROG_Dataset.scan_time` and the odometry struct's `"t"` field store this same timestamp as `float64`, matching the source `.h5` — **not** `float32`, which loses all sub-minute precision at Unix-epoch magnitude (~1.4e9); nothing in the training/eval pipeline reads either field directly, so this only matters if you do.

### What the FROG paper actually says about its own split

Checked against the paper (arXiv:2306.08531) rather than inferred, because the split's *motivation* decides how much authority it carries.

**Six sessions, all ~30 min, Table 1:**

| session | scans (raw) | annotated people | role |
| --- | --- | --- | --- |
| 10:31 | 64,238 | 127,600 | unused by the benchmark |
| **11:36** | 71,417 | **214,707** | train/val |
| **12:43** | 76,088 | **258,298** | train/val |
| 14:57 | 70,062 | 133,197 | unused |
| 15:53 | 60,758 | 133,023 | unused |
| 16:41 | 70,923 | 153,658 | test |

**The stated rationale, verbatim:**

> "The training/validation set is sourced from two different sequences recorded around the time of greatest attendance (around noon, maximizing the number of person annotations) and later randomly split in 90:10 proportion. The testing set is sourced from another different sequence."

So the two training sessions were chosen to **maximize person annotations** — and they are indeed the top two by that measure, by a wide margin.
That is a defensible choice for a detection benchmark and an explicit trade of *environment diversity* for *annotation density*.
It is not a claim that this split trains the best-generalizing model, and it should not be treated as one: this project's own A1 run overfits those two sessions hard (best epoch 1, gap +33.8pp).
Matching the split is required to compare against published numbers; it is not required to do good research (`TODO.md` A25).

The "randomly split in 90:10 proportion" confirms directly what this project found empirically: the val half is a **per-frame random holdout**, which is why its frames sit ~38 ms from training frames.

**The benchmark excludes empty scans, by the paper's own rule:**

> "In both cases, scans with empty lists of person annotations are excluded from the benchmark."

and

> "the FROG dataset provides annotations for every single scan"

Both matter.
The first means this project's `circle_num > 0` filter on held-out splits **is the official protocol**, not an interpretation of it.
The second means the extras' un-populated frames are **verified empty**, not un-annotated — so they are valid data for measuring false positives on person-free scenes (`TODO.md` A20).

**The file layout follows from those two sentences, and the arithmetic checks exactly:**

| file | paper's raw scans | file on disk | removed |
| --- | --- | --- | --- |
| `frog_11-36_12-43_train_val.h5` | 147,505 | 120,396 | 27,109 (18.4%) |
| `frog_16-41_test.h5` | 70,923 | 50,088 | 20,835 (29.4%) |
| `frog_10-31.h5` | 64,238 | 64,238 | **0** |
| `frog_14-57.h5` | 70,062 | 70,062 | **0** |
| `frog_15-53.h5` | 60,758 | 60,758 | **0** |

The two benchmark files are **pre-filtered to populated scans**; the three extras are the **raw recordings**.
That is the whole explanation for the composition difference, and the lower removal rate for train_val (18.4% against test's 29.4%) is consistent with those being the two most crowded sessions.

### The three loading modes (`mode=`)

One dataset, three partitions, selected at load time — `FROG_Dataset(mode=...)`, `--frog-mode` on `train.py` / `evaluate.py` / `render_video.py`.
Recordings, not files, are the unit: `frog_11-36_12-43_train_val.h5` bundles two recordings ~24.6 h apart and `balanced` sends them to different splits.
**A recording a split does not own is not loaded at all**, so no temporal window can draw history across a split boundary.

| mode | train | val | test | what it answers |
| --- | --- | --- | --- | --- |
| `official` *(default)* | train_val, paper's `split == 0`, 106,400 scored | the three extras, populated, 122,405 | frog_16-41, 50,088 | the published benchmark |
| `transferred` | **identical to official** | **identical to official** | the three extras' **person-free** frames, 72,533 | what a detector does when nobody is there |
| `balanced` | 12-43 + 14-57, 133,367 | 11-36 + 10-31, 121,329 | 16-41 + 15-53, 110,846 | the realistic mix — every split ~19-21% empty |

`transferred` shares official's train and val *deliberately*, so one checkpoint serves both and a transferred number is the official model measured elsewhere rather than a separate experiment (`TODO.md` A32b).

**`balanced` is our partition, not the paper's**, so it ignores the paper's per-frame `split` field entirely and uses whole recordings.
Its pairing was chosen from the six possible populated-only/raw matchings to minimise the spread of empty frames — 20.6 / 19.8 / 19.1 % — and its residual prior shift (3.70 people per populated frame in train against 3.20 in test) is *smaller* than `official`'s own (3.93 -> 3.07).

**Two rules for `balanced`, both easy to violate by accident:**

- Its **test** split is `16-41 + 15-53` because the published baselines (LFE, DROW3, DR-SPAAM) were trained by their authors on 11-36 and 12-43. Neither is in it, so their numbers there are honest — and it contains frog_16-41 whole, so the official-test number is extractable from the same pass.
- Its **val** split *does* contain 11-36, which those authors trained on. **Never report a published baseline's number on `balanced` val.**

Whole recordings rather than patches, deliberately: dealing blocks of one recording across splits would give every split all six environments, but adjacent blocks share venue, lighting and the same people in the same clothes — a diversity gain nothing could verify, bought with the leakage property this project spent a session eliminating.

### The run-up: frames too close to a sequence start

`get_scan()` clamps at index 0 of its own sequence, so a frame closer than `(T-1) * dtime` scans to a sequence start gets a **padded, motion-free history** — a wrong input paired with a correct label, aimed squarely at what this project studies.
The 0.5 s session splitting made this matter: train is 54 sequences (shortest 90 scans), test 47 (median 688).

`RUNUP_SCANS = 40` drops those frames, and is a **fixed** width, never `(T-1) * dtime` — sizing it from `dtime` would give every point of a `dtime` sweep a different frame set, the same trap the guard band fell into.
40 covers `T=5, dtime=10`; a wider `dtime` means raising it and re-running everything at the new value.

Measured cost, and the asymmetry is structural:

| split | populated | kept | dropped | why |
| --- | --- | --- | --- | --- |
| train | 108,356 | 106,400 | **1.81%** | 54 sequences |
| val | 122,405 | 122,405 | **0.00%** | three unbroken raw recordings, zero pauses |
| test | 50,088 | 50,088 | **0.00%** | `_SPLIT_RUNUP["test"] = 0`, deliberately |

**Test keeps them on purpose.** The FROG paper scores every populated scan, so trimming 3.75% of test frames would break comparability with every published number in order to fix a smaller contaminant.
Pass `runup_scans=0` (or a positive value for test) to override per call when the temporal analysis wants the clean subset reported separately.

### `split` selects files, and it is a whitelist

`_SPLIT_TO_FILE_KEYS` maps `train`/`val` to `frog_11-36_12-43_train_val.h5` and `test` to `frog_16-41_test.h5`, and **only** those files are opened.
It has to be a whitelist rather than a glob or an exclusion, because both halves of the bug it replaces were real:

- the loader used to `glob("*.h5")` and mark **every** frame of every file it found as annotated whenever the split was not `train`/`val`, so `split="test"` returned all 120,396 frames of the *training* file — the test file having never been downloaded.
  Every FROG "test" number measured before 2026-09-10 is train-set accuracy.
- and the test file carries no `split` field, so once it *is* on disk, a glob-based `split="train"` would sweep all of it into training through the same "no split field, so every frame is a detection" branch.

Measured after the fix: train 102,356 annotated frames / val 12,040 / test **50,088** — the test split is its own recording rather than a copy of the training one.

### Every split is a disjoint set of whole recordings

FROG publishes six recordings.
They partition cleanly, and nothing is carved:

| split | recordings | annotated frames |
| --- | --- | --- |
| `train` | 11-36 + 12-43 (`train_val.h5`), the paper's own `split == 0` frames | **108,356** |
| `val` | 10-31, 14-57, 15-53 — whole | 195,058 |
| `test` | 16-41 — whole | 50,088 |

**A held-out split annotates only frames containing at least one person**, matching the composition both official files already have.
Measured: `train_val` and `frog_16-41_test` are **100%** populated, while the three raw extras are ~62%.
That is a curation difference — the authors trimmed the benchmark files to populated frames and left the extras raw — not a property of the scenes: conditioned on being populated the extras average 3.13-3.36 people/frame against test's 3.07.

**The extras' un-populated frames are genuinely person-free, not un-annotated**, checked directly because it decides whether they can be trusted.
At a 0 -> N transition the count jumps by a **median of 1**, and occupancy trails to ~1.1-1.4 before a gap and resumes at ~1.4-1.6 after it — people leaving and entering the field of view one at a time.
An annotator stopping mid-segment would instead show the count jumping straight to that segment's full occupancy.

**Annotation quality where it exists matches the benchmark's own.**
Mean |delta count| between consecutive annotated frames is 0.014-0.017 for the three extras against **0.022 for `frog_16-41_test`**, with 0.17-0.25% of steps changing by more than one against test's 0.17%.
The extras are, if anything, marginally smoother than the official test recording.
This is the statistic that says val is trustworthy for model selection.
Leaving the empty ~38% in val puts pure-false-positive-opportunity frames into a precision-recall metric that neither train nor test ever sees, and it is devastating: **val wp-AUC 50.8% against a train probe of 85.7%, a +35pp gap that was entirely composition artifact**.
The rule is a protocol match rather than a convenience, and the evidence is that it is a **no-op on the test recording**.
After it: train 108,356 frames at 3.93 people/frame, val 122,405 at 3.22, test 50,088 at 3.07 — val now sits between the two.

**Val is cross-recording on purpose.**
A val split carved out of the *training* recordings measures same-recording generalization, which systematically overestimates the cross-recording number the test set reports — so early stopping would select for the wrong thing.
Measured on the first attempt at exactly that (10 contiguous blocks inside the training recordings, 150-scan guard bands): val wp-AUC **82.7%** at epoch 1, far above DR-SPAAM's published cross-recording 75.3%, because it is not the same quantity.
The three extra recordings are fully annotated (60-65% of frames carry people, ~2 people/frame, ~26-29 min each), nothing else uses them, and they are continuous — zero pauses over 0.5 s — so a cross-recording val costs nothing at all.

**Train reproduces the paper's set exactly.**
Honouring `split == 0` gives 108,356 frames, precisely what DROW3/DR-SPAAM were trained on.
Using the whole file (120,396) would mean beating the published numbers partly on 11% more data.
FROG's `split == 1` half is a per-frame random holdout whose frames sit ~38 ms from training frames — useless as a holdout (a stride-9 train subsample and it score **identically**, 78.8% wp-AUC both) and not the authors' training data either, so it is simply dropped.

Because splits never share a recording, no window, no guard band and no near-duplicate frame can bridge them.
Guard bands are gone entirely, along with `_assign_split_labels()`; `git log` has them if a same-recording val is ever wanted again.

### Recordings, sessions, and why both thresholds exist

`_split_at_gaps()` is called at two scales, for two different jobs:

- **`RECORDING_GAP_S = 300.0`** separates the genuine recordings a file bundles.
  `frog_11-36_12-43_train_val.h5` contains two of the FROG paper's six sessions (11:36 and 12:43 starts) with a ~24.6-hour gap.
  Recordings own their odometry file and are the unit the train/val carve works on, so a val block never straddles one.
- **`SESSION_GAP_S = 0.5`** separates *recording pauses* within a recording — 60 of them across the two training recordings, up to **74.3 seconds** long, together 15-20% of each recording's wall time.
  `get_scan()` indexes by scan number, not time, so before this a nominal "0.38 s" `T=5`/`dtime=10` window that straddled a pause really spanned 74 seconds of two unrelated scenes; 2.4% of `dtime=10` windows did, scaling linearly with `dtime` and so biasing the sweep against exactly the wide windows it measured.

Sequences are the unit `get_scan()`'s clamp protects, so a session boundary is a hard wall: a temporal window never reaches across it.
Measured after the fix: the largest inter-scan gap inside any sequence is 464 ms (train) / 450 ms (test), down from 74,300 ms.
The two training recordings become 54 sequences, the test recording 47.
Per-scan interval *jitter* within a session is left alone as noise — there is nothing to be done about it, and `aligned_raw_scan()` works off real odometry timestamps rather than an assumed rate.

## FROG odometry — real official files, preferred over the ICP fallback

The FROG authors publish real, official per-session odometry as `frog_<HH-MM>_odom.npz` (`ts` + `data` arrays: X, Y, heading, ~10 Hz — sparser than scan rate, aligned via `np.interp`) at `https://robotics.upo.es/datasets/frog/laser2d_people/data/`.
**Heading is interpolated unwrapped and wrapped back** (`_load_odom()`, fixed 2026-09-14).
Before that, a 1° turn across ±180° was interpolated straight through 0°, which put single-frame heading jumps of up to 144° into the test recording — 54 of 50,041 consecutive-scan pairs above 2°, now 2, both across 124-248 ms gaps (`CHANGELOG.md`).
`FROG_Dataset.download()` defaults to `which="all"` and fetches every `.h5` plus its odometry; the constructor's auto-download asks for only the split it needs, so a training run never silently pulls ~1.4 GB it will not open.
`_session_odom_filenames()` parses `\d{2}-\d{2}` tokens out of the `.h5` filename to match each *recording* — not each session, since one recording now contains dozens — to its own real odometry file.
All six sessions' odometry is on disk as of 2026-09-10.

**`_load_odom()` raises; it never fakes.**
It used to be `_load_or_fake_odom()`, wrapping the whole load in `except Exception: pass` and substituting all-zero odometry on a missing file, a corrupt npz or a renamed key, silently.
Zero odometry is indistinguishable from a perfectly stationary robot, so that fallback turned "ego-motion compensation is broken" into "ego-motion compensation looks like a no-op" — and every conclusion about `--align-scans` and about `dtime` rests on the odometry being real.
Preference order: per-recording real file, then the shared file, then raise.

**The published heading is quantized to exactly 1.0 degree** — 361 unique values across the full ±π range, steps exact multiples of 1° (p90 2°, max 4°), sampled at 10 Hz against 26.2 Hz scans.
At FROG's 0.25°/beam that is a **4-beam step with ±2 beams of quantization noise** on every historical frame.
Below `dtime≈5` the true rotation is smaller than that quantum, so the rotation correction injects more jitter than it removes — expect odometry to start paying for itself only at wider windows, and treat "real odometry scored below no odometry at small `dtime`" as expected rather than as a bug.
Translation is unaffected: x/y are quantized at ~0.5 cm, well below the ~1.2 cm the robot moves per frame.

**Rebuilding the ICP pseudo-odometry sidecar** (`utils/generate_frog_odometry.py`, only relevant when a session has no real file) writes `<h5-stem>_odom.npz` next to the source `.h5`.
It computes real timestamp gaps itself and forces zero relative motion across any gap exceeding `max_dt` (default 0.1s) *before* asking ICP for an estimate at all — ICP's own magnitude clamp cannot be trusted alone, since a small, plausible-looking delta can come from matching two genuinely unrelated scenes across a gap.
Any training run that used a since-regenerated sidecar, or the pre-real-odometry fallback for a session that now has a real file, has stale results and should be re-evaluated.

## How much actually moves per frame (real odometry, FROG train split, 26.2 Hz / ~38.15 ms)

Measured, not assumed — this is what grounds any `dtime` choice in metres rather than intuition.
A dataset property, so unaffected by the 2026-09-10 measurement-bug retraction (`performance-log.md`), which is why it lives here rather than there.

**Robot's own ego-motion** (median per `dtime`-step, real odometry):

| `dtime` | span | median translation | median rotation |
| --- | --- | --- | --- |
| 1 | 38 ms | ~1.15 cm | ~0 deg |
| 5 | 191 ms | ~6.2 cm | ~0 deg |
| 10 | 382 ms | ~12.4 cm | ~0 deg |
| 20 | 763 ms | ~24.9 cm | ~0-0.07 deg |
| 40 | 1.5 s | ~49.8 cm | ~0-2.2 deg |

Small throughout — little for `aligned_raw_scan()` to have to correct regardless of `dtime`.

**Real person displacement**, matched directly against annotated positions (`det_wp`), not assumed from a walking-speed constant — greedy nearest-neighbour match per `dtime`-step, 2.5 m gate, ~13,000-15,000 matched pairs per row:

| `dtime` | span | median | mean | p90 |
| --- | --- | --- | --- | --- |
| 1 | 38 ms | 2.12 cm | 3.52 cm | 5.13 cm |
| 5 | 191 ms | 8.78 cm | 14.28 cm | 22.50 cm |
| 10 | 382 ms | 17.16 cm | 26.44 cm | 46.28 cm |
| 20 | 763 ms | 34.78 cm | 47.43 cm | 93.05 cm |
| 40 | 1.5 s | 67.14 cm | 81.56 cm | 160.70 cm |

Scales almost exactly linearly at ~1.7 cm per unit of `dtime` across this whole range.
At `dtime=1`, a person really does move only ~2 cm between sampled frames (well under a body width) — the "5 near-identical frames" intuition that prompted this check.
By `dtime=20`, ~35 cm — close to half a walking stride.

## JRDB (`JRDB_Dataset`)

DROW-compatible format, 541 beams, 270 deg FoV.
Requires free manual registration at jrdb.erc.monash.edu and manual extraction into `library/follow_the_drow/include/JRDB-data/` — never auto-downloaded, so its absence is normal (`gotchas.md`).

## `LiveDataset` (ROS deployment only)

No file format — a `collections.deque(maxlen=time_frame)` sliding window fed by `push_measure()` from live ROS topics.
Exists only for real-time inference; see `deployment.md`.
