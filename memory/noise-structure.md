# Noise structure

*keywords:* noise, denoising, denoiser, missing return, NaN, infinity, sentinel, clamp, 10 m, spike, jitter, outlier, sensor specs, UTM-30LX, S300, LMS500, rated range, accuracy, scan anomalies

What each dataset's range readings actually contain, what the sensors promise, and what loaders do to the readings before a model sees them.
Which denoising mechanism a dataset gets depends on what is found here (owner's framing, 2026-09-17):

| noise found | mechanism |
| --- | --- |
| missing returns, sentinel values | validity masking of the input, not denoising |
| single-frame A-B-A spikes | spike filter: replace the middle reading only when its neighbours agree |
| jitter that is large next to the 0.5 m match radius | temporal averaging (a fine temporal convolution: tested and removed 2026-09-17 as unneeded, `rejected-ideas.md`) |
| none of these | no denoising |

A-B-B steps are real motion and must never be filtered.
**Goal (owner, 2026-09-17): the model should be feedable with any dataset's raw data, without a dataset-specific preprocessor.** Whatever this file's measurements call for should be learned by the network or should work identically everywhere.
The measurements come from `utils/scan_anomalies.py` (TODO A49), run 2026-09-17 over every file of both datasets; full output in `results/scan_anomalies_{frog,drow}.json`. Robot motion is not compensated, so jitter and spike figures are upper bounds.

**Verdict, both datasets: the sensors are clean, and the only thing that needs handling is how "no return" is written.**

| | FROG | DROW |
| --- | --- | --- |
| no-return encoding | `+inf` (test, extras); **61.0 m** (train/val file only) | 29.96 m (also 29.98, 29.99) |
| invalid share | 9.53 % | 11.98 % |
| invalid in runs ≥ 26 frames | 63.6 % of invalid readings (≥ 0.65 s) | 65.3 % (≥ 2 s) |
| invalid in stretches ≥ 6 beams | 90.4 % | 58.0 % |
| flickers (1-2 frames) | 6.3 % of invalid readings | 9.6 % |
| invalid near annotated objects / elsewhere | 6.48 % / 10.37 % | 8.29 % / 12.66 % |
| spikes per valid triple | 0.23 %; 0.04-0.30 % inside 10 m | 0.34 % |
| spikes near objects / elsewhere | 0.30 % / 0.21 % | **1.09 % / 0.16 %** |
| median jitter under 10 m | 9-20 mm (rated ±30 mm) | 10 mm (1 cm rounding) |
| consecutive changes over 0.5 m, under 10 m | 1.8-3.6 % | 2.3-5.3 % |

What the numbers decide:

- **Invalid readings are open space, not dropouts.** They last long, cover wide stretches of beams, and are *rarer* on people than elsewhere. Healing from past frames would mostly copy stale surfaces into open space; at most the ~6-10 % flickers could benefit. A constant meaning "nothing within range" is the faithful encoding.
- **No spike filter.** On FROG spikes are rare and grow with range (mixed pixels at far edges). On DROW they are 6.8x more common near people: at 12.7 Hz a swinging leg crosses a beam within one frame, so A-B-A there is motion. Filtering it would erase legs, the same failure as the median denoiser.
- **No temporal averaging.** Jitter is at the sensor's rated accuracy, far below the 0.5 m match radius. This matches the fine-convolution result: no-fine and 5 taps tie at three seeds (TODO A50).

## Sensors

| | FROG | DROW | JRDB |
| --- | --- | --- | --- |
| scanner | Hokuyo UTM-30LX | SICK S300 (safety scanner) | two SICK LMS500 |
| rated range | 0.1-30 m (white sheet), max 60 m | "distance measuring range" 30 m | not looked up (not on disk) |
| rated accuracy | ±30 mm at 0.1-10 m, ±50 mm at 10-30 m | **not published** in the data sheet | not looked up |
| angular step | 0.25° | 0.5° | - |
| hardware field of view | 270° | 270° | - |
| in the data | 720 beams / 180° | 450 beams / 225°, 37 cm above ground | loader assumes 541 / 270° |
| nameplate rate | 40 Hz (25 ms/scan) | 12.5 Hz (paper) | - |
| **measured rate** | **40.0 Hz** (mean over linked intervals, every file) | **12.7 Hz** (per-file mean, 12.4-12.7) | - |
| timestamps | bunched: ~38 ms, ~38 ms, <0.1 ms; the median says 26.2 Hz | rounded to 0.05 s; the median says 10 Hz | - |

Sources: Hokuyo's [UTM-30LX product page](https://www.hokuyo-aut.jp/search/single.php?serial=169); SICK's [S30B-2011BA data sheet](https://www.sick.com/media/pdf/5/45/845/dataSheet_S30B-2011BA_1026820_en.pdf) (response time 80 ms, warning field 8 m at 15 % reflectivity); Beyer, Hermans, Linder, Arras & Leibe, [Deep Person Detection in 2D Range Data](https://arxiv.org/abs/1804.02463), §II ("a SICK S300 laser scanner at 12.5 Hz and an angular resolution of 1/2 degree at 37 cm above ground").

**DROW's timestamps are rounded to 0.05 s**, so its scan intervals are 0.05 s or 0.10 s (194k and 264k of them) and the median interval says 10 Hz. The true rate is the per-file mean, 12.7 Hz, matching the paper. Measured 2026-09-17 from the `time` column of every `DROWv2-data/*/*.csv`.

## FROG

**How missing returns are handled now (implemented 2026-09-17, owner's call; TODO A49):**

- The three-horizon detector reads **raw** scans (`FROG_Dataset(missing_return_m=None)`). The network applies one dataset-agnostic rule, `sanitize_ranges`: a reading that is non-finite, non-positive, or at or beyond `max_range_m` (`--max-range-m`, default 10 m) reads as `max_range_m`. The validity input channel is gone.
- **Why no validity channel and no healing:** invalid readings are open space in both datasets (verdict above). "No return" and "a return beyond the limit" mean the same thing to a detector with a range limit, so one value carries both.
- **Equivalence on FROG at 10 m:** the loader's old clamp plus the old 10 m clip produced exactly this input, with a validity channel that was always 1. So old one-tap checkpoints fold exactly into the new network (`fold_legacy_calibration`), and FROG results do not need re-running for this change.
- The baselines (LFE, DROW, DR-SPAAM) still get the loader's legacy clamp by default (`LEGACY_MISSING_RETURN_M` = 10 m), which is what every baseline number was measured with. Moving it into each baseline's own preprocessing is a TODO.

**FROG is annotated only out to 10 m.** The paper covers "the people in the field of view of the sensor (180°), at a maximum distance of 10 m" ([FROG, arXiv 2306.08531](https://arxiv.org/abs/2306.08531)). The authors' own detector, [`upo_laser_people_detector`](https://github.com/robotics-upo/upo_laser_people_detector), defaults `scan_far` to 10 m. So the loader's 10.0 is the annotation limit.
The recording itself is not cut at 10 m: about 226k finite readings lie between 30 and 60 m. Its non-finite readings are real missing returns, and they behave as open space (verdict above).

**Two encodings in one dataset.** `frog_11-36_12-43_train_val.h5` contains no inf at all; it writes no return as exactly 61.0 m (7.46M readings, 3.0 % of FROG's readings). The test file and the three extras use `+inf` (17.6M), plus 155 NaN and 689 `-inf`. Today both reach the network as a valid 10 m: the loader clamps inf, and `beam_features` clips 61.0. **Removing the loader clamp alone would give train and test different inputs.**

**Timestamps are bunched** (sensor table): 27.5 % of linked intervals are under a quarter of the 25 ms period. Stage 3 divides by them (TODO A50 phase 1b).

**Consecutive-scan disagreement of 1.30 cm (median, 2026-09-16) is biased low**: a beam missing in both frames scores |10 - 10| = 0. It is not evidence that FROG is clean.

## DROW

The loader (`drow_dataset.py`) reads the CSV ranges as they are, with no clamp. DROW's own `cutout` fills out-of-window points with `UNK = 29.99`. Measured 2026-09-17 over all 208.8M readings in `*/*.csv`, with `awk`:

- **No return is encoded as 29.96 m.** 24.82M readings (11.9 %) hold exactly 29.96, plus 146k at 29.98 and 41k at 29.99. These are just under the S300's 30 m range, so to a model they look like real far returns.
- No reading is zero, negative or non-finite; the smallest is 0.05 m. Ranges are rounded to 1 cm.

DROW annotations (`.wp`, `.wc`, `.wa`, 53,855 objects) are not range-limited the way FROG's are. 3,952 (7.3 %) lie beyond 10 m and 475 beyond 15 m. Four lie at 316-716 m and are label errors: `lunch_2015-11-26-12-04-23` frames 18065 and 18500 in its first chunk, `run_2015-11-25-11-46-08` frame 35, `run_2015-11-25-14-58-49-a` frame 5 (train split).
**A 10 m model range would drop 7.3 % of DROW's labels**, so the model's range limit is a deployment parameter, not a FROG constant.
How DROW's annotators handled far or missing readings is not stated beyond the paper's per-1 m histograms, which run to 15 m.

## JRDB

Not on disk (manual registration, `gotchas.md`). Nothing is known about its noise yet.
