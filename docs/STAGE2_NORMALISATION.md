# Stage 2's normalisation is not ConvNeXt's

TAKHeLiPeD's stage 2 is a 1-D U-Net built from ConvNeXt blocks, and in one component it departs from the block it cites.
ConvNeXt normalises each position on its own; every block of stage 2 normalises over the whole scan.
This page says what each does, how the difference got in, what it measures on both datasets, and what it means for anyone describing or reusing the architecture.
The frozen detector carries the deviation, and every published TAKHeLiPeD number on FROG was measured with it.

## The standard: ConvNeXt's LayerNorm, per position

A ConvNeXt block (Liu et al., *A ConvNet for the 2020s*, CVPR 2022) is a depthwise convolution, a LayerNorm, a 1x1 widening by four, GELU, a 1x1 narrowing, and a residual connection.
Its LayerNorm runs in the channels-last layout, so the mean and variance are taken **at each spatial position, over that position's channels alone**, followed by a learnt scale and shift per channel.
In one dimension, over beams, that means each beam is standardised against its own `C` features and nothing else.
A beam's normalised features depend on that beam's features, and so on the scan only as far as the convolutions reach.

## Ours: one mean and one variance per scan

Every stage-2 block normalises with `GroupNorm(1, C)` on a `[B, C, N]` tensor.
With one group, the mean and variance are taken **over all `C x N` values of the scan at once** -- every channel of every beam -- and then the same learnt per-channel scale and shift is applied.
This is not an invention of this project: it is the normalisation PoolFormer calls *Modified Layer Normalization*, which "computes the mean and variance along token and channel dimensions" where vanilla LayerNorm uses the channel dimension only, and which its authors implement with PyTorch's `GroupNorm` at a group number of 1 (Yu et al., *MetaFormer Is Actually What You Need for Vision*, CVPR 2022, supplementary material).
So the honest description of stage 2's block is **a ConvNeXt block with PoolFormer's Modified Layer Normalization in place of ConvNeXt's LayerNorm**.

Two consequences follow, and both were measured here rather than inferred:

- **The receptive field is the whole scan, whatever the kernels.**
  Measured by gradient on DROW, every output beam reads all 450 input beams, in the headline network and equally at kernel 3 with no bottleneck blocks and with one block per stage.
  Convolutions cannot do that at those sizes; the normalisation does, because every block divides every beam by a statistic of the whole scene.
  Under ConvNeXt's own normalisation the same headline network reads 256 beams.
- **The score level depends on the scene.**
  Every block rescales every beam by the scan's statistics, so the same person in an enclosed corridor and in an open hall reaches the head with different features.
  Centring the output logit cannot undo this, because the rescaling happens inside the network at every layer, not once at the end.

## How it got in

The record does not show it being chosen for this behaviour.
The project's earlier U-Net architectures used `GroupNorm` in place of `BatchNorm` for a stated reason -- `BatchNorm` degrades at batch size 1, which is how a robot runs a detector (Wu and He, *Group Normalization*, ECCV 2018).
When the ConvNeXt blocks were written, `GroupNorm(1, C)` stood in for LayerNorm, a common shortcut for channels-first tensors.
On 2026-10-19 the research write-up audited every borrowing in stage 2 against its paper and described this one as faithful: "`GroupNorm(1, C)` for LayerNorm, which is the same normalisation on a 1-D tensor".
That was wrong, and it stayed wrong until 2026-11-10, when a receptive-field measurement that convolutions could not explain found it while looking for the cause of stage 2's gap to the cutout detectors on DROW.

## What it measures

`--norm position` builds ConvNeXt's own normalisation (`ChannelNorm`: each beam standardised over its channels, its parameters named as `GroupNorm`'s so a checkpoint's keys fit either), and `--norm scan`, the default, keeps the scan-wide one.
Person AP at 0.5 m, official test splits; CPU timing on four threads, batch 1, against FROG's 25 ms scan period:

| | scan-wide (`GroupNorm(1, C)`, the frozen detector) | per position (ConvNeXt's), kernel 7 | per position, kernel 5 |
| --- | ---: | ---: | ---: |
| DROW, stage 2, two seeds | 29.03% | 37.95% | **41.79%** |
| FROG, stage 2, two seeds, same-day controls | **81.30%** | 79.90% | 79.80% |
| FROG, whole detector, three seeds | **86.29%** | -- | 84.51% |
| CPU, whole detector, median / p99 | **19.00 / 20.92 ms** | -- | 20.04 / 22.08 ms |

For reference, DROW3's cutout detector, trained on identical DROW frames under the same protocol, reaches 63.20%.
Kernel 5 is the per-position network's best field on DROW (the smaller fields tried stay within 0.7 pp of it) and costs FROG 0.1 pp more than kernel 7, which is why it is the variant carried end to end.

**The standard normalisation is better where the test scenes differ from the training ones, and the deviation is better where they do not.**
On DROW the per-position block gains 12.8 pp at stage 2.
On FROG it loses 1.5 pp at stage 2 and 1.78 pp through the whole detector, every seed below every seed of the frozen detector, for about 1 ms more per scan.
The two datasets differ in the way the second consequence above makes relevant: DROW's test recordings are enclosed (a median return of 1.74 m) where its training recordings are open (4.33 m), while FROG's test and training scenes match (3.72 against 3.61 m).
Geometry is not the whole story, though: rebalancing DROW's recordings by geometry did not by itself close the scan-wide network's train-test gap by the rule written before that run.

What the scan-wide statistics *give* FROG is not measured.
The reading consistent with the numbers is that, when test scenes resemble training scenes, a scene-level statistic is information the network can use; when they do not, it is a shift the network was never trained to absorb.
That is a hypothesis, not a finding.

## What this means

- **For describing the architecture.**
  Stage 2 should be called a ConvNeXt U-Net with Modified Layer Normalization, citing PoolFormer for it, and never "ConvNeXt blocks" without qualification.
  A reviewer who reimplements "ConvNeXt blocks" from the paper builds the per-position variant, which is a different detector: 1.78 pp lower on FROG and 12.8 pp higher at stage 2 on DROW.
- **For the frozen detector.**
  Nothing changes until the owner decides: the frozen architecture and every FROG number on file keep the scan-wide norm.
  Whether ConvNeXt's own normalisation becomes the default, or a drop-in enabled by a measured property of the data, is open, and its conditions are written up separately.
- **For reuse.**
  The flag switches every normalisation in stage 2 -- the ConvNeXt blocks, the residual block and the bottleneck's `MetricGather` alike.
  A new component normalised with `GroupNorm(1, C)` would bring the scan-wide reach back, whatever its kernels.

## Sources

- Liu, Mao, Wu, Feichtenhofer, Darrell, Xie, *A ConvNet for the 2020s*, CVPR 2022 -- the ConvNeXt block and its per-position LayerNorm.
- Yu, Luo, Zhou, Si, Zhou, Wang, Feng, Yan, *MetaFormer Is Actually What You Need for Vision*, CVPR 2022, and its supplementary material -- Modified Layer Normalization, implemented as `GroupNorm` with one group.
- Wu, He, *Group Normalization*, ECCV 2018 -- why the project's networks avoid `BatchNorm` at batch size 1.
- The measurements: `CHANGELOG.md`, the entries of 2026-11-10 (the receptive field and the first test of both norms), 2026-11-11 (kernel 5) and 2026-11-13 (FROG end to end); `memory/performance-log.md`.
