# DWT–ResNeXt Road Segmentation — Design

**Date:** 2026-09-25
**Status:** approved design, pre-implementation
**Project root:** `/scratch/hk4488/roadseg`

## 1. Goal

Implement and honestly evaluate the "DWT + Modified ResNeXt" road segmentation
method described in the draft paper, on the Massachusetts Roads and DeepGlobe
Road Extraction datasets.

The draft's claim is that decomposing the input with a single-level 2D discrete
wavelet transform, and feeding the resulting 12 channels into a ResNeXt encoder,
improves road extraction. **That claim has never been tested.** The primary
deliverable of this project is a controlled measurement of its effect size, not
a demonstration that it works.

Cityscapes is out of scope. Binary road segmentation on Cityscapes is saturated
(~97–98% IoU for any competent model), so it cannot discriminate between the
variants under test and would consume compute to produce a table of
indistinguishable numbers.

## 2. Success criteria

1. Every rung in §4 trains to convergence on both datasets, 3 seeds each.
2. The R1 − R0 difference is reported with a paired bootstrap confidence
   interval, and that interval is believed regardless of which side of zero it
   falls on.
3. Test sets are scored exactly once, after checkpoint and threshold selection
   is complete on validation data.
4. Results are reproducible from a config file and a recorded seed.

A null result for R1 is an acceptable and publishable outcome of this project.
The design exists to make a null result *informative* rather than ambiguous.

## 3. Non-goals

- Cityscapes, or any street-view / ego-perspective data.
- Road *graph* extraction (vectorized output). We predict rasters. APLS is
  computed by skeletonizing the predicted raster, which is a proxy, not a
  graph-extraction method.
- Beating the state of the art. The target is a trustworthy ablation.
- Any architecture search, hyperparameter tuning per rung, or per-dataset
  recipe divergence. All rungs share one recipe; that is what makes the
  comparison valid.

## 4. The ablation ladder

One model class, one training recipe, one set of seeds. Rungs differ only in
the downsampling and upsampling operators.

| Rung | Encoder downsampling | Decoder upsampling | Loss | Role |
|------|----------------------|--------------------|------|------|
| R0 | stock ResNeXt-50 32×4d (strided conv + maxpool) | bilinear + conv | BCE + Dice | control |
| R1 | Haar DWT at input only; conv1 12→64, **stride 1** | bilinear + conv | BCE + Dice | **the paper** |
| R2 | Haar DWT replaces *every* stride-2 downsample | IDWT | BCE + Dice | strong variant |
| R3 | = R2 | = R2 | BCE + Dice + soft-clDice | connectivity variant |

Reported quantity of interest: **R1 − R0**. R2 and R3 exist so that a null
result at R1 does not end the project.

Baselines for the paper's comparison table (not part of the ablation, trained
with the same recipe): **U-Net** and **D-LinkNet** (the DeepGlobe challenge
winner). These were proposed in the design conversation and are included;
they can be cut if compute becomes the binding constraint, since they are
independent of the R0–R3 ladder.

## 5. Architecture

### 5.1 Stride parity — the core design constraint

The draft has an unresolved resolution bug: DWT halves the input, nothing ever
restores that factor, and the predicted mask ends up at half the ground-truth
resolution. The fix is not to bolt an extra decoder block on the end. It is to
make the encoder's total stride **identical across all rungs**, by giving R1's
stem a stride-1 convolution:

```
R0 stem:  conv7×7 s2  (512 → 256, 3ch → 64)   →  maxpool s2  (→ 128)
R1 stem:  DWT (512 → 256, 3ch → 12ch)  →  conv7×7 s1 (256, 12ch → 64)  →  maxpool s2 (→ 128)
```

Both stems emit a 64-channel feature map at 256×256. Every subsequent stage,
every skip connection, and the entire decoder are then bit-for-bit identical in
shape between R0 and R1. The two rungs differ in exactly one thing: how the
512→256 reduction is performed. That is what makes R1 − R0 attributable.

An assertion `out.shape[-2:] == inp.shape[-2:]` is enforced in the model's
forward pass for every rung, so this bug class cannot silently return.

**Documented confound:** R1's 7×7 stem kernel operates on a half-resolution LL
band, so it sees a 14×14 footprint in original-image pixels, versus 7×7 for R0.
The receptive fields are not identical. This is intrinsic to the paper's design
rather than an artifact of our implementation, and it is stated in the results
rather than hidden.

### 5.2 Encoder

`torchvision.models.resnext50_32x4d`, ImageNet weights. Stage outputs for a
512×512 input:

| Stage | Resolution | Channels |
|-------|-----------|----------|
| stem (conv1) | 256 | 64 |
| layer1 | 128 | 256 |
| layer2 | 64 | 512 |
| layer3 | 32 | 1024 |
| layer4 (bottleneck) | 16 | 2048 |

Cardinality is **32 at every stage** — it is a property of the grouped
convolution, not a per-stage channel count. (The draft's figure labels it
C=32/64/128/256 across stages, conflating cardinality with width. The figure
needs correcting in the paper.)

### 5.3 Wavelet transform

Implemented as a fixed-weight depthwise `conv2d` (stride 2, four 2×2 kernels per
input channel) and its exact inverse as `conv_transpose2d` with the same
kernels, `requires_grad=False`. Orthonormal Haar: low = [1,1]/√2,
high = [1,−1]/√2; the four 2D kernels are their outer products, entries ±½.

This runs on GPU, is batched, is differentiable end-to-end, and is exactly
invertible — none of which a `pywt` call in the dataloader would be, and R2
requires wavelet ops *inside* the network where a CPU library cannot go. The
same kernel-construction code accepts db2 / bior2.2 coefficients if a
wavelet-family ablation is wanted later.

Channel ordering is fixed and documented: `[R-LL, R-LH, R-HL, R-HH, G-LL, …, B-HH]`.

### 5.4 Pretrained weight adaptation (R1, R2)

Changing conv1 from 3 to 12 input channels discards its ImageNet weights. Left
unhandled, R1 would lose several IoU to R0 for reasons having nothing to do with
wavelets, and the comparison would be worthless.

> **Amended 2026-10-11** after the Task 4 review. The original recipe below
> ("zero") was claimed to make R1 equivalent to R0 at initialisation. It does
> not, and an exact adaptation exists. The exact one is now the default.

**Default — exact adaptation (`hf_init="exact"`).** R0's stem computes
`y[i,j] = Σ W[c,u,v] · x[c, 2i+u−3, 2j+v−3]`. Zero-padding the 7×7 kernel by one
row on top and one column on the left makes its 8×8 support coincide with four
whole 2×2 blocks per axis. Because Haar is orthonormal, the inner product of a
kernel block with a pixel block equals the inner product of their transforms, so
the equivalent kernel in the wavelet domain is simply **the DWT of the padded
kernel**:

```
K  = pad(W, top=1, left=1)          # [64, 3, 8, 8]
G  = DWT(K)                         # [64, 12, 4, 4]
W' = zeros(64, 12, 7, 7);  W'[:, :, 1:5, 1:5] = G
```

With this initialisation **R1 is the same function as pretrained R0 at step 0**,
borders included (verified numerically to ~1e-6). R1 can then depart from R0
only by learning, within the larger function class its stem affords: a 7×7
stride-1 kernel over 12 sub-band channels has a 14×14 pixel footprint and
37,632 weights against R0's 9,408. That extra stem capacity *is* the paper's
proposal, stated plainly: a DWT followed by a convolution is a reparameterised,
larger strided convolution. R1 − R0 measures whether that helps.

**Secondary — `hf_init="zero"`.** `W'[:, 4c] = W[:, c] / 2`, high-frequency
bands zero. This is the pretrained kernel applied at stride 1 to a 2×2
average-pooled image: every filter is retuned to half its spatial frequency and
intra-block detail is invisible until the high-frequency weights train. It is
*not* equivalent to R0 (stem feature cosine ≈ 0.62 at initialisation), so a
difference measured under it mixes the operator with a pretrained-transfer
mismatch. Kept as a flag because it is the adaptation a reader of the draft
would most likely assume. `hf_init="small-random"` is likewise available.

### 5.5 Decoder

U-Net style, five stages, **explicit skip connections** — not the draft's
"skip connections can be used" hedge. Decoder widths (512, 256, 128, 64, 32):

| Block | Upsample to | Skip source | Out ch |
|-------|-------------|-------------|--------|
| d1 | 32 | layer3 (1024) | 512 |
| d2 | 64 | layer2 (512) | 256 |
| d3 | 128 | layer1 (256) | 128 |
| d4 | 256 | stem (64) | 64 |
| d5 | 512 | — | 32 |

Head: 1×1 conv → 1 logit channel. Sigmoid applied in the loss/metrics, not the
model (numerical stability with `BCEWithLogitsLoss`).

Upsampling operator is swappable, and the two options differ **only in the 2×
kernel** (amended 2026-10-11):

- `bilinear` (R0, R1): fixed bilinear interpolation, then a 3×3 conv.
- `wavelet` (R2, R3): `WaveletUnpool` — a per-channel learned 1→4 expansion
  followed by the inverse DWT, i.e. a learned 2×2 synthesis filter, initialised
  to nearest-neighbour — then **the same** 3×3 conv.

The originally specified wavelet upsampler (a 1×1 conv to 4C channels, then
IDWT) had 4/9 the parameters of the bilinear path's conv, 6.1M fewer across the
decoder, and a different operation order. It would have confounded the operator
with capacity in the opposite direction to the encoder. The amended version
adds 4·C parameters per stage and nothing else.

### 5.6 R2 — wavelet downsampling throughout

> **Amended 2026-10-11** after the Task 4 review. The original design replaced
> each stride-2 operation with a DWT followed by a *dense* 1×1 projection
> (4C → C). Measured consequences: R2's encoder had 36.8M parameters against
> R0's 23.0M (+60%); 2.75M pretrained shortcut weights were discarded; and the
> new projections were randomly initialised inside a pretrained network, so
> R2's features at initialisation had cosine similarity 0.40–0.54 to R0's
> (R1: 0.90–0.96). R2 − R0 would have measured the operator, plus capacity,
> plus lost pretraining.

Every stride-2 operation in the encoder is replaced by a `WaveletPool` placed
**in front of the original pretrained layer**, whose stride is set to 1:

`WaveletPool(C)` = DWT, then a depthwise 4→1 learned combination of each
channel's own four sub-bands. It is a learned per-channel 2×2 analysis filter
expressed in the Haar basis; it adds 4·C parameters and mixes no channels.

| Replaced operation | Becomes | Initialised as | At init |
|---|---|---|---|
| stem maxpool 3×3 s2 | `WaveletPool(64)` | LL only (2×2 average) | approximate |
| `layer{2,3,4}.0` grouped 3×3, s2 | `WaveletPool(C)` → same conv, s1 | LL only (2×2 average) | approximate |
| `layer{2,3,4}.0` shortcut 1×1, s2 | `WaveletPool(C)` → same conv, s1 | (½,½,½,½) = top-left pixel | **exact** |

All pretrained weights are kept, cardinality stays 32, and the encoder's
parameter count differs from R0's by roughly 0.2%. All four sub-bands remain
available to every downsampling step through learned weights, which is the
honest form of the wavelet argument: strided convolution and max-pooling fix
which high-frequency content is discarded; a learned combination of Haar
sub-bands lets the network choose.

What this version does **not** test is cross-channel mixing of sub-bands at the
downsampling step. That is a legitimate, stronger model, but it cannot be
separated from added capacity; if wanted it should be an explicit extra rung
with a parameter-matched control, not a silent property of R2.

**Seed pairing.** `WaveletPool` and `WaveletUnpool` are initialised
deterministically and draw nothing from the random stream, and the decoder is
constructed before the encoder. At a fixed seed every weight the rungs share is
therefore identical across rungs, and R1's outputs equal R0's at step 0. Both
properties are asserted by tests.

## 6. Data pipeline

### 6.1 No resizing of aerial imagery

The draft resizes 1500×1500 and 1024×1024 overhead imagery to 512×1024. This
both distorts the aspect ratio and reduces 2–4 px wide roads to sub-pixel
width — it deletes the structures the method is supposed to preserve. We crop
and tile instead, never resize.

### 6.2 Massachusetts Roads

- Official split: 1108 train / 14 val / 49 test. 1500×1500.
- **Nodata masking.** Many images carry white no-data borders. These are
  detected (pure-white regions) and excluded from both the training sampler and
  the metric denominators. Left unhandled they silently inflate every score —
  a well-known trap with this dataset.
- Train: random 512×512 crops, sampling biased so a configured fraction of
  crops contain road pixels (road prevalence is ~4–5%; uniform sampling yields
  mostly-empty crops).
- Val/test: deterministic sliding window, stride 256, reflection-padded at the
  borders, predictions averaged over overlaps and **stitched back to full
  1500×1500 before any metric is computed.**

### 6.3 DeepGlobe

- Only the 6226 training images have public masks; the challenge never released
  val/test labels. We re-split those 6226 into 70/15/15 **by image**, with a
  fixed recorded seed, and the paper states this plainly.
- Layout `<id>_sat.jpg` / `<id>_mask.png`, confirmed against the existing
  loader in `sam2_road_dac`.
- Same crop/tile protocol as Massachusetts. 1024×1024 source.

### 6.4 Implementation notes

Tiling is computed on the fly from full images via a precomputed index — no
duplicated preprocessed copies on scratch, and the tiling scheme can change
without re-preprocessing. Crop RNG is derived from a per-worker torch generator,
not `np.random`, to avoid the classic duplicated-crops-across-DataLoader-workers
bug (present in the existing `sam2_road_dac` loader).

Augmentation, identical for all rungs: horizontal/vertical flip, rot90, ±15°
rotation, scale jitter 0.9–1.1, mild colour jitter. Normalization with ImageNet
statistics, applied **before** the DWT so that R0 and R1 see identically
normalized inputs.

## 7. Loss and metrics

**Loss.** `BCEWithLogits + soft Dice`, equally weighted, for R0–R2. R3 adds
soft-clDice. Class imbalance is handled by the Dice term plus road-biased crop
sampling rather than by per-pixel weighting.

**Metrics**, all computed on stitched full-resolution images, never on tiles:

- IoU, F1/Dice, Precision, Recall
- Relaxed F1 at ρ = 3 px (the DeepGlobe convention)
- clDice (hard, skeleton-based)
- APLS on graphs skeletonized from the predicted and ground-truth rasters

Both dataset-aggregate IoU (Σintersection / Σunion) and mean-per-image IoU are
reported, because the literature is inconsistent about which it means and
reviewers check.

## 8. Statistical protocol

- 3 seeds per rung per dataset; report mean ± std. The full grid is
  **4 rungs × 2 datasets × 3 seeds = 24 runs**, plus **2 baselines × 2 datasets
  × 3 seeds = 12 runs**, for 36 total. The 24-run ladder is the priority; the
  12 baseline runs are the cuttable tail if compute binds.
- Paired bootstrap confidence intervals on per-image IoU for each rung-vs-R0
  difference (images are the resampling unit).
- Checkpoint selection and decision-threshold selection on **validation only**.
- The test set is evaluated once, at the end, after all selection is frozen.
- Every run records its config, git commit, and seed alongside its metrics.

## 9. Reuse from `sam2_road_dac`

`/scratch/hk4488/sam2_road_dac` is an existing road-extraction project of the
author's. Reusable components, with assessed condition:

| Source | Action |
|--------|--------|
| `src/losses/cl_dice.py` | **Copy as-is.** Correct Shit et al. soft-clDice with a sound `[0,1]` input guard. |
| `src/metrics/cl_dice.py` | **Copy as-is.** Correct hard clDice. |
| `src/metrics/apls.py` | **Copy and fix** — two defects, see below. |
| `scripts/sbatch_train.sh` | **Adapt as template.** Source of the env, account, and auto-resume pattern. |
| `src/data/deepglobe.py` | **Keep the file-layout knowledge only**; rewrite. It builds graph targets we do not need, uses `np.random` for crops, and has no road-presence bias. |
| `src/utils/checkpoint.py` | **Do not reuse.** It saves only `requires_grad` parameters, which silently drops BatchNorm running statistics (buffers, not parameters). Correct for its LoRA setting, broken for a fully-trained model. |

Components are copied rather than imported across projects, so this repo stays
self-contained for release alongside the paper.

### APLS defects to fix

1. `_nearest_node` is a brute-force O(N) scan per query, making the metric
   O(N²). Skeletonized 1500×1500 masks produce thousands of nodes. Replace with
   a `scipy.spatial.cKDTree`.
2. **No `max_snap_dist` threshold.** It snaps to the nearest node at *any*
   distance, so a prediction whose nearest node is far away still receives
   credit. The standard metric scores an unmatched node as 0. As written it
   systematically inflates APLS. Add the threshold and the zero-on-miss rule.

## 10. Environment and cluster

- Python env: `/scratch/hk4488/.conda/envs/unsloth_sft` — Python 3.11.15,
  torch 2.10.0+cu128, torchvision 0.25.0+cu128, numpy 2.3.5, scipy 1.17.1,
  networkx 3.6.1, PIL, pyyaml.
- Missing packages installed into a project-local `.deps/` on `PYTHONPATH`,
  following the `sam2_road_dac` pattern (there is no conda on the login node):
  `scikit-image`, `opencv-python-headless`, `albumentations`, `pytest`.
- **Not needed:** `pywt` (wavelet kernels are hardcoded conv weights) and
  `timm` (torchvision ships pretrained `resnext50_32x4d`).
- SLURM: `h200_public` partition, account `torch_pr_627_general`, one GPU per
  run, a SLURM **array job** over (rung × dataset × seed) so the grid runs
  concurrently. `--requeue`, `--open-mode=append`, and auto-resume from
  `latest.pt` for preemption survival. Logs to `/scratch/hk4488/logs/roadseg/`.
- Training: AMP bf16, AdamW, cosine schedule with warmup, batch 16 at 512²,
  gradient clipping.
- The login node is heavily loaded and OOM-prone; `import torch` there already
  timed out once. All non-trivial verification runs as a SLURM job.

## 11. Repository layout

```
/scratch/hk4488/roadseg/
  configs/            one YAML per rung × dataset
  src/roadseg/
    data/     prepare.py  dataset.py  tiling.py  nodata.py
    models/   wavelet.py  downsample.py  encoder.py  decoder.py  segmenter.py
              baselines/  unet.py  dlinknet.py
    losses.py  metrics.py  engine.py  train.py  eval.py  stats.py
  scripts/    sbatch_train_array.sh  sbatch_eval.sh  setup_deps.sh
  tests/
  docs/superpowers/specs/
```

## 12. Testing strategy

Test-first, with tests that can actually fail:

- `IDWT(DWT(x)) == x` to within 1e-5 — perfect reconstruction.
- Output shape equals input shape, for every rung.
- **Encoder stage shapes are identical across R0 and R1** — the stride-parity
  guarantee of §5.1, asserted directly.
- Weight adaptation: at initialization, R1's stem response on a smooth input
  matches R0's to within tolerance (validates the /2 LL rescaling).
- `stitch(tile(x)) == x` for the sliding-window tiler, including border padding.
- Metrics against hand-computed small cases; APLS on a pair of synthetic graphs
  with a known answer, including the unmatched-node-scores-zero rule.
- Nodata masking excludes the intended pixels from metric denominators.
- Overfit-one-batch integration test: loss → ~0 on a single batch.

## 13. Risks

| Risk | Mitigation |
|------|-----------|
| R1 − R0 ≈ 0, i.e. the paper's contribution is null | R2/R3 are built in the same codebase and run in the same grid; the finding is reportable either way. This is the expected-value reason for the ladder. |
| APLS from skeletonized rasters is noisy | Report it alongside relaxed-F1 and clDice rather than as the sole connectivity number. |
| Preemption on shared H200 partitions | `--requeue` + frequent checkpointing + auto-resume, built in from the first commit rather than retrofitted. |
| Login-node instability blocks iteration | All verification via SLURM; `/tmp` is full cluster-wide, so tool output is redirected to scratch. |

## 14. Open items

- Kaggle API token is not configured (`~/.kaggle/kaggle.json` absent). Required
  to download DeepGlobe. Massachusetts can come from the Toronto source.
  Neither dataset is currently present on `/scratch`.
