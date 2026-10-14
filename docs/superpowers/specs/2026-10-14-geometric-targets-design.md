# Paper 2 — Centre-line + width targets for road segmentation (design)

Date: 2026-10-14. Branch `feat/geo-targets` (from `feat/dwt-resnext-ablation` @ c0a268a).
Approved by the user in conversation (direction, headline metric, approach B with A as a rung,
section 1 of the design); the user then asked to finish the design and submit without further
review gates. Defaults chosen by the controller under that instruction are marked **[default]**.

## 1. Question and motivation

Road labels in both benchmarks are reliable about *where a road runs and how it connects*, and
unreliable about *where its edges are*. Measured on 15 Massachusetts / 30 DeepGlobe training masks
(skeleton width = 2·EDT − 1):

| | width p5 / p50 / p95 | per-image median width |
|---|---|---|
| Massachusetts | 5.0 / 5.3 / 7.0 px | 5.3 – 6.2 (fixed-width rasterised centre-lines) |
| DeepGlobe | 5.0 / 9.8 / 22.3 px | 4.7 – 37.1 (inconsistent between images) |

Paper 2 replaces "predict a free-form mask" by "predict a centre-line and a width, and draw the
mask from them" — a shape prior (parallel edges, consistent width, no isolated blobs) built into
the output representation. The headline metric is **standard road IoU** (user's choice), scored
exactly as in paper 1.

## 2. Representation (approved section 1)

Targets, from each training crop's FINAL (augmented) mask, by image processing only
(`src/roadseg/data/geometry.py`):

1. **Disc depth:** per mask pixel, the largest radius in `RADII = (0,1,2,3,4,5,6,8,10,13,16,20)`
   whose disc -- the octagon `render` draws, Minkowski sum of its 3x3 steps -- fits inside the
   mask (successive erosions by the same steps). Taken on the mask padded by edge replication
   (21 px) so a road leaving the crop is not treated as ending there.
2. **Centre pixels:** `skeletonize(mask)` (same padding), plus every mask pixel the skeleton's
   discs leave uncovered (cover step done after the crop).
3. Width class = disc depth at centre pixels; `IGNORE` (−100) elsewhere and on no-data pixels;
   centre target masked by validity in the loss.
4. Every disc lies inside the mask and every mask pixel is covered: the one-hot targets draw the
   label back **exactly** (test-enforced on synthetic masks; measured on real masks, §7).
5. Targets draw no randomness: for a seed, the crop stream equals R0's (test-enforced).

How this was reached (pre-launch checks on training masks only):
- v1 skeleton + width = nearest bin to `EDT − offset`: Massachusetts ceiling 0.806 / 0.910 / 0.881
  at offset 0.5 / 1.0 / 1.5; DeepGlobe 0.860 / 0.920 (offset 0.5 / 1.0). Below the 0.95 gate.
- v2 adding equal-depth neighbours of the skeleton (even widths): 0.910 -> 0.910, no help.
- Diagnosis on 4 Massachusetts masks: every error is a 1-px edge pixel on slanted roads (the
  labels' staircase edges vs a disc around a staircase skeleton); FP > FN.
- v3 (adopted) the exact cover above. Without the cover step, the skeleton alone reaches 0.882
  (Massachusetts) / 0.918 (DeepGlobe) with disc-depth radii.

Drawing (`models/geo.py: render`): `contrib = sigmoid(c) · softmax(w)` ([B,K,H,W]); a countdown
from max(RADII) to 0 inserts bin k when the count reaches its radius and dilates the running max
by one 3×3 step per count (square on odd counts, cross on even: octagonal disc). Each bin is
dilated by exactly its radius; ~20 max-pools, fp32, differentiable.

**Representation ceiling** (`scripts/geo_ceiling.py`, 60 training images per dataset, never
val/test): draw each mask back from its own one-hot targets and score IoU. Gate: ≥ 0.95 pooled on
both datasets before GPU time. Results are recorded in §7.

## 3. Rungs [default: names GA, GB]

All rungs share R0's ImageNet ResNeXt-50 U-Net encoder/decoder, recipe, seeds 0–2 and crop
stream. The control is the **first campaign's R0 runs** (commit c1d1e2f); the R0 code path is
unchanged on this branch (R0's resolved config gains no keys; ComboLoss and `model(x)` path
untouched; test-enforced), so they are reused rather than retrained.

| Rung | Heads | Mask used for loss/inference | Isolates |
|---|---|---|---|
| R0 | mask | mask head | control (paper 1) |
| GA | mask + centre (1) + width (K=12) | mask head; geometric heads auxiliary | geometric supervision (GA − R0) |
| GB | same as GA | drawn by `render` from centre + width; mask head kept but unused | representation (GB − GA) |

Initialisation: decoder, mask head and encoder are built in R0's order → identical to R0 at the
same seed; GA and GB are identical to each other (test-enforced). GB returns `logit(clamp(m,
1e-4, 1 − 1e-4))`, so thresholding, tiled inference and scoring are R0's code unchanged.

## 4. Loss [default weights, untuned, disclosed]

`GeoLoss = (BCE + Dice)(mask) + 1.0 · (BCE + Dice)(centre logits, centre target)
+ 1.0 · CE(width logits, class) over centre pixels`, all masked by validity, fp32.
For GB the mask term is BCE + Dice of the drawn probability.

## 5. Protocol (identical to paper 1)

Massachusetts 100 epochs / DeepGlobe 40; batch 16 × 512²; AdamW 3e-4, wd 1e-4, warm-up 2 epochs +
cosine; bf16; checkpoint = best validation IoU@0.5; threshold selected on validation; test split
scored once, with probability maps saved. 12 new runs (GA, GB × 2 datasets × 3 seeds), launched
via `bash scripts/launch.sh --campaign geo` from a frozen snapshot after the CPU rehearsal passes
(rehearsal grid now includes GA and GB). Estimated ~60–70 GPU-hours on L40S.

## 6. Pre-specified analysis (`stats.GEO_ANALYSIS_PLAN`, `--plan geo`)

- **Primary:** GB − R0, per-image test IoU at the validation-selected threshold, two-level
  bootstrap (seeds within arm, then images), per dataset; decision rule identical to paper 1:
  "difference" only if both the two-level interval and the seed-paired t interval exclude zero
  with the same sign; exactly one → inconclusive, run seeds 3–4 for R0 and GB before any claim.
- **Secondary (no multiplicity correction):** GA − R0; GB − GA (`--baseline GA`); relaxed F1,
  clDice, APLS; pooled IoU; IoU@0.5; n_gt floors.
- Committed before any GA/GB run is trained.

## 7. Results of the pre-launch checks

`scripts/geo_ceiling.py`, 60 training images per dataset (job 19821301, 2026-10-14):

| | drawn-back IoU, full images (worst) | 512 crops | skeleton alone | centre px / mask px | skeleton share | targets per 512 crop |
|---|---|---|---|---|---|---|
| Massachusetts | 1.000000 (1.000000) | 1.000000 | 0.8817 | 0.257 | ~0.54 | 49 ms |
| DeepGlobe | 1.000000 (1.000000) | 1.000000 | 0.9177 | 0.159 | ~0.48 | 44 ms |

Gate (≥ 0.95) passed. Note: about half of the centre pixels are 1-px staircase edge pixels with
radius 0–1, not centre-line; the "centre" target is a minimal disc cover anchored on the
skeleton. Because radius 0 can draw any pixel, the representation can express any mask: the shape
prior acts through the targets and the drawing operator, not as a hard constraint.

## 8. Known limitations (to disclose)

- Loss weights (1.0, 1.0) and RADII untuned.
- The target design changed twice after looking at TRAINING masks (never val/test); recorded in §2.
- The radius-0 bin makes the prior soft (see §7).
- GB keeps an unused mask head (same parameter set as GA, no effect on its output).
- Control R0 runs come from an earlier code commit (same R0 path; provenance recorded per run).
- The validation error-breakdown diagnostic (edge vs whole-segment errors of R0) was proposed as
  a gate; under the user's "finish and submit" it runs as a reported analysis, not a gate.
