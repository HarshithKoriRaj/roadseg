# Decisions made while building this experiment

Every item below is a decision taken on the author's behalf during implementation, in the
order of the pipeline rather than the order it was made. Each says what was decided, why, and
what it costs if it was the wrong call — so it can be reversed. The binding rule behind most of
them: **rungs must differ only in the down/up-sampling operator** (and, for R3, the loss), or
the comparison is not interpretable.

The draft paper's text and figures need to change wherever a decision departs from the draft;
those are marked **[paper]**.

## A. The models

1. **R1's stem starts as exactly the pretrained R0 stem** **[paper]**. The draft's (and the first
   spec's) adaptation — pretrained kernel on the LL band, zeros on LH/HL/HH — is *not* equivalent
   to R0: it is the pretrained kernel applied to a 2×2-average-pooled image (stem feature cosine
   0.62 vs R0). An exact adaptation exists: the DWT of the top/left zero-padded 7×7 kernel, placed
   at taps 1:5 of the 12-channel stride-1 7×7 conv. With it R1 ≡ R0 at step 0 (verified to 1.5e-8
   on the real path) and can only diverge by learning. Default `hf_init="exact"`; the draft's
   variant remains available as `--hf-init zero`. *If wrong:* one flag re-runs the other variant.
2. **R1's stem convolution is stride 1.** The DWT already halves the input, so total stride stays
   32 and every feature map has the same shape as in R0. This is also the fix for the draft's
   resolution bug (the predicted mask came out at half resolution). *Documented confound:* R1's
   7×7 kernel can learn a 14×14-pixel footprint; R0's cannot.
3. **R2 keeps every pretrained weight and adds ~0.2% parameters** **[paper]**. The first design
   replaced each stride-2 operation by a DWT plus a dense 4C→C projection: +60% encoder parameters
   (36.8M vs 23.0M), 2.75M pretrained shortcut weights discarded, features at initialisation only
   0.40–0.54 cosine to R0. Replaced by `WaveletPool`: DWT, then a learned per-channel combination
   of that channel's four sub-bands (4·C parameters), placed *in front of* the kept pretrained
   convolution whose stride is set to 1. Shortcuts are exactly function-preserving; the other
   sites start as 2×2 average pooling (cosine to R0 at init: 0.91–0.96). Encoder parameters:
   R0 22,979,904 · R1 23,008,128 · R2 23,022,720. *If wrong:* a dense cross-channel variant would
   be an explicit extra rung with a parameter-matched control.
4. **The wavelet upsampler differs from bilinear only in the 2× kernel.** The first design had 4/9
   the parameters of the bilinear path (−6.1M over the decoder). Now: learned per-channel 2×2
   synthesis filter (initialised to nearest-neighbour) followed by the *same* 3×3 convolution.
5. **Initialisation is paired across rungs.** Wavelet parameters are initialised deterministically
   and draw nothing from the random stream; the decoder and head are built before the encoder. At
   a fixed seed every shared weight is bitwise identical across R0–R3.
6. **R3's loss is R2's loss plus 0.5·(1 − clDice) — nothing else** **[paper]**. The clDice function
   reused from `sam2_road_dac` returns 0.5·Dice + 0.5·clDice, so "adding clDice" also re-weighted
   Dice by 1.5×. Fixed. *If wrong:* the weight 0.5 is untuned.
7. **Cardinality is 32 at every stage.** The draft's figure labels it 32/64/128/256 — that is the
   width, not the cardinality **[paper]**.
8. **D-LinkNet's centre block is the published cascade** of dilated convolutions, not parallel
   branches (the first implementation was wrong). U-Net 31.0M, D-LinkNet 31.1M parameters.
9. Haar kernels are stored in float64 and cast to the input's dtype (float32 kernels cannot meet a
   1e-10 reconstruction test).

## B. The data

10. **Cityscapes was dropped** (decided with the author): binary road segmentation there is
    saturated near 97–98% IoU and cannot discriminate the rungs **[paper]**.
11. **Aerial imagery is never resized** **[paper]**. The draft resizes 1500×1500 and 1024×1024
    images to 512×1024, which distorts geometry and makes 2–4 px roads sub-pixel. Training uses
    512×512 crops; evaluation tiles at stride 256 and stitches back to full resolution.
12. **Tiles are edge-flush, not reflection-padded** (the last tile is aligned to the image edge),
    so the network never sees padded borders.
13. **Data sources.** No Kaggle token was available. Massachusetts comes from the original Toronto
    host; DeepGlobe from the Hugging Face mirror `lammtfkday/deepglobe-roads` (6226 labelled pairs,
    matching the original's counts, regions and GSD), flattened to the original layout. All 2342
    Massachusetts files were decoded end to end and are intact. *Disclose:* pixel identity of the
    mirror with the Kaggle release is unverified.
14. **DeepGlobe is re-split 70/15/15 by image, seed 1337** (4358/933/935) **[paper]**. Only the
    6226 training images were ever released with labels; the draft's "1243 validation and 1101
    test images with annotations" is not correct. The mirror's own partition is not used. The
    test split is pinned by a sha256 fingerprint (`a18e6da5…`).
15. **Massachusetts no-data is masked** **[paper]**. Blank white regions cover 11% of pixels on
    average (up to 79% of an image) and the labels have roads drawn through them. No-data = white
    pixels in a connected white region of at least 2500 px; Massachusetts only. It is excluded from
    the loss, from crop sampling (crops under 50% valid are rejected), from checkpoint/threshold
    selection, and from every metric. *Consequence:* Massachusetts scores are not comparable with
    published numbers that count those pixels.
16. **Augmentation** follows the spec (flips, 90° rotations, ±15° rotation, 0.9–1.1 scale, mild
    colour jitter), all from one seeded generator; image, mask and no-data mask are verified to
    stay aligned. **The crop stream is identical across rungs for a given seed.**
17. Datasets are preloaded into RAM (per-crop cost 460 ms → ~28 ms); `.deps` must not shadow the
    conda environment's numpy/scipy/PIL/networkx/yaml (a test enforces it).

## C. Training

18. Loss is computed in fp32 (the forward pass runs in bf16); a non-finite loss or gradient stops
    the run instead of poisoning the weights.
19. Schedule: per-iteration linear warmup over 2 epochs from 0.01·lr, then cosine to 0.01·lr
    (the first plan ignored the warmup the spec asked for). AdamW, lr 3e-4, weight decay 1e-4,
    gradient clip 1.0, batch 16, 100 epochs (Massachusetts) / 40 (DeepGlobe). **One recipe for
    every rung, untuned.**
20. Per-epoch validation computes pooled IoU only (stride 512); the full metric set is computed
    once at the end (stride 256).

## D. Evaluation and statistics

21. **A no-data-only image has no score (NaN), not a perfect one**; an all-background prediction on
    a road-free image scores 1.0. Vacuous precision/recall cases are documented.
22. **Relaxed F1 uses a Euclidean ρ = 3 px**, not a 7×7 square, and credits a match that lies just
    inside a no-data region.
23. **APLS** was ported from `sam2_road_dac` with three defects fixed: no snap-distance threshold
    (inflated scores), a quadratic nearest-node scan, and a mask-to-graph rule that fragmented
    intersections and joined parallel roads. Both graphs empty → 1.0; one empty → 0.0. 26 s →
    3.9 s per image.
24. **The test split is scored exactly once per run**, by a separate command that refuses to run
    again; the checkpoint and threshold come from validation only. Probability maps are saved, and
    IoU at a fixed 0.5 is scored in the same pass, so later questions never need a second pass.
25. **Headline interval: a two-level bootstrap** (seeds within arm, then images) on per-image IoU.
    The first version resampled images only, which answers "these three trained models" rather
    than "this training procedure". A seed-paired t interval (df = 2) is reported beside it. In
    simulation the two-level interval covered 0.94–0.98 in most regimes and 0.86 when seed variance
    dominated at n = 935. **Pre-specified rule:** a difference is claimed only if *both* intervals
    exclude zero; if they disagree the result is "inconclusive at 3 seeds" and seeds 3–4 are run
    for R0 and R1 first.
26. **Only R1 − R0 per-image IoU is the primary analysis.** Everything else — other rungs,
    baselines, other metrics, pooled IoU, IoU@0.5, exclusions of near-empty images — is secondary
    and labelled so in the output. The analysis plan was committed before any result existed.
27. Aggregation refuses mislabelled runs, mismatched image sets within a rung, and non-standard
    scoring protocols.

## E. Running it

28. Jobs run from a **frozen snapshot** of the code and refuse to resume under different code.
29. **Preemption/requeue:** checkpoints every 5 minutes and at every epoch (atomic, fsynced);
    a signal checkpoints and exits for requeue; a resumed run finishes the interrupted epoch;
    a run killed at any point and resumed ends bitwise identical to an uninterrupted one (tested).
    Only the time-limit warning triggers a self-requeue — a crash or a `scancel` never does —
    and a task stops after 20 restarts.
30. Each job runs a one-minute GPU self-test before training; a run directory is locked against a
    second process; an explicit `cuda` device never falls back to CPU.
31. Launch gate: `scripts/launch.sh` refuses a dirty tree and requires a passed CPU rehearsal of
    the exact commit. A GPU canary passed on an L40S (all six architectures at batch 16 × 512²
    under bf16, peak 5–17 GB).
32. Partitions: this account can use only `l40s_public` and `h200_public`; a run may therefore
    span GPU types across requeues.

## F. Process decisions (no effect on results)

Feature branch in place rather than a worktree (paths are absolute); task reviews overlapped
with the next task's implementation; the full suite runs on a compute node (the login node
kills it); the test runner caps OpenMP threads; one implementer's commit trailer names a
different model than the others (left as the truthful author).

---

# Limitations the paper must disclose

- R2 and R3 change the encoder *and* the decoder operator together.
- Only the exact stem initialisation is run; the draft's LL/2 variant is not.
- One untuned recipe; weight decay is applied to all parameters including BatchNorm.
- Checkpoint and threshold are selected per run on validation — 14 images for Massachusetts.
- Massachusetts scores are masked by a custom no-data rule and are not comparable with
  published numbers.
- The DeepGlobe split is random by image; adjacent tiles may fall in different splits.
- Three seeds; the stated coverage of the headline interval.
- bf16 autocast, non-deterministic GPU kernels, mixed GPU types.
- Flush tiling, uniform overlap averaging, no test-time augmentation.
- U-Net is trained from scratch; D-LinkNet and the main model are ImageNet-pretrained.
- The DeepGlobe files come from a mirror whose pixel identity with the original is unverified.
- The soft skeleton uses a 3×3 square min-pool rather than the reference implementation's cross;
  the clDice term is 1 (with zero gradient) on road-free crops.
- Rehearsals and the canary scored a handful of test images with miniature models; no decision
  used those scores. Any relaunch after seeing aggregates would be a second look at the test set.
