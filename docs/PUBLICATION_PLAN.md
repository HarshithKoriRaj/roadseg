# From ablation to publishable paper — plan (2026-10-14)

Inputs read: `docs/report/roadseg_report.tex` (our DWT study, all 36 runs final),
`/scratch/hk4488/.cache/segroad.pdf` (Tao et al., Seg-Road, Remote Sens. 2023, 15, 1602),
`/scratch/hk4488/logs/roadseg/stats_{massachusetts,deepglobe}_final.txt`, the SDD ledger.

## 1. What changed between the draft and what we have now

| | Original draft (DWT + modified ResNeXt) | What was actually built and run |
|---|---|---|
| Claim | DWT sub-bands expose edges → better thin/occluded roads | Tested as R1 − R0, pre-specified, 3 seeds |
| Input | Resized 1500²/1024² → 512×1024 | Never resized; 512² crops, stitched full-image eval |
| Resolution bug | DWT halves input, never restored | Stride-1 stem so output stride matches R0 |
| Init | "zero" adaptation claimed equivalent — it is not | Exact function-preserving init (R1 ≡ R0 at step 0) + Proposition: DWT→conv = reparameterised strided conv |
| Decoder | Draft figure, implicit skips | U-Net, explicit skips, identical across rungs |
| Baselines | None controlled | U-Net (scratch), D-LinkNet, + R2 (wavelet pool everywhere), R3 (+clDice) |
| Result | "improves" | **Null.** Mass +0.0004 [−0.0040, +0.0060]; DG +0.0017 [−0.0018, +0.0059] |

As it stands this is a well-executed negative result on one tweak. That alone is publishable only
in venues that accept negative/rigorous work (TMLR, NeurIPS ICBINB workshop, CVPR EarthVision).
To reach IEEE GRSL / Remote Sensing / JSTARS it needs one more substantive finding (Section 4).

## 2. What Seg-Road does, and where it is weak (useful as motivation, cite neutrally)

Method: SegFormer-style spatial-reduction transformer encoder + light CNN decoder + PCS head
(8-channel neighbour-connectivity labels at r = 2, BCE, α = 0.2; inference = union of the 8
inverse-mapped maps). Reports road IoU 0.6720 (DG), 0.6838 (Mass).

Protocol problems a reviewer would accept as motivation for a controlled study:
1. **No validation split on DeepGlobe** (4696 train / 1530 test). Checkpoint/threshold selection
   is either on test or unreported.
2. **Single run, no variance, no CIs.** PCS ablation is one run of the smallest model
   (+1.1 pt DG, +1.9 pt Mass), never on Seg-Road-l.
3. **Re-run baselines far below their own papers:** D-LinkNet 0.5762 vs 0.6466 (original) vs
   0.6655 (ours). Gains over weak baselines are inflated.
4. **Metrics inconsistent:** road IoU 0.672 implies road F1 0.804; they report F1 0.914
   (P/R/F1 are evidently not road-class pixel metrics).
5. Pretrained on PASCAL VOC (non-standard); no connectivity metric (APLS) despite a
   connectivity claim; cross-dataset claim (Fig. 13) is qualitative only.

**Key fact from our data:** a stock ImageNet ResNeXt-50 U-Net (R0) gets 0.677 DG pooled road IoU —
level with Seg-Road-l (0.672), URoadNet (0.678), close to DSFC-Net (0.687). Same architecture
(D-LinkNet) spans 0.576 → 0.666 across protocols (9 pt), while our seed sd is 0.1–0.3 pt and
our minimum detectable effect is ~0.5–1.0 pt. **Protocol effects are an order of magnitude larger
than the architectural gains the literature claims.**

## 3. Recommended paper framing

**"Protocol, not architecture: a controlled re-evaluation of road extraction on Massachusetts and
DeepGlobe"** (working title). Three pillars:

1. **Wavelet front-ends under control** (done): Proposition + R1 null + R2/R3.
2. **Protocol sensitivity** (cheap, mostly from saved `test_probs/` + checkpoints): quantify how
   much one fixed model's IoU moves with each evaluation choice published papers make.
3. **Controlled replication of one claimed-SOTA idea** (PCS from Seg-Road) as a plug-in head on R0.

Plus one mechanistic experiment that explains *why* wavelet papers report gains (4b).

## 4. Experiments to add (ordered by value per GPU-hour)

Each must get a written, committed analysis plan **before** results are read (as for R1 − R0).
Re-reading test for a new, non-selecting analysis is allowed but disclosed.

| # | Experiment | New compute | What it buys |
|---|---|---|---|
| 4a | **Protocol sensitivity** on saved R0 probs: masked vs unmasked Mass; per-image vs pooled; val-selected vs 0.5 vs test-oracle threshold; road IoU vs mIoU; stitched vs 512² crops scored separately; Seg-Road's stride-512 crop (discards borders). Retrain-free. | ~0 (CPU) | Pillar 2; also gives Mass numbers comparable to published ones (unmasked) |
| 4b | **When does DWT help?** R0 vs R1 **from scratch** (no ImageNet), both datasets, 3 seeds. Hypothesis (pre-register): gain appears only without pretraining — explains literature gains vs weak baselines. Optionally the naive "LL-only" init as R1-naive. | 12 runs ≈ 60 GPU-h | Turns the null into an explanation; the strongest scientific addition |
| 4c | **PCS head on R0** (8-ch, r=2, α=0.2, inverse-mapped union at inference), 3 seeds × 2 datasets; report IoU **and APLS**. | 6 runs ≈ 30 GPU-h | Pillar 3: does Seg-Road's claimed +1–2 pt survive control? Either answer is publishable |
| 4d | **Cross-dataset transfer** DG→Mass and Mass→DG using existing 36 checkpoints (resample for 0.5 m vs 1 m GSD). Pre-specify R2 vs R0 as the test of the anti-aliasing hypothesis. | inference only, ~5 GPU-h | Independent test images for R2's inconclusive +0.4 pt; quantifies Seg-Road's qualitative claim |
| 4e | R2 confirmatory seeds 3–4 (R0, R2, DG) — only if 4d is suggestive | 4 runs ≈ 26 GPU-h | Resolves "inconclusive at 3 seeds" per the existing rule |
| 4f | Empirical support for the Proposition: project R0's pretrained/trained conv1 filters onto the Haar basis; spectral energy of road vs background patches | ~0 | An image-processing figure that makes the null intuitive |

Total new GPU ≈ 95–125 GPU-h (last campaign was ~180 GPU-h, ~28 h wall).

## 5. Must-fix before submission (from report Limitations)

- Unmasked Massachusetts numbers (4a) — reviewers will otherwise dismiss the Mass comparison.
- Verify DeepGlobe HF mirror pixel identity vs Kaggle release (needs Kaggle credentials), or at
  least checksum a sample.
- Near-duplicate/adjacent-tile leakage check across the DG random split (perceptual hash + border
  matching).
- Massachusetts selection uses 14 val images — report threshold sensitivity (4a covers it).
- APLS is our implementation (snap 8 px, zero-on-miss) — release it, state it is not Batra's.
- Do **not** reuse any sam2_road_dac APLS/clDice numbers (known inflated/confounded).

## 6. Venues

- **TMLR** — judges correctness and interest, not novelty; best fit for pillars 1–2 alone.
- **IEEE GRSL** (letter) or **Remote Sensing / JSTARS** — need 4b or 4c to carry a positive finding.
- **CVPR EarthVision / NeurIPS ICBINB workshops** — fastest route for the current null + 4a.
