# Ledger — paper 2, geometric targets (GA, GB)

Spec: `docs/superpowers/specs/2026-10-14-geometric-targets-design.md`.

- 2026-10-14: user chose label geometry → standard IoU headline → ladder R0 → GA → GB; approved
  design section 1, then "finish and submit" (no further review gates).
- Representation checks on training masks (jobs 19819122, 19819792, 19821301): v1 skeleton +
  EDT−offset 0.910 / 0.920 ceiling; v2 even-width ridge no gain; diagnosis = 1-px staircase edges;
  v3 exact disc cover adopted → 1.000000 on both datasets. Targets 44–49 ms per 512 crop.
- Commits: ffe5351 spec + plan (before any GA/GB run); 0841265 implementation; 862387b launcher fix.
- Full suite at 0841265 (19833392): 487 passed, 3 failed — all launcher tests: the dwt submission
  had gained `ROADSEG_CAMPAIGN=dwt` / `campaign=dwt`. Fixed in 862387b (dwt launch byte-identical to
  paper 1). Full suite at 862387b (19837881): 492 passed.
- CPU rehearsal of 0841265 from snapshot (19833393): SMOKE: ALL PASSED (grid incl. GA/GB, geo-plan
  aggregation, preemption drills, BF16-CPU PASS).
- Ruling: launch 862387b with `--no-smoke-check`. `git diff 0841265 862387b` touches only
  scripts/launch.sh and tests; src/, configs/, the array and aggregation scripts are identical,
  and jobs never execute launch.sh. Cost if wrong: one launcher-side failure, visible at submit.
- **LAUNCHED 2026-10-14 ~17:50: array 19841566 (12 tasks, campaign geo), aggregate 19841568,
  snapshot releases/862387b.** Control = paper-1 R0 runs (6 scored, commit c1d1e2f).
- No GPU canary was run for GA/GB; each task's CUDA preflight (2 real optimizer steps with the
  geometric targets + one eval batch) runs first, and sentinels stop fresh runs after failures on
  two hosts. Watch `runs/PREFLIGHT_FAILED.*` and the first epoch times (DataLoader target cost).
- 2026-10-15 11:10: Massachusetts 6/6 GA/GB done + test-scored (GA ~4.0 h, GB ~4.4 h; GB epoch
  ~157 s vs GA 141 s). DeepGlobe: GA at epoch 38/40, GB at 34/40; no failures, no sentinels.
- FIRST RESULT (Massachusetts, pre-specified plan, stats from releases/862387b):
  PRIMARY GB − R0 per-image IoU −0.0021, boot [−0.0060, +0.0016], t [−0.0056, +0.0013] →
  no_detected_difference. Secondary: GA − R0 +0.0024 boot [−0.0020, +0.0069] but t
  [+0.0018, +0.0031] → inconclusive; GB − GA −0.0046 boot [−0.0087, −0.0008], t [−0.0083,
  −0.0008] (both exclude 0; secondary). APLS R0 .8047 / GA .8039 / GB .7990; clDice .8812 /
  .8828 / .8831. Tables: logs/roadseg/stats_massachusetts_geo{,_vs_GA}.txt.
- 2026-10-15 13:30 CONFIRMATORY TEST REGISTERED BEFORE ANY DEEPGLOBE GA/GB RESULT WAS READ
  (deepglobe_test_geo*_summary.json written 12:09 by aggregate 19841568, not opened): the
  Massachusetts signal GA − R0 (+0.0024, t-only) is tested on DeepGlobe as a confirmatory
  hypothesis: GA − R0 per-image test IoU, read from vs_baseline.GA in
  deepglobe_test_geo_summary.json; same decision rule (both two-level bootstrap and seed-paired t
  exclude 0 with the same sign = confirmed). Direction predicted: GA > R0. GB − R0 remains the
  pre-specified primary.
- 2026-10-15 13:31 DEEPGLOBE RESULT (aggregate 19841568, read after the registration above):
  PRIMARY GB − R0 −0.0017 boot [−0.0059, +0.0028] t [−0.0129, +0.0095] → no_detected_difference.
  CONFIRMATORY GA − R0 +0.0073 boot [+0.0033, +0.0118] t [+0.0011, +0.0135], per-seed
  +0.0070 +0.0049 +0.0099 → CONFIRMED (predicted direction). Pooled IoU R0 .6771 → GA .6858
  (+0.0087). GB − GA −0.0090 (both intervals exclude 0). APLS R0 .7755 / GA .7753 / GB .7588.
  Summary across datasets: drawing the mask (GB) does not help and is worse than GA on both;
  geometric auxiliary supervision (GA) helps on DeepGlobe (confirmed) and is inconclusive (+0.0024,
  t-only) on Massachusetts. Full output: logs/roadseg/stats_geo_final_aggregate.txt.
- 2026-10-15 geo2 prepared: commit a6cc48b (GAc/GAw rungs, campaign geo2, GEO2_ANALYSIS_PLAN
  committed before any geo2 run). Suite 19963973: 497 passed. Rehearsal 19963975: ALL PASSED,
  SMOKE_PASSED = a6cc48b. Session dropped before launch (~6 h idle).
- **LAUNCHED 2026-10-15 23:21: array 20012726 (10 tasks, campaign geo2), aggregate 20012727,
  snapshot releases/a6cc48b** (rehearsal check enforced, not overridden).
- Error breakdown (descriptive, saved test probs at each run's validation threshold; job
  19964515, scripts/analysis/error_breakdown.py, logs/roadseg/geo_error_breakdown.json); pooled IoU
  reproduces the scored values exactly. Edge/near pixels (≤ 2 px of the GT boundary) are 69% of
  R0's errors on Massachusetts, 51% on DeepGlobe. GA vs R0 on DeepGlobe: TP +0.1%, FN ≈ −0.6%,
  FP near −6.1%, FP far −7.2% → GA's gain is almost entirely FEWER FALSE POSITIVES (recall by
  width unchanged). GB: FN edge −5.8% but FP near +4.6% (draws roads wider), FN interior +1.5%.
- 2026-10-16 geo2 COMPLETE (array 20012726 10/10, 0 failures; aggregate 20012727). Read 11:25.
  PRIMARY (geo2 plan) Massachusetts GA − R0, seeds 0–4: +0.0027 boot [−0.0012, +0.0068], t
  [−0.0008, +0.0061], per-seed +0.0024 +0.0027 +0.0022 −0.0010 +0.0069 → no_detected_difference
  (no further extension, per plan). Pooled R0 .6556 / GA .6588 (5 seeds).
  DeepGlobe attribution (descriptive, reading rule fixed in the plan): GAc − R0 +0.0005 → no
  difference; GAw − R0 +0.0043 boot [+0.0002, +0.0088] t [−0.0013, +0.0098] → inconclusive;
  GA − GAc +0.0068 and GA − GAw +0.0030, both "difference". Neither single term carries the gain
  by the rule → reported as NEEDING BOTH TERMS; the width term contributes most (pooled GAw
  .6849 vs GA .6858 vs R0 .6771), centre-line term alone ~0. Output:
  logs/roadseg/stats_geo2_final_aggregate.txt.
- 2026-10-16 LITERATURE SEARCH: closest prior = Deep Distance Transform (Wang et al., CVPR 2020,
  arXiv 1912.03383; medical CT): seg head + rounded-EDT "scale classes" at every foreground voxel
  (classification, not regression) + ball reconstruction from a pseudo-skeleton at inference ≈ GA
  + GB. Not found applied to roads. Road priors: Wei et al. ISPRS Annals 2020 (centre-line +
  width regression, for vector extraction), Batra CVPR 2019 (orientation), BrightEarth 2024
  (distance map), cascaded surface/centre-line/edge (IEEE 2022). Note: DeepGlobe is described in
  the literature as precisely annotated with varying widths -> frame as "label widths vary", not
  "inconsistent".
- Campaign geo3 (branch feat/geo3-baselines, commit ba1e8e9): GD (DDT targets, 20 classes, all
  road px) and GO (orientation from skeleton structure tensor, 18 bins + background), GA's model,
  centre weight 0. GEO3_ANALYSIS_PLAN committed before launch: primary GA vs GD on DeepGlobe.
  Suite 20064298 + rehearsal 20064301 submitted; launcher job 20064329 (afterok on both) runs
  `launch.sh --campaign geo3` so a session drop cannot stall the launch again.
- CHN6-CUG (third dataset) needs the user: download is behind a request form (name/email/
  institution) at grzy.cug.edu.cn/zhuqiqi. Ottawa (RoadNet, Google Drive) is free but 21 images.
- (The two entries above were written 2026-10-16 ~14:00 and parked uncommitted until launch: a
  commit then would have moved HEAD off the rehearsed sha. The brief commit 91406a3 was reset.)
- Suite 20064298 at ba1e8e9: 511 passed. Rehearsal 20064301: ALL PASSED (14:55). Launcher job
  20064329 ran `launch.sh --campaign geo3` → **LAUNCHED: array 20079336 (6 tasks), aggregate
  20079337, snapshot releases/ba1e8e9.**
- 2026-10-17 geo3 COMPLETE (array 20079336 6/6, 0 failures; aggregate 20079337). Read 13:00.
  PRIMARY (geo3 plan) DeepGlobe GA − GD per-image IoU +0.0046, boot [+0.0021, +0.0072], t
  [+0.0025, +0.0068], per-seed +0.0055 +0.0045 +0.0038 → DIFFERENCE, GA ahead (pooled +0.0024).
  Secondary vs R0: GD +0.0027 (no detected difference; pooled +0.0063), GO +0.0014 (no
  difference; pooled +0.0041). GA − GO −… inconclusive (boot only). GAw − GD +0.0016 boot
  [−0.0011, +0.0042] t [−0.0004, +0.0036] → no detected difference: the disc-cover WIDTH encoding
  alone is not distinguishable from DDT's; GA's edge over DDT needs the centre term too.
  Pooled IoU: R0 .6771, GO .6812, GD .6834, GAw .6849, GA .6858. APLS: GO .7768, R0 .7755,
  GA .7753, GAw .7748, GD .7717. Output: logs/roadseg/stats_geo3_final_aggregate.txt.
- 2026-10-17 user asked to download the Ottawa (RoadNet) dataset as the third dataset; gdown job
  20187991 (data/_dl/ottawa_raw).
- 2026-10-17 THIRD DATASET = Ottawa (RoadNet, Liu et al. TGRS 2019; user's choice over CHN6-CUG,
  which needs a request form). Downloaded via gdown (job 20188055; zip sha256 aacaa827…73fb6).
  20 scenes (not 21), 0.21 m, manual road-SURFACE masks (red), widths 14–54 px (median 31).
  scripts/prepare_ottawa.py: exact 2× box downsample to 0.42 m (widths 7–27 px; fits RADII;
  smallest scene 512×640), mask = 2×2 majority; official test scenes 1, 16–20; val = scenes 9, 14
  (default_rng(1337) from the authors' training scenes 2–15); train = the other 12. Right-edge
  no-imagery strips cropped (scene 2: 168 cols, scene 1: 46 cols); scene 1 (test) keeps 1.45%
  black pixels (strip not a clean full-height block) — same pixels for every arm, disclosed.
- Campaign "ott" (branch feat/ott-dataset): R0, GA, GD × seeds 0–4 on Ottawa, 64 crops/scene/
  epoch × 200 epochs = 9,600 steps. OTT_ANALYSIS_PLAN committed with this entry, before any
  Ottawa run: primary GA vs R0 (Ottawa, 5 seeds); GA vs GD, GD vs R0 secondary; 6 test scenes →
  coarse intervals, disclosed in the plan. Rehearsal gains an "ottawa" section; time limit 4 h.
- 2026-10-17 suite 20190651 at 8cb06c4: 515 passed, 1 FAILED — test_sbatch_directives pins the
  rehearsal's --time at 03:00:00; 8cb06c4 raised it to 04:00:00 (the grid grew). Rehearsal
  20190653: ALL PASSED in 2 h 48 m under the 4 h limit, SMOKE_PASSED = 8cb06c4. The afterok
  launcher 20190655 was therefore cancelled (as designed).
  Ruling: launch 8cb06c4 by hand (`launch.sh --campaign ott`, rehearsal check enforced), then fix
  the test's expected value in a later commit. The failure is a stale pinned constant, not a code
  defect; the launched code is exactly the rehearsed commit. Cost if wrong: none for the runs.
  **LAUNCHED 2026-10-17 18:01: array 20200093 (15 tasks, campaign ott), aggregate 20200094,
  snapshot releases/8cb06c4.**
- 2026-10-18 ott COMPLETE (array 20200093 15/15, 0 failures; aggregate 20200094). Read 11:12.
  PRIMARY (ott plan) Ottawa GA − R0 per-image IoU −0.0029, boot [−0.0081, +0.0024], t [−0.0105,
  +0.0047], per-seed −0.0104 +0.0020 −0.0047 −0.0061 +0.0045 → no_detected_difference (final, no
  more seeds). Secondary: GD − R0 −0.0048 inconclusive (boot only, R0 ahead); GA − GD +0.0019 no
  difference. Pooled IoU R0 .8861 / GA .8828 / GD .8808 (Ottawa is near ceiling). APLS R0 .8828 /
  GA .8394 / GD .8198 (secondary, 6 test scenes; large drop, not tested). Output:
  logs/roadseg/stats_ott_final_aggregate.txt.
  SYNTHESIS: GA − R0 is confirmed on DeepGlobe only (+0.0073); Massachusetts +0.0027 and Ottawa
  −0.0029 show no detected difference. The DeepGlobe gain did NOT replicate on Ottawa; "helps when
  label widths vary" is not supported (Ottawa widths vary but labels are precise surfaces and IoU
  is ~0.88). GB (drawn mask) null on both datasets it ran on.
- 2026-10-18 GEOMETRY FEEDBACK (user: "add something in the architecture"; agreed guardrails:
  choose on validation only, pre-register, report every variant). Branch feat/gf-feedback.
  GF = GA's targets/loss, geometry head moved into the decoder and its probabilities (13 ch)
  concatenated into every later stage, so the mask uses the geometry at inference.
  GFa: predicted after stage 3 (1/2 res), fed into stage 4. GFb: after stage 2 (1/4), fed into
  stages 3–4. Campaign gfsel = seed 0 of both on all 3 datasets; scripts/analysis/gf_select.py
  picks the higher MEAN validation IoU (threshold_val_iou; ties → GFa) and never opens test files;
  gfsel's aggregation prints nothing. gfconf_a|b = the winner's seeds 1–2 on all 3 datasets.
  Plans gf_a / gf_b (committed with this entry, before any GF run): primary GF? vs R0 per dataset,
  secondary vs GA; both variants and the selection scores are reported whichever wins.
  scripts/gf_chain.sh runs gfsel then (afterok) selection + confirmation without a human.
  NOTE: validation IoU so far tracks test (DG: GA > R0 on both; Ottawa: GA < R0 on both).
- 2026-10-18 tests 20245961 + rehearsal 20245962 passed at c430651; gf-chain 20245963 launched
  gfsel (array 20253709); selector 20253712 (afterok) — VALIDATION IoU, seed 0:
  GFa mass .6501 / dg .6879 / ott .9006, mean .7462; GFb .6498 / .6890 / .8986, mean .7458 →
  GFa selected (margin .0004, essentially a tie); launched gfconf_a (array 20281436, aggregate
  20281440) 2026-10-18 22:03. All runs completed, 0 failures. Read 2026-10-23 11:50.
- GF RESULTS (plan gf_a, per dataset; GFa seeds 0–2):
  PRIMARY GFa − R0: Massachusetts +0.0032 boot [−0.0018, +0.0085] t [−0.0031, +0.0077] → no
  detected difference; DeepGlobe +0.0056 boot [+0.0012, +0.0103] t [−0.0072, +0.0183] →
  INCONCLUSIVE (boot only; per-seed +0.0045 +0.0010 +0.0111); Ottawa −0.0032 → no detected
  difference. Secondary GFa − GA: +0.0006 / −0.0017 / −0.0002, no detected difference on any
  dataset → feeding the geometry back adds nothing measurable over GA. Pooled IoU GFa .6591 /
  .6850 / .8833. APLS (secondary): Ottawa R0 .8828, GA .8394, GFa .8644 (partial recovery);
  Mass/DG unchanged. No further seeds (plan). Output: logs/roadseg/stats_gf_a_final_aggregate.txt.
