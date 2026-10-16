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
