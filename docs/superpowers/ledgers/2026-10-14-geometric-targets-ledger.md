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
