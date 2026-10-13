# SDD ledger — plan: /scratch/hk4488/roadseg/docs/superpowers/plans/2026-09-25-dwt-resnext-road-segmentation.md

Spec: /scratch/hk4488/roadseg/docs/superpowers/specs/2026-09-25-dwt-resnext-road-segmentation-design.md (read, binding authority)
Branch: feat/dwt-resnext-ablation
Branch base: fa24b74

Ruling: feature branch in place, not a git worktree — the plan hardcodes
/scratch/hk4488/roadseg throughout (configs, sbatch PYTHONPATH, data_root, runs_dir).
A worktree at a different path breaks every one of those. Cost if wrong: master is
one `git checkout` away; no work is lost.

## Pre-flight scan

### Cross-task interface pairs (shared file or symbol)

| Pair | Produces → consumes | Finding |
|------|--------------------|---------|
| T2→T3 | `DWT2d(ch)`, `IDWT2d(ch)` → ops.py | clean |
| T2→T4 | `DWT2d(3)` → encoder stem | clean |
| T3→T4 | `WaveletDown(in,out)` → maxpool + `_wavelet_ize` | clean |
| T3→T5 | `make_up(kind,in,out)` → decoder | clean |
| T4→T5 | `ResNeXtEncoder.CHANNELS` = decoder `enc_channels` default | clean, both (64,256,512,1024,2048) |
| T4→T6 | `ResNeXtEncoder(rung,pretrained,hf_init)` | clean |
| T5→T6 | `UNetDecoder(enc_channels,widths,up)` | clean |
| T6→T15 | `build_model(cfg)` | clean |
| T6→T16 | `build_baseline(name)` called with 1 arg; T16 signature `(name, pretrained=True)` | clean, default covers |
| T7→T13 | `ComboLoss` → engine test | clean |
| T7→T15 | `ComboLoss(use_cldice=)` | clean |
| T8→T14 | `confusion`/`metrics_from_confusion`/`relaxed_f1`/`cl_dice_metric` | clean |
| T9→T14 | `apls`, `graph_from_mask` | clean |
| T10→T14 | `tile_image`, `stitch` | clean |
| T10→T12 | T12 header claims it consumes `tile_positions`; its code does not | **F-C** |
| T11→T12 | `build_index(root,dataset,split)` | clean |
| T12→T14 | `RoadFullDataset` item keys image/mask/valid/id ↔ `evaluate` | clean |
| T12→T15 | `RoadTileDataset(root,ds,split,tile=,road_bias=)` | clean |
| T13→T15 | `save_checkpoint` kw-only epoch/best/cfg; `train_one_epoch(...,use_amp=,device=)` | clean |
| T14→T15 | `evaluate`→`agg["dataset_iou"]`, `select_threshold` | clean |
| T15→T16 | array job ↔ `configs/{dataset}.yaml` | clean |

### Per-task internal consistency (tests vs. code the task specifies)

| Task | Finding |
|------|---------|
| T1 | clean — pytest.sh PYTHONPATH covers `.deps` + `src`, tests import `roadseg.*` |
| T2 | clean — buffer not parameter, so `len(parameters())==0` holds |
| T3 | clean |
| T4 | clean — R0/R1/R2 all yield [64,32,16,8,4] from a 128 input; constant-input parity holds algebraically (LL=2v, W'=W/2) |
| T5 | clean — `_feats(size=128)` gives f4 at 8, five ×2 ups → 256 = 2× stem, matches assertion |
| T6 | clean |
| T7 | clean |
| T8 | clean — relaxed-F1 offset cases verified by hand (ρ=3, offset 2 → p=r=1; IoU=0) |
| T9 | **F-A** — `test_scales_to_many_nodes` |
| T10 | clean — (600,513) and (1500,1500) start-position arithmetic verified |
| T11 | clean — n=5/masks=False yields 3 train rows, all missing → raises |
| T12 | clean — biased-sampling test ≈99% per-sample hit rate over 12 tries; 36/40 threshold safe |
| T13 | clean |
| T14 | clean — paired bootstrap degenerate cases verified |
| T15 | clean |
| T16 | **F-B** — DLinkNet forward |

### Rulings

Ruling: **F-A** — T9's `test_scales_to_many_nodes` asserts APLS on a ~32k-node graph
finishes in 30 s. The dominant cost there is 200 networkx Dijkstra calls, not the
nearest-node lookup the test means to police, so correct cKDTree code could still
blow the budget. Replacing the timing assertion with a completion-and-finite-value
check on a large graph, budget 120 s. The O(N²) defect stays prevented by the
implementation instruction and task review. Cost if wrong: a performance regression
in APLS ships unnoticed; it would show up as slow eval jobs, not wrong numbers.

Ruling: **F-B** — T16's DLinkNet forward computes `self.e2(e1)` twice
(`e2, e3 = self.e2(e1), self.e3(self.e2(e1))`). Correct output, wasted compute.
Implementer instructed to assign sequentially. Cost if wrong: none, it is strictly
a simplification.

Ruling: **F-C** — T12's Interfaces header lists `tile_positions` as consumed; the
dataset does random crops and never calls it. Dropping the claim from the header.
Cost if wrong: none, documentation only.

## Progress

Task 1: implementer interrupted by session end 2026-09-25 — files written, uncommitted, no report.
  Resumed 2026-10-10: files match brief verbatim; smoke test passes (67 s, torch import on login node).
Ruling: `.deps` shadows the env's numpy (2.4.6 over 2.3.5), scipy, pillow, networkx, pyyaml,
  because `pip install --target` pulls transitive deps and `.deps` is first on PYTHONPATH.
  Prune those from `.deps` and make setup_deps.sh do it, so the conda env is the single source
  of core numerics and recorded results name one numpy. Cost if wrong: a pruned package needing
  the newer version fails at import — loud, caught by the smoke test.
Commit attribution changed this session to: Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Task 1: fresh implementer (abe4c66670486d7df) committed + pruned; controller amended author only → f82e8e1.
  Repo-local git identity set to hk4488 <hk4488@nyu.edu>.
Task 1: review — spec ✅, quality Approved, 1 Important: no test enforces the no-shadowing invariant.
  ⚠️ commit trailer — resolved by controller: present and correct.
Task 1: minor (deferred): prune step in setup_deps.sh never exercised by a genuinely fresh install.
Task 1: minor (deferred): `.superpowers/` .gitignore line bundled into the scaffolding commit.
Task 1: fix round 1/5 (1 addressed, 0 open — added test_core_packages_resolve_from_conda_env_not_deps, proven to fail on a reintroduced symlink; commits f82e8e1..2e178b8)
Task 1: complete (commits fa24b74..2e178b8, review clean)
Ruling: Task 2 implementer stored Haar kernel buffers in float64 (plan code built them in float32).
  The plan's own test demands 1e-10 reconstruction on float64 input, unreachable with float32-rounded
  1/√2; forward() casts to input dtype, so float32/bf16 paths are unchanged. Accepted — plan defect.
  Cost if wrong: a few bytes of buffer per wavelet layer; none on numerics.
Task 2: minor (deferred): DWT2d.forward doesn't validate input channel count; IDWT2d does (asymmetry inherited from plan).
Task 2: complete (commits 2e178b8..313d650, review clean)
Ruling: Task 3 implementer used the plan's stale trailer "Claude Opus 5". Controller amended the
  commit message only (c3d2cf6 → ddea65a, content identical) and sed-replaced all 17 stale trailers
  in the plan file (uncommitted; commit after Task 3 completes) so later briefs carry 5.5.
  Cost if wrong: none — commit metadata and plan text only.
Task 3: minor (deferred): make_up's unknown-kind branch untested (only make_down's is).
Task 3: minor (deferred): no test feeds odd spatial dims into WaveletDown (Task 6 guards the model entry point).
Task 3: complete (commits 313d650..ddea65a, review clean)
Plan attribution fix committed separately before Task 4.
Ruling: DATA — no Kaggle token. Massachusetts fetched from the original Toronto host (canonical source);
  DeepGlobe from HF mirror lammtfkday/deepglobe-roads (6226 labelled 1024² pairs, matches original
  counts/regions/GSD), flattened to the original Kaggle layout data/deepglobe/train/<id>_{sat.jpg,mask.png}
  so the SPEC's 70/15/15 seed-1337 split applies — the mirror's own 5000/600/626 seed-42 partition is
  NOT used. Fetched now by a scratchpad script (background, log /scratch/hk4488/logs/roadseg/fetch_data.log);
  Task 11's download_data.sh must be rewritten to these sources + layout, not Kaggle.
  Cost if wrong: mirror differs from the original DeepGlobe release — mitigated by count/size checks;
  a reviewer could object the paper should use the original Kaggle files, which is a re-download, not rework.
DATA status 2026-10-11: DeepGlobe complete (6226 pairs). Massachusetts 96% then Toronto timeouts; refetch running
  detached via .superpowers/fetch_mass.py (log /scratch/hk4488/logs/roadseg/fetch_mass.log, exits 0 when complete).

Task 4: session died again mid-dispatch; files on disk passed 8/8. Fresh implementer a362e99da022e427c verified
  vs brief, cardinality {32}×16 on R0/R1/R2, committed d1c4b30. (Brief's "9 passed" is a plan miscount; 8 is right.)
Task 4: review (opus) — spec ✅ (faithful transcription), quality NEEDS FIXES: 3 Important, all plan-mandated.

Ruling: **T4-F3 (R1 stem init not function-preserving)** — plan/spec §5.4 prescribed LL/2 + zero HF and claimed
  equivalence to R0; reviewer showed that is the pretrained kernel at stride 1 on an avg-pooled image (stem cosine
  0.62 vs R0), and that an EXACT init exists: G = DWT2d(pad(W, top/left 1)) placed at W'[:, :, 1:5, 1:5], giving
  R1 ≡ R0 at step 0 incl. borders. Spec's binding intent (§4, §5.1: R1−R0 attributable to the operator) outranks
  its §5.4 recipe. DECISION: hf_init="exact" is the new DEFAULT; "zero" (LL/2) and "small-random" stay as flags.
  Cost if wrong: if the authors want the "fresh" LL/2 init as the headline, it is one flag away (--hf-init zero);
  no code is lost.
Ruling: **T4-F1 + T4-F2 (R2 random-init projections discard pretraining; +60% encoder params)** — plan's
  WaveletDown (dense 1×1, 4C→C) replaced by WaveletPool(ch, init): DWT + depthwise 4→1 learned sub-band
  combination (4·C params, deterministic init, no RNG use), placed IN FRONT of the KEPT pretrained conv whose
  stride is set to 1. Shortcuts use init="select" (weights ½,½,½,½ = top-left pixel) → exactly the original
  stride-2 1×1; conv2-front and maxpool use init="ll" (2×2 average). R2−R0 param delta ≈ +0.2%.
  StridedDown / WaveletDown / make_down are deleted (dead after this). Spec §5.6 amended accordingly.
  Cost if wrong: a dense cross-channel sub-band projection might be a stronger model; it is also unattributable
  (capacity + lost pretraining). It can be added later as an explicit extra rung without touching R0–R3.
Ruling: **Decoder parity (pre-empting the same defect in Task 5)** — plan's WaveletUp (1×1 in→4·out, then IDWT)
  has 4/9 the params of BilinearUp's 3×3 conv (−6.1M over the decoder) and a different op order. Replaced by
  WaveletUp(in,out) = WaveletUnpool(in) [depthwise 1→4, init (2,0,0,0) = nearest-neighbour, then IDWT] followed
  by the SAME 3×3 conv as BilinearUp, both named `.conv`. Rungs then differ only in the 2× kernel.
  Cost if wrong: same as above — a different wavelet decoder is an extra rung, not a rework.
Ruling: **Seed pairing across rungs** (reviewer ⚠️) — extra layers must not shift the RNG stream for shared
  layers: WaveletPool/WaveletUnpool hold nn.Parameter with deterministic values (no nn.Conv2d → no RNG draw);
  RoadSegmenter (Task 6) builds decoder + head BEFORE the encoder. Tests assert shared weights are identical
  across rungs at a fixed seed, and that R1's outputs equal R0's at init. Carry into Task 5 and Task 6 dispatches.
  Cost if wrong: none — it only removes a nuisance variable.
Carry-forward: default hf_init is now "exact" → Task 6 (RoadSegmenter/build_model) and Task 15 (configs/base.yaml).
Carry-forward: `enc.rung` keeps the given label ("R3" stays "R3"); arch selection uses a private field.
⚠️ resolved by controller: pretrained weights not cached → caching resnext50_32x4d + resnet34 to TORCH_HOME now.
⚠️ open until GPU smoke test (post Task 15): bf16 autocast path through DWT buffers unexercised.
Task 4: minor (deferred): load_adapted_stem_from / stem_conv_out are test-only API on the production class.
Task 4: fix round 1/5 (3 addressed, 0 open — exact stem init default; WaveletPool in front of kept pretrained
  convs; WaveletUp = WaveletUnpool + same 3×3 conv; commits d1c4b30..b8043b7). Measured: params R0 22,979,904 /
  R1 23,008,128 (+0.12%) / R2 23,022,720 (+0.19%); cosine vs R0 at init: R1 1.0 all stages; R2 1.000/.909/.931/.945/.955.
  Re-review (opus): all addressed, no new Critical/Important. bf16 autocast exercised on CPU by reviewer — ⚠️ closed
  for CPU; CUDA path still to be exercised at the GPU smoke test.
Task 4: minor (deferred): test_zero_init_is_avgpool_not_R0 never asserts it differs from R0 (name overstates).
Task 4: minor (deferred): shared-weights test doesn't assert R1.conv1 == R2.conv1 nor that R2's extra keys are only pools/DWT buffers.
Task 4: minor (deferred): R3 encoder never forwarded in a test (label only).
Task 4: minor (deferred): task-4-report.md:27 stale line mentions WaveletDown.
Task 4: note for paper: inside R2 blocks at init the main path averages 2×2 (centre 2i+0.5) while the shortcut
  selects the top-left pixel (2i); stem pool is 2×2 avg vs R0's 3×3 max — this is the 0.91–0.96 cosine.
Carry-forward (Task 6): R1/R2's throwaway nn.Conv2d(12,64,7) draws RNG that R0 doesn't → decoder + head MUST be
  constructed before the encoder (already ruled); test must assert decoder/head weights equal across R0/R1/R2 at a seed.
Task 4: complete (commits 99bbf90..b8043b7, review clean after 1 fix round)
Spec amendment committed cafc624.

Task 5: review — spec ✅, Approved, no Critical/Important. ⚠️ ops.py attr names (unpool/idwt/conv) — resolved: pairing test passes.
Task 5: minor (deferred): skip-gradient test only runs up="bilinear"; no assert len(widths)==5; shape test batch size 1 only.
Task 5: complete (commits cafc624..84b52ac, review clean)
Ruling: overlap task N's review (read-only, background) with task N+1's implementation (different files); still one
  implementer at a time. Cost if wrong: a late Important finding on N forces rework in N+1 where N+1 builds on N.
Ruling: full suite takes ~11 min on the login node (ResNeXt builds on loaded CPU). Leaf-module implementers (Tasks 7–10)
  run focused tests + smoke only; controller runs the full suite at checkpoints (after 10, 14, 16).
  Cost if wrong: a cross-module break is found a few tasks late instead of immediately.
DATA: Massachusetts complete 2026-10-11 12:06 (1108/14/49 sat+map verified). ImageNet resnext50_32x4d + resnet34 cached in TORCH_HOME.
Task 6: implemented 10c1df8 (13 tests incl. seed pairing across R0–R3, R1≡R0 logits at init, R3≡R2 arch, no dead params). In review.
Task 6: review — spec ✅, Approved, no Critical/Important. ⚠️ test results/seed pairing: implementer report shows 13 passed incl. pairing + R1≡R0; encoder builds body before throwaway stem conv (verified in Task 4 re-review).
Task 6: minor (deferred): shape check is a bare `assert` (stripped under python -O); unknown rung now surfaces as KeyError from _UP_FOR_RUNG before the encoder's ValueError; build_model untested; divisibility test covers R2 only.
Task 6: complete (commits 84b52ac..10c1df8, review clean)
Task 7: implemented ba128d7 (7 loss tests). In review. Note: soft_cl_dice_loss's [0,1] guard does .min()/.max() → GPU sync per step on R3 (minor perf).
Task 7: review — spec ✅, Approved, no Critical/Important. clDice copy verified identical to source.
Task 7: minor (deferred, FIX BEFORE LAUNCH): dice_loss/ComboLoss compute sums in bf16 under autocast (~0.4% rounding of the Dice denominator; clDice soft-skeleton sums likewise) → cast logits .float() at the top of dice_loss and ComboLoss.forward, with a bf16-input test.
Task 7: minor (deferred): test_gradient_flows doesn't exercise use_cldice=True; clDice adds a constant 0.5 on empty targets (inflates R3 loss values — never compare raw loss across rungs); [0,1] guard passes NaN silently and forces a GPU sync; empty-mask dice test (32², logit −20) doesn't represent the 512² regime.
Task 7: complete (commits 10c1df8..ba128d7, review clean)
Task 8: implemented c740124 (8 metric tests, warning-free). In review. Implementer notes: fully-nodata region scores 1.0 on every metric (finite by convention) → per-image means must not include fully-masked images; cl_dice_metric has no valid arg (eval passes pred&valid, gt&valid).
Task 8: review — spec ✅, NEEDS FIXES: 3 Important.
Ruling: T8-1 fully-invalid image → NaN (not 1.0) from metrics_from_confusion and relaxed_f1; callers nanmean/exclude. Plan's
  test_fully_nodata_region_is_not_nan (finite) replaced. Cost if wrong: none for honest numbers; a NaN can only surface a real edge case.
Ruling: T8-2 relaxed_f1 tolerance is Euclidean (EDT), not the plan's (2ρ+1)² square — matches the published relaxed-P/R convention.
  Cost if wrong: numbers slightly lower than a square-tolerance implementation; the paper states "Euclidean ρ=3".
Ruling: T8-3 relaxed_f1 masks only the counted set, not the reference set (removes one-sided no-data border error). clDice border
  truncation documented as a limitation. Cost if wrong: Massachusetts labels inside no-data regions are trusted as references within ρ px of the border.
Task 8: fix round 1 dispatched to a19b92c80a2694728 (FIX_BASE c740124… note Task 9's commit ccc83ef sits between).
Task 9: implemented ccc83ef. Implementer found the brief's graph_from_mask defective (fragmented off-lattice '+', falsely joined
  parallel roads 4 px apart, lost road ends) and rewrote it: nodes = sampled skeleton px + endpoints + junctions; edges = skeleton
  paths with true pixel length. Open concern: pure-Python loops, untimed on 1500². In review.
Task 8: fix round 1/5 (3 addressed, 0 open — NaN for zero valid px; Euclidean EDT tolerance; unmasked reference set; commits ccc83ef..477f2ba). Re-review: all addressed, no new breakage.
Task 8: complete (commits ba128d7..c740124 + fix 477f2ba, review clean after 1 fix round)
Carry-forward (Task 14): metrics return NaN for images with zero valid pixels → evaluate() must nanmean and the bootstrap must drop NaN pairs; report dataset-pooled and mean-per-image separately; only IoU/F1 are meaningful as per-image means.
Task 9: review — spec ✅ (graph_from_mask rewrite judged correct), NEEDS FIXES: 2 Important — (1) plan's snap-defect test passes on the old behaviour (reviewer showed dx=1e4 scores 0.0 even with max_snap_dist=1e12); (2) apls(G,G,sample=500) = 26 s on a dense 1500² mask (~7 h for the test sets). Fix round 1 dispatched to abe953775d65732f3 (FIX_BASE for the diff: 2b8143d, since Task 10's commit sits after ccc83ef).
Ruling: APLS empty-graph convention — both empty → 1.0, exactly one empty → 0.0 (mirrors the pixel-metric convention). Cost if wrong: none; it is a definition, stated in the paper.
Task 9: minor (deferred): pairs sampled uniformly over nodes, not road length (documented); test_runs_on_a_large_graph greps the module source for "cKDTree" (brittle, controller-mandated).
Task 10: implemented 2b8143d (15 tiling tests incl. multi-channel and smaller-than-tile). Implementer anchored .gitignore `data/` → `/data/` because it was ignoring src/roadseg/data. In review.

## Real-data facts (controller inspection 2026-10-11, 40-image samples)
Massachusetts: sat *.tiff RGB 1500², labels *.tif mode L values {0,255}; 1108/14/49 all paired. White (>=250 all ch) covers mean 11.4% of
  pixels, >1% in 16/40 images, >30% in 7/40, max 79%. Labels have roads drawn INSIDE white regions (mean 2.6% of white px, max 20%).
  Small white blobs (roofs/paint) also trip the >=250 rule. Road frac over valid px: mean 5.1%.
DeepGlobe: sat *_sat.jpg 1024² RGB; masks *_mask.png mode RGB, binary {0,255}; 6226 paired; no nodata; road frac mean 3.6%.
Ruling: nodata = white pixels belonging to a connected white component of area >= 2500 px (50×50), Massachusetts ONLY; DeepGlobe valid = all True.
  Plan's valid_mask (any pixel >=250) would punch holes in real roads (lane paint, roofs). Cost if wrong: a few genuine nodata slivers
  under 2500 px stay "valid" — negligible area; the threshold is a named constant.
Ruling: the training loss MUST be masked by `valid` (plan's train_one_epoch ignored it): labels over blank nodata would otherwise teach the
  model to predict roads on white pixels. ComboLoss gains an optional `valid` argument (Task 13 dispatch amends losses.py, together with the
  fp32 cast flagged in Task 7). Cost if wrong: none — with valid all-ones it reduces to the unmasked loss (pinned by a test).
Ruling: train crop sampler rejects crops with valid fraction < 0.5 (within the existing retry loop) so batches are not dominated by blank tiles.
  Cost if wrong: slight under-sampling of image borders next to nodata.
Carry-forward (Task 11): layout confirmed — massachusetts/{train,valid,test}/*.tiff + {split}_labels/*.tif; deepglobe/train/<id>_{sat.jpg,mask.png}.
  download script = Toronto (base on .superpowers/fetch_mass.py, proven) + HF lammtfkday/deepglobe-roads flattened; NOT Kaggle.
Task 9: fix round 1/5 (2 addressed, 0 open — snap test now fails without the threshold; batched scipy Dijkstra 26 s → 3.9 s, equal to nx brute force at 1e-9; empty-graph convention; commits 2b8143d..f6c4511). Re-review: all addressed, edge cases (dup sources, zero-length edges, L2=0, disconnected) checked.
Task 9: complete (ccc83ef + fix f6c4511, review clean after 1 fix round)
Task 10: review — spec ✅, NEEDS FIXES: 1 Important plan-mandated (overlap test vacuous). fix round 1/5 (1 addressed, 0 open — exact-value averaging test, stride/coverage guards, leak test; commit 4f0fa98). Re-review: all addressed.
Ruling: Task 10's fix ran concurrently with Task 9's (different files, explicit-path staging). Cost if wrong: an index.lock collision — none occurred.
Task 10: complete (2b8143d + fix 4f0fa98, review clean after 1 fix round)
Task 11: implementer a86437f25976faa8c wrote prepare.py / download_data.py / test_prepare.py but could not commit (ran git from a wrong cwd). Controller committed the files unchanged. Implementer deviations: brief's test_index_pairs_images_with_masks was arithmetically impossible (int(3*0.7)=2) → n=10/7 rows; index filters on .tif/.tiff suffix; no RED run.
  DeepGlobe split fingerprint (seed 1337): 4358/933/935; first test ids 101296,102699,104350,104607,1054; sha256(sorted test ids joined by \n) = a18e6da5754390b186234dbe6ba98357a1618c9a03d12688a0bdc4fbf65883fe
Task 11: review — 1 Important (truncated download can look complete). fix round 1/5 (1 addressed, 0 open — Content-Length check, full-decode verify, per-dataset skips; commit 681cab2). Re-review: all addressed.
  Controller independently decoded all 2342 Massachusetts files: 0 bad; labels all 2,250,110 bytes; sat 6,765,878–6,765,892 bytes. Data on disk is intact.
Task 11: minor (deferred): fetch()'s 404/no-final-sleep paths untested; DeepGlobe repair = re-download zips; int(len*0.70) float arithmetic (pinned by fingerprint test).
Task 11: complete (aa028f6 + fix 681cab2, review clean after 1 fix round)
Task 12: implemented bf929e6 (connected-component nodata Mass-only, valid-fraction crop rejection, spec augmentations from one torch Generator). Measured: Mass val valid frac 0.9998, road frac 0.069; __getitem__ 0.46 s (Mass) / 0.35 s (DeepGlobe) on the login node. In review.
Ruling: per-crop whole-image decode (+ per-crop CC labelling) makes the loader the bottleneck (~2–3 crops/s/worker). Datasets preload decoded uint8 images, masks and (Mass) valid masks into RAM once at construction (thread pool), shared copy-on-write by forked workers: ~12.5 GB Mass train, ~18.3 GB DeepGlobe train. Goes into Task 12's fix round; SLURM --mem raised accordingly in Task 15.
  Cost if wrong: RAM pressure on a shared node → fall back to preload=False (kept as a flag) with more workers.
Plan gaps found by controller while preparing Tasks 13–15 (amend files written: task-13-amend.md, task-14-amend.md):
Ruling: LR schedule = per-iteration linear warmup (2 epochs) + cosine, per spec §10; plan's train.py ignored warmup_epochs.
Ruling: per-epoch validation computes pooled IoU only (full_metrics=False, stride 512); relaxed-F1/clDice/APLS only at final scoring (stride 256). Cost if wrong: checkpoint selected at stride 512 may differ marginally from stride-256 optimum.
Ruling: add `python -m roadseg.eval --run … --split test` (refuses to rescore test without --force) and `roadseg.stats.aggregate_runs` + CLI (mean±std over seeds; paired bootstrap on per-image IoU averaged over seeds, images as the resampling unit). Plan had no entrypoint for either.
Task 12: review — spec ✅ (alignment, seeding, normalise-after-jitter, no resizing all correct), NEEDS FIXES: 2 Important test gaps (no alignment test through augmentation; no real-DataLoader worker-seeding test). ⚠️ `val` split accepted by build_index for both datasets — resolved: Task 11 tests cover it.
Task 12: fix round 1 dispatched to ac3cf8549e032b367 (alignment test with negative control, DataLoader seeding test, RAM preload ruling, minor guards). FIX_BASE bf929e6.
Task 13: implemented 39bd3ce (engine + masked fp32 loss; 29 tests). Deviation: brief's overfit-one-batch test used i.i.d. random labels a tiny conv net cannot memorise (1.22→1.08 vs required 0.61) → target changed to y=(x[:,:1]>0); BCE normalised over all valid pixels in the batch; unused self.bce removed. In review.
Task 13: review — spec ✅, Approved, no Critical/Important. Overfit-test change judged acceptable; BCE batch normalisation is what A1 specified.
Task 13: minor (deferred, FIX BEFORE LAUNCH): no guard on non-finite GRADIENTS (clip_grad_norm_ would spread NaN, optimizer.step poisons weights silently) → error_if_nonfinite=True / raise FloatingPointError; save_checkpoint has no flush+fsync before replace (node crash on network FS could leave a truncated latest.pt).
Task 13: minor (deferred): max_batches fetches one extra batch (use islice); empty loader returns 0.0 silently; Dice weights each batch element equally regardless of valid area (note for the paper); clDice skeletonises a hard 0 boundary at no-data edges (R3 on Massachusetts); bf16 loss test is weak evidence of fp32 accumulation.
Task 13: ⚠️ open until GPU smoke test: use_amp=True path (autocast forward, fp32 loss) never exercised on CPU.
Task 13: complete (commits bf929e6..39bd3ce, review clean)
Task 14: implemented 7e9b4a7 (eval + stats; 24 tests). Notes: commit trailer says "Claude Sonnet 5.5" (the subagent followed its own attribution reminder; left as is — it is the truthful author); CLI reads cfg/threshold/dataset/data_root from <run>/metrics.json → Task 15 must write those keys; --tile/--stride flags added; aggregate_runs assumes rung names contain no underscore. In review.
Task 12: fix round 1/5 implemented 50bfbea (alignment test 0.9996 vs 0.738 negative control; 2-worker DataLoader seeding test; RAM preload + limit; guards). Per item 460 ms → 27.5 ms (Mass) / 346 → 32.4 ms (DeepGlobe) on a loaded login node; 15 ms target missed — Ruling: accepted as node load (warpAffine ≈30 ms under load); revisit only if GPU utilisation is low in the first real job. Unrequested change: uint8 transpose then float32 normalise instead of permute().contiguous(). Re-review dispatched.
Task 14: review (opus) — spec ✅, NEEDS FIXES: 4 Important — (1) bootstrap p can be exactly 0.0; (2) CI covers image sampling only [plan-mandated A5]; (3) test protocol tunable/unrecorded, --force destroys evidence, saved cfg mutated; (4) aggregate_runs trusts dir names, tolerates mismatched id sets. Reviewer's Monte Carlo: no recentring needed; type-I ≈ 0.057 at n=49.
Ruling: T14-1 +1 Monte Carlo p-value. Cost if wrong: none (strictly more correct).
Ruling: T14-2 keep image-level paired bootstrap (estimand: these trained models) and add a two-level bootstrap resampling seeds within arm then images (estimand: the training procedure) as the HEADLINE interval, plus a seed-level SE. Cost if wrong: with 3 seeds the two-level interval is crude and wide — honest, but may read as "inconclusive" where the image-level one looked decisive; both are reported.
Ruling: T14-3 unmodified cfg + `eval` block + standard/nonstandard protocol flag; --force renames the old result to *.superseded.json; atomic writes; --limit for smoke. Cost if wrong: none.
Ruling: T14-4 cross-check run dir name against cfg (`cfg["name"]`, written by train.py); raise on within-rung id mismatch; warn + intersect across rungs; non-standard-protocol runs excluded unless allowed. Cost if wrong: an overly strict check blocks aggregation of a legitimate partial grid — it says why.
Ruling: T14-5 (controller) per-image CPU metrics in a spawn process pool via a torch-free image_metrics module; APLS on 935 images was ~80 min of idle GPU per DeepGlobe run. Cost if wrong: pool overhead on tiny sets (workers=0 path kept).
Task 14: fix round 1 dispatched to acd675bcc96bf434d (task-14-fix1.md). FIX_BASE 7e9b4a7 for eval/stats files.
Task 16: implemented 4fa2167 (U-Net 31.04 M, D-LinkNet 31.10 M params; forward fix; divisible-by-32 guard; build_model passes pretrained). In review.
USER REQUIREMENT 2026-10-11: "it can be preempted also so checkpoints must be in place." → task-15-amend.md §A5: SIGTERM/SIGUSR1 handler → checkpoint + exit 75 → scontrol requeue; mid-epoch checkpoints every 10 min with step_in_epoch; mid-epoch resume runs the remaining steps; idempotent finish phase; atomic+fsync writes; kill-and-resume tests.
CLUSTER FACTS (sbatch --test-only, 2026-10-11 14:15): account torch_pr_627_general may use ONLY l40s_public and h200_public for GPU (both accepted together: --partition=l40s_public,h200_public --gres=gpu:1); do not pass --qos; estimated start ≈ 2026-10-12 10:36 (L40S) / 10-14 20:32 (H200), independent of time limit; 1042 pending on l40s_public; per-user cap 16 GPUs; cpu_short starts in ~15 min.
Ruling: launch plan = CPU end-to-end rehearsal on cpu_short first (catches everything but CUDA specifics), then submit the 36-run array (%16) to both partitions with test scoring at the end of each job, plus a dependent aggregation job. A separate GPU canary would cost an extra ~20 h of queueing. Cost if wrong: a CUDA-only bug fails all 36 jobs within minutes of starting and costs one resubmission (~1 day of queue).
Task 12: fix round 1/5 (3 addressed + guards, 0 open; commit 50bfbea). Re-review: all addressed; draw order unchanged; no cache aliasing.
Task 12: minor (deferred): _normalize_chw returns a non-C-contiguous tensor (astype default order='K'); values/dtype identical; collate makes batches contiguous. One-line order="C" fix + is_contiguous assertion.
Task 12: minor (deferred): alignment test's negative control relies on 3 of 4 features (full-width band is flip-invariant).
Task 12: complete (bf929e6 + fix 50bfbea, review clean after 1 fix round)
Task 16: review — rulings ✅; 1 Important plan-mandated: D-block is parallel, published D-LinkNet is a cascade (d1=relu(c1(x)), d2=relu(c2(d1)), …, out = x+d1+d2+d3+d4).
Ruling: T16-1 make it the cascade, with an impulse-response test (reach ±15 > ±8). The plan's code was wrong. Cost if wrong: none — the cascade is the published model.
Task 16: fix round 1 dispatched to a2714dfccb1c92898. FIX_BASE 4fa2167.
Task 16: minor (deferred): U-Net's true stride is 16 (the %32 guard is stricter than needed, by ruling); shape asserts strip under -O; param counts 31.0/31.1 M vs ~34 M to be stated in the paper table.
Task 16: fix round 1/5 (1 addressed, 0 open — cascade D-block with impulse-reach test ±15; commit 6b0ee63). Re-review: addressed, arithmetic hand-checked.
Task 16: complete (4fa2167 + fix 6b0ee63, review clean after 1 fix round)
SESSION DROPS 2026-10-11 ~17:19 and ~17:31: both while the controller's turn was idle-waiting on background agents (~10 min). Foreground agent calls of 20+ min have survived.
Ruling: never end a turn with background work pending; long implementers run in the FOREGROUND; long shell commands run detached (setsid nohup → log on /scratch); implementers commit in small steps. Cost if wrong: less parallelism.
Task 14: fix round 1 partially landed — 3ce3393 (stats: F1, F2, F4). eval.py (F3, F5, minors) mid-edit, uncommitted; image_metrics.py untracked; stats.py has further uncommitted edits. Fresh foreground implementer dispatched to finish.
Task 14: fix round 1/5 implemented in two commits: 3ce3393 (stats F1/F2/F4, first implementer) + 39b4369 (eval F3/F5/minors, finished by fresh implementer aed62617100f40cbe, 86 min wall on the loaded node). 53 tests pass; two-level bootstrap 0.47 s at 935×3×10 000. Once-only guard uses os.link (fails if target exists). Re-review dispatched (range 6b0ee63..39b4369).
Ruling: scripts/pytest.sh caps OMP_NUM_THREADS (default 4) — oversubscription turned a 46 s test file into 13 min; controller edited the runner directly (infrastructure, no product code). scripts/sbatch_tests.sh runs the full suite on cpu_short because the login node SIGKILLs it (checkpoint 1 reached ~80% with zero failures before exit 137).
Task 14: fix round 1/5 (F1–F5 + minors all addressed, 0 open; commits 3ce3393 + 39b4369). Re-review: all addressed, no new Critical/Important.
Task 14: minor (deferred): spawn workers under `python -m roadseg.eval` re-import the main module and so import torch (startup cost only); superseded filename has 1 s resolution; an unexpected os.link error would discard a long result; `--no-apls` leaves protocol "standard"; seed_level uses full-run aggregate means, not the common-id subset.
Task 14: complete (7e9b4a7 + fixes 3ce3393, 39b4369; review clean after 1 fix round)
SESSION DROP overnight 2026-10-11→12 during Task 15's foreground implementer (opus). It had committed cfdd9b9, 8e3e904, 07da2a9, e835551; uncommitted: train.py (+2: remove torn *.tmp) and tests/test_train_cli.py (+113: persistent-worker crop-stream test, real SIGKILL-mid-epoch resume test). No report written.
Checkpoint: full suite on cpu_short (job 19599777, commit ffec79b): 245 passed in 71 s. Compute nodes run the suite ~10× faster than the login node → all verification goes through SLURM from here (survives session drops).
2026-10-12 10:15: submitted full suite incl. slow tests on the working tree (job 19637818) and the CPU smoke rehearsal (job 19637819).
Task 15: remainder committed by controller 876ee55 (tested: full suite incl. slow, job 19637818, 313 passed).
Task 15: review (opus) — NEEDS FIXES: 1 Critical (crashed job hangs: workers ignore SIGTERM, parent's exit joins them forever, wrapper then requeues indefinitely — reproduced), 4 Important (no run-dir lock; wrapper never passes --device cuda; scancel undone by the wrapper's requeue; provenance only at finish / moving tree). Launch-readiness: resume exactness HOLDS, rung pairing HOLDS, no test leakage HOLDS; others HOLD WITH CAVEAT; concurrency DOES NOT HOLD.
  Reviewer facts: scontrol reports PreemptMode=OFF, GraceTime=0 for both public partitions; KillWait=30 s.
Ruling: T15 fix round = task-15-fix1.md G1–G7: G1 prompt exit on every path + exit-code contract; G2 lockf on <run>/.lock + per-process temp names; G3 --device cuda, no fallback; G4 requeue only on USR1; G5 frozen snapshot via scripts/launch.sh + per-process code identity, refuse on mismatch; G6 (controller) GPU preflight per job + sentinel + --preflight-only + sbatch_canary.sh; G7 ckpt_minutes 5, preemptible finish, strict aggregate, SMOKE_PASSED acknowledgement.
  Cost if wrong: mostly added refusal paths — a too-strict guard stalls a run loudly rather than corrupting it.
Ruling: no separate day-long canary gate before the array (each job preflights instead); a canary is submitted in parallel for early signal. Cost if wrong: a CUDA-only bug costs 36 × ~1 min of GPU and one resubmission.
Task 15: fix round 1/5 implemented by ae88aa10cf3163f56 (opus, 2 h 40 min): 45d4fbd G1, 31cbad8 G4, 6092c25 G3, 23def86 G2, b44035d G5, 18b7c8e G6, f7c83c3 G7, 8a00751 G1 follow-up. Full suite job 19647591 at 8a00751: 363 passed. Deviations: USR1 + train exit 0 still requeues (180 s cannot score test); signal during validation scoring publishes metrics.json and returns 0; code identity read from the import root's COMMIT; --preflight-only ignores sentinels; eval locks for --split val too; --device cuda without CUDA exits 3; SMOKE_PASSED only from the full default rehearsal of a clean commit.
  Implementer concerns: (a) it corrupted one line of rehearsal 19637819 by editing train.py in place while that job ran on the live tree → that rehearsal is evidence for nothing; (b) a node-specific GPU fault writes the preflight sentinel and blocks every later start; (c) a requeue starting before a dead node's NFS lock lease expires exits 2 and is not requeued.
2026-10-12 13:16: submitted fresh CPU rehearsal at 8a00751 (job 19648631, live tree — tree must stay untouched until it finishes) and GPU canary from snapshot releases/8a00751 (job 19648632).
Ruling: canary submitted with --no-smoke-check (it is itself a test; the real launch stays gated on SMOKE_PASSED). Cost if wrong: a failure the rehearsal would have caught costs one canary queue cycle.
Task 15: re-review of fix round 1 (opus, 58 min): G1–G7 ALL ADDRESSED, all 8 deviations ACCEPTABLE, no new Critical. Wrapper state machine tabulated: crash never requeues, scancel never requeues, USR1 requeues exactly once, "already scored" is success, preflight failure never requeues. Every ending of roadseg.train terminates promptly with lock released (pool-crash paths probed by the reviewer: exit 1 in 2.6–5 s).
  New breakage in the fix diff: 1 Important operational (sentinel checked before "already complete"/resume; one node-specific preflight failure blocks every later start — the controller's own G6 ruling as written) + minors (no requeue cap; USR1 + slow finish → TIMEOUT without requeue; RunLock.release may raise in finally; threshold key lacks code identity; --preflight-only can write a sentinel into real runs/; launch.sh exports CANARY_*/SMOKE_* vars).
GPU CANARY PASSED — job 19648632, snapshot releases/8a00751, node gl038, NVIDIA L40S, 13:33–13:52 (queued only 17 min):
  preflight batch 16 × 512², bf16: R0 3.89 steps/s 11.91 GB | R1 3.87 / 11.91 | R2 3.84 / 12.97 | R3 3.65 / 12.97 | unet 2.97 / 17.21 | dlinknet 9.69 / 5.41.
  miniature R1 + R3 runs on GPU incl. test scoring OK; preemption drill: SIGTERM → "checkpoint saved at epoch 1 step 4" → resumed at epoch 1 step 4 → final state ok (2 history rows, 1 resume). ⚠️ CLOSED: bf16/CUDA paths (Task 4, Task 13) exercised on GPU.
  Throughput ⇒ ≈4 h per Massachusetts run (55.4k steps), ≈6–7 h per DeepGlobe run (87k steps) + validation; inside the 24 h limit.
Ruling: Task 15 fix round 2 (task-15-fix2.md, H1–H8): sentinel blocks only fresh runs and only with ≥2 distinct failing hosts; requeue cap (SLURM_RESTART_COUNT > 20); release() swallow; bounded lock retry on requeue; USR1 watchdog (kill + requeue after 120 s); eval refuses a checkpoint whose epoch ≠ best_epoch; `launch.sh --smoke` runs the rehearsal from a snapshot (never the live tree); small hardening.
  Cost if wrong: one more rehearsal (~70 min CPU). The canary at 8a00751 stays valid for the CUDA path: round 2 touches wrapper/sentinel/lock logic, not model, data, loss or AMP code.
Ruling: launch order = fix round 2 → scoped re-review → rehearsal from snapshot → launch.sh (array + aggregate) → THEN the final whole-branch review while jobs queue/run. A results-affecting finding from the final review means cancel + relaunch from a new snapshot; an operational one is fixed in the tree and applies to any relaunch.
  Cost if wrong: GPU-hours spent on runs that have to be redone.
CPU REHEARSAL PASSED — job 19648631 at 8a00751 (live tree, untouched during the run), 1 h 08 min: 24 runs (6 architectures × 2 datasets × 2 seeds): train → score test once → re-train is a no-op → re-score refused; aggregation for both datasets; preemption drills mid-epoch (epoch 0 step 2) and after first epoch (epoch 1 step 1) both resume to the correct final state; BF16-CPU: PASS. SMOKE_PASSED written for 8a00751.
Ruling: do NOT launch 8a00751 yet although launch.sh would accept it. GPUs arrived in 17 min for the canary, so queue age is not scarce; the campaign is ~180 GPU-hours on a shared partition; fix round 2 and the final whole-branch review run first (concurrently), then one rehearsal from a snapshot, then launch. Cost if wrong: ~3 h later results.
Task 15: fix round 2 dispatched to ae88aa10cf3163f56 (resumed, background). FIX_BASE 8a00751.
FINAL REVIEW dispatched (opus, foreground) on fa24b74..8a00751.

## FINAL WHOLE-BRANCH REVIEW (opus, fa24b74..8a00751, 19 min) — verdict: LAUNCH AFTER THE FIXES LISTED
No defect found that can bias R1 − R0. Probed on the real launch path: all 318 non-fc ImageNet tensors bitwise equal in R0; R1/R2/R3 differ only in conv1 = exact adaptation; decoder + head bitwise equal R0–R3; max|R1−R0| logits 1.5e-8; R3 ≡ R2 architecture; encoder params 22,979,904 / 23,008,128 / 23,022,720; resolved configs differ only in name/rung; one AdamW group; crops depend only on (seed, token); eval rebuild + strict load diff 0.0; DeepGlobe fingerprint a18e6da5… on disk, 4358/933/935 disjoint, no byte-identical image across splits; Massachusetts 1108/14/49 disjoint; LR 3e-6 → 3e-4 (step 1108 / 4358) → 3e-6 over 55,400 / 87,160 steps.
Validity: (1) only the operator differs — HOLDS WITH CAVEAT (R2/R3 also change the decoder upsampler, by spec; R2 not function-preserving at init, max logit diff 3e-3; R3 loss); (2) pretraining real — HOLDS; (3) normalisation + DWT — HOLDS; (4) labels/no-data — HOLDS WITH CAVEAT (tiles edge-flush, spec §6.2 said reflection padding, no ruling); (5) no leakage — HOLDS; (6) statistics — HOLDS WITH CAVEAT (two-level interval covers 0.94–0.98 typically but 0.86 when seed variance dominates at n=935; empty-GT images score 1 or 0, each Massachusetts test image is 2.0 pt); (7) reproducibility — HOLDS WITH CAVEAT (cudnn.benchmark, mixed GPU types, provenance gaps).
MUST FIX: R3's extra term = 0.5·(1−softDice, eps 1e-6) + 0.5·(1−clDice), inherited from sam2_road_dac's soft_cl_dice_loss ("copy as-is", spec §9) → R3−R2 mixes clDice with 1.5× Dice.
Ruling: FW-1 R3 loss = BCE + Dice + cldice_weight·(1 − clDice), cldice_weight 0.5, no second Dice. Cost if wrong: λ=0.5 is untuned (disclosed); it is the weight clDice already carried.
Ruling: FW-2 test pass saves 8-bit probability maps and also scores IoU@0.5 — the test split is scored once, and later questions (threshold sensitivity, unmasked numbers, figures) must not need a second pass. Cost: ~0.2–1 GB per DeepGlobe run on scratch.
Ruling: FW-3 environment/provenance block; FW-4 no APLS on validation in the finish phase (idle GPU); FW-5 protocol flag tied to cfg; unet records pretrained:false. Cost if wrong: none.
Ruling: tiles are edge-flush (last tile aligned to the image edge), NOT reflection-padded as spec §6.2 said — the network never sees padded borders it was not trained on, and every tile is real imagery. Spec to be amended. Cost if wrong: none identified; overlap averaging is uneven near the last row/column.
Ruling: T14-2 revisited per final review — keep the two-level bootstrap as headline, ADD a seed-paired t interval (df = 2) beside it and state the coverage simulation; if R1−R0 lands near the interval edge, add seeds 3 and 4 for R0 and R1 (~45 GPU-h) before claiming anything. Post-launch stats commit, before any result is read.
Ruling: stats robustness outputs (pooled-IoU difference with bootstrap; difference excluding images below an n_gt floor; top-influence images; seed-paired t) are specified and committed BEFORE results are read, but AFTER launch — stats.py reads JSON only and is not on the training path. Cost if wrong: the dependent aggregation job from the launch snapshot prints the older table; the final table is produced by re-running stats from the later commit.
Final wave W1–W5 queued to ae88aa10cf3163f56 behind H8 (final-wave.md). One combined full-suite run at the end.
Limitations the paper must disclose (from the final review, beyond spec/ledger): R2/R3 change encoder and decoder operator together; only hf_init="exact" is run; one untuned recipe (lr 3e-4, wd on all params incl. BN, clip 1.0; 100/40 epochs); checkpoint+threshold selected per run on validation (14 images for Massachusetts); Massachusetts scores masked by a custom no-data rule → not comparable with published numbers; DeepGlobe split random by image (adjacent tiles may straddle splits); 3 seeds and the interval's coverage; only R1−R0 IoU is pre-specified; bf16, non-deterministic kernels, mixed GPU types; flush tiling, uniform averaging, no TTA; U-Net from scratch; HF mirror pixel-identity with the Kaggle release unverified; rehearsal/canary touched a few test images with miniature models (no decision used them); a relaunch after seeing aggregates would be a second look at test.
SESSION DROP 2026-10-12 ~15:00–21:30 with two implementers running. Recovered from git:
  Task 15 fix round 2 (ae88aa1): 9bc503c H1, efc16d8 H2, 30914cc H3+H4, b862bbc H5, 22ae022 H6, 5e29c06 H7, 7789427 H8.
  Final wave (ae88aa1): 58dba3f W1 (R3 loss = R2 loss + weighted clDice only), f441051 W2 (probability maps + IoU@0.5), 4cb9374 W3 (environment block), 8e8e5e5 W4 (no APLS on validation), fa4b329 W5.
  Stats wave (fresh opus implementer, stats-wave.md): 01947a5 S1 (analysis plan, primary row), 09a9d54 S2 (seed-paired t), e390e6c S3 (pooled-IoU diff), 7c21a11 S4 (near-empty sensitivity), a51b97e S5 (robustness block, --json-only), e89baf4 follow-up. No report file was written for the stats wave or the final wave (sessions cut off).
  Full suite job 19664683 at e390e6c: 435 passed, 1 FAILED — test_the_canary_script_rehearsed_on_cpu: the other implementer committed during the run, the code identity changed between two processes of one run in the live tree, and the provenance guard refused (the guard working as designed; a test-environment artifact). The implementer's uncommitted fix (script rehearsals run from a frozen copy with a COMMIT file) committed by the controller as c1d1e2f.
2026-10-12 21:36: full suite at c1d1e2f submitted (job 19676269, live tree — do not edit until it ends) and CPU rehearsal from snapshot releases/c1d1e2f via `launch.sh --smoke` (job 19676272).
Combined scoped re-review of the fix wave dispatched (opus, foreground): 8a00751..c1d1e2f.
COMBINED RE-REVIEW of the fix wave (opus, 8a00751..c1d1e2f, 19 min): verdict LAUNCH. H1–H8, W1–W5, S1–S5 all ADDRESSED; no must-fix.
  Probed: ComboLoss(False) and dice_loss bitwise equal to base → R0–R2 and baselines unchanged; r3 == r2 + 0.5·cl_dice_term bit for bit; empty target → term exactly 1.0 with exactly zero gradient; fp32 under bf16 autocast; models/, data/, engine.py, metrics.py, image_metrics.py, apls.py byte-identical to 8a00751; wrapper decision table confirmed over 26 stub scenarios (crash never requeues, scancel never requeues, USR1 requeues once, preflight failure never requeues, "already scored" is success, >20 restarts stops); primary analysis identical at base and head on the same input (the stats wave only adds); seed-paired t matches scipy.ttest_1samp; pooled-IoU bootstrap matches a naive loop; environment block works on real data/weights, sha256 matches torchvision filename hashes; preflight exercises the new R3 loss path with 2 real optimizer steps.
  New breakage: 1 Important NOT blocking — bare `return` reached from a trap handler (sbatch_train_array.sh:165, :183) returns 128+signal and `set -e` kills the wrapper on a SECOND USR1 while the watchdog is pending (SLURM sends it once) → fix `return 0`. Minors: orphaned watchdog `sleep` (harmless, cgroup reaps it); uncaught read error while hashing weights (startup only); --force re-score overwrites <split>_probs/ while keeping the superseded metrics.
  Reviewer facts: this cluster has PreemptMode=REQUEUE (cluster-level), KillWait=30, proctrack/cgroup; so the restart cap of 20 also counts SLURM's own preemption requeues.
  Should fix before results are read: ANALYSIS_PLAN has no rule for a disagreement between the primary two-level interval and the seed-paired t (df=2 can be spuriously narrow as well as wide); print tiny p-values as <0.0001.
Ruling: launch c1d1e2f as reviewed once the full suite (19676269) and the snapshot rehearsal (19676272) pass. The `return 0` fix and the analysis-plan disagreement rule go into a follow-up commit AFTER launch (launch.sh requires HEAD == the rehearsed commit) and BEFORE any result is read; the array's wrapper is frozen at submit time, so that fix applies only to a relaunch.
  Cost if wrong: a second USR1 (operator action) or a microsecond race leaves one task un-requeued → manual resubmit of that task.
Ruling (pre-specified now, to be written into ANALYSIS_PLAN): the primary conclusion uses the two-level bootstrap interval; a claimed difference requires BOTH the two-level interval and the seed-paired t interval to exclude zero; if they disagree the result is reported as "inconclusive at 3 seeds" and seeds 3–4 are run for R0 and R1 before any claim. Cost if wrong: conservative — a real small effect may be reported as inconclusive.
Watch after launch: runs/PREFLIGHT_FAILED.* in the first hour; tasks stopping at restart 21.
Full suite job 19676269 at c1d1e2f: 444 passed. CPU rehearsal from snapshot (job 19676272, releases/c1d1e2f): SMOKE: ALL PASSED 22:33 (24 runs, aggregation, 2 preemption drills, BF16-CPU: PASS); SMOKE_PASSED = c1d1e2f.
*** LAUNCHED 2026-10-12 ~22:40: array job 19679158 (36 tasks, %16, --partition=l40s_public,h200_public), aggregate job 19679159 (afterany), code = releases/c1d1e2f (commit c1d1e2fe39beb2c55d3ad4a8e04c8e28c30990ec). runs/ was empty except LAUNCHES.log; no sentinels. ***
Post-launch follow-up: 0ac3c2e (trap handlers return 0 + double-USR1 stub test; sonnet implementer, cut off after this commit) and d7893e8 (decision rule + <0.0001 p formatting; code by the cut-off implementer, tests and one key-set update by the controller — Ruling: controller finished it natively because sessions kept dropping mid-dispatch; cost if wrong: one unreviewed-author change, mitigated by the review below). Review (sonnet): APPROVED, no Critical/Important; reviewer confirmed the wrapper test fails with exit 138 without the fix; decide() checked over 9 interval combinations + unavailable/degenerate. Minors: a secondary rung's `decision` is distinguished only by the sibling `role` key; degenerate-t → inconclusive is stricter than the plan text (documented, tested).
The decision rule was first committed in writing (README.md, docs/DECISIONS.md, 0fc8039, 2026-10-12 22:39) before the first run finished; the code implementing it (d7893e8) was committed 2026-10-13 ~08:55 before any test score was read by anyone.
STATUS 2026-10-13 09:01: Massachusetts ladder complete (12/12 trained + test-scored); DeepGlobe ladder 12 runs training (epochs 8–23 of 40); baselines 4 running, 8 pending; 0 failures, 0 requeues, 0 sentinels.
FIRST RESULT 2026-10-13 09:05 — Massachusetts ladder (12 runs), analysed with stats at d7893e8 (plan + decision rule committed before any score was read); output saved to /scratch/hk4488/logs/roadseg/stats_massachusetts_2026-10-13.txt and runs/massachusetts_test_summary.json:
  pooled test IoU: R0 0.6562±0.0011 | R1 0.6554±0.0024 | R2 0.6545±0.0001 | R3 0.6553±0.0018 (49 images, 3 seeds).
  PRIMARY R1−R0 per-image IoU: +0.0004, two-level bootstrap 95% CI [−0.0040, +0.0060], p=0.90; seed-paired t [−0.0095, +0.0104], p=0.87 (d_s −0.0042, +0.0029, +0.0026). DECISION: no_detected_difference.
  Secondary: R2−R0 −0.0009 [−0.0060, +0.0049]; R3−R0 −0.0018 [−0.0061, +0.0020]. Robustness (pooled IoU, IoU@0.5, n_gt floors) all consistent; no image excluded at any floor.
Full suite at final head d7893e8 submitted: job 19708284.
