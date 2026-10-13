# roadseg — does a DWT front-end help a ResNeXt segment roads?

A controlled test of the "DWT + Modified ResNeXt" proposal on Massachusetts Roads and DeepGlobe.
One model class, one recipe, one set of seeds; the rungs differ only in how resolution is
reduced and restored.

| Rung | Encoder downsampling | Decoder upsampling | Loss |
|------|----------------------|--------------------|------|
| R0 | stock ResNeXt-50 32×4d | bilinear + conv | BCE + Dice |
| R1 | Haar DWT at the input (the paper's proposal) | bilinear + conv | BCE + Dice |
| R2 | learned Haar-basis pooling at every downsample | learned Haar-basis unpooling + conv | BCE + Dice |
| R3 | = R2 | = R2 | BCE + Dice + 0.5·clDice |

Baselines: U-Net (from scratch), D-LinkNet (ResNet-34). Three seeds each, both datasets: 36 runs.
The primary result is **R1 − R0 in per-image test IoU**. A null result is a valid outcome.

- Design and its amendments: `docs/superpowers/specs/2026-09-25-dwt-resnext-road-segmentation-design.md`
- Every decision taken during the build, and the limitations to disclose: `docs/DECISIONS.md`

## Where things are

| Path | What |
|------|------|
| `src/roadseg/` | models, data, losses, metrics, training, evaluation, statistics |
| `configs/` | `base.yaml`, `massachusetts.yaml`, `deepglobe.yaml` |
| `scripts/launch.sh` | the only way to submit jobs |
| `data/` | the two datasets (not in git) — `scripts/download_data.py` rebuilds and verifies them |
| `releases/<sha>/` | frozen code snapshots the jobs actually run |
| `runs/<dataset>_<name>_s<seed>/` | one directory per run |
| `runs/LAUNCHES.log` | what was submitted, when, from which commit |
| `/scratch/hk4488/logs/roadseg/` | SLURM logs |

A finished run directory holds `metrics.json` (config, validation-selected threshold, validation
scores, provenance), `test_metrics.json` (test scores, per image), `test_probs/` (8-bit
probability maps), `history.json`, `best.pt`, `latest.pt`.

## Watching a launch

```bash
squeue -u $USER                                   # what is queued / running
tail -n 3 /scratch/hk4488/logs/roadseg/roadseg_<arrayid>_<task>.out
ls runs/*/test_metrics.json | wc -l               # finished runs (36 when done)
ls runs/PREFLIGHT_FAILED.* 2>/dev/null            # a GPU self-test failed — read the file
bash scripts/sbatch_train_array.sh --print-map    # task index -> dataset / model / seed
```

Jobs checkpoint every 5 minutes and at every epoch, and resume by themselves after a preemption
or a time-limit requeue. A task that ends FAILED did not requeue on purpose (a crash, a refused
start, or more than 20 restarts): read its log, then resubmit the same task from the same
snapshot (absolute paths; the snapshot of a launch is in `runs/LAUNCHES.log`):

```bash
SNAP=/scratch/hk4488/roadseg/releases/<sha>
sbatch --array=<task> --export=ALL,ROADSEG_CODE=$SNAP $SNAP/scripts/sbatch_train_array.sh
```

It continues from its checkpoint. A run refuses to continue under different code.

## Results

The aggregation job submitted with a launch writes `runs/<dataset>_test_summary.json` when every
task has ended. To produce the table by hand (CPU only, seconds):

```bash
cd /scratch/hk4488/roadseg
PY=/scratch/hk4488/.conda/envs/unsloth_sft/bin/python
PYTHONPATH=.deps:src $PY -m roadseg.stats --runs-dir runs --dataset massachusetts --split test
PYTHONPATH=.deps:src $PY -m roadseg.stats --runs-dir runs --dataset deepglobe     --split test
```

The row marked primary is R1 vs R0. Read `docs/DECISIONS.md` §D before interpreting anything:
a difference is claimed only if both the two-level bootstrap interval and the seed-paired t
interval exclude zero. Everything else in the table is secondary.

**The test split is scored once per run.** `roadseg.eval` refuses to score it again; do not use
`--force` to "check something" — the saved probability maps answer most later questions without
a second pass. Re-launching after looking at aggregates is a second look at the test set and
must be disclosed.

## Changing the code

```bash
sbatch scripts/sbatch_tests.sh        # full suite on a compute node (~25 min); never on the login node
bash scripts/launch.sh --smoke        # CPU rehearsal of the exact commit (~70 min); required before a launch
bash scripts/launch.sh --canary       # GPU self-test of every architecture (~20 min)
bash scripts/launch.sh --dry-run      # show what a launch would submit
bash scripts/launch.sh                # submit the 36-run array and the aggregation job
```
