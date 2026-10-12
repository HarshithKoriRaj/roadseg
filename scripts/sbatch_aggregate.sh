#!/bin/bash
#SBATCH --job-name=roadseg-aggregate
#SBATCH --output=/scratch/hk4488/logs/roadseg/%x_%j.out
#SBATCH --error=/scratch/hk4488/logs/roadseg/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_627_general
#
# Aggregate the test results of every finished run: mean +- std over seeds per rung and
# the paired bootstrap of each rung against R0, for both datasets. Submit behind the array:
#
#   jid=$(sbatch --parsable scripts/sbatch_train_array.sh)
#   sbatch --dependency=afterany:$jid scripts/sbatch_aggregate.sh
#
# Extra arguments are passed to `python -m roadseg.stats` (e.g. --metric cl_dice).
# Writes $RUNS/<dataset>_test_summary.json. A dataset whose R0 runs are not finished fails
# loudly; the other one is still aggregated and the job exits non-zero.

set -uo pipefail

PROJECT=/scratch/hk4488/roadseg
PY="${ROADSEG_PY:-/scratch/hk4488/.conda/envs/unsloth_sft/bin/python}"
RUNS="${ROADSEG_RUNS:-$PROJECT/runs}"
export PYTHONPATH="$PROJECT/.deps:$PROJECT/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$PROJECT"
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  runs $RUNS  $(date)"
echo "finished runs: $(ls "$RUNS"/*/test_metrics.json 2>/dev/null | wc -l) of 36"

status=0
for dataset in massachusetts deepglobe; do
    echo "=== $dataset ==="
    if ! "$PY" -m roadseg.stats --runs-dir "$RUNS" --dataset "$dataset" --split test "$@"; then
        echo "AGGREGATE FAILED for $dataset" >&2
        status=1
    fi
done
exit "$status"
