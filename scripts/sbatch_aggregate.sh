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
# Aggregate the test results: mean +- std over seeds per rung and the paired bootstrap of each
# rung against R0, for both datasets. scripts/launch.sh submits it behind the array with
# --dependency=afterany, i.e. it runs whatever happened to the 36 tasks. So:
#
#   * all 36 runs scored            -> the tables, exit 0
#   * anything missing              -> the missing runs are listed, the tables over what IS there
#                                      are still printed (marked INCOMPLETE), and the job exits
#                                      non-zero: a table over 35 runs must not pass for the result
#   * --allow-partial               -> same output, exit 0 (if the aggregation itself succeeds)
#
# By hand, e.g. after rerunning a failed task:  bash scripts/sbatch_aggregate.sh [--allow-partial]
# Other arguments are passed to `python -m roadseg.stats` (e.g. --metric cl_dice).
# Writes $RUNS/<dataset>_test_summary.json. A dataset whose R0 runs are not finished fails
# loudly; the other one is still aggregated and the job exits non-zero.

set -uo pipefail

PROJECT=/scratch/hk4488/roadseg
CODE="${ROADSEG_CODE:-$PROJECT}"        # the frozen snapshot of scripts/launch.sh; default: live tree
PY="${ROADSEG_PY:-/scratch/hk4488/.conda/envs/unsloth_sft/bin/python}"
RUNS="${ROADSEG_RUNS:-$PROJECT/runs}"
CAMPAIGN="${ROADSEG_CAMPAIGN:-dwt}"     # geo: paper 2's GA/GB runs, analysed with --plan geo
export ROADSEG_CODE="$CODE" ROADSEG_CAMPAIGN="$CAMPAIGN"
export PYTHONPATH="$PROJECT/.deps:$CODE/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

allow_partial=0
stats_args=()
for arg in "$@"; do
    if [[ "$arg" == "--allow-partial" ]]; then
        allow_partial=1
    else
        stats_args+=("$arg")
    fi
done

cd "$PROJECT"
commit=$(cat "$CODE/COMMIT" 2>/dev/null || git -C "$CODE" rev-parse HEAD 2>/dev/null || echo unknown)
echo "host $(hostname)  code $CODE  commit $commit  runs $RUNS  $(date)"

# ---- which of the 36 runs are scored? (the same map the array uses) --------------------------
missing=()
total=0
while read -r _ _ _ _ name; do
    total=$(( total + 1 ))
    [[ -s "$RUNS/$name/test_metrics.json" ]] || missing+=("$name")
done < <(bash "$CODE/scripts/sbatch_train_array.sh" --print-map)
expected=36
plan_args=()
if [[ "$CAMPAIGN" == geo ]]; then
    expected=12
    plan_args=(--plan geo)
    # its control: the first campaign's R0 runs
    for dataset in massachusetts deepglobe; do
        for seed in 0 1 2; do
            total=$(( total + 1 ))
            [[ -s "$RUNS/${dataset}_R0_s${seed}/test_metrics.json" ]] || missing+=("${dataset}_R0_s${seed}")
        done
    done
    expected=18
elif [[ "$CAMPAIGN" == geo2 ]]; then
    plan_args=(--plan geo2)
    # its controls: seeds 0-2 of R0 and GA on both datasets (campaigns dwt and geo)
    for dataset in massachusetts deepglobe; do
        for name in R0 GA; do
            for seed in 0 1 2; do
                total=$(( total + 1 ))
                [[ -s "$RUNS/${dataset}_${name}_s${seed}/test_metrics.json" ]] || missing+=("${dataset}_${name}_s${seed}")
            done
        done
    done
    expected=22
fi
if (( total != expected )); then
    echo "AGGREGATE FAILED: the run map has $total entries, expected $expected" >&2
    exit 1
fi
echo "scored runs: $(( total - ${#missing[@]} )) of $total"
for name in ${missing[@]+"${missing[@]}"}; do
    reason="no test_metrics.json"
    [[ -e "$RUNS/$name/metrics.json" ]] || reason="training not finished (no metrics.json)"
    [[ -d "$RUNS/$name" ]] || reason="never started (no run directory)"
    echo "MISSING $name: $reason"
done
ls "$RUNS"/PREFLIGHT_FAILED.* 2>/dev/null | sed 's/^/NOTE preflight sentinel: /'

status=0
for dataset in massachusetts deepglobe; do
    echo "=== $dataset$( (( ${#missing[@]} )) && echo '  (INCOMPLETE: not all runs are scored, see above)') ==="
    if ! "$PY" -m roadseg.stats --runs-dir "$RUNS" --dataset "$dataset" --split test \
            ${plan_args[@]+"${plan_args[@]}"} ${stats_args[@]+"${stats_args[@]}"}; then
        echo "AGGREGATE FAILED for $dataset" >&2
        status=1
    fi
    if [[ "$CAMPAIGN" == geo2 && "$dataset" == deepglobe ]]; then
        echo "--- $dataset: GA vs its parts (secondary: GAc, GAw against GA) ---"
        "$PY" -m roadseg.stats --runs-dir "$RUNS" --dataset "$dataset" --split test --plan geo2 \
            --baseline GA ${stats_args[@]+"${stats_args[@]}"} || status=1
    fi
    if [[ "$CAMPAIGN" == geo ]]; then
        echo "--- $dataset: GB vs GA (secondary: the representation, supervision held fixed) ---"
        "$PY" -m roadseg.stats --runs-dir "$RUNS" --dataset "$dataset" --split test --plan geo \
            --baseline GA ${stats_args[@]+"${stats_args[@]}"} || status=1
    fi
done

if (( ${#missing[@]} )); then
    if (( allow_partial )); then
        echo "INCOMPLETE: ${#missing[@]} of $total runs are missing; accepted because of --allow-partial"
    else
        echo "INCOMPLETE: ${#missing[@]} of $total runs are missing (listed above). The tables above" \
             "are NOT the result. Rerun the missing tasks, then aggregate again" \
             "(--allow-partial accepts a partial set)." >&2
        status=1
    fi
fi
exit "$status"
