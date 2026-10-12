#!/bin/bash
#SBATCH --job-name=roadseg
#SBATCH --output=/scratch/hk4488/logs/roadseg/%x_%A_%a.out
#SBATCH --error=/scratch/hk4488/logs/roadseg/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --partition=l40s_public,h200_public
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_627_general
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --array=0-35%16
#
# One array task = one run: train, then score the test split once.
#
#   mkdir -p /scratch/hk4488/logs/roadseg        # SLURM does not create the log directory
#   sbatch scripts/sbatch_train_array.sh         # all 36 runs, at most 16 at a time
#   sbatch --array=0-23%16 scripts/sbatch_train_array.sh    # the ablation ladder only
#   bash scripts/sbatch_train_array.sh --print-map          # show the mapping; touches nothing
#
# Index -> run (seed is the fastest index):
#
#   index   dataset         name                         formula
#   -----   -------------   --------------------------   ------------------------------------
#    0-11   massachusetts   R0 R0 R0 R1 R1 R1 ... R3     ladder, i in 0..23:
#   12-23   deepglobe       R0 R0 R0 R1 R1 R1 ... R3       seed    = i % 3
#                                                          rung    = (R0 R1 R2 R3)[(i / 3) % 4]
#                                                          dataset = (massachusetts deepglobe)[i / 12]
#   24-29   massachusetts   unet x3, dlinknet x3         baselines, j = i - 24:
#   30-35   deepglobe       unet x3, dlinknet x3           seed    = j % 3
#                                                          model   = (unet dlinknet)[(j / 3) % 2]
#                                                          dataset = (massachusetts deepglobe)[j / 6]
#
# Preemption. The partitions are preemptible and the job is --requeue. SLURM sends USR1 to
# this shell 180 s before the time limit (--signal=B:USR1@180) and TERM on preemption. Both
# are forwarded to python as SIGTERM; train.py then checkpoints after the current optimizer
# step and exits 75, and this script asks for a requeue. The requeued task resumes from
# latest.pt (--resume auto). Test scoring writes its result atomically and only once, so an
# interrupted scoring is simply redone and a finished one is never repeated.

set -euo pipefail

PROJECT=/scratch/hk4488/roadseg
PY="${ROADSEG_PY:-/scratch/hk4488/.conda/envs/unsloth_sft/bin/python}"
RUNS="${ROADSEG_RUNS:-$PROJECT/runs}"
LOGS="${ROADSEG_LOGS:-/scratch/hk4488/logs/roadseg}"

RUNGS=(R0 R1 R2 R3)
BASELINES=(unet dlinknet)
DATASETS=(massachusetts deepglobe)
SEEDS=(0 1 2)

# map_index i -> DATASET, NAME, SEED, SELECT (the train.py flag pair that selects the model)
map_index() {
    local i=$1
    if ! [[ "$i" =~ ^[0-9]+$ ]] || (( i > 35 )); then
        echo "array index '$i' is outside 0-35" >&2
        return 1
    fi
    if (( i < 24 )); then
        SEED=${SEEDS[$(( i % 3 ))]}
        NAME=${RUNGS[$(( (i / 3) % 4 ))]}
        DATASET=${DATASETS[$(( i / 12 ))]}
        SELECT=(--rung "$NAME")
    else
        local j=$(( i - 24 ))
        SEED=${SEEDS[$(( j % 3 ))]}
        NAME=${BASELINES[$(( (j / 3) % 2 ))]}
        DATASET=${DATASETS[$(( j / 6 ))]}
        SELECT=(--model "$NAME")
    fi
}

if [[ "${1:-}" == "--print-map" ]]; then
    for i in $(seq 0 35); do
        map_index "$i"
        echo "$i $DATASET $NAME $SEED ${DATASET}_${NAME}_s${SEED}"
    done
    exit 0
fi

map_index "${SLURM_ARRAY_TASK_ID:?run as an array task (or use --print-map)}"
RUN="$RUNS/${DATASET}_${NAME}_s${SEED}"
mkdir -p "$LOGS" "$RUNS"

export PYTHONPATH="$PROJECT/.deps:$PROJECT/src"
export TORCH_HOME=/scratch/hk4488/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

echo "task $SLURM_ARRAY_TASK_ID -> dataset=$DATASET name=$NAME seed=$SEED run=$RUN"
echo "host $(hostname)  job ${SLURM_JOB_ID:-?}  restart ${SLURM_RESTART_COUNT:-0}  $(date)"
nvidia-smi -L || true

# ---- signal forwarding -------------------------------------------------------------------
# bash runs a trap only between commands, so a foreground python would hide the signal until
# it exited. Each step therefore runs in the background and the shell sits in `wait`, which a
# trapped signal interrupts at once.
child=""
got_signal=0
on_signal() {
    got_signal=1
    echo "signal received at $(date); forwarding SIGTERM to pid ${child:-<none>}"
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null || true
    fi
}
trap on_signal USR1 TERM

# run_step cmd...  -> STEP_RC = the command's own exit status
run_step() {
    local rc=0
    "$@" &
    child=$!
    if (( got_signal )); then              # the signal landed before $child was known
        kill -TERM "$child" 2>/dev/null || true
    fi
    STEP_RC=0
    wait "$child" || STEP_RC=$?
    # A trapped signal makes `wait` return (>128) while the child is still running.
    while kill -0 "$child" 2>/dev/null; do
        STEP_RC=0
        wait "$child" || STEP_RC=$?
    done
    # If the child exited between the interrupted `wait` and the check above, STEP_RC is
    # still the interruption; its real status is waiting to be collected (127 = already was).
    wait "$child" 2>/dev/null || rc=$?
    if (( rc != 0 && rc != 127 )); then
        STEP_RC=$rc
    fi
    child=""
}

requeue() {
    local ref="${SLURM_JOB_ID:-}"
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        ref="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    fi
    echo "requeueing $ref: $1"
    if ! scontrol requeue "$ref"; then
        echo "scontrol requeue $ref FAILED; resubmit this task by hand (it resumes from latest.pt)" >&2
        exit 75
    fi
    exit 0
}

cd "$PROJECT"

# ---- 1. train ----------------------------------------------------------------------------
run_step "$PY" -m roadseg.train --config "configs/${DATASET}.yaml" "${SELECT[@]}" \
    --seed "$SEED" --resume auto --runs-dir "$RUNS"
echo "train exited with $STEP_RC"
if (( STEP_RC == 75 )); then
    requeue "preempted, checkpoint saved"
elif (( got_signal )); then
    # Signalled while python could not checkpoint (still importing: killed, 143), or it
    # finished in the grace period (0): either way there is no time left to score.
    requeue "signalled (train exit $STEP_RC)"
elif (( STEP_RC != 0 )); then
    exit "$STEP_RC"
fi

# ---- 2. score the test split, once -------------------------------------------------------
rm -f "$RUN"/.test_metrics.json.*.tmp       # left behind if an earlier scoring was killed
run_step "$PY" -m roadseg.eval --run "$RUN" --split test --workers 8
echo "eval exited with $STEP_RC"
if (( STEP_RC == 0 )); then
    echo "DONE $RUN"
elif (( STEP_RC == 2 )) && [[ -s "$RUN/test_metrics.json" ]]; then
    # eval.py refuses (exit 2) to score the test split twice. 2 is also argparse's usage
    # error, so it only counts as success when the result is really there.
    echo "already scored: $RUN/test_metrics.json exists; nothing to do"
elif (( got_signal )); then
    requeue "signalled during test scoring (eval exit $STEP_RC); nothing was written"
else
    exit "$STEP_RC"
fi
