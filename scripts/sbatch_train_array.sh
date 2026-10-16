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
#   bash scripts/launch.sh                       # THE way to submit: all 36 runs, at most 16
#                                                # at a time, from a frozen snapshot of HEAD
#   bash scripts/sbatch_train_array.sh --print-map          # show the mapping; touches nothing
#
# Do not sbatch this file by hand: without ROADSEG_CODE (set by launch.sh) each requeue would
# import whatever the live repository holds at that moment.
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
# Campaign "geo" (paper 2; ROADSEG_CAMPAIGN=geo, set by `launch.sh --campaign geo`, which
# also submits --array=0-11): the geometric-target rungs. Its control is the R0 runs above.
#
#    0-5    massachusetts   GA GA GA GB GB GB            seed    = i % 3
#    6-11   deepglobe       GA GA GA GB GB GB            rung    = (GA GB)[(i / 3) % 2]
#                                                        dataset = (massachusetts deepglobe)[i / 6]
#
# Campaign "geo2" (ROADSEG_CAMPAIGN=geo2, --array=0-9): seeds 3-4 of R0 and GA on Massachusetts
# (the decision rule's follow-up of an inconclusive result) and GA's attribution on DeepGlobe.
#
#    0-1    massachusetts   R0 s3 s4
#    2-3    massachusetts   GA s3 s4
#    4-6    deepglobe       GAc s0 s1 s2                  (GA, centre-line loss term only)
#    7-9    deepglobe       GAw s0 s1 s2                  (GA, width loss term only)
#
# Campaign "geo3" (ROADSEG_CAMPAIGN=geo3, --array=0-5): prior auxiliary targets on GA's model.
#
#    0-2    deepglobe       GD s0 s1 s2                   (Deep Distance Transform scale classes)
#    3-5    deepglobe       GO s0 s1 s2                   (road orientation, Batra et al.)
#
# Signals. Both are forwarded to python as SIGTERM; train.py then checkpoints after the
# current optimizer step and exits 75. What happens next depends on WHICH signal it was:
#
#   USR1  the time-limit warning, sent to this shell only, 180 s before the limit
#         (--signal=B:USR1@180). Nobody else will restart the task, so this script asks for
#         a requeue -- if python stopped for the signal (75, killed by it, or training just
#         finished with no time left to score). This is the ONLY path to `scontrol requeue`.
#   TERM  scancel, or a real preemption. Never requeued here: requeueing would undo the
#         user's scancel, and a preempted job is requeued by SLURM itself (--requeue).
#
# A failure is never requeued, whatever signal coincides with it: 1 crashed, 2 refused,
# 3 preflight failed would all fail the same way again (exit codes: src/roadseg/runtime.py).
# The requeued task resumes from latest.pt (--resume auto). Test scoring writes its result
# atomically and only once, so an interrupted scoring is simply redone and a finished one is
# never repeated.

set -euo pipefail

PROJECT=/scratch/hk4488/roadseg
# Code and configs come from $ROADSEG_CODE: the frozen snapshot that scripts/launch.sh creates
# and exports, so every requeue of every task imports the same commit. Data, .deps, runs and
# logs stay under the project directory. (sbatch copied this script at submit time, so its
# own logic is frozen too.) Default: the live tree -- for tests and rehearsals only.
CODE="${ROADSEG_CODE:-$PROJECT}"
PY="${ROADSEG_PY:-/scratch/hk4488/.conda/envs/unsloth_sft/bin/python}"
RUNS="${ROADSEG_RUNS:-$PROJECT/runs}"
LOGS="${ROADSEG_LOGS:-/scratch/hk4488/logs/roadseg}"

CAMPAIGN="${ROADSEG_CAMPAIGN:-dwt}"
RUNGS=(R0 R1 R2 R3)
GEO_RUNGS=(GA GB)
BASELINES=(unet dlinknet)
DATASETS=(massachusetts deepglobe)
SEEDS=(0 1 2)

# map_index i -> DATASET, NAME, SEED, SELECT (the train.py flag pair that selects the model)
map_index() {
    local i=$1
    if [[ "$CAMPAIGN" == geo ]]; then
        if ! [[ "$i" =~ ^[0-9]+$ ]] || (( i > 11 )); then
            echo "array index '$i' is outside 0-11 (campaign geo)" >&2
            return 1
        fi
        SEED=${SEEDS[$(( i % 3 ))]}
        NAME=${GEO_RUNGS[$(( (i / 3) % 2 ))]}
        DATASET=${DATASETS[$(( i / 6 ))]}
        SELECT=(--rung "$NAME")
        return 0
    elif [[ "$CAMPAIGN" == geo2 ]]; then
        local g2_data=(massachusetts massachusetts massachusetts massachusetts deepglobe deepglobe
                       deepglobe deepglobe deepglobe deepglobe)
        local g2_name=(R0 R0 GA GA GAc GAc GAc GAw GAw GAw)
        local g2_seed=(3 4 3 4 0 1 2 0 1 2)
        if ! [[ "$i" =~ ^[0-9]+$ ]] || (( i > 9 )); then
            echo "array index '$i' is outside 0-9 (campaign geo2)" >&2
            return 1
        fi
        DATASET=${g2_data[$i]}; NAME=${g2_name[$i]}; SEED=${g2_seed[$i]}
        SELECT=(--rung "$NAME")
        return 0
    elif [[ "$CAMPAIGN" == geo3 ]]; then
        if ! [[ "$i" =~ ^[0-9]+$ ]] || (( i > 5 )); then
            echo "array index '$i' is outside 0-5 (campaign geo3)" >&2
            return 1
        fi
        local g3_name=(GD GO)
        DATASET=deepglobe; NAME=${g3_name[$(( i / 3 ))]}; SEED=$(( i % 3 ))
        SELECT=(--rung "$NAME")
        return 0
    elif [[ "$CAMPAIGN" != dwt ]]; then
        echo "unknown ROADSEG_CAMPAIGN '$CAMPAIGN' (dwt, geo, geo2 or geo3)" >&2
        return 1
    fi
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
    last=35
    [[ "$CAMPAIGN" == geo ]] && last=11
    [[ "$CAMPAIGN" == geo2 ]] && last=9
    [[ "$CAMPAIGN" == geo3 ]] && last=5
    for i in $(seq 0 "$last"); do
        map_index "$i"
        echo "$i $DATASET $NAME $SEED ${DATASET}_${NAME}_s${SEED}"
    done
    exit 0
fi

map_index "${SLURM_ARRAY_TASK_ID:?run as an array task (or use --print-map)}"
RUN="$RUNS/${DATASET}_${NAME}_s${SEED}"
mkdir -p "$LOGS" "$RUNS"

if [[ ! -d "$CODE/src/roadseg" || ! -f "$CODE/configs/${DATASET}.yaml" ]]; then
    echo "no code at ROADSEG_CODE=$CODE (src/roadseg, configs/${DATASET}.yaml): not started" >&2
    exit 2
fi
export ROADSEG_CODE="$CODE"
export PYTHONPATH="$PROJECT/.deps:$CODE/src"
export TORCH_HOME=/scratch/hk4488/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

echo "task $SLURM_ARRAY_TASK_ID -> dataset=$DATASET name=$NAME seed=$SEED run=$RUN"
echo "host $(hostname)  job ${SLURM_JOB_ID:-?}  restart ${SLURM_RESTART_COUNT:-0}  $(date)"
if [[ -f "$CODE/COMMIT" ]]; then
    echo "code $CODE  commit $(cat "$CODE/COMMIT") (frozen snapshot)"
else
    echo "code $CODE  commit $(git -C "$CODE" rev-parse HEAD 2>/dev/null || echo unknown)" \
         "- WARNING: the LIVE tree, not a snapshot; submit with scripts/launch.sh"
fi
nvidia-smi -L || true

# ---- restart cap -------------------------------------------------------------------------
# Every requeue (ours at the time limit, SLURM's after a preemption) counts. A task that has
# been restarted this often is not making progress -- e.g. a step that can never finish inside
# the time limit -- and would otherwise take a GPU-day per attempt, for ever.
MAX_RESTARTS=20
if (( ${SLURM_RESTART_COUNT:-0} > MAX_RESTARTS )); then
    echo "task $SLURM_ARRAY_TASK_ID ($RUN) has been restarted ${SLURM_RESTART_COUNT} times" \
         "(limit $MAX_RESTARTS): NOT running it again and NOT requeueing. Read the earlier" \
         "attempts in this log, fix the cause, then resubmit this index by hand." >&2
    exit 1
fi

# ---- signal forwarding -------------------------------------------------------------------
# bash runs a trap only between commands, so a foreground python would hide the signal until
# it exited. Each step therefore runs in the background and the shell sits in `wait`, which a
# trapped signal interrupts at once.
child=""
got_usr1=0          # time-limit warning: checkpoint, then requeue
got_term=0          # scancel or preemption: checkpoint, then get out of the way
forward() {
    echo "$1 received at $(date); forwarding SIGTERM to pid ${child:-<none>}"
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null || true
    fi
}

# ---- time-limit watchdog -----------------------------------------------------------------
# USR1 leaves 180 s. A python that does not stop in time (the finish phase's validation
# scoring cannot be interrupted) would run into the limit: the job ends TIMEOUT and nobody
# requeues it. So WATCHDOG_SECONDS after USR1 was forwarded, a step that is still running is
# SIGKILLed -- its whole process group, DataLoader workers included -- and the task is
# requeued. That is safe: every file appears by an atomic rename, and resuming after a kill at
# any instant is what the kill-sweep tests establish. Cancelled when the step ends by itself;
# never fires after TERM.
WATCHDOG_SECONDS="${ROADSEG_USR1_WATCHDOG:-120}"
watchdog=""          # pid of the pending watchdog
watchdog_fired=0
start_watchdog() {
    if [[ -n "$watchdog" || -z "$child" ]] || (( got_term )); then
        return 0
    fi
    ( trap 'kill "$nap" 2>/dev/null; exit 0' TERM
      sleep "$WATCHDOG_SECONDS" &
      nap=$!
      wait "$nap" && kill -USR2 "$$" 2>/dev/null ) &
    watchdog=$!
}
stop_watchdog() {
    if [[ -n "$watchdog" ]]; then
        kill -TERM "$watchdog" 2>/dev/null || true
        wait "$watchdog" 2>/dev/null || true
        watchdog=""
    fi
}
on_watchdog() {
    watchdog=""
    if (( got_term )) || [[ -z "$child" ]] || ! kill -0 "$child" 2>/dev/null; then
        return 0                           # TERM takes precedence; or the step is already gone
    fi
    watchdog_fired=1
    echo "watchdog: pid $child is still running ${WATCHDOG_SECONDS} s after USR1 was forwarded;" \
         "SIGKILL to its process group (the state on disk is kill-safe), then requeue"
    kill -KILL -- "-$child" 2>/dev/null || true     # the group (the step runs under setsid)
    pkill -KILL -P "$child" 2>/dev/null || true     # ... and, without setsid, its children
    kill -KILL "$child" 2>/dev/null || true
}

on_usr1() { got_usr1=1; forward "USR1 (time limit in 180 s)"; start_watchdog; }
on_term() { got_term=1; stop_watchdog; forward "TERM (scancel or preemption)"; }
trap on_usr1 USR1
trap on_term TERM
trap on_watchdog USR2

# Each step is the leader of its own process group, so the watchdog can kill all of it.
# (SLURM tracks the job by cgroup, so its own signals still reach every process.)
SETSID=()
if command -v setsid > /dev/null 2>&1; then
    SETSID=(setsid)
fi

# run_step cmd...  -> STEP_RC = the command's own exit status
run_step() {
    local rc=0
    ${SETSID[@]+"${SETSID[@]}"} "$@" &
    child=$!
    if (( got_usr1 || got_term )); then    # the signal landed before $child was known
        kill -TERM "$child" 2>/dev/null || true
        if (( got_usr1 )); then
            start_watchdog
        fi
    fi
    STEP_RC=0
    wait "$child" || STEP_RC=$?
    # A trapped signal makes `wait` return (>128) while the child is still running.
    while kill -0 "$child" 2>/dev/null; do
        STEP_RC=0
        wait "$child" || STEP_RC=$?
    done
    stop_watchdog
    # If the child exited between the interrupted `wait` and the check above, STEP_RC is
    # still the interruption; its real status is waiting to be collected (127 = already was).
    wait "$child" 2>/dev/null || rc=$?
    if (( rc != 127 )); then               # also 0: the child's real status, not the interruption
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

# terminated <step>: TERM arrived. Whatever python did with it, do not requeue.
terminated() {
    echo "TERM during $1 (exit $STEP_RC): not requeueing. A scancel means stop; a preempted" \
         "job is requeued by SLURM itself and resumes from latest.pt."
    exit 143
}

cd "$PROJECT"

# ---- 1. train ----------------------------------------------------------------------------
run_step "$PY" -m roadseg.train --config "$CODE/configs/${DATASET}.yaml" "${SELECT[@]}" \
    --seed "$SEED" --resume auto --runs-dir "$RUNS" --device cuda
echo "train exited with $STEP_RC"
if (( got_term )); then
    terminated "training"
elif (( watchdog_fired )); then
    requeue "time limit: train did not stop within ${WATCHDOG_SECONDS} s of USR1 and was killed"
elif (( got_usr1 )) && (( STEP_RC == 75 || STEP_RC == 143 || STEP_RC == 0 )); then
    # 75: checkpoint saved. 143: killed by the forwarded signal before its handlers existed
    # (still importing). 0: training finished in the grace period, but 180 s cannot score it.
    requeue "time limit (train exit $STEP_RC)"
elif (( STEP_RC == 75 )); then
    echo "train exited 75 but this shell saw no USR1: not requeueing (python was signalled" \
         "directly; if SLURM is preempting the job it requeues it itself)" >&2
    exit 75
elif (( STEP_RC != 0 )); then
    echo "train FAILED with exit $STEP_RC (1 crashed, 2 refused, 3 preflight): not requeued" >&2
    exit "$STEP_RC"
fi

# ---- 2. score the test split, once -------------------------------------------------------
# (Temp files of an earlier, killed scoring are cleared by roadseg.eval itself, under the
# run's lock -- a blind rm here could delete the temp file of a scorer that is still alive.)
# --save-probs: the test split is scored once, so keep its probability maps (8-bit PNG).
run_step "$PY" -m roadseg.eval --run "$RUN" --split test --workers 8 --device cuda --save-probs
echo "eval exited with $STEP_RC"
if (( STEP_RC == 0 )); then
    echo "DONE $RUN"
elif (( STEP_RC == 2 )) && [[ -s "$RUN/test_metrics.json" ]]; then
    # eval.py refuses (exit 2) to score the test split twice. 2 is also argparse's usage
    # error, so it only counts as success when the result is really there.
    echo "already scored: $RUN/test_metrics.json exists; nothing to do"
elif (( got_term )); then
    terminated "test scoring"
elif (( watchdog_fired )); then
    requeue "time limit: eval did not stop within ${WATCHDOG_SECONDS} s of USR1 and was killed; nothing was written"
elif (( got_usr1 && STEP_RC == 143 )); then
    requeue "time limit during test scoring (eval killed by the signal); nothing was written"
else
    echo "eval FAILED with exit $STEP_RC: not requeued" >&2
    exit "$STEP_RC"
fi
