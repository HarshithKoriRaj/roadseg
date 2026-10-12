#!/bin/bash
#SBATCH --job-name=roadseg-smoke
#SBATCH --output=/scratch/hk4488/logs/roadseg/%x_%j.out
#SBATCH --error=/scratch/hk4488/logs/roadseg/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_627_general
#
# End-to-end CPU rehearsal of everything the GPU array will do, on real data with real
# ImageNet weights, in miniature (2 epochs x 3 steps, 8 training / 2 validation / 3 test
# images, tile 256). Run it before submitting the array:
#
#   sbatch scripts/sbatch_smoke_cpu.sh        # ends with "SMOKE: ALL PASSED" or stops at the failure
#
#  1. every dataset x (R0 R1 R2 R3 unet dlinknet) x seeds (0 1): train -> score the test split
#     -> train again (must say "already complete") -> score again (must refuse, exit 2)
#  2. aggregate both datasets (--allow-nonstandard: these scores are not the protocol)
#  3. preemption drill, mid-epoch: SIGTERM a running job -> exit 75 with a mid-epoch
#     latest.pt -> --resume auto finishes it; one history row is marked resumed
#  4. preemption drill, after the first epoch: same, resuming in epoch 1
#  5. bf16 autocast on CPU for R1 and R2 (the precision path the GPU runs use). CPU bf16
#     operator coverage differs from CUDA, so this may fail without failing the rehearsal;
#     its outcome is printed as "BF16-CPU: PASS" or "BF16-CPU: FAIL <error>".
#
# Everything goes to $RUNS (wiped at the start); the real runs/ directory is never touched.
# For a quicker local rehearsal the grid can be narrowed, e.g.
#   SMOKE_DATASETS=massachusetts SMOKE_NAMES="R0 R1" SMOKE_SEEDS=0 SMOKE_TILE=128 bash scripts/sbatch_smoke_cpu.sh

set -euo pipefail

PROJECT=/scratch/hk4488/roadseg
PY="${ROADSEG_PY:-/scratch/hk4488/.conda/envs/unsloth_sft/bin/python}"
RUNS="${ROADSEG_RUNS_SMOKE:-$PROJECT/runs_smoke}"
read -r -a DATASETS <<<"${SMOKE_DATASETS:-massachusetts deepglobe}"
read -r -a NAMES <<<"${SMOKE_NAMES:-R0 R1 R2 R3 unet dlinknet}"
read -r -a SEEDS <<<"${SMOKE_SEEDS:-0 1}"
TILE="${SMOKE_TILE:-256}"
STRIDE=$(( TILE / 2 ))
WORKERS="${SMOKE_WORKERS:-2}"

export PYTHONPATH="$PROJECT/.deps:$PROJECT/src"
export TORCH_HOME=/scratch/hk4488/.cache/torch
export OMP_NUM_THREADS="${SMOKE_THREADS:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }
fail() { echo "SMOKE: FAILED - $*" >&2; exit 1; }
select_flag() { if [[ "$1" == R[0-9] ]]; then echo "--rung $1"; else echo "--model $1"; fi; }

# wait_for <seconds> <pid> <command...>: poll until the command succeeds; fail if the
# process dies or the time runs out first.
wait_for() {
    local deadline=$(( SECONDS + $1 )) pid=$2
    shift 2
    until "$@"; do
        kill -0 "$pid" 2>/dev/null || return 1
        (( SECONDS < deadline )) || return 1
        sleep 0.2
    done
}

[[ "$(basename "$RUNS")" == *smoke* ]] || fail "refusing to wipe '$RUNS': not a smoke directory"
cd "$PROJECT"
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  runs $RUNS  $(date)"
echo "grid: datasets=(${DATASETS[*]}) names=(${NAMES[*]}) seeds=(${SEEDS[*]}) tile=$TILE"
rm -rf "$RUNS"
mkdir -p "$RUNS"

SMALL=(--limit-train 8 --limit-val 2 --batch-size 2 --workers "$WORKERS" --tile "$TILE" --device cpu)
SHORT=(--epochs 2 --max-train-batches 3)

# ---- 1. the grid ---------------------------------------------------------------------------
for dataset in "${DATASETS[@]}"; do
    for name in "${NAMES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run="$RUNS/${dataset}_${name}_s${seed}"
            # shellcheck disable=SC2046
            train=("$PY" -m roadseg.train --config "configs/${dataset}.yaml" $(select_flag "$name")
                   --seed "$seed" "${SHORT[@]}" "${SMALL[@]}" --no-amp --runs-dir "$RUNS")
            score=("$PY" -m roadseg.eval --run "$run" --split test --tile "$TILE"
                   --stride "$STRIDE" --limit 3 --workers "$WORKERS")

            step "train $run"
            "${train[@]}"
            for f in metrics.json history.json best.pt latest.pt; do
                [[ -s "$run/$f" ]] || fail "$run/$f is missing after training"
            done

            step "score the test split of $run"
            "${score[@]}"
            [[ -s "$run/test_metrics.json" ]] || fail "$run/test_metrics.json is missing"

            step "train again: must be a no-op"
            out=$("${train[@]}")
            echo "$out"
            grep -q "already complete" <<<"$out" || fail "second training did not say 'already complete'"

            step "score again: must refuse"
            rc=0
            "${score[@]}" || rc=$?
            (( rc == 2 )) || fail "second test scoring exited $rc; expected the once-only refusal (2)"
        done
    done
done

# ---- 2. aggregation --------------------------------------------------------------------------
for dataset in "${DATASETS[@]}"; do
    step "aggregate $dataset"
    "$PY" -m roadseg.stats --runs-dir "$RUNS" --dataset "$dataset" --split test --allow-nonstandard
    [[ -s "$RUNS/${dataset}_test_summary.json" ]] || fail "no summary for $dataset"
done

# ---- 3 + 4. preemption drills ----------------------------------------------------------------
# ckpt_minutes 0 writes latest.pt after every optimizer step, so "a checkpoint exists" is a
# reliable sign that the job is inside an epoch.
DRILL="$RUNS/drills"
mkdir -p "$DRILL"
cat > "$DRILL/ckpt_every_step.yaml" <<EOF
_base: $PROJECT/configs/${DATASETS[0]}.yaml
ckpt_minutes: 0
EOF

# drill <label> <file whose appearance triggers the SIGTERM> <epoch the resume must land in>
drill() {
    local label=$1 trigger=$2 epoch=$3
    local dir="$DRILL/$label"
    local run="$dir/${DATASETS[0]}_R1_s0"
    local train=("$PY" -m roadseg.train --config "$DRILL/ckpt_every_step.yaml" --rung R1 --seed 0
                 --epochs 2 --max-train-batches 12 "${SMALL[@]}" --no-amp --runs-dir "$dir"
                 --resume auto)
    local rc=0 pid saved out

    step "preemption drill '$label': start, wait for $trigger, SIGTERM"
    "${train[@]}" > "$dir.log" 2>&1 &
    pid=$!
    wait_for 1800 "$pid" test -s "$run/$trigger" || { cat "$dir.log"; fail "drill $label: $trigger never appeared"; }
    kill -TERM "$pid"
    wait "$pid" || rc=$?
    cat "$dir.log"
    (( rc == 75 )) || fail "drill $label: exit $rc after SIGTERM, expected 75"
    grep -q "preempted: checkpoint saved at epoch $epoch step" "$dir.log" \
        || fail "drill $label: no 'preempted: checkpoint saved at epoch $epoch' line"
    [[ ! -e "$run/metrics.json" ]] || fail "drill $label: metrics.json exists after a preemption"
    saved=$("$PY" -c "import sys, torch
e = torch.load(sys.argv[1], map_location='cpu', weights_only=False)['extra']
print(e['epoch'], e['step_in_epoch'])" "$run/latest.pt")
    echo "latest.pt holds (epoch, step_in_epoch) = ($saved)"
    if [[ "$label" == mid-epoch ]]; then
        (( ${saved#* } > 0 )) || fail "drill $label: step_in_epoch is not > 0"
    fi

    step "preemption drill '$label': --resume auto"
    out=$("${train[@]}")
    echo "$out"
    grep -q "resumed from .* at epoch ${saved% *} step ${saved#* }" <<<"$out" \
        || fail "drill $label: the log does not show a resume at epoch ${saved% *} step ${saved#* }"
    "$PY" - "$run" "$epoch" <<'EOF' || fail "drill $label: wrong final state"
import json, sys
from pathlib import Path
run, epoch = Path(sys.argv[1]), int(sys.argv[2])
hist = json.loads((run / "history.json").read_text())
m = json.loads((run / "metrics.json").read_text())
assert [r["epoch"] for r in hist] == [0, 1], hist
assert m["history"] == hist and m["resumes"] == 1, (m["resumes"], hist)
resumed = [r["epoch"] for r in hist if r.get("resumed")]
assert resumed in ([epoch], []), resumed       # [] only if it stopped exactly on the boundary
print(f"final state ok: 2 history rows, resumed rows {resumed}, 1 resume")
EOF
}

drill mid-epoch latest.pt 0
drill after-first-epoch history.json 1

# ---- 5. bf16 autocast on CPU (informational) ---------------------------------------------------
step "bf16 autocast on CPU (R1, R2; allowed to fail)"
bf16="PASS"
for name in R1 R2; do
    if out=$("$PY" -m roadseg.train --config configs/massachusetts.yaml --rung "$name" --seed 0 \
                 "${SHORT[@]}" "${SMALL[@]}" --runs-dir "$RUNS/bf16" 2>&1); then
        echo "$out" | tail -n 4
    else
        echo "$out"
        bf16="FAIL $name: $(tail -n 1 <<<"$out")"
        break
    fi
done
echo "BF16-CPU: $bf16"

echo
echo "SMOKE: ALL PASSED  $(date)"
