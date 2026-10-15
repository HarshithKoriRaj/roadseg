#!/bin/bash
#SBATCH --job-name=roadseg-canary
#SBATCH --output=/scratch/hk4488/logs/roadseg/%x_%j.out
#SBATCH --error=/scratch/hk4488/logs/roadseg/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --partition=l40s_public,h200_public
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_627_general
#
# GPU canary: the first thing to run on a GPU, before the 36-task array spends a day in the
# queue. One job, no --requeue, about 25 minutes. Submit it through the launcher, so that it
# runs the same frozen snapshot the array will:
#
#   bash scripts/launch.sh --canary       # ends with "CANARY: ALL PASSED" or stops at the failure
#
#  1. `roadseg.train --preflight-only --device cuda` for R0 R1 R2 R3 unet dlinknet with the
#     DeepGlobe configuration (batch 16, tile 512): 2 optimizer steps + 1 eval batch each on a
#     throwaway model, with the real loss, bf16 autocast and the gradient guard. Prints a table
#     of steps/s and peak GB per architecture -- does every one of them fit and run?
#  2. a real miniature run on the GPU: R1 and R3 on Massachusetts (1 epoch, real data, real
#     ImageNet weights, DataLoader workers, validation, finish phase), then the test scoring.
#  3. a preemption drill on the GPU: start R2, SIGTERM the whole process group after ~20
#     optimizer steps, expect exit 75 with a mid-run checkpoint, resume to completion.
#
# Everything goes to $RUNS (wiped at the start); the real runs/ directory is never touched.
#
# To rehearse THIS SCRIPT on a CPU node (tests/test_train_cli.py does), every size is a knob:
#   CANARY_DEVICE=cpu CANARY_NAMES="R3 unet" CANARY_TRAIN_EXTRA="--preflight --no-amp --tile 128 --batch-size 2" \
#   CANARY_MINI="--epochs 1 --max-train-batches 3 --limit-train 4 --limit-val 1 --workers 2" \
#   CANARY_EVAL_EXTRA="--tile 128 --stride 64 --workers 2" CANARY_DRILL_STEPS=2 bash scripts/sbatch_canary.sh

set -euo pipefail

PROJECT=/scratch/hk4488/roadseg
CODE="${ROADSEG_CODE:-$PROJECT}"        # the frozen snapshot of scripts/launch.sh; default: live tree
PY="${ROADSEG_PY:-/scratch/hk4488/.conda/envs/unsloth_sft/bin/python}"
RUNS="${ROADSEG_RUNS_CANARY:-$PROJECT/runs_canary}"
DEVICE="${CANARY_DEVICE:-cuda}"
DRILL_STEPS="${CANARY_DRILL_STEPS:-20}"
read -r -a NAMES <<<"${CANARY_NAMES:-R0 R1 R2 R3 unet dlinknet}"
read -r -a MINI <<<"${CANARY_MINI:---epochs 1 --max-train-batches 30 --limit-train 32 --limit-val 2 --workers 4}"
read -r -a TRAIN_EXTRA <<<"${CANARY_TRAIN_EXTRA:-}"
read -r -a EVAL_EXTRA <<<"${CANARY_EVAL_EXTRA:-}"

export ROADSEG_CODE="$CODE"
export PYTHONPATH="$PROJECT/.deps:$CODE/src"
export TORCH_HOME=/scratch/hk4488/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${CANARY_THREADS:-1}"

step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }
fail() { echo "CANARY: FAILED - $*" >&2; exit 1; }
select_flag() { if [[ "$1" == R[0-9] || "$1" == G[AB]* ]]; then echo "--rung $1"; else echo "--model $1"; fi; }

[[ "$(basename "$RUNS")" == *canary* ]] || fail "refusing to wipe '$RUNS': not a canary directory"
[[ -d "$CODE/src/roadseg" ]] || fail "no code at ROADSEG_CODE=$CODE"
commit=$(cat "$CODE/COMMIT" 2>/dev/null || git -C "$CODE" rev-parse HEAD 2>/dev/null || echo unknown)
cd "$PROJECT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-?}  code $CODE  commit $commit  runs $RUNS  $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
rm -rf "$RUNS"
mkdir -p "$RUNS"

# ---- 1. preflight every architecture at the largest configuration ----------------------------
rows=()
failed=()
for name in "${NAMES[@]}"; do
    step "preflight $name (DeepGlobe configuration)"
    rc=0
    # shellcheck disable=SC2046
    out=$("$PY" -m roadseg.train --config "$CODE/configs/deepglobe.yaml" $(select_flag "$name") \
              --seed 0 --preflight-only --device "$DEVICE" --runs-dir "$RUNS" \
              "${TRAIN_EXTRA[@]}" 2>&1) || rc=$?
    echo "$out"
    line=$(grep "^preflight ok:" <<<"$out" || true)
    if (( rc != 0 )) || [[ -z "$line" ]]; then
        rows+=("$(printf '%-10s %10s %9s   exit %s' "$name" FAILED - "$rc")")
        failed+=("$name")
        continue
    fi
    sps=$(sed -n 's/.* steps_per_s=\([0-9.]*\).*/\1/p' <<<"$line")
    gb=$(sed -n 's/.* peak_gb=\([0-9.]*\).*/\1/p' <<<"$line")
    rows+=("$(printf '%-10s %10s %9s' "$name" "$sps" "$gb")")
done
step "preflight table (batch and tile of the DeepGlobe configuration, device $DEVICE)"
printf '%-10s %10s %9s\n' name steps/s "peak GB"
printf '%s\n' "${rows[@]}"
(( ${#failed[@]} == 0 )) || fail "preflight failed for: ${failed[*]} (see $RUNS/PREFLIGHT_FAILED.*)"

# ---- 2. a real miniature run, then the test scoring ------------------------------------------
for name in R1 R3; do
    run="$RUNS/massachusetts_${name}_s0"
    step "miniature run $name on Massachusetts"
    "$PY" -m roadseg.train --config "$CODE/configs/massachusetts.yaml" --rung "$name" --seed 0 \
        --device "$DEVICE" "${MINI[@]}" --runs-dir "$RUNS" "${TRAIN_EXTRA[@]}"
    for f in metrics.json history.json best.pt latest.pt; do
        [[ -s "$run/$f" ]] || fail "$run/$f is missing after training"
    done

    step "score the test split of $name"
    "$PY" -m roadseg.eval --run "$run" --split test --limit 2 --device "$DEVICE" "${EVAL_EXTRA[@]}"
    [[ -s "$run/test_metrics.json" ]] || fail "$run/test_metrics.json is missing"

    "$PY" - "$run" "$DEVICE" "$commit" <<'EOF' || fail "$name: metrics.json does not describe this job"
import json, sys
from pathlib import Path
run, device, commit = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
m = json.loads((run / "metrics.json").read_text())
t = json.loads((run / "test_metrics.json").read_text())
assert (m["device_name"] == "cpu") == (device == "cpu"), m["device_name"]
assert m["code_commit"] == commit, (m["code_commit"], commit)
assert len(m["processes"]) == 1 and m["processes"][0]["device_name"] == m["device_name"], m["processes"]
assert m["resumes"] == 0 and m["code_changes"] == [], (m["resumes"], m["code_changes"])
assert t["eval"]["code_commit"] == commit and t["eval"]["limit"] == 2, t["eval"]
print(f"ok: {run.name} on {m['device_name']}: val IoU {m['val']['dataset_iou']:.4f}, "
      f"test IoU {t['aggregate']['dataset_iou']:.4f} (2 images), {m['wall_seconds']:.0f} s")
EOF
done

# ---- 3. preemption drill ---------------------------------------------------------------------
# ckpt_minutes 0 writes latest.pt after every optimizer step, so counting its replacements
# counts steps. The SIGTERM goes to the whole process group (python and its DataLoader
# workers), as SLURM delivers a job-wide signal.
cat > "$RUNS/ckpt_every_step.yaml" <<EOF
_base: $CODE/configs/massachusetts.yaml
ckpt_minutes: 0
EOF
drill="$RUNS/drill"
run="$drill/massachusetts_R2_s0"
train=("$PY" -m roadseg.train --config "$RUNS/ckpt_every_step.yaml" --rung R2 --seed 0
       --device "$DEVICE" "${MINI[@]}" --epochs 2 --runs-dir "$drill" --resume auto
       "${TRAIN_EXTRA[@]}")

step "preemption drill: start R2, SIGTERM the process group after ~$DRILL_STEPS steps"
mkdir -p "$drill"
setsid "${train[@]}" > "$drill.log" 2>&1 &
pid=$!
steps=0
last=""
deadline=$(( SECONDS + 900 ))
while (( steps < DRILL_STEPS )); do
    kill -0 "$pid" 2>/dev/null || { cat "$drill.log"; fail "drill: training ended before $DRILL_STEPS steps"; }
    (( SECONDS < deadline )) || { cat "$drill.log"; kill -KILL -- "-$pid" 2>/dev/null; fail "drill: no $DRILL_STEPS steps in 15 min"; }
    now=$(stat -c '%i.%Y.%s' "$run/latest.pt" 2>/dev/null || true)
    if [[ -n "$now" && "$now" != "$last" ]]; then
        steps=$(( steps + 1 ))
        last=$now
    fi
    sleep 0.05
done
kill -TERM -- "-$pid"
rc=0
wait "$pid" || rc=$?
cat "$drill.log"
(( rc == 75 )) || fail "drill: exit $rc after SIGTERM, expected 75"
grep -q "preempted: checkpoint saved at epoch" "$drill.log" || fail "drill: no 'preempted: checkpoint saved' line"
[[ ! -e "$run/metrics.json" ]] || fail "drill: metrics.json exists after a preemption"
if pgrep -g "$pid" > /dev/null 2>&1; then
    fail "drill: processes of the preempted job are still alive: $(pgrep -g "$pid" | tr '\n' ' ')"
fi

step "preemption drill: --resume auto"
out=$("${train[@]}")
echo "$out"
grep -q "resumed from .* at epoch" <<<"$out" || fail "drill: the log does not show a resume"
"$PY" - "$run" <<'EOF' || fail "drill: wrong final state"
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
hist = json.loads((run / "history.json").read_text())
m = json.loads((run / "metrics.json").read_text())
assert [r["epoch"] for r in hist] == [0, 1], hist
assert m["history"] == hist and m["resumes"] == 1, (m["resumes"], hist)
assert len(m["processes"]) == 2 and m["code_changes"] == [], (m["processes"], m["code_changes"])
print(f"final state ok: 2 history rows, resumed rows {[r['epoch'] for r in hist if r.get('resumed')]}, "
      f"2 processes, 1 resume")
EOF

echo
echo "CANARY: ALL PASSED  $(date)"
