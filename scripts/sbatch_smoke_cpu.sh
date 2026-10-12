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
# images, tile 256). Submit it through the launcher, which freezes HEAD into a snapshot and
# rehearses THAT (all code -- src and configs -- comes from $ROADSEG_CODE), so the working
# tree may be edited while it runs:
#
#   bash scripts/launch.sh --smoke            # ends with "SMOKE: ALL PASSED" or stops at the failure
#
# (Run by hand without ROADSEG_CODE it rehearses the live tree: then the tree must be clean
# and must not change until it ends, or no acknowledgement is written.)
#
# On success it writes $RUNS/SMOKE_PASSED holding the commit it rehearsed; scripts/launch.sh
# refuses to submit any other commit. The acknowledgement is written only by the full default
# rehearsal of an unchanged, committed tree -- not by a narrowed one, not if the code moved.
#
#  1. grid: every dataset x (R0 R1 R2 R3 unet dlinknet) x seeds (0 1): preflight + train ->
#     score the test split -> train again (must say "already complete") -> score again (must
#     refuse, exit 2)
#  2. aggregate: both datasets (--allow-nonstandard: these scores are not the protocol)
#  3. drills: SIGTERM the whole process group of a running job (python and its DataLoader
#     workers, as a job-wide SLURM signal arrives) -> exit 75, no process left behind ->
#     --resume auto finishes it. Once mid-epoch (one history row is marked resumed), once
#     after the first epoch (resuming in epoch 1).
#  4. bf16: bf16 autocast on CPU for R1 and R2 (the precision path the GPU runs use). CPU bf16
#     operator coverage differs from CUDA, so this may fail without failing the rehearsal;
#     its outcome is printed as "BF16-CPU: PASS" or "BF16-CPU: FAIL <error>".
#
# Everything goes to $RUNS (wiped at the start); the real runs/ directory is never touched.
# For a quicker local rehearsal the grid can be narrowed (no acknowledgement is written), e.g.
#   SMOKE_DATASETS=massachusetts SMOKE_NAMES="R0 R1" SMOKE_SEEDS=0 SMOKE_TILE=128 bash scripts/sbatch_smoke_cpu.sh
#   SMOKE_STEPS="drills" SMOKE_TILE=128 bash scripts/sbatch_smoke_cpu.sh     # sections: grid aggregate drills bf16

set -euo pipefail

PROJECT=/scratch/hk4488/roadseg
CODE="${ROADSEG_CODE:-$PROJECT}"        # a frozen snapshot, or (default) the live tree
PY="${ROADSEG_PY:-/scratch/hk4488/.conda/envs/unsloth_sft/bin/python}"
RUNS="${ROADSEG_RUNS_SMOKE:-$PROJECT/runs_smoke}"

# The acknowledgement is only for the full default rehearsal.
full=1
for knob in SMOKE_DATASETS SMOKE_NAMES SMOKE_SEEDS SMOKE_TILE SMOKE_STEPS; do
    [[ -z "${!knob:-}" ]] || full=0
done
read -r -a DATASETS <<<"${SMOKE_DATASETS:-massachusetts deepglobe}"
read -r -a NAMES <<<"${SMOKE_NAMES:-R0 R1 R2 R3 unet dlinknet}"
read -r -a SEEDS <<<"${SMOKE_SEEDS:-0 1}"
TILE="${SMOKE_TILE:-256}"
STRIDE=$(( TILE / 2 ))
WORKERS="${SMOKE_WORKERS:-2}"
STEPS=" ${SMOKE_STEPS:-grid aggregate drills bf16} "

export ROADSEG_CODE="$CODE"
export PYTHONPATH="$PROJECT/.deps:$CODE/src"
export TORCH_HOME=/scratch/hk4488/.cache/torch
export OMP_NUM_THREADS="${SMOKE_THREADS:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }
fail() { echo "SMOKE: FAILED - $*" >&2; exit 1; }
select_flag() { if [[ "$1" == R[0-9] ]]; then echo "--rung $1"; else echo "--model $1"; fi; }
section() { [[ "$STEPS" == *" $1 "* ]]; }

# code_sha: the commit this rehearsal runs. A frozen snapshot names it in COMMIT. In a working
# tree it is HEAD, but only while nothing differs from HEAD ("<sha>+dirty" otherwise).
code_sha() {
    local head
    if [[ -f "$CODE/COMMIT" ]]; then
        cat "$CODE/COMMIT"
        return
    fi
    head=$(git -C "$CODE" rev-parse HEAD 2>/dev/null || echo unknown)
    if [[ -n "$(git -C "$CODE" status --porcelain 2>/dev/null || echo unknown)" ]]; then
        head="$head+dirty"
    fi
    echo "$head"
}

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
[[ -d "$CODE/src/roadseg" ]] || fail "no code at ROADSEG_CODE=$CODE"
cd "$PROJECT"
SHA=$(code_sha)
echo "host $(hostname)  code $CODE  commit $SHA  runs $RUNS  $(date)"
echo "grid: datasets=(${DATASETS[*]}) names=(${NAMES[*]}) seeds=(${SEEDS[*]}) tile=$TILE sections=($STEPS)"
rm -rf "$RUNS"                          # ... and with it any earlier SMOKE_PASSED
mkdir -p "$RUNS"

SMALL=(--limit-train 8 --limit-val 2 --batch-size 2 --workers "$WORKERS" --tile "$TILE" --device cpu)
SHORT=(--epochs 2 --max-train-batches 3)

# ---- 1. the grid ---------------------------------------------------------------------------
if section grid; then
for dataset in "${DATASETS[@]}"; do
    for name in "${NAMES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run="$RUNS/${dataset}_${name}_s${seed}"
            # --preflight: the device self-test every GPU job starts with, forced on the CPU
            # shellcheck disable=SC2046
            train=("$PY" -m roadseg.train --config "$CODE/configs/${dataset}.yaml" $(select_flag "$name")
                   --seed "$seed" "${SHORT[@]}" "${SMALL[@]}" --no-amp --preflight --runs-dir "$RUNS")
            score=("$PY" -m roadseg.eval --run "$run" --split test --tile "$TILE"
                   --stride "$STRIDE" --limit 3 --workers "$WORKERS")

            step "train $run"
            out=$("${train[@]}")
            echo "$out"
            grep -q "^preflight ok:" <<<"$out" || fail "$run: no 'preflight ok' line"
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
fi

# ---- 2. aggregation --------------------------------------------------------------------------
if section aggregate; then
for dataset in "${DATASETS[@]}"; do
    step "aggregate $dataset"
    "$PY" -m roadseg.stats --runs-dir "$RUNS" --dataset "$dataset" --split test --allow-nonstandard
    [[ -s "$RUNS/${dataset}_test_summary.json" ]] || fail "no summary for $dataset"
done
fi

# ---- 3. preemption drills --------------------------------------------------------------------
# ckpt_minutes 0 writes latest.pt after every optimizer step, so "a checkpoint exists" is a
# reliable sign that the job is inside an epoch.
DRILL="$RUNS/drills"

# drill <label> <file whose appearance triggers the SIGTERM> <epoch the resume must land in>
drill() {
    local label=$1 trigger=$2 epoch=$3
    local dir="$DRILL/$label"
    local run="$dir/${DATASETS[0]}_R1_s0"
    local train=("$PY" -m roadseg.train --config "$DRILL/ckpt_every_step.yaml" --rung R1 --seed 0
                 --epochs 2 --max-train-batches 12 "${SMALL[@]}" --no-amp --runs-dir "$dir"
                 --resume auto)
    local rc=0 pid saved out

    step "preemption drill '$label': start, wait for $trigger, SIGTERM the process group"
    # setsid: its own process group, like a job. The signal then goes to python AND to its
    # DataLoader workers, exactly as SLURM delivers a job-wide SIGTERM.
    setsid "${train[@]}" > "$dir.log" 2>&1 &
    pid=$!
    wait_for 1800 "$pid" test -s "$run/$trigger" \
        || { cat "$dir.log"; kill -KILL -- "-$pid" 2>/dev/null || true; fail "drill $label: $trigger never appeared"; }
    kill -TERM -- "-$pid"
    wait "$pid" || rc=$?
    cat "$dir.log"
    (( rc == 75 )) || fail "drill $label: exit $rc after SIGTERM, expected 75"
    grep -q "preempted: checkpoint saved at epoch $epoch step" "$dir.log" \
        || fail "drill $label: no 'preempted: checkpoint saved at epoch $epoch' line"
    [[ ! -e "$run/metrics.json" ]] || fail "drill $label: metrics.json exists after a preemption"
    if pgrep -g "$pid" > /dev/null 2>&1; then
        fail "drill $label: processes of the preempted job are still alive: $(pgrep -g "$pid" | tr '\n' ' ')"
    fi
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
assert len(m["processes"]) == 2 and m["code_changes"] == [], (m["processes"], m["code_changes"])
resumed = [r["epoch"] for r in hist if r.get("resumed")]
assert resumed in ([epoch], []), resumed       # [] only if it stopped exactly on the boundary
print(f"final state ok: 2 history rows, resumed rows {resumed}, 1 resume")
EOF
}

if section drills; then
    mkdir -p "$DRILL"
    cat > "$DRILL/ckpt_every_step.yaml" <<EOF
_base: $CODE/configs/${DATASETS[0]}.yaml
ckpt_minutes: 0
EOF
    drill mid-epoch latest.pt 0
    drill after-first-epoch history.json 1
fi

# ---- 4. bf16 autocast on CPU (informational) ---------------------------------------------------
if section bf16; then
    step "bf16 autocast on CPU (R1, R2; allowed to fail)"
    bf16="PASS"
    for name in R1 R2; do
        if out=$("$PY" -m roadseg.train --config "$CODE/configs/massachusetts.yaml" --rung "$name" --seed 0 \
                     "${SHORT[@]}" "${SMALL[@]}" --runs-dir "$RUNS/bf16" 2>&1); then
            echo "$out" | tail -n 4
        else
            echo "$out"
            bf16="FAIL $name: $(tail -n 1 <<<"$out")"
            break
        fi
    done
    echo "BF16-CPU: $bf16"
fi

echo
echo "SMOKE: ALL PASSED  $(date)"

# ---- the acknowledgement scripts/launch.sh asks for -------------------------------------------
now=$(code_sha)
if (( ! full )); then
    echo "SMOKE_PASSED not written: this was a narrowed rehearsal (SMOKE_* is set);" \
         "launch.sh needs the full default one"
elif [[ "$SHA" == *+dirty || "$SHA" == unknown* ]]; then
    echo "SMOKE_PASSED not written: the rehearsal did not run a commit ($SHA)." \
         "Commit, then rehearse again."
elif [[ "$now" != "$SHA" ]]; then
    echo "SMOKE_PASSED not written: the code changed during the rehearsal ($SHA -> $now):" \
         "it tested a moving tree. Rehearse again and leave the tree alone."
else
    echo "$SHA" > "$RUNS/SMOKE_PASSED"
    echo "wrote $RUNS/SMOKE_PASSED ($SHA): scripts/launch.sh will now submit this commit"
fi
