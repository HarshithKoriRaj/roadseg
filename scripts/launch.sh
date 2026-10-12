#!/bin/bash
# The ONE way to submit the experiment. Never `sbatch scripts/sbatch_train_array.sh` by hand:
# a hand-submitted task imports whatever is in the live repository each time it is requeued.
#
#   bash scripts/launch.sh --dry-run     # show what would happen; creates and submits nothing
#   bash scripts/launch.sh --canary      # GPU canary only (scripts/sbatch_canary.sh, ~40 min)
#   bash scripts/launch.sh               # the 36-task array + the aggregation job behind it
#
# What it does:
#   1. refuses a dirty tree (`git status --porcelain` non-empty): what runs must be a commit;
#   2. refuses unless the CPU rehearsal passed AT THIS COMMIT: runs_smoke/SMOKE_PASSED, written
#      by scripts/sbatch_smoke_cpu.sh, must hold HEAD's sha (--no-smoke-check overrides, loudly);
#   3. freezes the code: releases/<short sha>/ = `git archive HEAD`, plus a COMMIT file with
#      the full sha (reused if it already exists; read-only);
#   4. submits FROM the snapshot with --export=ALL,ROADSEG_CODE=<snapshot>, so every task and
#      every requeue of it imports the same code and records it (train.py refuses to resume a
#      checkpoint with different code). ROADSEG_PY / ROADSEG_RUNS / ROADSEG_LOGS are unset
#      first: those are test overrides and must not leak into the real jobs;
#   5. prints the job ids and appends them to runs/LAUNCHES.log.
#
# Data, .deps, runs/ and the logs stay under the project directory; only code and configs
# come from the snapshot.
#
# Overridable for tests: ROADSEG_PROJECT (the repository), ROADSEG_RELEASES (snapshot root),
# ROADSEG_SBATCH (the sbatch binary), ROADSEG_SMOKE_FLAG, ROADSEG_LAUNCH_LOG, ROADSEG_SLURM_LOGS.

set -euo pipefail

PROJECT="${ROADSEG_PROJECT:-/scratch/hk4488/roadseg}"
RELEASES="${ROADSEG_RELEASES:-$PROJECT/releases}"
SBATCH="${ROADSEG_SBATCH:-sbatch}"
SMOKE_FLAG="${ROADSEG_SMOKE_FLAG:-$PROJECT/runs_smoke/SMOKE_PASSED}"
LAUNCH_LOG="${ROADSEG_LAUNCH_LOG:-$PROJECT/runs/LAUNCHES.log}"
SLURM_LOGS="${ROADSEG_SLURM_LOGS:-/scratch/hk4488/logs/roadseg}"

dry=0
canary=0
smoke_check=1
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry=1 ;;
        --canary) canary=1 ;;
        --no-smoke-check) smoke_check=0 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "launch.sh: unknown argument '$arg' (see --help)" >&2; exit 2 ;;
    esac
done

refuse() { echo "launch.sh: REFUSED - $*" >&2; exit 2; }

# ---- 1. a commit, not a working tree ---------------------------------------------------------
sha=$(git -C "$PROJECT" rev-parse HEAD)
short=$(git -C "$PROJECT" rev-parse --short HEAD)
changes=$(git -C "$PROJECT" status --porcelain)
if [[ -n "$changes" ]]; then
    echo "$changes" >&2
    refuse "the tree at $PROJECT is dirty (above). Commit or remove the changes: jobs run a commit."
fi

# ---- 2. the CPU rehearsal passed at this commit ----------------------------------------------
if (( smoke_check )); then
    [[ -f "$SMOKE_FLAG" ]] || refuse "no $SMOKE_FLAG: the CPU rehearsal has not passed. Run" \
        "'sbatch scripts/sbatch_smoke_cpu.sh' at this commit ($short) and wait for 'SMOKE: ALL PASSED'" \
        "(--no-smoke-check overrides)."
    smoked=$(tr -d '[:space:]' < "$SMOKE_FLAG")
    [[ "$smoked" == "$sha" ]] || refuse "the CPU rehearsal passed at ${smoked:-<empty>}, but HEAD" \
        "is $sha. Rehearse this commit first (--no-smoke-check overrides)."
    echo "smoke: rehearsal passed at $short"
else
    echo "launch.sh: WARNING --no-smoke-check: submitting $short WITHOUT a passed CPU rehearsal" \
         "at this commit. Every failure the rehearsal would have caught now costs GPU queue time." >&2
fi

# ---- 3. the frozen snapshot ------------------------------------------------------------------
SNAP="$RELEASES/$short"
if [[ -e "$SNAP" ]]; then
    have=$(cat "$SNAP/COMMIT" 2>/dev/null || true)
    [[ "$have" == "$sha" ]] || refuse "$SNAP exists but its COMMIT is '${have:-<missing>}'," \
        "not $sha. Remove it and launch again."
    echo "snapshot: $SNAP (already there, commit $sha)"
elif (( dry )); then
    echo "snapshot: $SNAP (would be created: git archive $sha + COMMIT)"
else
    tmp="$RELEASES/.$short.$$.tmp"
    mkdir -p "$tmp"
    if ! { git -C "$PROJECT" archive "$sha" | tar -x -C "$tmp"; }; then
        rm -rf "$tmp"
        echo "launch.sh: git archive $sha failed; nothing was submitted" >&2
        exit 1
    fi
    echo "$sha" > "$tmp/COMMIT"
    chmod -R a-w "$tmp"                 # frozen: nobody edits what the jobs import
    chmod u+w "$tmp"                    # (the directory itself must stay renamable)
    mv "$tmp" "$SNAP"
    chmod a-w "$SNAP"
    echo "snapshot: $SNAP (created, commit $sha)"
fi

# ---- 4. submit from the snapshot -------------------------------------------------------------
unset ROADSEG_PY ROADSEG_RUNS ROADSEG_LOGS ROADSEG_RUNS_SMOKE ROADSEG_RUNS_CANARY
EXPORT="--export=ALL,ROADSEG_CODE=$SNAP"
stamp=$(date +%Y-%m-%dT%H:%M:%S%z)

# submit <label> <sbatch args...>  -> JOB = the job id (a placeholder in a dry run)
submit() {
    local label=$1 out
    shift
    if (( dry )); then
        echo "would run: $SBATCH --parsable $*"
        JOB="<$label job id>"
        return
    fi
    out=$("$SBATCH" --parsable "$@")
    JOB="${out%%;*}"
    [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "launch.sh: sbatch answered '$out', not a job id" >&2; exit 1; }
}

if (( dry )); then
    echo "DRY RUN: nothing is created and nothing is submitted"
else
    mkdir -p "$SLURM_LOGS" "$(dirname "$LAUNCH_LOG")"       # SLURM does not create the log dir
fi

if (( canary )); then
    submit canary "$EXPORT" "$SNAP/scripts/sbatch_canary.sh"
    jid=$JOB
    echo "canary job: $jid   (log: $SLURM_LOGS/roadseg-canary_$jid.out; ends with 'CANARY: ALL PASSED')"
    line="$stamp sha=$sha snapshot=$SNAP canary=$jid"
else
    submit array "$EXPORT" "$SNAP/scripts/sbatch_train_array.sh"
    jid=$JOB
    submit aggregate "--dependency=afterany:$jid" "$EXPORT" "$SNAP/scripts/sbatch_aggregate.sh"
    agg=$JOB
    echo "array job: $jid   (36 tasks, 16 at a time; logs: $SLURM_LOGS/roadseg_${jid}_<task>.out)"
    echo "aggregate job: $agg   (runs after every task of $jid has ended)"
    line="$stamp sha=$sha snapshot=$SNAP array=$jid aggregate=$agg"
fi
echo "code: $SNAP (commit $sha)"
if (( ! dry )); then
    echo "$line" >> "$LAUNCH_LOG"
    echo "recorded in $LAUNCH_LOG"
fi
