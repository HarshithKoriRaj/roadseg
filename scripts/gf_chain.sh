#!/bin/bash
# Geometry feedback, end to end without a human in the loop (sessions drop):
#
#   bash scripts/gf_chain.sh            # launch campaign gfsel, then queue the selector job
#   bash scripts/gf_chain.sh --select   # (the selector job itself) pick on validation, launch gfconf_*
#
# Step 1 runs scripts/launch.sh --campaign gfsel (all of launch.sh's checks apply) and submits a
# selector job that starts only when EVERY gfsel task succeeded (afterok). The selector runs
# scripts/analysis/gf_select.py -- validation IoU only, rule fixed in stats._gf_plan -- and then
# scripts/launch.sh --campaign gfconf_a|gfconf_b. HEAD must still be the rehearsed commit then:
# nothing may be committed until the confirmation campaign has launched.
set -euo pipefail
PROJECT=/scratch/hk4488/roadseg
PY=/scratch/hk4488/.conda/envs/unsloth_sft/bin/python
LOGS=/scratch/hk4488/logs/roadseg
cd "$PROJECT"

if [[ "${1:-}" == "--select" ]]; then
    out=$(PYTHONPATH="$PROJECT/.deps:$PROJECT/src" "$PY" scripts/analysis/gf_select.py --runs-dir runs)
    echo "$out"
    campaign=$(awk '/^campaign /{print $2}' <<<"$out")
    [[ "$campaign" == gfconf_a || "$campaign" == gfconf_b ]] || { echo "no campaign selected" >&2; exit 1; }
    bash scripts/launch.sh --campaign "$campaign"
    exit 0
fi

out=$(bash scripts/launch.sh --campaign gfsel)
echo "$out"
array=$(sed -n 's/^array job: \([0-9]*\).*/\1/p' <<<"$out")
[[ "$array" =~ ^[0-9]+$ ]] || { echo "gfsel was not launched" >&2; exit 1; }
sel=$(sbatch --parsable --job-name=gf-select --partition=cpu_short --account=torch_pr_627_general \
      --cpus-per-task=1 --mem=4G --time=00:20:00 --dependency="afterok:$array" --kill-on-invalid-dep=yes \
      --output="$LOGS/gf-select_%j.out" --wrap "bash $PROJECT/scripts/gf_chain.sh --select")
echo "selector job: $sel (afterok:$array)"
echo "$(date +%Y-%m-%dT%H:%M:%S%z) gf_chain: gfsel array=$array selector=$sel" >> runs/LAUNCHES.log
