#!/bin/bash
set -euo pipefail
PROJECT=/scratch/hk4488/roadseg
PY=/scratch/hk4488/.conda/envs/unsloth_sft/bin/python
export PYTHONPATH="$PROJECT/.deps:$PROJECT/src:${PYTHONPATH:-}"
export TORCH_HOME=/scratch/hk4488/.cache/torch
exec "$PY" -m pytest "$@"
