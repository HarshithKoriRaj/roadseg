#!/bin/bash
set -euo pipefail
PROJECT=/scratch/hk4488/roadseg
PY=/scratch/hk4488/.conda/envs/unsloth_sft/bin/python
export PYTHONPATH="$PROJECT/.deps:$PROJECT/src:${PYTHONPATH:-}"
export TORCH_HOME=/scratch/hk4488/.cache/torch
# Shared nodes: letting torch spawn one thread per core oversubscribes badly
# (a 46 s test file took 13 min on a loaded login node). Cap it; override if needed.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
exec "$PY" -m pytest "$@"
