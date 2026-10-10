#!/bin/bash
set -euo pipefail
PROJECT=/scratch/hk4488/roadseg
PY=/scratch/hk4488/.conda/envs/unsloth_sft/bin/python
mkdir -p "$PROJECT/.deps"
"$PY" -m pip install --no-cache-dir --target "$PROJECT/.deps" \
    "scikit-image>=0.24" "opencv-python-headless>=4.10" \
    "albumentations>=1.4" "pytest>=8.0"

# .deps is first on PYTHONPATH, so any transitive copies of these core
# packages here would shadow the conda env's own (the single source of truth).
rm -rf "$PROJECT/.deps"/{numpy,numpy-*.dist-info,numpy.libs,scipy,scipy-*.dist-info,scipy.libs,PIL,pillow-*.dist-info,pillow.libs,networkx,networkx-*.dist-info,yaml,_yaml,pyyaml-*.dist-info}

echo "deps installed to $PROJECT/.deps"
