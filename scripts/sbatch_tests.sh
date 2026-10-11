#!/bin/bash
#SBATCH --job-name=roadseg-tests
#SBATCH --output=/scratch/hk4488/logs/roadseg/%x_%j.out
#SBATCH --error=/scratch/hk4488/logs/roadseg/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_627_general

# Full test suite on a compute node. The login node kills heavy processes
# (the suite builds many ResNeXt-50s and reads real imagery), so run it here.
set -euo pipefail
cd /scratch/hk4488/roadseg
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
export OMP_NUM_THREADS=8
bash scripts/pytest.sh -q -p no:cacheprovider "$@"
echo "TESTS: ALL PASSED  $(date)"
