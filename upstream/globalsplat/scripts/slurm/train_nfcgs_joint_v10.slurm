#!/usr/bin/env bash
#SBATCH -J gs-joint-v10
#SBATCH --nodes=1
#SBATCH -w ariel-v10
#SBATCH --gres=gpu:normal:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-joint-v10-%j.out
#SBATCH -e logs/slurm/slurm-gs-joint-v10-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# The runner performs preflight once, then starts all eight ranks through srun.
# 1 scene/GPU * 8 GPUs * accumulate 1 = effective batch 8.
exec python scripts/run_nfcgs.py train --joint --devices 8 --launcher srun \
  --batch-size 1 --accumulate 1 --workers 8 "$@"
