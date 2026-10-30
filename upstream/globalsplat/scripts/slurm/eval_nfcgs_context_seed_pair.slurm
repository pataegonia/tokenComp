#!/usr/bin/env bash
#SBATCH -J gs-seed-pair-eval
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-seed-pair-eval-%j.out
#SBATCH -e logs/slurm/slurm-gs-seed-pair-eval-%j.err
set -euo pipefail
cd "${REPO_DIR:?}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=8
RUN="${CONTEXT_SEED_OUTPUT_ROOT:?}/${CONTEXT_SEED_ARM:?}/job_${TRAIN_JOB_ID:?}"
exec srun --ntasks=1 python scripts/run_nfcgs_context_seed_pair.py eval --arm "${CONTEXT_SEED_ARM}" \
  --dataset-root "${DATASET_ROOT:?}" --run-dir "${RUN}"
