#!/usr/bin/env bash
#SBATCH -J gs-a-validation
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-arch-eval-%j.out
#SBATCH -e logs/slurm/slurm-gs-arch-eval-%j.err
set -euo pipefail
cd "${REPO_DIR:?}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN="${PILOT_OUTPUT_ROOT:?}/${PILOT_ARM:?}/job_${TRAIN_JOB_ID:?}"
exec srun --ntasks=1 python scripts/run_nfcgs_architecture_pilot.py "${PILOT_EVAL_MODE:?}" \
  --arm "${PILOT_ARM}" --preparation "${PREPARATION_DIR:?}" --dataset-root "${DATASET_ROOT:?}" \
  --checkpoint "${RUN}/checkpoints/nfcgs_main/version_0/step000050000.ckpt" \
  --output "${RUN}/eval_${PILOT_EVAL_MODE}"
