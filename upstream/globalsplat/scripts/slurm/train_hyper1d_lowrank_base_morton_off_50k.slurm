#!/usr/bin/env bash
#SBATCH -J gs-lrbase-off50k
#SBATCH --nodes=1
#SBATCH -w ariel-v11
#SBATCH --gres=gpu:normal:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-lrbase-off50k-%j.out
#SBATCH -e logs/slurm/slurm-gs-lrbase-off50k-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
export REPO_DIR
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Match the Morton-ON run: per-GPU batch 2 x 4 GPUs x accumulation 1 = 8.
vanilla="${VANILLA_CHECKPOINT:-${REPO_DIR}/checkpoints/pretrained/globalsplat-re10k-32k.ckpt}"
output="${REPO_DIR}/outputs/hyper1d_lowrank_base_50k/rank56/morton_off/lambda0p0256/job_${SLURM_JOB_ID:-local}"
printf '[low-rank base MSH] node=ariel-v8 gpus=4 morton=off rank=56 paths=2 lambda=0.0256 steps=50000 output=%s\n' "${output}"

exec python scripts/run_hyper1d.py train \
  --vanilla-checkpoint "${vanilla}" \
  --architecture legacy --paths 2 --base-rank 56 --strides 4x \
  --lambda 0.0256 --devices 4 --launcher srun --batch-size 2 --accumulate 1 \
  --output "${output}" --no-time-limit --max-steps 50000 \
  --warmup-steps 1000 --checkpoint-every 500 --validate-every 2000 "$@"
