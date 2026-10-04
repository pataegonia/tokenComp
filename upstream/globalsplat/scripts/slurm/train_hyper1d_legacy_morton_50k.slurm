#!/usr/bin/env bash
#SBATCH -J gs-hyper1d-morton50k
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v9
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-hyper1d-morton50k-%j.out
#SBATCH -e logs/slurm/slurm-gs-hyper1d-morton50k-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
export REPO_DIR

# Legacy here means the original one-path Hyper1D with adapters (N=192/M=320).
# The old Nonlinear32 score+residual two-path baseline was already Morton ON.
# Initialize a new codec so this is a 50K training run with Morton enabled
# throughout, rather than changing token order only after training.
vanilla="${VANILLA_CHECKPOINT:-${REPO_DIR}/checkpoints/pretrained/globalsplat-re10k-32k.ckpt}"
output="${REPO_DIR}/outputs/hyper1d_legacy_morton_50k/lambda0p0256/job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

exec bash "${REPO_DIR}/scripts/slurm/train_hyper1d_12h.slurm" \
  --vanilla-checkpoint "${vanilla}" \
  --architecture legacy --strides 4x --morton --lambda 0.0256 \
  --output "${output}" --no-time-limit --max-steps 50000 \
  --warmup-steps 1000 --checkpoint-every 500 --validate-every 2000 "$@"
