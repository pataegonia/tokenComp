#!/usr/bin/env bash
#SBATCH -J gs-hyper1d-plain4
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v9
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-hyper1d-plain4-%j.out
#SBATCH -e logs/slurm/slurm-gs-hyper1d-plain4-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
export REPO_DIR
output="${REPO_DIR}/outputs/hyper1d_plain4/job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
exec bash "${REPO_DIR}/scripts/slurm/train_hyper1d_12h.slurm" \
  --architecture plain4 --output "${output}" \
  --no-time-limit --max-steps 50000 \
  --checkpoint-every 500 --validate-every 2000 "$@"
