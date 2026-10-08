#!/usr/bin/env bash
# Submitted by scripts/resume_hyper1d_rd4.py; GPU/task counts are overridden per group.
#SBATCH -J gs-h1d-rd4-resume
#SBATCH --nodes=1
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -x ariel-v[7]
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-h1d-rd4-resume-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-h1d-rd4-resume-%A_%a.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:?missing submit directory}}"
export REPO_DIR
cd "${REPO_DIR}"
: "${RD_RESUME_PLAN:?submit using scripts/resume_hyper1d_rd4.py}"
: "${RD_RESUME_GROUP:?missing recovery group}"
: "${SLURM_ARRAY_TASK_ID:?submit as an array}"

export TMPDIR="${GLOBALSPLAT_TMP_ROOT:-/ceph_data/clue9986/tmp}/gs_r_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPDIR}"
export TMP="${TMPDIR}" TEMP="${TMPDIR}"
conda_base="${CONDA_BASE:-/ceph_data/clue9986/anaconda3}"
if command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base)"
fi
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

srun --ntasks=1 --kill-on-bad-exit=1 python scripts/resume_hyper1d_rd4.py \
  --run-task "${RD_RESUME_PLAN}" "${RD_RESUME_GROUP}" "${SLURM_ARRAY_TASK_ID}"
