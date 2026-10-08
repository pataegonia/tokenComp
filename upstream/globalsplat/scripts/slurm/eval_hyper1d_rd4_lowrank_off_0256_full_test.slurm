#!/usr/bin/env bash
#SBATCH -J gs-rd4-lr0256-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -x ariel-v[7]
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm/slurm-gs-rd4-lr0256-eval-%j.out
#SBATCH -e logs/slurm/slurm-gs-rd4-lr0256-eval-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"

checkpoint="${REPO_DIR}/outputs/hyper1d_rd4_50k/lowrank_off/lambda0p0256/job_442943_0/checkpoints/hyper1d_12h/version_0/step000050000.ckpt"
output="${REPO_DIR}/outputs/hyper1d_rd4_50k/full_test/lowrank_off/lambda0p0256/job_${SLURM_JOB_ID}"
if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing evaluation checkpoint: ${checkpoint}" >&2
  exit 2
fi

export TMPDIR="${GLOBALSPLAT_TMP_ROOT:-/ceph_data/clue9986/tmp}/gs_${SLURM_JOB_ID}"
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

printf '[RD4 full test] variant=lowrank_off lambda=0.0256 checkpoint=%s output=%s\n' "${checkpoint}" "${output}"
srun --ntasks=1 python scripts/run_hyper1d.py eval \
  --checkpoint "${checkpoint}" --lambda 0.0256 \
  --all-test --workers 0 --output "${output}"

summary="${output}/evaluation/hyper1d_12h/scores_all_avg.json"
if [[ ! -f "${summary}" ]]; then
  echo "Evaluation ended without summary: ${summary}" >&2
  exit 2
fi
printf '[RD4 evaluated] %s\n' "${summary}"
