#!/usr/bin/env bash
#SBATCH -J gs-h1d-rd4-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH --array=0-27%4
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm/slurm-gs-h1d-rd4-eval-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-h1d-rd4-eval-%A_%a.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
export REPO_DIR
cd "${REPO_DIR}"
source "${REPO_DIR}/scripts/slurm/hyper1d_rd4_common.sh"
rd_eval_task "${SLURM_ARRAY_TASK_ID:?submit as an array}"
tag="$(rd_lambda_tag "${RD_LAMBDA}")"

checkpoint="$(rd_fixed_checkpoint "${RD_VARIANT}" "${RD_LAMBDA}")"
if [[ -z "${checkpoint}" ]]; then
  manifest="$(rd_manifest_path "${RD_VARIANT}" "${RD_LAMBDA}")"
  if [[ ! -f "${manifest}" ]]; then
    echo "Missing trained checkpoint manifest: ${manifest}" >&2
    exit 2
  fi
  IFS= read -r checkpoint < "${manifest}"
fi
if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing evaluation checkpoint: ${checkpoint}" >&2
  exit 2
fi
output="${REPO_DIR}/outputs/hyper1d_rd4_50k/full_test/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/${RD_VARIANT}/lambda${tag}"

export TMPDIR="${GLOBALSPLAT_TMP_ROOT:-/ceph_data/clue9986/tmp}/gs_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

printf '[RD4 full test] variant=%s lambda=%s checkpoint=%s output=%s\n' \
  "${RD_VARIANT}" "${RD_LAMBDA}" "${checkpoint}" "${output}"
srun --ntasks=1 python scripts/run_hyper1d.py eval \
  --checkpoint "${checkpoint}" --lambda "${RD_LAMBDA}" \
  --all-test --workers 0 --output "${output}"

summary="${output}/evaluation/hyper1d_12h/scores_all_avg.json"
if [[ ! -f "${summary}" ]]; then
  echo "Evaluation ended without summary: ${summary}" >&2
  exit 2
fi
printf '[RD4 evaluated] %s\n' "${summary}"
