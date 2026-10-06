#!/usr/bin/env bash
#SBATCH -J gs-h1d-rd4-1g
#SBATCH --nodes=1
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH --array=0-10%4
#SBATCH -w ariel-v10
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-h1d-rd4-1g-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-h1d-rd4-1g-%A_%a.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
export REPO_DIR
cd "${REPO_DIR}"
source "${REPO_DIR}/scripts/slurm/hyper1d_rd4_common.sh"
rd_one_task "${SLURM_ARRAY_TASK_ID:?submit as an array}"
rd_variant_config "${RD_VARIANT}"
RD_ONE_JOB_ID="${SLURM_ARRAY_JOB_ID:?missing array job ID}"

vanilla="${VANILLA_CHECKPOINT:-${REPO_DIR}/checkpoints/pretrained/globalsplat-re10k-32k.ckpt}"
if [[ ! -f "${vanilla}" ]]; then
  echo "Missing vanilla checkpoint: ${vanilla}" >&2
  exit 2
fi
tag="$(rd_lambda_tag "${RD_LAMBDA}")"
output="${REPO_DIR}/outputs/hyper1d_rd4_50k/${RD_VARIANT}/lambda${tag}/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
checkpoint="${output}/checkpoints/hyper1d_12h/version_0/step000050000.ckpt"
manifest="$(rd_manifest_path "${RD_VARIANT}" "${RD_LAMBDA}")"

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

morton_arg=()
if [[ "${RD_MORTON}" == on ]]; then morton_arg=(--morton); fi
printf '[RD4 train] variant=%s lambda=%s gpu=1 output=%s\n' "${RD_VARIANT}" "${RD_LAMBDA}" "${output}"
srun --ntasks=1 python scripts/run_hyper1d.py train \
  --vanilla-checkpoint "${vanilla}" --architecture "${RD_ARCH}" \
  --paths "${RD_PATHS}" --strides 4x "${morton_arg[@]}" \
  --lambda "${RD_LAMBDA}" --batch-size 2 --accumulate 4 \
  --output "${output}" --no-time-limit --max-steps 50000 \
  --warmup-steps 1000 --checkpoint-every 500 --validate-every 2000

if [[ ! -f "${checkpoint}" ]]; then
  echo "Training ended without step-50000 checkpoint: ${checkpoint}" >&2
  exit 2
fi
mkdir -p "$(dirname "${manifest}")"
printf '%s\n' "${checkpoint}" > "${manifest}.tmp.${SLURM_JOB_ID}"
mv "${manifest}.tmp.${SLURM_JOB_ID}" "${manifest}"
printf '[RD4 ready] %s\n' "${checkpoint}"
