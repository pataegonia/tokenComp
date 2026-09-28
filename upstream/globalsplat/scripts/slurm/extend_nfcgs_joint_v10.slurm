#!/usr/bin/env bash
#SBATCH -J gs-joint-ext-v10
#SBATCH --nodes=1
#SBATCH -w ariel-v10
#SBATCH --gres=gpu:normal:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-joint-ext-v10-%j.out
#SBATCH -e logs/slurm/slurm-gs-joint-ext-v10-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

: "${PARENT_JOB_ID:?submit with scripts/slurm/submit_nfcgs_joint_extension_v10.sh}"
RATE_LAMBDA="${RATE_LAMBDA:-0.0064}"
EXTEND_FROM_STEP="${EXTEND_FROM_STEP:-220000}"
TARGET_MAX_STEPS="${TARGET_MAX_STEPS:-500000}"
EXTENSION_WARMUP_STEPS="${EXTENSION_WARMUP_STEPS:-10000}"
EXTENSION_START_LR="${EXTENSION_START_LR:-0.00001}"
EXTENSION_PEAK_LR="${EXTENSION_PEAK_LR:-0.0001}"
EXTENSION_FINAL_LR="${EXTENSION_FINAL_LR:-0.00001}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
PARENT_LOG="${PARENT_LOG:-${REPO_DIR}/logs/slurm/slurm-gs-joint-v10-${PARENT_JOB_ID}.out}"

if [[ -n "${PARENT_CHECKPOINT:-}" ]]; then
  SOURCE_CHECKPOINT="${PARENT_CHECKPOINT}"
else
  SOURCE_CHECKPOINT="$(python scripts/resolve_nfcgs_slurm_checkpoint.py "${PARENT_LOG}")"
fi

TMP_ROOT="${SLURM_TMPDIR:-${REPO_DIR}/outputs/.nfcgs_tmp}"
mkdir -p "${TMP_ROOT}"
PATCHED_CHECKPOINT="${TMP_ROOT}/nfcgs-extension-${SLURM_JOB_ID}.ckpt"
trap 'rm -f -- "${PATCHED_CHECKPOINT}"' EXIT

python scripts/prepare_nfcgs_extension_checkpoint.py \
  "${SOURCE_CHECKPOINT}" "${PATCHED_CHECKPOINT}" \
  --expected-start-step "${EXTEND_FROM_STEP}" \
  --target-step "${TARGET_MAX_STEPS}" \
  --warmup-steps "${EXTENSION_WARMUP_STEPS}" \
  --start-lr "${EXTENSION_START_LR}" \
  --peak-lr "${EXTENSION_PEAK_LR}" \
  --final-lr "${EXTENSION_FINAL_LR}"

LAMBDA_TAG="${RATE_LAMBDA/./p}"
EXTENSION_OUTPUT="${EXTENSION_OUTPUT:-${REPO_DIR}/outputs/nfcgs_main/joint_extension/lambda${LAMBDA_TAG}/from_job_${PARENT_JOB_ID}_to_${TARGET_MAX_STEPS}}"
echo "PARENT_CHECKPOINT=${SOURCE_CHECKPOINT}"
echo "EXTENSION_OUTPUT=${EXTENSION_OUTPUT}"

python scripts/run_nfcgs.py train --joint --devices 8 --launcher srun \
  --checkpoint "${PATCHED_CHECKPOINT}" --resume \
  --rate-lambda "${RATE_LAMBDA}" --lr "${EXTENSION_PEAK_LR}" \
  --max-steps "${TARGET_MAX_STEPS}" --checkpoint-every "${CHECKPOINT_EVERY}" \
  --batch-size 1 --accumulate 1 --workers 8 --output "${EXTENSION_OUTPUT}"
