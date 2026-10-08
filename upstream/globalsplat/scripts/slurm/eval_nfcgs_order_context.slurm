#!/usr/bin/env bash
#SBATCH -J gs-order-context-eval
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-order-context-eval-%j.out
#SBATCH -e logs/slurm/slurm-gs-order-context-eval-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source scripts/slurm/nfcgs_order_context_settings.sh
: "${TRAIN_JOB_ID:?submit with submit_nfcgs_order_context.sh}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
STEP_TAG="$(printf '%09d' "${MAX_STEPS}")"
CHECKPOINT="${RUN_ROOT}/job_${TRAIN_JOB_ID}/checkpoints/nfcgs_main/version_0/step${STEP_TAG}.ckpt"
OUTPUT="${RUN_ROOT}/job_${TRAIN_JOB_ID}/eval_all"
[[ -s "${CHECKPOINT}" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }
python scripts/check_nfcgs_checkpoint.py "${CHECKPOINT}" --step "${MAX_STEPS}"
echo "TOKEN_ORDER=${TOKEN_ORDER} CONTEXT_SCHEDULE=${CONTEXT_SCHEDULE} CHECKPOINT=${CHECKPOINT} EVAL_OUTPUT=${OUTPUT}"
exec srun --ntasks=1 python scripts/run_nfcgs.py eval \
  --checkpoint "${CHECKPOINT}" --rate-lambda "${RATE_LAMBDA}" "${CODEC_ARGS[@]}" --output "${OUTPUT}"
