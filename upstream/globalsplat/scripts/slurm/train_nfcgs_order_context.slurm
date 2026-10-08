#!/usr/bin/env bash
#SBATCH -J gs-order-context
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 4-00:00:00
#SBATCH -o logs/slurm/slurm-gs-order-context-%j.out
#SBATCH -e logs/slurm/slurm-gs-order-context-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source scripts/slurm/nfcgs_order_context_settings.sh
: "${SOURCE_CHECKPOINT:?submit with submit_nfcgs_order_context.sh or set SOURCE_CHECKPOINT}"
[[ -s "${SOURCE_CHECKPOINT}" ]] || { echo "missing checkpoint: ${SOURCE_CHECKPOINT}" >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
OUTPUT="${RUN_ROOT}/job_${SLURM_JOB_ID:?submit with sbatch}"
echo "TOKEN_ORDER=${TOKEN_ORDER} CONTEXT_SCHEDULE=${CONTEXT_SCHEDULE} STAGES=${STAGES} KERNEL=${KERNEL}"
echo "SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT} TRAIN_OUTPUT=${OUTPUT} MAX_STEPS=${MAX_STEPS}"
echo "GPUS=2 BATCH_PER_GPU=2 ACCUMULATE=2 EFFECTIVE_BATCH=8 LR=0.0001 MSH=on"
exec python scripts/run_nfcgs.py train --devices 2 --launcher srun \
  --checkpoint "${SOURCE_CHECKPOINT}" --rate-lambda "${RATE_LAMBDA}" \
  "${CODEC_ARGS[@]}" --allow-score-path-conversion \
  --scope all --lr 0.0001 --batch-size 2 --accumulate 2 --workers 8 \
  --max-steps "${MAX_STEPS}" --checkpoint-every "${CHECKPOINT_EVERY}" --output "${OUTPUT}"
