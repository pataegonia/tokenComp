#!/usr/bin/env bash
#SBATCH -J gs-ns-resume
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-ns-resume-%j.out
#SBATCH -e logs/slurm/slurm-gs-ns-resume-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source scripts/slurm/nfcgs_order_context_settings.sh
: "${RESUME_CHECKPOINT:?submit with submit_nfcgs_nn_score_resume.sh}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
OUTPUT="${RUN_ROOT}/resume_job_${SLURM_JOB_ID:?submit with sbatch}"
COPY="${OUTPUT}/checkpoints/nfcgs_main/version_0/resume.ckpt"
python scripts/prepare_nfcgs_nn_score_resume.py "${RESUME_CHECKPOINT}" \
  --target-step "${MAX_STEPS}" --copy-to "${COPY}"
echo "TOKEN_ORDER=${TOKEN_ORDER} CONTEXT_SCHEDULE=${CONTEXT_SCHEDULE} STAGES=${STAGES} KERNEL=${KERNEL}"
echo "NN_BACKEND=${NN_ORDER_BACKEND} NN_DISTANCE_DEVICE=${NN_ORDER_DISTANCE_DEVICE} NN_CANDIDATES=${NN_ORDER_CANDIDATES}"
echo "SOURCE_CHECKPOINT=${RESUME_CHECKPOINT} TRAIN_OUTPUT=${OUTPUT} MAX_STEPS=${MAX_STEPS}"
echo "GPUS=2 BATCH_PER_GPU=2 ACCUMULATE=2 EFFECTIVE_BATCH=8 FULL_STATE_RESUME=true"
exec python scripts/run_nfcgs.py train --devices 2 --launcher srun \
  --checkpoint "${COPY}" --resume --rate-lambda "${RATE_LAMBDA}" "${CODEC_ARGS[@]}" \
  --scope all --lr 0.0001 --batch-size 2 --accumulate 2 --workers 8 \
  --max-steps "${MAX_STEPS}" --checkpoint-every "${CHECKPOINT_EVERY}" --output "${OUTPUT}"
