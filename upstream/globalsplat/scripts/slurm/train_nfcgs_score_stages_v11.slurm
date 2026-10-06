#!/usr/bin/env bash
#SBATCH -J gs-score-stages
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v11
#SBATCH --gres=gpu:normal:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 4-00:00:00
#SBATCH --array=0-2%1
#SBATCH -o logs/slurm/slurm-gs-score-stages-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-score-stages-%A_%a.err

# Matched 2/3/4-stage comparison: kernel 5 + quantized STE context in every arm.
# MSH stays enabled. Four GPUs * batch 2 * accumulation 1 = effective batch 8.
set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

RATE_LAMBDA="${RATE_LAMBDA:-0.0256}"
MAX_STEPS="${MAX_STEPS:-50000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
case "${RATE_LAMBDA}" in
  0.0064|0.0256) ;;
  *) echo "RATE_LAMBDA must be 0.0064 or 0.0256" >&2; exit 2 ;;
esac
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_STEPS must be positive" >&2; exit 2; }
[[ "${CHECKPOINT_EVERY}" =~ ^[1-9][0-9]*$ ]] || { echo "CHECKPOINT_EVERY must be positive" >&2; exit 2; }
(( MAX_STEPS % CHECKPOINT_EVERY == 0 )) || { echo "MAX_STEPS must be divisible by CHECKPOINT_EVERY" >&2; exit 2; }
case "${SLURM_ARRAY_TASK_ID:?submit as an array}" in
  0) STAGES=2 ;;
  1) STAGES=3 ;;
  2) STAGES=4 ;;
  *) echo "invalid array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
TAG="${RATE_LAMBDA/./p}"
SOURCE="${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda${TAG}/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda${TAG}_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt"
[[ -s "${SOURCE}" ]] || { echo "missing source checkpoint: ${SOURCE}" >&2; exit 1; }
OUTPUT="${REPO_DIR}/outputs/nfcgs_score_stages/lambda${TAG}/job_${SLURM_ARRAY_JOB_ID:?submit with sbatch}/stages${STAGES}"
echo "SOURCE_CHECKPOINT=${SOURCE}"
echo "STAGES=${STAGES} KERNEL=5 CONTEXT_QUANTIZATION=ste SCORE_PATH=minimal"
echo "TRAIN_OUTPUT=${OUTPUT} RATE_LAMBDA=${RATE_LAMBDA} MAX_STEPS=${MAX_STEPS}"
echo "GPUS=4 BATCH_PER_GPU=2 ACCUMULATE=1 EFFECTIVE_BATCH=8 LR=0.0001"
exec python scripts/run_nfcgs.py train --devices 4 --launcher srun \
  --checkpoint "${SOURCE}" --rate-lambda "${RATE_LAMBDA}" \
  --score-path minimal --allow-score-path-conversion \
  --score-spatial-stages "${STAGES}" --score-spatial-kernel 5 \
  --score-context-quantization ste \
  --scope all --lr 0.0001 --batch-size 2 --accumulate 1 --workers 8 \
  --max-steps "${MAX_STEPS}" --checkpoint-every "${CHECKPOINT_EVERY}" \
  --output "${OUTPUT}"
