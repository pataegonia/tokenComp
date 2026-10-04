#!/usr/bin/env bash
#SBATCH -J gs-score-minimal
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v11
#SBATCH --gres=gpu:normal:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 4-00:00:00
#SBATCH -o logs/slurm/slurm-gs-score-minimal-%j.out
#SBATCH -e logs/slurm/slurm-gs-score-minimal-%j.err

# Weights-only warm start from the main Split 10k checkpoint.
# GlobalSplat stays frozen; active codec parameters, including MSH, are trained.
# Effective batch stays 8: 4 GPUs * 2 scenes/GPU * accumulation 1.
set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

RATE_LAMBDA="${RATE_LAMBDA:-0.0256}"
SCORE_PATH="${SCORE_PATH:-minimal}"
MAX_STEPS="${MAX_STEPS:-50000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
case "${RATE_LAMBDA}" in
  0.0064|0.0256) ;;
  *) echo "RATE_LAMBDA must be 0.0064 or 0.0256" >&2; exit 2 ;;
esac
case "${SCORE_PATH}" in
  minimal) CONVERSION_ARGS=(--allow-score-path-conversion) ;;
  full) CONVERSION_ARGS=() ;;
  *) echo "SCORE_PATH must be minimal or full" >&2; exit 2 ;;
esac
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_STEPS must be positive" >&2; exit 2; }
[[ "${CHECKPOINT_EVERY}" =~ ^[1-9][0-9]*$ ]] || { echo "CHECKPOINT_EVERY must be positive" >&2; exit 2; }
(( MAX_STEPS % CHECKPOINT_EVERY == 0 )) || { echo "MAX_STEPS must be divisible by CHECKPOINT_EVERY" >&2; exit 2; }

TAG="${RATE_LAMBDA/./p}"
SOURCE="${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda${TAG}/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda${TAG}_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt"
[[ -s "${SOURCE}" ]] || { echo "missing source checkpoint: ${SOURCE}" >&2; exit 1; }
OUTPUT="${REPO_DIR}/outputs/nfcgs_score_path_ablation/${SCORE_PATH}/lambda${TAG}/job_${SLURM_JOB_ID:?submit with sbatch}"
echo "SOURCE_CHECKPOINT=${SOURCE}"
echo "SCORE_PATH=${SCORE_PATH} RATE_LAMBDA=${RATE_LAMBDA} MAX_STEPS=${MAX_STEPS}"
echo "TRAIN_OUTPUT=${OUTPUT}"
echo "GPUS=4 BATCH_PER_GPU=2 ACCUMULATE=1 EFFECTIVE_BATCH=8 LR=0.0001"

# Preflight once, then let the runner start one DDP process per GPU via srun.
exec python scripts/run_nfcgs.py train --devices 4 --launcher srun \
  --checkpoint "${SOURCE}" \
  --rate-lambda "${RATE_LAMBDA}" \
  --score-path "${SCORE_PATH}" \
  "${CONVERSION_ARGS[@]}" \
  --scope all --lr 0.0001 --batch-size 2 --accumulate 1 --workers 8 \
  --max-steps "${MAX_STEPS}" --checkpoint-every "${CHECKPOINT_EVERY}" \
  --output "${OUTPUT}"
