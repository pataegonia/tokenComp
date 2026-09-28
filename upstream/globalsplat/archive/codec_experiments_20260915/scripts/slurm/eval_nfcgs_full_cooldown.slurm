#!/usr/bin/env bash
#SBATCH -J gs-full-lowlr-eval
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-full-lowlr-eval-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-full-lowlr-eval-%A_%a.err

set -euo pipefail
export REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run from the GlobalSplat repository root" >&2; exit 1;
}
source "${REPO_DIR}/scripts/slurm/nfcgs_full_cooldown_grid.sh"

case "${1:-run}" in
  plan)
    for task in {0..3}; do
      select_full_cooldown_condition "${task}"
      echo "TASK=${task} ARM=${ARM_TAG} PREDICTOR=${SCORE_SPATIAL_PREDICTOR} LAMBDA=${RATE_LAMBDA} EXPECTED_STEP=10000"
    done
    exit 0
    ;;
  run) ;;
  *) echo "Usage: bash $0 plan | sbatch [--array=TASKS] $0" >&2; exit 1 ;;
esac

select_full_cooldown_condition "${SLURM_ARRAY_TASK_ID:-}"
FULL_COOLDOWN_ROOT="${FULL_COOLDOWN_ROOT:-${REPO_DIR}/outputs/nfcgs_full_lowlr10k}"
FULL_COOLDOWN_RUN_TAG="${FULL_COOLDOWN_RUN_TAG:-default}"
SCORE_CONTEXT_TAG=m1c1s1
if [[ "${SCORE_SPATIAL_PREDICTOR}" != linear ]]; then
  SCORE_CONTEXT_TAG="${SCORE_CONTEXT_TAG}_${SCORE_SPATIAL_PREDICTOR}h${SCORE_SPATIAL_HIDDEN}"
fi
EXPERIMENT="nfcgs_full_lowlr10k_${ARM_TAG}_rank56_lambda${LAMBDA_TAG}_residual_on_morton_on_${SCORE_CONTEXT_TAG}"
EXPECTED_STEP="${FULL_COOLDOWN_EVAL_STEP:-${FULL_COOLDOWN_MAX_STEPS:-10000}}"
printf -v CHECKPOINT_NAME 'step%09d.ckpt' "$((10#${EXPECTED_STEP}))"
TRAIN_ROOT="${FULL_COOLDOWN_ROOT}/${FULL_COOLDOWN_RUN_TAG}/${ARM_TAG}"
export CHECKPOINT="${FULL_COOLDOWN_CHECKPOINT:-${TRAIN_ROOT}/rank56/lambda${LAMBDA_TAG}/residual_on/checkpoints/${EXPERIMENT}/version_0/${CHECKPOINT_NAME}}"
export OUTPUT_ROOT="${FULL_COOLDOWN_EVAL_ROOT:-${REPO_DIR}/outputs/nfcgs_full_lowlr10k_eval}/${FULL_COOLDOWN_RUN_TAG}/${ARM_TAG}"
export EXPECTED_STEP
export PROTOCOL="${FULL_COOLDOWN_PROTOCOL:-all}"
export CONTEXT="${FULL_COOLDOWN_CONTEXT:-12}"
export MAX_SCENES="${FULL_COOLDOWN_MAX_SCENES:-null}"
export NUM_WORKERS="${FULL_COOLDOWN_EVAL_WORKERS:-4}"
export PRECISION="${FULL_COOLDOWN_PRECISION:-bf16-mixed}"
export SAVE_IMAGES="${FULL_COOLDOWN_SAVE_IMAGES:-false}"

echo "FULL_COOLDOWN_EVAL_TASK=${SLURM_ARRAY_TASK_ID} RUN=${FULL_COOLDOWN_RUN_TAG} ARM=${ARM_TAG} LAMBDA=${RATE_LAMBDA}"
echo "CHECKPOINT=${CHECKPOINT}"
export -n RANK
exec bash "${REPO_DIR}/scripts/slurm/eval_nfcgs_rank56.slurm"
