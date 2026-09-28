#!/usr/bin/env bash
#SBATCH -J gs-spatial-pred-eval
#SBATCH --array=6-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-spatial-pred-eval-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-spatial-pred-eval-%A_%a.err

set -euo pipefail
export REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run from the GlobalSplat repository root" >&2; exit 1;
}
source "${REPO_DIR}/scripts/slurm/nfcgs_spatial_predictor_grid.sh"

case "${1:-run}" in
  plan)
    for task in {0..8}; do
      select_spatial_predictor_condition "${task}"
      echo "TASK=${task} ARM=${PREDICTOR_TAG} PREDICTOR=${SCORE_SPATIAL_PREDICTOR} LAMBDA=${RATE_LAMBDA} PARENT_TASK=${PARENT_TASK_ID} EXPECTED_STEP=25000"
    done
    exit 0
    ;;
  run) ;;
  *) echo "Usage: bash $0 plan | sbatch [--array=TASKS] $0" >&2; exit 1 ;;
esac

select_spatial_predictor_condition "${SLURM_ARRAY_TASK_ID:-}"
SPATIAL_PREDICTOR_ROOT="${SPATIAL_PREDICTOR_ROOT:-${REPO_DIR}/outputs/nfcgs_spatial_predictor25k}"
SPATIAL_PREDICTOR_RUN_TAG="${SPATIAL_PREDICTOR_RUN_TAG:-default}"
SCORE_CONTEXT_TAG="${PARENT_PRIOR_TAG}"
if [[ "${SCORE_SPATIAL_PREDICTOR}" != linear ]]; then
  SCORE_CONTEXT_TAG="${SCORE_CONTEXT_TAG}_${SCORE_SPATIAL_PREDICTOR}h${SCORE_SPATIAL_HIDDEN}"
fi
EXPERIMENT="nfcgs_spatial_predictor25k_${PREDICTOR_TAG}_rank56_lambda${LAMBDA_TAG}_residual_on_morton_on_${SCORE_CONTEXT_TAG}"
EXPECTED_STEP="${SPATIAL_PREDICTOR_EVAL_STEP:-${SPATIAL_PREDICTOR_MAX_STEPS:-25000}}"
printf -v CHECKPOINT_NAME 'step%09d.ckpt' "$((10#${EXPECTED_STEP}))"
TRAIN_ROOT="${SPATIAL_PREDICTOR_ROOT}/${SPATIAL_PREDICTOR_RUN_TAG}/${PREDICTOR_TAG}/${PARENT_PRIOR_TAG}"
export CHECKPOINT="${SPATIAL_PREDICTOR_CHECKPOINT:-${TRAIN_ROOT}/rank56/lambda${LAMBDA_TAG}/residual_on/checkpoints/${EXPERIMENT}/version_0/${CHECKPOINT_NAME}}"
export OUTPUT_ROOT="${SPATIAL_PREDICTOR_EVAL_ROOT:-${REPO_DIR}/outputs/nfcgs_spatial_predictor25k_eval}/${SPATIAL_PREDICTOR_RUN_TAG}/${PREDICTOR_TAG}/${PARENT_PRIOR_TAG}"
export EXPECTED_STEP
export PROTOCOL="${SPATIAL_PREDICTOR_PROTOCOL:-all}"
export CONTEXT="${SPATIAL_PREDICTOR_CONTEXT:-12}"
export MAX_SCENES="${SPATIAL_PREDICTOR_MAX_SCENES:-null}"
export NUM_WORKERS="${SPATIAL_PREDICTOR_EVAL_WORKERS:-4}"
export PRECISION="${SPATIAL_PREDICTOR_PRECISION:-bf16-mixed}"
export SAVE_IMAGES="${SPATIAL_PREDICTOR_SAVE_IMAGES:-false}"

echo "SPATIAL_PREDICTOR_EVAL_TASK=${SLURM_ARRAY_TASK_ID} RUN=${SPATIAL_PREDICTOR_RUN_TAG} ARM=${PREDICTOR_TAG} PREDICTOR=${SCORE_SPATIAL_PREDICTOR} LAMBDA=${RATE_LAMBDA}"
echo "CHECKPOINT=${CHECKPOINT}"
export -n RANK
exec bash "${REPO_DIR}/scripts/slurm/eval_nfcgs_rank56.slurm"
