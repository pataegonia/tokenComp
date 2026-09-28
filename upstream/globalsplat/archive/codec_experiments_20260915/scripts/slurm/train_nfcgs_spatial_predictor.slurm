#!/usr/bin/env bash
#SBATCH -J gs-spatial-pred
#SBATCH --array=6-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm/slurm-gs-spatial-pred-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-spatial-pred-%A_%a.err

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
      echo "TASK=${task} ARM=${PREDICTOR_TAG} PREDICTOR=${SCORE_SPATIAL_PREDICTOR} LAMBDA=${RATE_LAMBDA} PARENT_TASK=${PARENT_TASK_ID} PARENT_PRIOR=${PARENT_PRIOR_TAG}"
    done
    exit 0
    ;;
  run) ;;
  *) echo "Usage: bash $0 plan | sbatch [--array=TASKS] $0" >&2; exit 1 ;;
esac

select_spatial_predictor_condition "${SLURM_ARRAY_TASK_ID:-}"
PARENT_ROOT="${SPATIAL_PREDICTOR_PARENT_ROOT:-${REPO_DIR}/outputs/nfcgs_nonlinear_score_context}"
PARENT_RUN_TAG="${SPATIAL_PREDICTOR_PARENT_RUN_TAG:-20260910_092000}"
PARENT_EXPERIMENT="nfcgs_nonlinear_score_context_joint_rank56_lambda${LAMBDA_TAG}_residual_on_morton_on_${PARENT_PRIOR_TAG}"
export WARM_START_CHECKPOINT="${PARENT_ROOT}/${PARENT_RUN_TAG}/${PARENT_PRIOR_TAG}/rank56/lambda${LAMBDA_TAG}/residual_on/checkpoints/${PARENT_EXPERIMENT}/version_0/step000050000.ckpt"

SPATIAL_PREDICTOR_ROOT="${SPATIAL_PREDICTOR_ROOT:-${REPO_DIR}/outputs/nfcgs_spatial_predictor25k}"
SPATIAL_PREDICTOR_RUN_TAG="${SPATIAL_PREDICTOR_RUN_TAG:-default}"
export OUTPUT_ROOT="${SPATIAL_PREDICTOR_ROOT}/${SPATIAL_PREDICTOR_RUN_TAG}/${PREDICTOR_TAG}/${PARENT_PRIOR_TAG}"
export EXPERIMENT_PREFIX="nfcgs_spatial_predictor25k_${PREDICTOR_TAG}"
export FEATURE_CODEC_TRAIN_SCOPE=all
export CONTEXT_VIEWS=24
export SUBSET_CONSISTENCY=true
export MAX_STEPS="${SPATIAL_PREDICTOR_MAX_STEPS:-25000}"
export MICRO_BATCH="${SPATIAL_PREDICTOR_MICRO_BATCH:-2}"
export ACCUMULATE="${SPATIAL_PREDICTOR_ACCUMULATE:-4}"
export NUM_WORKERS="${SPATIAL_PREDICTOR_NUM_WORKERS:-8}"
export PRECISION="${SPATIAL_PREDICTOR_PRECISION:-bf16-mixed}"
export SEED="${SPATIAL_PREDICTOR_SEED:-111123}"
export DUMMY_GPU_LOAD=false
export RESUME_CHECKPOINT="${SPATIAL_PREDICTOR_RESUME_CHECKPOINT:-}"

echo "SPATIAL_PREDICTOR_TASK=${SLURM_ARRAY_TASK_ID} RUN=${SPATIAL_PREDICTOR_RUN_TAG} ARM=${PREDICTOR_TAG} PREDICTOR=${SCORE_SPATIAL_PREDICTOR} LAMBDA=${RATE_LAMBDA}"
echo "PARENT_TASK=${PARENT_TASK_ID} PARENT_PRIOR=${PARENT_PRIOR_TAG} WARM_START_CHECKPOINT=${WARM_START_CHECKPOINT}"
echo "NODE=ariel-v12 GRES=gpu:1 BATCH=micro${MICRO_BATCH}xaccumulate${ACCUMULATE} STEPS=${MAX_STEPS}"
# RANK is a codec width here; do not leak it as a distributed-process rank.
export -n RANK
exec bash "${REPO_DIR}/scripts/slurm/train_nfcgs_paper_recipe.slurm"
