#!/usr/bin/env bash
#SBATCH -J gs-nl-score-ctx
#SBATCH --array=0-1,10-11
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm/slurm-gs-nl-score-ctx-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-nl-score-ctx-%A_%a.err

set -euo pipefail
export REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run from the GlobalSplat repository root" >&2; exit 1;
}
source "${REPO_DIR}/scripts/slurm/nfcgs_nonlinear_score_context_grid.sh"

case "${1:-run}" in
  plan)
    for task in {0..11}; do
      select_nonlinear_score_context_condition "${task}"
      echo "TASK=${task} PRIOR=${PRIOR_TAG} RANK=${RANK} LAMBDA=${RATE_LAMBDA} TRANSFORM=nonlinear32"
    done
    exit 0
    ;;
  run) ;;
  *) echo "Usage: bash $0 plan | sbatch [--array=TASKS] $0" >&2; exit 1 ;;
esac

select_nonlinear_score_context_condition "${SLURM_ARRAY_TASK_ID:-}"
NONLINEAR_TRAIN_ROOT="${NONLINEAR_TRAIN_ROOT:-${REPO_DIR}/outputs/nfcgs_paper24_transform_train}"
BASE_EXPERIMENT="nfcgs_transform_nonlinear32_rank${RANK}_lambda${LAMBDA_TAG}_residual_on_morton_on"
export WARM_START_CHECKPOINT="${NONLINEAR_CONTEXT_WARM_START_CHECKPOINT:-${NONLINEAR_TRAIN_ROOT}/nonlinear32/rank${RANK}/lambda${LAMBDA_TAG}/residual_on/checkpoints/${BASE_EXPERIMENT}/version_0/step000050000.ckpt}"

NONLINEAR_CONTEXT_ROOT="${NONLINEAR_CONTEXT_ROOT:-${REPO_DIR}/outputs/nfcgs_nonlinear_score_context}"
NONLINEAR_CONTEXT_RUN_TAG="${NONLINEAR_CONTEXT_RUN_TAG:-default}"
export OUTPUT_ROOT="${NONLINEAR_CONTEXT_ROOT}/${NONLINEAR_CONTEXT_RUN_TAG}/${PRIOR_TAG}"
export EXPERIMENT_PREFIX=nfcgs_nonlinear_score_context_joint
export FEATURE_CODEC_TRAIN_SCOPE=all
export CONTEXT_VIEWS=24
export SUBSET_CONSISTENCY=true
export MAX_STEPS="${NONLINEAR_CONTEXT_MAX_STEPS:-50000}"
export MICRO_BATCH="${NONLINEAR_CONTEXT_MICRO_BATCH:-2}"
export ACCUMULATE="${NONLINEAR_CONTEXT_ACCUMULATE:-4}"
export NUM_WORKERS="${NONLINEAR_CONTEXT_NUM_WORKERS:-8}"
export PRECISION="${NONLINEAR_CONTEXT_PRECISION:-bf16-mixed}"
export SEED="${NONLINEAR_CONTEXT_SEED:-111123}"
export DUMMY_GPU_LOAD=false
export RESUME_CHECKPOINT="${NONLINEAR_CONTEXT_RESUME_CHECKPOINT:-}"

echo "NONLINEAR_CONTEXT_TASK=${SLURM_ARRAY_TASK_ID} RUN=${NONLINEAR_CONTEXT_RUN_TAG} PRIOR=${PRIOR_TAG} RANK=${RANK} LAMBDA=${RATE_LAMBDA}"
echo "WARM_START_CHECKPOINT=${WARM_START_CHECKPOINT}"
echo "NODE=ariel-v12 GRES=gpu:1 BATCH=micro${MICRO_BATCH}xaccumulate${ACCUMULATE} STEPS=${MAX_STEPS}"
# RANK is a codec width here; do not leak it as a distributed-process rank.
export -n RANK
exec bash "${REPO_DIR}/scripts/slurm/train_nfcgs_paper_recipe.slurm"
