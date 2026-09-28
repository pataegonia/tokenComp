#!/usr/bin/env bash
#SBATCH -J gs-score-prob
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 2-00:00:00
#SBATCH -o logs/slurm/slurm-gs-score-prob-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-score-prob-%A_%a.err

set -euo pipefail
export REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run from the GlobalSplat repository root" >&2; exit 1;
}
source "${REPO_DIR}/scripts/slurm/nfcgs_score_probability_grid.sh"

case "${1:-run}" in
  plan)
    for task in {0..7}; do
      select_score_probability_condition "${task}"
      echo "TASK=${task} ARM=${ENTROPY_TAG} ENTROPY=${SCORE_SPATIAL_ENTROPY} LAMBDA=${RATE_LAMBDA} PARENT_TASK=${PARENT_TASK_ID} SCOPE=score_probability"
    done
    exit 0
    ;;
  run) ;;
  *) echo "Usage: bash $0 plan | sbatch [--array=TASKS] $0" >&2; exit 1 ;;
esac

select_score_probability_condition "${SLURM_ARRAY_TASK_ID:-}"
PARENT_ROOT="${SCORE_PROBABILITY_PARENT_ROOT:-${REPO_DIR}/outputs/nfcgs_nonlinear_score_context}"
PARENT_RUN_TAG="${SCORE_PROBABILITY_PARENT_RUN_TAG:-20260910_092000}"
PARENT_EXPERIMENT="nfcgs_nonlinear_score_context_joint_rank56_lambda${LAMBDA_TAG}_residual_on_morton_on_${PARENT_PRIOR_TAG}"
export WARM_START_CHECKPOINT="${PARENT_ROOT}/${PARENT_RUN_TAG}/${PARENT_PRIOR_TAG}/rank56/lambda${LAMBDA_TAG}/residual_on/checkpoints/${PARENT_EXPERIMENT}/version_0/step000050000.ckpt"

SCORE_PROBABILITY_ROOT="${SCORE_PROBABILITY_ROOT:-${REPO_DIR}/outputs/nfcgs_score_probability10k}"
SCORE_PROBABILITY_RUN_TAG="${SCORE_PROBABILITY_RUN_TAG:-default}"
export OUTPUT_ROOT="${SCORE_PROBABILITY_ROOT}/${SCORE_PROBABILITY_RUN_TAG}/${ENTROPY_TAG}"
export EXPERIMENT_PREFIX="nfcgs_score_probability10k_${ENTROPY_TAG}"
export FEATURE_CODEC_TRAIN_SCOPE=score_probability
export CONTEXT_VIEWS=24
export SUBSET_CONSISTENCY=true
export MAX_STEPS="${SCORE_PROBABILITY_MAX_STEPS:-10000}"
export MICRO_BATCH="${SCORE_PROBABILITY_MICRO_BATCH:-2}"
export ACCUMULATE="${SCORE_PROBABILITY_ACCUMULATE:-4}"
export NUM_WORKERS="${SCORE_PROBABILITY_NUM_WORKERS:-8}"
export PRECISION="${SCORE_PROBABILITY_PRECISION:-bf16-mixed}"
export SEED="${SCORE_PROBABILITY_SEED:-111123}"
export OPTIMIZER_LR="${SCORE_PROBABILITY_LR:-0.0001}"
export OPTIMIZER_LR_MILESTONES="${SCORE_PROBABILITY_LR_MILESTONES:-[7000]}"
export OPTIMIZER_LR_GAMMA="${SCORE_PROBABILITY_LR_GAMMA:-0.1}"
# Quantiles define the reconstruction lattice. They are frozen by the scope;
# disabling auxiliary updates makes that invariant explicit at the recipe level.
export QUANTILE_UPDATE_INTERVAL=0
export DUMMY_GPU_LOAD=false
export RESUME_CHECKPOINT="${SCORE_PROBABILITY_RESUME_CHECKPOINT:-}"

echo "SCORE_PROBABILITY_TASK=${SLURM_ARRAY_TASK_ID} RUN=${SCORE_PROBABILITY_RUN_TAG} ARM=${ENTROPY_TAG} ENTROPY=${SCORE_SPATIAL_ENTROPY} LAMBDA=${RATE_LAMBDA}"
echo "PARENT_TASK=${PARENT_TASK_ID} WARM_START_CHECKPOINT=${WARM_START_CHECKPOINT}"
echo "NODE=ariel-v12 GRES=gpu:1 BATCH=micro${MICRO_BATCH}xaccumulate${ACCUMULATE} STEPS=${MAX_STEPS} LR=${OPTIMIZER_LR}"
export -n RANK
exec bash "${REPO_DIR}/scripts/slurm/train_nfcgs_paper_recipe.slurm"
