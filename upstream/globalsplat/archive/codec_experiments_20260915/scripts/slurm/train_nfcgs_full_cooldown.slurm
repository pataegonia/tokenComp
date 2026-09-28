#!/usr/bin/env bash
#SBATCH -J gs-full-lowlr
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 2-00:00:00
#SBATCH -o logs/slurm/slurm-gs-full-lowlr-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-full-lowlr-%A_%a.err

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
      echo "TASK=${task} ARM=${ARM_TAG} PREDICTOR=${SCORE_SPATIAL_PREDICTOR} LAMBDA=${RATE_LAMBDA} PARENT_TASK=${PARENT_TASK_ID} LR=1e-6"
    done
    exit 0
    ;;
  run) ;;
  *) echo "Usage: bash $0 plan | sbatch [--array=TASKS] $0" >&2; exit 1 ;;
esac

select_full_cooldown_condition "${SLURM_ARRAY_TASK_ID:-}"
PARENT_ROOT="${FULL_COOLDOWN_PARENT_ROOT:-${REPO_DIR}/outputs/nfcgs_nonlinear_score_context}"
PARENT_RUN_TAG="${FULL_COOLDOWN_PARENT_RUN_TAG:-20260910_092000}"
PARENT_EXPERIMENT="nfcgs_nonlinear_score_context_joint_rank56_lambda${LAMBDA_TAG}_residual_on_morton_on_${PARENT_PRIOR_TAG}"
export WARM_START_CHECKPOINT="${PARENT_ROOT}/${PARENT_RUN_TAG}/${PARENT_PRIOR_TAG}/rank56/lambda${LAMBDA_TAG}/residual_on/checkpoints/${PARENT_EXPERIMENT}/version_0/step000050000.ckpt"

FULL_COOLDOWN_ROOT="${FULL_COOLDOWN_ROOT:-${REPO_DIR}/outputs/nfcgs_full_lowlr10k}"
FULL_COOLDOWN_RUN_TAG="${FULL_COOLDOWN_RUN_TAG:-default}"
export OUTPUT_ROOT="${FULL_COOLDOWN_ROOT}/${FULL_COOLDOWN_RUN_TAG}/${ARM_TAG}"
export EXPERIMENT_PREFIX="nfcgs_full_lowlr10k_${ARM_TAG}"
export FEATURE_CODEC_TRAIN_SCOPE=all
export CONTEXT_VIEWS=24
export SUBSET_CONSISTENCY=true
export MAX_STEPS="${FULL_COOLDOWN_MAX_STEPS:-10000}"
export MICRO_BATCH="${FULL_COOLDOWN_MICRO_BATCH:-2}"
export ACCUMULATE="${FULL_COOLDOWN_ACCUMULATE:-4}"
export NUM_WORKERS="${FULL_COOLDOWN_NUM_WORKERS:-8}"
export PRECISION="${FULL_COOLDOWN_PRECISION:-bf16-mixed}"
export SEED="${FULL_COOLDOWN_SEED:-111123}"
export OPTIMIZER_LR="${FULL_COOLDOWN_LR:-0.000001}"
# Keep LR constant even though the shared recipe retains its archived milestones.
export OPTIMIZER_LR_GAMMA=1.0
export QUANTILE_UPDATE_INTERVAL="${FULL_COOLDOWN_QUANTILE_INTERVAL:-500}"
export DUMMY_GPU_LOAD=false
export RESUME_CHECKPOINT="${FULL_COOLDOWN_RESUME_CHECKPOINT:-}"

echo "FULL_COOLDOWN_TASK=${SLURM_ARRAY_TASK_ID} RUN=${FULL_COOLDOWN_RUN_TAG} ARM=${ARM_TAG} PREDICTOR=${SCORE_SPATIAL_PREDICTOR} LAMBDA=${RATE_LAMBDA}"
echo "PARENT_TASK=${PARENT_TASK_ID} WARM_START_CHECKPOINT=${WARM_START_CHECKPOINT}"
echo "NODE=ariel-v12 GRES=gpu:1 BATCH=micro${MICRO_BATCH}xaccumulate${ACCUMULATE} STEPS=${MAX_STEPS} LR=${OPTIMIZER_LR}"
export -n RANK
exec bash "${REPO_DIR}/scripts/slurm/train_nfcgs_paper_recipe.slurm"
