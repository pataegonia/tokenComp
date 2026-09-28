#!/usr/bin/env bash
#SBATCH -J gs-score-context
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm/slurm-gs-score-context-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-score-context-%A_%a.err

set -euo pipefail
export REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run from the GlobalSplat repository root" >&2; exit 1;
}
source "${REPO_DIR}/scripts/slurm/nfcgs_score_context_grid.sh"

case "${1:-run}" in
  plan)
    for task in {0..7}; do
      select_score_context_condition "${task}"
      echo "TASK=${task} PRIOR=${PRIOR_TAG} LAMBDA=${RATE_LAMBDA} RANK=56 RESIDUAL=on MORTON=on"
    done
    exit 0
    ;;
  run) ;;
  *) echo "Usage: bash $0 plan | sbatch [--array=0-7] $0" >&2; exit 1 ;;
esac

select_score_context_condition "${SLURM_ARRAY_TASK_ID:-}"
TRAIN_MODE="${TRAIN_MODE:-screen}"
case "${TRAIN_MODE}" in
  screen)
    export MAX_STEPS="${MAX_STEPS:-5000}"
    export FEATURE_CODEC_TRAIN_SCOPE=score_context
    export CONTEXT_VIEWS="${CONTEXT_VIEWS:-13}"
    export SUBSET_CONSISTENCY="${SUBSET_CONSISTENCY:-false}"
    ;;
  joint)
    export MAX_STEPS="${MAX_STEPS:-50000}"
    export FEATURE_CODEC_TRAIN_SCOPE=all
    export CONTEXT_VIEWS="${CONTEXT_VIEWS:-24}"
    export SUBSET_CONSISTENCY="${SUBSET_CONSISTENCY:-true}"
    ;;
  *) echo "ERROR: TRAIN_MODE must be screen or joint" >&2; exit 1 ;;
esac

BASE_TRAIN_ROOT="${BASE_TRAIN_ROOT:-${REPO_DIR}/outputs/nfcgs_paper24_subset_train}"
BASE_EXPERIMENT="nfcgs_paper24_subset_rank56_lambda${LAMBDA_TAG}_residual_on_morton_on"
export WARM_START_CHECKPOINT="${WARM_START_CHECKPOINT:-${BASE_TRAIN_ROOT}/rank56/lambda${LAMBDA_TAG}/residual_on/checkpoints/${BASE_EXPERIMENT}/version_0/step000050000.ckpt}"
SCORE_CONTEXT_ROOT="${SCORE_CONTEXT_ROOT:-${REPO_DIR}/outputs/nfcgs_score_context}"
SCORE_CONTEXT_RUN_TAG="${SCORE_CONTEXT_RUN_TAG:-default}"
export OUTPUT_ROOT="${SCORE_CONTEXT_ROOT}/${SCORE_CONTEXT_RUN_TAG}/${TRAIN_MODE}/${PRIOR_TAG}"
export EXPERIMENT_PREFIX="nfcgs_score_context_${TRAIN_MODE}"
export MICRO_BATCH="${MICRO_BATCH:-2}"
export ACCUMULATE="${ACCUMULATE:-4}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export PRECISION="${PRECISION:-bf16-mixed}"
export SEED="${SEED:-111123}"
export DUMMY_GPU_LOAD=false
export RESUME_CHECKPOINT="${SCORE_CONTEXT_RESUME_CHECKPOINT:-}"

echo "SCORE_CONTEXT_TASK=${SLURM_ARRAY_TASK_ID} RUN=${SCORE_CONTEXT_RUN_TAG} MODE=${TRAIN_MODE} PRIOR=${PRIOR_TAG}"
echo "WARM_START_CHECKPOINT=${WARM_START_CHECKPOINT}"
exec bash "${REPO_DIR}/scripts/slurm/train_nfcgs_paper_recipe.slurm"
