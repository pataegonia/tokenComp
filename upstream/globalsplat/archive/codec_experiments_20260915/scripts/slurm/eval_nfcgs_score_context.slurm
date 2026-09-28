#!/usr/bin/env bash
#SBATCH -J gs-score-ctx-eval
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-score-ctx-eval-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-score-ctx-eval-%A_%a.err

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
      echo "TASK=${task} PRIOR=${PRIOR_TAG} LAMBDA=${RATE_LAMBDA}"
    done
    exit 0
    ;;
  run) ;;
  *) echo "Usage: bash $0 plan | sbatch [--array=0-7] $0" >&2; exit 1 ;;
esac

select_score_context_condition "${SLURM_ARRAY_TASK_ID:-}"
TRAIN_MODE="${TRAIN_MODE:-screen}"
case "${TRAIN_MODE}" in
  screen) EXPECTED_STEP="${EXPECTED_STEP:-5000}"; MAX_SCENES="${MAX_SCENES:-300}" ;;
  joint) EXPECTED_STEP="${EXPECTED_STEP:-50000}"; MAX_SCENES="${MAX_SCENES:-null}" ;;
  *) echo "ERROR: TRAIN_MODE must be screen or joint" >&2; exit 1 ;;
esac
[[ "${EXPECTED_STEP}" =~ ^[0-9]+$ ]] || { echo "ERROR: EXPECTED_STEP must be numeric for score-context evaluation" >&2; exit 1; }
VERSION="${SCORE_CONTEXT_VERSION:-0}"
SCORE_CONTEXT_ROOT="${SCORE_CONTEXT_ROOT:-${REPO_DIR}/outputs/nfcgs_score_context}"
SCORE_CONTEXT_RUN_TAG="${SCORE_CONTEXT_RUN_TAG:-default}"
EXPERIMENT="nfcgs_score_context_${TRAIN_MODE}_rank56_lambda${LAMBDA_TAG}_residual_on_morton_on_${PRIOR_TAG}"
export CHECKPOINT="${CHECKPOINT:-${SCORE_CONTEXT_ROOT}/${SCORE_CONTEXT_RUN_TAG}/${TRAIN_MODE}/${PRIOR_TAG}/rank56/lambda${LAMBDA_TAG}/residual_on/checkpoints/${EXPERIMENT}/version_${VERSION}/step$(printf '%09d' "${EXPECTED_STEP}").ckpt}"
export OUTPUT_ROOT="${SCORE_CONTEXT_EVAL_ROOT:-${REPO_DIR}/outputs/nfcgs_score_context_eval}/${SCORE_CONTEXT_RUN_TAG}/${TRAIN_MODE}/${PRIOR_TAG}"
export EXPECTED_STEP MAX_SCENES
export PROTOCOL="${PROTOCOL:-all}"
export CONTEXT="${CONTEXT:-12}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SAVE_IMAGES="${SAVE_IMAGES:-false}"

echo "SCORE_CONTEXT_EVAL_TASK=${SLURM_ARRAY_TASK_ID} RUN=${SCORE_CONTEXT_RUN_TAG} MODE=${TRAIN_MODE} PRIOR=${PRIOR_TAG} MAX_SCENES=${MAX_SCENES}"
echo "CHECKPOINT=${CHECKPOINT}"
exec bash "${REPO_DIR}/scripts/slurm/eval_nfcgs_rank56.slurm"
