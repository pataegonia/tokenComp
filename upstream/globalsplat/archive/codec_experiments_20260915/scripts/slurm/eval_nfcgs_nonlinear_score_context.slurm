#!/usr/bin/env bash
#SBATCH -J gs-nl-score-ctx-eval
#SBATCH --array=0-1,10-11
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-nl-score-ctx-eval-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-nl-score-ctx-eval-%A_%a.err

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
      echo "TASK=${task} PRIOR=${PRIOR_TAG} RANK=${RANK} LAMBDA=${RATE_LAMBDA} TRANSFORM=nonlinear32 EXPECTED_STEP=50000"
    done
    exit 0
    ;;
  run) ;;
  *) echo "Usage: bash $0 plan | sbatch [--array=TASKS] $0" >&2; exit 1 ;;
esac

select_nonlinear_score_context_condition "${SLURM_ARRAY_TASK_ID:-}"
NONLINEAR_CONTEXT_ROOT="${NONLINEAR_CONTEXT_ROOT:-${REPO_DIR}/outputs/nfcgs_nonlinear_score_context}"
NONLINEAR_CONTEXT_RUN_TAG="${NONLINEAR_CONTEXT_RUN_TAG:-default}"
SCORE_CONTEXT_SUFFIX=""
if [[ "${PRIOR_TAG}" != factorized ]]; then
  SCORE_CONTEXT_SUFFIX="_${PRIOR_TAG}"
fi
EXPERIMENT="nfcgs_nonlinear_score_context_joint_rank${RANK}_lambda${LAMBDA_TAG}_residual_on_morton_on${SCORE_CONTEXT_SUFFIX}"
EXPECTED_STEP="${NONLINEAR_CONTEXT_EVAL_STEP:-${NONLINEAR_CONTEXT_MAX_STEPS:-50000}}"
printf -v CHECKPOINT_NAME 'step%09d.ckpt' "$((10#${EXPECTED_STEP}))"
export CHECKPOINT="${NONLINEAR_CONTEXT_CHECKPOINT:-${NONLINEAR_CONTEXT_ROOT}/${NONLINEAR_CONTEXT_RUN_TAG}/${PRIOR_TAG}/rank${RANK}/lambda${LAMBDA_TAG}/residual_on/checkpoints/${EXPERIMENT}/version_0/${CHECKPOINT_NAME}}"
export OUTPUT_ROOT="${NONLINEAR_CONTEXT_EVAL_ROOT:-${REPO_DIR}/outputs/nfcgs_nonlinear_score_context_eval}/${NONLINEAR_CONTEXT_RUN_TAG}/${PRIOR_TAG}"
export EXPECTED_STEP
export PROTOCOL="${PROTOCOL:-all}"
export CONTEXT="${CONTEXT:-12}"
export MAX_SCENES="${NONLINEAR_CONTEXT_MAX_SCENES:-null}"
export NUM_WORKERS="${NONLINEAR_CONTEXT_EVAL_WORKERS:-4}"
export PRECISION="${NONLINEAR_CONTEXT_PRECISION:-bf16-mixed}"
export SAVE_IMAGES="${NONLINEAR_CONTEXT_SAVE_IMAGES:-false}"

echo "NONLINEAR_CONTEXT_EVAL_TASK=${SLURM_ARRAY_TASK_ID} RUN=${NONLINEAR_CONTEXT_RUN_TAG} PRIOR=${PRIOR_TAG} RANK=${RANK} LAMBDA=${RATE_LAMBDA}"
echo "CHECKPOINT=${CHECKPOINT}"
export -n RANK
exec bash "${REPO_DIR}/scripts/slurm/eval_nfcgs_rank56.slurm"
