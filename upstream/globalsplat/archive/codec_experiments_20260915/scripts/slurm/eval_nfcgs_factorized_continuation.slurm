#!/usr/bin/env bash
#SBATCH -J gs-factorized-cont-eval
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-factorized-cont-eval-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-factorized-cont-eval-%A_%a.err

set -euo pipefail
export REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run from the GlobalSplat repository root" >&2; exit 1;
}
source "${REPO_DIR}/scripts/slurm/nfcgs_factorized_continuation_grid.sh"

case "${1:-run}" in
  plan)
    for task in 0 1 2 3; do
      select_factorized_continuation_condition "${task}"
      echo "TASK=${task} PRIOR=factorized RANK=${RANK} LAMBDA=${RATE_LAMBDA} EXPECTED_STEP=50000"
    done
    exit 0
    ;;
  run) ;;
  *) echo "Usage: bash $0 plan | sbatch [--array=0-3] $0" >&2; exit 1 ;;
esac

select_factorized_continuation_condition "${SLURM_ARRAY_TASK_ID:-}"
FACTOR_CONT_ROOT="${FACTOR_CONT_ROOT:-${REPO_DIR}/outputs/nfcgs_factorized_continuation}"
FACTOR_CONT_RUN_TAG="${FACTOR_CONT_RUN_TAG:-default}"
EXPERIMENT="nfcgs_factorized_continuation_rank${RANK}_lambda${LAMBDA_TAG}_residual_on_morton_on"
EXPECTED_STEP="${FACTOR_CONT_EVAL_STEP:-${FACTOR_CONT_MAX_STEPS:-50000}}"
printf -v CHECKPOINT_NAME 'step%09d.ckpt' "$((10#${EXPECTED_STEP}))"
export CHECKPOINT="${FACTOR_CONT_CHECKPOINT:-${FACTOR_CONT_ROOT}/${FACTOR_CONT_RUN_TAG}/rank${RANK}/lambda${LAMBDA_TAG}/residual_on/checkpoints/${EXPERIMENT}/version_0/${CHECKPOINT_NAME}}"
export OUTPUT_ROOT="${FACTOR_CONT_EVAL_ROOT:-${REPO_DIR}/outputs/nfcgs_factorized_continuation_eval}/${FACTOR_CONT_RUN_TAG}"
export EXPECTED_STEP
export PROTOCOL="${PROTOCOL:-all}"
export CONTEXT="${CONTEXT:-12}"
export MAX_SCENES="${FACTOR_CONT_MAX_SCENES:-null}"
export NUM_WORKERS="${FACTOR_CONT_EVAL_WORKERS:-4}"
export PRECISION="${FACTOR_CONT_PRECISION:-bf16-mixed}"
export SAVE_IMAGES="${FACTOR_CONT_SAVE_IMAGES:-false}"

echo "FACTORIZED_CONT_EVAL_TASK=${SLURM_ARRAY_TASK_ID} RUN=${FACTOR_CONT_RUN_TAG} RANK=${RANK} LAMBDA=${RATE_LAMBDA}"
echo "CHECKPOINT=${CHECKPOINT}"
# GRID_TASK_ID lets the common evaluator recover codec width without exporting
# RANK as a distributed-process environment variable.
export -n RANK
exec bash "${REPO_DIR}/scripts/slurm/eval_nfcgs_rank56.slurm"
