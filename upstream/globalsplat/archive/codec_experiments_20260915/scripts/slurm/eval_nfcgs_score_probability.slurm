#!/usr/bin/env bash
#SBATCH -J gs-score-prob-eval
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-score-prob-eval-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-score-prob-eval-%A_%a.err

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
      echo "TASK=${task} ARM=${ENTROPY_TAG} ENTROPY=${SCORE_SPATIAL_ENTROPY} LAMBDA=${RATE_LAMBDA} EXPECTED_STEP=10000"
    done
    exit 0
    ;;
  run) ;;
  *) echo "Usage: bash $0 plan | sbatch [--array=TASKS] $0" >&2; exit 1 ;;
esac

select_score_probability_condition "${SLURM_ARRAY_TASK_ID:-}"
SCORE_PROBABILITY_ROOT="${SCORE_PROBABILITY_ROOT:-${REPO_DIR}/outputs/nfcgs_score_probability10k}"
SCORE_PROBABILITY_RUN_TAG="${SCORE_PROBABILITY_RUN_TAG:-default}"
SCORE_CONTEXT_TAG=m1c1s1
if [[ "${SCORE_SPATIAL_ENTROPY}" != shared ]]; then
  SCORE_CONTEXT_TAG="${SCORE_CONTEXT_TAG}_${SCORE_SPATIAL_ENTROPY}"
fi
EXPERIMENT="nfcgs_score_probability10k_${ENTROPY_TAG}_rank56_lambda${LAMBDA_TAG}_residual_on_morton_on_${SCORE_CONTEXT_TAG}"
EXPECTED_STEP="${SCORE_PROBABILITY_EVAL_STEP:-${SCORE_PROBABILITY_MAX_STEPS:-10000}}"
printf -v CHECKPOINT_NAME 'step%09d.ckpt' "$((10#${EXPECTED_STEP}))"
TRAIN_ROOT="${SCORE_PROBABILITY_ROOT}/${SCORE_PROBABILITY_RUN_TAG}/${ENTROPY_TAG}"
export CHECKPOINT="${SCORE_PROBABILITY_CHECKPOINT:-${TRAIN_ROOT}/rank56/lambda${LAMBDA_TAG}/residual_on/checkpoints/${EXPERIMENT}/version_0/${CHECKPOINT_NAME}}"
export OUTPUT_ROOT="${SCORE_PROBABILITY_EVAL_ROOT:-${REPO_DIR}/outputs/nfcgs_score_probability10k_eval}/${SCORE_PROBABILITY_RUN_TAG}/${ENTROPY_TAG}"
export EXPECTED_STEP
export PROTOCOL="${SCORE_PROBABILITY_PROTOCOL:-all}"
export CONTEXT="${SCORE_PROBABILITY_CONTEXT:-12}"
export MAX_SCENES="${SCORE_PROBABILITY_MAX_SCENES:-null}"
export NUM_WORKERS="${SCORE_PROBABILITY_EVAL_WORKERS:-4}"
export PRECISION="${SCORE_PROBABILITY_PRECISION:-bf16-mixed}"
export SAVE_IMAGES="${SCORE_PROBABILITY_SAVE_IMAGES:-false}"

echo "SCORE_PROBABILITY_EVAL_TASK=${SLURM_ARRAY_TASK_ID} RUN=${SCORE_PROBABILITY_RUN_TAG} ARM=${ENTROPY_TAG} ENTROPY=${SCORE_SPATIAL_ENTROPY} LAMBDA=${RATE_LAMBDA}"
echo "CHECKPOINT=${CHECKPOINT}"
export -n RANK
exec bash "${REPO_DIR}/scripts/slurm/eval_nfcgs_rank56.slurm"
