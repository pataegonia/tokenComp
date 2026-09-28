#!/usr/bin/env bash
#SBATCH -J gs-factorized-cont
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm/slurm-gs-factorized-cont-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-factorized-cont-%A_%a.err

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
      echo "TASK=${task} PRIOR=factorized RANK=${RANK} LAMBDA=${RATE_LAMBDA} RESIDUAL=on MORTON=on"
    done
    exit 0
    ;;
  run) ;;
  *) echo "Usage: bash $0 plan | sbatch [--array=0-3] $0" >&2; exit 1 ;;
esac

select_factorized_continuation_condition "${SLURM_ARRAY_TASK_ID:-}"
BASE_TRAIN_ROOT="${BASE_TRAIN_ROOT:-${REPO_DIR}/outputs/nfcgs_paper24_subset_train}"
BASE_EXPERIMENT="nfcgs_paper24_subset_rank${RANK}_lambda${LAMBDA_TAG}_residual_on_morton_on"
export WARM_START_CHECKPOINT="${FACTOR_CONT_WARM_START_CHECKPOINT:-${BASE_TRAIN_ROOT}/rank${RANK}/lambda${LAMBDA_TAG}/residual_on/checkpoints/${BASE_EXPERIMENT}/version_0/step000050000.ckpt}"

FACTOR_CONT_ROOT="${FACTOR_CONT_ROOT:-${REPO_DIR}/outputs/nfcgs_factorized_continuation}"
FACTOR_CONT_RUN_TAG="${FACTOR_CONT_RUN_TAG:-default}"
export OUTPUT_ROOT="${FACTOR_CONT_ROOT}/${FACTOR_CONT_RUN_TAG}"
export EXPERIMENT_PREFIX=nfcgs_factorized_continuation
export FEATURE_CODEC_TRAIN_SCOPE=all
export CONTEXT_VIEWS=24
export SUBSET_CONSISTENCY=true
export MAX_STEPS="${FACTOR_CONT_MAX_STEPS:-50000}"
export MICRO_BATCH="${FACTOR_CONT_MICRO_BATCH:-2}"
export ACCUMULATE="${FACTOR_CONT_ACCUMULATE:-4}"
export NUM_WORKERS="${FACTOR_CONT_NUM_WORKERS:-8}"
export PRECISION="${FACTOR_CONT_PRECISION:-bf16-mixed}"
export SEED="${FACTOR_CONT_SEED:-111123}"
export DUMMY_GPU_LOAD=false
export RESUME_CHECKPOINT="${FACTOR_CONT_RESUME_CHECKPOINT:-}"

echo "FACTORIZED_CONT_TASK=${SLURM_ARRAY_TASK_ID} RUN=${FACTOR_CONT_RUN_TAG} RANK=${RANK} LAMBDA=${RATE_LAMBDA}"
echo "WARM_START_CHECKPOINT=${WARM_START_CHECKPOINT}"
echo "CONTROL=matching score-context warm-start and 50k joint budget, context flags all off"
# RANK is a codec width here; do not leak it as a distributed-process rank.
export -n RANK
exec bash "${REPO_DIR}/scripts/slurm/train_nfcgs_paper_recipe.slurm"
