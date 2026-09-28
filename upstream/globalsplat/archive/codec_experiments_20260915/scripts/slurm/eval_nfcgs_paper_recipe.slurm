#!/usr/bin/env bash
#SBATCH -J gs-nfcgs-paper-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-nfcgs-paper-eval-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-nfcgs-paper-eval-%A_%a.err

set -euo pipefail

# Run from the GlobalSplat repo root. These are the 12 version_0 / step-50000
# files confirmed after the paper24 training grid completed. Recipe identity
# and checkpoint selection intentionally do not depend on inherited env vars.
#
#   bash scripts/slurm/eval_nfcgs_paper_recipe.slurm check
#   sbatch --array=0-7 scripts/slurm/eval_nfcgs_paper_recipe.slurm on
#   sbatch --array=0,2,4,6 scripts/slurm/eval_nfcgs_paper_recipe.slurm off

export REPO_DIR="$(pwd -P)"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run this script from the GlobalSplat repository root" >&2
  exit 1
}
export TRAIN_OUTPUT_ROOT="${REPO_DIR}/outputs/nfcgs_paper24_subset_train"
export OUTPUT_ROOT="${REPO_DIR}/outputs/nfcgs_paper24_subset_eval_corrected"
export EXPECTED_STEP=50000
export TRANSFORM=linear TRANSFORM_HIDDEN=32
unset GRID_TASK_ID
export PROTOCOL=all CONTEXT=12 PRECISION=bf16-mixed SAVE_IMAGES=false
export CHECKPOINT_ONLY=false DRY_RUN=false SLURM_EXPORT_ENV=ALL

select_paper_checkpoint() {
  local task="$1" morton="$2" residual_tag lambda_tag arm experiment
  case "${task}" in
    0) RANK=56; RATE_LAMBDA=0.0064; USE_RESIDUAL=true ;;
    1) RANK=56; RATE_LAMBDA=0.0064; USE_RESIDUAL=false ;;
    2) RANK=56; RATE_LAMBDA=0.0256; USE_RESIDUAL=true ;;
    3) RANK=56; RATE_LAMBDA=0.0256; USE_RESIDUAL=false ;;
    4) RANK=80; RATE_LAMBDA=0.0064; USE_RESIDUAL=true ;;
    5) RANK=80; RATE_LAMBDA=0.0064; USE_RESIDUAL=false ;;
    6) RANK=80; RATE_LAMBDA=0.0256; USE_RESIDUAL=true ;;
    7) RANK=80; RATE_LAMBDA=0.0256; USE_RESIDUAL=false ;;
    *) echo "ERROR: submit with --array=0-7 (on) or --array=0,2,4,6 (off)" >&2; exit 1 ;;
  esac
  residual_tag=off
  [[ "${USE_RESIDUAL}" == "false" ]] || residual_tag=on
  lambda_tag="${RATE_LAMBDA/./p}"
  arm="${TRAIN_OUTPUT_ROOT}/rank${RANK}/lambda${lambda_tag}/residual_${residual_tag}"
  USE_MORTON=true
  if [[ "${morton}" == "off" ]]; then
    [[ "${USE_RESIDUAL}" == "true" ]] || {
      echo "ERROR: the paper24 Morton-off grid only contains residual-on tasks 0,2,4,6" >&2
      exit 1
    }
    USE_MORTON=false
    arm="${arm}/morton_off"
  fi
  experiment="nfcgs_paper24_subset_rank${RANK}_lambda${lambda_tag}_residual_${residual_tag}_morton_${morton}"
  export CHECKPOINT="${arm}/checkpoints/${experiment}/version_0/step000050000.ckpt"
  export RANK RATE_LAMBDA USE_RESIDUAL USE_MORTON
  [[ -f "${CHECKPOINT}" ]] || {
    echo "ERROR: confirmed paper24 checkpoint is missing: ${CHECKPOINT}" >&2
    exit 1
  }
  echo "PAPER24_CHECKPOINT=${CHECKPOINT}"
}

case "${1:-}" in
  check)
    for task in 0 1 2 3 4 5 6 7; do
      select_paper_checkpoint "${task}" on
    done
    for task in 0 2 4 6; do
      select_paper_checkpoint "${task}" off
    done
    echo "All 12 paper24 checkpoint files exist (tensor loading is checked by each eval task)."
    exit 0
    ;;
  on|off)
    select_paper_checkpoint "${SLURM_ARRAY_TASK_ID:-}" "$1"
    ;;
  *)
    echo "Usage: bash $0 check | sbatch --array=... $0 on|off" >&2
    exit 1
    ;;
esac

# Reuse the validated actual-bitstream evaluator with an explicit checkpoint.
# Its #SBATCH directives are ignored here; resources come from this script.
exec bash "${REPO_DIR}/scripts/slurm/eval_nfcgs_rank56.slurm"
