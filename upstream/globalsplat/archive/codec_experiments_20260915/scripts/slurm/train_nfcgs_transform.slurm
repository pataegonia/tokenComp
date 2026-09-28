#!/usr/bin/env bash
#SBATCH -J gs-nfcgs-transform
#SBATCH --array=8-15
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm/slurm-gs-nfcgs-transform-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-nfcgs-transform-%A_%a.err

set -euo pipefail
# Run from the repository root. Default submission trains only the eight NEW
# nonlinear arms. --array=0-15 also retrains linear controls in a separate tree.
# bash scripts/slurm/train_nfcgs_transform.slurm plan
# mkdir -p logs/slurm
# sbatch scripts/slurm/train_nfcgs_transform.slurm
export REPO_DIR="$(pwd -P)"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run from the GlobalSplat repository root" >&2; exit 1;
}
source "${REPO_DIR}/scripts/slurm/nfcgs_transform_grid.sh"
export MICRO_BATCH=1 ACCUMULATE=8 NUM_WORKERS=8 SEED=111123 PRECISION=bf16-mixed
export DUMMY_GPU_LOAD=false
export MAX_STEPS="${MAX_STEPS:-50000}"
# Do not inherit an unrelated old run's initializer/resume path.
export CODEC_INIT=""
export RESUME_CHECKPOINT="${TRANSFORM_RESUME_CHECKPOINT:-}"
TRAIN_ROOT="${TRANSFORM_TRAIN_ROOT:-${REPO_DIR}/outputs/nfcgs_paper24_transform_train}"

case "${1:-run}" in
  plan)
    for task in {0..15}; do
      select_transform_condition "${task}"
      echo "TASK=${task} TRANSFORM=${TRANSFORM} RANK=${RANK} LAMBDA=${RATE_LAMBDA} RESIDUAL=${USE_RESIDUAL} MORTON=true"
    done
    echo "New training: tasks 8-15; micro=1 accumulate=8 seed=111123 steps=${MAX_STEPS}"
    exit 0
    ;;
  run) ;;
  *) echo "Usage: bash $0 plan | sbatch [--array=0-15] $0" >&2; exit 1 ;;
esac
select_transform_condition "${SLURM_ARRAY_TASK_ID:-}"
export OUTPUT_ROOT="${TRAIN_ROOT}/${TRANSFORM_TAG}"
echo "TRANSFORM_TASK=${SLURM_ARRAY_TASK_ID} OUTPUT_ROOT=${OUTPUT_ROOT}"
exec bash "${REPO_DIR}/scripts/slurm/train_nfcgs_paper_recipe.slurm"
