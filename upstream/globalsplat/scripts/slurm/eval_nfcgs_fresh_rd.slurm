#!/usr/bin/env bash
#SBATCH -J gs-fresh-eval
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:normal:1
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-fresh-eval-%j.out
#SBATCH -e logs/slurm/slurm-gs-fresh-eval-%j.err
set -euo pipefail
source "${REPO_DIR:?}/scripts/slurm/fresh_rd_environment.sh"
extra=()
if [[ "${FRESH_EVAL_MODE:-test}" == validation ]]; then extra+=(--step "${FRESH_EVAL_STEP:?}"); fi
python scripts/run_nfcgs_fresh_rd.py "${FRESH_EVAL_MODE:-test}" --cell "${CELL:?}" \
  --rate-lambda "${RATE_LAMBDA:?}" --seed-index "${SEED_INDEX:?}" \
  --budget "${FRESH_BUDGET:?}" --preparation "$PREPARATION_DIR" --run-dir "${FRESH_RUN_DIR:?}" \
  --dataset-root "$DATASET_ROOT" "${extra[@]}"
