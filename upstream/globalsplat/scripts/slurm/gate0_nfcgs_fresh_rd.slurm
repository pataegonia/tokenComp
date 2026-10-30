#!/usr/bin/env bash
#SBATCH -J gs-fresh-gate0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:normal:1
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-fresh-gate0-%j.out
#SBATCH -e logs/slurm/slurm-gs-fresh-gate0-%j.err
set -euo pipefail
source "${REPO_DIR:?}/scripts/slurm/fresh_rd_environment.sh"
python scripts/check_nfcgs_fresh_rd_gate0.py "${GATE0_TASK:?}" --root "$FRESH_OUTPUT_ROOT" \
  --preparation "$PREPARATION_DIR" --native-checkpoint "${FRESH_NATIVE_CHECKPOINT:?}" --dataset-root "$DATASET_ROOT"
