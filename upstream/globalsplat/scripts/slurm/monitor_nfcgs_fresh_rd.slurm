#!/usr/bin/env bash
#SBATCH -J gs-fresh-monitor
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:normal:1
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-fresh-monitor-%j.out
#SBATCH -e logs/slurm/slurm-gs-fresh-monitor-%j.err
set -euo pipefail
source "${REPO_DIR:?}/scripts/slurm/fresh_rd_environment.sh"
python scripts/monitor_nfcgs_fresh_rd.py --root "$FRESH_OUTPUT_ROOT" --state "${FRESH_STATE:?}" \
  --phase "${MONITOR_PHASE:-A}" --wave "${MONITOR_WAVE:-1}" --budget "${FRESH_BUDGET:-100000}" \
  --preparation "$PREPARATION_DIR" --native-checkpoint "${FRESH_NATIVE_CHECKPOINT:?}" --dataset-root "$DATASET_ROOT"
