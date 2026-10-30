#!/usr/bin/env bash
#SBATCH -J gs-fx-topk-smoke
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:normal:1
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 03:00:00
#SBATCH -o logs/slurm/slurm-gs-topk-smoke-%j.out
#SBATCH -e logs/slurm/slurm-gs-topk-smoke-%j.err
# Plan §14 smoke: short audited Top-K trainings for speed, memory and K movement.
# Submit from the separate copy: TOPK_SMOKE_CELLS="tk_h tk_ns" sbatch scripts/slurm/smoke_nfcgs_topk.slurm
set -euo pipefail
export REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
export DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
export FRESH_OUTPUT_ROOT="${FRESH_OUTPUT_ROOT:-${REPO_DIR}/outputs/nfcgs_fresh_rd/branches/B120000}"
export PREPARATION_DIR="${PREPARATION_DIR:-${REPO_DIR}/outputs/nfcgs_architecture_pilots/initialization_20261019}"
source "${REPO_DIR}/scripts/slurm/fresh_rd_environment.sh"
out="${REPO_DIR}/outputs/nfcgs_topk_smoke/job_${SLURM_JOB_ID:?submit with sbatch}"
for cell in ${TOPK_SMOKE_CELLS:-tk_h tk_ns}; do
  echo "TOPK_SMOKE_CELL=${cell}"
  python scripts/run_nfcgs_fresh_rd.py train --cell "${cell}" --rate-lambda 0.0128 --seed-index 0 --stage A \
    --budget 100000 --audit-max-steps "${TOPK_SMOKE_STEPS:-300}" --audit-trace "${out}/${cell}/trace.json" \
    --preparation "${PREPARATION_DIR}" --run-dir "${out}/${cell}" --dataset-root "${DATASET_ROOT}"
done
echo "TOPK_SMOKE_COMPLETE output=${out}"
