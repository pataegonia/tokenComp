#!/usr/bin/env bash
#SBATCH -J gs-fx-topk-ft
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:normal:1
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 2-00:00:00
#SBATCH -o logs/slurm/slurm-gs-topk-ft-%j.out
#SBATCH -e logs/slurm/slurm-gs-topk-ft-%j.err
# Plan §14 Top-K fine-tune (2026-10-30). Submit from the separate copy:
#   TKFT_VARIANT=topk TKFT_LAMBDA=0.0128 sbatch scripts/slurm/finetune_nfcgs_topk.slurm   (and TKFT_VARIANT=control)
set -euo pipefail
export REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
export DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
export FRESH_OUTPUT_ROOT="${FRESH_OUTPUT_ROOT:-${REPO_DIR}/outputs/nfcgs_fresh_rd/branches/B120000}"
export PREPARATION_DIR="${PREPARATION_DIR:-${REPO_DIR}/outputs/nfcgs_architecture_pilots/initialization_20261019}"
source "${REPO_DIR}/scripts/slurm/fresh_rd_environment.sh"
LAMBDA="${TKFT_LAMBDA:?0.0032, 0.0128 or 0.0512}"
python scripts/run_nfcgs_topk_finetune.py --variant "${TKFT_VARIANT:?topk or control}" --rate-lambda "${LAMBDA}" \
  ${TKFT_SOURCE_SHA256:+--source-sha256 "${TKFT_SOURCE_SHA256}"} \
  --run-dir "${REPO_DIR}/outputs/nfcgs_topk_finetune/${TKFT_VARIANT}/lambda${LAMBDA//./p}/job_${SLURM_JOB_ID:?submit with sbatch}" \
  --preparation "${PREPARATION_DIR}" --output-root "${FRESH_OUTPUT_ROOT}" --dataset-root "${DATASET_ROOT}"
