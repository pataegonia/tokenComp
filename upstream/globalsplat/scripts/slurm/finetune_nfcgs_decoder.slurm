#!/usr/bin/env bash
#SBATCH -J gs-fx-decoder-ft
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:normal:1
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 2-00:00:00
#SBATCH -o logs/slurm/slurm-gs-decoder-ft-%j.out
#SBATCH -e logs/slurm/slurm-gs-decoder-ft-%j.err
# Plan §13.4. Submit from the separate copy:
#   FT_VARIANT=finetune sbatch scripts/slurm/finetune_nfcgs_decoder.slurm   (and FT_VARIANT=control)
set -euo pipefail
export REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
export DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
export FRESH_OUTPUT_ROOT="${FRESH_OUTPUT_ROOT:-${REPO_DIR}/outputs/nfcgs_fresh_rd/branches/B120000}"
export PREPARATION_DIR="${PREPARATION_DIR:-${REPO_DIR}/outputs/nfcgs_architecture_pilots/initialization_20261019}"
source "${REPO_DIR}/scripts/slurm/fresh_rd_environment.sh"
python scripts/run_nfcgs_decoder_finetune.py --variant "${FT_VARIANT:?finetune or control}" \
  ${FT_SOURCE_SHA256:+--source-sha256 "${FT_SOURCE_SHA256}"} \
  --run-dir "${REPO_DIR}/outputs/nfcgs_decoder_finetune/${FT_VARIANT}/job_${SLURM_JOB_ID:?submit with sbatch}" \
  --preparation "${PREPARATION_DIR}" --output-root "${FRESH_OUTPUT_ROOT}" --dataset-root "${DATASET_ROOT}"
