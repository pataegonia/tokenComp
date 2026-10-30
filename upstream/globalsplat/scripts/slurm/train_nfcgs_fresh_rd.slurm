#!/usr/bin/env bash
#SBATCH -J gs-fresh-train
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:normal:1
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-fresh-train-%j.out
#SBATCH -e logs/slurm/slurm-gs-fresh-train-%j.err
set -euo pipefail
source "${REPO_DIR:?}/scripts/slurm/fresh_rd_environment.sh"
rate_tag="${RATE_LAMBDA//./p}"
run="${FRESH_RUN_DIR:-${FRESH_OUTPUT_ROOT}/${CELL:?}/lambda${rate_tag}/s${SEED_INDEX:?}/job_${SLURM_JOB_ID}}"
extra=()
if [[ "${FRESH_RESUME:-false}" == true ]]; then extra+=(--resume); fi
if [[ "${FRESH_STAGE:-A}" != full || "${FRESH_RESUME:-false}" != true || ! -f "$run/stage_full_complete.json" ]]; then
  python scripts/run_nfcgs_fresh_rd.py train --cell "$CELL" --rate-lambda "$RATE_LAMBDA" \
    --seed-index "$SEED_INDEX" --stage "${FRESH_STAGE:-A}" --budget "${FRESH_BUDGET:-100000}" \
    --preparation "$PREPARATION_DIR" --run-dir "$run" --dataset-root "$DATASET_ROOT" "${extra[@]}"
fi
if [[ "${FRESH_TEST_AFTER_TRAIN:-false}" == true ]]; then
  python scripts/run_nfcgs_fresh_rd.py test --cell "$CELL" --rate-lambda "$RATE_LAMBDA" \
    --seed-index "$SEED_INDEX" --stage "${FRESH_STAGE:-full}" --budget "${FRESH_BUDGET:?}" \
    --preparation "$PREPARATION_DIR" --run-dir "$run" --dataset-root "$DATASET_ROOT"
fi
