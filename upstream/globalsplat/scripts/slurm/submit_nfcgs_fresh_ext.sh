#!/usr/bin/env bash
# Submit plan §13 fresh 120k runs (train then full test in one allocation)
# from the separate server copy. Usage: bash submit_nfcgs_fresh_ext.sh CELL:LAMBDA ...
# Example: bash scripts/slurm/submit_nfcgs_fresh_ext.sh nx_on:0.0128 msh_n:0.0032
set -euo pipefail
REPO_DIR="$(pwd -P)"
MAIN_REPO="/ceph_data/${USER}/tokencomp/upstream/globalsplat"
if [[ "${REPO_DIR}" == "$(readlink -f "${MAIN_REPO}" 2>/dev/null || echo "${MAIN_REPO}")" ]]; then
  echo "ERROR: run this from the separate copy, not the Wave 2 repository" >&2
  exit 1
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 CELL:LAMBDA [CELL:LAMBDA ...]" >&2
  exit 1
fi
ROOT="${REPO_DIR}/outputs/nfcgs_fresh_rd/branches/B120000"  # shares the frozen test frame cohort
PREP="${REPO_DIR}/outputs/nfcgs_architecture_pilots/initialization_20261019"
for spec in "$@"; do
  cell="${spec%%:*}"
  rate="${spec##*:}"
  # Validate the cell, lambda and the full command before anything is queued.
  python scripts/run_nfcgs_fresh_rd.py train --cell "${cell}" --rate-lambda "${rate}" --seed-index 0 \
    --stage full --budget 120000 --preparation "${PREP}" --run-dir "${ROOT}/${cell}/dry_run" --dry-run >/dev/null
done
for spec in "$@"; do
  cell="${spec%%:*}"
  rate="${spec##*:}"
  sbatch --job-name="gs-fx-${cell}-l${rate#0.}-s0-full120k" \
    --export=ALL,REPO_DIR="${REPO_DIR}",DATASET_ROOT=/data3/local_datasets/re10k,FRESH_OUTPUT_ROOT="${ROOT}",PREPARATION_DIR="${PREP}",CELL="${cell}",RATE_LAMBDA="${rate}",SEED_INDEX=0,FRESH_STAGE=full,FRESH_BUDGET=120000,FRESH_TEST_AFTER_TRAIN=true,FRESH_RESUME=false \
    scripts/slurm/train_nfcgs_fresh_rd.slurm
done
