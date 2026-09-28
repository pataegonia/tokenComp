#!/usr/bin/env bash
# Submit the v10 continuation after a successful parent training job.

set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 PARENT_JOB_ID" >&2
  exit 2
fi
PARENT_JOB_ID="$1"
if [[ ! "${PARENT_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "PARENT_JOB_ID must be numeric" >&2
  exit 2
fi

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd "${REPO_DIR}"
mkdir -p logs/slurm

REQUIRED_FILES=(
  scripts/prepare_nfcgs_extension_checkpoint.py
  scripts/resolve_nfcgs_slurm_checkpoint.py
  scripts/run_nfcgs.py
  scripts/slurm/extend_nfcgs_joint_v10.slurm
)
MISSING=0
for FILE in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${FILE}" ]]; then
    echo "missing required file: ${REPO_DIR}/${FILE}" >&2
    MISSING=1
  fi
done
if [[ "${MISSING}" -ne 0 ]]; then
  echo "extension job was not submitted" >&2
  exit 1
fi

SUBMITTED="$(sbatch --parsable \
  --dependency="afterok:${PARENT_JOB_ID}" \
  --export="ALL,PARENT_JOB_ID=${PARENT_JOB_ID}" \
  scripts/slurm/extend_nfcgs_joint_v10.slurm)"
JOB_ID="${SUBMITTED%%;*}"
echo "submitted extension job ${JOB_ID} afterok:${PARENT_JOB_ID}"
echo "squeue -j ${JOB_ID} -o '%.18i %.20j %.10T %.30R'"
