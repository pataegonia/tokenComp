#!/usr/bin/env bash
# Sourced by the fresh RD jobs only.
set -euo pipefail
cd "${REPO_DIR:?missing REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR="${NFCGS_TMP_ROOT:-/ceph_data/${USER}/.tmp/nfcgs}/fresh_rd_${SLURM_JOB_ID:?}"
mkdir -p "$TMPDIR" logs/slurm
: "${FRESH_OUTPUT_ROOT:?}" "${DATASET_ROOT:?}" "${PREPARATION_DIR:?}"
echo "FRESH_NODE=${SLURMD_NODENAME:-$(hostname)} DATASET_ROOT=${DATASET_ROOT} JOB=${SLURM_JOB_ID}"
for split in train test; do
  if [[ ! -s "${DATASET_ROOT}/${split}/index.json" ]]; then
    echo "ERROR: missing RE10K index ${DATASET_ROOT}/${split}/index.json on ${SLURMD_NODENAME:-$(hostname)}" >&2
    exit 1
  fi
done
