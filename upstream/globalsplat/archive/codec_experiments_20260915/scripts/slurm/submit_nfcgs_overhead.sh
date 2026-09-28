#!/usr/bin/env bash
# Submit one read-only, sequential overhead benchmark on a single physical GPU.
set -euo pipefail
REPO_DIR="$(pwd -P)"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run from the GlobalSplat repository root" >&2; exit 1;
}
MODE="${1:-submit}"
[[ "${MODE}" == submit || "${MODE}" == dry-run ]] || {
  echo "Usage: bash $0 [submit|dry-run]" >&2; exit 1;
}
PROFILE_RUN_TAG="${PROFILE_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export PROFILE_RUN_TAG
if [[ "${MODE}" == dry-run ]]; then
  bash scripts/slurm/profile_nfcgs_overhead.slurm plan
  echo "DRY_RUN no job submitted"
  exit 0
fi
command -v sbatch >/dev/null 2>&1 || {
  echo "ERROR: sbatch is not available; run this on the Slurm login node" >&2; exit 1;
}
mkdir -p "${REPO_DIR}/logs/slurm"
EXPORTS="ALL,PROFILE_RUN_TAG=${PROFILE_RUN_TAG},PROFILE_SUITE=${PROFILE_SUITE:-all},PROFILE_WARMUP_SCENES=${PROFILE_WARMUP_SCENES:-4},PROFILE_MEASURE_SCENES=${PROFILE_MEASURE_SCENES:-32}"
JOB_RAW="$(sbatch --parsable --nodelist=ariel-v12 --gres=gpu:1 --export="${EXPORTS}" scripts/slurm/profile_nfcgs_overhead.slurm)"
JOB_ID="${JOB_RAW%%;*}"
[[ "${JOB_ID}" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected sbatch response: ${JOB_RAW}" >&2; exit 1; }
echo "submitted overhead=${JOB_RAW} run=${PROFILE_RUN_TAG} suite=${PROFILE_SUITE:-all} warmup=${PROFILE_WARMUP_SCENES:-4} measure=${PROFILE_MEASURE_SCENES:-32}"

