#!/usr/bin/env bash
# Submit only the matched 32-scene evals and their comparison report.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 TRAIN_ARRAY_JOB_ID" >&2
  exit 2
fi

TRAIN_ID="$1"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd "${REPO_DIR}"
mkdir -p logs/slurm
MAX_STEPS="${MAX_STEPS:-20000}"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_STEPS must be positive" >&2
  exit 2
}

STEP_TAG="$(printf '%09d' "${MAX_STEPS}")"
ROOT="${REPO_DIR}/outputs/nfcgs_main/score_mean_offset_ablation/job_${TRAIN_ID}"
for VARIANT in baseline no_mean_offset; do
  CHECKPOINT="${ROOT}/${VARIANT}/checkpoints/nfcgs_main/version_0/step${STEP_TAG}.ckpt"
  [[ -f "${CHECKPOINT}" ]] || {
    echo "missing trained checkpoint: ${CHECKPOINT}" >&2
    exit 1
  }
done

EVAL_SUBMISSION="$(sbatch --parsable \
  --export="ALL,TRAIN_ARRAY_JOB_ID=${TRAIN_ID},MAX_STEPS=${MAX_STEPS}" \
  scripts/slurm/eval_nfcgs_mean_offset_ablation_v11.slurm)"
EVAL_ID="${EVAL_SUBMISSION%%;*}"
REPORT_SUBMISSION="$(sbatch --parsable \
  --dependency="afterok:${EVAL_ID}" \
  --export="ALL,TRAIN_ARRAY_JOB_ID=${TRAIN_ID}" \
  scripts/slurm/report_nfcgs_mean_offset_ablation.slurm)"
REPORT_ID="${REPORT_SUBMISSION%%;*}"

echo "eval array ${EVAL_ID}: baseline (0), no_mean_offset (1)"
echo "dependent report ${REPORT_ID}: comparison.json"
echo "squeue -j ${EVAL_ID},${REPORT_ID} -o '%.18i %.24j %.10T %.30R'"
