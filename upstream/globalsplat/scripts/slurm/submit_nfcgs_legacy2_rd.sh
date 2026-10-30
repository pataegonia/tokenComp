#!/usr/bin/env bash
# Fill the two missing rates of the Morton even/odd kernel5 + STE control.
set -euo pipefail
case "${1:-}" in
  "") ;;
  --dry-run) export DRY_RUN=1 ;;
  *) echo "usage: bash $0 [--dry-run]" >&2; exit 2 ;;
esac
REPO_DIR="${REPO_DIR:-$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd -P -- "${REPO_DIR}"
REPO_DIR="$(pwd -P)"
export REPO_DIR
# Match the existing lambda=.0256 even/odd run, including its common parent.
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt}"
export SOURCE_CHECKPOINT
export KERNEL=5 MAX_STEPS=50000 CHECKPOINT_EVERY=5000 SCORE_ORDER_SCALE=""
STATE_DIR="${LEGACY2_STATE_DIR:-${REPO_DIR}/logs/slurm/order_context_legacy2_rd_20261012}"
RATES=(0.0064 0.0128)
point() {
  local rate="$1" preview="$2"
  TOKEN_ORDER=morton CONTEXT_SCHEDULE=legacy STAGES=2 RATE_LAMBDA="${rate}" \
    DRY_RUN="${preview}" SUBMISSION_RECORD="${STATE_DIR}/submissions.tsv" \
    SUBMISSION_STATE_FILE="${STATE_DIR}/morton_legacy2_${rate/./p}.state" \
    TRAIN_JOB_NAME="gs-oc-m2-${rate/./p}" EVAL_JOB_NAME="gs-oc-e-m2-${rate/./p}" \
    bash scripts/slurm/submit_nfcgs_order_context.sh
}
# Validate both points and any recorded state before submitting the first job.
for rate in "${RATES[@]}"; do point "${rate}" 1; done
[[ "${DRY_RUN:-0}" != 1 ]] || { echo "PREVIEW ONLY: 2 training points; no jobs submitted"; exit 0; }
mkdir -p -- "${STATE_DIR}"
LOCK_DIR="${STATE_DIR}/.submit-lock"
if ! mkdir -- "${LOCK_DIR}"; then
  echo "Another submission is active, or a previous submission left ${LOCK_DIR}. No jobs submitted." >&2
  exit 1
fi
trap 'rmdir -- "${LOCK_DIR}"' EXIT
printf 'order\tschedule\tstages\tlambda\ttrain_job\teval_job\tsource_checkpoint\toutput\n' > "${STATE_DIR}/submissions.tsv"
echo "SUBMISSION_STATE_DIR=${STATE_DIR}"
for rate in "${RATES[@]}"; do point "${rate}" 0; done
echo "Legacy2 ready: 2 independent training jobs (2 normal GPUs each, up to 4 training GPUs), each with its own afterok full-scene eval."
echo "SUBMISSION_RECORD=${STATE_DIR}/submissions.tsv"
echo "Rerun this same command to reuse recorded job IDs and submit only missing jobs."
