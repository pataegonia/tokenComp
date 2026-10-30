#!/usr/bin/env bash
# Add lambda .0064/.0128 to the existing NN-score .0256 experiment.
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
# All three rates start from the same FullSplit .0256 step10k parent.
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt}"
export SOURCE_CHECKPOINT
export TOKEN_ORDER=nn_score CONTEXT_SCHEDULE=legacy STAGES=3 KERNEL=5
export NN_ORDER_BACKEND=knn NN_ORDER_DISTANCE_DEVICE=input NN_ORDER_CANDIDATES=64
export MAX_STEPS=50000 CHECKPOINT_EVERY=5000 SCORE_ORDER_SCALE="" TRAIN_TIME_LIMIT=6-00:00:00
# A previous resume helper may have exported its eval path. These new rates
# must use their own newly submitted training directories.
unset EVAL_RUN_DIR
STATE_DIR="${NN_SCORE_RD_STATE_DIR:-${REPO_DIR}/logs/slurm/nn_score_rd_gpu_20261013}"
RATES=(0.0064 0.0128)
point() {
  local rate="$1" preview="$2"
  RATE_LAMBDA="${rate}" DRY_RUN="${preview}" \
    SUBMISSION_RECORD="${STATE_DIR}/submissions.tsv" \
    SUBMISSION_STATE_FILE="${STATE_DIR}/nn_score_legacy3_${rate/./p}.state" \
    TRAIN_JOB_NAME="gs-ns-${rate/./p}" EVAL_JOB_NAME="gs-ns-e-${rate/./p}" \
    bash scripts/slurm/submit_nfcgs_order_context.sh
}
# Verify both points, parent and existing IDs before submitting any new job.
for rate in "${RATES[@]}"; do point "${rate}" 1; done
[[ "${DRY_RUN:-0}" != 1 ]] || { echo "PREVIEW ONLY: 2 NN-score training points; no jobs submitted"; exit 0; }
mkdir -p -- "${STATE_DIR}"
LOCK="${STATE_DIR}/.submit-lock"
mkdir -- "${LOCK}" || { echo "Another submission is active, or a stale lock exists: ${LOCK}; no jobs submitted." >&2; exit 1; }
trap 'rmdir -- "${LOCK}"' EXIT
# Recheck state while owning the lock, before rebuilding the submission table.
for rate in "${RATES[@]}"; do point "${rate}" 1; done
printf 'order\tschedule\tstages\tlambda\ttrain_job\teval_job\tsource_checkpoint\toutput\n' > "${STATE_DIR}/submissions.tsv"
for rate in "${RATES[@]}"; do point "${rate}" 0; done
echo "NN-score RD ready: 2 training jobs (2 normal GPUs each, effective batch 8), with 2 afterok full-scene evals."
echo "SUBMISSION_RECORD=${STATE_DIR}/submissions.tsv"
echo "Rerun the same command to reuse job IDs and retry missing eval submissions."
