#!/usr/bin/env bash
# Missing legacy3 factorial control: original slots, no Score spatial prediction.
set -euo pipefail
case "${1:-}" in
  "") ;;
  --dry-run) export DRY_RUN=1 ;;
  *) echo "usage: bash $0 [--dry-run]" >&2; exit 2 ;;
esac
(( $# <= 1 )) || { echo "unexpected extra arguments" >&2; exit 2; }
REPO_DIR="${REPO_DIR:-$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd -P -- "${REPO_DIR}"
REPO_DIR="$(pwd -P)"
export REPO_DIR
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt}"
export SOURCE_CHECKPOINT
export TOKEN_ORDER=none SPATIAL_CONTEXT=false CONTEXT_SCHEDULE=legacy STAGES=3 KERNEL=5
export MAX_STEPS=50000 CHECKPOINT_EVERY=5000 TRAIN_TIME_LIMIT=6-00:00:00
export NN_ORDER_BACKEND=legacy NN_ORDER_DISTANCE_DEVICE=cpu NN_ORDER_CANDIDATES=64
export SCORE_ORDER_SCALE=""
unset EVAL_RUN_DIR
STATE_DIR="${SORT_CONTEXT_BOTH_OFF_STATE_DIR:-${REPO_DIR}/logs/slurm/sort_context_both_off_rd_20261016}"
RATES=(0.0064 0.0128 0.0256)
# Checkpoint transactions must use the Ceph staging fix before starting new jobs.
grep -Fq 'CHECKPOINT_TMPDIR=' scripts/slurm/train_nfcgs_order_context.slurm || {
  echo "Upload the updated scripts/slurm/train_nfcgs_order_context.slurm (CHECKPOINT_TMPDIR fix) first; no jobs submitted." >&2
  exit 1
}
point() {
  local rate="$1" preview="$2"
  RATE_LAMBDA="${rate}" DRY_RUN="${preview}" \
    SUBMISSION_RECORD="${STATE_DIR}/submissions.tsv" \
    SUBMISSION_STATE_FILE="${STATE_DIR}/both_off_${rate/./p}.state" \
    TRAIN_JOB_NAME="gs-bo-${rate/./p}" EVAL_JOB_NAME="gs-bo-e-${rate/./p}" \
    bash scripts/slurm/submit_nfcgs_order_context.sh
}
preview_all() {
  local rate
  for rate in "${RATES[@]}"; do point "${rate}" 1; done
}
# Validate every point and any saved IDs before the first submission.
preview_all
[[ "${DRY_RUN:-0}" != 1 ]] || { echo "PREVIEW ONLY: 3 training points; no jobs submitted"; exit 0; }
mkdir -p -- "${STATE_DIR}"
LOCK="${STATE_DIR}/.submit-lock"
mkdir -- "${LOCK}" || { echo "Submission lock exists: ${LOCK}; no jobs submitted." >&2; exit 1; }
trap 'rmdir -- "${LOCK}"' EXIT
preview_all
printf 'order\tschedule\tstages\tlambda\ttrain_job\teval_job\tsource_checkpoint\toutput\n' > "${STATE_DIR}/submissions.tsv"
for rate in "${RATES[@]}"; do point "${rate}" 0; done
echo "Both-OFF control ready: 3 training jobs (2 GPUs each), with 3 afterok full-scene evals."
echo "SUBMISSION_RECORD=${STATE_DIR}/submissions.tsv"
echo "Rerun the same command to reuse job IDs and retry missing submissions."
