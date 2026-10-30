#!/usr/bin/env bash
# Three context-only schedules (original slots) and Hilbert-only, three rates.
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
export CONTEXT_SCHEDULE=legacy STAGES=3 KERNEL=5 MAX_STEPS=50000 CHECKPOINT_EVERY=5000
export NN_ORDER_BACKEND=legacy NN_ORDER_DISTANCE_DEVICE=cpu NN_ORDER_CANDIDATES=64
export SCORE_ORDER_SCALE="" TRAIN_TIME_LIMIT=6-00:00:00
unset EVAL_RUN_DIR
STATE_DIR="${SORT_CONTEXT_ABLATION_STATE_DIR:-${REPO_DIR}/logs/slurm/sort_context_ablation_rd_20261013}"
# Keep the original arms/state names so already-submitted jobs are reused.
ARMS=(context_only hilbert_only context_quarter2 context_dyadic4)
RATES=(0.0064 0.0128 0.0256)
point() {
  local arm="$1" rate="$2" preview="$3" order context tag scheduffle stages
  case "${arm}" in
    context_only) order=none; context=true; tag=co; schedule=legacy; stages=3 ;;
    hilbert_only) order=hilbert; context=false; tag=ho; schedule=legacy; stages=3 ;;
    context_quarter2) order=none; context=true; tag=cq2; schedule=quarter2; stages=2 ;;
    context_dyadic4) order=none; context=true; tag=cd4; schedule=dyadic4; stages=4 ;;
  esac
  TOKEN_ORDER="${order}" SPATIAL_CONTEXT="${context}" CONTEXT_SCHEDULE="${schedule}" STAGES="${stages}" \
    RATE_LAMBDA="${rate}" DRY_RUN="${preview}" \
    SUBMISSION_RECORD="${STATE_DIR}/submissions.tsv" \
    SUBMISSION_STATE_FILE="${STATE_DIR}/${arm}_${rate/./p}.state" \
    TRAIN_JOB_NAME="gs-${tag}-${rate/./p}" EVAL_JOB_NAME="gs-${tag}-e-${rate/./p}" \
    bash scripts/slurm/submit_nfcgs_order_context.sh
}
preview_all() {
  local arm rate
  for arm in "${ARMS[@]}"; do
    for rate in "${RATES[@]}"; do point "${arm}" "${rate}" 1; done
  done
}
# Validate all twelve configurations and recorded IDs before the first submission.
preview_all
[[ "${DRY_RUN:-0}" != 1 ]] || { echo "PREVIEW ONLY: 12 training points; no jobs submitted"; exit 0; }
mkdir -p -- "${STATE_DIR}"
LOCK="${STATE_DIR}/.submit-lock"
mkdir -- "${LOCK}" || { echo "Submission lock exists: ${LOCK}; no jobs submitted." >&2; exit 1; }
trap 'rmdir -- "${LOCK}"' EXIT
preview_all
printf 'order\tschedule\tstages\tlambda\ttrain_job\teval_job\tsource_checkpoint\toutput\n' > "${STATE_DIR}/submissions.tsv"
for arm in "${ARMS[@]}"; do
  for rate in "${RATES[@]}"; do point "${arm}" "${rate}" 0; done
done
echo "Sort/context ablation ready: 12 training jobs (2 GPUs each), with 12 afterok full-scene evals."
echo "SUBMISSION_RECORD=${STATE_DIR}/submissions.tsv"
echo "Rerun the same command to reuse job IDs and retry missing eval submissions."
