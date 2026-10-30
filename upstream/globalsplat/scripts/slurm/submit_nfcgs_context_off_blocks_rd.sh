#!/usr/bin/env bash
# Context OFF: one full-56-channel stream versus four full-56-channel dyadic streams.
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
read -r -a ORDERS <<< "${CONTEXT_OFF_ORDERS:-hilbert}"
(( ${#ORDERS[@]} >= 1 && ${#ORDERS[@]} <= 2 )) || { echo "CONTEXT_OFF_ORDERS must be none, hilbert, or 'none hilbert'" >&2; exit 2; }
SEEN=" "
for order in "${ORDERS[@]}"; do
  case "${order}" in none|hilbert) ;; *) echo "invalid CONTEXT_OFF_ORDERS: ${order}" >&2; exit 2 ;; esac
  [[ "${SEEN}" != *" ${order} "* ]] || { echo "duplicate order: ${order}" >&2; exit 2; }
  SEEN+="${order} "
done
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt}"
export SOURCE_CHECKPOINT
export SPATIAL_CONTEXT=false KERNEL=5 MAX_STEPS=50000 CHECKPOINT_EVERY=5000 TRAIN_TIME_LIMIT=6-00:00:00
export SCORE_SLICE_CHANNELS=56
export NN_ORDER_BACKEND=legacy NN_ORDER_DISTANCE_DEVICE=cpu NN_ORDER_CANDIDATES=64 SCORE_ORDER_SCALE=""
unset EVAL_RUN_DIR
STATE_DIR="${CONTEXT_OFF_BLOCKS_STATE_DIR:-${REPO_DIR}/logs/slurm/context_off_blocks_rd_20261017}"
RATES=(0.0064 0.0128 0.0256)
grep -Fq 'FLAG_SINGLE' globalsplat/compression/score_context.py || {
  echo "Upload updated globalsplat/compression/score_context.py before submitting; no jobs submitted." >&2; exit 1;
}
grep -Fq 'single_block_marker' globalsplat/compression/checkpoint.py || {
  echo "Upload updated globalsplat/compression/checkpoint.py before submitting; no jobs submitted." >&2; exit 1;
}
grep -Fq 'CHECKPOINT_TMPDIR=' scripts/slurm/train_nfcgs_order_context.slurm || {
  echo "Upload the checkpoint TMPDIR fix before submitting; no jobs submitted." >&2; exit 1;
}
point() {
  local order="$1" schedule="$2" stages="$3" rate="$4" preview="$5"
  local short=one
  [[ "${schedule}" != dyadic4 ]] || short=d4
  TOKEN_ORDER="${order}" CONTEXT_SCHEDULE="${schedule}" STAGES="${stages}" RATE_LAMBDA="${rate}" DRY_RUN="${preview}" \
    SUBMISSION_RECORD="${STATE_DIR}/submissions.tsv" \
    SUBMISSION_STATE_FILE="${STATE_DIR}/${order}_${schedule}_${rate/./p}.state" \
    TRAIN_JOB_NAME="gs-of-${order}-${short}-${rate/./p}" EVAL_JOB_NAME="gs-of-e-${order}-${short}-${rate/./p}" \
    bash scripts/slurm/submit_nfcgs_order_context.sh
}
all_points() {
  local preview="$1" order schedule stages rate
  for order in "${ORDERS[@]}"; do
    for schedule in single dyadic4; do
      stages=1
      [[ "${schedule}" != dyadic4 ]] || stages=4
      for rate in "${RATES[@]}"; do point "${order}" "${schedule}" "${stages}" "${rate}" "${preview}"; done
    done
  done
}
# Validate all saved state and CLI settings before accepting even the first job.
all_points 1
COUNT=$(( ${#ORDERS[@]} * 6 ))
[[ "${DRY_RUN:-0}" != 1 ]] || { echo "PREVIEW ONLY: ${COUNT} training points; full-56-channel Score streams=1/4; no jobs submitted"; exit 0; }
mkdir -p -- "${STATE_DIR}"
LOCK="${STATE_DIR}/.submit-lock"
mkdir -- "${LOCK}" || { echo "Submission lock exists: ${LOCK}; no jobs submitted." >&2; exit 1; }
trap 'rmdir -- "${LOCK}"' EXIT
all_points 1
printf 'order\tschedule\tstages\tlambda\ttrain_job\teval_job\tsource_checkpoint\toutput\n' > "${STATE_DIR}/submissions.tsv"
all_points 0
echo "Context-OFF blocks ready: ${COUNT} training jobs (2 GPUs each) plus ${COUNT} afterok full-scene evals."
echo "SCORE_CHANNEL_GROUPS=56 SINGLE_STRINGS=1 DYADIC4_STRINGS=4"
echo "SUBMISSION_RECORD=${STATE_DIR}/submissions.tsv"
echo "Rerun the same command to reuse accepted IDs and retry missing submissions."
