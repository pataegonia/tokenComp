#!/usr/bin/env bash
# Hilbert dyadic4 at rate=0; six selected methods at .0032/.0512.
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
export KERNEL=5 MAX_STEPS=50000 CHECKPOINT_EVERY=5000 TRAIN_TIME_LIMIT=6-00:00:00
export NN_ORDER_BACKEND=legacy NN_ORDER_DISTANCE_DEVICE=cpu NN_ORDER_CANDIDATES=64 SCORE_ORDER_SCALE=""
unset EVAL_RUN_DIR
STATE_DIR="${EXTREMES_STATE_DIR:-${REPO_DIR}/logs/slurm/order_context_extremes_20261017}"
ARMS=(hilbert_dyadic4 morton_dyadic4 hilbert_legacy3 morton_legacy3 none_dyadic4 hilbert_context_off_single)
RATES=(0.0032 0.0512)
grep -Fq 'FLAG_SINGLE' globalsplat/compression/score_context.py || {
  echo "Upload the single-stream codec changes before submitting; no jobs submitted." >&2; exit 1;
}
grep -Fq 'channel regrouping requires' globalsplat/compression/checkpoint.py || {
  echo "Upload the full-channel warm-start changes before submitting; no jobs submitted." >&2; exit 1;
}
grep -Fq 'CHECKPOINT_TMPDIR=' scripts/slurm/train_nfcgs_order_context.slurm || {
  echo "Upload the Ceph checkpoint TMPDIR fix before submitting; no jobs submitted." >&2; exit 1;
}
point() {
  local arm="$1" rate="$2" preview="$3" order schedule stages context channels short
  context=true; channels=16
  case "${arm}" in
    hilbert_dyadic4) order=hilbert; schedule=dyadic4; stages=4; short=hd4 ;;
    morton_dyadic4) order=morton; schedule=dyadic4; stages=4; short=md4 ;;
    hilbert_legacy3) order=hilbert; schedule=legacy; stages=3; short=hl3 ;;
    morton_legacy3) order=morton; schedule=legacy; stages=3; short=ml3 ;;
    none_dyadic4) order=none; schedule=dyadic4; stages=4; short=nd4 ;;
    hilbert_context_off_single) order=hilbert; schedule=single; stages=1; context=false; channels=56; short=h1off ;;
    *) echo "invalid arm: ${arm}" >&2; return 2 ;;
  esac
  TOKEN_ORDER="${order}" CONTEXT_SCHEDULE="${schedule}" STAGES="${stages}" SPATIAL_CONTEXT="${context}" \
    SCORE_SLICE_CHANNELS="${channels}" RATE_LAMBDA="${rate}" DRY_RUN="${preview}" \
    SUBMISSION_RECORD="${STATE_DIR}/submissions.tsv" \
    SUBMISSION_STATE_FILE="${STATE_DIR}/${arm}_${rate/./p}.state" \
    TRAIN_JOB_NAME="gs-x-${short}-${rate/./p}" EVAL_JOB_NAME="gs-x-e-${short}-${rate/./p}" \
    bash scripts/slurm/submit_nfcgs_order_context.sh
}
all_points() {
  local preview="$1" arm rate
  point hilbert_dyadic4 0 "${preview}"
  for rate in "${RATES[@]}"; do
    for arm in "${ARMS[@]}"; do point "${arm}" "${rate}" "${preview}"; done
  done
}
# Validate every configuration and saved ID before the first submission.
all_points 1
[[ "${DRY_RUN:-0}" != 1 ]] || { echo "PREVIEW ONLY: 13 training points; no jobs submitted"; exit 0; }
mkdir -p -- "${STATE_DIR}"
LOCK="${STATE_DIR}/.submit-lock"
mkdir -- "${LOCK}" || { echo "Submission lock exists: ${LOCK}; no jobs submitted." >&2; exit 1; }
trap 'rmdir -- "${LOCK}"' EXIT
all_points 1
printf 'order\tschedule\tstages\tlambda\ttrain_job\teval_job\tsource_checkpoint\toutput\n' > "${STATE_DIR}/submissions.tsv"
all_points 0
echo "RD extension ready: 13 training jobs (2 GPUs each), plus 13 afterok full-scene evals."
echo "RATE_ZERO=Hilbert+dyadic4; CONTEXT_OFF=Hilbert+single+channels56 (one Score string)."
echo "SUBMISSION_RECORD=${STATE_DIR}/submissions.tsv"
echo "Rerun the same command to reuse accepted IDs and retry missing train/eval submissions."
