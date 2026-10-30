#!/usr/bin/env bash
# Six missing L3 factorial endpoints and the matched D4 spatial-OFF control.
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
STATE_DIR="${PAPER_CONTROLS_STATE_DIR:-${REPO_DIR}/logs/slurm/paper_controls_20261018}"
ARMS=(none_legacy3_on hilbert_legacy3_off12 none_legacy3_off12)
RATES=(0.0032 0.0512)

grep -Fq 'CHECKPOINT_TMPDIR=' scripts/slurm/train_nfcgs_order_context.slurm || {
  echo "Upload the Ceph checkpoint TMPDIR fix before submitting; no jobs submitted." >&2; exit 1;
}

point() {
  local arm="$1" rate="$2" preview="$3" order schedule stages context short
  schedule=legacy; stages=3; context=false
  case "${arm}" in
    none_legacy3_on) order=none; context=true; short=nl3on ;;
    hilbert_legacy3_off12) order=hilbert; short=hl3off ;;
    none_legacy3_off12) order=none; short=nl3off ;;
    hilbert_dyadic4_off16) order=hilbert; schedule=dyadic4; stages=4; short=hd4off ;;
    *) echo "invalid arm: ${arm}" >&2; return 2 ;;
  esac
  TOKEN_ORDER="${order}" CONTEXT_SCHEDULE="${schedule}" STAGES="${stages}" SPATIAL_CONTEXT="${context}" \
    SCORE_SLICE_CHANNELS=16 RATE_LAMBDA="${rate}" DRY_RUN="${preview}" \
    SUBMISSION_RECORD="${STATE_DIR}/submissions.tsv" \
    SUBMISSION_STATE_FILE="${STATE_DIR}/${arm}_${rate/./p}.state" \
    TRAIN_JOB_NAME="gs-p-${short}-${rate/./p}" EVAL_JOB_NAME="gs-p-e-${short}-${rate/./p}" \
    bash scripts/slurm/submit_nfcgs_order_context.sh
}

all_points() {
  local preview="$1" arm rate
  for rate in "${RATES[@]}"; do
    for arm in "${ARMS[@]}"; do point "${arm}" "${rate}" "${preview}"; done
  done
  point hilbert_dyadic4_off16 0.0128 "${preview}"
}

# Validate all seven configurations and saved IDs before any sbatch call.
all_points 1
[[ "${DRY_RUN:-0}" != 1 ]] || { echo "PREVIEW ONLY: 7 training points; no jobs submitted"; exit 0; }
mkdir -p -- "${STATE_DIR}"
LOCK="${STATE_DIR}/.submit-lock"
mkdir -- "${LOCK}" || { echo "Submission lock exists: ${LOCK}; no jobs submitted." >&2; exit 1; }
trap 'rmdir -- "${LOCK}"' EXIT
all_points 1
printf 'order\tschedule\tstages\tlambda\ttrain_job\teval_job\tsource_checkpoint\toutput\n' > "${STATE_DIR}/submissions.tsv"
all_points 0
echo "Paper controls ready: 7 training jobs (2 GPUs each), plus 7 afterok full-scene evals."
echo "SCORE_STRINGS=L3:12,D4:16; SCORE_SLICE_CHANNELS=16; MSH=on; MAX_STEPS=50000."
echo "SUBMISSION_RECORD=${STATE_DIR}/submissions.tsv"
echo "Rerun the same command to reuse accepted IDs and retry missing train/eval submissions."
