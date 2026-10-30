#!/usr/bin/env bash
# Four training points: Morton quarter2 x three rates, then a Score-only NN probe.
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
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt}"
export SOURCE_CHECKPOINT
# Pin the agreed recipe even if the shell still exports a previous sweep's knobs.
export KERNEL=5 MAX_STEPS=50000 CHECKPOINT_EVERY=5000 SCORE_ORDER_SCALE=""
STATE_DIR="${FOLLOWUP_STATE_DIR:-${REPO_DIR}/logs/slurm/order_context_followup_20261011}"
POINTS=("morton quarter2 2 0.0064 mq2" "morton quarter2 2 0.0128 mq2"
        "morton quarter2 2 0.0256 mq2" "nn_score legacy 3 0.0256 ns3")
point() {
  local order="$1" schedule="$2" stages="$3" rate="$4" label="$5" preview="$6"
  TOKEN_ORDER="${order}" CONTEXT_SCHEDULE="${schedule}" STAGES="${stages}" RATE_LAMBDA="${rate}" \
    DRY_RUN="${preview}" SUBMISSION_RECORD="${STATE_DIR}/submissions.tsv" \
    SUBMISSION_STATE_FILE="${STATE_DIR}/${order}_${schedule}_${rate/./p}.state" \
    TRAIN_JOB_NAME="gs-oc-${label}-${rate/./p}" EVAL_JOB_NAME="gs-oc-e-${label}-${rate/./p}" \
    bash scripts/slurm/submit_nfcgs_order_context.sh
}
# Check all active points, including existing state, before the first sbatch.
for spec in "${POINTS[@]}"; do
  read -r order schedule stages rate label <<< "${spec}"
  point "${order}" "${schedule}" "${stages}" "${rate}" "${label}" 1
done
[[ "${DRY_RUN:-0}" != 1 ]] || { echo "PREVIEW ONLY: 4 training points; no jobs submitted"; exit 0; }
mkdir -p -- "${STATE_DIR}"
LOCK_DIR="${STATE_DIR}/.submit-lock"
if ! mkdir -- "${LOCK_DIR}"; then
  echo "Another submission is active, or a previous submission left ${LOCK_DIR}. No jobs submitted." >&2
  exit 1
fi
trap 'rmdir -- "${LOCK_DIR}"' EXIT
printf 'order\tschedule\tstages\tlambda\ttrain_job\teval_job\tsource_checkpoint\toutput\n' > "${STATE_DIR}/submissions.tsv"
echo "SUBMISSION_STATE_DIR=${STATE_DIR}"
for spec in "${POINTS[@]}"; do
  read -r order schedule stages rate label <<< "${spec}"
  point "${order}" "${schedule}" "${stages}" "${rate}" "${label}" 0
done
echo "Followup ready: 4 training jobs (2 normal GPUs each, up to 8 training GPUs), each with its own afterok full-scene eval."
echo "SUBMISSION_RECORD=${STATE_DIR}/submissions.tsv"
echo "Rerun this same command to reuse recorded job IDs and submit only missing jobs."
