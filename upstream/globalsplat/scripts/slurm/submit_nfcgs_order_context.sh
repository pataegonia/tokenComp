#!/usr/bin/env bash
# Select order and context independently. Submits one train and afterok full eval.
set -euo pipefail
if [[ -n "${REPO_DIR:-}" ]]; then
  if ! cd -P -- "${REPO_DIR}"; then
    echo "Cannot enter REPO_DIR=${REPO_DIR}. Run 'cd /', then cd to the existing repository path." >&2
    exit 1
  fi
else
  if ! SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" \
      || ! cd -P -- "${SCRIPT_DIR}/../.."; then
    echo "Cannot resolve repository directory. The shell may be inside a deleted/replaced directory." >&2
    echo "Run 'cd /', cd to the existing repository path, and retry (or set an absolute REPO_DIR)." >&2
    exit 1
  fi
fi
if ! REPO_DIR="$(pwd -P)" || [[ "${REPO_DIR}" != /* ]]; then
  echo "Cannot resolve an absolute repository path; no jobs submitted. Re-enter the repository from 'cd /'." >&2
  exit 1
fi
[[ -f scripts/run_nfcgs.py ]] || { echo "Not a GlobalSplat repository: ${REPO_DIR}" >&2; exit 1; }
source scripts/slurm/nfcgs_order_context_settings.sh
for file in scripts/slurm/{train,eval}_nfcgs_order_context.slurm; do
  [[ -s "${file}" ]] || { echo "missing server file: ${file}" >&2; exit 1; }
  [[ "$(head -n 1 "${file}")" == '#!/usr/bin/env bash' ]] || { echo "invalid shebang or CRLF/BOM: ${file}" >&2; exit 1; }
done
# Only .0064/.0256 have FullSplit parents. Added rates warm start from .0256.
SOURCE_TAG="${TAG}"
case "${RATE_LAMBDA}" in 0.0064|0.0256) ;; *) SOURCE_TAG=0p0256 ;; esac
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda${SOURCE_TAG}/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda${SOURCE_TAG}_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt}"
[[ -s "${SOURCE_CHECKPOINT}" ]] || {
  echo "Missing source checkpoint (no jobs submitted): ${SOURCE_CHECKPOINT}" >&2
  echo "Check the existing FullSplit parent after re-entering the repository; set SOURCE_CHECKPOINT explicitly if its path changed." >&2
  exit 1
}
if ! SOURCE_CHECKPOINT="$(realpath -e -- "${SOURCE_CHECKPOINT}")"; then
  echo "Source checkpoint could not be resolved; no jobs submitted." >&2
  exit 1
fi
echo "REPO_DIR=${REPO_DIR} SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT}"
export REPO_DIR SOURCE_CHECKPOINT
# An optional per-point state file makes an interrupted sweep safe to resume.
# Persist the training ID before submitting eval so a rejected eval cannot
# orphan a successfully submitted training job in the sweep's records.
TRAIN_JOB_ID=""
EVAL_JOB_ID=""
if [[ -n "${SUBMISSION_STATE_FILE:-}" ]]; then
  # Preserve old state fingerprints for the original defaults. Optimized
  # submissions must not reuse a legacy training/eval ID accidentally.
  NN_SIGNATURE=()
  if [[ "${NN_ORDER_BACKEND}" != legacy || "${NN_ORDER_CANDIDATES}" != 64 || "${NN_ORDER_DISTANCE_DEVICE}" != cpu ]]; then
    NN_SIGNATURE=(nn-implementation "${NN_ORDER_BACKEND}" "${NN_ORDER_CANDIDATES}" "${NN_ORDER_DISTANCE_DEVICE}")
  fi
  [[ -z "${TRAIN_TIME_LIMIT:-}" ]] || NN_SIGNATURE+=(train-time-limit "${TRAIN_TIME_LIMIT}")
  [[ "${SPATIAL_CONTEXT}" == true ]] || NN_SIGNATURE+=(spatial-context false)
  [[ "${SCORE_SLICE_CHANNELS}" == 16 ]] || NN_SIGNATURE+=(score-slice-channels "${SCORE_SLICE_CHANNELS}")
  SUBMISSION_SIGNATURE="$(printf '%s\0' order-context-v1 "${REPO_DIR}" "${SOURCE_CHECKPOINT}" \
    "${TOKEN_ORDER}" "${CONTEXT_SCHEDULE}" "${STAGES}" "${KERNEL}" "${RATE_LAMBDA}" \
    "${MAX_STEPS}" "${CHECKPOINT_EVERY}" "${SCORE_ORDER_SCALE}" "${NN_SIGNATURE[@]}" | sha256sum)"
  SUBMISSION_SIGNATURE="${SUBMISSION_SIGNATURE%% *}"
  if [[ -e "${SUBMISSION_STATE_FILE}" ]]; then
    STATE_VERSION="" STATE_SIGNATURE="" STATE_TRAIN="" STATE_EVAL="" STATE_EXTRA=""
    if ! read -r STATE_VERSION STATE_SIGNATURE STATE_TRAIN STATE_EVAL STATE_EXTRA < "${SUBMISSION_STATE_FILE}" \
        || [[ "${STATE_VERSION}" != v1 || "${STATE_SIGNATURE}" != "${SUBMISSION_SIGNATURE}" \
          || ! "${STATE_TRAIN}" =~ ^[0-9]+$ || -n "${STATE_EXTRA}" ]] \
        || [[ "${STATE_EVAL}" != - && ! "${STATE_EVAL}" =~ ^[0-9]+$ ]]; then
      echo "Invalid or mismatched submission state: ${SUBMISSION_STATE_FILE}; no jobs submitted." >&2
      exit 1
    fi
    TRAIN_JOB_ID="${STATE_TRAIN}"
    [[ "${STATE_EVAL}" == - ]] || EVAL_JOB_ID="${STATE_EVAL}"
  fi
fi
save_submission_state() {
  [[ -n "${SUBMISSION_STATE_FILE:-}" ]] || return 0
  mkdir -p -- "$(dirname -- "${SUBMISSION_STATE_FILE}")"
  local temporary_state
  temporary_state="$(mktemp "${SUBMISSION_STATE_FILE}.tmp.XXXXXX")"
  printf 'v1\t%s\t%s\t%s\n' "${SUBMISSION_SIGNATURE}" "${TRAIN_JOB_ID}" "${EVAL_JOB_ID:--}" > "${temporary_state}"
  mv -f -- "${temporary_state}" "${SUBMISSION_STATE_FILE}"
}
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  python scripts/run_nfcgs.py train --checkpoint "${SOURCE_CHECKPOINT}" \
    --rate-lambda "${RATE_LAMBDA}" "${CODEC_ARGS[@]}" --allow-score-path-conversion --dry-run >/dev/null
  echo "PREVIEW ORDER=${TOKEN_ORDER} SCHEDULE=${CONTEXT_SCHEDULE} STAGES=${STAGES} LAMBDA=${RATE_LAMBDA} GPUS=2 EFFECTIVE_BATCH=8 MAX_STEPS=${MAX_STEPS}"
  echo "PREVIEW SPATIAL_CONTEXT=${SPATIAL_CONTEXT} SCORE_PATH=minimal"
  echo "PREVIEW NN_BACKEND=${NN_ORDER_BACKEND} NN_DISTANCE_DEVICE=${NN_ORDER_DISTANCE_DEVICE} NN_CANDIDATES=${NN_ORDER_CANDIDATES}"
  [[ -z "${TRAIN_TIME_LIMIT:-}" ]] || echo "PREVIEW TRAIN_TIME_LIMIT=${TRAIN_TIME_LIMIT}"
  echo "PREVIEW TRAIN_NAME=${TRAIN_JOB_NAME:-gs-order-context} EVAL_NAME=${EVAL_JOB_NAME:-gs-order-context-eval} OUTPUT=${RUN_ROOT}/job_<TRAIN_JOB_ID> DEPENDENCY=afterok:<TRAIN_JOB_ID>"
  exit 0
fi
mkdir -p logs/slurm
TRAIN_OPTIONS=()
EVAL_OPTIONS=()
[[ -z "${TRAIN_JOB_NAME:-}" ]] || TRAIN_OPTIONS+=(--job-name="${TRAIN_JOB_NAME}")
[[ -z "${TRAIN_TIME_LIMIT:-}" ]] || TRAIN_OPTIONS+=(--time="${TRAIN_TIME_LIMIT}")
[[ -z "${EVAL_JOB_NAME:-}" ]] || EVAL_OPTIONS+=(--job-name="${EVAL_JOB_NAME}")
if [[ -z "${TRAIN_JOB_ID}" ]]; then
  TRAIN_SUBMISSION="$(sbatch --parsable "${TRAIN_OPTIONS[@]}" --export=ALL scripts/slurm/train_nfcgs_order_context.slurm)"
  TRAIN_JOB_ID="${TRAIN_SUBMISSION%%;*}"
  [[ "${TRAIN_JOB_ID}" =~ ^[0-9]+$ ]] || { echo "invalid sbatch response: ${TRAIN_SUBMISSION}" >&2; exit 1; }
  echo "TRAIN_JOB=${TRAIN_JOB_ID} ORDER=${TOKEN_ORDER} SCHEDULE=${CONTEXT_SCHEDULE} STAGES=${STAGES}"
  save_submission_state
else
  echo "REUSING TRAIN_JOB=${TRAIN_JOB_ID} STATE=${SUBMISSION_STATE_FILE}"
fi
export TRAIN_JOB_ID
if [[ -z "${EVAL_JOB_ID}" ]]; then
  if ! EVAL_SUBMISSION="$(sbatch --parsable "${EVAL_OPTIONS[@]}" --dependency="afterok:${TRAIN_JOB_ID}" --export=ALL scripts/slurm/eval_nfcgs_order_context.slurm)"; then
    echo "Train ${TRAIN_JOB_ID} was submitted; eval submission failed." >&2
    if [[ -n "${SUBMISSION_STATE_FILE:-}" ]]; then
      echo "Its ID is saved in ${SUBMISSION_STATE_FILE}; rerun the same sweep to submit only missing jobs." >&2
    else
      echo "Submit eval only with TRAIN_JOB_ID=${TRAIN_JOB_ID} and --dependency=afterok:${TRAIN_JOB_ID}." >&2
    fi
    exit 1
  fi
  EVAL_JOB_ID="${EVAL_SUBMISSION%%;*}"
  [[ "${EVAL_JOB_ID}" =~ ^[0-9]+$ ]] || { echo "invalid eval sbatch response: ${EVAL_SUBMISSION}; train ${TRAIN_JOB_ID} is already submitted" >&2; exit 1; }
  save_submission_state
else
  echo "REUSING EVAL_JOB=${EVAL_JOB_ID} STATE=${SUBMISSION_STATE_FILE}"
fi
if [[ -n "${SUBMISSION_RECORD:-}" ]]; then
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${TOKEN_ORDER}" "${CONTEXT_SCHEDULE}" "${STAGES}" \
    "${RATE_LAMBDA}" "${TRAIN_JOB_ID}" "${EVAL_JOB_ID}" "${SOURCE_CHECKPOINT}" "${RUN_ROOT}/job_${TRAIN_JOB_ID}" >> "${SUBMISSION_RECORD}"
fi
echo "TRAIN_JOB=${TRAIN_JOB_ID} EVAL_JOB=${EVAL_JOB_ID} OUTPUT=${RUN_ROOT}/job_${TRAIN_JOB_ID}"
echo "squeue -j ${TRAIN_JOB_ID},${EVAL_JOB_ID} -o '%.18i %.24j %.10T %.30R'"
