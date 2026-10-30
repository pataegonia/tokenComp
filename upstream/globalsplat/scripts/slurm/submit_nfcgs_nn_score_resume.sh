#!/usr/bin/env bash
# Resume Score-only NN with CPU/input-device distances and effective batch 8.
set -euo pipefail
PREVIEW=0
REPLACE_JOBS=""
while (( $# )); do
  case "$1" in
    --dry-run) PREVIEW=1; shift ;;
    --replace-jobs)
      (( $# >= 2 )) || { echo "--replace-jobs needs comma-separated job IDs" >&2; exit 2; }
      REPLACE_JOBS="$2"; shift 2
      [[ "${REPLACE_JOBS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "invalid replacement job IDs" >&2; exit 2; } ;;
    *) echo "usage: bash $0 [--dry-run] [--replace-jobs 448650,448651]" >&2; exit 2 ;;
  esac
done
REPO_DIR="${REPO_DIR:-$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd -P -- "${REPO_DIR}"
REPO_DIR="$(pwd -P)"
export REPO_DIR
export TOKEN_ORDER=nn_score CONTEXT_SCHEDULE=legacy STAGES=3 KERNEL=5 RATE_LAMBDA=0.0256
export NN_ORDER_BACKEND=knn NN_ORDER_CANDIDATES=64 SCORE_ORDER_SCALE=""
export NN_ORDER_DISTANCE_DEVICE="${NN_ORDER_DISTANCE_DEVICE:-cpu}"
export MAX_STEPS="${NN_SCORE_TARGET_STEPS:-50000}" CHECKPOINT_EVERY=5000
source scripts/slurm/nfcgs_order_context_settings.sh
for file in scripts/prepare_nfcgs_nn_score_resume.py scripts/run_nfcgs.py \
            scripts/slurm/train_nfcgs_nn_score_resume.slurm scripts/slurm/eval_nfcgs_order_context.slurm; do
  [[ -s "${file}" ]] || { echo "missing server file: ${file}; no jobs changed" >&2; exit 1; }
done
for file in scripts/slurm/train_nfcgs_nn_score_resume.slurm scripts/slurm/eval_nfcgs_order_context.slurm; do
  [[ "$(head -n 1 "${file}")" == '#!/usr/bin/env bash' ]] || { echo "invalid shebang/CRLF/BOM: ${file}" >&2; exit 1; }
done
SOURCE_RUN="${NN_SCORE_SOURCE_RUN:-${REPO_DIR}/outputs/nfcgs_order_context/nn_score/legacy_stages3_k5_ste/lambda0p0256/job_448650}"
if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
  # Numbered checkpoints do not get replaced like last.ckpt does during training.
  shopt -s nullglob
  for candidate in "${SOURCE_RUN}"/checkpoints/nfcgs_main/version_0/step?????????.ckpt; do
    [[ -s "${candidate}" && "${candidate##*/}" =~ ^step[0-9]{9}\.ckpt$ ]] && RESUME_CHECKPOINT="${candidate}"
  done
  shopt -u nullglob
  if [[ -z "${RESUME_CHECKPOINT:-}" && -s "${SOURCE_RUN}/checkpoints/nfcgs_main/version_0/last.ckpt" ]]; then
    RESUME_CHECKPOINT="${SOURCE_RUN}/checkpoints/nfcgs_main/version_0/last.ckpt"
  fi
fi
[[ -n "${RESUME_CHECKPOINT:-}" && -s "${RESUME_CHECKPOINT}" ]] || {
  echo "No saved Score checkpoint found under ${SOURCE_RUN}; no jobs changed." >&2; exit 1;
}
RESUME_CHECKPOINT="$(realpath -e -- "${RESUME_CHECKPOINT}")"
export RESUME_CHECKPOINT
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# Activate globalsplat before running this helper. Fail before cancelling any
# old jobs if Numba, the weights, optimizer state or saved configuration is bad.
python -c 'from globalsplat.compression.token_order import _compiled_greedy_knn; _compiled_greedy_knn()'
python scripts/prepare_nfcgs_nn_score_resume.py "${RESUME_CHECKPOINT}" --target-step "${MAX_STEPS}"
python scripts/run_nfcgs.py train --checkpoint "${RESUME_CHECKPOINT}" --resume \
  "${CODEC_ARGS[@]}" --rate-lambda "${RATE_LAMBDA}" --devices 2 --batch-size 2 --accumulate 2 \
  --max-steps "${MAX_STEPS}" --dry-run >/dev/null
SOURCE_JOB=""
if [[ "${RESUME_CHECKPOINT}" =~ /job_([0-9]+)/ ]]; then SOURCE_JOB="${BASH_REMATCH[1]}"; fi
STATE_FILE="${NN_SCORE_RESUME_STATE_FILE:-${REPO_DIR}/logs/slurm/nn_score_resume_${SOURCE_JOB:-custom}_${NN_ORDER_DISTANCE_DEVICE}_knn.state}"
# Keep existing CPU fingerprints valid; GPU submissions use a distinct state.
DEVICE_SIGNATURE=()
[[ "${NN_ORDER_DISTANCE_DEVICE}" == cpu ]] || DEVICE_SIGNATURE=(distance-device "${NN_ORDER_DISTANCE_DEVICE}")
SIGNATURE="$(printf '%s\0' nn-score-resume-v1 "${REPO_DIR}" "${RESUME_CHECKPOINT}" "${MAX_STEPS}" "${DEVICE_SIGNATURE[@]}" | sha256sum)"
SIGNATURE="${SIGNATURE%% *}"
TRAIN_JOB_ID="" EVAL_JOB_ID=""
load_state() {
  [[ -e "${STATE_FILE}" ]] || return 0
  local version signature train eval extra
  if ! read -r version signature train eval extra < "${STATE_FILE}" \
      || [[ "${version}" != v1 || "${signature}" != "${SIGNATURE}" || -n "${extra}" \
         || ! "${train}" =~ ^[0-9]+$ ]] \
      || [[ "${eval}" != - && ! "${eval}" =~ ^[0-9]+$ ]]; then
    echo "Invalid or mismatched resume state: ${STATE_FILE}; no jobs changed." >&2; exit 1
  fi
  TRAIN_JOB_ID="${train}"
  [[ "${eval}" == - ]] || EVAL_JOB_ID="${eval}"
}
load_state
echo "RESUME_CHECKPOINT=${RESUME_CHECKPOINT} NN_BACKEND=knn DISTANCE=${NN_ORDER_DISTANCE_DEVICE} GPUS=2 EFFECTIVE_BATCH=8"
if (( PREVIEW )); then
  echo "PREVIEW REPLACE_JOBS=${REPLACE_JOBS:-none} TRAIN=${TRAIN_JOB_ID:-new} EVAL=${EVAL_JOB_ID:-new} MAX_STEPS=${MAX_STEPS}"
  echo "PREVIEW OUTPUT=${RUN_ROOT}/resume_job_<TRAIN_JOB_ID> TIME_LIMIT=6-days; no jobs changed"
  exit 0
fi
mkdir -p logs/slurm "$(dirname -- "${STATE_FILE}")"
LOCK="${STATE_FILE}.lock"
mkdir -- "${LOCK}" || { echo "Resume submission lock exists: ${LOCK}; no jobs changed." >&2; exit 1; }
trap 'rmdir -- "${LOCK}"' EXIT
load_state
save_state() {
  local temporary
  temporary="$(mktemp "${STATE_FILE}.tmp.XXXXXX")"
  printf 'v1\t%s\t%s\t%s\n' "${SIGNATURE}" "${TRAIN_JOB_ID}" "${EVAL_JOB_ID:--}" > "${temporary}"
  mv -f -- "${temporary}" "${STATE_FILE}"
}
active_replacements() {
  local job queue
  queue="$(squeue -h -u "${USER}" -o '%i')" || return 1
  while read -r job; do
    [[ -n "${job}" ]] || continue
    [[ ",${REPLACE_JOBS}," != *",${job},"* ]] || printf '%s\n' "${job}"
  done <<< "${queue}"
}
if [[ -z "${TRAIN_JOB_ID}" ]]; then
  # Full checkpoint validation above must finish BEFORE cancelling old jobs.
  if [[ -n "${REPLACE_JOBS}" ]]; then
    ACTIVE="$(active_replacements)"
    if [[ -n "${ACTIVE}" ]]; then
      ACTIVE="${ACTIVE//$'\n'/,}"
      scancel "${ACTIVE}"
      for (( attempt=0; attempt<30; attempt++ )); do
        ACTIVE="$(active_replacements)"
        [[ -n "${ACTIVE}" ]] || break
        sleep 1
      done
      ACTIVE="$(active_replacements)"
      [[ -z "${ACTIVE}" ]] || { echo "Old jobs have not stopped; retry later. No new jobs submitted." >&2; exit 1; }
    fi
  elif [[ -n "${SOURCE_JOB}" ]]; then
    QUEUE="$(squeue -h -u "${USER}" -o '%i')"
    while read -r job; do
      [[ "${job}" != "${SOURCE_JOB}" ]] || {
        echo "Source training ${SOURCE_JOB} is active; use --replace-jobs <train>,<eval> to switch it." >&2; exit 1;
      }
    done <<< "${QUEUE}"
  fi
  RESPONSE="$(sbatch --parsable --export=ALL scripts/slurm/train_nfcgs_nn_score_resume.slurm)"
  TRAIN_JOB_ID="${RESPONSE%%;*}"
  [[ "${TRAIN_JOB_ID}" =~ ^[0-9]+$ ]] || { echo "invalid sbatch response: ${RESPONSE}" >&2; exit 1; }
  save_state
else
  echo "REUSING TRAIN_JOB=${TRAIN_JOB_ID}"
fi
export TRAIN_JOB_ID EVAL_RUN_DIR="${RUN_ROOT}/resume_job_${TRAIN_JOB_ID}"
if [[ -z "${EVAL_JOB_ID}" ]]; then
  if ! RESPONSE="$(sbatch --parsable --job-name=gs-ns-resume-eval --export=ALL \
      --dependency="afterok:${TRAIN_JOB_ID}" scripts/slurm/eval_nfcgs_order_context.slurm)"; then
    echo "Train ${TRAIN_JOB_ID} submitted; eval failed. Rerun this command to retry eval only." >&2; exit 1
  fi
  EVAL_JOB_ID="${RESPONSE%%;*}"
  [[ "${EVAL_JOB_ID}" =~ ^[0-9]+$ ]] || { echo "invalid eval sbatch response: ${RESPONSE}" >&2; exit 1; }
  save_state
else
  echo "REUSING EVAL_JOB=${EVAL_JOB_ID}"
fi
echo "TRAIN_JOB=${TRAIN_JOB_ID} EVAL_JOB=${EVAL_JOB_ID} OUTPUT=${EVAL_RUN_DIR}"
echo "STATE=${STATE_FILE}"
