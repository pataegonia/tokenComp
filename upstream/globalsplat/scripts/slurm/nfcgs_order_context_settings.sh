#!/usr/bin/env bash
# Sourced by submit/train/eval so all three use the identical codec settings.
TOKEN_ORDER="${TOKEN_ORDER:-morton}"
CONTEXT_SCHEDULE="${CONTEXT_SCHEDULE:-legacy}"
case "${TOKEN_ORDER}" in morton|hilbert|nn_xyz|nn_score) ;; *) echo "invalid TOKEN_ORDER" >&2; exit 2 ;; esac
case "${CONTEXT_SCHEDULE}" in
  legacy) STAGES="${STAGES:-3}" ;;
  quarter2) STAGES="${STAGES:-2}"; [[ "${STAGES}" == 2 ]] || { echo "quarter2 requires STAGES=2" >&2; exit 2; } ;;
  dyadic4) STAGES="${STAGES:-4}"; [[ "${STAGES}" == 4 ]] || { echo "dyadic4 requires STAGES=4" >&2; exit 2; } ;;
  *) echo "invalid CONTEXT_SCHEDULE" >&2; exit 2 ;;
esac
case "${STAGES}" in 2|3|4) ;; *) echo "STAGES must be 2, 3 or 4" >&2; exit 2 ;; esac
KERNEL="${KERNEL:-5}"
case "${KERNEL}" in 3|5|7) ;; *) echo "KERNEL must be 3, 5 or 7" >&2; exit 2 ;; esac
if [[ "${CONTEXT_SCHEDULE}" == legacy && "${STAGES}" != 2 && "${KERNEL}" == 3 ]]; then
  echo "legacy 3/4 stages require KERNEL=5 or 7" >&2; exit 2
fi
RATE_LAMBDA="${RATE_LAMBDA:-0.0256}"
case "${RATE_LAMBDA}" in 0.0064|0.0256) ;; *) echo "invalid RATE_LAMBDA" >&2; exit 2 ;; esac
MAX_STEPS="${MAX_STEPS:-50000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ && "${CHECKPOINT_EVERY}" =~ ^[1-9][0-9]*$ ]] || { echo "steps must be positive integers" >&2; exit 2; }
(( MAX_STEPS % CHECKPOINT_EVERY == 0 )) || { echo "MAX_STEPS must be divisible by CHECKPOINT_EVERY" >&2; exit 2; }
TAG="${RATE_LAMBDA/./p}"
SCHEDULE_TAG="${CONTEXT_SCHEDULE}_stages${STAGES}_k${KERNEL}_ste"
RUN_ROOT="${REPO_DIR}/outputs/nfcgs_order_context/${TOKEN_ORDER}/${SCHEDULE_TAG}/lambda${TAG}"
CODEC_ARGS=(--score-path minimal --token-order "${TOKEN_ORDER}"
  --score-context-schedule "${CONTEXT_SCHEDULE}" --score-spatial-stages "${STAGES}"
  --score-spatial-kernel "${KERNEL}" --score-context-quantization ste)
if [[ -n "${SCORE_ORDER_SCALE:-}" ]]; then
  [[ "${TOKEN_ORDER}" == nn_score && -s "${SCORE_ORDER_SCALE}" ]] || { echo "SCORE_ORDER_SCALE needs nn_score and an existing JSON file" >&2; exit 2; }
  SCORE_ORDER_SCALE="$(realpath "${SCORE_ORDER_SCALE}")"
  CODEC_ARGS+=(--score-order-scale "${SCORE_ORDER_SCALE}")
fi
export TOKEN_ORDER CONTEXT_SCHEDULE STAGES KERNEL RATE_LAMBDA MAX_STEPS CHECKPOINT_EVERY
export SCORE_ORDER_SCALE="${SCORE_ORDER_SCALE:-}"
