#!/usr/bin/env bash
# Sourced by submit/train/eval so all three use the identical codec settings.
TOKEN_ORDER="${TOKEN_ORDER:-morton}"
CONTEXT_SCHEDULE="${CONTEXT_SCHEDULE:-legacy}"
case "${TOKEN_ORDER}" in none|morton|hilbert|nn_xyz|nn_score) ;; *) echo "invalid TOKEN_ORDER" >&2; exit 2 ;; esac
SPATIAL_CONTEXT="${SPATIAL_CONTEXT:-true}"
case "${SPATIAL_CONTEXT}" in true|false) ;; *) echo "SPATIAL_CONTEXT must be true or false" >&2; exit 2 ;; esac
NN_ORDER_BACKEND="${NN_ORDER_BACKEND:-legacy}"
NN_ORDER_CANDIDATES="${NN_ORDER_CANDIDATES:-64}"
NN_ORDER_DISTANCE_DEVICE="${NN_ORDER_DISTANCE_DEVICE:-cpu}"
case "${NN_ORDER_BACKEND}" in legacy|row|knn) ;; *) echo "invalid NN_ORDER_BACKEND" >&2; exit 2 ;; esac
case "${NN_ORDER_DISTANCE_DEVICE}" in cpu|input) ;; *) echo "invalid NN_ORDER_DISTANCE_DEVICE" >&2; exit 2 ;; esac
[[ "${NN_ORDER_CANDIDATES}" =~ ^[1-9][0-9]*$ ]] || { echo "NN_ORDER_CANDIDATES must be positive" >&2; exit 2; }
if [[ "${NN_ORDER_BACKEND}" == legacy && "${NN_ORDER_DISTANCE_DEVICE}" != cpu ]]; then
  echo "legacy NN backend requires NN_ORDER_DISTANCE_DEVICE=cpu" >&2; exit 2
fi
if [[ "${NN_ORDER_BACKEND}" != legacy && "${TOKEN_ORDER}" != nn_xyz && "${TOKEN_ORDER}" != nn_score ]]; then
  echo "optimized NN backend requires TOKEN_ORDER=nn_xyz or nn_score" >&2; exit 2
fi
case "${CONTEXT_SCHEDULE}" in
  legacy) STAGES="${STAGES:-3}" ;;
  quarter2) STAGES="${STAGES:-2}"; [[ "${STAGES}" == 2 ]] || { echo "quarter2 requires STAGES=2" >&2; exit 2; } ;;
  dyadic4) STAGES="${STAGES:-4}"; [[ "${STAGES}" == 4 ]] || { echo "dyadic4 requires STAGES=4" >&2; exit 2; } ;;
  single) STAGES="${STAGES:-1}"; [[ "${STAGES}" == 1 && "${SPATIAL_CONTEXT}" == false ]] || { echo "single requires STAGES=1 and SPATIAL_CONTEXT=false" >&2; exit 2; } ;;
  *) echo "invalid CONTEXT_SCHEDULE" >&2; exit 2 ;;
esac
case "${STAGES}" in 1|2|3|4) ;; *) echo "STAGES must be 1, 2, 3 or 4" >&2; exit 2 ;; esac
[[ "${STAGES}" != 1 || "${CONTEXT_SCHEDULE}" == single ]] || { echo "STAGES=1 requires single schedule" >&2; exit 2; }
KERNEL="${KERNEL:-5}"
SCORE_SLICE_CHANNELS="${SCORE_SLICE_CHANNELS:-16}"
[[ "${SCORE_SLICE_CHANNELS}" =~ ^[1-9][0-9]*$ ]] && (( SCORE_SLICE_CHANNELS <= 56 )) || { echo "SCORE_SLICE_CHANNELS must be 1..56" >&2; exit 2; }
case "${KERNEL}" in 3|5|7) ;; *) echo "KERNEL must be 3, 5 or 7" >&2; exit 2 ;; esac
if [[ "${CONTEXT_SCHEDULE}" == legacy && "${STAGES}" != 2 && "${KERNEL}" == 3 ]]; then
  echo "legacy 3/4 stages require KERNEL=5 or 7" >&2; exit 2
fi
RATE_LAMBDA="${RATE_LAMBDA:-0.0256}"
case "${RATE_LAMBDA}" in 0|0.0032|0.0064|0.0128|0.0256|0.0512) ;; *) echo "RATE_LAMBDA must be 0, 0.0032, 0.0064, 0.0128, 0.0256 or 0.0512" >&2; exit 2 ;; esac
MAX_STEPS="${MAX_STEPS:-50000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ && "${CHECKPOINT_EVERY}" =~ ^[1-9][0-9]*$ ]] || { echo "steps must be positive integers" >&2; exit 2; }
(( MAX_STEPS % CHECKPOINT_EVERY == 0 )) || { echo "MAX_STEPS must be divisible by CHECKPOINT_EVERY" >&2; exit 2; }
TAG="${RATE_LAMBDA/./p}"
SCHEDULE_TAG="${CONTEXT_SCHEDULE}_stages${STAGES}_k${KERNEL}_ste"
[[ "${SPATIAL_CONTEXT}" == true ]] || SCHEDULE_TAG+="_context_off"
[[ "${SCORE_SLICE_CHANNELS}" == 16 ]] || SCHEDULE_TAG+="_c${SCORE_SLICE_CHANNELS}"
if [[ "${NN_ORDER_BACKEND}" != legacy || "${NN_ORDER_CANDIDATES}" != 64 || "${NN_ORDER_DISTANCE_DEVICE}" != cpu ]]; then
  SCHEDULE_TAG+="_nn_${NN_ORDER_BACKEND}_${NN_ORDER_DISTANCE_DEVICE}_k${NN_ORDER_CANDIDATES}"
fi
RUN_ROOT="${REPO_DIR}/outputs/nfcgs_order_context/${TOKEN_ORDER}/${SCHEDULE_TAG}/lambda${TAG}"
CODEC_ARGS=(--score-path minimal --token-order "${TOKEN_ORDER}"
  --nn-order-backend "${NN_ORDER_BACKEND}" --nn-order-candidates "${NN_ORDER_CANDIDATES}"
  --nn-order-distance-device "${NN_ORDER_DISTANCE_DEVICE}"
  --score-context-schedule "${CONTEXT_SCHEDULE}" --score-spatial-stages "${STAGES}"
  --score-slice-channels "${SCORE_SLICE_CHANNELS}"
  --score-spatial-kernel "${KERNEL}" --score-context-quantization ste)
[[ "${SPATIAL_CONTEXT}" == true ]] || CODEC_ARGS+=(--no-spatial-context)
if [[ -n "${SCORE_ORDER_SCALE:-}" ]]; then
  [[ "${TOKEN_ORDER}" == nn_score && -s "${SCORE_ORDER_SCALE}" ]] || { echo "SCORE_ORDER_SCALE needs nn_score and an existing JSON file" >&2; exit 2; }
  SCORE_ORDER_SCALE="$(realpath "${SCORE_ORDER_SCALE}")"
  CODEC_ARGS+=(--score-order-scale "${SCORE_ORDER_SCALE}")
fi
export TOKEN_ORDER CONTEXT_SCHEDULE STAGES KERNEL RATE_LAMBDA MAX_STEPS CHECKPOINT_EVERY
export SPATIAL_CONTEXT
export SCORE_SLICE_CHANNELS
export NN_ORDER_BACKEND NN_ORDER_CANDIDATES NN_ORDER_DISTANCE_DEVICE
export SCORE_ORDER_SCALE="${SCORE_ORDER_SCALE:-}"
