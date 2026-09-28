#!/usr/bin/env bash
# Shared mapping for the 16-condition, Morton-on transform experiment.
# 0..7: linear controls; 8..15: nonlinear32, with the same rank/lambda/residual order.
select_transform_condition() {
  local task="$1"
  case "${task}" in
    0|1|2|3|4|5|6|7) TRANSFORM=linear; TRANSFORM_TAG=linear ;;
    8|9|10|11|12|13|14|15) TRANSFORM=nonlinear; TRANSFORM_TAG=nonlinear32 ;;
    *) echo "ERROR: transform grid expects task id 0 through 15" >&2; return 1 ;;
  esac
  GRID_TASK_ID=$((task % 8))
  case "${GRID_TASK_ID}" in
    0|1) RANK=56; RATE_LAMBDA=0.0064 ;;
    2|3) RANK=56; RATE_LAMBDA=0.0256 ;;
    4|5) RANK=80; RATE_LAMBDA=0.0064 ;;
    6|7) RANK=80; RATE_LAMBDA=0.0256 ;;
  esac
  USE_RESIDUAL=true; RESIDUAL_TAG=on
  if (( GRID_TASK_ID % 2 )); then USE_RESIDUAL=false; RESIDUAL_TAG=off; fi
  TRANSFORM_HIDDEN=32
  USE_MORTON=true
  LAMBDA_TAG="${RATE_LAMBDA/./p}"
  EXPERIMENT_PREFIX="nfcgs_transform_${TRANSFORM_TAG}"
  EXPERIMENT="${EXPERIMENT_PREFIX}_rank${RANK}_lambda${LAMBDA_TAG}_residual_${RESIDUAL_TAG}_morton_on"
  export TRANSFORM TRANSFORM_TAG TRANSFORM_HIDDEN GRID_TASK_ID RANK RATE_LAMBDA
  export USE_RESIDUAL USE_MORTON RESIDUAL_TAG LAMBDA_TAG EXPERIMENT_PREFIX EXPERIMENT
}
