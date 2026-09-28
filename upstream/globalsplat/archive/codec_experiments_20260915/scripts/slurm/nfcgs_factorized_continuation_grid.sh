#!/usr/bin/env bash
# Compute-matched factorized controls for the score-context experiment.
select_factorized_continuation_condition() {
  local task="$1"
  case "${task}" in
    0) RANK=56; RATE_LAMBDA=0.0064; GRID_TASK_ID=0 ;;
    1) RANK=56; RATE_LAMBDA=0.0256; GRID_TASK_ID=2 ;;
    2) RANK=80; RATE_LAMBDA=0.0064; GRID_TASK_ID=4 ;;
    3) RANK=80; RATE_LAMBDA=0.0256; GRID_TASK_ID=6 ;;
    *) echo "ERROR: factorized continuation grid expects task id 0 through 3" >&2; return 1 ;;
  esac

  USE_RESIDUAL=true
  USE_MORTON=true
  TRANSFORM=linear
  TRANSFORM_HIDDEN=32
  SCORE_MEAN_CONDITION=false
  SCORE_CHANNEL_CONTEXT=false
  SCORE_SPATIAL_CONTEXT=false
  SCORE_SLICE_CHANNELS=16
  SCORE_CONTEXT_HIDDEN=64
  LAMBDA_TAG="${RATE_LAMBDA/./p}"

  export RATE_LAMBDA GRID_TASK_ID RANK USE_RESIDUAL USE_MORTON
  export TRANSFORM TRANSFORM_HIDDEN LAMBDA_TAG
  export SCORE_MEAN_CONDITION SCORE_CHANNEL_CONTEXT SCORE_SPATIAL_CONTEXT
  export SCORE_SLICE_CHANNELS SCORE_CONTEXT_HIDDEN
}
