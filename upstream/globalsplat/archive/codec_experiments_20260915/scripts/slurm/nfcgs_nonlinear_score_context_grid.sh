#!/usr/bin/env bash
# Nonlinear32 x score-prior grid. Even tasks use lambda .0064, odd tasks .0256.
select_nonlinear_score_context_condition() {
  local task="$1"
  case "${task}" in
    0|1) SCORE_MEAN_CONDITION=false; SCORE_CHANNEL_CONTEXT=false; SCORE_SPATIAL_CONTEXT=false; PRIOR_TAG=factorized ;;
    2|3) SCORE_MEAN_CONDITION=true;  SCORE_CHANNEL_CONTEXT=false; SCORE_SPATIAL_CONTEXT=false; PRIOR_TAG=m1c0s0 ;;
    4|5) SCORE_MEAN_CONDITION=true;  SCORE_CHANNEL_CONTEXT=true;  SCORE_SPATIAL_CONTEXT=false; PRIOR_TAG=m1c1s0 ;;
    6|7) SCORE_MEAN_CONDITION=true;  SCORE_CHANNEL_CONTEXT=false; SCORE_SPATIAL_CONTEXT=true;  PRIOR_TAG=m1c0s1 ;;
    8|9) SCORE_MEAN_CONDITION=true;  SCORE_CHANNEL_CONTEXT=true;  SCORE_SPATIAL_CONTEXT=true;  PRIOR_TAG=m1c1s1 ;;
    10|11) SCORE_MEAN_CONDITION=false; SCORE_CHANNEL_CONTEXT=false; SCORE_SPATIAL_CONTEXT=false; PRIOR_TAG=factorized ;;
    *) echo "ERROR: nonlinear score-context grid expects task id 0 through 11" >&2; return 1 ;;
  esac

  if (( task >= 10 )); then
    RANK=80
  else
    RANK=56
  fi
  if (( task % 2 == 0 )); then
    RATE_LAMBDA=0.0064
    if (( RANK == 56 )); then GRID_TASK_ID=0; else GRID_TASK_ID=4; fi
  else
    RATE_LAMBDA=0.0256
    if (( RANK == 56 )); then GRID_TASK_ID=2; else GRID_TASK_ID=6; fi
  fi

  USE_RESIDUAL=true
  USE_MORTON=true
  TRANSFORM=nonlinear
  TRANSFORM_HIDDEN=32
  SCORE_SLICE_CHANNELS="${SCORE_SLICE_CHANNELS:-16}"
  SCORE_CONTEXT_HIDDEN="${SCORE_CONTEXT_HIDDEN:-64}"
  LAMBDA_TAG="${RATE_LAMBDA/./p}"

  export RATE_LAMBDA GRID_TASK_ID RANK USE_RESIDUAL USE_MORTON
  export TRANSFORM TRANSFORM_HIDDEN LAMBDA_TAG PRIOR_TAG
  export SCORE_MEAN_CONDITION SCORE_CHANNEL_CONTEXT SCORE_SPATIAL_CONTEXT
  export SCORE_SLICE_CHANNELS SCORE_CONTEXT_HIDDEN
}
