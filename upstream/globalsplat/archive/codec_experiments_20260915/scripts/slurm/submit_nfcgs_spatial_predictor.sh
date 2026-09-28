#!/usr/bin/env bash
# Default: add only lambda .0256 Full P0/P1/P2 and their aftercorr evaluations.
set -euo pipefail
REPO_DIR="$(pwd -P)"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run from the GlobalSplat repository root" >&2; exit 1;
}
MODE="${1:-submit}"
if (( $# > 1 )) || [[ "${MODE}" != submit && "${MODE}" != dry-run ]]; then
  echo "Usage: bash $0 [submit|dry-run]" >&2; exit 1
fi
source "${REPO_DIR}/scripts/slurm/nfcgs_spatial_predictor_grid.sh"

TASKS="${TASKS:-6-8}"
# Validate before any submission; allow normal Slurm lists/ranges and a cap.
if [[ ! "${TASKS}" =~ ^[0-8](-[0-8])?(,[0-8](-[0-8])?)*(%[1-9][0-9]*)?$ ]]; then
  echo "ERROR: TASKS must select task ids 0 through 8 (for example 6-8)" >&2; exit 1
fi
IFS=, read -r -a TASK_RANGES <<< "${TASKS%%%*}"
declare -A SELECTED=()
for RANGE in "${TASK_RANGES[@]}"; do
  FIRST="${RANGE%-*}"
  LAST="${RANGE##*-}"
  if (( FIRST > LAST )); then
    echo "ERROR: descending task range: ${RANGE}" >&2; exit 1
  fi
  for (( TASK=FIRST; TASK<=LAST; TASK++ )); do SELECTED[${TASK}]=1; done
done
SPATIAL_PREDICTOR_RUN_TAG="${SPATIAL_PREDICTOR_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PARENT_ROOT="${SPATIAL_PREDICTOR_PARENT_ROOT:-${REPO_DIR}/outputs/nfcgs_nonlinear_score_context}"
PARENT_RUN_TAG="${SPATIAL_PREDICTOR_PARENT_RUN_TAG:-20260910_092000}"
declare -A CHECKED_PARENTS=()
for TASK in {0..8}; do
  [[ -n "${SELECTED[${TASK}]:-}" ]] || continue
  select_spatial_predictor_condition "${TASK}"
  echo "TASK=${TASK} ARM=${PREDICTOR_TAG} LAMBDA=${RATE_LAMBDA} PARENT_TASK=${PARENT_TASK_ID} PARENT_PRIOR=${PARENT_PRIOR_TAG}"
  PARENT_EXPERIMENT="nfcgs_nonlinear_score_context_joint_rank56_lambda${LAMBDA_TAG}_residual_on_morton_on_${PARENT_PRIOR_TAG}"
  PARENT_CHECKPOINT="${PARENT_ROOT}/${PARENT_RUN_TAG}/${PARENT_PRIOR_TAG}/rank56/lambda${LAMBDA_TAG}/residual_on/checkpoints/${PARENT_EXPERIMENT}/version_0/step000050000.ckpt"
  [[ -z "${CHECKED_PARENTS[${PARENT_CHECKPOINT}]:-}" ]] || continue
  CHECKED_PARENTS[${PARENT_CHECKPOINT}]=1
  echo "PARENT_CHECKPOINT=${PARENT_CHECKPOINT}"
  [[ "${MODE}" != dry-run ]] || continue
  [[ -f "${PARENT_CHECKPOINT}" ]] || {
    echo "ERROR: missing selected spatial-predictor parent: ${PARENT_CHECKPOINT}" >&2
    exit 1
  }
done
# The grid uses RANK for codec width, not for Lightning's process rank.
export -n RANK

if [[ "${MODE}" == dry-run ]]; then
  echo "DRY_RUN tasks=${TASKS} run=${SPATIAL_PREDICTOR_RUN_TAG} node=ariel-v12 gres=gpu:1"
  echo "Would submit training array and matching aftercorr eval; no jobs submitted, parent files not checked."
  exit 0
fi
command -v sbatch >/dev/null 2>&1 || {
  echo "ERROR: sbatch is not available; run this script on a Slurm login node" >&2
  exit 1
}
mkdir -p "${REPO_DIR}/logs/slurm"

EXPORTS="ALL,SPATIAL_PREDICTOR_RUN_TAG=${SPATIAL_PREDICTOR_RUN_TAG},SPATIAL_PREDICTOR_PARENT_ROOT=${PARENT_ROOT},SPATIAL_PREDICTOR_PARENT_RUN_TAG=${PARENT_RUN_TAG}"
TRAIN_JOB_RAW="$(sbatch --parsable --array="${TASKS}" --nodelist=ariel-v12 --gres=gpu:1 --export="${EXPORTS}" scripts/slurm/train_nfcgs_spatial_predictor.slurm)"
TRAIN_JOB="${TRAIN_JOB_RAW%%;*}"
[[ "${TRAIN_JOB}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: unexpected sbatch train response: ${TRAIN_JOB_RAW}" >&2; exit 1;
}
EVAL_JOB_RAW="$(sbatch --parsable --array="${TASKS}" --nodelist=ariel-v12 --gres=gpu:1 --dependency="aftercorr:${TRAIN_JOB}" --export="${EXPORTS}" scripts/slurm/eval_nfcgs_spatial_predictor.slurm)"
EVAL_JOB="${EVAL_JOB_RAW%%;*}"
[[ "${EVAL_JOB}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: unexpected sbatch eval response: ${EVAL_JOB_RAW}" >&2; exit 1;
}

echo "submitted train=${TRAIN_JOB_RAW} eval=${EVAL_JOB_RAW} run=${SPATIAL_PREDICTOR_RUN_TAG} tasks=${TASKS} node=ariel-v12 gres=gpu:1"
