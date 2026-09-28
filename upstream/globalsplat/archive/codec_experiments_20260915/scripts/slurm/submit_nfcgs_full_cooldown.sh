#!/usr/bin/env bash
# Submit a fair P0/P2 low-LR continuation from the original Full parents.
set -euo pipefail
REPO_DIR="$(pwd -P)"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run from the GlobalSplat repository root" >&2; exit 1;
}
MODE="${1:-submit}"
if (( $# > 1 )) || [[ "${MODE}" != submit && "${MODE}" != dry-run ]]; then
  echo "Usage: bash $0 [submit|dry-run]" >&2; exit 1
fi
source "${REPO_DIR}/scripts/slurm/nfcgs_full_cooldown_grid.sh"

TASKS="${TASKS:-0-3}"
if [[ ! "${TASKS}" =~ ^[0-3](-[0-3])?(,[0-3](-[0-3])?)*(%[1-9][0-9]*)?$ ]]; then
  echo "ERROR: TASKS must select task ids 0 through 3 (for example 0-3)" >&2; exit 1
fi
IFS=, read -r -a TASK_RANGES <<< "${TASKS%%%*}"
declare -A SELECTED=()
for RANGE in "${TASK_RANGES[@]}"; do
  FIRST="${RANGE%-*}"
  LAST="${RANGE##*-}"
  (( FIRST <= LAST )) || { echo "ERROR: descending task range: ${RANGE}" >&2; exit 1; }
  for (( TASK=FIRST; TASK<=LAST; TASK++ )); do SELECTED[${TASK}]=1; done
done

FULL_COOLDOWN_RUN_TAG="${FULL_COOLDOWN_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PARENT_ROOT="${FULL_COOLDOWN_PARENT_ROOT:-${REPO_DIR}/outputs/nfcgs_nonlinear_score_context}"
PARENT_RUN_TAG="${FULL_COOLDOWN_PARENT_RUN_TAG:-20260910_092000}"
declare -A CHECKED_PARENTS=()
for TASK in {0..3}; do
  [[ -n "${SELECTED[${TASK}]:-}" ]] || continue
  select_full_cooldown_condition "${TASK}"
  PARENT_EXPERIMENT="nfcgs_nonlinear_score_context_joint_rank56_lambda${LAMBDA_TAG}_residual_on_morton_on_${PARENT_PRIOR_TAG}"
  PARENT_CHECKPOINT="${PARENT_ROOT}/${PARENT_RUN_TAG}/${PARENT_PRIOR_TAG}/rank56/lambda${LAMBDA_TAG}/residual_on/checkpoints/${PARENT_EXPERIMENT}/version_0/step000050000.ckpt"
  echo "TASK=${TASK} ARM=${ARM_TAG} LAMBDA=${RATE_LAMBDA} PARENT_TASK=${PARENT_TASK_ID}"
  [[ -z "${CHECKED_PARENTS[${PARENT_CHECKPOINT}]:-}" ]] || continue
  CHECKED_PARENTS[${PARENT_CHECKPOINT}]=1
  echo "PARENT_CHECKPOINT=${PARENT_CHECKPOINT}"
  [[ "${MODE}" == dry-run || -f "${PARENT_CHECKPOINT}" ]] || {
    echo "ERROR: missing selected Full parent: ${PARENT_CHECKPOINT}" >&2; exit 1;
  }
done
export -n RANK

if [[ "${MODE}" == dry-run ]]; then
  echo "DRY_RUN tasks=${TASKS} run=${FULL_COOLDOWN_RUN_TAG} node=ariel-v12 gres=gpu:1 lr=${FULL_COOLDOWN_LR:-0.000001}"
  echo "Would submit training array and matching aftercorr eval; no jobs submitted, parent files not checked."
  exit 0
fi
command -v sbatch >/dev/null 2>&1 || {
  echo "ERROR: sbatch is not available; run this script on a Slurm login node" >&2; exit 1;
}
mkdir -p "${REPO_DIR}/logs/slurm"

EXPORTS="ALL,FULL_COOLDOWN_RUN_TAG=${FULL_COOLDOWN_RUN_TAG},FULL_COOLDOWN_PARENT_ROOT=${PARENT_ROOT},FULL_COOLDOWN_PARENT_RUN_TAG=${PARENT_RUN_TAG}"
TRAIN_JOB_RAW="$(sbatch --parsable --array="${TASKS}" --nodelist=ariel-v12 --gres=gpu:1 --export="${EXPORTS}" scripts/slurm/train_nfcgs_full_cooldown.slurm)"
TRAIN_JOB="${TRAIN_JOB_RAW%%;*}"
[[ "${TRAIN_JOB}" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected sbatch train response: ${TRAIN_JOB_RAW}" >&2; exit 1; }
EVAL_JOB_RAW="$(sbatch --parsable --array="${TASKS}" --nodelist=ariel-v12 --gres=gpu:1 --dependency="aftercorr:${TRAIN_JOB}" --export="${EXPORTS}" scripts/slurm/eval_nfcgs_full_cooldown.slurm)"
EVAL_JOB="${EVAL_JOB_RAW%%;*}"
[[ "${EVAL_JOB}" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected sbatch eval response: ${EVAL_JOB_RAW}" >&2; exit 1; }

echo "submitted train=${TRAIN_JOB_RAW} eval=${EVAL_JOB_RAW} run=${FULL_COOLDOWN_RUN_TAG} tasks=${TASKS} node=ariel-v12 gres=gpu:1"
