#!/usr/bin/env bash
# Submit matching train/eval arrays. By default every architecture/rate arm is
# submitted for final joint training with no client-side concurrency throttle.
set -euo pipefail
REPO_DIR="$(pwd -P)"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run from the GlobalSplat repository root" >&2; exit 1;
}
mkdir -p "${REPO_DIR}/logs/slurm"
TRAIN_MODE="${TRAIN_MODE:-joint}"
TASKS="${TASKS:-0-7}"
case "${TRAIN_MODE}" in
  screen|joint) ;;
  *) echo "ERROR: TRAIN_MODE must be screen or joint" >&2; exit 1 ;;
esac
command -v sbatch >/dev/null 2>&1 || {
  echo "ERROR: sbatch is not available; run this script on a Slurm login node" >&2
  exit 1
}
SCORE_CONTEXT_RUN_TAG="${SCORE_CONTEXT_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXPORTS="ALL,TRAIN_MODE=${TRAIN_MODE},SCORE_CONTEXT_RUN_TAG=${SCORE_CONTEXT_RUN_TAG}"
TRAIN_JOB_RAW="$(sbatch --parsable --array="${TASKS}" --nodelist=ariel-v12 --gres=gpu:1 --export="${EXPORTS}" scripts/slurm/train_nfcgs_score_context.slurm)"
TRAIN_JOB="${TRAIN_JOB_RAW%%;*}"
[[ "${TRAIN_JOB}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: unexpected sbatch train response: ${TRAIN_JOB_RAW}" >&2
  exit 1
}
EVAL_JOB_RAW="$(sbatch --parsable --array="${TASKS}" --nodelist=ariel-v12 --gres=gpu:1 --dependency="aftercorr:${TRAIN_JOB}" --export="${EXPORTS}" scripts/slurm/eval_nfcgs_score_context.slurm)"
EVAL_JOB="${EVAL_JOB_RAW%%;*}"
[[ "${EVAL_JOB}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: unexpected sbatch eval response: ${EVAL_JOB_RAW}" >&2
  exit 1
}
echo "submitted train=${TRAIN_JOB_RAW} eval=${EVAL_JOB_RAW} run=${SCORE_CONTEXT_RUN_TAG} mode=${TRAIN_MODE} tasks=${TASKS} node=ariel-v12 gres=gpu:1"
