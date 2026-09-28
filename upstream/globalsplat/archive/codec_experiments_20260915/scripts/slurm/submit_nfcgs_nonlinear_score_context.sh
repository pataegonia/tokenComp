#!/usr/bin/env bash
# Submit nonlinear32 continuation controls before selecting a context winner.
set -euo pipefail
REPO_DIR="$(pwd -P)"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run from the GlobalSplat repository root" >&2; exit 1;
}
command -v sbatch >/dev/null 2>&1 || {
  echo "ERROR: sbatch is not available; run this script on a Slurm login node" >&2
  exit 1
}
mkdir -p "${REPO_DIR}/logs/slurm"

# Context arms are intentionally opt-in until the linear context results select
# a winner. The four defaults are Rank-56/80 factorized continuation controls.
TASKS="${TASKS:-0-1,10-11}"
NONLINEAR_CONTEXT_RUN_TAG="${NONLINEAR_CONTEXT_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
NONLINEAR_TRAIN_ROOT="${NONLINEAR_TRAIN_ROOT:-${REPO_DIR}/outputs/nfcgs_paper24_transform_train}"
for CONDITION in 56:0p0064 56:0p0256 80:0p0064 80:0p0256; do
  IFS=: read -r CONDITION_RANK LAMBDA_TAG <<< "${CONDITION}"
  BASE_EXPERIMENT="nfcgs_transform_nonlinear32_rank${CONDITION_RANK}_lambda${LAMBDA_TAG}_residual_on_morton_on"
  BASE_CHECKPOINT="${NONLINEAR_TRAIN_ROOT}/nonlinear32/rank${CONDITION_RANK}/lambda${LAMBDA_TAG}/residual_on/checkpoints/${BASE_EXPERIMENT}/version_0/step000050000.ckpt"
  [[ -f "${BASE_CHECKPOINT}" ]] || {
    echo "ERROR: missing nonlinear32 warm-start checkpoint: ${BASE_CHECKPOINT}" >&2
    exit 1
  }
done

EXPORTS="ALL,NONLINEAR_CONTEXT_RUN_TAG=${NONLINEAR_CONTEXT_RUN_TAG}"
TRAIN_JOB_RAW="$(sbatch --parsable --array="${TASKS}" --nodelist=ariel-v12 --gres=gpu:1 --export="${EXPORTS}" scripts/slurm/train_nfcgs_nonlinear_score_context.slurm)"
TRAIN_JOB="${TRAIN_JOB_RAW%%;*}"
[[ "${TRAIN_JOB}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: unexpected sbatch train response: ${TRAIN_JOB_RAW}" >&2; exit 1;
}
EVAL_JOB_RAW="$(sbatch --parsable --array="${TASKS}" --nodelist=ariel-v12 --gres=gpu:1 --dependency="aftercorr:${TRAIN_JOB}" --export="${EXPORTS}" scripts/slurm/eval_nfcgs_nonlinear_score_context.slurm)"
EVAL_JOB="${EVAL_JOB_RAW%%;*}"
[[ "${EVAL_JOB}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: unexpected sbatch eval response: ${EVAL_JOB_RAW}" >&2; exit 1;
}

echo "submitted train=${TRAIN_JOB_RAW} eval=${EVAL_JOB_RAW} run=${NONLINEAR_CONTEXT_RUN_TAG} tasks=${TASKS} node=ariel-v12 gres=gpu:1"
