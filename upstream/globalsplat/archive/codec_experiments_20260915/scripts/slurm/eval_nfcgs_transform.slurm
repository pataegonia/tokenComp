#!/usr/bin/env bash
#SBATCH -J gs-nfcgs-transform-eval
#SBATCH --array=8-15
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-nfcgs-transform-eval-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-nfcgs-transform-eval-%A_%a.err

set -euo pipefail
# Tasks 0..7 use confirmed version_0 linear checkpoints unless
# TRANSFORM_LINEAR_SOURCE=fresh is explicitly selected after retraining them.
# Tasks 8..15 evaluate the new nonlinear32 checkpoints. No "latest" discovery.
# bash scripts/slurm/eval_nfcgs_transform.slurm plan
# bash scripts/slurm/eval_nfcgs_transform.slurm check
# conda activate globalsplat
# bash scripts/slurm/eval_nfcgs_transform.slurm validate 8-14
# sbatch --array=8-14 scripts/slurm/eval_nfcgs_transform.slurm
# sbatch scripts/slurm/eval_nfcgs_transform.slurm
export REPO_DIR="$(pwd -P)"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run from the GlobalSplat repository root" >&2; exit 1;
}
source "${REPO_DIR}/scripts/slurm/nfcgs_transform_grid.sh"
NEW_TRAIN_ROOT="${TRANSFORM_TRAIN_ROOT:-${REPO_DIR}/outputs/nfcgs_paper24_transform_train}"
NEW_EVAL_ROOT="${TRANSFORM_EVAL_ROOT:-${REPO_DIR}/outputs/nfcgs_paper24_transform_eval}"
VERSION="${TRANSFORM_VERSION:-0}"
STEP="${TRANSFORM_EVAL_STEP:-50000}"
LINEAR_SOURCE="${TRANSFORM_LINEAR_SOURCE:-existing}"
[[ "${VERSION}" =~ ^[0-9]+$ && "${STEP}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: TRANSFORM_VERSION and TRANSFORM_EVAL_STEP must be nonnegative integers" >&2; exit 1;
}
case "${LINEAR_SOURCE}" in
  existing|fresh) ;;
  *) echo "ERROR: TRANSFORM_LINEAR_SOURCE must be existing or fresh" >&2; exit 1 ;;
esac
export PROTOCOL=all CONTEXT=12 PRECISION=bf16-mixed SAVE_IMAGES=false
export DRY_RUN=false CHECKPOINT_ONLY=false SLURM_EXPORT_ENV=ALL
# This is the transform ablation, not the separate score-context experiment.
# Do not inherit a limited scene count or score-context flags from another run.
export MAX_SCENES=null
export SCORE_MEAN_CONDITION=false SCORE_CHANNEL_CONTEXT=false SCORE_SPATIAL_CONTEXT=false
export SCORE_SLICE_CHANNELS=16 SCORE_CONTEXT_HIDDEN=64

select_transform_checkpoint() {
  select_transform_condition "$1"
  # RANK here means codec width, not Lightning's distributed process rank.
  # The common evaluator recovers width from the exported GRID_TASK_ID.
  export -n RANK
  local version="${VERSION}" experiment="${EXPERIMENT}"
  EXPECTED_STEP="${STEP}"
  TRAIN_OUTPUT_ROOT="${NEW_TRAIN_ROOT}/${TRANSFORM_TAG}"
  if [[ "${TRANSFORM}" == "linear" && "${LINEAR_SOURCE}" == "existing" ]]; then
    TRAIN_OUTPUT_ROOT="${REPO_DIR}/outputs/nfcgs_paper24_subset_train"
    experiment="nfcgs_paper24_subset_rank${RANK}_lambda${LAMBDA_TAG}_residual_${RESIDUAL_TAG}_morton_on"
    version=0
    EXPECTED_STEP=50000
  fi
  local filename
  printf -v filename 'step%09d.ckpt' "$((10#${EXPECTED_STEP}))"
  CHECKPOINT="${TRAIN_OUTPUT_ROOT}/rank${RANK}/lambda${LAMBDA_TAG}/residual_${RESIDUAL_TAG}/checkpoints/${experiment}/version_${version}/${filename}"
  OUTPUT_ROOT="${NEW_EVAL_ROOT}/${TRANSFORM_TAG}"
  export CHECKPOINT TRAIN_OUTPUT_ROOT OUTPUT_ROOT EXPECTED_STEP
  echo "TASK=$1 TRANSFORM=${TRANSFORM} RANK=${RANK} LAMBDA=${RATE_LAMBDA} RESIDUAL=${USE_RESIDUAL} CHECKPOINT=${CHECKPOINT}"
}

select_preflight_tasks() {
  local spec="$1" token first last task
  local -a parts
  local -A seen=()
  TASKS=()
  [[ "${spec}" =~ ^[0-9,-]+$ && "${spec}" != ,* && "${spec}" != *, && "${spec}" != *,,* ]] || {
    echo "ERROR: invalid task selection: ${spec}" >&2; return 1;
  }
  IFS=',' read -r -a parts <<< "${spec}"
  for token in "${parts[@]}"; do
    [[ "${token}" =~ ^([0-9]{1,2})(-([0-9]{1,2}))?$ ]] || {
      echo "ERROR: invalid task selection: ${spec}" >&2; return 1;
    }
    first=$((10#${BASH_REMATCH[1]}))
    last=$((10#${BASH_REMATCH[3]:-${BASH_REMATCH[1]}}))
    (( first <= last && last <= 15 )) || {
      echo "ERROR: task selection must use ascending ranges within 0 through 15" >&2; return 1;
    }
    for ((task=first; task<=last; task++)); do
      if [[ -z "${seen[${task}]:-}" ]]; then
        TASKS+=("${task}")
        seen[${task}]=1
      fi
    done
  done
}

usage() {
  echo "Usage: bash $0 {plan|check|validate} [TASKS, e.g. 8-14 or 8,10-14] | sbatch [--array=8-15] $0" >&2
}

case "${1:-run}" in
  plan|check|validate)
    (( $# <= 2 )) || { usage; exit 1; }
    MODE="$1"
    select_preflight_tasks "${2-0-15}"
    if [[ "${MODE}" == "validate" ]]; then
      # CPU-only: activate the project environment first, or supply its Python.
      VALIDATE_PYTHON="${TRANSFORM_PYTHON:-python}"
      command -v "${VALIDATE_PYTHON}" >/dev/null || {
        echo "ERROR: activate the globalsplat environment or set TRANSFORM_PYTHON" >&2; exit 1;
      }
      export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    fi
    FAILURES=0
    for task in "${TASKS[@]}"; do
      select_transform_checkpoint "${task}"
      if [[ "${MODE}" == "plan" ]]; then continue; fi
      if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "ERROR: requested transform checkpoint is missing: ${CHECKPOINT}" >&2
        FAILURES=$((FAILURES + 1))
        continue
      fi
      if [[ "${MODE}" == "validate" ]]; then
        if ! "${VALIDATE_PYTHON}" "${REPO_DIR}/scripts/check_nfcgs_checkpoint.py" "${CHECKPOINT}" \
          --rank "${RANK}" --step "${EXPECTED_STEP}" \
          --transform "${TRANSFORM}" --transform-hidden "${TRANSFORM_HIDDEN}" \
          --residual "${USE_RESIDUAL}" --morton "${USE_MORTON}" \
          --score-mean-condition "${SCORE_MEAN_CONDITION}" \
          --score-channel-context "${SCORE_CHANNEL_CONTEXT}" \
          --score-spatial-context "${SCORE_SPATIAL_CONTEXT}" \
          --score-slice-channels "${SCORE_SLICE_CHANNELS}" \
          --score-context-hidden "${SCORE_CONTEXT_HIDDEN}"; then
          echo "ERROR: checkpoint validation failed for task ${task}" >&2
          FAILURES=$((FAILURES + 1))
        fi
      fi
    done
    (( FAILURES == 0 )) || {
      echo "ERROR: ${FAILURES}/${#TASKS[@]} selected checkpoints failed ${MODE}; do not submit this selection." >&2; exit 1;
    }
    case "${MODE}" in
      check) echo "All ${#TASKS[@]} selected checkpoint files exist; use validate to check contents before submission." ;;
      validate) echo "All ${#TASKS[@]} selected checkpoints passed CPU loading, codec configuration, and global-step validation." ;;
    esac
    exit 0
    ;;
  run) (( $# <= 1 )) || { usage; exit 1; } ;;
  *) usage; exit 1 ;;
esac
select_transform_checkpoint "${SLURM_ARRAY_TASK_ID:-}"
[[ -f "${CHECKPOINT}" ]] || {
  echo "ERROR: requested transform checkpoint is missing: ${CHECKPOINT}" >&2; exit 1;
}
exec bash "${REPO_DIR}/scripts/slurm/eval_nfcgs_rank56.slurm"
