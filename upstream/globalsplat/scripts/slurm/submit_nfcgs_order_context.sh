#!/usr/bin/env bash
# Select order and context independently. Submits one train and afterok full eval.
set -euo pipefail
if [[ -n "${REPO_DIR:-}" ]]; then
  if ! cd -P -- "${REPO_DIR}"; then
    echo "Cannot enter REPO_DIR=${REPO_DIR}. Run 'cd /', then cd to the existing repository path." >&2
    exit 1
  fi
else
  if ! SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" \
      || ! cd -P -- "${SCRIPT_DIR}/../.."; then
    echo "Cannot resolve repository directory. The shell may be inside a deleted/replaced directory." >&2
    echo "Run 'cd /', cd to the existing repository path, and retry (or set an absolute REPO_DIR)." >&2
    exit 1
  fi
fi
if ! REPO_DIR="$(pwd -P)" || [[ "${REPO_DIR}" != /* ]]; then
  echo "Cannot resolve an absolute repository path; no jobs submitted. Re-enter the repository from 'cd /'." >&2
  exit 1
fi
[[ -f scripts/run_nfcgs.py ]] || { echo "Not a GlobalSplat repository: ${REPO_DIR}" >&2; exit 1; }
source scripts/slurm/nfcgs_order_context_settings.sh
for file in scripts/slurm/{train,eval}_nfcgs_order_context.slurm; do
  [[ -s "${file}" ]] || { echo "missing server file: ${file}" >&2; exit 1; }
  [[ "$(head -n 1 "${file}")" == '#!/usr/bin/env bash' ]] || { echo "invalid shebang or CRLF/BOM: ${file}" >&2; exit 1; }
done
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda${TAG}/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda${TAG}_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt}"
[[ -s "${SOURCE_CHECKPOINT}" ]] || {
  echo "Missing source checkpoint (no jobs submitted): ${SOURCE_CHECKPOINT}" >&2
  echo "Check the existing FullSplit parent after re-entering the repository; set SOURCE_CHECKPOINT explicitly if its path changed." >&2
  exit 1
}
if ! SOURCE_CHECKPOINT="$(realpath -e -- "${SOURCE_CHECKPOINT}")"; then
  echo "Source checkpoint could not be resolved; no jobs submitted." >&2
  exit 1
fi
echo "REPO_DIR=${REPO_DIR} SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT}"
export REPO_DIR SOURCE_CHECKPOINT
mkdir -p logs/slurm
TRAIN_SUBMISSION="$(sbatch --parsable --export=ALL scripts/slurm/train_nfcgs_order_context.slurm)"
TRAIN_JOB_ID="${TRAIN_SUBMISSION%%;*}"
[[ "${TRAIN_JOB_ID}" =~ ^[0-9]+$ ]] || { echo "invalid sbatch response: ${TRAIN_SUBMISSION}" >&2; exit 1; }
export TRAIN_JOB_ID
echo "TRAIN_JOB=${TRAIN_JOB_ID} ORDER=${TOKEN_ORDER} SCHEDULE=${CONTEXT_SCHEDULE} STAGES=${STAGES}"
EVAL_SUBMISSION="$(sbatch --parsable --dependency="afterok:${TRAIN_JOB_ID}" --export=ALL scripts/slurm/eval_nfcgs_order_context.slurm)"
EVAL_JOB_ID="${EVAL_SUBMISSION%%;*}"
echo "TRAIN_JOB=${TRAIN_JOB_ID} EVAL_JOB=${EVAL_JOB_ID} OUTPUT=${RUN_ROOT}/job_${TRAIN_JOB_ID}"
echo "squeue -j ${TRAIN_JOB_ID},${EVAL_JOB_ID} -o '%.18i %.24j %.10T %.30R'"
