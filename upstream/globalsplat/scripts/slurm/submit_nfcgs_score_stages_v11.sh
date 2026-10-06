#!/usr/bin/env bash
# STAGE_TASKS=1: only 3 stages. Default 0-2%1: matched 2/3/4 stages, serially.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd "${REPO_DIR}"
mkdir -p logs/slurm
RATE_LAMBDA="${RATE_LAMBDA:-0.0256}"
MAX_STEPS="${MAX_STEPS:-50000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
STAGE_TASKS="${STAGE_TASKS:-0-2%1}"
case "${STAGE_TASKS}" in
  0|1|2|0-2|1-2|0-2%[123]|1-2%[123]) ;;
  *) echo "STAGE_TASKS must be 0, 1, 2, 0-2 or 1-2 (ranges may add %1, %2 or %3)" >&2; exit 2 ;;
esac
case "${RATE_LAMBDA}" in
  0.0064|0.0256) ;;
  *) echo "RATE_LAMBDA must be 0.0064 or 0.0256" >&2; exit 2 ;;
esac
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_STEPS must be positive" >&2; exit 2; }
[[ "${CHECKPOINT_EVERY}" =~ ^[1-9][0-9]*$ ]] || { echo "CHECKPOINT_EVERY must be positive" >&2; exit 2; }
(( MAX_STEPS % CHECKPOINT_EVERY == 0 )) || { echo "MAX_STEPS must be divisible by CHECKPOINT_EVERY" >&2; exit 2; }
for file in scripts/slurm/{train,eval}_nfcgs_score_stages_v11.slurm; do
  [[ -s "${file}" ]] || { echo "missing server file: ${file}" >&2; exit 1; }
  [[ "$(head -n 1 "${file}")" == '#!/usr/bin/env bash' ]] || { echo "invalid shebang or CRLF/BOM: ${file}" >&2; exit 1; }
done
TAG="${RATE_LAMBDA/./p}"
SOURCE="${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda${TAG}/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda${TAG}_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt"
[[ -s "${SOURCE}" ]] || { echo "missing source checkpoint: ${SOURCE}" >&2; exit 1; }
TRAIN_SUBMISSION="$(sbatch --parsable --array="${STAGE_TASKS}" \
  --export="ALL,RATE_LAMBDA=${RATE_LAMBDA},MAX_STEPS=${MAX_STEPS},CHECKPOINT_EVERY=${CHECKPOINT_EVERY}" \
  scripts/slurm/train_nfcgs_score_stages_v11.slurm)"
TRAIN_ID="${TRAIN_SUBMISSION%%;*}"
echo "TRAIN_ARRAY=${TRAIN_ID} TASKS=${STAGE_TASKS}"
EVAL_SUBMISSION="$(sbatch --parsable --array="${STAGE_TASKS}" \
  --dependency="afterok:${TRAIN_ID}" \
  --export="ALL,TRAIN_ARRAY_JOB_ID=${TRAIN_ID},RATE_LAMBDA=${RATE_LAMBDA},MAX_STEPS=${MAX_STEPS}" \
  scripts/slurm/eval_nfcgs_score_stages_v11.slurm)"
EVAL_ID="${EVAL_SUBMISSION%%;*}"
echo "TRAIN_ARRAY=${TRAIN_ID} EVAL_ARRAY=${EVAL_ID} TASKS=${STAGE_TASKS} (0=2 stages, 1=3 stages, 2=4 stages)"
echo "OUTPUT=outputs/nfcgs_score_stages/lambda${TAG}/job_${TRAIN_ID}"
echo "squeue -j ${TRAIN_ID},${EVAL_ID} -o '%.18i %.24j %.10T %.30R'"
