#!/usr/bin/env bash
# Submit matched training arms and their dependent 32-scene evaluations.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd "${REPO_DIR}"
mkdir -p logs/slurm
MAX_STEPS="${MAX_STEPS:-20000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_STEPS must be positive" >&2; exit 2; }
[[ "${CHECKPOINT_EVERY}" =~ ^[1-9][0-9]*$ ]] || { echo "CHECKPOINT_EVERY must be positive" >&2; exit 2; }
(( MAX_STEPS % CHECKPOINT_EVERY == 0 )) || { echo "MAX_STEPS must be divisible by CHECKPOINT_EVERY" >&2; exit 2; }
SOURCE="${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0064/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0064_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt"
[[ -f "${SOURCE}" ]] || { echo "missing source checkpoint: ${SOURCE}" >&2; exit 1; }

TRAIN_SUBMISSION="$(sbatch --parsable \
  --export="ALL,MAX_STEPS=${MAX_STEPS},CHECKPOINT_EVERY=${CHECKPOINT_EVERY}" \
  scripts/slurm/train_nfcgs_mean_offset_ablation_v11.slurm)"
TRAIN_ID="${TRAIN_SUBMISSION%%;*}"
EVAL_SUBMISSION="$(sbatch --parsable \
  --dependency="afterok:${TRAIN_ID}" \
  --export="ALL,TRAIN_ARRAY_JOB_ID=${TRAIN_ID},MAX_STEPS=${MAX_STEPS}" \
  scripts/slurm/eval_nfcgs_mean_offset_ablation_v11.slurm)"
EVAL_ID="${EVAL_SUBMISSION%%;*}"
REPORT_SUBMISSION="$(sbatch --parsable \
  --dependency="afterok:${EVAL_ID}" \
  --export="ALL,TRAIN_ARRAY_JOB_ID=${TRAIN_ID}" \
  scripts/slurm/report_nfcgs_mean_offset_ablation.slurm)"
REPORT_ID="${REPORT_SUBMISSION%%;*}"
echo "training array ${TRAIN_ID}: baseline (0), no_mean_offset (1)"
echo "dependent eval array ${EVAL_ID}: 32 scenes per arm"
echo "dependent report ${REPORT_ID}: comparison.json"
echo "squeue -j ${TRAIN_ID},${EVAL_ID},${REPORT_ID} -o '%.18i %.24j %.10T %.30R'"
