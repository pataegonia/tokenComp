#!/usr/bin/env bash
#SBATCH -J gs-score-stages-eval
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 24:00:00
#SBATCH --array=0-2%1
#SBATCH -o logs/slurm/slurm-gs-score-stages-eval-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-score-stages-eval-%A_%a.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
: "${TRAIN_ARRAY_JOB_ID:?submit with submit_nfcgs_score_stages_v11.sh}"
RATE_LAMBDA="${RATE_LAMBDA:-0.0256}"
MAX_STEPS="${MAX_STEPS:-50000}"
case "${SLURM_ARRAY_TASK_ID:?submit as an array}" in
  0) STAGES=2 ;;
  1) STAGES=3 ;;
  2) STAGES=4 ;;
  *) echo "invalid array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
TAG="${RATE_LAMBDA/./p}"
STEP_TAG="$(printf '%09d' "${MAX_STEPS}")"
ROOT="${REPO_DIR}/outputs/nfcgs_score_stages/lambda${TAG}/job_${TRAIN_ARRAY_JOB_ID}"
CHECKPOINT="${ROOT}/stages${STAGES}/checkpoints/nfcgs_main/version_0/step${STEP_TAG}.ckpt"
OUTPUT="${ROOT}/stages${STAGES}_eval_all"
[[ -s "${CHECKPOINT}" ]] || { echo "missing trained checkpoint: ${CHECKPOINT}" >&2; exit 1; }
python scripts/check_nfcgs_checkpoint.py "${CHECKPOINT}" --step "${MAX_STEPS}"
echo "STAGES=${STAGES} CHECKPOINT=${CHECKPOINT} EVAL_OUTPUT=${OUTPUT}"
exec srun --ntasks=1 python scripts/run_nfcgs.py eval \
  --checkpoint "${CHECKPOINT}" --rate-lambda "${RATE_LAMBDA}" \
  --score-path minimal --score-spatial-stages "${STAGES}" \
  --score-spatial-kernel 5 --score-context-quantization ste \
  --output "${OUTPUT}"
