#!/usr/bin/env bash
#SBATCH -J gs-mean-offset-eval
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v11
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 02:00:00
#SBATCH --array=0-1
#SBATCH -o logs/slurm/slurm-gs-mean-offset-eval-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-mean-offset-eval-%A_%a.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

: "${TRAIN_ARRAY_JOB_ID:?submit with submit_nfcgs_mean_offset_ablation_v11.sh}"
MAX_STEPS="${MAX_STEPS:-20000}"
STEP_TAG="$(printf '%09d' "${MAX_STEPS}")"
case "${SLURM_ARRAY_TASK_ID:?submit as an array}" in
  0) VARIANT=baseline; MODE_ARGS=() ;;
  1) VARIANT=no_mean_offset; MODE_ARGS=(--no-score-mean-offset) ;;
  *) echo "invalid array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
ROOT="${REPO_DIR}/outputs/nfcgs_main/score_mean_offset_ablation/job_${TRAIN_ARRAY_JOB_ID}"
CHECKPOINT="${ROOT}/${VARIANT}/checkpoints/nfcgs_main/version_0/step${STEP_TAG}.ckpt"
OUTPUT="${ROOT}/${VARIANT}_eval32"
[[ -f "${CHECKPOINT}" ]] || { echo "missing trained checkpoint: ${CHECKPOINT}" >&2; exit 1; }
python scripts/check_nfcgs_checkpoint.py "${CHECKPOINT}" --step "${MAX_STEPS}"
echo "VARIANT=${VARIANT}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "EVAL_OUTPUT=${OUTPUT}"

exec srun --ntasks=1 python scripts/run_nfcgs.py eval \
  --rate-lambda 0.0064 \
  --checkpoint "${CHECKPOINT}" \
  --max-scenes 32 \
  --dump-score-context \
  --output "${OUTPUT}" \
  "${MODE_ARGS[@]}"
