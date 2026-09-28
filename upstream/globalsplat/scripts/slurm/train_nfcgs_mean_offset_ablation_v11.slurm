#!/usr/bin/env bash
#SBATCH -J gs-mean-offset
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v11
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 2-00:00:00
#SBATCH --array=0-1
#SBATCH -o logs/slurm/slurm-gs-mean-offset-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-mean-offset-%A_%a.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

MAX_STEPS="${MAX_STEPS:-20000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
SOURCE="${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0064/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0064_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt"
case "${SLURM_ARRAY_TASK_ID:?submit as an array}" in
  0) VARIANT=baseline; MODE_ARGS=() ;;
  1) VARIANT=no_mean_offset; MODE_ARGS=(--no-score-mean-offset) ;;
  *) echo "invalid array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
OUTPUT="${REPO_DIR}/outputs/nfcgs_main/score_mean_offset_ablation/job_${SLURM_ARRAY_JOB_ID}/${VARIANT}"
[[ -f "${SOURCE}" ]] || { echo "missing source checkpoint: ${SOURCE}" >&2; exit 1; }
python scripts/check_nfcgs_checkpoint.py "${SOURCE}" --step 10000
echo "SOURCE_CHECKPOINT=${SOURCE}"
echo "VARIANT=${VARIANT}"
echo "TRAIN_OUTPUT=${OUTPUT}"

exec srun --ntasks=1 python scripts/run_nfcgs.py train \
  --checkpoint "${SOURCE}" \
  --reset-score-mean-offset \
  --rate-lambda 0.0064 \
  --scope all \
  --lr 0.0001 \
  --batch-size 2 \
  --accumulate 4 \
  --workers 8 \
  --max-steps "${MAX_STEPS}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --output "${OUTPUT}" \
  "${MODE_ARGS[@]}"
