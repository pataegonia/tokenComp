#!/usr/bin/env bash
#SBATCH -J gs-score-pred
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v11
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm/slurm-gs-score-pred-%j.out
#SBATCH -e logs/slurm/slurm-gs-score-pred-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

CHECKPOINT="${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0064/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0064_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt"
OUTPUT="${REPO_DIR}/outputs/nfcgs_main/split_score_prediction_v11/job_${SLURM_JOB_ID:-manual}"
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing Split checkpoint: ${CHECKPOINT}" >&2; exit 1; }
python scripts/check_nfcgs_checkpoint.py "${CHECKPOINT}" --step 10000
echo "CHECKPOINT=${CHECKPOINT}"
echo "EVAL_OUTPUT=${OUTPUT}"

exec srun --ntasks=1 python scripts/run_nfcgs.py eval \
  --rate-lambda 0.0064 \
  --checkpoint "${CHECKPOINT}" \
  --max-scenes 32 \
  --dump-score-context \
  --output "${OUTPUT}"
