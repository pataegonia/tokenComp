#!/usr/bin/env bash
#SBATCH -J gs-joint500k-eval
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v10
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-joint500k-eval-%j.out
#SBATCH -e logs/slurm/slurm-gs-joint500k-eval-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

CHECKPOINT="${REPO_DIR}/outputs/nfcgs_main/joint_extension/lambda0p0064/from_job_425420_to_500000/checkpoints/nfcgs_main_joint/version_0/step000500000.ckpt"
OUTPUT="${REPO_DIR}/outputs/nfcgs_main/joint_extension_eval/lambda0p0064/step500000/job_${SLURM_JOB_ID:-manual}"
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing 500k checkpoint: ${CHECKPOINT}" >&2; exit 1; }
python scripts/check_nfcgs_checkpoint.py "${CHECKPOINT}" --step 500000
echo "CHECKPOINT=${CHECKPOINT}"
echo "EVAL_OUTPUT=${OUTPUT}"

exec srun --ntasks=1 python scripts/run_nfcgs.py eval \
  --rate-lambda 0.0064 \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}"
