#!/usr/bin/env bash
#SBATCH -J gs-nonlinear50k-100
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v11
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm/slurm-gs-nonlinear50k-100-%j.out
#SBATCH -e logs/slurm/slurm-gs-nonlinear50k-100-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
VALIDATION_SCENES="${1:?pass the validation scene JSON path as the first argument}"
[[ -f "${VALIDATION_SCENES}" ]] || { echo "missing validation JSON: ${VALIDATION_SCENES}" >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=8
RATE_LAMBDA="${RATE_LAMBDA:-0.0064}"
case "${RATE_LAMBDA}" in
  0.0064|0.0256) ;;
  *) echo "RATE_LAMBDA must be 0.0064 or 0.0256" >&2; exit 2 ;;
esac
TAG="${RATE_LAMBDA/./p}"
CHECKPOINT="${REPO_DIR}/outputs/nfcgs_paper24_transform_train/nonlinear32/rank56/lambda${TAG}/residual_on/checkpoints/nfcgs_transform_nonlinear32_rank56_lambda${TAG}_residual_on_morton_on/version_0/step000050000.ckpt"
[[ -s "${CHECKPOINT}" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }
OUTPUT="${REPO_DIR}/outputs/nfcgs_nonlinear50k_test100/lambda${TAG}/job_${SLURM_JOB_ID}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "VALIDATION_SCENES=${VALIDATION_SCENES}"
echo "OUTPUT=${OUTPUT}"
exec srun --ntasks=1 python scripts/eval_nonlinear50k_subset.py \
  --checkpoint "${CHECKPOINT}" \
  --validation-scenes "${VALIDATION_SCENES}" \
  --dataset-root "${DATASET_ROOT:-/data3/local_datasets/re10k}" \
  --count 100 --seed 111123 --workers 4 --output "${OUTPUT}"
