#!/usr/bin/env bash
# Sender: context images -> one E2EM0301 bitstream file per full-eval scene.
#SBATCH -J gs-encode-full
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v11
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-encode-full-%j.out
#SBATCH -e logs/slurm/slurm-gs-encode-full-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0064/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0064_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
BITSTREAM_DIR="${BITSTREAM_DIR:-${REPO_DIR}/outputs/nfcgs_main/sender_receiver_full/existing_b/manual_${SLURM_JOB_ID:-encode}/bitstreams}"
[[ -f "${CHECKPOINT}" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }
python scripts/check_nfcgs_checkpoint.py "${CHECKPOINT}" --step 10000

echo "ROLE=sender"
echo "B_MODE=existing"
echo "SCENES=full"
echo "BITSTREAM_DIR=${BITSTREAM_DIR}"
exec srun --ntasks=1 python scripts/eval_nfcgs_sender_receiver.py encode \
  --checkpoint "${CHECKPOINT}" \
  --expected-step 10000 \
  --dataset-root "${DATASET_ROOT}" \
  --bitstream-dir "${BITSTREAM_DIR}" \
  --workers "${WORKERS:-4}" \
  --precision "${PRECISION:-bf16-mixed}"
