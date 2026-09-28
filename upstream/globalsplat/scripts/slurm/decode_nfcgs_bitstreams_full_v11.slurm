#!/usr/bin/env bash
# Receiver: saved E2EM0301 files -> tokens -> Gaussians -> full-scene metrics.
#SBATCH -J gs-decode-full
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v11
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-decode-full-%j.out
#SBATCH -e logs/slurm/slurm-gs-decode-full-%j.err

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
: "${BITSTREAM_DIR:?BITSTREAM_DIR must point to the completed sender output}"
OUTPUT="${OUTPUT:-${BITSTREAM_DIR%/bitstreams}/decoder_eval}"
[[ -f "${CHECKPOINT}" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${BITSTREAM_DIR}/COMPLETE.json" ]] || {
  echo "sender output is incomplete: ${BITSTREAM_DIR}" >&2
  exit 1
}
python scripts/check_nfcgs_checkpoint.py "${CHECKPOINT}" --step 10000

echo "ROLE=receiver"
echo "INPUT=bitstream files only (plus target cameras/ground truth for evaluation)"
echo "BITSTREAM_DIR=${BITSTREAM_DIR}"
echo "EVAL_OUTPUT=${OUTPUT}"
exec srun --ntasks=1 python scripts/eval_nfcgs_sender_receiver.py decode \
  --checkpoint "${CHECKPOINT}" \
  --expected-step 10000 \
  --dataset-root "${DATASET_ROOT}" \
  --bitstream-dir "${BITSTREAM_DIR}" \
  --output "${OUTPUT}" \
  --workers "${WORKERS:-4}" \
  --precision "${PRECISION:-bf16-mixed}"
