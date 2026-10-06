#!/usr/bin/env bash
#SBATCH -J gs-score-minimal-eval
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v11
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-score-minimal-eval-EEE%j.out
#SBATCH -e logs/slurm/slurm-gs-score-minimal-eval-%j.err

# Evaluate the completed minimal job 441307 over all processable test scenes.
# Keep its original two-stage even/odd / kernel-3 / noise configuration.
# run_nfcgs.py defaults to those spatial settings in both old and new runners.
set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

TRAIN_ROOT="${REPO_DIR}/outputs/nfcgs_score_path_ablation/minimal/lambda0p0256/job_441307"
CHECKPOINT="${TRAIN_ROOT}/checkpoints/nfcgs_main/version_0/step000050000.ckpt"
OUTPUT="${TRAIN_ROOT}/eval_all/job_${SLURM_JOB_ID:?submit with sbatch}"
[[ -s "${CHECKPOINT}" ]] || { echo "missing trained checkpoint: ${CHECKPOINT}" >&2; exit 1; }
python scripts/check_nfcgs_checkpoint.py "${CHECKPOINT}" --step 50000
echo "CHECKPOINT=${CHECKPOINT}"
echo "SCORE_PATH=minimal SPATIAL_STAGES=2 KERNEL=3 CONTEXT_QUANTIZATION=noise RATE_LAMBDA=0.0256"
echo "GPUS=1 EVAL_SCOPE=all CONTEXT_VIEWS=12 TARGET_VIEWS=8 ACTUAL_BITSTREAM=true"
echo "EVAL_OUTPUT=${OUTPUT}/nfcgs_main"

# No scene limit: actual encode -> decode -> render on every eligible test scene.
exec srun --ntasks=1 python scripts/run_nfcgs.py eval \
  --checkpoint "${CHECKPOINT}" --rate-lambda 0.0256 \
  --score-path minimal --devices 1 --workers 4 \
  --output "${OUTPUT}"
