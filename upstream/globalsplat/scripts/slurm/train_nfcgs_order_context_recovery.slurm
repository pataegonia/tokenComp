#!/usr/bin/env bash
# Full-state resume when a checkpoint exists; otherwise the original warm start.
#SBATCH -J gs-oc-recovery
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-oc-recovery-%j.out
#SBATCH -e logs/slurm/slurm-gs-oc-recovery-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
# Lightning/fsspec transactions stage checkpoint bytes in tempfile.gettempdir().
# Keep that staging space on Ceph, using a short path for multiprocessing sockets.
NFCGS_TMP_ROOT="${NFCGS_TMP_ROOT:-/ceph_data/${USER:?missing USER}/.tmp/nfcgs}"
export TMPDIR="${NFCGS_TMP_ROOT}/job_${SLURM_JOB_ID:?submit with sbatch}"
mkdir -p -- "${TMPDIR}"
echo "CHECKPOINT_TMPDIR=${TMPDIR}"
source scripts/slurm/nfcgs_order_context_settings.sh
: "${RECOVERY_MODE:?submit with recover_nfcgs_sort_context_ablation.py}"
: "${RECOVERY_CHECKPOINT:?missing source checkpoint}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
OUTPUT="${RUN_ROOT}/recovery_job_${SLURM_JOB_ID:?submit with sbatch}"
LOAD="${RECOVERY_CHECKPOINT}"
MODE_ARGS=()
case "${RECOVERY_MODE}" in
  resume)
    LOAD="${OUTPUT}/checkpoints/nfcgs_main/version_0/resume.ckpt"
    python scripts/recover_nfcgs_sort_context_ablation.py --copy-checkpoint "${RECOVERY_CHECKPOINT}" \
      --copy-to "${LOAD}" --expected-config "${RECOVERY_CODEC_CONFIG:?missing expected config}"
    MODE_ARGS+=(--resume)
    ;;
  restart) MODE_ARGS+=(--allow-score-path-conversion) ;;
  *) echo "invalid RECOVERY_MODE=${RECOVERY_MODE}" >&2; exit 2 ;;
esac
echo "RECOVERY_MODE=${RECOVERY_MODE} TOKEN_ORDER=${TOKEN_ORDER} CONTEXT_SCHEDULE=${CONTEXT_SCHEDULE} SPATIAL_CONTEXT=${SPATIAL_CONTEXT}"
echo "SOURCE_CHECKPOINT=${RECOVERY_CHECKPOINT} TRAIN_OUTPUT=${OUTPUT} MAX_STEPS=${MAX_STEPS}"
echo "GPUS=2 BATCH_PER_GPU=2 ACCUMULATE=2 EFFECTIVE_BATCH=8 MSH=on"
exec python scripts/run_nfcgs.py train --devices 2 --launcher srun \
  --checkpoint "${LOAD}" "${MODE_ARGS[@]}" --rate-lambda "${RATE_LAMBDA}" "${CODEC_ARGS[@]}" \
  --scope all --lr 0.0001 --batch-size 2 --accumulate 2 --workers 8 \
  --max-steps "${MAX_STEPS}" --checkpoint-every "${CHECKPOINT_EVERY}" --output "${OUTPUT}"
