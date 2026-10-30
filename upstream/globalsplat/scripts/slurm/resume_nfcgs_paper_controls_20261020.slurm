#!/usr/bin/env bash
#SBATCH -J gs-paper-control-resume
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-paper-control-resume-%j.out
#SBATCH -e logs/slurm/slurm-gs-paper-control-resume-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
: "${SLURM_JOB_ID:?submit using the recovery controller}"
: "${RESUME_ORIGINAL_TRAIN:?missing original training ID}"
: "${RESUME_RUN_DIR:?missing original run directory}"
: "${RESUME_CHECKPOINT:?missing full-state checkpoint}"
: "${RESUME_EXPECTED_STEP:?missing checkpoint step}"
: "${RESUME_EXPECTED_SHA256:?missing checkpoint SHA-256}"
: "${RESUME_EXPECTED_CONFIG:?missing codec config}"
[[ "${RESUME_ORIGINAL_TRAIN}" == 456951 || "${RESUME_ORIGINAL_TRAIN}" == 457099 ]] || {
  echo 'this recovery is restricted to trains 456951 and 457099' >&2; exit 2;
}
NFCGS_TMP_ROOT="${NFCGS_TMP_ROOT:-/ceph_data/${USER:?missing USER}/.tmp/nfcgs}"
export TMPDIR="${NFCGS_TMP_ROOT}/job_${SLURM_JOB_ID}"
mkdir -p -- "${TMPDIR}"
source scripts/slurm/nfcgs_order_context_settings.sh
[[ "${SPATIAL_CONTEXT}" == false && "${SCORE_SLICE_CHANNELS}" == 16 && "${MAX_STEPS}" == 50000 ]] || {
  echo 'resume requires matched OFF16/OFF12 and an absolute 50000 step target' >&2; exit 2;
}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled OMP_NUM_THREADS=8
SNAPSHOT="${RESUME_RUN_DIR}/checkpoints/nfcgs_main/version_0/resume_job_${SLURM_JOB_ID}.ckpt"
python scripts/recover_nfcgs_paper_controls_20261020.py \
  --copy-source "${RESUME_CHECKPOINT}" --copy-target "${SNAPSHOT}" \
  --expected-step "${RESUME_EXPECTED_STEP}" --expected-sha256 "${RESUME_EXPECTED_SHA256}" \
  --expected-config "${RESUME_EXPECTED_CONFIG}"
echo "ORIGINAL_TRAIN=${RESUME_ORIGINAL_TRAIN} RESUME_JOB=${SLURM_JOB_ID} TRAIN_OUTPUT=${RESUME_RUN_DIR}"
echo 'RESUME=full-state TARGET_GLOBAL_STEP=50000 GPUS=2 BATCH_PER_GPU=2 ACCUMULATE=2 MODEL_SEED=111123 DATA_SEED=403'
exec python scripts/run_nfcgs.py train --devices 2 --launcher srun --resume \
  --checkpoint "${SNAPSHOT}" --rate-lambda "${RATE_LAMBDA}" "${CODEC_ARGS[@]}" \
  --scope all --lr 0.0001 --batch-size 2 --accumulate 2 --workers 8 \
  --max-steps 50000 --checkpoint-every 5000 --output "${RESUME_RUN_DIR}"
