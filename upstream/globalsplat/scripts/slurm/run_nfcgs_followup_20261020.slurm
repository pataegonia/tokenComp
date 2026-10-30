#!/usr/bin/env bash
#SBATCH -J gs-followup20
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -p batch_ugrad
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-followup20-%j.out
#SBATCH -e logs/slurm/slurm-gs-followup20-%j.err

set -euo pipefail
[[ $# == 1 ]] || { echo 'task required: boundary/native/fixed-render' >&2; exit 2; }
TASK="$1"
case "${TASK}" in boundary|native|fixed-render) ;; *) echo 'invalid task' >&2; exit 2 ;; esac
: "${SLURM_JOB_ID:?use the submit_nfcgs_followup_20261020.py launcher}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd -P -- "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
NFCGS_TMP_ROOT="${NFCGS_TMP_ROOT:-/ceph_data/${USER:?missing USER}/.tmp/nfcgs}"
export TMPDIR="${NFCGS_TMP_ROOT}/followup20_${TASK}_${SLURM_JOB_ID}"
mkdir -p -- "${TMPDIR}"
export -n RANK 2>/dev/null || true
OUTPUT="${FOLLOWUP_OUTPUT:-${REPO_DIR}/outputs/nfcgs_followup_20261020/${TASK}/job_${SLURM_JOB_ID}}"
ARGS=("${TASK}" --output "${OUTPUT}" --threads "${SLURM_CPUS_PER_TASK:-8}"
  --dataset-root "${DATASET_ROOT:-/data3/local_datasets/re10k}"
  --num-workers "${FOLLOWUP_NUM_WORKERS:-4}")
[[ -z "${FOLLOWUP_B0_RUN:-}" ]] || ARGS+=(--b0-run "${FOLLOWUP_B0_RUN}")
[[ -z "${FOLLOWUP_PATH_RUN:-}" ]] || ARGS+=(--path-run "${FOLLOWUP_PATH_RUN}")
[[ -z "${FOLLOWUP_FIXED_RUN:-}" ]] || ARGS+=(--fixed-run "${FOLLOWUP_FIXED_RUN}")
[[ -z "${FOLLOWUP_TARGET_BUNDLE:-}" ]] || ARGS+=(--bundle "${FOLLOWUP_TARGET_BUNDLE}")
[[ -z "${FOLLOWUP_CHECKPOINT:-}" ]] || ARGS+=(--checkpoint "${FOLLOWUP_CHECKPOINT}")
[[ -z "${FOLLOWUP_NATIVE_CHECKPOINT:-}" ]] || ARGS+=(--native-checkpoint "${FOLLOWUP_NATIVE_CHECKPOINT}")
[[ -z "${FOLLOWUP_REFERENCE:-}" ]] || ARGS+=(--reference "${FOLLOWUP_REFERENCE}")
[[ "${FOLLOWUP_RESUME:-0}" != 1 ]] || ARGS+=(--resume)
echo "FOLLOWUP_TASK=${TASK} OUTPUT=${OUTPUT} TMPDIR=${TMPDIR}"
exec srun --ntasks=1 python scripts/run_nfcgs_followup_20261020.py "${ARGS[@]}"
