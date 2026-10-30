#!/usr/bin/env bash
#SBATCH -J gs-seed-pair-train
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-seed-pair-train-%j.out
#SBATCH -e logs/slurm/slurm-gs-seed-pair-train-%j.err
set -euo pipefail
cd "${REPO_DIR:?}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=8
export TMPDIR="${NFCGS_TMP_ROOT:-/ceph_data/${USER:?}/.tmp/nfcgs}/job_${SLURM_JOB_ID:?}"
mkdir -p -- "${TMPDIR}"
echo "CHECKPOINT_TMPDIR=${TMPDIR} ARM=${CONTEXT_SEED_ARM:?} MODEL_SEED=111124 DATA_LOADER_SEED=404"
RUN="${CONTEXT_SEED_OUTPUT_ROOT:?}/${CONTEXT_SEED_ARM}/job_${SLURM_JOB_ID}"
exec python scripts/run_nfcgs_context_seed_pair.py train --arm "${CONTEXT_SEED_ARM}" \
  --checkpoint "${SOURCE_CHECKPOINT:?}" --dataset-root "${DATASET_ROOT:?}" --run-dir "${RUN}" --launcher srun
