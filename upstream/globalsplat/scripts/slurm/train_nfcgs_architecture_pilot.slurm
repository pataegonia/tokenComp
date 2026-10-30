#!/usr/bin/env bash
#SBATCH -J gs-a-train
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-arch-train-%j.out
#SBATCH -e logs/slurm/slurm-gs-arch-train-%j.err
set -euo pipefail
cd "${REPO_DIR:?}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="/ceph_data/${USER:?}/.tmp/nfcgs/job_${SLURM_JOB_ID:?}"
mkdir -p -- "${TMPDIR}"
echo "CHECKPOINT_TMPDIR=${TMPDIR} ARM=${PILOT_ARM:?} INITIALIZER=${PREPARATION_DIR:?}/${PILOT_ARM}.ckpt"
python scripts/smoke_nfcgs_architecture_pilot.py --checkpoint "${PREPARATION_DIR}/${PILOT_ARM}.ckpt"
exec python scripts/run_nfcgs_architecture_pilot.py train --arm "${PILOT_ARM}" \
  --preparation "${PREPARATION_DIR}" --dataset-root "${DATASET_ROOT:?}" --launcher srun \
  --output "${PILOT_OUTPUT_ROOT:?}/${PILOT_ARM}/job_${SLURM_JOB_ID}"
