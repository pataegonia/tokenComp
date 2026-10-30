#!/usr/bin/env bash
#SBATCH -J gs-a-prepare
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -p batch_ugrad
#SBATCH -t 4:00:00
#SBATCH -o logs/slurm/slurm-gs-arch-prepare-%j.out
#SBATCH -e logs/slurm/slurm-gs-arch-prepare-%j.err
set -euo pipefail
cd "${REPO_DIR:?missing REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
extra=()
if [[ -n "${INIT_SEED:-}" ]]; then extra+=(--seed "$INIT_SEED"); fi
if [[ -n "${ARMS:-}" ]]; then
  read -r -a arms <<< "$ARMS"
  extra+=(--arms "${arms[@]}")
fi
if [[ -n "${HOLDOUT_REFERENCE:-}" ]]; then extra+=(--holdout-reference "$HOLDOUT_REFERENCE"); fi
python scripts/prepare_nfcgs_architecture_pilots.py --vanilla "${VANILLA_CHECKPOINT:?}" \
  --dataset-root "${DATASET_ROOT:?}" --output "${PREPARATION_DIR:?}" "${extra[@]}"
