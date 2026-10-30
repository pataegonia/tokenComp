#!/usr/bin/env bash
#SBATCH -J gs-token-adjacency
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -p batch_ugrad
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm/slurm-gs-token-adjacency-%j.out
#SBATCH -e logs/slurm/slurm-gs-token-adjacency-%j.err

# CPU-only: reads the cached matched-scene contexts, no dataset, no GPU.
set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
: "${SLURM_JOB_ID:?submit with sbatch}"
MANIFEST="${ADJACENCY_MANIFEST:-${REPO_DIR}/config/plans/nfcgs_path_diagnostics_fresh120k_h_on_lambda0p0128.json}"
OUTPUT="${ADJACENCY_OUTPUT:-${REPO_DIR}/outputs/nfcgs_token_adjacency/job_${SLURM_JOB_ID}}"
THREADS="${SLURM_CPUS_PER_TASK:-16}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}"
echo "TOKEN_ADJACENCY_MANIFEST=${MANIFEST} OUTPUT=${OUTPUT} NODE=${SLURMD_NODENAME:-$(hostname)}"
python scripts/analyze_nfcgs_token_adjacency.py --manifest "${MANIFEST}" --output "${OUTPUT}" \
  --device cpu --threads "${THREADS}" --limit "${ADJACENCY_LIMIT:-0}"
