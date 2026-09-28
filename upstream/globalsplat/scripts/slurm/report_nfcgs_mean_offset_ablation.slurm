#!/usr/bin/env bash
#SBATCH -J gs-mean-offset-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -p batch_ugrad
#SBATCH -t 00:10:00
#SBATCH -o logs/slurm/slurm-gs-mean-offset-report-%j.out
#SBATCH -e logs/slurm/slurm-gs-mean-offset-report-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
: "${TRAIN_ARRAY_JOB_ID:?submit with submit_nfcgs_mean_offset_ablation_v11.sh}"

python scripts/report_nfcgs_mean_offset_ablation.py \
  "${REPO_DIR}/outputs/nfcgs_main/score_mean_offset_ablation/job_${TRAIN_ARRAY_JOB_ID}"
