#!/usr/bin/env bash
#SBATCH -J gs-hyper1d-eval
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v11
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm/slurm-gs-hyper1d-eval-%j.out
#SBATCH -e logs/slurm/slurm-gs-hyper1d-eval-%j.err

set -euo pipefail
if [[ $# -eq 0 ]]; then
  echo "Usage: sbatch scripts/slurm/eval_hyper1d.slurm CHECKPOINT [run_hyper1d eval options]" >&2
  exit 2
fi
if [[ "$1" == --help || "$1" == -h ]]; then
  echo "Usage: sbatch scripts/slurm/eval_hyper1d.slurm CHECKPOINT [run_hyper1d eval options]"
  echo "Defaults: sample 32 test scenes, seed 111123, excluding the run's validation scenes."
  echo "Use --max-scenes, --sample-seed, --output or --save-images to customize evaluation."
  exit 0
fi
checkpoint="$1"
shift
if [[ "${checkpoint}" == --* ]]; then
  echo "The first argument must be a Hyper1D checkpoint path." >&2
  exit 2
fi
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
dry_run=false
for arg in "$@"; do
  if [[ "${arg}" == --dry-run ]]; then
    dry_run=true
  fi
done
if [[ "${dry_run}" == false && ! -f "${checkpoint}" ]]; then
  echo "Missing evaluation checkpoint: ${checkpoint}" >&2
  exit 2
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec srun --ntasks=1 python scripts/run_hyper1d.py eval \
  --checkpoint "${checkpoint}" --sample-test --max-scenes 32 \
  --sample-seed 111123 --workers 0 "$@"
