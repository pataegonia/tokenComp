#!/usr/bin/env bash
#SBATCH -J gs-hyper1d-resume
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v9
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-hyper1d-resume-%j.out
#SBATCH -e logs/slurm/slurm-gs-hyper1d-resume-%j.err

set -euo pipefail
if [[ $# -eq 0 ]]; then
  echo "Usage: sbatch scripts/slurm/resume_hyper1d_12h.slurm CHECKPOINT [run_hyper1d options]" >&2
  exit 2
fi
if [[ "$1" == --help || "$1" == -h ]]; then
  echo "Usage: sbatch scripts/slurm/resume_hyper1d_12h.slurm CHECKPOINT [run_hyper1d options]"
  echo "Defaults: resume to 32000 total steps, save every 500 steps, train for up to 11 more hours."
  echo "Pass --output with the original run directory to continue its logs and validation reports."
  exit 0
fi
checkpoint="$1"
shift
if [[ "${checkpoint}" == --* ]]; then
  echo "The first argument must be a full training checkpoint path." >&2
  exit 2
fi

export REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
dry_run=false
for arg in "$@"; do
  if [[ "${arg}" == --dry-run ]]; then
    dry_run=true
  fi
done
if [[ "${dry_run}" == false && ! -f "${checkpoint}" ]]; then
  echo "Missing resume checkpoint: ${checkpoint}" >&2
  exit 2
fi

# Reuse environment setup and srun from the pilot wrapper. Later arguments can
# override the step/time/save defaults; --max-steps is an absolute step limit.
exec bash "${REPO_DIR}/scripts/slurm/train_hyper1d_12h.slurm" \
  --checkpoint "${checkpoint}" --resume \
  --max-steps 32000 --train-hours 11 \
  --checkpoint-every 500 --validate-every 2000 "$@"
