#!/usr/bin/env bash
#SBATCH -J gs-hyper1d-12h
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v9
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-hyper1d-%j.out
#SBATCH -e logs/slurm/slurm-gs-hyper1d-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# A plain `sbatch this_script.slurm` starts the default vanilla-initialized pilot.
# Explicit --checkpoint/--vanilla-checkpoint arguments always take precedence.
run_args=("$@")
source_given=false
dry_run=false
for arg in "${run_args[@]}"; do
  case "${arg}" in
    --checkpoint|--checkpoint=*|--vanilla-checkpoint|--vanilla-checkpoint=*) source_given=true ;;
    --dry-run) dry_run=true ;;
  esac
done
if [[ "${source_given}" == false ]]; then
  vanilla_checkpoint="${VANILLA_CHECKPOINT:-${REPO_DIR}/checkpoints/pretrained/globalsplat-re10k-32k.ckpt}"
  if [[ "${dry_run}" == false && ! -f "${vanilla_checkpoint}" ]]; then
    echo "Missing vanilla checkpoint: ${vanilla_checkpoint}" >&2
    echo "Pass --vanilla-checkpoint /path/to/globalsplat-re10k-32k.ckpt or --checkpoint /path/to/hyper1d.ckpt." >&2
    exit 2
  fi
  run_args=(--vanilla-checkpoint "${vanilla_checkpoint}" "${run_args[@]}")
  echo "[hyper1d] default vanilla checkpoint: ${vanilla_checkpoint}"
fi
exec srun --ntasks=1 python scripts/run_hyper1d.py train "${run_args[@]}"
