#!/usr/bin/env bash
#SBATCH -J gs-msh-fulltest
#SBATCH --nodes=1
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH --array=0-1%2
#SBATCH -t 12:00:00
#SBATCH -w ariel-v11
#SBATCH -o logs/slurm/slurm-gs-msh-fulltest-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-msh-fulltest-%A_%a.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"

case "${SLURM_ARRAY_TASK_ID:?submit with sbatch as an array}" in
  0) variant=single_msh ;;
  1) variant=base_residual_msh ;;
  *) echo "Invalid array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

train_root="${REPO_DIR}/outputs/hyper1d_msh_compare_50k"
checkpoint="${train_root}/${variant}/morton_on/lambda0p0256/job_438115_${SLURM_ARRAY_TASK_ID}/checkpoints/hyper1d_12h/version_0/step000050000.ckpt"
output="${train_root}/full_test/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/${variant}"
if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing evaluation checkpoint: ${checkpoint}" >&2
  exit 2
fi

conda_base="${CONDA_BASE:-/ceph_data/clue9986/anaconda3}"
if command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base)"
fi
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

printf '[full test] variant=%s checkpoint=%s output=%s\n' "${variant}" "${checkpoint}" "${output}"
srun --ntasks=1 python scripts/run_hyper1d.py eval \
  --checkpoint "${checkpoint}" --all-test --workers 0 --output "${output}" "$@"
