#!/usr/bin/env bash
#SBATCH -J gs-msh-off-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm/slurm-gs-msh-off-eval-%j.out
#SBATCH -e logs/slurm/slurm-gs-msh-off-eval-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"

checkpoint="${REPO_DIR}/outputs/hyper1d/20260928_161549_981799/checkpoints/hyper1d_12h/version_0/step000050000.ckpt"
output="${REPO_DIR}/outputs/hyper1d_msh_compare_50k/single_msh/morton_off/lambda0p0256/full_test/job_${SLURM_JOB_ID}"
if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing evaluation checkpoint: ${checkpoint}" >&2
  exit 2
fi

# Keep temporary files off node-local /tmp; a short path also fits Unix sockets.
export TMPDIR="${GLOBALSPLAT_TMP_ROOT:-/ceph_data/clue9986/tmp}/gs_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
export TMP="${TMPDIR}" TEMP="${TMPDIR}"

conda_base="${CONDA_BASE:-/ceph_data/clue9986/anaconda3}"
if command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base)"
fi
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

printf '[Morton OFF 1-path full test] checkpoint=%s output=%s\n' "${checkpoint}" "${output}"
exec srun --ntasks=1 python scripts/run_hyper1d.py eval \
  --checkpoint "${checkpoint}" --all-test --workers 0 --output "${output}"
