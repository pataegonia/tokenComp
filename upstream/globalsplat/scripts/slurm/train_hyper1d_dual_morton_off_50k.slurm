#!/usr/bin/env bash
#SBATCH -J gs-msh2-off50k
#SBATCH --nodes=1
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-msh2-off50k-%j.out
#SBATCH -e logs/slurm/slurm-gs-msh2-off50k-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
export REPO_DIR
cd "${REPO_DIR}"

vanilla="${VANILLA_CHECKPOINT:-${REPO_DIR}/checkpoints/pretrained/globalsplat-re10k-32k.ckpt}"
output="${REPO_DIR}/outputs/hyper1d_msh_compare_50k/base_residual_msh/morton_off/lambda0p0256/job_${SLURM_JOB_ID}"
if [[ ! -f "${vanilla}" ]]; then
  echo "Missing vanilla checkpoint: ${vanilla}" >&2
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

printf '[Morton OFF 2-path train] checkpoint=%s output=%s\n' "${vanilla}" "${output}"
exec bash "${REPO_DIR}/scripts/slurm/train_hyper1d_12h.slurm" \
  --vanilla-checkpoint "${vanilla}" \
  --architecture legacy --paths 2 --strides 4x --lambda 0.0256 \
  --output "${output}" --no-time-limit --max-steps 50000 \
  --warmup-steps 1000 --checkpoint-every 500 --validate-every 2000
