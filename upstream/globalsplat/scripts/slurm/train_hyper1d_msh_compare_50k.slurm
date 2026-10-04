#!/usr/bin/env bash
#SBATCH -J gs-msh-compare50k
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v11
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH -p batch_ugrad
#SBATCH --array=0-1%2
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-msh-compare50k-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-msh-compare50k-%A_%a.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
export REPO_DIR
cd "${REPO_DIR}"

# Both runs initialize from vanilla with the same seed and base MSH weights.
# Task 1 adds a separately parameterized MSH for x - decoded_base(x).
case "${SLURM_ARRAY_TASK_ID:?submit this script with sbatch as an array}" in
  0) paths=1; variant=single_msh ;;
  1) paths=2; variant=base_residual_msh ;;
  *) echo "Invalid array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

vanilla="${VANILLA_CHECKPOINT:-${REPO_DIR}/checkpoints/pretrained/globalsplat-re10k-32k.ckpt}"
output="${REPO_DIR}/outputs/hyper1d_msh_compare_50k/${variant}/morton_on/lambda0p0256/job_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}"
printf '[MSH comparison] variant=%s paths=%s morton=on lambda=0.0256 steps=50000 output=%s\n' "${variant}" "${paths}" "${output}"

exec bash "${REPO_DIR}/scripts/slurm/train_hyper1d_12h.slurm" \
  --vanilla-checkpoint "${vanilla}" \
  --architecture legacy --paths "${paths}" --strides 4x --morton --lambda 0.0256 \
  --output "${output}" --no-time-limit --max-steps 50000 \
  --warmup-steps 1000 --checkpoint-every 500 --validate-every 2000 "$@"
