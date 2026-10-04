#!/usr/bin/env bash
#SBATCH -J gs-hyper1d-hq50k
#SBATCH --nodes=1
#SBATCH --nodelist=ariel-v9
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH --array=0-3%4
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-gs-hyper1d-hq50k-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-hyper1d-hq50k-%A_%a.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
export REPO_DIR
cd "${REPO_DIR}"

# Each task starts from the same vanilla checkpoint within its architecture.
# Lower lambda weakens the bitrate penalty in distortion + lambda * BPGA.
case "${SLURM_ARRAY_TASK_ID:?submit this script with sbatch as an array}" in
  0) architecture=legacy; lambda=0.0128 ;;
  1) architecture=plain4; lambda=0.0128 ;;
  2) architecture=legacy; lambda=0.0064 ;;
  3) architecture=plain4; lambda=0.0064 ;;
  *) echo "Invalid array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

vanilla="${VANILLA_CHECKPOINT:-${REPO_DIR}/checkpoints/pretrained/globalsplat-re10k-32k.ckpt}"
if [[ ! -f "${vanilla}" ]]; then
  echo "Missing vanilla checkpoint: ${vanilla}" >&2
  exit 2
fi

lambda_tag="${lambda//./p}"
output="${REPO_DIR}/outputs/hyper1d_high_quality_50k/${architecture}/lambda${lambda_tag}/job_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}"
printf '[hyper1d sweep] architecture=%s lambda=%s output=%s\n' "${architecture}" "${lambda}" "${output}"

exec bash "${REPO_DIR}/scripts/slurm/train_hyper1d_12h.slurm" \
  --vanilla-checkpoint "${vanilla}" \
  --architecture "${architecture}" --lambda "${lambda}" \
  --output "${output}" --no-time-limit --max-steps 50000 \
  --warmup-steps 1000 --checkpoint-every 500 --validate-every 2000
