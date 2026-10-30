#!/usr/bin/env bash
# Same GPU, same cached input scenes, all checkpoint variants in one allocation.
#SBATCH -J gs-complexity
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -p batch_ugrad
#SBATCH -t 06:00:00
#SBATCH -o logs/slurm/slurm-gs-complexity-%j.out
#SBATCH -e logs/slurm/slurm-gs-complexity-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
MODE="${1:-run}"
[[ "${MODE}" == run || "${MODE}" == plan ]] || {
  echo "usage: bash $0 plan | sbatch $0" >&2
  exit 2
}
MANIFEST="${PROFILE_MANIFEST:-${REPO_DIR}/logs/slurm/nfcgs_complexity_manifest_2026-10-16.json}"
[[ -f "${MANIFEST}" ]] || { echo "missing manifest: ${MANIFEST}" >&2; exit 1; }
OUTPUT="${PROFILE_OUTPUT:-${REPO_DIR}/outputs/nfcgs_complexity/job_${SLURM_JOB_ID:-plan}}"
ARGS=(--manifest "${MANIFEST}" --mode actual --output "${OUTPUT}"
  --device cuda --precision bf16 --tokens 4096 --threads 8 --seed 0
  --dataset-root "${DATASET_ROOT:-/data3/local_datasets/re10k}"
  --num-workers "${PROFILE_NUM_WORKERS:-4}" --warmup "${PROFILE_WARMUP:-4}"
  --scenes "${PROFILE_SCENES:-32}" --repeats "${PROFILE_REPEATS:-3}")
[[ "${PROFILE_SKIP_MISSING:-0}" != 1 ]] || ARGS+=(--skip-missing)
[[ "${PROFILE_RESUME:-0}" != 1 ]] || ARGS+=(--resume)
if [[ "${MODE}" == plan ]]; then
  printf '%q ' python scripts/profile_nfcgs_complexity.py "${ARGS[@]}"
  printf '\n'
  exit 0
fi
: "${SLURM_JOB_ID:?submit with sbatch}"
NFCGS_TMP_ROOT="${NFCGS_TMP_ROOT:-/ceph_data/${USER:?missing USER}/.tmp/nfcgs}"
export TMPDIR="${NFCGS_TMP_ROOT}/profile_${SLURM_JOB_ID}"
mkdir -p -- "${TMPDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 WANDB_MODE=disabled
export -n RANK 2>/dev/null || true
echo "COMPLEXITY_TMPDIR=${TMPDIR} OUTPUT=${OUTPUT}"
exec srun --ntasks=1 python scripts/profile_nfcgs_complexity.py "${ARGS[@]}"
