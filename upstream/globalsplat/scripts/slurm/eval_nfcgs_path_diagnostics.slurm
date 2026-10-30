#!/usr/bin/env bash
#SBATCH -J gs-path-diagnostics
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -p batch_ugrad
#SBATCH -t 08:00:00
#SBATCH -o logs/slurm/slurm-gs-path-diagnostics-%j.out
#SBATCH -e logs/slurm/slurm-gs-path-diagnostics-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
MODE="${1:-run}"
[[ $# -le 1 && ( "${MODE}" == run || "${MODE}" == plan ) ]] || {
  echo "usage: bash $0 plan | sbatch $0" >&2; exit 2;
}
MANIFEST="${PATH_DIAGNOSTIC_MANIFEST:-${REPO_DIR}/config/plans/nfcgs_path_diagnostics_20261019.json}"
for file in globalsplat/compression/path_diagnostics.py scripts/eval_nfcgs_path_diagnostics.py \
  "${MANIFEST}" scripts/profile_nfcgs_complexity.py; do
  [[ -s "${file}" ]] || { echo "missing diagnostic dependency: ${file}" >&2; exit 1; }
done
INPUT_RUN="${PATH_DIAGNOSTIC_INPUT_RUN:-${REPO_DIR}/outputs/nfcgs_legacy3_anchors/job_454995}"
OUTPUT="${PATH_DIAGNOSTIC_OUTPUT:-${REPO_DIR}/outputs/nfcgs_path_diagnostics/job_${SLURM_JOB_ID:-plan}}"
ARGS=(--manifest "${MANIFEST}"
  --input-run "${INPUT_RUN}" --output "${OUTPUT}" --scenes 256 --tokens 4096 --seed 0
  --expected-input-fingerprint 069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29
  --device cuda --precision bf16 --threads 8 --num-workers "${PATH_DIAGNOSTIC_NUM_WORKERS:-4}"
  --dataset-root "${DATASET_ROOT:-/data3/local_datasets/re10k}")
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=8 WANDB_MODE=disabled
if [[ "${MODE}" == plan ]]; then
  exec python scripts/eval_nfcgs_path_diagnostics.py "${ARGS[@]}" --plan
fi
: "${SLURM_JOB_ID:?submit with sbatch}"
NFCGS_TMP_ROOT="${NFCGS_TMP_ROOT:-/ceph_data/${USER:?missing USER}/.tmp/nfcgs}"
export TMPDIR="${NFCGS_TMP_ROOT}/path_diagnostics_${SLURM_JOB_ID}"
mkdir -p -- "${TMPDIR}"
export -n RANK 2>/dev/null || true
echo "PATH_DIAGNOSTICS_INPUT=${INPUT_RUN} OUTPUT=${OUTPUT} TMPDIR=${TMPDIR}"
srun --ntasks=1 python scripts/eval_nfcgs_path_diagnostics.py "${ARGS[@]}"
echo "PATH_DIAGNOSTICS_REPORT=${OUTPUT}/REPORT.md"
