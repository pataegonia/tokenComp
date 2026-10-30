#!/usr/bin/env bash
#SBATCH -J gs-fixed-order
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -p batch_ugrad
#SBATCH -t 08:00:00
#SBATCH -o logs/slurm/slurm-gs-fixed-order-%j.out
#SBATCH -e logs/slurm/slurm-gs-fixed-order-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
MODE="${1:-run}"
[[ $# -le 1 && ( "${MODE}" == run || "${MODE}" == plan ) ]] || {
  echo "usage: bash $0 plan | sbatch $0" >&2; exit 2;
}
for file in scripts/analyze_nfcgs_fixed_order.py scripts/eval_nfcgs_independent_receiver.py \
  scripts/slurm/eval_nfcgs_independent_receiver.slurm scripts/eval_nfcgs_path_diagnostics.py \
  scripts/profile_nfcgs_complexity.py globalsplat/compression/mechanism.py; do
  [[ -s "${file}" ]] || { echo "missing fixed-order dependency: ${file}" >&2; exit 1; }
done
OUTPUT="${FIXED_ORDER_OUTPUT:-${REPO_DIR}/outputs/nfcgs_fixed_order/job_${SLURM_JOB_ID:-plan}}"
ARGS=(--input-run "${FIXED_ORDER_INPUT_RUN:-${REPO_DIR}/outputs/nfcgs_legacy3_anchors/job_454995}"
  --path-run "${FIXED_ORDER_PATH_RUN:-${REPO_DIR}/outputs/nfcgs_path_diagnostics/job_458004}"
  --output "${OUTPUT}" --scenes 256 --tokens 4096 --seed 0 --precision bf16 --device cuda
  --threads 8 --num-workers 0 --trace-scenes "${FIXED_ORDER_TRACE_SCENES:-1}")
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=8 WANDB_MODE=disabled
if [[ "${MODE}" == plan ]]; then
  exec python scripts/analyze_nfcgs_fixed_order.py plan "${ARGS[@]}"
fi
: "${SLURM_JOB_ID:?submit with sbatch}"
NFCGS_TMP_ROOT="${NFCGS_TMP_ROOT:-/ceph_data/${USER:?missing USER}/.tmp/nfcgs}"
export TMPDIR="${NFCGS_TMP_ROOT}/fixed_order_${SLURM_JOB_ID}"
mkdir -p -- "${TMPDIR}"
export -n RANK 2>/dev/null || true
if [[ "${FIXED_ORDER_RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi
echo "FIXED_ORDER_OUTPUT=${OUTPUT} TMPDIR=${TMPDIR}"
srun --ntasks=1 python scripts/analyze_nfcgs_fixed_order.py run "${ARGS[@]}"
echo "FIXED_ORDER_REPORT=${OUTPUT}/REPORT.md"
