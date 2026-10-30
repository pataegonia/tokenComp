#!/usr/bin/env bash
#SBATCH -J gs-independent-receiver
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -p batch_ugrad
#SBATCH -t 08:00:00
#SBATCH -o logs/slurm/slurm-gs-independent-receiver-%j.out
#SBATCH -e logs/slurm/slurm-gs-independent-receiver-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
MODE="${1:-run}"
[[ $# -le 1 && ( "${MODE}" == run || "${MODE}" == plan ) ]] || {
  echo "usage: bash $0 plan | sbatch $0" >&2; exit 2;
}
for file in scripts/eval_nfcgs_independent_receiver.py scripts/eval_nfcgs_path_diagnostics.py \
  scripts/profile_nfcgs_complexity.py config/plans/nfcgs_path_diagnostics_20261019.json \
  globalsplat/compression/path_diagnostics.py scripts/slurm/eval_nfcgs_path_diagnostics.slurm; do
  [[ -s "${file}" ]] || { echo "missing receiver dependency: ${file}" >&2; exit 1; }
done
OUTPUT="${INDEPENDENT_RECEIVER_OUTPUT:-${REPO_DIR}/outputs/nfcgs_independent_receiver/job_${SLURM_JOB_ID:-plan}}"
SOURCE="${INDEPENDENT_RECEIVER_PATH_RUN:-${REPO_DIR}/outputs/nfcgs_path_diagnostics/job_458004}"
CHECKPOINT="${REPO_DIR}/outputs/nfcgs_order_context/hilbert/dyadic4_stages4_k5_ste/lambda0p0128/job_447746/checkpoints/nfcgs_main/version_0/step000050000.ckpt"
SOURCE_ARGS=(--path-run "${SOURCE}" --manifest "${REPO_DIR}/config/plans/nfcgs_path_diagnostics_20261019.json")
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=8 WANDB_MODE=disabled
if [[ "${MODE}" == plan ]]; then
  exec python scripts/eval_nfcgs_independent_receiver.py plan "${SOURCE_ARGS[@]}"
fi
: "${SLURM_JOB_ID:?submit with sbatch}"
NFCGS_TMP_ROOT="${NFCGS_TMP_ROOT:-/ceph_data/${USER:?missing USER}/.tmp/nfcgs}"
export TMPDIR="${NFCGS_TMP_ROOT}/independent_receiver_${SLURM_JOB_ID}"
mkdir -p -- "${TMPDIR}"
export -n RANK 2>/dev/null || true
echo "INDEPENDENT_RECEIVER_SOURCE=${SOURCE} OUTPUT=${OUTPUT} TMPDIR=${TMPDIR}"
# The exporter exits completely before the packet receiver starts.
srun --ntasks=1 python scripts/eval_nfcgs_independent_receiver.py prepare "${SOURCE_ARGS[@]}" \
  --bundle "${OUTPUT}/bundle" --dataset-root "${DATASET_ROOT:-/data3/local_datasets/re10k}" \
  --threads 8 --num-workers "${INDEPENDENT_RECEIVER_NUM_WORKERS:-4}"
srun --ntasks=1 python scripts/eval_nfcgs_independent_receiver.py receive \
  --bundle "${OUTPUT}/bundle" --checkpoint "${CHECKPOINT}" --output "${OUTPUT}/receiver" \
  --device cuda --precision bf16 --threads 8
echo "INDEPENDENT_RECEIVER_REPORT=${OUTPUT}/receiver/REPORT.md"
