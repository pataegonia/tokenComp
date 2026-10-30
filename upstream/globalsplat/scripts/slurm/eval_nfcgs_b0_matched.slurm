#!/usr/bin/env bash
#SBATCH -J gs-b0-matched
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -p batch_ugrad
#SBATCH -t 08:00:00
#SBATCH -o logs/slurm/slurm-gs-b0-matched-%j.out
#SBATCH -e logs/slurm/slurm-gs-b0-matched-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
MODE="${1:-run}"
[[ $# -le 1 && ( "${MODE}" == run || "${MODE}" == plan ) ]] || {
  echo "usage: bash $0 plan | sbatch $0" >&2; exit 2;
}
for file in scripts/eval_nfcgs_b0_matched.py scripts/eval_nfcgs_path_diagnostics.py \
  scripts/profile_nfcgs_complexity.py globalsplat/compression/path_diagnostics.py \
  config/plans/nfcgs_path_diagnostics_20261019.json scripts/slurm/eval_nfcgs_path_diagnostics.slurm; do
  [[ -s "${file}" ]] || { echo "missing B0 dependency: ${file}" >&2; exit 1; }
done
OUTPUT="${B0_MATCHED_OUTPUT:-${REPO_DIR}/outputs/nfcgs_b0_matched/job_${SLURM_JOB_ID:-plan}}"
ARGS=(--native-checkpoint "${NATIVE_CHECKPOINT:-${REPO_DIR}/checkpoints/pretrained/globalsplat-re10k-32k.ckpt}"
  --manifest "${REPO_DIR}/config/plans/nfcgs_path_diagnostics_20261019.json"
  --input-run "${B0_MATCHED_INPUT_RUN:-${REPO_DIR}/outputs/nfcgs_legacy3_anchors/job_454995}"
  --expected-input-fingerprint 069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29
  --output "${OUTPUT}" --scenes 256 --tokens 4096 --seed 0 --device cuda --precision bf16
  --threads 8 --num-workers "${B0_MATCHED_NUM_WORKERS:-4}"
  --dataset-root "${DATASET_ROOT:-/data3/local_datasets/re10k}")
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=8 WANDB_MODE=disabled
if [[ "${MODE}" == plan ]]; then
  exec python scripts/eval_nfcgs_b0_matched.py "${ARGS[@]}" --plan
fi
: "${SLURM_JOB_ID:?submit with sbatch}"
NFCGS_TMP_ROOT="${NFCGS_TMP_ROOT:-/ceph_data/${USER:?missing USER}/.tmp/nfcgs}"
export TMPDIR="${NFCGS_TMP_ROOT}/b0_matched_${SLURM_JOB_ID}"
mkdir -p -- "${TMPDIR}"
export -n RANK 2>/dev/null || true
echo "B0_MATCHED_OUTPUT=${OUTPUT} TMPDIR=${TMPDIR}"
srun --ntasks=1 python scripts/eval_nfcgs_b0_matched.py "${ARGS[@]}"
echo "B0_MATCHED_REPORT=${OUTPUT}/REPORT.md"
