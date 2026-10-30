#!/usr/bin/env bash
#SBATCH -J gs-l3-anchors
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -p batch_ugrad
#SBATCH -t 08:00:00
#SBATCH -o logs/slurm/slurm-gs-l3-anchors-%j.out
#SBATCH -e logs/slurm/slurm-gs-l3-anchors-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
MODE="${1:-run}"
[[ "${MODE}" == run || "${MODE}" == plan ]] || { echo "usage: bash $0 plan | sbatch $0" >&2; exit 2; }
MANIFEST="${ANCHOR_MANIFEST:-${REPO_DIR}/logs/slurm/nfcgs_legacy3_anchors_manifest_2026-10-16.json}"
[[ -s "${MANIFEST}" ]] || { echo "missing manifest: ${MANIFEST}" >&2; exit 1; }
for file in globalsplat/compression/mechanism.py scripts/profile_nfcgs_complexity.py scripts/analyze_nfcgs_mechanism.py scripts/report_nfcgs_mechanism.py; do
  [[ -s "${file}" ]] || { echo "missing analysis file: ${file}" >&2; exit 1; }
done
grep -Fq 'self.stage_observer' globalsplat/compression/score_context.py || {
  echo "Upload the updated globalsplat/compression/score_context.py first." >&2; exit 1;
}
OUTPUT="${ANCHOR_OUTPUT:-${REPO_DIR}/outputs/nfcgs_legacy3_anchors/job_${SLURM_JOB_ID:-plan}}"
ARGS=(--scope legacy3-anchors --manifest "${MANIFEST}" --output "${OUTPUT}" --device cuda --precision bf16
  --scenes "${ANCHOR_SCENES:-256}" --trace-scenes "${ANCHOR_TRACE_SCENES:-8}"
  --tokens 4096 --seed 0 --threads 8 --num-workers "${ANCHOR_NUM_WORKERS:-4}"
  --dataset-root "${DATASET_ROOT:-/data3/local_datasets/re10k}")
[[ "${ANCHOR_RESUME:-0}" != 1 ]] || ARGS+=(--resume)
if [[ "${MODE}" == plan ]]; then
  exec python scripts/analyze_nfcgs_mechanism.py "${ARGS[@]}" --plan
fi
: "${SLURM_JOB_ID:?submit with sbatch}"
NFCGS_TMP_ROOT="${NFCGS_TMP_ROOT:-/ceph_data/${USER:?missing USER}/.tmp/nfcgs}"
export TMPDIR="${NFCGS_TMP_ROOT}/l3_anchors_${SLURM_JOB_ID}"
mkdir -p -- "${TMPDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 WANDB_MODE=disabled
export -n RANK 2>/dev/null || true
echo "ANCHOR_TMPDIR=${TMPDIR} OUTPUT=${OUTPUT}"
srun --ntasks=1 python scripts/analyze_nfcgs_mechanism.py "${ARGS[@]}"
REPORT="${OUTPUT}/report_${SLURM_JOB_ID}"
python scripts/report_nfcgs_mechanism.py --input "${OUTPUT}" --output "${REPORT}"
echo "ANCHOR_REPORT=${REPORT}"
