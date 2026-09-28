#!/usr/bin/env bash
#SBATCH -J gs-entropy-probe
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm/slurm-gs-entropy-probe-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-entropy-probe-%A_%a.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: submit from the GlobalSplat repository root" >&2; exit 1;
}
MODE="${1:-run}"
case "${MODE}" in
  plan) TASK_ID="${PROBE_TASK:-0}" ;;
  run) TASK_ID="${SLURM_ARRAY_TASK_ID:-${PROBE_TASK:-0}}" ;;
  *) echo "Usage: bash $0 plan | sbatch --array=0-1 $0" >&2; exit 1 ;;
esac
case "${TASK_ID}" in
  0) LAMBDA_TAG=0p0064 ;;
  1) LAMBDA_TAG=0p0256 ;;
  *) echo "ERROR: task 0=.0064, task 1=.0256" >&2; exit 1 ;;
esac
PARENT_ROOT="${PROBE_PARENT_ROOT:-${REPO_DIR}/outputs/nfcgs_score_probability10k}"
PARENT_RUN="${PROBE_PARENT_RUN:-20260913_112316}"
EXPERIMENT="nfcgs_score_probability10k_e1_split_rank56_lambda${LAMBDA_TAG}_residual_on_morton_on_m1c1s1_split"
CHECKPOINT="${PROBE_CHECKPOINT:-${PARENT_ROOT}/${PARENT_RUN}/e1_split/rank56/lambda${LAMBDA_TAG}/residual_on/checkpoints/${EXPERIMENT}/version_0/step000010000.ckpt}"
FIT_SCENES="${PROBE_FIT_SCENES:-128}"
EVAL_SCENES="${PROBE_EVAL_SCENES:-128}"
CACHE_SCENES="${PROBE_CACHE_SCENES:-0}"
[[ "${FIT_SCENES}" =~ ^[1-9][0-9]*$ && "${EVAL_SCENES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: fit/eval scene counts must be positive integers" >&2; exit 1;
}
[[ "${CACHE_SCENES}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: cache scene count must be a nonnegative integer" >&2; exit 1;
}
RUN_TAG="${PROBE_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${PROBE_OUTPUT_ROOT:-${REPO_DIR}/outputs/nfcgs_entropy_probe}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}/lambda${LAMBDA_TAG}/job_${SLURM_JOB_ID:-manual}_${TASK_ID}"
echo "ENTROPY_PROBE_TASK=${TASK_ID} LAMBDA=${LAMBDA_TAG} FIT_TRAIN=${FIT_SCENES} EVAL_TEST=${EVAL_SCENES}"
echo "CACHE_SCENES_PER_SPLIT=${CACHE_SCENES}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
[[ "${MODE}" != plan ]] || exit 0
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing Split checkpoint" >&2; exit 1; }
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=8
export WANDB_MODE=disabled
# Prevent a rank-width environment variable from being interpreted as DDP rank.
export -n RANK 2>/dev/null || true
srun --ntasks=1 python scripts/probe_nfcgs_entropy.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATASET_ROOT:-/data3/local_datasets/re10k}" \
  --output-dir "${OUTPUT_DIR}" \
  --fit-scenes "${FIT_SCENES}" --eval-scenes "${EVAL_SCENES}" \
  --prior-strength "${PROBE_PRIOR_STRENGTH:-4096}" \
  --num-workers "${PROBE_NUM_WORKERS:-4}" \
  --cache-scenes-per-split "${CACHE_SCENES}" \
  --threads 8 --seed 0 --device cuda --precision bf16 --expected-step 10000
