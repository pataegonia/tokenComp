#!/usr/bin/env bash
#SBATCH -J gs-nfcgs-grid-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 2-00:00:00
#SBATCH -o logs/slurm/slurm-gs-nfcgs-r56-train-%j.out
#SBATCH -e logs/slurm/slurm-gs-nfcgs-r56-train-%j.err

set -euo pipefail

# Submit from the GlobalSplat repository root (the log directory must exist
# before sbatch opens its stdout/stderr files):
#   mkdir -p logs/slurm
#   sbatch scripts/slurm/train_nfcgs_rank56.slurm
#
# Checkpoints default to checkpoints/{pretrained,codec_init} under REPO_DIR.
# Optional overrides:
#   VANILLA_CHECKPOINT=/path/to/globalsplat-re10k-32k.ckpt
#   CODEC_INIT=/path/to/re10k_ctx12_s64_t512_rank56.pt
#   CODEC_INIT_RANK56=/path/to/rank56_init.pt
#   CODEC_INIT_RANK80=/path/to/rank80_init.pt
#   RESUME_CHECKPOINT=/path/to/last.ckpt
#   USE_RESIDUAL=true|false
#   USE_MORTON=true|false
#
# Submit the full rank/lambda/residual grid as an eight-job array:
#   sbatch --array=0-7 scripts/slurm/train_nfcgs_rank56.slurm
#
# Array mapping:
#   0/1: rank 56, lambda 0.0064, residual on/off
#   2/3: rank 56, lambda 0.0256, residual on/off
#   4/5: rank 80, lambda 0.0064, residual on/off
#   6/7: rank 80, lambda 0.0256, residual on/off
#
# Without an array, RANK, RATE_LAMBDA, USE_RESIDUAL, and USE_MORTON can be set
# manually. To add only the informative Morton-off residual-on arms to an
# existing Morton-on grid:
#   sbatch --export=ALL,USE_MORTON=false --array=0,2,4,6 \
#     scripts/slurm/train_nfcgs_rank56.slurm
#
# The archived run used batch/GPU=8. A 48-GiB single GPU may not fit that
# micro-batch, so the defaults below use batch=1 and accumulation=8 while
# preserving an effective batch size of 8.

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-globalsplat}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
MVSPLAT_ROOT="${MVSPLAT_ROOT:-${REPO_DIR}/third_party/ZPressor/mvsplat}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_DIR}/checkpoints}"
VANILLA_CHECKPOINT="${VANILLA_CHECKPOINT:-${CHECKPOINT_ROOT}/pretrained/globalsplat-re10k-32k.ckpt}"
CODEC_INIT_RANK56="${CODEC_INIT_RANK56:-${CHECKPOINT_ROOT}/codec_init/re10k_ctx12_s64_t512_rank56.pt}"
CODEC_INIT_RANK80="${CODEC_INIT_RANK80:-${CHECKPOINT_ROOT}/codec_init/re10k_ctx12_s64_t512_rank80.pt}"
CODEC_INIT="${CODEC_INIT:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/nfcgs_rank56_train}"
MICRO_BATCH="${MICRO_BATCH:-1}"
ACCUMULATE="${ACCUMULATE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-50000}"
PRECISION="${PRECISION:-bf16-mixed}"
SEED="${SEED:-111123}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  case "${SLURM_ARRAY_TASK_ID}" in
    0) RANK=56; RATE_LAMBDA=0.0064; USE_RESIDUAL=true ;;
    1) RANK=56; RATE_LAMBDA=0.0064; USE_RESIDUAL=false ;;
    2) RANK=56; RATE_LAMBDA=0.0256; USE_RESIDUAL=true ;;
    3) RANK=56; RATE_LAMBDA=0.0256; USE_RESIDUAL=false ;;
    4) RANK=80; RATE_LAMBDA=0.0064; USE_RESIDUAL=true ;;
    5) RANK=80; RATE_LAMBDA=0.0064; USE_RESIDUAL=false ;;
    6) RANK=80; RATE_LAMBDA=0.0256; USE_RESIDUAL=true ;;
    7) RANK=80; RATE_LAMBDA=0.0256; USE_RESIDUAL=false ;;
    *) echo "ERROR: rank/lambda/residual grid expects array task id 0 through 7" >&2; exit 1 ;;
  esac
fi
RANK="${RANK:-56}"
RATE_LAMBDA="${RATE_LAMBDA:-0.0256}"
case "${USE_RESIDUAL:-true}" in
  true|1|yes) USE_RESIDUAL=true; RESIDUAL_TAG=on ;;
  false|0|no) USE_RESIDUAL=false; RESIDUAL_TAG=off ;;
  *) echo "ERROR: USE_RESIDUAL must be true or false" >&2; exit 1 ;;
esac
case "${USE_MORTON:-true}" in
  true|1|yes) USE_MORTON=true; MORTON_TAG=on ;;
  false|0|no) USE_MORTON=false; MORTON_TAG=off ;;
  *) echo "ERROR: USE_MORTON must be true or false" >&2; exit 1 ;;
esac
[[ "${RANK}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: RANK must be a positive integer" >&2
  exit 1
}
[[ "${RATE_LAMBDA}" =~ ^0\.[0-9]+$ ]] || {
  echo "ERROR: RATE_LAMBDA must be a positive decimal smaller than one" >&2
  exit 1
}
LAMBDA_TAG="${RATE_LAMBDA/./p}"
if [[ -z "${CODEC_INIT}" ]]; then
  if [[ "${RANK}" == "56" && -n "${CODEC_INIT_RANK56:-}" ]]; then
    CODEC_INIT="${CODEC_INIT_RANK56}"
  elif [[ "${RANK}" == "80" && -n "${CODEC_INIT_RANK80:-}" ]]; then
    CODEC_INIT="${CODEC_INIT_RANK80}"
  fi
fi

echo "GRID_COMBINATION=rank${RANK}/lambda${LAMBDA_TAG}/residual_${RESIDUAL_TAG}/morton_${MORTON_TAG}"
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi

[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: REPO_DIR is not the GlobalSplat repo root: ${REPO_DIR}" >&2
  exit 1
}
[[ -f "${DATASET_ROOT}/train/index.json" ]] || {
  echo "ERROR: missing RE10K train index: ${DATASET_ROOT}/train/index.json" >&2
  exit 1
}
[[ -d "${MVSPLAT_ROOT}/src/dataset" ]] || {
  echo "ERROR: missing vendored MVSplat dataset code: ${MVSPLAT_ROOT}" >&2
  exit 1
}
if [[ -z "${RESUME_CHECKPOINT}" ]]; then
  [[ -f "${VANILLA_CHECKPOINT}" ]] || {
    echo "ERROR: missing vanilla GlobalSplat-32K checkpoint: ${VANILLA_CHECKPOINT}" >&2
    exit 1
  }
fi
if [[ -n "${CODEC_INIT}" && ! -f "${CODEC_INIT}" ]]; then
  echo "ERROR: CODEC_INIT does not exist: ${CODEC_INIT}" >&2
  exit 1
fi
if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "ERROR: RESUME_CHECKPOINT does not exist: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_EXTENSIONS_DIR="${TMPDIR:-/tmp}/globalsplat_torch_ext_${USER:-job}"

RUN_ROOT="${OUTPUT_ROOT}/rank${RANK}/lambda${LAMBDA_TAG}/residual_${RESIDUAL_TAG}"
if [[ "${USE_MORTON}" == "false" ]]; then
  RUN_ROOT="${RUN_ROOT}/morton_off"
fi
INITIAL_CHECKPOINT="${RUN_ROOT}/initialization/nfcgs_rank${RANK}_from_vanilla.ckpt"
mkdir -p "${RUN_ROOT}" "${TORCH_EXTENSIONS_DIR}"
cd "${REPO_DIR}"

python -c "import compressai, hydra, pytorch_lightning, torch; assert torch.cuda.is_available(), 'CUDA PyTorch is required'; print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'gpu=', torch.cuda.get_device_name(0), 'compressai=', compressai.__version__)"

LOAD_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  LOAD_ARGS+=("checkpointing.load=${RESUME_CHECKPOINT}" "checkpointing.resume=true")
  echo "Resuming full state: ${RESUME_CHECKPOINT}"
else
  if [[ ! -f "${INITIAL_CHECKPOINT}" ]]; then
    INIT_ARGS=(--vanilla "${VANILLA_CHECKPOINT}" --output "${INITIAL_CHECKPOINT}" --seed "${SEED}" --rank "${RANK}")
    if [[ -n "${CODEC_INIT}" ]]; then
      INIT_ARGS+=(--codec-init "${CODEC_INIT}")
    fi
    srun --ntasks=1 python scripts/initialize_nfcgs_from_vanilla.py "${INIT_ARGS[@]}"
  fi
  LOAD_ARGS+=("checkpointing.load=${INITIAL_CHECKPOINT}" "checkpointing.resume=false")
  echo "Starting codec training from: ${INITIAL_CHECKPOINT}"
fi

echo "DATASET_ROOT=${DATASET_ROOT}"
echo "12 context / 8 target; 4096 tokens x 8 = 32768 Gaussians"
echo "MICRO_BATCH=${MICRO_BATCH} ACCUMULATE=${ACCUMULATE} MAX_STEPS=${MAX_STEPS}"
echo "RANK=${RANK} RATE_LAMBDA=${RATE_LAMBDA} USE_RESIDUAL=${USE_RESIDUAL} USE_MORTON=${USE_MORTON}"
echo "CODEC_INIT=${CODEC_INIT:-<constructor initialization>}"
echo "RUN_ROOT=${RUN_ROOT}"

srun --ntasks=1 python -m globalsplat.main \
  +experiment=re10k_32k_nfcgs_rank56 \
  "${LOAD_ARGS[@]}" \
  model.feature_codec.rank="${RANK}" \
  model.feature_codec.use_residual="${USE_RESIDUAL}" \
  model.feature_codec.use_morton="${USE_MORTON}" \
  loss.rate_lambda="${RATE_LAMBDA}" \
  experiment_name="nfcgs_rank${RANK}_lambda${LAMBDA_TAG}_residual_${RESIDUAL_TAG}_morton_${MORTON_TAG}" \
  "dataset.dataset_roots=[${DATASET_ROOT}]" \
  dataset.mvsplat_root="${MVSPLAT_ROOT}" \
  optimizer.batch_size="${MICRO_BATCH}" \
  optimizer.num_workers="${NUM_WORKERS}" \
  trainer.accumulate_grad_batches="${ACCUMULATE}" \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.precision="${PRECISION}" \
  output_dir="${RUN_ROOT}/checkpoints" \
  log_dir="${RUN_ROOT}/tensorboard" \
  seed="${SEED}" \
  hydra.run.dir="${RUN_ROOT}/hydra/${SLURM_JOB_ID:-manual}" \
  hydra.output_subdir=.hydra

echo "Training outputs: ${RUN_ROOT}"
