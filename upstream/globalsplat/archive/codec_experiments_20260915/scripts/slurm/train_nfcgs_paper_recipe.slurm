#!/usr/bin/env bash
#SBATCH -J gs-nfcgs-paper
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 3-00:00:00
#SBATCH -w ariel-v12
#SBATCH -o logs/slurm/slurm-gs-nfcgs-paper-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-nfcgs-paper-%A_%a.err

set -euo pipefail

# GlobalSplat paper-view rerun of the NFC-GS grid:
#   context pool 24 -> two 13-view forwards, 12 shared targets
#   alpha/depth subset consistency = 1e-3 / 1e-2
#   archived NFC-GS codec entropy family and exact PCA/statistics initialization
#
# Submit the rank/lambda/residual grid:
#   mkdir -p logs/slurm
#   sbatch --array=0-7 scripts/slurm/train_nfcgs_paper_recipe.slurm
#
# Add the four residual-on Morton-off controls:
#   sbatch --export=ALL,USE_MORTON=false --array=0,2,4,6 \
#     scripts/slurm/train_nfcgs_paper_recipe.slurm
#
# Array mapping:
#   0/1: rank 56, lambda 0.0064, residual on/off
#   2/3: rank 56, lambda 0.0256, residual on/off
#   4/5: rank 80, lambda 0.0064, residual on/off
#   6/7: rank 80, lambda 0.0256, residual on/off

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
TRANSFORM="${TRANSFORM:-linear}"
TRANSFORM_HIDDEN="${TRANSFORM_HIDDEN:-32}"
SCORE_MEAN_CONDITION="${SCORE_MEAN_CONDITION:-false}"
SCORE_CHANNEL_CONTEXT="${SCORE_CHANNEL_CONTEXT:-false}"
SCORE_SPATIAL_CONTEXT="${SCORE_SPATIAL_CONTEXT:-false}"
SCORE_SPATIAL_PREDICTOR="${SCORE_SPATIAL_PREDICTOR:-linear}"
SCORE_SPATIAL_ENTROPY="${SCORE_SPATIAL_ENTROPY:-shared}"
SCORE_SPATIAL_HIDDEN="${SCORE_SPATIAL_HIDDEN:-32}"
SCORE_SLICE_CHANNELS="${SCORE_SLICE_CHANNELS:-16}"
SCORE_CONTEXT_HIDDEN="${SCORE_CONTEXT_HIDDEN:-64}"
FEATURE_CODEC_TRAIN_SCOPE="${FEATURE_CODEC_TRAIN_SCOPE:-all}"
WARM_START_CHECKPOINT="${WARM_START_CHECKPOINT:-}"
CONTEXT_VIEWS="${CONTEXT_VIEWS:-24}"
SUBSET_CONSISTENCY="${SUBSET_CONSISTENCY:-true}"
OPTIMIZER_LR="${OPTIMIZER_LR:-}"
OPTIMIZER_LR_MILESTONES="${OPTIMIZER_LR_MILESTONES:-}"
OPTIMIZER_LR_GAMMA="${OPTIMIZER_LR_GAMMA:-}"
QUANTILE_UPDATE_INTERVAL="${QUANTILE_UPDATE_INTERVAL:-}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-}"
case "${TRANSFORM}" in
  linear) DEFAULT_ROOT="${REPO_DIR}/outputs/nfcgs_paper24_subset_train"; DEFAULT_PREFIX=nfcgs_paper24_subset ;;
  nonlinear) DEFAULT_ROOT="${REPO_DIR}/outputs/nfcgs_paper24_transform_train/nonlinear${TRANSFORM_HIDDEN}"; DEFAULT_PREFIX="nfcgs_transform_nonlinear${TRANSFORM_HIDDEN}" ;;
  *) echo "ERROR: TRANSFORM must be linear or nonlinear" >&2; exit 1 ;;
esac
[[ "${TRANSFORM_HIDDEN}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: TRANSFORM_HIDDEN must be positive" >&2; exit 1; }
for VALUE in SCORE_MEAN_CONDITION SCORE_CHANNEL_CONTEXT SCORE_SPATIAL_CONTEXT; do
  case "${!VALUE}" in true|false) ;; *) echo "ERROR: ${VALUE} must be true or false" >&2; exit 1 ;; esac
done
case "${FEATURE_CODEC_TRAIN_SCOPE}" in all|score_context|score_probability) ;; *) echo "ERROR: FEATURE_CODEC_TRAIN_SCOPE must be all, score_context, or score_probability" >&2; exit 1 ;; esac
case "${SCORE_SPATIAL_PREDICTOR}" in linear|residual3|residual7) ;; *) echo "ERROR: SCORE_SPATIAL_PREDICTOR must be linear, residual3, or residual7" >&2; exit 1 ;; esac
if [[ "${SCORE_SPATIAL_PREDICTOR}" != linear && "${SCORE_SPATIAL_CONTEXT}" != true ]]; then
  echo "ERROR: residual SCORE_SPATIAL_PREDICTOR requires SCORE_SPATIAL_CONTEXT=true" >&2; exit 1
fi
case "${SCORE_SPATIAL_ENTROPY}" in shared|split|gaussian|conditional_scale) ;; *) echo "ERROR: SCORE_SPATIAL_ENTROPY must be shared, split, gaussian, or conditional_scale" >&2; exit 1 ;; esac
if [[ "${SCORE_SPATIAL_ENTROPY}" != shared && "${SCORE_SPATIAL_CONTEXT}" != true ]]; then
  echo "ERROR: non-shared SCORE_SPATIAL_ENTROPY requires SCORE_SPATIAL_CONTEXT=true" >&2; exit 1
fi
[[ "${SCORE_SPATIAL_HIDDEN}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: SCORE_SPATIAL_HIDDEN must be positive" >&2; exit 1; }
[[ "${SCORE_SLICE_CHANNELS}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: SCORE_SLICE_CHANNELS must be positive" >&2; exit 1; }
[[ "${SCORE_CONTEXT_HIDDEN}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: SCORE_CONTEXT_HIDDEN must be positive" >&2; exit 1; }
[[ "${CONTEXT_VIEWS}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: CONTEXT_VIEWS must be positive" >&2; exit 1; }
case "${SUBSET_CONSISTENCY}" in true|false) ;; *) echo "ERROR: SUBSET_CONSISTENCY must be true or false" >&2; exit 1 ;; esac
SCORE_CONTEXT_TAG=factorized
SCORE_CONTEXT_SUFFIX=""
if [[ "${SCORE_MEAN_CONDITION}" == true || "${SCORE_CHANNEL_CONTEXT}" == true || "${SCORE_SPATIAL_CONTEXT}" == true ]]; then
  [[ "${SCORE_MEAN_CONDITION}" == true ]] && SCORE_M=1 || SCORE_M=0
  [[ "${SCORE_CHANNEL_CONTEXT}" == true ]] && SCORE_C=1 || SCORE_C=0
  [[ "${SCORE_SPATIAL_CONTEXT}" == true ]] && SCORE_S=1 || SCORE_S=0
  SCORE_CONTEXT_TAG="m${SCORE_M}c${SCORE_C}s${SCORE_S}"
  if [[ "${SCORE_SPATIAL_PREDICTOR}" != linear ]]; then
    SCORE_CONTEXT_TAG="${SCORE_CONTEXT_TAG}_${SCORE_SPATIAL_PREDICTOR}h${SCORE_SPATIAL_HIDDEN}"
  fi
  if [[ "${SCORE_SPATIAL_ENTROPY}" != shared ]]; then
    SCORE_CONTEXT_TAG="${SCORE_CONTEXT_TAG}_${SCORE_SPATIAL_ENTROPY}"
  fi
  SCORE_CONTEXT_SUFFIX="_${SCORE_CONTEXT_TAG}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_ROOT}}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-${DEFAULT_PREFIX}}"
# Each sample becomes a batched pair of 13-view branches.  Batch 1 leaves the
# RTX A5000 under-filled (about 6 GiB allocated and visible launch gaps), so use
# two samples per forward by default.  Accumulation keeps the archived effective
# batch size of 8 and can still be overridden at submission time if needed.
MICRO_BATCH="${MICRO_BATCH:-2}"
ACCUMULATE="${ACCUMULATE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-50000}"
PRECISION="${PRECISION:-bf16-mixed}"
SEED="${SEED:-111123}"
# Cosmetic GPU load: wait until training has warmed up, fill VRAM while leaving
# 1024 MiB free, and submit short FP16 GEMM bursts once per second.  Disable it
# with DUMMY_GPU_LOAD=false or tune each value through sbatch --export.
DUMMY_GPU_LOAD="${DUMMY_GPU_LOAD:-true}"
DUMMY_GPU_RESERVE_MIB="${DUMMY_GPU_RESERVE_MIB:-1024}"
DUMMY_GPU_WAIT_USED_MIB="${DUMMY_GPU_WAIT_USED_MIB:-4096}"
DUMMY_GPU_SETTLE_SECONDS="${DUMMY_GPU_SETTLE_SECONDS:-45}"
DUMMY_GPU_CHUNK_MIB="${DUMMY_GPU_CHUNK_MIB:-256}"
DUMMY_GPU_MATMUL_SIZE="${DUMMY_GPU_MATMUL_SIZE:-4096}"
DUMMY_GPU_BURST_OPS="${DUMMY_GPU_BURST_OPS:-16}"
DUMMY_GPU_PERIOD_SECONDS="${DUMMY_GPU_PERIOD_SECONDS:-1.0}"

GRID_TASK_ID="${GRID_TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -n "${GRID_TASK_ID}" ]]; then
  case "${GRID_TASK_ID}" in
    0) RANK=56; RATE_LAMBDA=0.0064; USE_RESIDUAL=true ;;
    1) RANK=56; RATE_LAMBDA=0.0064; USE_RESIDUAL=false ;;
    2) RANK=56; RATE_LAMBDA=0.0256; USE_RESIDUAL=true ;;
    3) RANK=56; RATE_LAMBDA=0.0256; USE_RESIDUAL=false ;;
    4) RANK=80; RATE_LAMBDA=0.0064; USE_RESIDUAL=true ;;
    5) RANK=80; RATE_LAMBDA=0.0064; USE_RESIDUAL=false ;;
    6) RANK=80; RATE_LAMBDA=0.0256; USE_RESIDUAL=true ;;
    7) RANK=80; RATE_LAMBDA=0.0256; USE_RESIDUAL=false ;;
    *) echo "ERROR: grid expects array task id 0 through 7" >&2; exit 1 ;;
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
  if [[ "${RANK}" == "56" ]]; then
    CODEC_INIT="${CODEC_INIT_RANK56}"
  elif [[ "${RANK}" == "80" ]]; then
    CODEC_INIT="${CODEC_INIT_RANK80}"
  else
    echo "ERROR: no default codec artifact for rank ${RANK}" >&2
    exit 1
  fi
fi

RUN_ROOT="${OUTPUT_ROOT}/rank${RANK}/lambda${LAMBDA_TAG}/residual_${RESIDUAL_TAG}"
if [[ "${USE_MORTON}" == "false" ]]; then
  RUN_ROOT="${RUN_ROOT}/morton_off"
fi
EXPERIMENT="${EXPERIMENT_PREFIX}_rank${RANK}_lambda${LAMBDA_TAG}_residual_${RESIDUAL_TAG}_morton_${MORTON_TAG}${SCORE_CONTEXT_SUFFIX}"
if [[ -n "${SCORE_CONTEXT_SUFFIX}" ]]; then
  INITIAL_CHECKPOINT="${RUN_ROOT}/initialization/nfcgs_rank${RANK}_${SCORE_CONTEXT_TAG}.ckpt"
else
  INITIAL_CHECKPOINT="${RUN_ROOT}/initialization/nfcgs_rank${RANK}_exact_artifact.ckpt"
fi

echo "GRID_COMBINATION=rank${RANK}/lambda${LAMBDA_TAG}/residual_${RESIDUAL_TAG}/morton_${MORTON_TAG}"
echo "TRANSFORM=${TRANSFORM} HIDDEN=${TRANSFORM_HIDDEN} EXPERIMENT=${EXPERIMENT}"
echo "SCORE_CONTEXT=mean:${SCORE_MEAN_CONDITION},channel:${SCORE_CHANNEL_CONTEXT},spatial:${SCORE_SPATIAL_CONTEXT},predictor:${SCORE_SPATIAL_PREDICTOR},spatial_entropy:${SCORE_SPATIAL_ENTROPY},spatial_hidden:${SCORE_SPATIAL_HIDDEN},slice:${SCORE_SLICE_CHANNELS},hidden:${SCORE_CONTEXT_HIDDEN} scope=${FEATURE_CODEC_TRAIN_SCOPE}"
if [[ "${SUBSET_CONSISTENCY}" == true ]]; then
  echo "RECIPE=paper context_pool=${CONTEXT_VIEWS} branch_views=13+13 shared_targets=12 subset_consistency=on"
else
  echo "RECIPE=screen context_views=${CONTEXT_VIEWS} single_branch shared_targets=12 subset_consistency=off"
fi
echo "ENTROPY=score:${SCORE_CONTEXT_TAG} residual_y:conditional_gaussian residual_z:factorized"
echo "INITIALIZATION=geometry_basis+pca_analysis+pca_synthesis+score_scale+residual_mean_std"
echo "BATCH=micro${MICRO_BATCH} accumulate${ACCUMULATE} effective$((MICRO_BATCH * ACCUMULATE)) workers${NUM_WORKERS}"
echo "OPTIMIZER_OVERRIDES=lr:${OPTIMIZER_LR:-default},milestones:${OPTIMIZER_LR_MILESTONES:-default},gamma:${OPTIMIZER_LR_GAMMA:-default},quantile_interval:${QUANTILE_UPDATE_INTERVAL:-default},val_epoch_interval:${CHECK_VAL_EVERY_N_EPOCH:-default}"
echo "DUMMY_GPU_LOAD=${DUMMY_GPU_LOAD} reserve=${DUMMY_GPU_RESERVE_MIB}MiB wait_used=${DUMMY_GPU_WAIT_USED_MIB}MiB settle=${DUMMY_GPU_SETTLE_SECONDS}s burst_ops=${DUMMY_GPU_BURST_OPS} period=${DUMMY_GPU_PERIOD_SECONDS}s"
echo "RUN_ROOT=${RUN_ROOT}"
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
    echo "ERROR: missing vanilla checkpoint: ${VANILLA_CHECKPOINT}" >&2
    exit 1
  }
  [[ -f "${CODEC_INIT}" ]] || {
    echo "ERROR: missing rank-${RANK} codec initialization artifact: ${CODEC_INIT}" >&2
    exit 1
  }
  if [[ -n "${WARM_START_CHECKPOINT}" && ! -f "${WARM_START_CHECKPOINT}" ]]; then
    echo "ERROR: missing warm-start checkpoint: ${WARM_START_CHECKPOINT}" >&2
    exit 1
  fi
else
  [[ -f "${RESUME_CHECKPOINT}" ]] || {
    echo "ERROR: missing resume checkpoint: ${RESUME_CHECKPOINT}" >&2
    exit 1
  }
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_EXTENSIONS_DIR="${TMPDIR:-/tmp}/globalsplat_torch_ext_${USER:-job}"

mkdir -p "${RUN_ROOT}" "${TORCH_EXTENSIONS_DIR}"
cd "${REPO_DIR}"

python -c "import compressai, hydra, pytorch_lightning, torch; assert torch.cuda.is_available(), 'CUDA PyTorch is required'; print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'gpu=', torch.cuda.get_device_name(0), 'compressai=', compressai.__version__)"

LOAD_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  LOAD_ARGS+=("checkpointing.load=${RESUME_CHECKPOINT}" "checkpointing.resume=true")
  echo "Resuming full state: ${RESUME_CHECKPOINT}"
else
  if [[ ! -f "${INITIAL_CHECKPOINT}" ]]; then
    WARM_START_ARGS=()
    if [[ -n "${WARM_START_CHECKPOINT}" ]]; then
      WARM_START_ARGS+=(--warm-start "${WARM_START_CHECKPOINT}")
    fi
    srun --ntasks=1 python scripts/initialize_nfcgs_from_vanilla.py \
      --vanilla "${VANILLA_CHECKPOINT}" \
      --codec-init "${CODEC_INIT}" \
      --output "${INITIAL_CHECKPOINT}" \
      --seed "${SEED}" \
      --rank "${RANK}" \
      --transform "${TRANSFORM}" --transform-hidden "${TRANSFORM_HIDDEN}" \
      --use-residual "${USE_RESIDUAL}" --use-morton "${USE_MORTON}" \
      --score-mean-condition "${SCORE_MEAN_CONDITION}" \
      --score-channel-context "${SCORE_CHANNEL_CONTEXT}" \
      --score-spatial-context "${SCORE_SPATIAL_CONTEXT}" \
      --score-spatial-predictor "${SCORE_SPATIAL_PREDICTOR}" \
      --score-spatial-entropy "${SCORE_SPATIAL_ENTROPY}" \
      --score-spatial-hidden "${SCORE_SPATIAL_HIDDEN}" \
      --score-slice-channels "${SCORE_SLICE_CHANNELS}" \
      --score-context-hidden "${SCORE_CONTEXT_HIDDEN}" \
      "${WARM_START_ARGS[@]}"
  fi
  LOAD_ARGS+=("checkpointing.load=${INITIAL_CHECKPOINT}" "checkpointing.resume=false")
  echo "Starting codec training from: ${INITIAL_CHECKPOINT}"
fi

OPTIMIZER_ARGS=()
[[ -n "${OPTIMIZER_LR}" ]] && OPTIMIZER_ARGS+=("optimizer.lr=${OPTIMIZER_LR}")
[[ -n "${OPTIMIZER_LR_MILESTONES}" ]] && OPTIMIZER_ARGS+=("optimizer.lr_milestones=${OPTIMIZER_LR_MILESTONES}")
[[ -n "${OPTIMIZER_LR_GAMMA}" ]] && OPTIMIZER_ARGS+=("optimizer.lr_gamma=${OPTIMIZER_LR_GAMMA}")
[[ -n "${QUANTILE_UPDATE_INTERVAL}" ]] && OPTIMIZER_ARGS+=("loss.quantile_update_interval=${QUANTILE_UPDATE_INTERVAL}")
[[ -n "${CHECK_VAL_EVERY_N_EPOCH}" ]] && OPTIMIZER_ARGS+=("trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}")

DUMMY_PID=""
cleanup_dummy_gpu_load() {
  if [[ -n "${DUMMY_PID}" ]] && kill -0 "${DUMMY_PID}" 2>/dev/null; then
    kill "${DUMMY_PID}" 2>/dev/null || true
    wait "${DUMMY_PID}" 2>/dev/null || true
  fi
}
trap cleanup_dummy_gpu_load EXIT INT TERM

case "${DUMMY_GPU_LOAD}" in
  true|1|yes)
    python -u scripts/gpu_dummy_load.py \
      --reserve-mib "${DUMMY_GPU_RESERVE_MIB}" \
      --wait-used-mib "${DUMMY_GPU_WAIT_USED_MIB}" \
      --settle-seconds "${DUMMY_GPU_SETTLE_SECONDS}" \
      --chunk-mib "${DUMMY_GPU_CHUNK_MIB}" \
      --matmul-size "${DUMMY_GPU_MATMUL_SIZE}" \
      --burst-ops "${DUMMY_GPU_BURST_OPS}" \
      --period-seconds "${DUMMY_GPU_PERIOD_SECONDS}" &
    DUMMY_PID=$!
    echo "Dummy GPU load helper started (pid=${DUMMY_PID})"
    ;;
  false|0|no)
    echo "Dummy GPU load helper disabled"
    ;;
  *)
    echo "ERROR: DUMMY_GPU_LOAD must be true or false" >&2
    exit 1
    ;;
esac

TRAIN_STATUS=0
srun --ntasks=1 python -m globalsplat.main \
  +experiment=re10k_32k_nfcgs_paper \
  "${LOAD_ARGS[@]}" \
  model.feature_codec.rank="${RANK}" \
  model.feature_codec.use_residual="${USE_RESIDUAL}" \
  model.feature_codec.use_morton="${USE_MORTON}" \
  model.feature_codec.transform="${TRANSFORM}" \
  model.feature_codec.transform_hidden="${TRANSFORM_HIDDEN}" \
  model.feature_codec.score_mean_condition="${SCORE_MEAN_CONDITION}" \
  model.feature_codec.score_channel_context="${SCORE_CHANNEL_CONTEXT}" \
  model.feature_codec.score_spatial_context="${SCORE_SPATIAL_CONTEXT}" \
  model.feature_codec.score_spatial_predictor="${SCORE_SPATIAL_PREDICTOR}" \
  model.feature_codec.score_spatial_entropy="${SCORE_SPATIAL_ENTROPY}" \
  model.feature_codec.score_spatial_hidden="${SCORE_SPATIAL_HIDDEN}" \
  model.feature_codec.score_slice_channels="${SCORE_SLICE_CHANNELS}" \
  model.feature_codec.score_context_hidden="${SCORE_CONTEXT_HIDDEN}" \
  model.feature_codec_train_scope="${FEATURE_CODEC_TRAIN_SCOPE}" \
  loss.rate_lambda="${RATE_LAMBDA}" \
  loss.subset_consistency="${SUBSET_CONSISTENCY}" \
  experiment_name="${EXPERIMENT}" \
  "dataset.dataset_roots=[${DATASET_ROOT}]" \
  dataset.mvsplat_root="${MVSPLAT_ROOT}" \
  dataset.num_context_views="${CONTEXT_VIEWS}" \
  optimizer.batch_size="${MICRO_BATCH}" \
  optimizer.num_workers="${NUM_WORKERS}" \
  trainer.accumulate_grad_batches="${ACCUMULATE}" \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.precision="${PRECISION}" \
  "${OPTIMIZER_ARGS[@]}" \
  output_dir="${RUN_ROOT}/checkpoints" \
  log_dir="${RUN_ROOT}/tensorboard" \
  seed="${SEED}" \
  hydra.run.dir="${RUN_ROOT}/hydra/${SLURM_JOB_ID:-manual}" \
  hydra.output_subdir=.hydra || TRAIN_STATUS=$?

cleanup_dummy_gpu_load
trap - EXIT INT TERM

if (( TRAIN_STATUS != 0 )); then
  echo "ERROR: training exited with status ${TRAIN_STATUS}" >&2
  exit "${TRAIN_STATUS}"
fi

echo "Training outputs: ${RUN_ROOT}"
