#!/usr/bin/env bash
#SBATCH -J gs-nfcgs-grid-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-nfcgs-grid-eval-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-nfcgs-grid-eval-%A_%a.err

set -euo pipefail

# Submit from the cloned GlobalSplat repository root:
#   mkdir -p logs/slurm
#   sbatch --array=0-7 scripts/slurm/eval_nfcgs_rank56.slurm
#
# The array mapping is identical to train_nfcgs_rank56.slurm:
#   0/1: rank 56, lambda 0.0064, residual on/off
#   2/3: rank 56, lambda 0.0256, residual on/off
#   4/5: rank 80, lambda 0.0064, residual on/off
#   6/7: rank 80, lambda 0.0256, residual on/off
#
# Each task automatically selects the highest numbered step*.ckpt under its
# own checkpoints/ directory (excluding nested Morton-off controls).
# Set EXPECTED_STEP to a number to require that exact
# global step. CHECKPOINT can still override discovery for a manual run.
# CHECKPOINT_ONLY=true prints the selected file without starting GPU evaluation.
# It must be an integrated checkpoint containing `model.feature_codec.*`, not
# the vanilla globalsplat-re10k-32k.ckpt.
# Default PROTOCOL=all evaluates every processable raw RE10K test scene with
# 12 disjoint context + 8 target views and no per-scene optimization. Set
# PROTOCOL=fixed for the standard 5,601-scene C3G index (3 target views).

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-globalsplat}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${REPO_DIR}/outputs/nfcgs_rank56_train}"
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/nfcgs_grid_eval}"
CONTEXT="${CONTEXT:-12}"
PROTOCOL="${PROTOCOL:-all}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-bf16-mixed}"
SAVE_IMAGES="${SAVE_IMAGES:-false}"
MAX_SCENES="${MAX_SCENES:-null}"
EXPECTED_STEP="${EXPECTED_STEP:-auto}"
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
case "${TRANSFORM}" in
  linear|nonlinear) ;;
  *) echo "ERROR: TRANSFORM must be linear or nonlinear" >&2; exit 1 ;;
esac
[[ "${TRANSFORM_HIDDEN}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: TRANSFORM_HIDDEN must be positive" >&2; exit 1; }
case "${SCORE_SPATIAL_PREDICTOR}" in linear|residual3|residual7) ;; *) echo "ERROR: SCORE_SPATIAL_PREDICTOR must be linear, residual3, or residual7" >&2; exit 1 ;; esac
if [[ "${SCORE_SPATIAL_PREDICTOR}" != linear && "${SCORE_SPATIAL_CONTEXT}" != true ]]; then
  echo "ERROR: residual SCORE_SPATIAL_PREDICTOR requires SCORE_SPATIAL_CONTEXT=true" >&2; exit 1
fi
case "${SCORE_SPATIAL_ENTROPY}" in shared|split|gaussian|conditional_scale) ;; *) echo "ERROR: SCORE_SPATIAL_ENTROPY must be shared, split, gaussian, or conditional_scale" >&2; exit 1 ;; esac
if [[ "${SCORE_SPATIAL_ENTROPY}" != shared && "${SCORE_SPATIAL_CONTEXT}" != true ]]; then
  echo "ERROR: non-shared SCORE_SPATIAL_ENTROPY requires SCORE_SPATIAL_CONTEXT=true" >&2; exit 1
fi
[[ "${SCORE_SPATIAL_HIDDEN}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: SCORE_SPATIAL_HIDDEN must be positive" >&2; exit 1; }
if [[ "${TRANSFORM}" == "nonlinear" && -z "${CHECKPOINT}" ]]; then
  echo "ERROR: nonlinear evaluation requires an explicit CHECKPOINT (use eval_nfcgs_transform.slurm)" >&2
  exit 1
fi
if [[ "${EXPECTED_STEP}" != "auto" && ! "${EXPECTED_STEP}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EXPECTED_STEP must be auto or a nonnegative integer" >&2
  exit 1
fi
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
TRAIN_RUN_ROOT="${TRAIN_OUTPUT_ROOT}/rank${RANK}/lambda${LAMBDA_TAG}/residual_${RESIDUAL_TAG}"
if [[ "${USE_MORTON}" == "false" ]]; then
  TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT}/morton_off"
fi
# Both the original and paper-view recipes use this directory, with different
# experiment-name prefixes below it. The sibling morton_off/ arm must never
# participate in Morton-on checkpoint discovery.
CHECKPOINT_BASE="${TRAIN_RUN_ROOT}/checkpoints"

echo "GRID_COMBINATION=rank${RANK}/lambda${LAMBDA_TAG}/residual_${RESIDUAL_TAG}/morton_${MORTON_TAG}"
echo "CHECKPOINT_BASE=${CHECKPOINT_BASE}"
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi

if [[ -z "${CHECKPOINT}" ]]; then
  [[ -d "${CHECKPOINT_BASE}" ]] || {
    echo "ERROR: checkpoint directory does not exist: ${CHECKPOINT_BASE}" >&2
    exit 1
  }
  mapfile -d '' -t CHECKPOINT_CANDIDATES < <(
    find "${CHECKPOINT_BASE}" -type f -name 'step*.ckpt' -print0
  )
  if (( ${#CHECKPOINT_CANDIDATES[@]} == 0 )); then
    echo "ERROR: no numbered training checkpoint under ${CHECKPOINT_BASE}" >&2
    exit 1
  fi
  BEST_STEP=-1
  BEST_CHECKPOINT=""
  for CANDIDATE in "${CHECKPOINT_CANDIDATES[@]}"; do
    CANDIDATE_NAME="$(basename "${CANDIDATE}")"
    [[ "${CANDIDATE_NAME}" =~ ([0-9]+)\.ckpt$ ]] || continue
    CANDIDATE_STEP=$((10#${BASH_REMATCH[1]}))
    if [[ "${EXPECTED_STEP}" != "auto" && "${CANDIDATE_STEP}" -ne "${EXPECTED_STEP}" ]]; then
      continue
    fi
    if (( CANDIDATE_STEP > BEST_STEP )) || \
       { (( CANDIDATE_STEP == BEST_STEP )) && [[ "${CANDIDATE}" > "${BEST_CHECKPOINT}" ]]; }; then
      BEST_STEP="${CANDIDATE_STEP}"
      BEST_CHECKPOINT="${CANDIDATE}"
    fi
  done
  if [[ -z "${BEST_CHECKPOINT}" ]]; then
    echo "ERROR: no step-${EXPECTED_STEP} checkpoint under ${CHECKPOINT_BASE}" >&2
    printf 'Available checkpoint: %s\n' "${CHECKPOINT_CANDIDATES[@]}" >&2
    exit 1
  fi
  CHECKPOINT="${BEST_CHECKPOINT}"
fi

[[ -f "${CHECKPOINT}" ]] || {
  echo "ERROR: missing integrated NFC-GS checkpoint: ${CHECKPOINT}" >&2
  exit 1
}
echo "CHECKPOINT=${CHECKPOINT}"
if [[ "${CHECKPOINT_ONLY:-false}" == "true" ]]; then
  exit 0
fi

case "${PROTOCOL}" in
  all)
    [[ "${CONTEXT}" == "12" ]] || {
      echo "ERROR: PROTOCOL=all currently uses the recovered 12-context/8-target setup" >&2
      exit 2
    }
    DATASET_CONFIG="re10k_eval_all_ctx12"
    PROTOCOL_TAG="all_test"
    ;;
  fixed)
    case "${CONTEXT}" in
      12|24|36) DATASET_CONFIG="re10k_eval_ctx${CONTEXT}" ;;
      *) echo "ERROR: fixed CONTEXT must be 12, 24, or 36; got ${CONTEXT}" >&2; exit 2 ;;
    esac
    PROTOCOL_TAG="fixed_c3g"
    ;;
  *) echo "ERROR: PROTOCOL must be all or fixed; got ${PROTOCOL}" >&2; exit 2 ;;
esac

[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: REPO_DIR is not the GlobalSplat repo root: ${REPO_DIR}" >&2
  exit 1
}
[[ -f "${DATASET_ROOT}/test/index.json" ]] || {
  echo "ERROR: missing RE10K test index: ${DATASET_ROOT}/test/index.json" >&2
  exit 1
}
[[ -f "${CHECKPOINT}" ]] || {
  echo "ERROR: missing integrated NFC-GS checkpoint: ${CHECKPOINT}" >&2
  exit 1
}
if [[ "${PROTOCOL}" == "fixed" ]]; then
  [[ -f "${REPO_DIR}/assets/eval_index/re10k/c3g_re10k_ctx_${CONTEXT}v_trg_3v.json" ]] || {
    echo "ERROR: missing fixed ${CONTEXT}-view evaluation index" >&2
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

RUN_OUTPUT="${OUTPUT_ROOT}/rank${RANK}/lambda${LAMBDA_TAG}/residual_${RESIDUAL_TAG}/morton_${MORTON_TAG}/${PROTOCOL_TAG}_ctx${CONTEXT}/job_${SLURM_JOB_ID:-manual}"
mkdir -p "${RUN_OUTPUT}"
cd "${REPO_DIR}"

python -c "import compressai, gsplat, hydra, pytorch_lightning, torch; assert torch.cuda.is_available(), 'CUDA PyTorch is required'; print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'gpu=', torch.cuda.get_device_name(0), 'compressai=', compressai.__version__)"
python scripts/check_nfcgs_checkpoint.py "${CHECKPOINT}" \
  --rank "${RANK}" --step "${EXPECTED_STEP}" \
  --transform "${TRANSFORM}" --transform-hidden "${TRANSFORM_HIDDEN}" \
  --residual "${USE_RESIDUAL}" --morton "${USE_MORTON}" \
  --score-mean-condition "${SCORE_MEAN_CONDITION}" \
  --score-channel-context "${SCORE_CHANNEL_CONTEXT}" \
  --score-spatial-context "${SCORE_SPATIAL_CONTEXT}" \
  --score-spatial-predictor "${SCORE_SPATIAL_PREDICTOR}" \
  --score-spatial-entropy "${SCORE_SPATIAL_ENTROPY}" \
  --score-spatial-hidden "${SCORE_SPATIAL_HIDDEN}" \
  --score-slice-channels "${SCORE_SLICE_CHANNELS}" \
  --score-context-hidden "${SCORE_CONTEXT_HIDDEN}"

echo "REPO_DIR=${REPO_DIR}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "RANK=${RANK} RATE_LAMBDA=${RATE_LAMBDA} USE_RESIDUAL=${USE_RESIDUAL} USE_MORTON=${USE_MORTON}"
echo "PROTOCOL=${PROTOCOL} CONTEXT=${CONTEXT} SCENES=all PRECISION=${PRECISION}"
if [[ "${PROTOCOL}" == "all" ]]; then
  python -c "import json,sys; print('raw test index scenes=', len(json.load(open(sys.argv[1]))))" "${DATASET_ROOT}/test/index.json"
else
  echo "fixed C3G scenes=5601 target_views=3"
fi
echo "RUN_OUTPUT=${RUN_OUTPUT}"

srun --ntasks=1 python -m globalsplat.main \
  model=globalsplat_nfcgs_rank56 \
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
  +experiment=re10k_32k \
  dataset="${DATASET_CONFIG}" \
  mode=test \
  checkpointing.load="${CHECKPOINT}" \
  "dataset.dataset_roots=[${DATASET_ROOT}]" \
  dataset.augment=false \
  optimizer.batch_size=1 \
  optimizer.num_workers="${NUM_WORKERS}" \
  trainer.devices=1 \
  trainer.num_nodes=1 \
  trainer.strategy=auto \
  trainer.precision="${PRECISION}" \
  test.max_scenes="${MAX_SCENES}" \
  test.actual_bitstream=true \
  test.compute_scores=true \
  test.save_image="${SAVE_IMAGES}" \
  test.save_gt_image=false \
  test.save_input_images=false \
  test.save_video=false \
  test.output_path="${RUN_OUTPUT}" \
  experiment_name="nfcgs_rank${RANK}_lambda${LAMBDA_TAG}_residual_${RESIDUAL_TAG}_morton_${MORTON_TAG}_${PROTOCOL_TAG}_ctx${CONTEXT}" \
  hydra.run.dir="${RUN_OUTPUT}/hydra" \
  hydra.output_subdir=.hydra

echo "Results: ${RUN_OUTPUT}/nfcgs_rank${RANK}_lambda${LAMBDA_TAG}_residual_${RESIDUAL_TAG}_morton_${MORTON_TAG}_${PROTOCOL_TAG}_ctx${CONTEXT}"
