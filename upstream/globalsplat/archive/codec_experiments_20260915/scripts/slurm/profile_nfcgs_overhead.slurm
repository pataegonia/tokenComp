#!/usr/bin/env bash
#SBATCH -J gs-overhead
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 03:00:00
#SBATCH -o logs/slurm/slurm-gs-overhead-%j.out
#SBATCH -e logs/slurm/slurm-gs-overhead-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: submit from the GlobalSplat repository root" >&2; exit 1;
}
MODE="${1:-run}"
[[ "${MODE}" == run || "${MODE}" == plan ]] || {
  echo "Usage: bash $0 plan | sbatch $0" >&2; exit 1;
}
PROFILE_SUITE="${PROFILE_SUITE:-all}"
[[ "${PROFILE_SUITE}" == all || "${PROFILE_SUITE}" == context || "${PROFILE_SUITE}" == probability ]] || {
  echo "ERROR: PROFILE_SUITE must be all, context, or probability" >&2; exit 1;
}
WARMUP_SCENES="${PROFILE_WARMUP_SCENES:-4}"
MEASURE_SCENES="${PROFILE_MEASURE_SCENES:-32}"
[[ "${WARMUP_SCENES}" =~ ^[0-9]+$ && "${MEASURE_SCENES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: warmup must be nonnegative and measure scenes must be positive" >&2; exit 1;
}
RUN_TAG="${PROFILE_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${PROFILE_OUTPUT_ROOT:-${REPO_DIR}/outputs/nfcgs_overhead_profile}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
CONTEXT_ROOT="${PROFILE_CONTEXT_ROOT:-${REPO_DIR}/outputs/nfcgs_score_context/20260908_092102}"
FACTOR_ROOT="${PROFILE_FACTOR_ROOT:-${REPO_DIR}/outputs/nfcgs_factorized_continuation/20260908_184640}"
PROBABILITY_ROOT="${PROFILE_PROBABILITY_ROOT:-${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316}"

profile_one() {
  local suite="$1" label="$2" lambda_tag="$3" expected_step="$4" checkpoint="$5"
  local destination="${OUTPUT_DIR}/${suite}/lambda${lambda_tag}/${label}"
  echo "PROFILE suite=${suite} lambda=${lambda_tag} label=${label} checkpoint=${checkpoint}"
  [[ "${MODE}" == run ]] || return 0
  [[ -f "${checkpoint}" ]] || { echo "ERROR: missing checkpoint ${checkpoint}" >&2; exit 1; }
  if [[ -f "${destination}/summary.json" ]]; then
    echo "SKIP_COMPLETE ${destination}/summary.json"
    return 0
  fi
  if [[ -d "${destination}" && -n "$(find "${destination}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: preserving incomplete nonempty destination ${destination}" >&2
    exit 1
  fi
  srun --ntasks=1 python scripts/profile_nfcgs_overhead.py \
    --checkpoint "${checkpoint}" --expected-step "${expected_step}" \
    --suite "${suite}" --label "${label}" --lambda-tag "${lambda_tag}" \
    --dataset-root "${DATASET_ROOT:-/data3/local_datasets/re10k}" \
    --output-dir "${destination}" --warmup-scenes "${WARMUP_SCENES}" \
    --measure-scenes "${MEASURE_SCENES}" --num-workers "${PROFILE_NUM_WORKERS:-4}" \
    --threads 8 --seed 0 --device cuda --precision "${PROFILE_PRECISION:-bf16}"
}

context_suite() {
  local lambda_tag="$1" tag experiment checkpoint
  experiment="nfcgs_factorized_continuation_rank56_lambda${lambda_tag}_residual_on_morton_on"
  checkpoint="${FACTOR_ROOT}/rank56/lambda${lambda_tag}/residual_on/checkpoints/${experiment}/version_0/step000050000.ckpt"
  profile_one context factorized "${lambda_tag}" 50000 "${checkpoint}"
  for tag in m1c0s0 m1c1s0 m1c0s1 m1c1s1; do
    case "${tag}" in
      m1c0s0) label=mean ;;
      m1c1s0) label=mean_channel ;;
      m1c0s1) label=mean_spatial ;;
      m1c1s1) label=full ;;
    esac
    experiment="nfcgs_score_context_joint_rank56_lambda${lambda_tag}_residual_on_morton_on_${tag}"
    checkpoint="${CONTEXT_ROOT}/joint/${tag}/rank56/lambda${lambda_tag}/residual_on/checkpoints/${experiment}/version_0/step000050000.ckpt"
    profile_one context "${label}" "${lambda_tag}" 50000 "${checkpoint}"
  done
}

probability_suite() {
  local lambda_tag="$1" directory suffix label experiment checkpoint
  for label in shared split gaussian conditional_scale; do
    case "${label}" in
      shared) directory=e0_shared; suffix=m1c1s1 ;;
      split) directory=e1_split; suffix=m1c1s1_split ;;
      gaussian) directory=e2_gaussian; suffix=m1c1s1_gaussian ;;
      conditional_scale) directory=e3_conditional_scale; suffix=m1c1s1_conditional_scale ;;
    esac
    experiment="nfcgs_score_probability10k_${directory}_rank56_lambda${lambda_tag}_residual_on_morton_on_${suffix}"
    checkpoint="${PROBABILITY_ROOT}/${directory}/rank56/lambda${lambda_tag}/residual_on/checkpoints/${experiment}/version_0/step000010000.ckpt"
    profile_one probability "${label}" "${lambda_tag}" 10000 "${checkpoint}"
  done
}

if [[ "${MODE}" == run ]]; then
  cd "${REPO_DIR}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-globalsplat}"
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  export OMP_NUM_THREADS=8
  export WANDB_MODE=disabled
  # Avoid Slurm's RANK variable being mistaken for the codec rank.
  export -n RANK 2>/dev/null || true
fi

echo "OVERHEAD_PLAN suite=${PROFILE_SUITE} run=${RUN_TAG} warmup=${WARMUP_SCENES} measure=${MEASURE_SCENES}"
for lambda_tag in 0p0064 0p0256; do
  [[ "${PROFILE_SUITE}" == probability ]] || context_suite "${lambda_tag}"
  [[ "${PROFILE_SUITE}" == context ]] || probability_suite "${lambda_tag}"
done
[[ "${MODE}" == run ]] || exit 0

python scripts/summarize_nfcgs_overhead.py --input-root "${OUTPUT_DIR}"
echo "OVERHEAD_ALL_COMPLETE output=${OUTPUT_DIR}"
