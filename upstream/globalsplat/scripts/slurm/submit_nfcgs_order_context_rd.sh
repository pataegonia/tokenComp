#!/usr/bin/env bash
# Default: add two lower-rate points to existing curves. RD_CONFIGS/RD_LAMBDAS select others.
set -euo pipefail
case "${1:-}" in
  "") ;;
  --dry-run) export DRY_RUN=1 ;;
  *) echo "usage: bash $0 [--dry-run]" >&2; exit 2 ;;
esac
REPO_DIR="${REPO_DIR:-$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd -P -- "${REPO_DIR}"
REPO_DIR="$(pwd -P)"
[[ -f scripts/run_nfcgs.py ]] || { echo "Not a GlobalSplat repository: ${REPO_DIR}" >&2; exit 1; }
export REPO_DIR
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt}"
export SOURCE_CHECKPOINT
read -r -a CONFIGS <<< "${RD_CONFIGS:-morton_legacy3 hilbert_legacy3 morton_dyadic4}"
read -r -a RATES <<< "${RD_LAMBDAS:-0.0064 0.0128}"
declare -A SEEN_CONFIGS=() SEEN_RATES=()
for config in "${CONFIGS[@]}"; do
  case "${config}" in morton_legacy3|hilbert_legacy3|morton_dyadic4|hilbert_quarter2|hilbert_dyadic4) ;; *) echo "invalid RD_CONFIGS entry: ${config}" >&2; exit 2 ;; esac
  [[ -z "${SEEN_CONFIGS[${config}]:-}" ]] || { echo "duplicate config: ${config}" >&2; exit 2; }
  SEEN_CONFIGS[${config}]=1
done
for rate in "${RATES[@]}"; do
  case "${rate}" in 0.0064|0.0128|0.0256) ;; *) echo "invalid RD_LAMBDAS entry: ${rate}" >&2; exit 2 ;; esac
  [[ -z "${SEEN_RATES[${rate}]:-}" ]] || { echo "duplicate rate: ${rate}" >&2; exit 2; }
  SEEN_RATES[${rate}]=1
done
(( ${#CONFIGS[@]} > 0 && ${#RATES[@]} > 0 )) || { echo "RD_CONFIGS and RD_LAMBDAS must not be empty" >&2; exit 2; }
point() {
  local config="$1" rate="$2" preview="$3" order schedule stages label
  case "${config}" in
    morton_legacy3) order=morton; schedule=legacy; stages=3; label=m3 ;;
    hilbert_legacy3) order=hilbert; schedule=legacy; stages=3; label=h3 ;;
    morton_dyadic4) order=morton; schedule=dyadic4; stages=4; label=d4 ;;
    hilbert_quarter2) order=hilbert; schedule=quarter2; stages=2; label=hq2 ;;
    hilbert_dyadic4) order=hilbert; schedule=dyadic4; stages=4; label=hd4 ;;
  esac
  TOKEN_ORDER="${order}" CONTEXT_SCHEDULE="${schedule}" STAGES="${stages}" RATE_LAMBDA="${rate}" \
    DRY_RUN="${preview}" TRAIN_JOB_NAME="gs-oc-${label}-${rate/./p}" EVAL_JOB_NAME="gs-oc-e-${label}-${rate/./p}" \
    bash scripts/slurm/submit_nfcgs_order_context.sh
}
# Validate EVERY requested point before submitting the first job.
for config in "${CONFIGS[@]}"; do
  for rate in "${RATES[@]}"; do point "${config}" "${rate}" 1; done
done
for config in "${CONFIGS[@]}"; do
  case "${config}" in
    morton_legacy3) echo "REFERENCE lambda0p0256: Morton legacy3 train=441552_1 eval=441553_1" ;;
    hilbert_legacy3) echo "REFERENCE lambda0p0256: Hilbert legacy3 train=445516 eval=445517" ;;
    morton_dyadic4) echo "REFERENCE lambda0p0256: Morton dyadic4 train=445518 eval=445519" ;;
  esac
done
[[ "${DRY_RUN:-0}" != 1 ]] || { echo "PREVIEW ONLY: no jobs submitted"; exit 0; }
mkdir -p logs/slurm
SUBMISSION_RECORD="${RD_SUBMISSION_RECORD:-${REPO_DIR}/logs/slurm/order_context_rd_$(date +%Y%m%d_%H%M%S)_$$.tsv}"
[[ ! -e "${SUBMISSION_RECORD}" ]] || { echo "submission record already exists: ${SUBMISSION_RECORD}" >&2; exit 1; }
printf 'order\tschedule\tstages\tlambda\ttrain_job\teval_job\tsource_checkpoint\toutput\n' > "${SUBMISSION_RECORD}"
export SUBMISSION_RECORD
echo "SUBMISSION_RECORD=${SUBMISSION_RECORD}"
for config in "${CONFIGS[@]}"; do
  for rate in "${RATES[@]}"; do point "${config}" "${rate}" 0; done
done
echo "RD submissions complete: $((${#CONFIGS[@]} * ${#RATES[@]})) training jobs, each with an afterok full-scene eval."
