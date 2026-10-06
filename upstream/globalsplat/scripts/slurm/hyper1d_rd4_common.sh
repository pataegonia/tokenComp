d#!/usr/bin/env bash
# Shared experiment IDs for the four-point Hyper1D rate-distortion sweep.
# Source this file after setting REPO_DIR.

case "${RD_LAMBDA4:-0.0032}" in
  0.0032) RD_LAMBDAS=(0.0256 0.0128 0.0064 0.0032) ;;
  0.0512) RD_LAMBDAS=(0.0512 0.0256 0.0128 0.0064) ;;
  *) echo 'RD_LAMBDA4 must be 0.0032 or 0.0512' >&2; return 2 ;;
esac
RD_LAMBDA4="${RD_LAMBDA4:-0.0032}"
RD_INCLUDE_PLAIN4="${RD_INCLUDE_PLAIN4:-1}"
if [[ "${RD_INCLUDE_PLAIN4}" != 0 && "${RD_INCLUDE_PLAIN4}" != 1 ]]; then
  echo 'RD_INCLUDE_PLAIN4 must be 0 or 1' >&2
  return 2
fi

rd_lambda_tag() {
  printf '%s' "${1//./p}"
}

rd_one_task() {
  case "$1" in
    0) RD_VARIANT=single_off; RD_LAMBDA="${RD_LAMBDA4}" ;;
    1) RD_VARIANT=single_on; RD_LAMBDA=0.0128 ;;
    2) RD_VARIANT=single_on; RD_LAMBDA=0.0064 ;;
    3) RD_VARIANT=single_on; RD_LAMBDA="${RD_LAMBDA4}" ;;
    4) RD_VARIANT=dual_off; RD_LAMBDA=0.0128 ;;
    5) RD_VARIANT=dual_off; RD_LAMBDA=0.0064 ;;
    6) RD_VARIANT=dual_off; RD_LAMBDA="${RD_LAMBDA4}" ;;
    7) RD_VARIANT=dual_on; RD_LAMBDA=0.0128 ;;
    8) RD_VARIANT=dual_on; RD_LAMBDA=0.0064 ;;
    9) RD_VARIANT=dual_on; RD_LAMBDA="${RD_LAMBDA4}" ;;
    10) RD_VARIANT=plain4; RD_LAMBDA="${RD_LAMBDA4}"
        [[ "${RD_INCLUDE_PLAIN4}" == 1 ]] || return 2 ;;
    *) echo "Invalid one-GPU sweep task: $1" >&2; return 2 ;;
  esac
}

rd_lowrank_task() {
  case "$1" in
    0) RD_VARIANT=lowrank_off; RD_LAMBDA=0.0256 ;;
    1) RD_VARIANT=lowrank_off; RD_LAMBDA=0.0128 ;;
    2) RD_VARIANT=lowrank_off; RD_LAMBDA=0.0064 ;;
    3) RD_VARIANT=lowrank_off; RD_LAMBDA="${RD_LAMBDA4}" ;;
    4) RD_VARIANT=lowrank_on; RD_LAMBDA=0.0128 ;;
    5) RD_VARIANT=lowrank_on; RD_LAMBDA=0.0064 ;;
    6) RD_VARIANT=lowrank_on; RD_LAMBDA="${RD_LAMBDA4}" ;;
    *) echo "Invalid four-GPU sweep task: $1" >&2; return 2 ;;
  esac
}

rd_eval_task() {
  local task="$1" variant_index lambda_index
  if (( task < 0 || task >= 24 + 4 * RD_INCLUDE_PLAIN4 )); then
    echo "Invalid evaluation task: $task" >&2
    return 2
  fi
  variant_index=$((task / 4))
  lambda_index=$((task % 4))
  case "${variant_index}" in
    0) RD_VARIANT=single_off ;;
    1) RD_VARIANT=single_on ;;
    2) RD_VARIANT=dual_off ;;
    3) RD_VARIANT=dual_on ;;
    4) RD_VARIANT=lowrank_off ;;
    5) RD_VARIANT=lowrank_on ;;
    6) RD_VARIANT=plain4 ;;
  esac
  RD_LAMBDA="${RD_LAMBDAS[lambda_index]}"
}

rd_variant_config() {
  RD_ARCH=legacy RD_PATHS=1 RD_RANK=0 RD_MORTON=off
  case "$1" in
    single_off) ;;
    single_on) RD_MORTON=on ;;
    dual_off) RD_PATHS=2 ;;
    dual_on) RD_PATHS=2; RD_MORTON=on ;;
    lowrank_off) RD_PATHS=2; RD_RANK=56 ;;
    lowrank_on) RD_PATHS=2; RD_RANK=56; RD_MORTON=on ;;
    plain4) RD_ARCH=plain4 ;;
    *) echo "Invalid Hyper1D variant: $1" >&2; return 2 ;;
  esac
}

rd_fixed_checkpoint() {
  local root="${REPO_DIR}/outputs" suffix='/checkpoints/hyper1d_12h/version_0/step000050000.ckpt'
  case "$1:$2" in
    single_off:0.0256) printf '%s\n' "${root}/hyper1d/20260928_161549_981799${suffix}" ;;
    single_off:0.0128) printf '%s\n' "${root}/hyper1d_high_quality_50k/legacy/lambda0p0128/job_437624_0${suffix}" ;;
    single_off:0.0064) printf '%s\n' "${root}/hyper1d_high_quality_50k/legacy/lambda0p0064/job_437624_2${suffix}" ;;
    single_on:0.0256) printf '%s\n' "${root}/hyper1d_msh_compare_50k/single_msh/morton_on/lambda0p0256/job_438115_0${suffix}" ;;
    dual_off:0.0256) printf '%s\n' "${root}/hyper1d_msh_compare_50k/base_residual_msh/morton_off/lambda0p0256/job_441056${suffix}" ;;
    dual_on:0.0256) printf '%s\n' "${root}/hyper1d_msh_compare_50k/base_residual_msh/morton_on/lambda0p0256/job_438115_1${suffix}" ;;
    lowrank_on:0.0256) printf '%s\n' "${root}/hyper1d_lowrank_base_50k/rank56/morton_on/lambda0p0256/job_441332${suffix}" ;;
    plain4:0.0256) printf '%s\n' "${root}/hyper1d_plain4/job_436321${suffix}" ;;
    plain4:0.0128) printf '%s\n' "${root}/hyper1d_high_quality_50k/plain4/lambda0p0128/job_437624_1${suffix}" ;;
    plain4:0.0064) printf '%s\n' "${root}/hyper1d_high_quality_50k/plain4/lambda0p0064/job_437624_3${suffix}" ;;
    *) printf '\n' ;;
  esac
}

rd_manifest_path() {
  local group job_id tag
  tag="$(rd_lambda_tag "$2")"
  case "$1" in
    lowrank_*) group=lowrank; job_id="${RD_LOWRANK_JOB_ID:?RD_LOWRANK_JOB_ID is required}" ;;
    *) group=one_gpu; job_id="${RD_ONE_JOB_ID:?RD_ONE_JOB_ID is required}" ;;
  esac
  printf '%s\n' "${REPO_DIR}/outputs/hyper1d_rd4_50k/manifests/${group}/job_${job_id}/${1}_lambda${tag}.txt"
}
