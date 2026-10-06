#!/usr/bin/env bash
# Submit missing 50k trainings, then full-test evaluations for four lambda points.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export REPO_DIR RD_LAMBDA4="${RD_LAMBDA4:-0.0032}"
export RD_INCLUDE_PLAIN4="${RD_INCLUDE_PLAIN4:-1}"
cd "${REPO_DIR}"
source "${REPO_DIR}/scripts/slurm/hyper1d_rd4_common.sh"
mkdir -p logs/slurm

vanilla="${VANILLA_CHECKPOINT:-${REPO_DIR}/checkpoints/pretrained/globalsplat-re10k-32k.ckpt}"
if [[ ! -f "${vanilla}" ]]; then
  echo "Missing vanilla checkpoint: ${vanilla}" >&2
  exit 2
fi

variants=(single_off single_on dual_off dual_on lowrank_off lowrank_on)
if [[ "${RD_INCLUDE_PLAIN4}" == 1 ]]; then variants+=(plain4); fi
for variant in "${variants[@]}"; do
  for lambda in "${RD_LAMBDAS[@]}"; do
    fixed="$(rd_fixed_checkpoint "${variant}" "${lambda}")"
    if [[ -n "${fixed}" && ! -f "${fixed}" ]]; then
      echo "Previously trained step-50000 checkpoint is missing: ${fixed}" >&2
      exit 2
    fi
  done
done

job_user="${USER:-${SLURM_JOB_USER:-}}"
if [[ -n "${job_user}" ]] && command -v squeue >/dev/null 2>&1; then
  active_sweep="$(squeue -h -u "${job_user}" -n gs-h1d-rd4-1g,gs-h1d-rd4-lr -o '%i %j')"
  if [[ -n "${active_sweep}" ]]; then
    echo "An RD4 training sweep is already active:" >&2
    echo "${active_sweep}" >&2
    exit 2
  fi
fi

# Do not start a second Morton-OFF low-rank lambda=0.0256 job while the older
# submission is still active and has not produced its final checkpoint.
old_root="${REPO_DIR}/outputs/hyper1d_lowrank_base_50k/rank56/morton_off/lambda0p0256"
existing_off=''
if [[ -d "${old_root}" ]]; then
  existing_off="$(find "${old_root}" -type f -path '*/checkpoints/hyper1d_12h/version_0/step000050000.ckpt' -print -quit)"
fi
if [[ -z "${existing_off}" && -n "${job_user}" ]] && command -v squeue >/dev/null 2>&1; then
  active="$(squeue -h -u "${job_user}" -n gs-lrbase-off50k -o '%i')"
  if [[ -n "${active}" ]]; then
    echo "Existing low-rank Morton-OFF job ${active} is active; wait for it to finish before submitting RD4." >&2
    exit 2
  fi
fi

one_array='0-10%4'
eval_array='0-27%4'
if [[ "${RD_INCLUDE_PLAIN4}" == 0 ]]; then
  one_array='0-9%4'
  eval_array='0-23%4'
fi
one_id="$(sbatch --parsable --array="${one_array}" \
  --export=ALL,RD_LAMBDA4="${RD_LAMBDA4}",RD_INCLUDE_PLAIN4="${RD_INCLUDE_PLAIN4}" \
  scripts/slurm/train_hyper1d_rd4_one_gpu.slurm)"
one_id="${one_id%%;*}"
lowrank_id="$(sbatch --parsable \
  --export=ALL,RD_LAMBDA4="${RD_LAMBDA4}",RD_INCLUDE_PLAIN4="${RD_INCLUDE_PLAIN4}" \
  scripts/slurm/train_hyper1d_rd4_lowrank.slurm)"
lowrank_id="${lowrank_id%%;*}"
eval_id="$(sbatch --parsable --array="${eval_array}" \
  --dependency="afterok:${one_id}:${lowrank_id}" \
  --export=ALL,RD_LAMBDA4="${RD_LAMBDA4}",RD_INCLUDE_PLAIN4="${RD_INCLUDE_PLAIN4}",RD_ONE_JOB_ID="${one_id}",RD_LOWRANK_JOB_ID="${lowrank_id}" \
  scripts/slurm/eval_hyper1d_rd4_full_test.slurm)"
eval_id="${eval_id%%;*}"

printf 'lambda points: %s\n' "${RD_LAMBDAS[*]}"
printf 'one-GPU training array: %s (at most 4 GPUs)\n' "${one_id}"
printf 'four-GPU low-rank array: %s (at most 8 GPUs)\n' "${lowrank_id}"
printf 'full-test evaluation array: %s (starts after both training arrays; at most 4 GPUs)\n' "${eval_id}"
printf 'maximum concurrent total: 12 GPUs during training\n'
summary_args=()
if [[ "${RD_INCLUDE_PLAIN4}" == 0 ]]; then summary_args=(--no-include-plain4); fi
printf 'after evaluation: python scripts/summarize_hyper1d_rd4.py --eval-job-id %s --lambda4 %s' "${eval_id}" "${RD_LAMBDA4}"
if (( ${#summary_args[@]} )); then printf ' %s' "${summary_args[@]}"; fi
printf '\n'
