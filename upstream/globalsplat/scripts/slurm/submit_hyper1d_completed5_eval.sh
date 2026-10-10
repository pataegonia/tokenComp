#!/usr/bin/env bash
# Evaluate the five newly completed 50k runs on the full test set.
set -euo pipefail

# Resolve from this file so an inherited REPO_DIR cannot select tokencomp.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export REPO_DIR
export RD_LAMBDA4=0.0032 RD_INCLUDE_PLAIN4=1
export RD_ONE_JOB_ID=442942 RD_LOWRANK_JOB_ID=442943
cd "${REPO_DIR}"

common="${REPO_DIR}/scripts/slurm/hyper1d_rd4_common.sh"
eval_script="${REPO_DIR}/scripts/slurm/eval_hyper1d_rd4_full_test.slurm"
for required in "${common}" "${eval_script}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing evaluation script: ${required}" >&2
    exit 2
  fi
done
source "${common}"

# RD4 evaluation IDs: dual_off .0032; dual_on .0128/.0064/.0032;
# plain4 .0032. Each task has a separate output directory.
tasks=(11 13 14 15 27)
for task in "${tasks[@]}"; do
  rd_eval_task "${task}"
  manifest="$(rd_manifest_path "${RD_VARIANT}" "${RD_LAMBDA}")"
  if [[ ! -s "${manifest}" ]]; then
    echo "Missing trained checkpoint manifest: ${manifest}" >&2
    exit 2
  fi
  checkpoint=''
  IFS= read -r checkpoint < "${manifest}" || [[ -n "${checkpoint}" ]]
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing evaluation checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  printf '[ready] task=%s variant=%s lambda=%s checkpoint=%s\n' \
    "${task}" "${RD_VARIANT}" "${RD_LAMBDA}" "${checkpoint}"
done

mkdir -p logs/slurm
job_id="$(sbatch --parsable \
  --array=11,13-15,27 \
  --exclude='ariel-v[6-7]' \
  --chdir="${REPO_DIR}" \
  --export="ALL,REPO_DIR=${REPO_DIR},RD_LAMBDA4=0.0032,RD_INCLUDE_PLAIN4=1,RD_ONE_JOB_ID=442942,RD_LOWRANK_JOB_ID=442943" \
  "${eval_script}")"
job_id="${job_id%%;*}"
printf 'Full-test evaluation array: %s\n' "${job_id}"
printf 'GPU requirement: 1 per task, up to 5 concurrently\n'
printf 'Logs: %s/logs/slurm/slurm-gs-h1d-rd4-eval-%s_<task>.out\n' \
  "${REPO_DIR}" "${job_id}"
