#!/usr/bin/env bash
# Submit the receiver immediately with an afterok dependency on the sender.
set -euo pipefail
[[ $# -eq 0 ]] || { echo "usage: $0" >&2; exit 2; }

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd "${REPO_DIR}"
mkdir -p logs/slurm

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/outputs/nfcgs_main/sender_receiver_full/existing_b/${RUN_ID}}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0064/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0064_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
BITSTREAM_DIR="${BITSTREAM_DIR:-${RUN_ROOT}/bitstreams}"
OUTPUT="${OUTPUT:-${RUN_ROOT}/decoder_eval}"
[[ -f "${CHECKPOINT}" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }

export REPO_DIR CHECKPOINT DATASET_ROOT BITSTREAM_DIR OUTPUT
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-globalsplat}"
export WORKERS="${WORKERS:-4}"
export PRECISION="${PRECISION:-bf16-mixed}"

ENCODE_SUBMISSION="$(sbatch --parsable \
  scripts/slurm/encode_nfcgs_bitstreams_full_v11.slurm)"
ENCODE_ID="${ENCODE_SUBMISSION%%;*}"
DECODE_SUBMISSION="$(sbatch --parsable \
  --dependency="afterok:${ENCODE_ID}" \
  scripts/slurm/decode_nfcgs_bitstreams_full_v11.slurm)"
DECODE_ID="${DECODE_SUBMISSION%%;*}"

echo "sender job:  ${ENCODE_ID}"
echo "receiver job: ${DECODE_ID} (afterok:${ENCODE_ID})"
echo "bitstreams:   ${BITSTREAM_DIR}"
echo "results:      ${OUTPUT}"
echo "squeue -j ${ENCODE_ID},${DECODE_ID} -o '%.18i %.24j %.10T %.30R'"
