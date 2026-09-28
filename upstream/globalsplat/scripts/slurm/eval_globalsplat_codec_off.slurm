#!/usr/bin/env bash
#SBATCH -J gs-codec-off-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/slurm-gs-codec-off-eval-%j.out
#SBATCH -e logs/slurm/slurm-gs-codec-off-eval-%j.err

set -euo pipefail

# Official uncompressed baseline; matches corrected paper eval data/precision.
# From the GlobalSplat repository root:
#   mkdir -p logs/slurm
#   sbatch scripts/slurm/eval_globalsplat_codec_off.slurm
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/pretrained/globalsplat-re10k-32k.ckpt}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-globalsplat}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_OUTPUT="${REPO_DIR}/outputs/globalsplat_codec_off_eval/all_test_ctx12/job_${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"

[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo 'ERROR: run from the GlobalSplat repository root' >&2; exit 1;
}
[[ -f "${CHECKPOINT}" && -f "${DATASET_ROOT}/test/index.json" ]] || {
  echo 'ERROR: missing official checkpoint or RE10K test index' >&2; exit 1;
}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_DIR}"
python - "${CHECKPOINT}" <<'PY'
import sys
import torch
assert torch.cuda.is_available(), 'CUDA is required'
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
state = checkpoint.get('state_dict', checkpoint)
assert not any(k.startswith('model.feature_codec.') for k in state), \
    'Use the official vanilla checkpoint, not a codec checkpoint'
print('CHECKPOINT=', sys.argv[1])
print('CODEC=off PROTOCOL=all CONTEXT=12 TARGET=8 PRECISION=bf16-mixed')
PY
mkdir -p "${RUN_OUTPUT}"
srun --ntasks=1 python -m globalsplat.main \
  model=globalsplat model.feature_codec=null \
  +experiment=re10k_32k dataset=re10k_eval_all_ctx12 mode=test \
  checkpointing.load="${CHECKPOINT}" \
  "dataset.dataset_roots=[${DATASET_ROOT}]" dataset.augment=false \
  optimizer.batch_size=1 optimizer.num_workers="${NUM_WORKERS}" \
  trainer.devices=1 trainer.num_nodes=1 trainer.strategy=auto \
  trainer.precision=bf16-mixed \
  test.max_scenes=null test.actual_bitstream=false test.compute_scores=true \
  test.save_image=false test.save_gt_image=false test.save_input_images=false \
  test.save_video=false test.output_path="${RUN_OUTPUT}" \
  experiment_name=globalsplat_codec_off_all_test_ctx12 \
  hydra.run.dir="${RUN_OUTPUT}/hydra" hydra.output_subdir=.hydra
echo "Results: ${RUN_OUTPUT}/globalsplat_codec_off_all_test_ctx12"
