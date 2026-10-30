#!/usr/bin/env bash
#SBATCH -J gs-nn-order-bench
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -p batch_ugrad
#SBATCH -t 00:30:00
#SBATCH -o logs/slurm/slurm-gs-nn-order-bench-%j.out
#SBATCH -e logs/slurm/slurm-gs-nn-order-bench-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p outputs/nn_order_benchmark
exec srun --ntasks=1 python scripts/benchmark_nfcgs_nn_order.py --device cuda \
  --tokens "${NN_ORDER_TOKENS:-4096}" --batch-size "${NN_ORDER_BATCH_SIZE:-2}" \
  --candidates "${NN_ORDER_CANDIDATES:-64}" --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --repeats 5 --warmup 1 \
  --output "outputs/nn_order_benchmark/job_${SLURM_JOB_ID:?submit with sbatch}.json"
