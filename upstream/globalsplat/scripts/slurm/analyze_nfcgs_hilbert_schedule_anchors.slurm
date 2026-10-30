#!/usr/bin/env bash
#SBATCH -J gs-hsched-anchors
#SBATCH --nodes=1
#SBATCH --exclude=ariel-v7
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -p batch_ugrad
#SBATCH -t 08:00:00
#SBATCH -o logs/slurm/slurm-gs-hsched-anchors-%j.out
#SBATCH -e logs/slurm/slurm-gs-hsched-anchors-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
MODE="${1:-run}"
[[ "${MODE}" == run || "${MODE}" == plan ]] || { echo "usage: bash $0 plan | sbatch $0" >&2; exit 2; }
for file in globalsplat/compression/mechanism.py scripts/profile_nfcgs_complexity.py scripts/analyze_nfcgs_mechanism.py scripts/report_nfcgs_mechanism.py; do
  [[ -s "${file}" ]] || { echo "missing analysis file: ${file}" >&2; exit 1; }
done
grep -Fq 'hilbert-schedule-anchors' scripts/analyze_nfcgs_mechanism.py || {
  echo "Upload the updated scripts/analyze_nfcgs_mechanism.py first." >&2; exit 1;
}
MANIFEST="${SCHEDULE_MANIFEST:-${REPO_DIR}/logs/slurm/nfcgs_hilbert_schedule_anchors_2026-10-17.json}"
if [[ ! -f "${MANIFEST}" ]]; then
  SCHEDULE_MANIFEST_PATH="${MANIFEST}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path.cwd()
common = dict(texture_channels=512, geometry_channels=512, geometry_observable_channels=224,
    rank=56, residual_n=192, residual_m=320, adapter_hidden=96, morton_bits=10,
    use_morton=True, token_order="hilbert", score_order_scale=None,
    nn_order_backend="legacy", nn_order_candidates=64, nn_order_distance_device="cpu",
    use_residual=True, use_centering=False, use_score_norm=False, transform="linear", transform_hidden=32,
    score_mean_condition=False, score_channel_context=False, score_spatial_context=True,
    score_spatial_predictor="linear", score_spatial_entropy="split", score_spatial_hidden=32,
    score_slice_channels=16, score_context_hidden=64, score_spatial_kernel=5, score_context_quantization="ste")
points = []
for label, schedule, stages, job in (
    ("hilbert_legacy3_0p0128", "legacy", 3, 447725),
    ("hilbert_quarter2_0p0128", "quarter2", 2, 447740),
    ("hilbert_dyadic4_0p0128", "dyadic4", 4, 447746)):
    checkpoint = root / "outputs/nfcgs_order_context/hilbert" / f"{schedule}_stages{stages}_k5_ste"
    checkpoint /= f"lambda0p0128/job_{job}/checkpoints/nfcgs_main/version_0/step000050000.ckpt"
    if not checkpoint.is_file():
        raise SystemExit(f"MISSING CHECKPOINT: {checkpoint}")
    points.append(dict(label=label, family=label.rsplit("_", 1)[0], checkpoint=str(checkpoint),
        rate_lambda=.0128, expected_step=50000, tokens=4096,
        config=dict(common, score_context_schedule=schedule, score_spatial_stages=stages)))
manifest = dict(version=1, reference="hilbert_legacy3_0p0128", scope="hilbert-schedule-anchors", points=points)
path = Path(os.environ["SCHEDULE_MANIFEST_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(f"MANIFEST_CREATED={path}", flush=True)
PY
fi
INPUT_RUN="${SCHEDULE_INPUT_RUN:-${REPO_DIR}/outputs/nfcgs_legacy3_anchors/job_454995}"
FINGERPRINT="${SCHEDULE_INPUT_FINGERPRINT:-069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29}"
OUTPUT="${SCHEDULE_OUTPUT:-${REPO_DIR}/outputs/nfcgs_hilbert_schedule_anchors/job_${SLURM_JOB_ID:-plan}}"
ARGS=(--scope hilbert-schedule-anchors --manifest "${MANIFEST}" --input-run "${INPUT_RUN}"
  --expected-input-fingerprint "${FINGERPRINT}" --output "${OUTPUT}" --device cuda --precision bf16
  --scenes 256 --trace-scenes "${SCHEDULE_TRACE_SCENES:-8}" --tokens 4096 --seed 0 --threads 8
  --num-workers "${SCHEDULE_NUM_WORKERS:-4}" --dataset-root "${DATASET_ROOT:-/data3/local_datasets/re10k}")
[[ "${SCHEDULE_RESUME:-0}" != 1 ]] || ARGS+=(--resume)
if [[ "${MODE}" == plan ]]; then
  exec python scripts/analyze_nfcgs_mechanism.py "${ARGS[@]}" --plan
fi
: "${SLURM_JOB_ID:?submit with sbatch}"
NFCGS_TMP_ROOT="${NFCGS_TMP_ROOT:-/ceph_data/${USER:?missing USER}/.tmp/nfcgs}"
export TMPDIR="${NFCGS_TMP_ROOT}/hsched_anchors_${SLURM_JOB_ID}"
mkdir -p -- "${TMPDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 WANDB_MODE=disabled
export -n RANK 2>/dev/null || true
echo "SCHEDULE_TMPDIR=${TMPDIR} INPUT_RUN=${INPUT_RUN} OUTPUT=${OUTPUT}"
srun --ntasks=1 python scripts/analyze_nfcgs_mechanism.py "${ARGS[@]}"
REPORT="${OUTPUT}/report_${SLURM_JOB_ID}"
python scripts/report_nfcgs_mechanism.py --input "${OUTPUT}" --output "${REPORT}"
echo "SCHEDULE_REPORT=${REPORT}"
