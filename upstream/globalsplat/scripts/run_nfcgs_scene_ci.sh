#!/usr/bin/env bash
# Analyze existing per-scene eval JSON files. No training, rendering, or GPU use.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(pwd -P)}"
[[ -f "${REPO_DIR}/pyproject.toml" && -d "${REPO_DIR}/globalsplat" ]] || {
  echo "ERROR: run from the GlobalSplat repository root" >&2; exit 1;
}
RUN_TAG="${SCENE_CI_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${SCENE_CI_OUTPUT_ROOT:-${REPO_DIR}/outputs/nfcgs_scene_ci}/${RUN_TAG}"
cd "${REPO_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONUNBUFFERED=1
python scripts/analyze_nfcgs_scene_ci.py \
  --manifest "${SCENE_CI_MANIFEST:-scripts/nfcgs_scene_ci_manifest_2026-09-15.json}" \
  --output-dir "${OUTPUT_DIR}" \
  --bootstrap-draws "${SCENE_CI_BOOTSTRAP_DRAWS:-20000}" \
  --confidence "${SCENE_CI_CONFIDENCE:-0.95}" \
  --seed "${SCENE_CI_SEED:-20260915}" \
  --bootstrap-batch-size "${SCENE_CI_BOOTSTRAP_BATCH_SIZE:-64}"
echo "SCENE_CI_OUTPUT=${OUTPUT_DIR}"
