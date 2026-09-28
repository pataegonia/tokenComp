#!/usr/bin/env bash
# Prune only the known 2026-09-14 Full low-LR checkpoint runs.
#
# Usage from the GlobalSplat repository root:
#   bash scripts/cleanup_nfcgs_full_lowlr_checkpoints.sh plan
#   bash scripts/cleanup_nfcgs_full_lowlr_checkpoints.sh apply
#
# The three evaluated step-10k checkpoints are always retained.  The default
# plan mode is read-only; apply refuses to run unless every retained checkpoint
# exists and is non-empty.

set -euo pipefail

MODE="${1:-plan}"
case "${MODE}" in
  plan|apply) ;;
  *)
    echo "Usage: bash $0 plan|apply" >&2
    exit 2
    ;;
esac

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
if [[ ! -f "${REPO_DIR}/pyproject.toml" || ! -d "${REPO_DIR}/globalsplat" ]]; then
  echo "ERROR: run from the GlobalSplat repository root" >&2
  exit 1
fi

OUTPUT_ROOT="${REPO_DIR}/outputs/nfcgs_full_lowlr10k"
OUTPUT_ROOT_REAL="$(realpath -m -- "${OUTPUT_ROOT}")"
EXPECTED_ROOT_REAL="$(realpath -m -- "${REPO_DIR}/outputs/nfcgs_full_lowlr10k")"
if [[ "${OUTPUT_ROOT_REAL}" != "${EXPECTED_ROOT_REAL}" ]]; then
  echo "ERROR: resolved output root is outside the expected experiment root" >&2
  exit 1
fi

COMPLETED_RUN="${OUTPUT_ROOT_REAL}/20260914_083141"
FAILED_RERUN="${OUTPUT_ROOT_REAL}/20260914_121549/p0_linear/rank56/lambda0p0064/residual_on"
FAILED_ORIGINAL="${COMPLETED_RUN}/p0_linear/rank56/lambda0p0064/residual_on"

P2_0064="${COMPLETED_RUN}/p2_residual7/rank56/lambda0p0064/residual_on"
P0_0256="${COMPLETED_RUN}/p0_linear/rank56/lambda0p0256/residual_on"
P2_0256="${COMPLETED_RUN}/p2_residual7/rank56/lambda0p0256/residual_on"

declare -a KEEP=(
  "${P2_0064}/checkpoints/nfcgs_full_lowlr10k_p2_residual7_rank56_lambda0p0064_residual_on_morton_on_m1c1s1_residual7h32/version_0/step000010000.ckpt"
  "${P0_0256}/checkpoints/nfcgs_full_lowlr10k_p0_linear_rank56_lambda0p0256_residual_on_morton_on_m1c1s1/version_0/step000010000.ckpt"
  "${P2_0256}/checkpoints/nfcgs_full_lowlr10k_p2_residual7_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_residual7h32/version_0/step000010000.ckpt"
)

declare -a COMPLETED_CONDITIONS=(
  "${P2_0064}"
  "${P0_0256}"
  "${P2_0256}"
)

declare -a FAILED_CONDITIONS=(
  "${FAILED_ORIGINAL}"
  "${FAILED_RERUN}"
)

declare -a TARGETS=()

is_under_output_root() {
  local target="$1"
  local resolved_parent
  resolved_parent="$(realpath -m -- "$(dirname -- "${target}")")"
  [[ "${resolved_parent}/" == "${OUTPUT_ROOT_REAL}/"* ]]
}

add_target() {
  local target="$1"
  if [[ -e "${target}" || -L "${target}" ]]; then
    if ! is_under_output_root "${target}"; then
      echo "ERROR: refusing target outside ${OUTPUT_ROOT_REAL}: ${target}" >&2
      exit 1
    fi
    TARGETS+=("${target}")
  fi
}

for condition in "${COMPLETED_CONDITIONS[@]}"; do
  while IFS= read -r -d '' target; do
    case "$(basename -- "${target}")" in
      step000010000.ckpt) ;;
      step*.ckpt|last.ckpt) add_target "${target}" ;;
    esac
  done < <(find "${condition}/checkpoints" \( -type f -o -type l \) -name '*.ckpt' -print0 2>/dev/null || true)

  while IFS= read -r -d '' target; do
    add_target "${target}"
  done < <(find "${condition}/initialization" \( -type f -o -type l \) -name '*.ckpt' -print0 2>/dev/null || true)
done

# Neither failed condition produced an evaluated step-10k checkpoint.  Remove
# every checkpoint artifact under only these two exact condition directories.
for condition in "${FAILED_CONDITIONS[@]}"; do
  while IFS= read -r -d '' target; do
    add_target "${target}"
  done < <(find "${condition}" \( -type f -o -type l \) -name '*.ckpt' -print0 2>/dev/null || true)
done

echo "MODE=${MODE}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT_REAL}"
echo "KEEP:"
for target in "${KEEP[@]}"; do
  if [[ -s "${target}" ]]; then
    size="$(stat -c '%s' -- "${target}")"
    printf '  %12d  %s\n' "${size}" "${target}"
  else
    echo "  MISSING_OR_EMPTY  ${target}"
    if [[ "${MODE}" == apply ]]; then
      echo "ERROR: refusing apply because a retained checkpoint is missing or empty" >&2
      exit 1
    fi
  fi
done

echo "REMOVE:"
total_bytes=0
for target in "${TARGETS[@]}"; do
  size="$(stat -c '%s' -- "${target}" 2>/dev/null || echo 0)"
  total_bytes=$((total_bytes + size))
  printf '  %12d  %s\n' "${size}" "${target}"
done

printf 'REMOVE_COUNT=%d REMOVE_BYTES=%d REMOVE_GIB=%.3f\n' \
  "${#TARGETS[@]}" "${total_bytes}" "$(awk -v bytes="${total_bytes}" 'BEGIN { print bytes / 1073741824 }')"

if [[ "${MODE}" == plan ]]; then
  echo "PLAN ONLY: no files were removed. Re-run with apply after reviewing every path above."
  exit 0
fi

for target in "${TARGETS[@]}"; do
  rm -f -- "${target}"
done

echo "REMOVED=${#TARGETS[@]} checkpoint files (${total_bytes} bytes)."
echo "The three evaluated step000010000.ckpt files listed under KEEP were preserved."
