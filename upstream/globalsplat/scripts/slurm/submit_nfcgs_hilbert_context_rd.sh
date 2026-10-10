#!/usr/bin/env bash
# Six new points: Hilbert quarter2/dyadic4 x lambda .0064/.0128/.0256.
set -euo pipefail
SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# Set the complete matrix explicitly, even if a previous sweep exported other values.
export RD_CONFIGS="hilbert_quarter2 hilbert_dyadic4"
export RD_LAMBDAS="0.0064 0.0128 0.0256"
exec bash "${SCRIPT_DIR}/submit_nfcgs_order_context_rd.sh" "$@"
