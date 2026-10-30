#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd -P -- "${REPO_DIR}"
export REPO_DIR="$(pwd -P)"
exec python scripts/submit_nfcgs_context_seed_pair.py "$@"
