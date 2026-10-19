#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PACKAGE_ROOT"

echo "[1/3] Verifying the archived query-level rankings..."
"$PYTHON_BIN" scripts/verify_rankings.py

echo "[2/3] Verifying the package checksums..."
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum --check manifests/SHA256SUMS.txt
else
  echo "sha256sum is unavailable; checksum verification was skipped." >&2
fi

echo "[3/3] Re-evaluating the selected residual rerankers on cached features..."
"$PYTHON_BIN" scripts/evaluate_cached_reranker.py --device auto

echo "All reproduction checks passed."
