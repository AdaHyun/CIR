#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${1:-dress}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${ROOT_DIR}/scripts/run_fashioniq_experiment.sh" \
  "${CATEGORY}" \
  "dqu_pfp" \
  "stage2" \
  "checkpoints/fashioniq_stage2/${CATEGORY}"
