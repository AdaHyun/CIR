#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${1:-dress}"
PSEUDO_PATH="${2:?pseudo path required}"
PSEUDO_MIN_SCORE="${3:-0.55}"
PSEUDO_START_EPOCH="${4:-10}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${ROOT_DIR}/scripts/run_fashioniq_experiment.sh" \
  "${CATEGORY}" \
  "dqu_pfp" \
  "stage4" \
  "checkpoints/fashioniq_stage4/${CATEGORY}" \
  "${PSEUDO_PATH}" \
  "${PSEUDO_START_EPOCH}" \
  "${PSEUDO_MIN_SCORE}" \
  "filtered_pseudo"
