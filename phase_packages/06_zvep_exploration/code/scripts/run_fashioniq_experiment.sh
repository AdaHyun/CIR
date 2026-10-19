#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${1:?category required}"
MODEL_VARIANT="${2:?model variant required}"
EXPERIMENT_RUN_ID="${3:?run id required}"
EXPERIMENT_MODEL_DIR="${4:?model dir required}"
PSEUDO_PATH="${5:-}"
PSEUDO_START_EPOCH="${6:-0}"
PSEUDO_MIN_SCORE="${7:-0.0}"
PSEUDO_LABEL="${8:-none}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${ROOT_DIR}/configs/fashioniq_${CATEGORY}.env"
PYTHON_BIN="${PYTHON_BIN:-python}"
PRETRAINED_VALUE="${PRETRAINED_PATH:-}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Unknown category: ${CATEGORY}"
  echo "Expected config: ${CONFIG_FILE}"
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_FILE}"

mkdir -p "${ROOT_DIR}/${EXPERIMENT_MODEL_DIR}"

if [[ -z "${PRETRAINED_VALUE}" ]]; then
  PRETRAINED_VALUE="${PRETRAINED}"
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" src/train.py \
  --dataset "${DATASET}" \
  --fashioniq_split "${FASHIONIQ_SPLIT}" \
  --fashioniq_path "${ROOT_DIR}/data/FashionIQ" \
  --lr "${LR}" \
  --clip_lr "${CLIP_LR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  --backbone "${BACKBONE}" \
  --pretrained "${PRETRAINED_VALUE}" \
  --model_variant "${MODEL_VARIANT}" \
  --model_dir "${ROOT_DIR}/${EXPERIMENT_MODEL_DIR}" \
  --pseudo_path "${PSEUDO_PATH}" \
  --pseudo_start_epoch "${PSEUDO_START_EPOCH}" \
  --pseudo_min_score "${PSEUDO_MIN_SCORE}" \
  --pseudo_label "${PSEUDO_LABEL}" \
  --i "${EXPERIMENT_RUN_ID}"
