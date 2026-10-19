#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${1:-dress}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${ROOT_DIR}/configs/fashioniq_${CATEGORY}.env"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Unknown category: ${CATEGORY}"
  echo "Expected config: ${CONFIG_FILE}"
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_FILE}"

mkdir -p "${ROOT_DIR}/${MODEL_DIR}"

cd "${ROOT_DIR}"
python src/train.py \
  --dataset "${DATASET}" \
  --fashioniq_split "${FASHIONIQ_SPLIT}" \
  --fashioniq_path "${ROOT_DIR}/data/FashionIQ" \
  --lr "${LR}" \
  --clip_lr "${CLIP_LR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  --backbone "${BACKBONE}" \
  --pretrained "${PRETRAINED}" \
  --model_dir "${ROOT_DIR}/${MODEL_DIR}" \
  --i "${RUN_ID}"
