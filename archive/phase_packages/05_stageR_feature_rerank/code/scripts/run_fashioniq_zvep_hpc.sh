#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${1:?category required}"
MODEL_VARIANT="${2:?model variant required}"
RUN_ID="${3:?run id required}"
MODEL_DIR="${4:?model dir required}"
NUM_EPOCHS_OVERRIDE="${5:-}"

ROOT=/data/user/hqi881/CIR/project_stage1_dqu_baseline
ENV_PATH=/data/user/hqi881/envs/dqu-stage1-cir
CONFIG_FILE="${ROOT}/configs/zvep/zvep_base.env"
FASHIONIQ_ROOT="${ROOT}/data/FashionIQ"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "missing config: ${CONFIG_FILE}"
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_FILE}"

if [[ -n "${NUM_EPOCHS_OVERRIDE}" ]]; then
  NUM_EPOCHS="${NUM_EPOCHS_OVERRIDE}"
fi

PSEUDO_PATH="${PSEUDO_PATH:-}"

mkdir -p "${ROOT}/${MODEL_DIR}"
cd "${ROOT}"

conda run -p "${ENV_PATH}" python src/train.py \
  --dataset "${CATEGORY}" \
  --fashioniq_split "${FASHIONIQ_SPLIT}" \
  --fashioniq_path "${FASHIONIQ_ROOT}" \
  --lr "${LR}" \
  --clip_lr "${CLIP_LR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --dropout_rate "${DROPOUT_RATE}" \
  --backbone "${BACKBONE}" \
  --pretrained "${PRETRAINED}" \
  --model_variant "${MODEL_VARIANT}" \
  --model_dir "${ROOT}/${MODEL_DIR}" \
  --pseudo_path "${PSEUDO_PATH}" \
  --pseudo_start_epoch "${PSEUDO_START_EPOCH}" \
  --pseudo_min_score "${PSEUDO_MIN_SCORE}" \
  --pseudo_label "${PSEUDO_LABEL}" \
  --pseudo_repeat "${PSEUDO_REPEAT}" \
  --mix_mode "${MIX_MODE}" \
  --pseudo_max_ratio "${PSEUDO_MAX_RATIO}" \
  --pseudo_ramp_epochs "${PSEUDO_RAMP_EPOCHS}" \
  --pseudo_quality_power "${PSEUDO_QUALITY_POWER}" \
  --pseudo_samples_per_epoch "${PSEUDO_SAMPLES_PER_EPOCH}" \
  --i "${RUN_ID}"
