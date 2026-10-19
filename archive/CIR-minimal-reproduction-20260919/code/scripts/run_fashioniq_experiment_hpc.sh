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
PSEUDO_REPEAT="${9:-1}"

ROOT=/data/user/hqi881/CIR/project_stage1_dqu_baseline
ENV_PATH=/data/user/hqi881/envs/dqu-stage1-cir
FASHIONIQ_ROOT=${ROOT}/data/FashionIQ
PRETRAINED_PATH=/data/user/hqi881/CIR/model/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin

mkdir -p "${ROOT}/${EXPERIMENT_MODEL_DIR}"
cd "${ROOT}"
conda run -p "${ENV_PATH}" python src/train.py \
  --dataset "${CATEGORY}" \
  --fashioniq_split val-split \
  --fashioniq_path "${FASHIONIQ_ROOT}" \
  --lr 1e-4 \
  --clip_lr 1e-6 \
  --batch_size 16 \
  --num_epochs 100 \
  --num_workers 8 \
  --backbone ViT-H-14 \
  --pretrained "${PRETRAINED_PATH}" \
  --model_variant "${MODEL_VARIANT}" \
  --model_dir "${ROOT}/${EXPERIMENT_MODEL_DIR}" \
  --pseudo_path "${PSEUDO_PATH}" \
  --pseudo_start_epoch "${PSEUDO_START_EPOCH}" \
  --pseudo_min_score "${PSEUDO_MIN_SCORE}" \
  --pseudo_label "${PSEUDO_LABEL}" \
  --pseudo_repeat "${PSEUDO_REPEAT}" \
  --i "${EXPERIMENT_RUN_ID}"
