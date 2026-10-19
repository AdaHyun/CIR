#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${1:?category required}"
RUN_ID="${2:?run id required}"
MODEL_DIR="${3:?model dir required}"
PSEUDO_PATH="${4:?pseudo path required}"
MIX_MODE="${5:?mix mode required}"
PSEUDO_START_EPOCH="${6:-0}"
PSEUDO_MAX_RATIO="${7:-0.25}"
PSEUDO_RAMP_EPOCHS="${8:-5}"
PSEUDO_QUALITY_POWER="${9:-1.0}"
PSEUDO_SAMPLES_PER_EPOCH="${10:-0}"
PSEUDO_MIN_SCORE="${11:-0.0}"
MODEL_VARIANT="${12:-dqu_pfp_refcond}"
PSEUDO_LABEL="${13:-structured}"

ROOT=/data/user/hqi881/CIR/project_stage1_dqu_baseline
ENV_PATH=/data/user/hqi881/envs/dqu-stage1-cir
FASHIONIQ_ROOT=${ROOT}/data/FashionIQ
PRETRAINED_PATH=/data/user/hqi881/CIR/model/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin

mkdir -p "${ROOT}/${MODEL_DIR}"
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
  --model_dir "${ROOT}/${MODEL_DIR}" \
  --pseudo_path "${PSEUDO_PATH}" \
  --pseudo_start_epoch "${PSEUDO_START_EPOCH}" \
  --pseudo_min_score "${PSEUDO_MIN_SCORE}" \
  --pseudo_label "${PSEUDO_LABEL}" \
  --mix_mode "${MIX_MODE}" \
  --pseudo_max_ratio "${PSEUDO_MAX_RATIO}" \
  --pseudo_ramp_epochs "${PSEUDO_RAMP_EPOCHS}" \
  --pseudo_quality_power "${PSEUDO_QUALITY_POWER}" \
  --pseudo_samples_per_epoch "${PSEUDO_SAMPLES_PER_EPOCH}" \
  --i "${RUN_ID}"

