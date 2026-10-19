#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${1:?category required}"
RUN_ID="${2:?run id required}"
MODEL_DIR="${3:?model dir required}"
BANK_PATH="${4:?bank path required}"
RESUME_CHECKPOINT="${5:?resume checkpoint required}"
NUM_EPOCHS="${6:-24}"
LR="${7:-5e-5}"
CLIP_LR="${8:-5e-7}"
SOFT_LOSS_WEIGHT="${9:-0.15}"
CONFLICT_LOSS_WEIGHT="${10:-0.08}"
CONFLICT_MARGIN="${11:-0.04}"
BATCH_SIZE="${12:-8}"
BANK_SOFT_K="${13:-3}"
BANK_START_EPOCH="${14:-0}"

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
  --lr "${LR}" \
  --clip_lr "${CLIP_LR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --num_workers 8 \
  --backbone ViT-H-14 \
  --pretrained "${PRETRAINED_PATH}" \
  --model_variant dqu_pfp_refcond \
  --model_dir "${ROOT}/${MODEL_DIR}" \
  --mix_mode concat \
  --bank_path "${BANK_PATH}" \
  --bank_soft_k "${BANK_SOFT_K}" \
  --bank_start_epoch "${BANK_START_EPOCH}" \
  --soft_loss_weight "${SOFT_LOSS_WEIGHT}" \
  --conflict_loss_weight "${CONFLICT_LOSS_WEIGHT}" \
  --conflict_margin "${CONFLICT_MARGIN}" \
  --resume_checkpoint "${RESUME_CHECKPOINT}" \
  --tolerance_epoch 8 \
  --lr_decay 12 \
  --max_decay_epoch 24 \
  --i "${RUN_ID}"
