#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${1:?category required}"
OUTPUT_PATH="${2:?output path required}"
MAX_QUERIES="${3:-0}"

ROOT=/data/user/hqi881/CIR/project_stage1_dqu_baseline
ENV_PATH=/data/user/hqi881/envs/dqu-stage1-cir
FASHIONIQ_ROOT=${ROOT}/data/FashionIQ
PRETRAINED_PATH=/data/user/hqi881/CIR/model/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
STAGED_REFCOND=${ROOT}/checkpoints/stageD_refcond/${CATEGORY}/${CATEGORY}_stageD_refcond_best_model.pt
STAGEA_DQU=${ROOT}/checkpoints/stageA_dqu_baseline_v3/${CATEGORY}/${CATEGORY}_stageA3_best_model.pt

mkdir -p "$(dirname "${OUTPUT_PATH}")"
cd "${ROOT}"
conda run -p "${ENV_PATH}" python scripts/build_multiteacher_bank.py \
  --category "${CATEGORY}" \
  --fashioniq_path "${FASHIONIQ_ROOT}" \
  --pretrained "${PRETRAINED_PATH}" \
  --teacher_a_checkpoint "${STAGED_REFCOND}" \
  --teacher_a_variant dqu_pfp_refcond \
  --teacher_b_checkpoint "${STAGEA_DQU}" \
  --teacher_b_variant dqu \
  --output_path "${OUTPUT_PATH}" \
  --max_queries "${MAX_QUERIES}"
