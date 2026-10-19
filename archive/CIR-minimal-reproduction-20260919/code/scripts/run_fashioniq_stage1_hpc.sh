#!/usr/bin/env bash
set -euo pipefail
CATEGORY="${1:-dress}"
ROOT=/data/user/hqi881/CIR/project_stage1_dqu_baseline
ENV_PATH=/data/user/hqi881/envs/dqu-stage1-cir
FASHIONIQ_ROOT=/data/user/hqi881/CIR/dataset/FashionIQ
PRETRAINED_PATH=/data/user/hqi881/CIR/model/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
mkdir -p "${ROOT}/checkpoints/fashioniq/${CATEGORY}"
cd "${ROOT}"
conda run -p "${ENV_PATH}" python src/train.py --dataset "${CATEGORY}" --fashioniq_split val-split --fashioniq_path "${FASHIONIQ_ROOT}" --lr 1e-4 --clip_lr 1e-6 --batch_size 16 --num_epochs 100 --num_workers 8 --backbone ViT-H-14 --pretrained "${PRETRAINED_PATH}" --model_dir "${ROOT}/checkpoints/fashioniq/${CATEGORY}" --i stage1
