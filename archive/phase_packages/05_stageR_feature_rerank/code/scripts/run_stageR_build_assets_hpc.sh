#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CATEGORY="${CATEGORY:?CATEGORY is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
FASHIONIQ_PATH="${FASHIONIQ_PATH:-/data/user/hqi881/CIR/dataset/FashionIQ}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/artifacts/stageR_feature_rerank_20260423}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_TOPK="${TRAIN_TOPK:-64}"
VAL_TOPK="${VAL_TOPK:-100}"

OUT_DIR="$RUN_ROOT/assets/$CATEGORY"
mkdir -p "$OUT_DIR"

python scripts/build_fashioniq_rerank_assets.py \
  --category "$CATEGORY" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUT_DIR" \
  --fashioniq_path "$FASHIONIQ_PATH" \
  --batch_size "$BATCH_SIZE" \
  --train_topk "$TRAIN_TOPK" \
  --val_topk "$VAL_TOPK"
