#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CATEGORY="${CATEGORY:?CATEGORY is required}"
ASSET_DIR="${ASSET_DIR:?ASSET_DIR is required}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/artifacts/stageR_feature_rerank_20260423}"
RUN_NAME="${RUN_NAME:-human_only}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
DROPOUT="${DROPOUT:-0.2}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
TRAIN_K="${TRAIN_K:-16}"
TRAIN_POOL="${TRAIN_POOL:-32}"
VAL_K="${VAL_K:-100}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.0}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-0.07}"
BASE_SCORE_SCALE="${BASE_SCORE_SCALE:-1.0}"

OUT_DIR="$RUN_ROOT/runs/$RUN_NAME/$CATEGORY"
mkdir -p "$OUT_DIR"

python scripts/train_fashioniq_feature_reranker.py \
  --category "$CATEGORY" \
  --asset_dir "$ASSET_DIR" \
  --output_dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --dropout "$DROPOUT" \
  --hidden_dim "$HIDDEN_DIM" \
  --train_k "$TRAIN_K" \
  --train_pool "$TRAIN_POOL" \
  --val_k "$VAL_K" \
  --distill_weight "$DISTILL_WEIGHT" \
  --distill_temperature "$DISTILL_TEMPERATURE" \
  --base_score_scale "$BASE_SCORE_SCALE"
