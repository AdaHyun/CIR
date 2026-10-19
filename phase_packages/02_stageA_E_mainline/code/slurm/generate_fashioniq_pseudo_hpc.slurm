#!/bin/bash
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH -o /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/%x_%j.out
#SBATCH -e /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/%x_%j.err
set -euo pipefail

CATEGORY="${CATEGORY:?CATEGORY required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT required}"
OUTPUT_JSONL="${OUTPUT_JSONL:?OUTPUT_JSONL required}"
OUTPUT_SUMMARY="${OUTPUT_SUMMARY:?OUTPUT_SUMMARY required}"
OUTPUT_CSV="${OUTPUT_CSV:-}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.60}"
MIN_DIRECTION_MARGIN="${MIN_DIRECTION_MARGIN:-0.0}"
MIN_NUM_CHANGES="${MIN_NUM_CHANGES:-1}"
MAX_NUM_CHANGES="${MAX_NUM_CHANGES:-2}"
KEEP_TOP_N="${KEEP_TOP_N:-0}"
MAX_RECORDS_PER_MODIFICATION="${MAX_RECORDS_PER_MODIFICATION:-0}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-128}"
NEIGHBOR_TOPK="${NEIGHBOR_TOPK:-10}"

module load miniconda3
ROOT=/data/user/hqi881/CIR/project_stage1_dqu_baseline
ENV_PATH=/data/user/hqi881/envs/dqu-stage1-cir
FASHIONIQ_ROOT=${ROOT}/data/FashionIQ

echo "[$(date)] category=${CATEGORY} checkpoint=${CHECKPOINT} threshold=${SCORE_THRESHOLD} min_direction_margin=${MIN_DIRECTION_MARGIN} min_changes=${MIN_NUM_CHANGES} max_changes=${MAX_NUM_CHANGES} keep_top_n=${KEEP_TOP_N} max_records_per_modification=${MAX_RECORDS_PER_MODIFICATION}"
cd "${ROOT}"
conda run -p "${ENV_PATH}" python scripts/generate_fashioniq_pseudo.py \
  --category "${CATEGORY}" \
  --fashioniq_path "${FASHIONIQ_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --output_summary "${OUTPUT_SUMMARY}" \
  ${OUTPUT_CSV:+--output_csv "${OUTPUT_CSV}"} \
  --score_threshold "${SCORE_THRESHOLD}" \
  --min_direction_margin "${MIN_DIRECTION_MARGIN}" \
  --min_num_changes "${MIN_NUM_CHANGES}" \
  --max_num_changes "${MAX_NUM_CHANGES}" \
  --keep_top_n "${KEEP_TOP_N}" \
  --max_records_per_modification "${MAX_RECORDS_PER_MODIFICATION}" \
  --query_batch_size "${QUERY_BATCH_SIZE}" \
  --neighbor_topk "${NEIGHBOR_TOPK}"
