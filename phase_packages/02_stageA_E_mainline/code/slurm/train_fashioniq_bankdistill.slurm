#!/bin/bash
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH -o /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/%x_%j.out
#SBATCH -e /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/%x_%j.err
set -euo pipefail

CATEGORY="${CATEGORY:?CATEGORY required}"
RUN_ID="${RUN_ID:?RUN_ID required}"
MODEL_DIR="${MODEL_DIR:?MODEL_DIR required}"
BANK_PATH="${BANK_PATH:?BANK_PATH required}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:?RESUME_CHECKPOINT required}"
NUM_EPOCHS="${NUM_EPOCHS:-24}"
LR="${LR:-5e-5}"
CLIP_LR="${CLIP_LR:-5e-7}"
SOFT_LOSS_WEIGHT="${SOFT_LOSS_WEIGHT:-0.15}"
CONFLICT_LOSS_WEIGHT="${CONFLICT_LOSS_WEIGHT:-0.08}"
CONFLICT_MARGIN="${CONFLICT_MARGIN:-0.04}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BANK_SOFT_K="${BANK_SOFT_K:-3}"
BANK_START_EPOCH="${BANK_START_EPOCH:-0}"

module load miniconda3
cd /data/user/hqi881/CIR/project_stage1_dqu_baseline
mkdir -p /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm
echo "[$(date)] bankdistill category=${CATEGORY} run_id=${RUN_ID} model_dir=${MODEL_DIR} bank=${BANK_PATH}"
bash scripts/run_fashioniq_bankdistill_hpc.sh \
  "${CATEGORY}" \
  "${RUN_ID}" \
  "${MODEL_DIR}" \
  "${BANK_PATH}" \
  "${RESUME_CHECKPOINT}" \
  "${NUM_EPOCHS}" \
  "${LR}" \
  "${CLIP_LR}" \
  "${SOFT_LOSS_WEIGHT}" \
  "${CONFLICT_LOSS_WEIGHT}" \
  "${CONFLICT_MARGIN}" \
  "${BATCH_SIZE}" \
  "${BANK_SOFT_K}" \
  "${BANK_START_EPOCH}"
