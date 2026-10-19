#!/bin/bash
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH -o /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/%x_%j.out
#SBATCH -e /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/%x_%j.err
set -euo pipefail

CATEGORY="${CATEGORY:?CATEGORY required}"
MODEL_VARIANT="${MODEL_VARIANT:?MODEL_VARIANT required}"
RUN_ID="${RUN_ID:?RUN_ID required}"
MODEL_DIR="${MODEL_DIR:?MODEL_DIR required}"
JOB_NAME_SAFE="${JOB_NAME_SAFE:-fiq_run}"
PSEUDO_PATH="${PSEUDO_PATH:-}"
PSEUDO_START_EPOCH="${PSEUDO_START_EPOCH:-0}"
PSEUDO_MIN_SCORE="${PSEUDO_MIN_SCORE:-0.0}"
PSEUDO_LABEL="${PSEUDO_LABEL:-none}"
PSEUDO_REPEAT="${PSEUDO_REPEAT:-1}"

module load miniconda3
cd /data/user/hqi881/CIR/project_stage1_dqu_baseline
echo "[$(date)] category=${CATEGORY} model_variant=${MODEL_VARIANT} run_id=${RUN_ID} model_dir=${MODEL_DIR} pseudo_path=${PSEUDO_PATH} pseudo_label=${PSEUDO_LABEL} pseudo_repeat=${PSEUDO_REPEAT}"
bash scripts/run_fashioniq_experiment_hpc.sh \
  "${CATEGORY}" \
  "${MODEL_VARIANT}" \
  "${RUN_ID}" \
  "${MODEL_DIR}" \
  "${PSEUDO_PATH}" \
  "${PSEUDO_START_EPOCH}" \
  "${PSEUDO_MIN_SCORE}" \
  "${PSEUDO_LABEL}" \
  "${PSEUDO_REPEAT}"
