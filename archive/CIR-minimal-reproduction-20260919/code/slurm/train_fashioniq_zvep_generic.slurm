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
NUM_EPOCHS_OVERRIDE="${NUM_EPOCHS_OVERRIDE:-}"

module load miniconda3
cd /data/user/hqi881/CIR/project_stage1_dqu_baseline
mkdir -p /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm
echo "[$(date)] category=${CATEGORY} model_variant=${MODEL_VARIANT} run_id=${RUN_ID} model_dir=${MODEL_DIR} num_epochs=${NUM_EPOCHS_OVERRIDE:-base}"
bash scripts/run_fashioniq_zvep_hpc.sh \
  "${CATEGORY}" \
  "${MODEL_VARIANT}" \
  "${RUN_ID}" \
  "${MODEL_DIR}" \
  "${NUM_EPOCHS_OVERRIDE}"
