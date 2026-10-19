#!/bin/bash
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=08:00:00
#SBATCH -o /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/%x_%j.out
#SBATCH -e /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/%x_%j.err
set -euo pipefail

CATEGORY="${CATEGORY:?CATEGORY required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH required}"
MAX_QUERIES="${MAX_QUERIES:-0}"

module load miniconda3
cd /data/user/hqi881/CIR/project_stage1_dqu_baseline
mkdir -p /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm
echo "[$(date)] build bank category=${CATEGORY} output=${OUTPUT_PATH} max_queries=${MAX_QUERIES}"
bash scripts/run_build_multiteacher_bank_hpc.sh "${CATEGORY}" "${OUTPUT_PATH}" "${MAX_QUERIES}"
