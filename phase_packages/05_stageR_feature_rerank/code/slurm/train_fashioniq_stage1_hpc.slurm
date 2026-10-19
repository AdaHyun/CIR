#!/bin/bash
#SBATCH -p acd_u
#SBATCH --job-name=dqu_fiq_hpc
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH -o /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/%x_%j.out
#SBATCH -e /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/%x_%j.err
set -euo pipefail
CATEGORY="${CATEGORY:-dress}"
module load miniconda3
cd /data/user/hqi881/CIR/project_stage1_dqu_baseline
bash scripts/run_fashioniq_stage1_hpc.sh "${CATEGORY}"
