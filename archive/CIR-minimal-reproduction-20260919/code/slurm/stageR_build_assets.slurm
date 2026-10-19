#!/bin/bash
#SBATCH -J stageR_build
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o slurm/stageR_build_%j.out
#SBATCH -e slurm/stageR_build_%j.err

set -euo pipefail

PROJECT_ROOT="/data/user/hqi881/CIR/project_stage1_dqu_baseline"
cd "$PROJECT_ROOT"

source /share/miniconda3/etc/profile.d/conda.sh
conda activate /data/user/hqi881/envs/dqu-stage1-cir

bash scripts/run_stageR_build_assets_hpc.sh
