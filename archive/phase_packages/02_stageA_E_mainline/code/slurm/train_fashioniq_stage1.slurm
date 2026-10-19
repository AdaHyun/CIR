#!/bin/bash
#SBATCH -p acd_u
#SBATCH --job-name=dqu_fiq
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH -o slurm/%x_%j.out
#SBATCH -e slurm/%x_%j.err

set -euo pipefail

CATEGORY="${CATEGORY:-dress}"
REPO_ROOT="${REPO_ROOT:-/data/user/$USER/project_stage1_dqu_baseline}"
ENV_PATH="${ENV_PATH:-/data/user/$USER/envs/dqu-stage1}"

mkdir -p "${REPO_ROOT}/slurm"
cd "${REPO_ROOT}"

module load miniconda3
conda activate "${ENV_PATH}"

export TOKENIZERS_PARALLELISM=false

bash scripts/run_fashioniq_stage1.sh "${CATEGORY}"
