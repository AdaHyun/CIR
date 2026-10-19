#!/bin/bash
#SBATCH -p acd_u
#SBATCH --job-name=dqu_fiq_array
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH -o slurm/%x_%A_%a.out
#SBATCH -e slurm/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/user/$USER/project_stage1_dqu_baseline}"
ENV_PATH="${ENV_PATH:-/data/user/$USER/envs/dqu-stage1}"
CATEGORIES=(dress shirt toptee)
CATEGORY="${CATEGORIES[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "${REPO_ROOT}/slurm"
cd "${REPO_ROOT}"

module load miniconda3
conda activate "${ENV_PATH}"

export TOKENIZERS_PARALLELISM=false

bash scripts/run_fashioniq_stage1.sh "${CATEGORY}"
