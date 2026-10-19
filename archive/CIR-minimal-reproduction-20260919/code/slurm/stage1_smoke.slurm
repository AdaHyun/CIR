#!/bin/bash
#SBATCH -p acd_u
#SBATCH --job-name=dqu_smoke
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=00:20:00
#SBATCH -o /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/dqu_smoke_%j.out
#SBATCH -e /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/dqu_smoke_%j.err
set -euo pipefail
module load miniconda3
cd /data/user/hqi881/CIR/project_stage1_dqu_baseline
conda run -p /data/user/hqi881/envs/dqu-stage1-cir python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
conda run -p /data/user/hqi881/envs/dqu-stage1-cir python -c "import open_clip; print(open_clip.__version__)"
conda run -p /data/user/hqi881/envs/dqu-stage1-cir python scripts/check_dataset.py --dataset all
