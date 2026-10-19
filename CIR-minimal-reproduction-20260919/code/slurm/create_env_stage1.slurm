#!/bin/bash
#SBATCH -p acd_u
#SBATCH --job-name=dqu_env
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=02:00:00
#SBATCH -o /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/dqu_env_%j.out
#SBATCH -e /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/dqu_env_%j.err
set -euo pipefail
module load miniconda3
if [ -d /data/user/hqi881/envs/dqu-stage1-cir ]; then
 rm -rf /data/user/hqi881/envs/dqu-stage1-cir
fi
conda create -y -p /data/user/hqi881/envs/dqu-stage1-cir python=3.9.18 pip "pytorch=2.0.1=*cuda118*" "torchvision=0.15.2=*cuda118*"
conda run -p /data/user/hqi881/envs/dqu-stage1-cir pip install -r /data/user/hqi881/CIR/project_stage1_dqu_baseline/requirements.txt
conda run -p /data/user/hqi881/envs/dqu-stage1-cir python -c "import torch, open_clip; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
