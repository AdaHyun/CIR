#!/bin/bash
#SBATCH -p acd_u
#SBATCH --job-name=dqu_dress_smk3
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=00:40:00
#SBATCH -o /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/dqu_dress_smk3_%j.out
#SBATCH -e /data/user/hqi881/CIR/project_stage1_dqu_baseline/slurm/dqu_dress_smk3_%j.err
set -euo pipefail
module load miniconda3
cd /data/user/hqi881/CIR/project_stage1_dqu_baseline
conda run -p /data/user/hqi881/envs/dqu-stage1-cir python src/train.py --dataset dress --fashioniq_split val-split --fashioniq_path /data/user/hqi881/CIR/project_stage1_dqu_baseline/data/FashionIQ --lr 1e-4 --clip_lr 1e-6 --batch_size 8 --num_epochs 1 --num_workers 0 --backbone ViT-H-14 --pretrained /data/user/hqi881/CIR/model/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin --model_dir /data/user/hqi881/CIR/project_stage1_dqu_baseline/checkpoints/fashioniq/dress_smoke3 --i smoke3
