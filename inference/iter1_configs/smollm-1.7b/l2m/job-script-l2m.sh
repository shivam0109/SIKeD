#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=smollm1-7b_l2m
#SBATCH --output=smollm1-7b_l2m.out
#SBATCH --error=smollm1-7b_l2m.err

python ../../../inference_all_chk_all_data.py l2m.yaml
