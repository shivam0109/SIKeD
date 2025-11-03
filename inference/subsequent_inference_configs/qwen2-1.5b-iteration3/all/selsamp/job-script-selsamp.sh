#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=5:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=qwen1_selsamp_all
#SBATCH --output=qwen1_selsamp_all.out
#SBATCH --error=qwen1_selsamp_all.err

python ../../../../../../inference_all_chk_all_data.py selsamp.yaml
