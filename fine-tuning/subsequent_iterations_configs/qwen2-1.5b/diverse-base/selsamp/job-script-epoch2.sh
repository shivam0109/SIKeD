#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=9:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=qwen2-1b_selsamp_diverse_iter2
#SBATCH --output=qwen2-1b_selsamp_diverse_iter2.out
#SBATCH --error=qwen2-1b_selsamp_diverse_iter2.err

python ../../../../../../lora_train_and_save.py epoch2.yaml