#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:50:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=qwen_gsm_selsamp
#SBATCH --output=qwen_gsm_selsamp.out
#SBATCH --error=qwen_gsm_selsamp.err

python ../../lora_train_and_save.py selsamp.yaml
