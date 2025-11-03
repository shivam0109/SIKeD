#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=5:50:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=gemma7b_gsm_selsamp
#SBATCH --output=gemma7b_gsm_selsamp.out
#SBATCH --error=gemma7b_gsm_selsamp.err

python ../../lora_train_and_save.py selsamp.yaml
