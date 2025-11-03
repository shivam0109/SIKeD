#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:50:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=gemma7b_gsm_pot
#SBATCH --output=gemma7b_gsm_pot.out
#SBATCH --error=gemma7b_gsm_pot.err

python ../../lora_train_and_save.py pot.yaml
