#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:50:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=gemma2b_gsm_selsamp_pot
#SBATCH --output=gemma2b_gsm_selsamp_pot.out
#SBATCH --error=gemma2b_gsm_selsamp_pot.err

python ../../lora_train_and_save.py selsamp-pot.yaml
