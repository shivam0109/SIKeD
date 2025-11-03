#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:50:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=smollm1-7b_gsm_selsamp_pot
#SBATCH --output=smollm1-7b_gsm_selsamp_pot.out
#SBATCH --error=smollm1-7b_gsm_selsamp_pot.err

python ../../lora_train_and_save.py selsamp-pot.yaml
