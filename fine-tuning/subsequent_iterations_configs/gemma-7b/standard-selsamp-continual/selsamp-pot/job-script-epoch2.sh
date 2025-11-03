#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:10:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=gemma7b_selsamp_pot_continual_iter2
#SBATCH --output=gemma7b_selsamp_pot_continual_iter2.out
#SBATCH --error=gemma7b_selsamp_pot_continual_iter2.err

python ../../../../../../lora_train_and_save.py epoch2.yaml