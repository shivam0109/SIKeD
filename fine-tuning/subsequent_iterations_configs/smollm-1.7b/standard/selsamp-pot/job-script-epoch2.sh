#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=smollm-1-7b_selsamp-pot_standard_iter2
#SBATCH --output=smollm-1-7b_selsamp-pot_standard_iter2.out
#SBATCH --error=smollm-1-7b_selsamp-pot_standard_iter2.err

python ../../../../../../lora_train_and_save.py epoch2.yaml