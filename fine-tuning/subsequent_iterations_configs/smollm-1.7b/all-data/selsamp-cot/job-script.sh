#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=smollm17_selsamp_all_cot_iter2
#SBATCH --output=smollm17_selsamp_all_cot_iter2.out
#SBATCH --error=smollm17_selsamp_all_cot_iter2.err

python ../../../../../../lora_train_and_save.py epoch2.yaml