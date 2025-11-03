#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=gemma2b_selsamp_all_cot_iter3
#SBATCH --output=gemma2b_selsamp_all_cot_iter3.out
#SBATCH --error=gemma2b_selsamp_all_cot_iter3.err

python ../../../../../../lora_train_and_save.py epoch3.yaml