#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=gemma7b_selsamp_mixed_cot_iter4
#SBATCH --output=gemma7b_selsamp_mixed_cot_iter4.out
#SBATCH --error=gemma7b_selsamp_mixed_cot_iter4.err

python ../../../../../../lora_train_and_save.py epoch4.yaml