#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=qwen-1b_selsamp_mixed_l2m_iter3
#SBATCH --output=qwen-1b_selsamp_mixed_l2m_iter3.out
#SBATCH --error=qwen-1b_selsamp_mixed_l2m_iter3.err

python ../../../../../../lora_train_and_save.py epoch3.yaml