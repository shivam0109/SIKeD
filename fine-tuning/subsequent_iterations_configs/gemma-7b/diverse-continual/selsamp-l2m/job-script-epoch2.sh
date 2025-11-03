#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=6:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=gemma7b_selsamp-l2m_diverse-continual_iter2
#SBATCH --output=gemma7b_selsamp-l2m_diverse-continual_iter2.out
#SBATCH --error=gemma7b_selsamp-l2m_diverse-continual_iter2.err

python ../../../../../../lora_train_and_save.py epoch2.yaml