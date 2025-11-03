#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=smollm-1-7b_selsamp-l2m_mixed_iter3
#SBATCH --output=smollm-1-7b_selsamp-l2m_mixed_iter3.out
#SBATCH --error=smollm-1-7b_selsamp-l2m_mixed_iter3.err

python ../../../../../../lora_train_and_save.py epoch3.yaml