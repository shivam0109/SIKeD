#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=smollm17_selsamp_all_l2m_iter3
#SBATCH --output=smollm17_selsamp_all_l2m_iter3.out
#SBATCH --error=smollm17_selsamp_all_l2m_iter3.err

python ../../../../../../lora_train_and_save.py epoch3.yaml