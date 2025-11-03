#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=gemma-7b_selsamp_mixed_l2m_iter4
#SBATCH --output=gemma-7b_selsamp_mixed_l2m_iter4.out
#SBATCH --error=gemma-7b_selsamp_mixed_l2m_iter4.err

python ../../../../../../lora_train_and_save.py epoch4.yaml