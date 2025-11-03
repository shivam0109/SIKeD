#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=qwen2-hb_selsamp_all_l2m_iter2
#SBATCH --output=qwen2-hb_selsamp_all_l2m_iter2.out
#SBATCH --error=qwen2-hb_selsamp_all_l2m_iter2.err

python ../../../../../../lora_train_and_save.py epoch2.yaml