#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=qwen-hb_selsamp_mixed_pot_iter4
#SBATCH --output=qwen-hb_selsamp_mixed_pot_iter4.out
#SBATCH --error=qwen-hb_selsamp_mixed_pot_iter4.err

python ../../../../../../lora_train_and_save.py epoch4.yaml