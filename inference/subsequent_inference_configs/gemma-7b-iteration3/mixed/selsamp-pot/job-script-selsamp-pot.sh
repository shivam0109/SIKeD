#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=gemma-7b_selsamp_mixed_pot
#SBATCH --output=gemma-7b_selsamp_mixed_pot.out
#SBATCH --error=gemma-7b_selsamp_mixed_pot.err

python ../../../../../../inference_all_chk_all_data.py selsamp-pot.yaml
