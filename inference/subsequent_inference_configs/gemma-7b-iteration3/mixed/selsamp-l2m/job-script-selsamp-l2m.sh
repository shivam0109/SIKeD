#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=gemma-7b_selsamp_mixed_l2m
#SBATCH --output=gemma-7b_selsamp_mixed_l2m.out
#SBATCH --error=gemma-7b_selsamp_mixed_l2m.err

python ../../../../../../inference_all_chk_all_data.py selsamp-l2m.yaml
