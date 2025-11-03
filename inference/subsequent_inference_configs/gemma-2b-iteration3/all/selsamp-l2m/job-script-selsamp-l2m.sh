#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=gemma-2b_selsamp_all_l2m
#SBATCH --output=gemma-2b_selsamp_all_l2m.out
#SBATCH --error=gemma-2b_selsamp_all_l2m.err

python ../../../../../../inference_all_chk_all_data.py selsamp-l2m.yaml
