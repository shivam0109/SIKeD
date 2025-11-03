#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=9:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=gemma-7b_selsamp_standard
#SBATCH --output=gemma-7b_selsamp_standard.out
#SBATCH --error=gemma-7b_selsamp_standard.err

python ../../../../../../inference_all_chk_all_data.py selsamp.yaml
