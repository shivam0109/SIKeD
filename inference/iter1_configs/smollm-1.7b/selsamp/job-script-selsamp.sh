#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=9:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=smollm1-7b_selsamp
#SBATCH --output=smollm1-7b_selsamp.out
#SBATCH --error=smollm1-7b_selsamp.err

python ../../../inference_all_chk_all_data.py selsamp.yaml
