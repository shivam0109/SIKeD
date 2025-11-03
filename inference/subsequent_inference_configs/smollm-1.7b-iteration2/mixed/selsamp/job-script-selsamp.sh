#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=smollm1-7b_selsamp_mixed
#SBATCH --output=smollm1-7b_selsamp_mixed.out
#SBATCH --error=smollm1-7b_selsamp_mixed.err

python ../../../../../../inference_all_chk_all_data.py selsamp.yaml
