#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=5:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=gemma-2b_selsamp_mixed
#SBATCH --output=gemma-2b_selsamp_mixed.out
#SBATCH --error=gemma-2b_selsamp_mixed.err

python ../../../../../../inference_all_chk_all_data.py selsamp.yaml
