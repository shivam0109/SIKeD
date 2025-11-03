#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=6:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=gemma-2b_selsamp_diverse_continual
#SBATCH --output=gemma-2b_selsamp_diverse_continual.out
#SBATCH --error=gemma-2b_selsamp_diverse_continual.err

python ../../../../../../inference_all_chk_all_data.py selsamp.yaml
