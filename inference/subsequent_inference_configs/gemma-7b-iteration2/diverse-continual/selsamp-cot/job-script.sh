#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:10:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=gemma-7b_selsamp-cot_diverse_continual
#SBATCH --output=gemma-7b_selsamp-cot_diverse_continual.out
#SBATCH --error=gemma-7b_selsamp-cot_diverse_continual.err

python ../../../../../../inference_all_chk_all_data.py selsamp-cot.yaml
