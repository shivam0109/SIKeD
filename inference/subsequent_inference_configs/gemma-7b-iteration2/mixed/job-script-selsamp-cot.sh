#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g
#SBATCH --job-name=gemma-7b_selsamp_mixed_cot
#SBATCH --output=gemma-7b_selsamp_mixed_cot.out
#SBATCH --error=gemma-7b_selsamp_mixed_cot.err

python ../../../../../inference_all_chk_all_data.py selsamp-cot.yaml
