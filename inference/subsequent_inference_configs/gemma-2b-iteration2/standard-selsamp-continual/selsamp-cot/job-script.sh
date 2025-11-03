#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=gemma-2b_selsamp-cot_standard-selsamp_continual
#SBATCH --output=gemma-2b_selsamp-cot_standard-selsamp_continual.out
#SBATCH --error=gemma-2b_selsamp-cot_standard-selsamp_continual.err

python ../../../../../../inference_all_chk_all_data.py selsamp-cot.yaml
