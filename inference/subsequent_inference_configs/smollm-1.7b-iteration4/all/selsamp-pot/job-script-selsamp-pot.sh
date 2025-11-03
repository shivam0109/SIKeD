#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=smollm1-7b_selsamp-pot_all
#SBATCH --output=smollm1-7b_selsamp-pot_all.out
#SBATCH --error=smollm1-7b_selsamp-pot_all.err

python ../../../../../../inference_all_chk_all_data.py selsamp-pot.yaml
