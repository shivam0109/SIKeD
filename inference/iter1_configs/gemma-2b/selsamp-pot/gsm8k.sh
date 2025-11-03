#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=selsamp-pot_gsm8k
#SBATCH --output=selsamp-pot_gsm8k.out
#SBATCH --error=selsamp-pot_gsm8k.err

python ../../../inference_all_checkpoints.py gsm8k.yaml
