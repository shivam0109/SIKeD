#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:10:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g
#SBATCH --job-name=pot_asdiv
#SBATCH --output=pot_asdiv.out
#SBATCH --error=pot_asdiv.err

python ../../../inference_all_checkpoints.py asdiv.yaml
