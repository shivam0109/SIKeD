#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=pot_multiarith
#SBATCH --output=pot_multiarith.out
#SBATCH --error=pot_multiarith.err

python ../../../inference_all_checkpoints.py multiarith.yaml
