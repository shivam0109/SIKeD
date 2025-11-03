#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:10:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=selsamp_multiarith
#SBATCH --output=selsamp_multiarith.out
#SBATCH --error=selsamp_multiarith.err

python ../../../inference_all_checkpoints.py multiarith.yaml
