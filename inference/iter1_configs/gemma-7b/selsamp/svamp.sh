#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:10:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=selsamp_svamp
#SBATCH --output=selsamp_svamp.out
#SBATCH --error=selsamp_svamp.err

python ../../../inference_all_checkpoints.py svamp.yaml
