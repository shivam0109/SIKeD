#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=cot_svamp
#SBATCH --output=cot_svamp.out
#SBATCH --error=cot_svamp.err

python ../../../inference_all_checkpoints.py svamp.yaml
