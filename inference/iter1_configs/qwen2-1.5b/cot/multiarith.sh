#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=cot_multiarith
#SBATCH --output=cot_multiarith.out
#SBATCH --error=cot_multiarith.err

python ../../../inference_all_checkpoints.py multiarith.yaml
