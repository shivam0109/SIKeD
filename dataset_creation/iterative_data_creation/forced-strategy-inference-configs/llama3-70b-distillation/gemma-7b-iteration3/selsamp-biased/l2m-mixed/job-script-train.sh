#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=10:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=gemma7b-l2m-annot-train-mixed
#SBATCH --output=gemma7b-l2m-annot-train-mixed.out
#SBATCH --error=gemma7b-l2m-annot-train-mixed.err

python ../../../../../forced_strategy_inference.py l2m-train.yaml