#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=5:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=gemma7b-pot-annot-train
#SBATCH --output=gemma7b-pot-annot-train.out
#SBATCH --error=gemma7b-pot-annot-train.err

python ../../../../../forced_strategy_inference.py pot-train.yaml