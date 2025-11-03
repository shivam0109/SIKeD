#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=0:29:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=pot-annot-train
#SBATCH --output=pot-annot-train.out
#SBATCH --error=pot-annot-train.err

python ../../../../../forced_strategy_inference.py pot-train.yaml