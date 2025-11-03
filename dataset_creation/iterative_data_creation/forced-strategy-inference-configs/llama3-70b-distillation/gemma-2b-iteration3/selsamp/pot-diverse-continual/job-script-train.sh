#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=gemma2b-pot-annot-train-diverse-continual
#SBATCH --output=gemma2b-pot-annot-train-diverse-continual.out
#SBATCH --error=gemma2b-pot-annot-train-diverse-continual.err

python ../../../../../forced_strategy_inference.py pot-train.yaml