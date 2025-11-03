#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=qwen215-pot-annot-train-standard-selsamp-continual
#SBATCH --output=qwen215-pot-annot-train-standard-selsamp-continual.out
#SBATCH --error=qwen215-pot-annot-train-standard-selsamp-continual.err

python ../../../../../forced_strategy_inference.py pot-train.yaml