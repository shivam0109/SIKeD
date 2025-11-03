#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=qwen215-l2m-annot-train-all
#SBATCH --output=qwen215-l2m-annot-train-all.out
#SBATCH --error=qwen215-l2m-annot-train-all.err

python ../../../../../forced_strategy_inference.py l2m-train.yaml