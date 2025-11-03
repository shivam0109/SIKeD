#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=10:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=gemma7b-cot-annot-train-all
#SBATCH --output=gemma7b-cot-annot-train-all.out
#SBATCH --error=gemma7b-cot-annot-train-all.err

python ../../../../../forced_strategy_inference.py cot-train.yaml