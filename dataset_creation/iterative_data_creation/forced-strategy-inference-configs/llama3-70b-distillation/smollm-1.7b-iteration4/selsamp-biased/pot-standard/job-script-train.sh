#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=smollm-1-7b-pot-annot-train-standard
#SBATCH --output=smollm-1-7b-pot-annot-train-standard.out
#SBATCH --error=smollm-1-7b-pot-annot-train-standard.err

python ../../../../../forced_strategy_inference.py pot-train.yaml