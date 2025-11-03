#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=0:29:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=pot-annot-test
#SBATCH --output=pot-annot-test.out
#SBATCH --error=pot-annot-test.err

python ../../../../../forced_strategy_inference.py pot-test.yaml