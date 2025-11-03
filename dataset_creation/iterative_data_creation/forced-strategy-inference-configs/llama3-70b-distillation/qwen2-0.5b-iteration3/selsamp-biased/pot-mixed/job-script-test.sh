#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=0:29:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=qwen205-pot-annot-test-mixed
#SBATCH --output=qwen205-pot-annot-test-mixed.out
#SBATCH --error=qwen205-pot-annot-test-mixed.err

python ../../../../../forced_strategy_inference.py pot-test.yaml