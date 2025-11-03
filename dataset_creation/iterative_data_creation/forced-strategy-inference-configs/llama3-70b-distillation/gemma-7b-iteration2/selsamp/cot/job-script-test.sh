#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=0:29:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=gemma7b-cot-annot-test
#SBATCH --output=gemma7b-cot-annot-test.out
#SBATCH --error=gemma7b-cot-annot-test.err

python ../../../../../forced_strategy_inference.py cot-test.yaml