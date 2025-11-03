#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=0:29:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=gemma7b-l2m-annot-test
#SBATCH --output=gemma7b-l2m-annot-test.out
#SBATCH --error=gemma7b-l2m-annot-test.err

python ../../../../../forced_strategy_inference.py l2m-test.yaml