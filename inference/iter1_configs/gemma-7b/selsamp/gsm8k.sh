#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:10:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=selsamp_gsm8k
#SBATCH --output=selsamp_gsm8k.out
#SBATCH --error=selsamp_gsm8k.err

python ../../../inference_all_checkpoints.py gsm8k.yaml
