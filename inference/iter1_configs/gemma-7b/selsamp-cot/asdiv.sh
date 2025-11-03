#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:10:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g
#SBATCH --job-name=selsamp-cot_asdiv
#SBATCH --output=selsamp-cot_asdiv.out
#SBATCH --error=selsamp-cot_asdiv.err

python ../../../inference_all_checkpoints.py asdiv.yaml
