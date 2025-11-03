#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:10:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g
#SBATCH --job-name=selsamp-pot_asdiv
#SBATCH --output=selsamp-pot_asdiv.out
#SBATCH --error=selsamp-pot_asdiv.err

python ../../../../../../inference_all_checkpoints.py asdiv.yaml
