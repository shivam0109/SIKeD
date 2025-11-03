#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:10:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=selsamp-pot_multiarith
#SBATCH --output=selsamp-pot_multiarith.out
#SBATCH --error=selsamp-pot_multiarith.err

python ../../../../../../inference_all_checkpoints.py multiarith.yaml
