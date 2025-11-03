#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:10:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g
#SBATCH --job-name=selsamp-l2m_asdiv
#SBATCH --output=selsamp-l2m_asdiv.out
#SBATCH --error=selsamp-l2m_asdiv.err

python ../../../../../../inference_all_checkpoints.py asdiv.yaml
