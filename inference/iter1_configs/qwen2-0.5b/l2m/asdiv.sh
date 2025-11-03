#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=l2m_asdiv
#SBATCH --output=l2m_asdiv.out
#SBATCH --error=l2m_asdiv.err

python ../../../inference_all_checkpoints.py asdiv.yaml
