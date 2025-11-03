#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:10:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=selsamp-l2m_multiarith_mixed
#SBATCH --output=selsamp-l2m_multiarith_mixed.out
#SBATCH --error=selsamp-l2m_multiarith_mixed.err

python ../../../../../../inference_all_checkpoints.py multiarith.yaml
