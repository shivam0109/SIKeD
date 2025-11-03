#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:10:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --job-name=gemma-2b_selsamp-l2m_diverse_base
#SBATCH --output=gemma-2b_selsamp-l2m_diverse_base.out
#SBATCH --error=gemma-2b_selsamp-l2m_diverse_base.err

python ../../../../../../inference_all_chk_all_data.py selsamp-l2m.yaml
