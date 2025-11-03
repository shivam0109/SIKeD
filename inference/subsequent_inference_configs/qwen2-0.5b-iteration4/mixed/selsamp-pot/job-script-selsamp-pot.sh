#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=qwenh_selsamp_mixed_pot
#SBATCH --output=qwenh_selsamp_mixed_pot.out
#SBATCH --error=qwenh_selsamp_mixed_pot.err

python ../../../../../../inference_all_chk_all_data.py selsamp-pot.yaml
