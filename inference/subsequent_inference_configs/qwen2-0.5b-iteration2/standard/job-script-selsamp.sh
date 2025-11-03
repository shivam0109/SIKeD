#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=5:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=qwenh_selsamp
#SBATCH --output=qwenh_selsamp.out
#SBATCH --error=qwenh_selsamp.err

python ../../../../../inference_all_chk_all_data.py selsamp.yaml
