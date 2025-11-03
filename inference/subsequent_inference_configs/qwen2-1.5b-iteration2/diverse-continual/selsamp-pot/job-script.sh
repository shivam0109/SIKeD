#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=3:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=qwen215_selsamp-pot_diverse_continual
#SBATCH --output=qwen215_selsamp-pot_diverse_continual.out
#SBATCH --error=qwen215_selsamp-pot_diverse_continual.err

python ../../../../../../inference_all_chk_all_data.py selsamp-pot.yaml
