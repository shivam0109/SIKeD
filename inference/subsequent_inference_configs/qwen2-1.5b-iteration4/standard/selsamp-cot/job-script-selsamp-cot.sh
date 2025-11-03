#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:58:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=qwen215_selsamp_cot
#SBATCH --output=qwen215_selsamp_cot.out
#SBATCH --error=qwen215_selsamp_cot.err

python ../../../../../../inference_all_chk_all_data.py selsamp-cot.yaml
