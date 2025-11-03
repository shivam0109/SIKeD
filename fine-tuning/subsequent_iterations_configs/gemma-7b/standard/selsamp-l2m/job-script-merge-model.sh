#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=0:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=merge_model_l2m
#SBATCH --output=merge_model_l2m.out
#SBATCH --error=merge_model_l2m.err

python ../../../../../merge_model_for_retrain.py /cluster/work/sachan/shivam/improving-prompting-strategies/model/knowledge_distillation/llama3-70b/lora/gemma-7b/selsamp-l2m/final_model
