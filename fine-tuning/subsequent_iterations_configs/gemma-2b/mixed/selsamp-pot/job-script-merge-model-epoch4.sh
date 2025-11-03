#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=0:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=merge_model
#SBATCH --output=merge_model.out
#SBATCH --error=merge_model.err

python ../../../../../merge_model_for_retrain.py /cluster/work/sachan/shivam/improving-prompting-strategies/model/knowledge_distillation/llama3-70b/lora/action-reward/iteration3/gemma-2b/mixed/selsamp-pot/checkpoint-100
