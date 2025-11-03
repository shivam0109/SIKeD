#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --mem-per-cpu=32G
#SBATCH --time=17:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=ar_annot_l2m
#SBATCH --output=ar_annot_l2m.out
#SBATCH --error=ar_annot_l2m.err


module load gcc/8.2.0
module load python_gpu/3.10.4
module load eth_proxy
python ../annotation.py l2m.yaml