#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --mem-per-cpu=32G
#SBATCH --time=17:59:59
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --job-name=ar_annot_pot
#SBATCH --output=ar_annot_pot.out
#SBATCH --error=ar_annot_pot.err


module load gcc/8.2.0
module load python_gpu/3.10.4
module load eth_proxy
python ../annotation.py pot.yaml