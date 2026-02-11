#!/bin/bash

#SBATCH -p h100_sn
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 00-00:30:00
#SBATCH --qos h100_sn
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate mpnn_cu2.4

python ../../../run/generate_json.py @json.flags

python ../../../run/run_protein_mpnn.py @proteinmpnn.flags
