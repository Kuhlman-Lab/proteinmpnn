#!/bin/bash

#SBATCH -J stability_pred_last
#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 02-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j-%x.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nzrandol@unc.edu

source ~/.bashrc
module add cuda/11.2
conda activate mpnn

python ../stability_pred.py --num_seq_per_target 10000 --batch_size 25 --test_set_path test.csv \
--pdb_path wildtype_structure_prediction_af2.pdb --sort_by_rank --decoding_order mutant_last
