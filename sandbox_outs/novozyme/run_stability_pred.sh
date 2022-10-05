#!/bin/bash

source activate mpnn

python ../stability_pred.py --num_seq_per_target 500 --batch_size 2 --test_set_path test.csv \
--pdb_path wildtype_structure_prediction_af2.pdb --sort_by_rank --decoding_order mutant_last