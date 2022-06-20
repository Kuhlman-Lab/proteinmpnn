#!/bin/bash

#source ~/.bashrc
#conda activate mpnn

#python /proj/kuhl_lab/proteinmpnn/run/generate_json.py @json.flags
python ../../../generate_json.py @json.flags
#python /proj/kuhl_lab/proteinmpnn/run/run_protein_mpnn.py @proteinmpnn_res_specs.json
python ../../../run_protein_mpnn.py @proteinmpnn.flags
