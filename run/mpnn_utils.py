from itertools import chain
import os, glob, re
from typing import Dict
import numpy as np
from protein_mpnn.protein_mpnn_utils import StructureDatasetPDB, parse_PDB
from helper_scripts import make_fixed_positions_dict, make_tied_positions_dict

# Model config for storing hyperparameters
MODEL_CONFIG = {'hidden_dim': 128,
                'num_layers': 3}

# Chain letter alphabet
init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
extra_alphabet = [str(item) for item in list(np.arange(300))]
chain_alphabet = init_alphabet + extra_alphabet


def determine_weight_directory() -> str:
    """ Determines what directory the model weights lives in.

    Output:
        str: A string of the absolute path where the model weights live.
    """

    # Get name of parent directory of file and join with model_weights/
    file_path = os.path.realpath(__file__)
    k = file_path.rfind(os.sep)
    model_weight_dir = os.path.join(file_path[:k], 'model_weights')

    return model_weight_dir


def get_pdb_dataset(pdb_dir: str) -> StructureDatasetPDB:
    """ Parses pdb directory and returns a dataset containing coordinates
    of every parsed protein chain.

    Input:
        pdb_dir (str): A string of the path where all input pdb files are 
            contained.

    Output:
        StructureDatasetPDB: A dataset containing the parsed proteins and
            their backbone coordinates.
    """ 

    if os.path.isdir(pdb_dir):
        # Find all pdb files
        pdb_files = glob.glob(os.path.join(pdb_dir, '*.pdb'))
        if len(pdb_files) == 0:
            raise ValueError(f'No .pdb files detected in pdb_dir: {pdb_dir}')
        else:
            # Parse every pdb file and add parsed dict to overall list
            pdb_dict_list = []
            for pdb_file in pdb_files:
                pdb_dict_list += parse_PDB(pdb_file)

        # Construct dataset from pdb_dict_list
        dataset_valid = StructureDatasetPDB(pdb_dict_list, max_length=20000)
    else:
        raise ValueError(f'Could not find pdb_dir: {pdb_dir}')

    return dataset_valid


def get_pdb_dataset_func(pdb_list: list) -> StructureDatasetPDB:
    """ Takes list of pdb strings and returns a dataset containing coordinates
    of every parsed protein chain.

    Input:
        pdb_list (list): A list of all input pdb strings.

    Output:
        StructureDatasetPDB: A dataset containing the parsed proteins and
            their backbone coordinates.
    """
    if len(pdb_list) == 0:
        raise ValueError(f'No .pdb files detected in pdb_dir: {pdb_dir}')

    else:
        # Parse every pdb file and add parsed dict to overall list
        pdb_dict_list = []
        for pdb_file in pdb_files:
            pdb_dict_list += parse_PDB(pdb_file)

    # Construct dataset from pdb_dict_list
    dataset_valid = StructureDatasetPDB(pdb_dict_list, max_length=20000)

    return dataset_valid


def fixed_positions_args(design_spec_dict, protein):

    design_res = design_spec_dict['designable']
    design_positions = {}
    for res in design_res:
        if res['chain'] not in design_positions:
            design_positions[res['chain']] = []
        
        if res['resid'] not in design_positions[res['chain']]:
            design_positions[res['chain']].append(str(res['resid']))

    design_chains = list(design_positions.keys())
    all_chains = [item[-1:] for item in list(protein) if item[:9]=='seq_chain']
    chain_seqs = [protein[f'seq_chain_{letter}'] for letter in all_chains]
    chain_lengths = [len(seq) for seq in chain_seqs]
    chain_idxs = [list(range(1, length+1)) for length in chain_lengths]
    chain_idxs = [[str(idx) for idx in chain] for chain in chain_idxs]

    fixed_positions = {}
    for i, chain in enumerate(all_chains):
        # If not in design_chains then chain is entirely fixed
        if chain not in design_chains:
            fixed_positions[chain] = chain_idxs[i]
        # Else the chain is either partially fixed or not fixed at all
        else:
            # If design positions does not equal the full range of the chain indices, then some are fixed
            if design_positions[chain] != chain_idxs[i]:
                fixed_pos = set(chain_idxs[i]).difference(set(design_positions[chain]))
                fixed_pos = list(fixed_pos)
                fixed_pos.sort()
                fixed_positions[chain] = fixed_pos

    chain_list = list(fixed_positions.keys())
    position_list = list(fixed_positions.values())
    position_list = [' '.join(pos) for pos in position_list]

    return [protein], ', '.join(position_list), ' '.join(chain_list)


def tied_positions_args(design_spec_dict, protein):

    symmetric_res = design_spec_dict['symmetric']
    chain_positions = {}
    for res in symmetric_res:
        split_res = []
        for res_item in res:
            split_item = re.split('(\d+)', res_item)
            chain_id = split_item[0]
            res_idx = split_item[1]
            split_res.append( (chain_id, res_idx) )
        
        for res_item in split_res:
            if res_item[0] not in chain_positions:
                chain_positions[res_item[0]] = []
                
            if res_item[1] not in chain_positions[res_item[0]]:
                chain_positions[res_item[0]].append(res_item[1])

    position_list = [' '.join(chain_pos) for chain_pos in list(chain_positions.values())]

    return [protein], 0, ', '.join(position_list), ' '.join(list(chain_positions.keys()))


def transform_inputs(design_spec_dict: Dict[str, Dict[str, np.ndarray]], protein):
    
    # Loaded from chain_id_json
    # Additional example of design at line 144
    # Usage in Examples 2, 4, 5 (VAN + CA)
    # /helper_scripts/assign_fixed_chains.py
    # Note 1: Used --jsonl_path (as created by 
    #     /helper_scripts/parse_multiple_chains.py) as --input_path
    #     This is the same outputs found in the protein_dict acquired by 
    #     pdb_ds[i] for index i
    # Input Args:
    #     --input_path see Note 1
    #     --output_path doesn't matter
    #     --chain_list "A C"
    # After looking through and making some changes to assign_fixed_chains.py
    # I think I have deduced that this dictionary isn't necessary because 
    # everything should be handled by fixed_positions and tied_positions.
    chain_id_dict = None

    # Loaded from fixed_positions_json
    # Usage in Examples 4, 5, and 6 (no 6 for CA)
    # /helper_scripts/make_fixed_positions_dict.py
    # Note 1: Used --jsonl_path (as created by 
    #     /helper_scripts/parse_multiple_chains.py) as --input_path
    #     This is the same outputs found in the protein_dict acquired by 
    #     pdb_ds[i] for index i
    # Input Args:
    #     --input_path see Note 1
    #     --output_path doesn't matter 
    #     --chain_list "A C"
    #     --position_list "1 2 3 4 5 6 7 8, 1 2 3 4 5 6 7 8"
    fixed_positions_dict = make_fixed_positions_dict.main(*fixed_positions_args(design_spec_dict, protein))
    if fixed_positions_dict == {}:
        fixed_positions_dict = None
    #fixed_positions_dict = None

    # Loaded from tied_positions_json
    # Usage in Examples 5 and 6 (VAN + CA)
    # /helper_scripts/make_tied_positions_dict.py
    # Note 1: Used --jsonl_path (as created by 
    #     /helper_scripts/parse_multiple_chains.py) as --input_path
    #     This is the same outputs found in the protein_dict acquired by 
    #     pdb_ds[i] for index i
    # Input Args:
    #     --input_path see Note 1
    #     --output_path doesn't matter
    #     --chain_list "A C"
    #     --position_list "1 2 3 4 5 6 7 8, 1 2 3 4 5 6 7 8" 
    tied_positions_dict = make_tied_positions_dict.main(*tied_positions_args(design_spec_dict, protein))
    if tied_positions_dict == {}:
        tied_positions_dict = None

    # Loaded from omit_AA_json
    # From looking at tied_featurize, structure of the dictionary should be something like
    # omit_AA_dict[pdb_name] = {chain_letter: [(res_idx, omit_AAs), ...]}
    omit_AA_dict = {protein['name']: {}}
    for chain_letter in chain_alphabet:
        chain_seq = protein.get('seq_chain_' + chain_letter)
        if chain_seq != None:
            omit_AA_dict[protein['name']][chain_letter] = []
            for res in design_spec_dict['designable']:
                if res['chain'] == chain_letter:
                    # Determine which residues to omit
                    if res['MutTo'] != 'all':
                        if 'hydphob' in res['MutTo']:
                            # Exclude hydrophilics
                            omit_AAs = 'CDEHKNPQRSTX'
                        elif 'hydphil' in res['MutTo']:
                            # Exclude hydrophobics
                            omit_AAs = 'ACFGILMPVWYX'
                        elif 'all' in res['MutTo'] and '-' in res['MutTo'] and '+' not in res['MutTo']:
                            # Make list exclude no residues
                            omit_AAs = 'X'
                        elif set(res['MutTo']).issubset(set('ACDEFGHIKLMNPQRSTVWYX')):
                            # Omit everything but what is provided
                            omit_AAs = 'ACDEFGHIKLMNPQRSTVWYX'
                            for aa in list(res['MutTo']):
                                if aa in omit_AAs:
                                    omit_AAs = list(omit_AAs)
                                    omit_AAs.remove(aa)
                                    omit_AAs = ''.join(omit_AAs)
                        else:
                            # Provide a default of X to omit_AAs
                            omit_AAs = 'X'
                        
                        if '-' in res['MutTo']:
                            # Add to omit list
                            to_omit = res['MutTo'].split('-')[-1].split('+')[0]
                            omit_AAs += to_omit
                        if '+' in res['MutTo']:
                            # Remove from omit list
                            to_not_omit = res['MutTo'].split('+')[-1].split('-')[0]
                            for aa in to_not_omit:
                                if aa in omit_AAs:
                                    omit_AAs = list(omit_AAs)
                                    omit_AAs.remove(aa)
                                    omit_AAs = ''.join(omit_AAs)
                        omit_AA_dict[protein['name']][chain_letter].append([res['resid'], omit_AAs])

    # Loaded from pssm_json
    # There is no example of usage for this file. So I'm leaving it as
    # None for now.
    pssm_dict = None

    # Loaded from bias_AA_json
    # There is no example of usage for this file. So I'm leaving it as
    # None for now.
    bias_AA_dict = None

    # Loaded from bias_by_res_json
    # There is no example of usage for this file. So I'm leaving it as
    # None for now.
    bias_by_res_dict = None


    return chain_id_dict, fixed_positions_dict, pssm_dict, omit_AA_dict, bias_AA_dict, tied_positions_dict, bias_by_res_dict
