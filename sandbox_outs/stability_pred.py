import argparse
import time, os
import numpy as np
import torch
import copy
from protein_mpnn_utils import _scores, tied_featurize, parse_PDB
from protein_mpnn_utils import StructureDatasetPDB, ProteinMPNN

def determine_weight_directory() -> str:
    longleaf = 'longleaf' in os.path.expanduser('~')

    if longleaf:
        weight_path = '/proj/kuhl_lab/alphafold/alphafold/data/'
    else:
        weight_path = '/home/nzrandolph/git/ProteinMPNN/run/model_weights/'

    return weight_path


def place_mutant_order(randn, mut_idx, last=False):

    # Create random order
    decoding_order = torch.argsort(torch.abs(randn))

    # Determine which order to look for
    order = randn.shape[-1] - 1 if last else 0

    # Update current order res to mutant's index
    decoding_order[:, torch.argmax((decoding_order == order).int(), dim=-1)] = decoding_order[:, [mut_idx] * randn.shape[0]]

    # Update mutant to order res
    decoding_order[:, [mut_idx] * randn.shape[0]] = order

    return decoding_order


def predict_stability(model, protein, chain_id_dict, batch_copies, num_batches, device, decoding_order=None):

    # Form batch of clones
    batch_clones = [copy.deepcopy(protein) for _ in range(batch_copies)]

    # Featurize
    X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(batch_clones, device, chain_id_dict, *([None] * 5))

    # Loop over number of batches
    scores = np.array([])
    for _ in range(num_batches):
        
        # Create decoding order
        randn = torch.randn(chain_M.shape, device=device)
        if decoding_order != None:
            decoding_order = place_mutant_order(randn, protein['mut_idx'], last=decoding_order == 'mutant_last')

        # Run model
        log_probs = model(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn, use_input_decoding_order=decoding_order != None, decoding_order=decoding_order)
        
        # Compute score
        mask_for_loss = mask * chain_M * chain_M_pos
        score = _scores(S, log_probs, mask_for_loss).cpu().data.numpy()
        
        # Accumulate
        scores = np.concatenate((scores, score))

    return np.mean(scores), np.std(scores)


def stability_prediction(args):

    # Construct the checkpoint path
    if args.path_to_model_weights:
        model_folder_path = args.path_to_model_weights
    else: 
        model_folder_path = determine_weight_directory()
    if model_folder_path[-1] != '/':
        model_folder_path = model_folder_path + '/'
    checkpoint_path = model_folder_path + f'{args.model_name}.pt'

    # Determine number of batches to run
    BATCH_COPIES = args.batch_size
    NUM_BATCHES = args.num_seq_per_target // BATCH_COPIES

    # Default parameters and other variables for ProteinMPNN
    hidden_dim = 128
    num_layers = 3
    max_length = 20000
    backbone_noise = 0.00

    # Determine which device to run on
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Parse the input PDB to obtain structure and sequence
    pdb_dict_list = parse_PDB(args.pdb_path)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)
    wt_seq = pdb_dict_list[0]['seq']

    # Construct chain_id_dict with all chains "designable"
    all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9] == 'seq_chain']
    chain_id_dict = {}
    chain_id_dict[pdb_dict_list[0]['name']] = (all_chain_list, [])

    if args.test_set_path:
        # Read the sequence test set
        test_set_file = args.test_set_path
        with open(test_set_file, 'r') as f:
            test_set_list = f.readlines()

        # Parse the test set and obtain sequences and ids
        test_set = [mut.split(',')[:2] for mut in test_set_list[1:]]
        test_set_seqs = [mut[1] for mut in test_set]
        
        # Insert gaps where a deletion is and record mutant location
        test_set_mut_idx = []
        for i, seq in enumerate(test_set_seqs):
            if len(seq) < len(wt_seq):
                k = next((idx for idx, res in enumerate(seq) if res != wt_seq[idx]), None)
                test_set_seqs[i] = seq[:k] + '-' + seq[k:]
            test_set_mut_idx.append(next((idx for idx, res in enumerate(seq) if res != wt_seq[idx]), None))
    else:
        test_set_seqs = []

    # Construct the model from saved checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Construct output folder
    base_folder = args.out_folder
    if base_folder[-1] != '/':
        base_folder = base_folder + '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    with torch.no_grad():

        for protein in dataset_valid:

            # Log and time
            print(f"Predicting stability for {protein['name']}...")
            time_start = time.time()

            # Obtain native sequence score for comparision
            native_score, native_std = predict_stability(model, protein, chain_id_dict, BATCH_COPIES, NUM_BATCHES, device)

            # Write native outputs
            out_path = base_folder + f"{protein['name']}_stability.csv"
            if args.decoding_order == 'mutant_first':
                out_path = base_folder + f"{protein['name']}_stability_first.csv"
            elif args.decoding_order == 'mutant_last':
                out_path = base_folder + f"{protein['name']}_stability_last.csv"
            with open(out_path, 'w') as csv_file:
                csv_file.write('ID,SEQ,SCORE.MEAN,SCORE.STD,DELTA.MEAN,STABILITY.MEAN,RANKING,+/-\n')
                csv_file.write(f"native,{protein['seq']},{native_score:.5f},{native_std:.5f},0,NA,NA,NA\n")

            out_list = []
            for i in range(len(test_set_seqs[:2])):

                # Update the sequence with the mutated sequence
                protein['seq_chain_A'] = test_set_seqs[i]
                protein['seq'] = test_set_seqs[i]
                protein['mut_idx'] = test_set_mut_idx[i]

                # Score the mutant sequence
                mutant_score, mutant_std = predict_stability(model, protein, chain_id_dict, BATCH_COPIES, NUM_BATCHES, device, args.decoding_order)

                # Append to output list
                out_list.append((test_set[i][0], test_set_seqs[i], mutant_score, mutant_std))

            # Reorder based on rank
            ranked_out_list = sorted(out_list, key=lambda x: x[2] - native_score, reverse=not args.descending_Tm)

            # Append to outfile
            with open(out_path, 'a') as csv_file:
                file_out_list = ranked_out_list if args.sort_by_rank else out_list
                for i in range(len(file_out_list)):
                    ranking = i if args.sort_by_rank else ranked_out_list.index(out_list[i])
                    delta = file_out_list[i][2] - native_score
                    csv_file.write(f"{file_out_list[i][0]},{file_out_list[i][1]},{file_out_list[i][2]:.5f},{file_out_list[i][3]},{delta:.5f},{-1 * delta:.5f},{ranking},{'+' if delta < 0.0 else '-'}\n")

            # Print timing
            print(f'Finished prediction. Elapsed time = {time.time() - time_start:.3f}.')


if __name__ == '__main__':
    
    # Construct parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_model_weights", type=str, default="", help="Path to model weights folder;") 
    parser.add_argument("--model_name", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")
    parser.add_argument("--num_seq_per_target", type=int, default=1, help="Number of sequences to generate per target")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    parser.add_argument("--decoding_order", type=str, default=None, choices=['mutant_first', 'mutant_last'])
    parser.add_argument("--test_set_path", type=str, default='', help="Path csv containing mutant sequences to rank")
    parser.add_argument("--out_folder", type=str, default='.', help="Path to a folder to output sequences, e.g. /home/out/")
    
    parser.add_argument("--pdb_path", type=str, help="Path to a single PDB to be designed")
    parser.add_argument("--descending_Tm", action='store_true')
    parser.add_argument('--sort_by_rank', action='store_true')

    # Get arguments
    args = parser.parse_args()

    # Predict
    stability_prediction(args)


