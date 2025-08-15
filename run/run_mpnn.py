import os
import numpy as np
import time
import torch
import json
import copy
from protein_mpnn.protein_mpnn_utils import ProteinMPNN, tied_featurize, _scores, _S_to_seq
from mpnn_utils import determine_weight_directory, MODEL_CONFIG, MODEL_NAMES, get_pdb_dataset, transform_inputs
from generate_json import FileArgumentParser

def parse_args_from_file(mpnn_flags_file, parser):
        
    try:
        # Prepend @ to the filename to tell argparse this is an arguments file
        args = parser.parse_args(['@' + mpnn_flags_file])
        return args
    except FileNotFoundError:
        raise FileNotFoundError(f"Arguments file not found: {mpnn_flags_file}")
    except Exception as e:
        raise ValueError(f"Error parsing arguments file: {str(e)}")   
    
def get_arguments(mpnn_flags_file): 
    parser = FileArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument("--model_name", 
                        type=str, default="v_48_020",
                        choices=MODEL_NAMES,
                        help="ProteinMPNN model name. E.g. "
                        "v_48_010 = vanilla model with 48 edges 0.10A noise")
    parser.add_argument("--backbone_noise", type=float, default=0.00, help="Standard deviation of Gaussian noise to add to backbone atoms")
    parser.add_argument("--num_seq_per_target", type=int, default=1, help="Number of sequences to generate per target")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    parser.add_argument("--sampling_temp", type=str, default="0.1", help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.")
    parser.add_argument("--destabilize", action="store_true", help="Include to invert aa probabilities by making less favored amino acids more common.")
    
    parser.add_argument("--out_folder", type=str, help="Path to a folder to output sequences, e.g. /home/out/")
    parser.add_argument("--pdb_dir", type=str, default='', help="Path to a single PDB to be designed")
    parser.add_argument("--design_specs_json", type=str, help="Path to a folder with parsed pdb into jsonl")
    parser.add_argument("--af2_formatted_output", action='store_true', help="Whether or not to include another output file that is in AF2 format for direct structure prediction after design.")
    parser.add_argument("--global_omit_AAs", type=str, default='X', help='AAs to globally omit from all designable positions (e.g. PC for no proline or cysteine). Note that it is generally advisable to include X in this list. Default is X.')
    parser.add_argument('--experimental', action='store_true', help='Enables experimental parsing of allowable mutations')
    parser.add_argument('--bidirectional', action='store_true', help="Enable bidirectional coding constraints. Default is off.")
    parser.add_argument('--bias_AA_dict', default=None, type=str, help='Path to json file containing global amino acid bias dictionary for MPNN. Default is None.')
    parser.add_argument('--bias_by_res_dict', default=None, type=str, help='Path to json file containing per residue bias dictionary for MPNN. Default is None.')
    parser.add_argument('--pairwise', action='store_true', help='Enables parsing for experimental pairwise mutation clusters (experimental).')
    parser.add_argument('--dump_all_probs', action='store_true', help='If enabled, a file (probs.pkl) containing the aa probabilities at each position is saved.')
    parser.add_argument('--mcmc', action='store_true', help='If enabled, bidirectional coding uses MCMC routine. Must have --bidirectional flag enabled.')
    args = parse_args_from_file(mpnn_flags_file, parser)
    
    return args

def main(mpnn_flags_file, design_run=True, json_data=None, pdb_paths=None):
    
    #pdb_dir, design_specs_json, model_name="v_48_020", backbone_noise=0.00, 
    # num_seq_per_target=1, batch_size=1, sampling_temp="0.1", af2_formatted_output=False, 
    # bidir=False, bias_AA_dict=None, bias_by_res_dict=None, dump_probs=False
    
    args = get_arguments(mpnn_flags_file)
    #create directory to write pdb files
    pdb_dir = "MPNN_pdbs"
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)
    
    for p in pdb_paths:
        if os.path.exists(p):
            os.system(f"cp {p} {pdb_dir}")
    
    # Extract hyperparameters from model config
    hidden_dim = MODEL_CONFIG['hidden_dim']
    num_layers = MODEL_CONFIG['num_layers']

    # Determine path of the model weights
    model_weight_dir = determine_weight_directory()
    ckpt_path = os.path.join(model_weight_dir, f'{args.model_name}.pt')
    if not os.path.isfile(ckpt_path):
        raise ValueError('Invalid model name provided. Choose ca_48_002, ca_48_020, v_48_002, '
                         'v_48_010, v_48_020, v_48_030, s_48_010, or s_48_020.')

    # NOT SURE WHAT THESE DO
    NUM_BATCHES = args.num_seq_per_target // args.batch_size
    BATCH_COPIES = args.batch_size

    # Construct list of temperatures
    temperatures = [float(item) for item in args.sampling_temp.split()]

    # Default is 'X'
    omit_AAs_list = 'X'
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32) # 1 if omitting, 0 if not

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Parse the PDB directory and get dataset of the proteins
    pdb_ds = get_pdb_dataset(pdb_dir)

    #amino acid bias
    bias_AAs_np = np.zeros(len(alphabet))
    if args.bias_AA_dict:
        for n, AA in enumerate(alphabet):
            if AA in list(args.bias_AA_dict.keys()):
                bias_AAs_np[n] = args.bias_AA_dict[AA]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    num_edges = ckpt['num_edges']
    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim,
                        num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=args.backbone_noise,
                        k_neighbors=num_edges)
    model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Inference
    with torch.no_grad():
        outputs = []
        for ix, protein in enumerate(pdb_ds):
            output = []
            design_specs_dict = json_data
            score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            chain_id_dict, fixed_positions_dict, pssm_dict, omit_AA_dict, bias_AA_dict_decoy, tied_positions_dict, bias_by_res_dict_decoy = transform_inputs(design_specs_dict, protein)
            print('=' * 50)
            print(chain_id_dict, '\n', fixed_positions_dict, '\n', pssm_dict, '\n',
                  omit_AA_dict, '\n', tied_positions_dict, '\n', args.bias_by_res_dict)
            print('=' * 50)
            
            # quit()
            
            X, S, mask, _, chain_M, chain_encoding_all, _, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, _, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, args.bias_by_res_dict)

            # Setting pssm threshold to 0 for now. TODO: CHANGE LATER
            pssm_threshold = 0.0
            pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float()
            name_ = batch_clones[0]['name']

            # Do the inference
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            mask_for_loss = mask*chain_M*chain_M_pos
            scores, _ = _scores(S, log_probs, mask_for_loss)
            native_score = scores.cpu().data.numpy()

            # Generate some sequences
            print(f'Generating sequences for: {name_}')
            if args.bidirectional:
                print(f'Bidirectional coding requested for: {name_}')
            t0 = time.time()
            for temp in temperatures:
                for j in range(NUM_BATCHES):
                    randn_2 = torch.randn(chain_M.shape, device=X.device)
                    # pssm flags set to 0
                    pssm_multi = 0.0
                    pssm_log_odds_flag = 0
                    pssm_bias_flag = 0
                    if tied_positions_dict == None:
                        sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, 
                                                   residue_idx, mask=mask, temerature=temp, 
                                                   omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, 
                                                   chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, 
                                                   pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, 
                                                   pssm_log_odds_flag=bool(pssm_log_odds_flag), 
                                                   pssm_log_odds_mask=pssm_log_odds_mask, 
                                                   pssm_bias_flag=bool(pssm_bias_flag), bias_by_res=bias_by_res_all)
                    elif args.mcmc: # MCMC based bidirectional sampling
                        sample_dict = model.mcmc_sample(X, mask, residue_idx, chain_encoding_all, temperature=temp)        
                    else:
                        sample_dict = model.tied_sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, 
                                                        temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, 
                                                        chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, 
                                                        pssm_bias=pssm_bias, pssm_multi=pssm_multi, 
                                                        pssm_log_odds_flag=bool(pssm_log_odds_flag), 
                                                        pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), 
                                                        tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta, 
                                                        bias_by_res=bias_by_res_all, bidir=args.bidirectional, bidir_table_dir=model_weight_dir)
                    S_sample = sample_dict["S"]
                    log_probs = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
                    mask_for_loss = mask*chain_M*chain_M_pos
                    scores, _ = _scores(S_sample, log_probs, mask_for_loss)
                    scores = scores.cpu().data.numpy()
                    all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                    all_log_probs_list.append(log_probs.cpu().data.numpy())
                    S_sample_list.append(S_sample.cpu().data.numpy())
                    for b_ix in range(BATCH_COPIES):
                        masked_chain_length_list = masked_chain_length_list_list[b_ix]
                        masked_list = masked_list_list[b_ix]
                        seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21), axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
                        seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                        score = scores[b_ix]
                        score_list.append(score)
                        native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                        if b_ix == 0 and j==0 and temp==temperatures[0]:
                            start = 0
                            end = 0
                            list_of_AAs = []
                            for mask_l in masked_chain_length_list:
                                end += mask_l
                                list_of_AAs.append(native_seq[start:end])
                                start = end
                            native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                            l0 = 0
                            for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                l0 += mc_length
                                native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                                l0 += 1
                            sorted_masked_chain_letters = np.argsort(masked_list_list[0])
                            print_masked_chains = [masked_list_list[0][i] for i in sorted_masked_chain_letters]
                            sorted_visible_chain_letters = np.argsort(visible_list_list[0])
                            print_visible_chains = [visible_list_list[0][i] for i in sorted_visible_chain_letters]
                            native_score_print = np.format_float_positional(np.float32(native_score.mean()), unique=False, precision=4)
                            output.append((name_, native_score_print, print_visible_chains, print_masked_chains, args.model_name, native_seq))
                        start = 0
                        end = 0
                        list_of_AAs = []
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(seq[start:end])
                            start = end

                        seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                        l0 = 0
                        for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                            l0 += mc_length
                            seq = seq[:l0] + '/' + seq[l0:]
                            l0 += 1
                        score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
                        seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4)
                        output.append((temp,b_ix,score_print,seq_rec_print,seq))

            outputs.append(output)
            '''
            if dump_probs:
                all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                with open('probs.pkl', 'wb') as temp_file:
                    pickle.dump(all_probs_list, temp_file)
            '''
        t1 = time.time()
        dt = round(float(t1-t0), 4)
        num_seqs = len(temperatures)*NUM_BATCHES*BATCH_COPIES
        total_length = X.shape[1]
        print(f'{num_seqs} sequences of length {total_length} generated in {dt} seconds')

    return [",".join(seq.split("/"))]
