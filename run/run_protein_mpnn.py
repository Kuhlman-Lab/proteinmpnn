import os
import numpy as np
import time
import torch
import json
import copy
import pickle
from protein_mpnn.protein_mpnn_utils import ProteinMPNN, tied_featurize, _scores, _S_to_seq
from mpnn_utils import determine_weight_directory, MODEL_CONFIG, MODEL_NAMES, get_pdb_dataset, transform_inputs
from generate_json import FileArgumentParser

def decide_model_weights():
    longleaf = 'longleaf' in os.getcwd()

    if longleaf:
        weight_path = '/proj/kuhl_lab/alphafold/alphafold/alphafold/data/'
    else:
        weight_path = '/home/nzrandolph/git/alphfold/alphafold/alphafold/data/'

    return weight_path


def run_protein_mpnn(args):

    # Extract hyperparameters from model config
    hidden_dim = MODEL_CONFIG['hidden_dim']
    num_layers = MODEL_CONFIG['num_layers']

    # Determine path of the model weights
    model_weight_dir = determine_weight_directory()
    ckpt_path = os.path.join(model_weight_dir, f'{args.model_name}.pt')
    if not os.path.isfile(ckpt_path):
        raise ValueError(f'Invalid model name provided. Choose one of {MODEL_NAMES}.')

    # Get and set up output directory
    output_folder = args.out_folder
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    # NOT SURE WHAT THESE DO
    NUM_BATCHES = args.num_seq_per_target // args.batch_size
    BATCH_COPIES = args.batch_size

    # Construct list of temperatures
    temperatures = [float(item) for item in args.sampling_temp.split()]

    # Default is 'X'
    omit_AAs_list = args.global_omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32) # 1 if omitting, 0 if not

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Parse the PDB directory and get dataset of the proteins
    pdb_ds = get_pdb_dataset(args.pdb_dir)

    # Load the residue designability specs json
    if os.path.isfile(args.design_specs_json):
        with open(args.design_specs_json, 'r') as json_file:
            json_list = json_file.read()
        design_specs_jsons = json_list
    else:
        raise ValueError(f'The design specs json was not found: {args.design_specs_json}')

    if not args.bidirectional and args.mcmc:
        raise ValueError("MCMC cannot be enabled without bidirectional flag.")

    # This should be tested
    bias_AA_dict = None
    bias_AAs_np = np.zeros(len(alphabet))
    if args.bias_AA_dict:
        if os.path.isfile(args.bias_AA_dict):
            with open(args.bias_AA_dict, 'r') as json_file:
                bias_AA_dict = json.load(json_file)
        for n, AA in enumerate(alphabet):
            if AA in list(bias_AA_dict.keys()):
                bias_AAs_np[n] = bias_AA_dict[AA]
    
    bias_by_res_dict = None
    if args.bias_by_res_dict:
        if os.path.isfile(args.bias_by_res_dict):
            with open(args.bias_by_res_dict, 'r') as json_file:
                bias_by_res_dict = json.load(json_file)

    # Load the checkpoint and set up model
    ckpt = torch.load(ckpt_path, map_location=device)
    num_edges = ckpt['num_edges']
    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, 
                        num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=args.backbone_noise, 
                        k_neighbors=num_edges)
    model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Inference
    with torch.no_grad():
        for ix, protein in enumerate(pdb_ds):
            design_specs_dict = json.loads(design_specs_jsons)
            score_list = []
            S_sample_list = []
            probs_list = []
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            chain_id_dict, fixed_positions_dict, pssm_dict, omit_AA_dict, bias_AA_dict_decoy, tied_positions_dict, bias_by_res_decoy = transform_inputs(design_specs_dict, protein, experimental=args.experimental)
            X, S, mask, _, chain_M, chain_encoding_all, _, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, _, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict)
            # Setting pssm threshold to 0 for now. TODO: CHANGE LATER
            pssm_threshold = 0.0
            pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float()
            name_ = batch_clones[0]['name']

            # Do the inference
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            mask_for_loss = mask*chain_M*chain_M_pos
            scores, scores_per_res = _scores(S, log_probs, mask_for_loss)
            native_score, native_scores_per_res = scores.cpu().data.numpy(), scores_per_res
            native_mask_for_loss = mask_for_loss
            # Generate some sequences
            ali_file = os.path.join(output_folder,  batch_clones[0]['name'] + '.fasta')
            af2_file = os.path.join(output_folder, batch_clones[0]['name'] + '.csv')
            prob_file = os.path.join(output_folder, batch_clones[0]['name'] + '.npz')

            # multi-state splitting files
            if 'chain_key' in design_specs_dict:
                split_ali = [os.path.join(output_folder, fi + '.fasta') for fi in design_specs_dict['chain_key'].keys()]
                split_af2 = [os.path.join(output_folder, fi + '.csv') for fi in design_specs_dict['chain_key'].keys()]
            
            print(f'Generating sequences for: {name_}')
            if args.bidirectional:
                print(f'Bidirectional coding requested for: {name_}')
            t0 = time.time()
            with open(ali_file, 'w') as f:
                for temp in temperatures:
                    for j in range(NUM_BATCHES):
                        randn_2 = torch.randn(chain_M.shape, device=X.device)
                        # pssm flags set to 0
                        pssm_multi = 0.0
                        pssm_log_odds_flag = 0
                        pssm_bias_flag = 0

                        if args.pairwise:
                            sample_dict, log_probs, true_chain_mask = model.pairwise_sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), bias_by_res=bias_by_res_all, invert_probs=args.destabilize)
                            scores, scores_per_res = sample_dict["score"], sample_dict["score_per_res"]
                            mask_for_loss = true_chain_mask
                            S_sample = sample_dict["S"]
                        else:
                            if tied_positions_dict == None:
                                sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), bias_by_res=bias_by_res_all, invert_probs=args.destabilize)

                            elif args.mcmc: # MCMC based bidirectional sampling
                                sample_dict = model.mcmc_sample(X, mask, residue_idx, chain_encoding_all, temperature=temp)                    

                            else:
                                sample_dict = model.tied_sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), 
                                                                tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta, bias_by_res=bias_by_res_all, invert_probs=args.destabilize, bidir=args.bidirectional, bidir_table_dir=model_weight_dir)
                            S_sample = sample_dict["S"]

                            log_probs = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
                            mask_for_loss = mask*chain_M*chain_M_pos
                            scores, scores_per_res = _scores(S_sample, log_probs, mask_for_loss)
                            scores = scores.cpu().data.numpy()
                        
                        if args.dump: # collect probabilities for each seq
                            probs_list.append(sample_dict["probs"].cpu().data.numpy())


                        S_sample_list.append(S_sample.cpu().data.numpy())
                        for b_ix in range(BATCH_COPIES):
                            masked_chain_length_list = masked_chain_length_list_list[b_ix]
                            masked_list = masked_list_list[b_ix]
                            seq_recovery_per_res = torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21), axis=-1)*mask_for_loss[b_ix]
                            seq_recovery_rate = torch.sum(seq_recovery_per_res)/torch.sum(mask_for_loss[b_ix])
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
                                f.write('>{}, score={}, fixed_chains={}, designed_chains={}, model_name={}\n{}\n'.format(name_, native_score_print, print_visible_chains, print_masked_chains, args.model_name, native_seq)) #write the native sequence

                                # FOR NATIVE SEQUENCES
                                # Calculating per-chain scores and splitting states into separate .fasta files
                                if 'chain_key' in design_specs_dict:
                                    for fsplit, af2split in zip(split_ali, split_af2):
                                        with open(fsplit, 'w') as fsp:
                                            fbase = os.path.splitext(os.path.basename(fsplit))[0]
                                            chains_original = list(design_specs_dict['chain_key'][fbase].keys())
                                            chains_to_recover = list(design_specs_dict['chain_key'][fbase].values())
                                            chain_idx = [masked_list_list[0].index(ctr) for ctr in chains_to_recover]
                                            native_seq_split = '/'.join([native_seq.split('/')[idx] for idx in chain_idx])
                                            scores_per_chain = {}
                                            # make mask for per-state scoring as well
                                            state_idx = torch.zeros(chain_encoding_all.shape, dtype=torch.bool, device=device)

                                            for ctr, idx in zip(chains_original, chain_idx):
                                                # grab the indices for scoring each chain
                                                score_idx = chain_encoding_all == idx + 1
                                                chain_score = torch.sum(native_scores_per_res[score_idx], dim=-1) / torch.sum(native_mask_for_loss[score_idx], dim=-1)
                                                chain_score = np.format_float_positional(np.float32(chain_score.cpu().data.numpy().mean()), unique=False, precision=4)
                                                scores_per_chain[ctr] = chain_score
                                                state_idx += score_idx

                                            state_idx = state_idx.bool()
                                            score_per_state = torch.sum(native_scores_per_res[state_idx], dim=-1) / torch.sum(native_mask_for_loss[state_idx], dim=-1)
                                            score_per_state = np.format_float_positional(np.float32(score_per_state.cpu().data.numpy().mean()), unique=False, precision=4)
                                            fsp.write('>{}, fixed_chains={}, designed_chains={}, state_score={}, scores_per_chain={}, model={}\n{}\n'.format(fbase, print_visible_chains, chains_original, 
                                                                                                                                            score_per_state, scores_per_chain, args.model_name, native_seq_split))
                    
                                            if args.af2_formatted_output:
                                                with open(af2split, 'w') as faf2:
                                                    af2_seqs = native_seq_split.split('/')
                                                    af2_seqs = ',' + ','.join(af2_seqs)
                                                    # need to sanitize comment to remove any commas or AF2 parsing will fail
                                                    comment = f'{name_} fixed_chains={print_visible_chains} designed_chains={chains_original} state_score={score_per_state} scores_per_chain={scores_per_chain} model_name={args.model_name}'.replace(',', '')
                                                    faf2.write(f'{af2_seqs} # {comment}\n')    

                                if args.af2_formatted_output:
                                    with open(af2_file, 'w') as af2:
                                        af2_seqs = native_seq.split('/')
                                        af2_seqs = ','+','.join(af2_seqs)
                                        # need to sanitize comment to remove any commas or AF2 parsing will fail
                                        comment = f'{name_} score={native_score_print} fixed_chains={print_visible_chains} designed_chains={print_masked_chains} model_name={args.model_name}'.replace(',', '')
                                        af2.write(f'{af2_seqs} # {comment}\n')
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
                            f.write('>T={}, sample={}, score={}, seq_recovery={}\n{}\n'.format(temp,b_ix,score_print,seq_rec_print,seq)) #write generated sequence

                            # FOR DESIGNED SEQUENCES
                            if 'chain_key' in design_specs_dict:
                                # split generated seq and save each to separate file
                                for fsplit, af2split in zip(split_ali, split_af2):
                                    with open(fsplit, 'a') as fsp:
                                        fbase = os.path.splitext(os.path.basename(fsplit))[0]
                                        chains_original = list(design_specs_dict['chain_key'][fbase].keys())
                                        chains_to_recover = list(design_specs_dict['chain_key'][fbase].values())
                                        chain_idx = [masked_list.index(ctr) for ctr in chains_to_recover]
                                        seq_split = '/'.join([seq.split('/')[idx] for idx in chain_idx])

                                        scores_per_chain, seq_recovery_per_chain = {}, {}
                                        # make mask for per-state scoring as well
                                        state_idx = torch.zeros(chain_encoding_all.shape, dtype=torch.bool, device=device)
                                        
                                        for ctr, idx in zip(chains_original, chain_idx):
                                                # grab the indices for scoring each chain
                                                score_idx = chain_encoding_all == idx + 1  # [1, L]
                                                chain_score = torch.sum(scores_per_res[score_idx], dim=-1) / torch.sum(mask_for_loss[score_idx], dim=-1)
                                                chain_score = np.format_float_positional(np.float32(chain_score.cpu().data.numpy().mean()), unique=False, precision=4)
                                                scores_per_chain[ctr] = chain_score
                                                # use this for seq recovery too
                                                chain_seq_rec = torch.sum(seq_recovery_per_res.unsqueeze(0)[score_idx], dim=-1) / torch.sum(mask_for_loss[score_idx], dim=-1)
                                                seq_recovery_per_chain[ctr] = np.format_float_positional(np.float32(chain_seq_rec.cpu().data.numpy().mean()), unique=False, precision=4)
                                                state_idx += score_idx

                                        state_idx = state_idx.bool()
                                        score_per_state = torch.sum(scores_per_res[state_idx], dim=-1) / torch.sum(mask_for_loss[state_idx], dim=-1)
                                        score_per_state = np.format_float_positional(np.float32(score_per_state.cpu().data.numpy().mean()), unique=False, precision=4)
                                        seq_recovery_per_state = torch.sum(seq_recovery_per_res.unsqueeze(0)[state_idx], dim=-1) / torch.sum(mask_for_loss[state_idx], dim=-1)
                                        seq_recovery_per_state = np.format_float_positional(np.float32(seq_recovery_per_state.cpu().data.numpy().mean()), unique=False, precision=4)
                                        fsp.write('>{}, sample={}, state_score={}, scores_per_chain={}, state_recovery={}, seq_recovery_per_chain={}, model={}\n{}\n'.format(fbase, b_ix, score_per_state, scores_per_chain, seq_recovery_per_state, seq_recovery_per_chain, args.model_name, seq_split))

                                    if args.af2_formatted_output:
                                        with open(af2split, 'a') as faf2:
                                            af2_seqs = seq_split.split('/')
                                            af2_seqs = ',' + ','.join(af2_seqs)
                                            # need to sanitize comment to remove any commas or AF2 parsing will fail
                                            comment = f'T={temp} sample={b_ix} state_score={score_per_state} scores_per_chain={scores_per_chain} state_seq_recovery={seq_recovery_per_state} seq_recovery_per_chain={seq_recovery_per_chain}'.replace(',', '')
                                            faf2.write(f'{af2_seqs} # {comment}\n')


                            if args.af2_formatted_output:
                                with open(af2_file, 'a') as af2:
                                    af2_seqs = seq.split('/')
                                    af2_seqs = ','+','.join(af2_seqs)
                                    # need to sanitize comment to remove any commas or AF2 parsing will fail
                                    comment = f'T={temp} sample={b_ix} score={score_print} seq_recovery={seq_rec_print}'.replace(',', '')
                                    af2.write(f'{af2_seqs} # {comment}\n')
            
            # Average probs for different decoding runs
            if args.dump:
                probs = np.squeeze(np.mean(np.stack(probs_list), axis=0), axis=0)
                with open(prob_file, 'wb') as pf:
                    np.save(pf, probs)

            t1 = time.time()
            dt = round(float(t1-t0), 4)
            num_seqs = len(temperatures)*NUM_BATCHES*BATCH_COPIES
            total_length = X.shape[1]
            print(f'{num_seqs} sequences of length {total_length} generated in {dt} seconds')


def run_protein_mpnn_func(pdb_dir, design_specs_json, model_name="v_48_020", backbone_noise=0.00, num_seq_per_target=1, batch_size=1, sampling_temp="0.1", af2_formatted_output=False, bidir=False, bias_AA_dict=None, bias_by_res_dict=None, dump_probs=False):
    # Extract hyperparameters from model config
    hidden_dim = MODEL_CONFIG['hidden_dim']
    num_layers = MODEL_CONFIG['num_layers']

    # Determine path of the model weights
    model_weight_dir = determine_weight_directory()
    ckpt_path = os.path.join(model_weight_dir, f'{model_name}.pt')
    if not os.path.isfile(ckpt_path):
        raise ValueError('Invalid model name provided. Choose ca_48_002, ca_48_020, v_48_002, '
                         'v_48_010, v_48_020, v_48_030, s_48_010, or s_48_020.')

    # NOT SURE WHAT THESE DO
    NUM_BATCHES = num_seq_per_target // batch_size
    BATCH_COPIES = batch_size

    # Construct list of temperatures
    temperatures = [float(item) for item in sampling_temp.split()]

    # Default is 'X'
    omit_AAs_list = 'X'
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32) # 1 if omitting, 0 if not

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Parse the PDB directory and get dataset of the proteins
    pdb_ds = get_pdb_dataset(pdb_dir)

    #amino acid bias
    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
        for n, AA in enumerate(alphabet):
            if AA in list(bias_AA_dict.keys()):
                bias_AAs_np[n] = bias_AA_dict[AA]

    ckpt = torch.load(ckpt_path, map_location=device)
    num_edges = ckpt['num_edges']
    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim,
                        num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise,
                        k_neighbors=num_edges)
    model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Inference
    with torch.no_grad():
        outputs = []
        for ix, protein in enumerate(pdb_ds):
            output = []
            design_specs_dict = json.loads(design_specs_json)
            score_list = []
            all_probs_list = []
            S_sample_list = []
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            chain_id_dict, fixed_positions_dict, pssm_dict, omit_AA_dict, bias_AA_dict_decoy, tied_positions_dict, bias_by_res_dict_decoy = transform_inputs(design_specs_dict, protein)
            print('=' * 50)
            print(chain_id_dict, '\n', fixed_positions_dict, '\n', pssm_dict, '\n',
                  omit_AA_dict, '\n', tied_positions_dict, '\n', bias_by_res_dict)
            print('=' * 50)
                        
            X, S, mask, _, chain_M, chain_encoding_all, _, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, _, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict)

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
            if bidir:
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
                        sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temerature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), bias_by_res=bias_by_res_all)
                    else:
                        sample_dict = model.tied_sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta, bias_by_res=bias_by_res_all, bidir=bidir, bidir_table_dir=model_weight_dir)
                    S_sample = sample_dict["S"]
                    log_probs = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
                    mask_for_loss = mask*chain_M*chain_M_pos
                    scores, _ = _scores(S_sample, log_probs, mask_for_loss)
                    scores = scores.cpu().data.numpy()
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
                            output.append((name_, native_score_print, print_visible_chains, print_masked_chains, model_name, native_seq))
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
        t1 = time.time()
        dt = round(float(t1-t0), 4)
        num_seqs = len(temperatures)*NUM_BATCHES*BATCH_COPIES
        total_length = X.shape[1]
        print(f'{num_seqs} sequences of length {total_length} generated in {dt} seconds')

    return outputs


if __name__ == "__main__":
    # Construct the parser and its arguments.
    parser = FileArgumentParser(description='Script that runs ProteinMPNN.', 
                                fromfile_prefix_chars='@')
    
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
    parser.add_argument('--mcmc', action='store_true', help='If enabled, bidirectional coding uses MCMC routine. Must have --bidirectional flag enabled.')
    parser.add_argument('--dump', action='store_true', help='If enabled, raw probability tables (L x 20) will be dumped for each sequence.')
    args = parser.parse_args()
    run_protein_mpnn(args) 
