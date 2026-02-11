import numpy as np
import torch
import time
import csv
import os

# Change mutation function to disallow stop codons from ever being chosen
# Add a function to remove stop codons if they are present after a full round of sampling


def BPs_to_AAs(fwd_sequence, rev_sequence, num_codons, shift):
    aa_list1, aa_list2 = [], []
    for i in range(num_codons):
        fwd_codon = fwd_sequence[i * 3: i * 3 + 3]
        rev_codon = rev_sequence[i * 3: i * 3 + 3]
        aa_list1 += codon_to_amino_acid[fwd_codon]
        aa_list2 += codon_to_amino_acid[rev_codon]

    aa_list1 = ''.join(aa_list1)
    aa_list2 = ''.join(aa_list2)
    return aa_list1, aa_list2

def sample_all(position, num_nas, shift, fwd_sequence, rev_sequence, probs1, probs2, score_array1, score_array2):
    """
    position = position to sample
    fwd_sequence = string representation of seq 1 of length L
    rev_sequence = string representation of seq 2 of length L

    probs1 = raw logits for seq 1 of shape [L, 21]
    probs2 = raw logits for seq 1 of shape [L, 21]

    codons = tensor of codon indices to score and replace
    shift = how many base pairs to shift when retrieving AA from codon

    score_array1 = tensor of previous score1 of shape [L]
    score_array2 = tensor of previous score2 of shape [L]
    """

    new_fwd_sequenceA = fwd_sequence[:position] + 'A' + fwd_sequence[position + 1:]
    new_fwd_sequenceT = fwd_sequence[:position] + 'T' + fwd_sequence[position + 1:]
    new_fwd_sequenceC = fwd_sequence[:position] + 'C' + fwd_sequence[position + 1:]
    new_fwd_sequenceG = fwd_sequence[:position] + 'G' + fwd_sequence[position + 1:]

    # Calculates the reverse compliment position and generates the corresponding sequences
    rev_position = num_nas - position - 1
    new_rev_sequenceA = rev_sequence[:rev_position] + 'T' + rev_sequence[rev_position + 1:]
    new_rev_sequenceT = rev_sequence[:rev_position] + 'A' + rev_sequence[rev_position + 1:]
    new_rev_sequenceC = rev_sequence[:rev_position] + 'G' + rev_sequence[rev_position + 1:]
    new_rev_sequenceG = rev_sequence[:rev_position] + 'C' + rev_sequence[rev_position + 1:]

    # convert from bp index to codon index and scores the new sequences
    fwd_idx = position // 3
    rev_idx = (rev_position) // 3
    new_score_array1A, new_score_array2A = score(new_fwd_sequenceA, new_rev_sequenceA, probs1, probs2, fwd_idx, rev_idx, score_array1, score_array2)
    new_score_array1T, new_score_array2T = score(new_fwd_sequenceT, new_rev_sequenceT, probs1, probs2, fwd_idx, rev_idx, score_array1, score_array2)
    new_score_array1C, new_score_array2C = score(new_fwd_sequenceC, new_rev_sequenceC, probs1, probs2, fwd_idx, rev_idx, score_array1, score_array2)
    new_score_array1G, new_score_array2G = score(new_fwd_sequenceG, new_rev_sequenceG, probs1, probs2, fwd_idx, rev_idx, score_array1, score_array2)

    # Create lists of score arrays and sequences for min calculations and indexing
    score_arrays1 = [new_score_array1A, new_score_array1T, new_score_array1C, new_score_array1G]
    score_arrays2 = [new_score_array2A, new_score_array2T, new_score_array2C, new_score_array2G]

    sequence_list1 = [new_fwd_sequenceA, new_fwd_sequenceT, new_fwd_sequenceC, new_fwd_sequenceG]
    sequence_list2 = [new_rev_sequenceA, new_rev_sequenceT, new_rev_sequenceC, new_rev_sequenceG]

    return score_arrays1, score_arrays2, sequence_list1, sequence_list2

def sample_all_same_strand(position, num_nas, shift, fwd_sequence, rev_sequence, probs1, probs2, score_array1, score_array2):
    """
    position = position to sample
    fwd_sequence = string representation of seq 1 of length L
    rev_sequence = string representation of seq 2 of length L

    probs1 = raw logits for seq 1 of shape [L, 21]
    probs2 = raw logits for seq 1 of shape [L, 21], flipped along L axis

    codons = tensor of codon indices to score and replace
    shift = how many base pairs to shift when retrieving AA from codon

    score_array1 = tensor of previous score1 of shape [L]
    score_array2 = tensor of previous score2 of shape [L]
    """

    new_fwd_sequenceA = fwd_sequence[:position] + 'A' + fwd_sequence[position + 1:]
    new_fwd_sequenceT = fwd_sequence[:position] + 'T' + fwd_sequence[position + 1:]
    new_fwd_sequenceC = fwd_sequence[:position] + 'C' + fwd_sequence[position + 1:]
    new_fwd_sequenceG = fwd_sequence[:position] + 'G' + fwd_sequence[position + 1:]

    # Calculates the reverse compliment position and generates the corresponding sequences
    new_rev_sequenceA = fwd_sequence[shift:position] + 'A' + fwd_sequence[position + 1:] + fwd_sequence[-shift:]
    new_rev_sequenceT = fwd_sequence[shift:position] + 'T' + fwd_sequence[position + 1:] + fwd_sequence[-shift:]
    new_rev_sequenceC = fwd_sequence[shift:position] + 'C' + fwd_sequence[position + 1:] + fwd_sequence[-shift:]
    new_rev_sequenceG = fwd_sequence[shift:position] + 'G' + fwd_sequence[position + 1:] + fwd_sequence[-shift:]

    # convert from bp index to codon index and scores the new sequences
    fwd_idx = position // 3
    rev_idx = (num_nas - position - 1) // 3
    new_score_array1A, new_score_array2A = score(new_fwd_sequenceA, new_rev_sequenceA, probs1, probs2, fwd_idx, rev_idx, score_array1, score_array2)
    new_score_array1T, new_score_array2T = score(new_fwd_sequenceT, new_rev_sequenceT, probs1, probs2, fwd_idx, rev_idx, score_array1, score_array2)
    new_score_array1C, new_score_array2C = score(new_fwd_sequenceC, new_rev_sequenceC, probs1, probs2, fwd_idx, rev_idx, score_array1, score_array2)
    new_score_array1G, new_score_array2G = score(new_fwd_sequenceG, new_rev_sequenceG, probs1, probs2, fwd_idx, rev_idx, score_array1, score_array2)

    # Create lists of score arrays and sequences for min calculations and indexing
    score_arrays1 = [new_score_array1A, new_score_array1T, new_score_array1C, new_score_array1G]
    score_arrays2 = [new_score_array2A, new_score_array2T, new_score_array2C, new_score_array2G]

    sequence_list1 = [new_fwd_sequenceA, new_fwd_sequenceT, new_fwd_sequenceC, new_fwd_sequenceG]
    sequence_list2 = [new_rev_sequenceA, new_rev_sequenceT, new_rev_sequenceC, new_rev_sequenceG]

    return score_arrays1, score_arrays2, sequence_list1, sequence_list2

def score_all_codons(seq1, seq2, probs1, probs2):
    """
    Score every codon in the sequences.
    Returns two score arrays of shape [num_codons]
    """
    num_codons = probs1.shape[0]
    score_array1 = torch.zeros(num_codons, device='cpu')
    score_array2 = torch.zeros(num_codons, device='cpu')

    for i in range(num_codons):
        fwd_codon = seq1[i*3 : i*3+3]
        rev_codon = seq2[i*3 : i*3+3]

        # Forward codon
        try:
            aa1 = codon_to_amino_acid[fwd_codon]
            if aa1 == 'Z':
                score_array1[i] = float('inf')
            else:
                score_array1[i] = -torch.log(probs1[i, amino_acid_position[aa1]])
        except (KeyError, IndexError):
            score_array1[i] = 0.0

        # Reverse codon
        try:
            aa2 = codon_to_amino_acid[rev_codon]
            if aa2 == 'Z':
                score_array2[i] = float('inf')
            else:
                score_array2[i] = -torch.log(probs2[i, amino_acid_position[aa2]])
        except (KeyError, IndexError):
            score_array2[i] = 0.0

    return score_array1, score_array2

'''
def score(seq1, seq2, probs1, probs2, codons, score_array1, score_array2):
    """
    seq1 = string representation of seq 1 of length L
    seq2 = string representation of seq 2 of length L

    probs1 = raw logits for seq 1 of shape [L, 21]
    probs2 = raw logits for seq 1 of shape [L, 21], flipped along L axis

    codons = tensor of codon indices to score and replace
    shift = how many base pairs to shift when retrieving AA from codon

    score_array1 = tensor of previous score1 of shape [L]
    score_array2 = tensor of previous score2 of shape [L]
    """
    sa1 = score_array1.clone()
    sa2 = score_array2.clone()

    # Fill in arrays of [L, ] score values in quick loop
    num_codons = codons.numel()
    aa1_arr, aa2_arr = torch.zeros(num_codons, device='cpu', dtype=torch.long), torch.zeros(num_codons, device='cpu', dtype=torch.long)
    for n, i in enumerate(codons):
        fwd_codon = seq1[i * 3: i * 3 + 3]
        rev_codon = seq2[i * 3: i * 3 + 3]

        #score all positions normally except overhang codons don't matter for that strand so we always want to score them as zero
        try:
            aa1 = codon_to_amino_acid[fwd_codon]
            if aa1 == 'Z':  # If it's a stop codon, assign maximum penalty
                sa1[i] = torch.tensor(float('inf'), dtype=torch.float32)
            else:
                aa1 = 'X' if aa1 == 'Z' else aa1  # This line is now redundant but kept for safety
                idx1 = amino_acid_position[aa1]
                aa1_arr[n] = idx1
                sa1[i] = -torch.log(probs1[i, idx1])
        except (KeyError, IndexError):
            pass

        # check for rev complements
        try:
            aa2 = codon_to_amino_acid[rev_codon]
            if aa2 == 'Z':  # If it's a stop codon, assign maximum penalty
                sa2[i] = torch.tensor(float('inf'), dtype=torch.float32)
            else:
                aa2 = 'X' if aa2 == 'Z' else aa2  # This line is now redundant but kept for safety
                idx2 = amino_acid_position[aa2]
                aa2_arr[n] = idx2
                sa2[i] = -torch.log(probs2[i, idx2])
        except (KeyError, IndexError):
            pass

    return sa1, sa2
'''


def score(seq1, seq2, probs1, probs2, fwd_idx, rev_idx, score_array1, score_array2, debug=False):
    sa1 = score_array1.clone()
    sa2 = score_array2.clone()
    num_codons = len(score_array1)

    # --- Forward codon ---
    try:
        fwd_codon = seq1[fwd_idx*3 : fwd_idx*3+3]
        if len(fwd_codon) < 3:
            raise IndexError
        aa1 = codon_to_amino_acid[fwd_codon]
        sa1[fwd_idx] = float('inf') if aa1 == 'Z' else -torch.log(probs1[fwd_idx, amino_acid_position[aa1]])
    except (KeyError, IndexError):
        #print(f"Error scoring codons at indices fwd_idx={fwd_idx}, rev_idx={rev_idx}. Check if the sequences are properly formatted and the indices are within bounds.")
        pass

    # --- Reverse codon ---
    try:
        rev_codon = seq2[rev_idx*3 : rev_idx*3+3]
        if len(rev_codon) < 3:
            raise IndexError
        aa2 = codon_to_amino_acid[rev_codon]
        sa2[rev_idx] = float('inf') if aa2 == 'Z' else -torch.log(probs2[rev_idx, amino_acid_position[aa2]])
    except (KeyError, IndexError):
        #print(f"Error scoring codons at indices fwd_idx={fwd_idx}, rev_idx={rev_idx}. Check if the sequences are properly formatted and the indices are within bounds.")
        pass

    if debug:
        print(f"[DEBUG] fwd_idx={fwd_idx}, fwd_codon={seq1[fwd_idx*3:fwd_idx*3+3]}, sa1={sa1[fwd_idx]}")
        print(f"[DEBUG] rev_idx={rev_idx}, rev_codon={seq2[rev_idx*3:rev_idx*3+3]}, sa2={sa2[rev_idx]}")

    return sa1, sa2
    
'''
def z_mutator(fwd_sequence, rev_sequence, num_codons, shift):
    """
    Removes stop codons from the sequences and returns the modified sequences without caring about scores.
    Useful to prevent the algorithm from getting stuck in a local minimum with stop codons.
    """
    fwd_zs, rev_zs, aa_list1, aa_list2  = find_zs(fwd_sequence, rev_sequence, num_codons)
    zs_exist = len(fwd_zs) > 0 or len(rev_zs) > 0
    while zs_exist:
        for i in fwd_zs:
            fwd_sequence = fwd_sequence[:i * 3] + 'C' + fwd_sequence[i * 3 + 1:]
            rev_sequence = rev_sequence[:len(rev_sequence) - i * 3] + 'G' + rev_sequence[len(rev_sequence) - i * 3 + 1:]
        for i in rev_zs:
            rev_sequence = rev_sequence[:i * 3] + 'C' + rev_sequence[i * 3 + 1:]
            fwd_sequence = fwd_sequence[:len(fwd_sequence) - i * 3] + 'G' + fwd_sequence[len(fwd_sequence) - i * 3 + 1:]
        fwd_zs, rev_zs, aa_list1, aa_list2 = find_zs(fwd_sequence, rev_sequence, num_codons)
        zs_exist = len(fwd_zs) > 0 or len(rev_zs) > 0
    # If we removed all stop codons, return the modified sequences

    return fwd_sequence, rev_sequence
'''


# Define Dictionary of Amino Acids
codon_to_amino_acid = {
    'TTT': 'F', 'TTC': 'F',  # Phenylalanine
    'TTA': 'L', 'TTG': 'L',  # Leucine
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',  # Leucine
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',  # Isoleucine
    'ATG': 'M',  # Methionine (Start codon)
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',  # Valine
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',  # Serine
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',  # Proline
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',  # Threonine
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',  # Alanine
    'TAT': 'Y', 'TAC': 'Y',  # Tyrosine
    'TAA': 'Z', 'TAG': 'Z', 'TGA': 'Z',  # Stop codons (Z)
    'CAT': 'H', 'CAC': 'H',  # Histidine
    'CAA': 'Q', 'CAG': 'Q',  # Glutamine
    'AAT': 'N', 'AAC': 'N',  # Asparagine
    'AAA': 'K', 'AAG': 'K',  # Lysine
    'GAT': 'D', 'GAC': 'D',  # Aspartic acid
    'GAA': 'E', 'GAG': 'E',  # Glutamic acid
    'TGT': 'C', 'TGC': 'C',  # Cysteine
    'TGG': 'W',  # Tryptophan
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',  # Arginine
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'  # Glycine
}

# Define Dictionary of Amino Acid Positions
amino_acid_position = {
    'A': 0,  # Alanine
    'C': 1,  # Cysteine
    'D': 2,  # Aspartic acid
    'E': 3,  # Glutamic acid
    'F': 4,  # Phenylalanine
    'G': 5,  # Glycine
    'H': 6,  # Histidine
    'I': 7,  # Isoleucine
    'K': 8,  # Lysine
    'L': 9,  # Leucine
    'M': 10, # Methionine
    'N': 11, # Asparagine
    'P': 12, # Proline
    'Q': 13, # Glutamine
    'R': 14, # Arginine
    'S': 15, # Serine
    'T': 16, # Threonine
    'V': 17, # Valine
    'W': 18, # Tryptophan
    'Y': 19,  # Tyrosine
    'X': 20 # Variable codon
}

# Define Nucleic Acid Complement Dictionary
nucleic_acid_complement = {
    'A': 'T',
    'T': 'A',
    'C': 'G',
    'G': 'C'
}

def na_sample(probs1, probs2):
    '''
    # Define the maximum scores possible for each sequence individually
    max_score1 = torch.sum(torch.max(probs1, axis=-1))
    max_score2 = torch.sum(torch.max(probs2, axis=-1))
    '''
    metro_file = os.path.join(os.getcwd(), 'metro.flags')
    if not os.path.isfile(metro_file):
        raise ValueError(f"Missing flag file expected at {metro_file}")
    strand, shift, temperature, use_gradient, gradient_start, gradient_end, num_mutations, metropolis, overhang_residues = run_metropolis_from_flags(metro_file)

    # Ensure that the two proteins have the same length
    if probs1.shape != probs2.shape:
        raise ValueError('Tables must have the same shape')

    if strand == 'same':
        sample_all_function = sample_all_same_strand
        score_function = score
    else:
        sample_all_function = sample_all
        score_function = score

    # Fill in penalty for stop codon using "X" position in prob arrays
    stop_penalty = 10000 # should be as large as possible 

    # The number of codons in the pair of proteins
    num_codons = probs1.shape[0] 
    num_nas = num_codons * 3 + shift + overhang_residues * 3 # +1 for single frame shift, +2 for double frame shift, +3 per overhang base

    temp_list = []
    if use_gradient:
        temp_list = np.linspace(gradient_start, gradient_end, num_mutations)
    else:
        temp_list = [temperature] * num_mutations

    '''
    # Create a random DNA sequence for testing
    fwd_sequence = ''
    for NA in range(num_nas):
        fwd_sequence += np.random.choice(['A', 'T', 'C', 'G'])
    
    # Create the reverse complement of the DNA sequence
    rev_sequence = ''
    for BP in fwd_sequence[::-1]:
        rev_sequence += nucleic_acid_complement[BP]
    '''
    
    # Create an all A DNA sequence for testing
    fwd_sequence = 'A' * num_nas
    rev_sequence = 'T' * num_nas

    # Do initial scoring for baseline values
    probs1, probs2 = probs1.to('cpu'), probs2.to('cpu')

    #init1 = torch.full((num_codons,), dtype=torch.float, device='cpu', fill_value=torch.inf)
    #init2 = torch.full((num_codons,), dtype=torch.float, device='cpu', fill_value=torch.inf)
    #codons = torch.arange(num_codons, dtype=torch.long, device='cpu')
    score_array1, score_array2 = score_all_codons(fwd_sequence, rev_sequence, probs1, probs2)
    score1, score2 = torch.sum(score_array1), torch.sum(score_array2)
    score_overall = (score1 * score2)
    best_iter = 0
    metro_used = 0
    base_set = ['A', 'T', 'C', 'G']

    tick = time.time()
    no_z = False
    rounds = 0
    while not no_z:
        for i in range(num_mutations):
            # only select non-overhanging positions for simplicity
            position = np.random.randint(0, num_nas)
            char = fwd_sequence[position]

            score_arrays1, score_arrays2, sequence_list1, sequence_list2 = sample_all_function(position, num_nas, shift, fwd_sequence, rev_sequence, probs1, probs2, score_array1, score_array2)

            # Compute combined scores with each given mutation set
            scores1 = torch.tensor([torch.sum(arr) for arr in score_arrays1])
            scores2 = torch.tensor([torch.sum(arr) for arr in score_arrays2])
            combined_scores = scores1 * scores2
            
            # Get the indices of the minimum values. After 1000 iterations, disallow accepting the same sequence
            if i > 1000:
                combined_scores[base_set.index(char)] = torch.tensor(float('inf'))

            min_indices = torch.nonzero(combined_scores == combined_scores.min()).squeeze()

            # If there are multiple indices with the same minimum value, randomly choose one. 
            if min_indices.dim() > 0:  # If there are multiple indices
                random_min = min_indices[torch.randint(0, len(min_indices), (1,))]
            else:  # If there's only one index
                random_min = min_indices

            # Determine if we should accept the new sequence
            if combined_scores[random_min] < score_overall:
                    new_score = combined_scores[random_min]
                    fwd_sequence, rev_sequence = sequence_list1[random_min], sequence_list2[random_min]
                    score1, score2 = scores1[random_min], scores2[random_min]
                    score_overall = new_score
                    score_array1, score_array2 = score_arrays1[random_min], score_arrays2[random_min]
                    best_iter = i
            
            # If new score is worse we randomly sample a number and use the Metropolis Algorithm to determine if we should accept the new sequence
            elif metropolis:
                    metro_rand = np.random.rand()
                    energy = ((combined_scores[random_min]) - score_overall)/score_overall
                    energy = energy.detach().cpu().numpy()
                    metro_score = 1 - np.exp(-energy/temp_list[i]) #metropolis energy = e^(-E/T)
                    new_score = combined_scores[random_min]
                    if metro_score < metro_rand:
                        fwd_sequence, rev_sequence = sequence_list1[random_min], sequence_list2[random_min]
                        score1, score2 = scores1[random_min], scores2[random_min]
                        score_overall = new_score
                        score_array1, score_array2 = score_arrays1[random_min], score_arrays2[random_min]
                        metro_used += 1
            if torch.isinf(score1) or torch.isinf(score2):
                print('A sequence has a stop codon during iter {i}')

        # Format sequence into final (transcribed) format
        final_AAs1, final_AAs2 = BPs_to_AAs(fwd_sequence, rev_sequence, num_codons, shift)
        final_AAs1 = ''.join(final_AAs1)
        final_AAs2 = ''.join(final_AAs2)

        pos = 0
        out_probs1 = torch.zeros_like(probs1, device='cuda')
        for char in final_AAs1:
            char = "X" if char == "Z" else char
            out_probs1[pos, amino_acid_position[char]] = 1
            pos += 1

        out_probs2 = torch.zeros_like(probs2, device='cuda')
        pos = 0
        for char in final_AAs2:
            char = "X" if char == "Z" else char
            out_probs2[pos, amino_acid_position[char]] = 1
            pos += 1

        # Check if there are any stop codons in the final sequence
        no_z = ('Z' not in final_AAs1) and ('Z' not in final_AAs2)
        '''
        if rounds >5 and not no_z:
            print(f'Stop codons found. Removing Zs from sequences: {final_AAs1}, {final_AAs2}')
            no_z = True
            fwd_sequence, rev_sequence = z_mutator(fwd_sequence, rev_sequence, num_codons, shift)
            score_array1, score_array2 = score(fwd_sequence, rev_sequence, probs1, probs2, codons, shift, init1, init2, stop_penalty)
            score_overall = (score1 * score2)
            final_AAs1, final_AAs2 = BPs_to_AAs(fwd_sequence, rev_sequence, num_codons, shift)
            final_AAs1 = ''.join(final_AAs1)
            final_AAs2 = ''.join(final_AAs2)
        '''
        rounds += 1



    elapsed = time.time() - tick
    print(f'Finished after {rounds} rounds:')
    print(f'Final Scores: {score1}, {score2}')
    print(f'Final AAs: {final_AAs1}, {final_AAs2}')
    print(f'Fwd Sequence (5p to 3p): {fwd_sequence}')
    print(f'Rev Sequence (5p to 3p): {rev_sequence[::-1]}')
    print(f'MCMC Runtime: {round(elapsed, 0)}s')
    print(f'Best Iteration: {best_iter}')
    print(f'Metropolis Used: {metro_used}')

    append_to_csv('mpnn_results.csv', [score1.item(), score2.item(), best_iter, metro_used, round(elapsed, 0), final_AAs1, final_AAs2, fwd_sequence, rev_sequence[::-1]])

    return out_probs1, out_probs2, score1, score2, final_AAs1, final_AAs2, fwd_sequence, rev_sequence

def load_flags(file_path):
    """
    Reads arguments from a flags file and returns them as a dictionary.
    
    :param file_path: Path to the flags file (e.g., 'metro.flags').
    :return: A dictionary of parsed key-value pairs.
    """
    args = {}
    try:
        with open(file_path, "r") as file:
            for line in file:
                # Skip empty lines and comments
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Split key-value pairs
                key, value = map(str.strip, line.split("=", 1))
                # Convert to appropriate data types
                if value.lower() in {"true", "false"}:
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string if not a number
                args[key] = value
    except FileNotFoundError:
        print(f"Error: Flags file '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading flags file '{file_path}': {e}")
    return args

def append_to_csv(file_path, data):
    """
    Appends data to an existing CSV file, creating it if necessary.
    
    Parameters:
    file_path (str): Path to the CSV file.
    data (list of lists): Data to append (each sublist is a row).
    """
    file_exists = os.path.exists(file_path)
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            header = [
                'Final Score 1', 
                'Final Score 2', 
                'Best Iteration',
                'Metropolis Used', 
                'Runtime', 
                'Final AAs 1', 
                'Final AAs 2', 
                'Fwd Sequence', 
                'Rev Sequence',
            ]
            writer.writerow(header)  # Write headers if file does not exist
        
        writer.writerow(data)

def run_metropolis_from_flags(flags_file):
    """
    Runs the metropolis sampler using arguments from a flags file.
    :param flags_file: Path to the flags file.
    """
    args = load_flags(flags_file)
    #print(f"Loaded arguments: {args}")
    # Use the arguments to perform your logic
    strand = args.get("strand", 'opposite')
    shift = args.get("shift", 1)
    overhang_residues = args.get("overhang_residues", 0) #Number of bases on the 5' and 3' ends of the first and second proteins that do not overlap
    temperature = args.get("temperature", 0.007) #Accepts ~5 percent of the sampled data
    use_gradient = args.get("use_gradient", False)
    gradient_start = args.get("gradient_start", 0.5) #Accepts ~55 percent of the sampled data
    gradient_end = args.get("gradient_end", 0.05) #Accepts ~1 percent of the data
    num_mutations = args.get("num_mutations", 30000) #Number of random mutations to introduce to the sequence, should be ~300x the number of Amino Acids
    metropolis = args.get("metropolis", False) #Use the Metropolis Algorithm

    print('Using Updated Version of Metropolis Sampler')
    print(f"Running metropolis with strand={strand}, shift={shift}, temperature={temperature}, use_gradient={use_gradient}, gradient_start={gradient_start}, gradient_end={gradient_end}, num_mutations={num_mutations}, metropolis={metropolis}")
    return(strand, shift, temperature, use_gradient, gradient_start, gradient_end, num_mutations, metropolis, overhang_residues)    
