import os
from protein_mpnn.protein_mpnn_utils import MODEL_CONFIG    


def decide_model_weights():
    LONGLEAF = 'longleaf' in os.getcwd()

    if longeaf:
        weight_path = '/proj/kuhl_lab/alphafold/alphafold/alphafold/data/'
    else:
        weight_path = '/home/nzrandolph/git/alphfold/alphafold/alphafold/data/'

    return weight_path


def run_protein_mpnn(args):

    if args.path_to_model_weights:
        model_folder_path = args.path_to_model_weights
        if model_folder_path[-1] != '/':
            model_folder_path = model_folder_path + '/'
    else:
         file_path = os.path.realpath(__file__)
         k = file_path.rfind("/")
         model_folder_path = file_path[:k] + '/model_weights/'

    checkpoint_path = model_folder_path + f'{args.model_name}.pt'
    folder_for_outputs = args.out_folder

    # What does args.batch_size mean? in terms of performance, etc.?
    # For now I'm leaving it be.
    NUM_BATCHES = args.num_seq_per_target//args.batch_size
    BATCH_COPIES = args.batch_size
    temperatures = [float(item) for item in args.sampling_temp.split()]
    
