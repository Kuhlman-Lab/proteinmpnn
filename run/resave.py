import torch
from protein_mpnn.protein_mpnn_utils import ProteinMPNN
import os

ckpt_files = os.listdir('model_weights')

for ckpt_file in ckpt_files:
    if 'ca' in ckpt_file:
        continue

    print(ckpt_file)
    ckpt = torch.load(os.path.join('model_weights', ckpt_file))
    num_edges = ckpt['num_edges']
    print('Number of edges:', ckpt['num_edges'])
    noise_level = ckpt['noise_level']
    print(f'Training noise level: {noise_level}')

    model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, augment_eps=noise_level, k_neighbors=num_edges)
    model.load_state_dict(ckpt['model_state_dict'])

    model_state_dict = model.state_dict()
    ckpt = {'num_edges': num_edges, 'noise_level': noise_level,
            'model_state_dict': model_state_dict}
    
    torch.save(ckpt, os.path.join('new_weights', ckpt_file), _use_new_zipfile_serialization=False)
    
