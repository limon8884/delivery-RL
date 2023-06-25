from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from utils import *
from dispatch.utils import *
from networks.encoders.positional_encoder import PositionalEncoder
from networks.encoders.point_encoder import PointEncoder
from networks.utils import *
from objects.gamble_triple import GambleTriple


class GambleTripleEncoder(nn.Module):
    def __init__(self, number_enc_dim, d_model, path_weights=None, point_encoder: PointEncoder=None, point_enc_dim=None, device=None):
        super().__init__()
        self.device = device
        self.d_model = d_model
        assert (point_encoder or point_enc_dim) is not None, 'one of them should not be None'

        if point_encoder is None:
            assert point_enc_dim is not None, 'one of them should not be None'
            point_encoder = PointEncoder(point_enc_dim, device)

        self.order_enc = PositionalEncoder('o', 
                                               point_encoder, 
                                               num_enc_dim=number_enc_dim, 
                                               out_dim=d_model,  
                                               device=device
                                               )
        self.courier_enc = PositionalEncoder('c', 
                                                 point_encoder, 
                                                 num_enc_dim=number_enc_dim, 
                                                 out_dim=d_model,  
                                                 device=device
                                                )
        self.ar_enc = PositionalEncoder('ar', 
                                            point_encoder, 
                                            num_enc_dim=number_enc_dim, 
                                            out_dim=d_model,  
                                            device=device
                                        )   
        
        if path_weights is not None:
            self.load_weights(path_weights)

    def load_weights(self, path: str):
        self.order_enc.load_state_dict(torch.load(path + '/o.pt', map_location=self.device))
        self.courier_enc.load_state_dict(torch.load(path + '/c.pt', map_location=self.device))
        self.ar_enc.load_state_dict(torch.load(path + '/ar.pt', map_location=self.device))
        
    def forward(self, gamble_triple: GambleTriple, current_time: int):
        '''
        Attention: the function adds BOS-fake items to every GambleTriple to strugle zero-input problem
        '''
        o_tensor, o_ids = self.make_tensor_with_ids(gamble_triple.orders, item_type='o', current_time=current_time)
        c_tensor, c_ids = self.make_tensor_with_ids(gamble_triple.couriers, item_type='c', current_time=current_time)
        ar_tensor, ar_ids = self.make_tensor_with_ids(gamble_triple.active_routes, item_type='ar', current_time=current_time)

        tensors = {
            'o': o_tensor,
            'c': c_tensor,
            'ar': ar_tensor
        }
        ids = {
            'o': o_ids,
            'c': c_ids,
            'ar': ar_ids
        }

        return tensors, ids

    def make_tensor_with_ids(self, sequence, item_type, current_time):
        '''
        Input: a sequence of items
        Output: an embedding tensor of shape [len(seq) + 1, emb_size] and ids tensor of shape [len(seq) + 1]
        The function adds a BOS item with zeros-embedding
        '''
        
        bos_tensor = torch.zeros(self.d_model, device=self.device, dtype=torch.float)
        if item_type == 'o':
            tensors = torch.stack([bos_tensor] + [self.order_enc(item, current_time) for item in sequence])
        elif item_type == 'c':
            tensors = torch.stack([bos_tensor] + [self.courier_enc(item, current_time) for item in sequence])
        elif item_type == 'ar':
            tensors = torch.stack([bos_tensor] + [self.ar_enc(item, current_time) for item in sequence])   
        else:
            raise RuntimeError('unkown item type')
        
        ids = torch.tensor([-1] + [item.id for item in sequence], dtype=torch.int, device=self.device)

        return tensors, ids
    