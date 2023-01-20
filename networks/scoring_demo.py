from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from utils import *
from dispatch.utils import *
from networks.encoders import PointEncoder, PositionalEncoderDemo
from networks.utils import *

class ScoringNetDemo(nn.Module):
    def __init__(self, 
            mode='default', 
            n_layers=2, 
            d_model=256, 
            n_head=2, 
            dim_ff=256, 
            point_enc_dim=32, 
            number_enc_dim=8,
            device=None):
        super().__init__()
        self.point_encoder = PointEncoder(point_enc_dim=point_enc_dim, device=device)
        self.order_enc = PositionalEncoderDemo('o', self.point_encoder, num_enc_dim=number_enc_dim, out_dim=d_model, device=device)
        self.courier_enc = PositionalEncoderDemo('c', self.point_encoder, num_enc_dim=number_enc_dim, out_dim=d_model, device=device)        
        self.mode = mode
        self.device = device

        self.encoders_OC = [
            {
                'o': nn.TransformerDecoderLayer(
                    d_model=d_model, 
                    nhead=n_head, 
                    dim_feedforward=dim_ff, 
                    batch_first=True,
                    device=self.device
                ),
                'c': nn.TransformerDecoderLayer(
                    d_model=d_model, 
                    nhead=n_head, 
                    dim_feedforward=dim_ff, 
                    batch_first=True,
                    device=self.device
                ),
            } 
            for _ in range(n_layers)
        ]
    
    def forward(self, batch: List[GambleTriple], current_time: int):
        self.masks = None
        tensors = self.make_tensors_and_create_masks(batch, current_time)

        ord, crr = tensors
        ord, crr = self.encoder_OC(ord, crr)
        scores = self.bipartite_scores(ord, crr)

        if self.mode == 'square':
            return -torch.square(scores)
        return -scores

    def make_tensors_and_create_masks(self, batch: List[GambleTriple], current_time):
        batch_c = []
        batch_o = []
        # batch_ar = []
        for triple in batch:
            batch_o.append(triple.orders)
            batch_c.append(triple.couriers)
            # batch_ar.append(triple.active_routes)

        o_tensor, o_mask = self.make_tensor(batch_o, item_type='o', current_time=current_time)
        c_tensor, c_mask = self.make_tensor(batch_c, item_type='c', current_time=current_time)
        # ar_tensor, ar_mask = self.make_tensor(batch_ar, item_type='ar', current_time=current_time)
        self.masks = {
            'o': o_mask,
            'c': c_mask,
            # 'ar': ar_mask
        }

        return o_tensor, c_tensor#, ar_tensor

    def make_tensor(self, batch_sequences, item_type, current_time):
        samples = []
        lenghts = []
        for sequence in batch_sequences:
            if item_type == 'o':
                sample = torch.stack([self.order_enc(item, current_time) for item in sequence])
            elif item_type == 'c':
                sample = torch.stack([self.courier_enc(item, current_time) for item in sequence])
            # elif item_type == 'ar':
            #     sample = torch.stack([self.active_routes_enc(item, current_time) for item in sequence])
            samples.append(sample)
            lenghts.append(len(sequence))
        
        tens = pad_sequence(samples, batch_first=True)
        return tens, create_mask(lenghts, self.device)

    def encoder_OC(self, ord, crr):
        for encoders in self.encoders_OC:
            new_ord = encoders['o'](ord, crr, tgt_key_padding_mask=self.masks['o'], memory_key_padding_mask=self.masks['c'])
            new_crr = encoders['c'](crr, ord, tgt_key_padding_mask=self.masks['c'], memory_key_padding_mask=self.masks['o'])
            ord = new_ord
            crr = new_crr
        return ord, crr

    def bipartite_scores(self, ord, crr):
        crr_t = torch.transpose(crr, 1, 2)
        return torch.matmul(ord, crr_t)

    def get_mask(self):
        with torch.no_grad():
            om_ones = torch.where(self.masks['o'], 0, 1).unsqueeze(-1).float()
            cm_ones = torch.where(self.masks['c'], 0, 1).unsqueeze(-2).float()
            return torch.matmul(om_ones, cm_ones).float()

