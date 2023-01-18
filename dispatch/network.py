from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from utils import *
from dispatch.utils import *

def ExtractOrderFeatures(order: Order, current_time: int) -> List[float]:
    return [
            order.point_from.x, 
            order.point_from.y, 
            order.point_to.x, 
            order.point_to.y, 
            order.expire_time - current_time
    ]

def ExtractCourierFeatures(courier: Courier, current_time: int=0) -> List[float]:
    return [
            courier.position.x, 
            courier.position.y
    ]

def ExtractActiveRoutesFeatures(active_route: ActiveRoute, current_time: int=0) -> List[float]:
    if active_route.eta_period:
        dist = distance(active_route.courier.position, active_route.order.point_from)\
            + distance(active_route.order.point_from, active_route.order.point_to)
    else:
        dist = dist(active_route.courier.position, active_route.order.point_to)
    return [
        active_route.order.point_to.x, 
        active_route.order.point_to.y,
        dist
    ]

class PositionalEncoder(nn.Module):
    def __init__(self, type, pos_enc_dim, out_dim, device=None):
        super().__init__()
        assert pos_enc_dim % 2 == 0
        self.sin_layer = nn.Linear(1, pos_enc_dim // 2, device=device)
        self.cos_layer = nn.Linear(1, pos_enc_dim // 2, device=device)
        # self.trainable = trainable
        # self.freqs = torch.tensor([1 / 2**i for i in range(pos_enc_dim // 2)])
        if type == 'o':
            self.feature_extractor = ExtractOrderFeatures
            num_features = 5
        elif type == 'c':
            self.feature_extractor = ExtractCourierFeatures
            num_features = 2
        elif type == 'ar':
            self.feature_extractor = ExtractActiveRoutesFeatures
            num_features = 3
        else:
            raise Exception('not found type!')

        self.mlp = nn.Sequential(
            nn.LayerNorm(num_features * pos_enc_dim),
            nn.LeakyReLU(),
            nn.Linear(num_features * pos_enc_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.mlp.to(device=device)
        self.device = device

    def forward(self, item, current_time):
        x = self.feature_extractor(item, current_time)
        x = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(-1)
        # pos_enc = torch.cat([torch.sin(f * self.freqs), torch.cos(f * self.freqs)], dim=-1)
        # if self.trainable:
        pos_enc = torch.cat([torch.sin(self.sin_layer(x)), torch.cos(self.sin_layer(x))], dim=-1)
    
        return self.mlp(torch.flatten(pos_enc))
    

class ScoringNet(nn.Module):
    def __init__(self, mode='default', n_layers=2, d_model=256, n_head=2, dim_ff=256, pos_enc_dim=64, device=None):
        super().__init__()
        self.order_enc = PositionalEncoder(type='o', out_dim=d_model, pos_enc_dim=pos_enc_dim, device=device)
        self.courier_enc = PositionalEncoder(type='c', out_dim=d_model, pos_enc_dim=pos_enc_dim, device=device)
        self.active_routes_enc = PositionalEncoder(type='ar', out_dim=d_model, pos_enc_dim=pos_enc_dim, device=device)
        self.mode = mode
        self.device = device

        self.encoders_AR = {
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
            )
        }
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

        ord, crr = self.encoder_AR(tensors)
        ord, crr = self.encoder_OC(ord, crr)
        scores = self.bipartite_scores(ord, crr)

        if self.mode == 'square':
            return -torch.square(scores)
        return -scores

    def make_tensors_and_create_masks(self, batch: List[GambleTriple], current_time):
        batch_c = []
        batch_o = []
        batch_ar = []
        for triple in batch:
            batch_o.append(triple.orders)
            batch_c.append(triple.couriers)
            batch_ar.append(triple.active_routes)

        o_tensor, o_mask = self.make_tensor(batch_o, type='o', current_time=current_time)
        c_tensor, c_mask = self.make_tensor(batch_c, type='c', current_time=current_time)
        ar_tensor, ar_mask = self.make_tensor(batch_ar, type='ar', current_time=current_time)
        self.masks = {
            'o': o_mask,
            'c': c_mask,
            'ar': ar_mask
        }

        return o_tensor, c_tensor, ar_tensor

    # def make_order_tensor(self, orders: List[List[Order]], current_time: int):
    #     samples = []
    #     lenghts = []
    #     for order_list in orders:
    #         samples.append(torch.stack([self.order_enc(order, current_time) for order in order_list]))
    #         lenghts.append(len(order_list))
        
    #     tens = pad_sequence(samples, batch_first=True)
    #     return tens, create_mask(lenghts)

    def make_tensor(self, batch_sequences, type, current_time):
        samples = []
        lenghts = []
        for sequence in batch_sequences:
            if type == 'o':
                sample = torch.stack([self.order_enc(item, current_time) for item in sequence])
            elif type == 'c':
                sample = torch.stack([self.courier_enc(item, current_time) for item in sequence])
            elif type == 'ar':
                sample = torch.stack([self.active_routes_enc(item, current_time) for item in sequence])
            samples.append(sample)
            lenghts.append(len(sequence))
        
        tens = pad_sequence(samples, batch_first=True)
        return tens, create_mask(lenghts, self.device)

    def encoder_AR(self, tensors):
        ord, crr, ar = tensors
        new_ord = self.encoders_AR['o'](ord, ar, tgt_key_padding_mask=self.masks['o'], memory_key_padding_mask=self.masks['ar'])
        new_crr = self.encoders_AR['c'](crr, ar, tgt_key_padding_mask=self.masks['c'], memory_key_padding_mask=self.masks['ar'])

        return new_ord, new_crr

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
            om_ones = torch.where(self.masks['o'], 0, 1).unsqueeze(-1)
            cm_ones = torch.where(self.masks['c'], 0, 1).unsqueeze(-2)
            return torch.matmul(om_ones, cm_ones).float()

def create_mask(lenghts, device):
    max_len = max(lenghts)
    with torch.no_grad():
        result = torch.arange(max_len).expand(len(lenghts), max_len) >= torch.tensor(lenghts).unsqueeze(1)
        result.to(device)
        return result