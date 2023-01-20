from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from utils import *
from dispatch.utils import *

class FeatureExtractor:
    def __init__(self) -> None:
        self.num_features = {
            'o': {
                'points': 2,
                'numbers': 1,
            },
            'c': {
                'points': 1,
                'numbers': 0,
            },
            'ar': {
                'points': 1,
                'numbers': 1,
            },
        }
        
    def __call__(self, item, current_time, item_type):
        if item_type == 'o':
            result = {
                'points': [item.point_from, item.point_to], 
                'numbers': [item.expire_time - current_time],
            }
            assert len(result['points']) == self.num_features['o']['points']
            assert len(result['numbers']) == self.num_features['o']['numbers']
        elif item_type == 'c':
            result = {
                'points': [item.position], 
                'numbers': [],
            } 
            assert len(result['points']) == self.num_features['c']['points']
            assert len(result['numbers']) == self.num_features['c']['numbers']
        elif item_type == 'ar':
            if item.eta_period:
                dist = distance(item.courier.position, item.order.point_from)\
                    + distance(item.order.point_from, item.order.point_to)
            else:
                dist = dist(item.courier.position, item.order.point_to)

            result = {
                'points': [item.order.point_to], 
                'numbers': [dist],
            }
            assert len(result['points']) == self.num_features['ar']['points']
            assert len(result['numbers']) == self.num_features['ar']['numbers']
        else: 
             raise Exception('not found type!')
        
        return result

class PointEncoder(nn.Module):
    def __init__(self, point_enc_dim, device):
        super().__init__()
        assert point_enc_dim % 4 == 0
        self.point_enc_dim = point_enc_dim
        self.sin_layer = nn.Linear(1, point_enc_dim // 4, device=device)
        self.cos_layer = nn.Linear(1, point_enc_dim // 4, device=device)
        self.device = device

        # nn.init.uniform_(self.sin_layer.weight, 0, 100)
        # nn.init.uniform_(self.cos_layer.weight, 0, 100)
        # self.trainable = trainable
        # self.freqs = torch.tensor([1 / 2**i for i in range(pos_enc_dim // 2)])

    def forward(self, p: Point):
        x = torch.tensor([p.x, p.y], dtype=torch.float32, device=self.device).unsqueeze(-1)
        return torch.cat([torch.sin(self.sin_layer(x)), torch.cos(self.cos_layer(x))], dim=-1).flatten()
    
        # pos_enc = torch.cat([torch.sin(f * self.freqs), torch.cos(f * self.freqs)], dim=-1)
        # if self.trainable:

# class NumericEncoder(nn.Module):
#     def __init__(self, point_enc_dim, device):
#         super().__init__()
#         assert point_enc_dim % 2 == 0
#         self.sin_layer = nn.Linear(1, point_enc_dim // 2, device=device)
#         self.cos_layer = nn.Linear(1, point_enc_dim // 2, device=device)
#         self.device = device

#     def forward(self, x: float):
#         x = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(-1)
#         return torch.cat([torch.sin(self.sin_layer(x)), torch.cos(self.cos_layer(x))], dim=-1).flatten()

class PositionalEncoder(nn.Module):
    def __init__(self, item_type: str, point_encoder: PointEncoder, num_enc_dim, out_dim, device=None):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.point_encoder = point_encoder
        self.number_encoder = nn.Linear(1, num_enc_dim, device=device)
        self.item_type = item_type

        pt_enc_dim = self.point_encoder.point_enc_dim
        input_dim = self.feature_extractor.num_features[item_type]['points'] * pt_enc_dim \
                + self.feature_extractor.num_features[item_type]['numbers'] * num_enc_dim 
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.mlp.to(device=device)
        self.device = device

    def forward(self, item, current_time):
        features = self.feature_extractor(item, current_time, self.item_type)
        encoded_points = [self.point_encoder(p) for p in features['points']]
        encoded_numbers = [
                self.number_encoder(torch.tensor([n], dtype=torch.float32, device=self.device)) 
                for n in features['numbers']
        ]

        return self.mlp(torch.cat(encoded_points + encoded_numbers))
    

class ScoringNet(nn.Module):
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
        self.order_enc = PositionalEncoder('o', self.point_encoder, num_enc_dim=number_enc_dim, out_dim=d_model, device=device)
        self.courier_enc = PositionalEncoder('c', self.point_encoder, num_enc_dim=number_enc_dim, out_dim=d_model, device=device)
        self.active_routes_enc = PositionalEncoder('ar', self.point_encoder, num_enc_dim=number_enc_dim, out_dim=d_model, device=device)
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

        o_tensor, o_mask = self.make_tensor(batch_o, item_type='o', current_time=current_time)
        c_tensor, c_mask = self.make_tensor(batch_c, item_type='c', current_time=current_time)
        ar_tensor, ar_mask = self.make_tensor(batch_ar, item_type='ar', current_time=current_time)
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

    def make_tensor(self, batch_sequences, item_type, current_time):
        samples = []
        lenghts = []
        for sequence in batch_sequences:
            if item_type == 'o':
                sample = torch.stack([self.order_enc(item, current_time) for item in sequence])
            elif item_type == 'c':
                sample = torch.stack([self.courier_enc(item, current_time) for item in sequence])
            elif item_type == 'ar':
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
            om_ones = torch.where(self.masks['o'], 0, 1).unsqueeze(-1).float()
            cm_ones = torch.where(self.masks['c'], 0, 1).unsqueeze(-2).float()
            return torch.matmul(om_ones, cm_ones).float()

def create_mask(lenghts, device):
    max_len = max(lenghts)
    with torch.no_grad():
        result = torch.arange(max_len, device=device).expand(len(lenghts), max_len) >= torch.tensor(lenghts, device=device).unsqueeze(1)
        return result