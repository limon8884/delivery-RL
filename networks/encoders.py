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
        self.sin_layer_x = nn.Linear(1, point_enc_dim // 4, device=device)
        self.cos_layer_x = nn.Linear(1, point_enc_dim // 4, device=device)
        self.sin_layer_y = nn.Linear(1, point_enc_dim // 4, device=device)
        self.cos_layer_y = nn.Linear(1, point_enc_dim // 4, device=device)
        self.device = device

        # nn.init.uniform_(self.sin_layer.weight, 0, 100)
        # nn.init.uniform_(self.cos_layer.weight, 0, 100)
        # self.trainable = trainable
        # self.freqs = torch.tensor([1 / 2**i for i in range(pos_enc_dim // 2)])

    def forward(self, p: Point):
        x = torch.tensor([p.x], dtype=torch.float32, device=self.device)
        y = torch.tensor([p.y], dtype=torch.float32, device=self.device)
        return torch.cat([
            torch.sin(self.sin_layer_x(x)), 
            torch.cos(self.cos_layer_x(x)),
            torch.sin(self.sin_layer_y(y)), 
            torch.cos(self.cos_layer_y(y)),
        ], dim=-1).flatten()
    
        # pos_enc = torch.cat([torch.sin(f * self.freqs), torch.cos(f * self.freqs)], dim=-1)
        # if self.trainable:

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
    
class PositionalEncoderDemo(nn.Module):
    def __init__(self, item_type: str, point_encoder: PointEncoder, num_enc_dim, out_dim, use_grad=False, device=None):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.point_encoder = point_encoder
        self.number_encoder = nn.Linear(1, num_enc_dim, device=device)
        self.item_type = item_type
        self.use_grad = use_grad

        pt_enc_dim = self.point_encoder.point_enc_dim
        input_dim = pt_enc_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.mlp.to(device=device)
        self.device = device

    def forward(self, item, current_time):
        features = self.feature_extractor(item, current_time, self.item_type)
        encoded_point = self.point_encoder(features['points'][0])
        if not self.use_grad:
            encoded_point = encoded_point.detach()

        return self.mlp(encoded_point)
        
