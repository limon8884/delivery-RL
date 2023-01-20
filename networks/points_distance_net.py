from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from utils import *
from dispatch.utils import *
from networks.encoders import PositionalEncoder, PointEncoder
from networks.utils import *

class PointDistNet(nn.Module):
    def __init__(self, enc_pos_dim=32, out_dim=128, device=None):
        super().__init__()
        self.pe = PointEncoder(enc_pos_dim, device=device)
        input_dim = 2 * enc_pos_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, 1)
        ).to(device=device)
        self.device = device

    def forward(self, pts1, pts2):
        x = torch.stack([torch.cat([self.pe(p1), self.pe(p2)], dim=-1) for p1, p2 in zip(pts1, pts2)])
        return self.mlp(x).squeeze(-1)