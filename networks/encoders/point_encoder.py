import torch
import torch.nn as nn
from objects.point import Point


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

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, p: Point):
        with torch.no_grad():
            x = torch.tensor([p.x], dtype=torch.float32, device=self.device)
            y = torch.tensor([p.y], dtype=torch.float32, device=self.device)
            return torch.cat([
                torch.sin(self.sin_layer_x(x)),
                torch.cos(self.cos_layer_x(x)),
                torch.sin(self.sin_layer_y(y)),
                torch.cos(self.cos_layer_y(y)),
            ], dim=-1).flatten()
