import torch
import torch.nn as nn
import numpy as np

from src.objects import Point, Courier, Order, Route, Claim, Gamble


X_COORD_MEAN = 37.60
Y_COORD_MEAN = 55.75
X_COORD_STD = 0.25
Y_COORD_STD = 0.17


def normalize_coords(points: torch.Tensor, device) -> torch.Tensor:
    mean = torch.tensor([X_COORD_MEAN, Y_COORD_MEAN]).unsqueeze(0).to(device)
    std = torch.tensor([X_COORD_STD, Y_COORD_STD]).unsqueeze(0).to(device)
    points = (points - mean) / std
    return points


class BasePointEncoder(nn.Module):
    def __init__(self, point_emb_dim: int, device, normalize_coords=False):
        super().__init__()
        assert point_emb_dim % 4 == 0
        self.point_emb_dim = point_emb_dim
        self.device = device
        self.normalize_coords = normalize_coords

    def forward(self, points: torch.Tensor):
        raise NotImplementedError()


class LinearPointEncoder(BasePointEncoder):
    def __init__(self, point_emb_dim: int, device, normalize_coords=False, trainable=True, **kwargs):
        super().__init__(point_emb_dim, device, normalize_coords)
        self.layer = nn.Linear(2, point_emb_dim).to(device=self.device)
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, points: torch.Tensor):
        assert points.shape[-1] == 2, points.shape
        if self.normalize_coords:
            points = normalize_coords(points, self.device)
        x = self.layer(points)
        return x


class FreqPointEncoderUnited(BasePointEncoder):
    def __init__(self, point_emb_dim: int, device, normalize_coords=False, trainable=True, **kwargs):
        super().__init__(point_emb_dim, device, normalize_coords)
        self.layer = nn.Linear(2, point_emb_dim).to(device=self.device)
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, points: torch.Tensor):
        assert points.shape[-1] == 2, points.shape
        if self.normalize_coords:
            points = normalize_coords(points, self.device)
        
        x = self.layer(points)
        x = torch.cat([
            torch.sin(x[..., :self.point_emb_dim // 2]),
            torch.cos(x[..., self.point_emb_dim // 2:])
        ], dim=-1)
        return x


class FreqPointEncoder(BasePointEncoder):
    def __init__(self, point_emb_dim: int, device, normalize_coords=False, trainable=True, **kwargs):
        super().__init__(point_emb_dim, device, normalize_coords)
        self.sin_layer_x = nn.Linear(1, point_emb_dim // 4, device=device)
        self.cos_layer_x = nn.Linear(1, point_emb_dim // 4, device=device)
        self.sin_layer_y = nn.Linear(1, point_emb_dim // 4, device=device)
        self.cos_layer_y = nn.Linear(1, point_emb_dim // 4, device=device)
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, points: torch.Tensor):
        assert points.shape[-1] == 2, points.shape
        if self.normalize_coords:
            points = normalize_coords(points, self.device)
        x = torch.cat([
            torch.sin(self.sin_layer_x(points[..., 0].unsqueeze(-1))),
            torch.cos(self.cos_layer_x(points[..., 0].unsqueeze(-1))),
            torch.sin(self.sin_layer_y(points[..., 1].unsqueeze(-1))),
            torch.cos(self.cos_layer_y(points[..., 1].unsqueeze(-1))),
        ], dim=-1)
        return x


class SupportPointEncoder(BasePointEncoder):
    def __init__(self, point_emb_dim: int, support_points_interval: float, device, trainable=True, **kwargs):
        super().__init__(point_emb_dim, device, False)
        # n_support_points = int((2 * X_COORD_STD) * (2 * Y_COORD_STD) / support_points_interval**2)
        x_num = int(2 * X_COORD_STD / support_points_interval) + 1
        y_num = int(2 * Y_COORD_STD / support_points_interval) + 1
        self.support_points = torch.zeros((x_num * y_num, 2), requires_grad=False).to(self.device)
        for i in range(x_num):
            for j in range(y_num):
                x = X_COORD_MEAN - X_COORD_STD + i * support_points_interval
                y = Y_COORD_MEAN - Y_COORD_STD + j * support_points_interval
                idx = i * y_num + j
                self.support_points[idx, 0] = x
                self.support_points[idx, 1] = y
        self.support_embeddings = nn.Embedding(x_num * y_num, point_emb_dim).to(self.device)
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, points: torch.Tensor):
        assert points.shape[-1] == 2, points.shape
        dists = torch.square(points[:, None, :] - self.support_points[None, :, :]).sum(dim=-1)
        support_point_idxes = torch.argmin(dists, dim=-1)
        x = self.support_embeddings(support_point_idxes)
        return x


# class PointEncoder(nn.Module):
#     def __init__(self, point_emb_dim: int, device, normalize_coords=False):
#         super().__init__()
#         assert point_emb_dim % 4 == 0
#         self.sin_layer_x = nn.Linear(1, point_emb_dim // 4, device=device)
#         self.cos_layer_x = nn.Linear(1, point_emb_dim // 4, device=device)
#         self.sin_layer_y = nn.Linear(1, point_emb_dim // 4, device=device)
#         self.cos_layer_y = nn.Linear(1, point_emb_dim // 4, device=device)
#         self.device = device
#         self.normalize_coords = normalize_coords

#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, points: torch.Tensor):
#         assert points.shape[-1] == 2, points.shape
#         if self.normalize_coords:
#             mean = torch.tensor([X_COORD_MEAN, Y_COORD_MEAN]).unsqueeze(0).to(self.device)
#             std = torch.tensor([X_COORD_STD, Y_COORD_STD]).unsqueeze(0).to(self.device)
#             points = (points - mean) / std
#         with torch.no_grad():
#             return torch.cat([
#                 torch.sin(self.sin_layer_x(points[..., 0].unsqueeze(-1))),
#                 torch.cos(self.cos_layer_x(points[..., 0].unsqueeze(-1))),
#                 torch.sin(self.sin_layer_y(points[..., 1].unsqueeze(-1))),
#                 torch.cos(self.cos_layer_y(points[..., 1].unsqueeze(-1))),
#             ], dim=-1)

