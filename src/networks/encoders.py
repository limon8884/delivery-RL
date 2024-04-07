import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta

from src.objects import Point, Courier, Order, Route, Claim, Gamble


class PointEncoder(nn.Module):
    def __init__(self, point_emb_dim: int, device):
        super().__init__()
        assert point_emb_dim % 4 == 0
        self.sin_layer_x = nn.Linear(1, point_emb_dim // 4, device=device)
        self.cos_layer_x = nn.Linear(1, point_emb_dim // 4, device=device)
        self.sin_layer_y = nn.Linear(1, point_emb_dim // 4, device=device)
        self.cos_layer_y = nn.Linear(1, point_emb_dim // 4, device=device)
        self.device = device

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, points: torch.FloatTensor):
        assert points.shape[-1] == 2, points.shape
        with torch.no_grad():
            return torch.cat([
                torch.sin(self.sin_layer_x(points[..., 0].unsqueeze(-1))),
                torch.cos(self.cos_layer_x(points[..., 0].unsqueeze(-1))),
                torch.sin(self.sin_layer_y(points[..., 1].unsqueeze(-1))),
                torch.cos(self.cos_layer_y(points[..., 1].unsqueeze(-1))),
            ], dim=-1)


class NumberEncoder(nn.Module):
    def __init__(self, numbers_np_dim, number_embedding_dim, device, dropout) -> None:
        super().__init__()
        self.numbers_np_dim = numbers_np_dim
        self.number_embedding_dim = number_embedding_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(normalized_shape=(numbers_np_dim,)),
            nn.Linear(numbers_np_dim, number_embedding_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(number_embedding_dim, number_embedding_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout),
        ).to(device)
        self.device = device

    def forward(self, numbers_np: np.ndarray) -> torch.FloatTensor:
        x = torch.tensor(numbers_np, device=self.device, dtype=torch.float)
        x = self.mlp(x)
        return x


# class CoordSinCosEncoder(nn.Module):
#     def __init__(self, coords_np_dim, coords_embedding_dim, device, dropout):
#         super().__init__()
#         assert coords_embedding_dim % 2 == 0
#         self.coords_np_dim = coords_np_dim
#         self.coords_embedding_dim = coords_embedding_dim
#         self.bn = nn.BatchNorm1d(num_features=coords_np_dim, device=device)
#         self.sin_layer = nn.Linear(coords_np_dim, coords_embedding_dim // 2, device=device)
#         self.cos_layer = nn.Linear(coords_np_dim, coords_embedding_dim // 2, device=device)
#         self.device = device

#     def forward(self, coords_np: np.ndarray) -> torch.FloatTensor:
#         assert coords_np.ndim == 2 and coords_np.shape[1] == self.coords_np_dim
#         # x = torch.FloatTensor(coords_np, device=self.device)
#         x = torch.tensor(coords_np, device=self.device, dtype=torch.float)
#         if coords_np.shape[0] != 1 or not self.training:
#             x = self.bn(x)
#         return torch.cat([
#             torch.sin(self.sin_layer(x)),
#             torch.cos(self.cos_layer(x)),
#         ], dim=-1)


class CoordMLPEncoder(nn.Module):
    def __init__(self, coords_np_dim, point_embedding_dim, coords_embedding_dim, device, dropout):
        super().__init__()
        assert point_embedding_dim % 4 == 0
        self.pe = PointEncoder(point_embedding_dim, device=device)
        self.coords_np_dim = coords_np_dim
        input_dim = coords_np_dim // 2 * point_embedding_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(normalized_shape=(input_dim,)),
            nn.Linear(input_dim, coords_embedding_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(coords_embedding_dim, coords_embedding_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout),
        ).to(device)
        self.device = device

    def forward(self, coords_np: np.ndarray) -> torch.FloatTensor:
        assert coords_np.ndim == 2 and coords_np.shape[1] == self.coords_np_dim
        bs = coords_np.shape[0]
        points = torch.tensor(coords_np.reshape(bs, self.coords_np_dim // 2, 2),
                              dtype=torch.float, device=self.device)
        points_embs = self.pe(points).reshape(bs, -1)
        return self.mlp(points_embs)


class ItemEncoder(nn.Module):
    def __init__(self, feature_types: dict[tuple[int, int], str], item_embedding_dim: int, **kwargs) -> None:
        super().__init__()
        dropout = kwargs['dropout']
        device = kwargs['device']
        self.item_embedding_dim = item_embedding_dim
        coords_idxs = []
        numbers_idxs = []
        for (l, r), type_ in feature_types.items():
            if type_ == 'coords':
                coords_idxs.extend(list(range(l, r)))
            elif type_ == 'numbers':
                numbers_idxs.extend(list(range(l, r)))
            else:
                raise RuntimeError('no such type of feature')
        self.coords_idxs = np.array(sorted(coords_idxs))
        self.numbers_idxs = np.array(sorted(numbers_idxs))

        point_embedding_dim = kwargs['point_embedding_dim']
        coords_embedding_dim = len(coords_idxs) // 2 * kwargs['cat_points_embedding_dim']
        number_embedding_dim = len(numbers_idxs)
        self.coord_encoder = CoordMLPEncoder(coords_np_dim=len(self.coords_idxs),
                                             point_embedding_dim=point_embedding_dim,
                                             coords_embedding_dim=coords_embedding_dim,
                                             device=device, dropout=dropout)
        self.number_encoder = NumberEncoder(numbers_np_dim=len(self.numbers_idxs),
                                            number_embedding_dim=number_embedding_dim,
                                            device=device, dropout=dropout)
        self.mlp = self.mlp = nn.Sequential(
            nn.Linear(coords_embedding_dim + number_embedding_dim, item_embedding_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(item_embedding_dim, item_embedding_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(item_embedding_dim, item_embedding_dim),
        ).to(device)

    def forward(self, item_np: np.ndarray) -> torch.FloatTensor:
        assert item_np.ndim == 2 and item_np.shape[1] == len(self.coords_idxs) + len(self.numbers_idxs), (
            item_np.ndim, len(self.coords_idxs), len(self.numbers_idxs), item_np.shape[1]
            )
        coords_emb = self.coord_encoder(item_np[:, self.coords_idxs])
        numbers_emb = self.number_encoder(item_np[:, self.numbers_idxs])
        x = torch.cat([coords_emb, numbers_emb], dim=-1)
        x = self.mlp(x)
        return x


class GambleEncoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        order_embedding_dim = kwargs['order_embedding_dim']
        claim_embedding_dim = kwargs['claim_embedding_dim']
        courier_embedding_dim = kwargs['courier_embedding_dim']
        point_embedding_dim = kwargs['point_embedding_dim']
        cat_points_embedding_dim = kwargs['cat_points_embedding_dim']
        self.max_num_points_in_route = kwargs['max_num_points_in_route']
        device = kwargs['device']
        dropout = kwargs['dropout']
        self.claim_encoder = ItemEncoder(
            feature_types=Claim.numpy_feature_types(),
            item_embedding_dim=claim_embedding_dim,
            point_embedding_dim=point_embedding_dim * 2,
            cat_points_embedding_dim=cat_points_embedding_dim,
            dropout=dropout,
            device=device,
        )
        self.courier_encoder = ItemEncoder(
            feature_types=Courier.numpy_feature_types(),
            item_embedding_dim=courier_embedding_dim,
            point_embedding_dim=point_embedding_dim * 1,
            cat_points_embedding_dim=cat_points_embedding_dim,
            dropout=dropout,
            device=device,
        )
        self.order_encoder = ItemEncoder(
            feature_types=Order.numpy_feature_types(max_num_points_in_route=self.max_num_points_in_route),
            item_embedding_dim=order_embedding_dim,
            point_embedding_dim=point_embedding_dim * (1 + self.max_num_points_in_route),
            cat_points_embedding_dim=cat_points_embedding_dim,
            dropout=dropout,
            device=device,
        )

    def forward(self, gamble_np_dict: dict[str, np.ndarray | None]) -> dict[str, torch.FloatTensor | None]:
        return {
            'crr': self.courier_encoder(gamble_np_dict['crr']) if gamble_np_dict['crr'] is not None else None,
            'clm': self.claim_encoder(gamble_np_dict['clm']),
            'ord': self.order_encoder(gamble_np_dict['ord']) if gamble_np_dict['ord'] is not None else None,
        }
