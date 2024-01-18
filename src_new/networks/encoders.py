import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta

from src_new.objects import Point, Courier, Order, Route, Claim, Gamble


class NumberEncoder(nn.Module):
    def __init__(self, numbers_np_dim, number_embedding_dim, device) -> None:
        super().__init__()
        self.numbers_np_dim = numbers_np_dim
        self.number_embedding_dim = number_embedding_dim
        self.bn = nn.BatchNorm1d(num_features=numbers_np_dim)
        self.layer = nn.Linear(numbers_np_dim, number_embedding_dim, device=device)
        self.activation_func = nn.LeakyReLU()
        self.device = device

    def forward(self, numbers_np: np.ndarray) -> torch.FloatTensor:
        x = torch.FloatTensor(numbers_np, device=self.device)
        if numbers_np.shape[0] != 1 or not self.training:
            x = self.bn(x)
        x = self.layer(x)
        x = self.activation_func(x)
        return x


class CoordEncoder(nn.Module):
    def __init__(self, coords_np_dim, coords_embedding_dim, device):
        super().__init__()
        assert coords_embedding_dim % 2 == 0
        self.coords_np_dim = coords_np_dim
        self.coords_embedding_dim = coords_embedding_dim
        self.bn = nn.BatchNorm1d(num_features=coords_np_dim)
        self.sin_layer = nn.Linear(coords_np_dim, coords_embedding_dim // 2, device=device)
        self.cos_layer = nn.Linear(coords_np_dim, coords_embedding_dim // 2, device=device)
        self.device = device

    def forward(self, coords_np: np.ndarray) -> torch.FloatTensor:
        assert coords_np.ndim == 2 and coords_np.shape[1] == self.coords_np_dim
        x = torch.FloatTensor(coords_np, device=self.device)
        if coords_np.shape[0] != 1 or not self.training:
            x = self.bn(x)
        return torch.cat([
            torch.sin(self.sin_layer(x)),
            torch.cos(self.cos_layer(x)),
        ], dim=-1)


class ItemEncoder(nn.Module):
    def __init__(self, feature_types: dict[tuple[int, int], str], item_embedding_dim: int, **kwargs) -> None:
        super().__init__()
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

        self.coord_encoder = CoordEncoder(coords_np_dim=len(self.coords_idxs),
                                          coords_embedding_dim=kwargs['point_embedding_dim'],
                                          device=kwargs['device'])
        self.number_encoder = NumberEncoder(numbers_np_dim=len(self.numbers_idxs),
                                            number_embedding_dim=kwargs['number_embedding_dim'],
                                            device=kwargs['device'])
        self.mlp = self.mlp = nn.Sequential(
            nn.Linear(kwargs['point_embedding_dim'] + kwargs['number_embedding_dim'],
                      item_embedding_dim, device=kwargs['device']),
            nn.LeakyReLU(),
            nn.Linear(item_embedding_dim, item_embedding_dim, device=kwargs['device']),
            nn.LeakyReLU(),
        )

    def forward(self, item_np: np.ndarray) -> torch.FloatTensor:
        assert item_np.ndim == 2 and item_np.shape[1] == len(self.coords_idxs) + len(self.numbers_idxs)
        coords_emb = self.coord_encoder(item_np[:, self.coords_idxs])
        numbers_emb = self.number_encoder(item_np[:, self.numbers_idxs])
        x = torch.cat([coords_emb, numbers_emb], dim=-1)
        x = self.mlp(x)
        return x


# class CourierEncoder(nn.Module):
#     def __init__(self, courier_np_dim, courier_embedding_dim, point_embedding_dim, number_embedding_dim, device) -> None:
#         super().__init__()
#         self.courier_np_dim = courier_np_dim
#         self.courier_embedding_dim = courier_embedding_dim
#         self.coord_encoder = PointEncoder(point_embedding_dim=point_embedding_dim, device=device)
#         self.time_encoder = NumberEncoder(number_embedding_dim=number_embedding_dim, device=device)
#         self.mlp = nn.Sequential(
#             nn.Linear(point_embedding_dim + number_embedding_dim, courier_embedding_dim, device=device),
#             nn.LeakyReLU(),
#             nn.Linear(courier_embedding_dim, courier_embedding_dim, device=device),
#             nn.LeakyReLU(),
#         )

#     def forward(self, crr_np: np.ndarray, dttm: datetime) -> torch.FloatTensor:
#         assert crr_np.ndim == 2 and crr_np.shape[1] == self.courier_np_dim  # [batch_dim, courier_np_dim]
        
#         # coords_emb = self.coord_encoder(crr_np[])
#         time_emb = self.time_encoder((dttm - crr.start_dttm).total_seconds())
#         x = torch.cat([coords_emb, time_emb], dim=-1)
#         x = self.mlp(x)
#         return x


# class RouteEncoder(nn.Module):
#     def __init__(self, route_embedding_dim, point_embedding_dim,
#                  number_embedding_dim, max_num_points_in_route, device) -> None:
#         super().__init__()
#         self.route_embedding_dim = route_embedding_dim
#         self.max_num_points_in_route = max_num_points_in_route
#         self.coord_encoder = PointEncoder(point_embedding_dim=point_embedding_dim, device=device)
#         self.point_type_encoder = NumberEncoder(number_embedding_dim=number_embedding_dim, device=device)

#         raw_emb_dim = (point_embedding_dim + number_embedding_dim) * max_num_points_in_route
#         self.mlp = nn.Sequential(
#             nn.Linear(raw_emb_dim, route_embedding_dim),
#             nn.LeakyReLU(),
#             nn.Linear(route_embedding_dim, route_embedding_dim),
#             nn.LeakyReLU(),
#         )

#     def forward(self, route: Route) -> torch.FloatTensor:
#         assert len(route.route_points) <= self.max_num_points_in_route
#         embeddings_list = []
#         for point in route.route_points:
#             embeddings_list.append(self.coord_encoder(point.point))
#             embeddings_list.append(self.point_type_encoder(point.point_type.value))
#         zeroes_left = (self.coord_encoder.point_embedding_dim + self.point_type_encoder.number_embedding_dim) \
#             * (self.max_num_points_in_route - len(route.route_points))
#         embeddings_list.append(torch.zeros(size=(zeroes_left,)))
#         x = torch.cat(embeddings_list, dim=-1)
#         x = self.mlp(x)
#         return x


# class CourierOrderEncoder(nn.Module):
#     def __init__(self,
#                  embedding_dim,
#                  courier_embedding_dim,
#                  route_embedding_dim,
#                  point_embedding_dim,
#                  number_embedding_dim,
#                  max_num_points_in_route,
#                  device,
#                  ) -> None:
#         super().__init__()
#         self.courier_encoder = CourierEncoder(courier_embedding_dim, point_embedding_dim, number_embedding_dim, device)
#         self.route_encoder = RouteEncoder(route_embedding_dim, point_embedding_dim, number_embedding_dim,
#                                           max_num_points_in_route, device)
#         self.time_encoder = NumberEncoder(number_embedding_dim, device)
#         self.embedding_dim = embedding_dim
#         self.mlp = nn.Sequential(
#             nn.Linear(courier_embedding_dim + route_embedding_dim, embedding_dim, device=device),
#             nn.LeakyReLU(),
#             nn.Linear(embedding_dim, embedding_dim, device=device),
#             nn.LeakyReLU(),
#         )

#     def forward(self, item: Order | Courier, dttm: datetime) -> torch.FloatTensor:
#         if isinstance(item, Courier):
#             courier = item
#             route = Route([])
#         elif isinstance(item, Order):
#             courier = item.courier
#             route = item.route
#         else:
#             raise RuntimeError('item should be either courier or order')
#         crr_emb = self.courier_encoder(courier, dttm)
#         rt_emb = self.route_encoder(route)
#         both_emb = torch.cat([crr_emb, rt_emb], dim=-1)
#         emb = self.mlp(both_emb)
#         return emb


# class ClaimEncoder(nn.Module):
#     def __init__(self, claim_embedding_dim, point_embedding_dim, number_embedding_dim, device) -> None:
#         super().__init__()
#         self.claim_embedding_dim = claim_embedding_dim
#         self.source_coord_encoder = PointEncoder(point_embedding_dim=point_embedding_dim, device=device)
#         self.destination_coord_encoder = PointEncoder(point_embedding_dim=point_embedding_dim, device=device)
#         self.time_encoder = NumberEncoder(number_embedding_dim=number_embedding_dim, device=device)
#         self.source_waiting_encoder = NumberEncoder(number_embedding_dim=number_embedding_dim, device=device)
#         self.destination_waiting_encoder = NumberEncoder(number_embedding_dim=number_embedding_dim, device=device)
#         self.mlp = nn.Sequential(
#             nn.Linear(point_embedding_dim * 2 + number_embedding_dim * 3, claim_embedding_dim, device=device),
#             nn.LeakyReLU(),
#             nn.Linear(claim_embedding_dim, claim_embedding_dim, device=device),
#             nn.LeakyReLU(),
#         )

#     def forward(self, claim: Claim, dttm: datetime) -> torch.FloatTensor:
#         embeddings_list = []
#         embeddings_list.append(self.source_coord_encoder(claim.source_point))
#         embeddings_list.append(self.destination_coord_encoder(claim.destination_point))
#         embeddings_list.append(self.time_encoder((dttm - claim.creation_dttm).total_seconds()))
#         embeddings_list.append(self.source_waiting_encoder(claim.waiting_on_point_source.total_seconds()))
#         embeddings_list.append(self.destination_waiting_encoder(claim.waiting_on_point_destination.total_seconds()))
#         x = torch.cat(embeddings_list, dim=-1)
#         emb = self.mlp(x)
#         return emb


class GambleEncoder(nn.Module):
    def __init__(self,
                 order_embedding_dim,
                 claim_embedding_dim,
                 courier_embedding_dim,
                 point_embedding_dim,
                 number_embedding_dim,
                 max_num_points_in_route,
                 device,
                 ) -> None:
        super().__init__()
        self.claim_encoder = ItemEncoder(
            feature_types=Claim.numpy_feature_types(),
            item_embedding_dim=claim_embedding_dim,
            point_embedding_dim=point_embedding_dim * 2,
            number_embedding_dim=number_embedding_dim,
            device=device,
        )
        self.courier_encoder = ItemEncoder(
            feature_types=Courier.numpy_feature_types(),
            item_embedding_dim=courier_embedding_dim,
            point_embedding_dim=point_embedding_dim * 1,
            number_embedding_dim=number_embedding_dim,
            device=device,
        )
        self.order_encoder = ItemEncoder(
            feature_types=Order.numpy_feature_types(max_num_points_in_route=max_num_points_in_route),
            item_embedding_dim=order_embedding_dim,
            point_embedding_dim=point_embedding_dim * (1 + max_num_points_in_route),
            number_embedding_dim=number_embedding_dim,
            device=device,
        )

    def forward(self, gamble_np_dict: dict[str, np.ndarray]) -> dict[str, list[torch.FloatTensor]]:
        return {
            'crr': self.courier_encoder(gamble_np_dict['crr']),
            'clm': self.claim_encoder(gamble_np_dict['clm']),
            'ord': self.order_encoder(gamble_np_dict['ord']),
        }
