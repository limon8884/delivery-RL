import torch
import torch.nn as nn
from datetime import datetime, timedelta

from src_new.objects import Point, Courier, Order, Route, Claim, Gamble


class NumberEncoder(nn.Module):
    def __init__(self, number_embedding_dim, device) -> None:
        super().__init__()
        self.number_embedding_dim = number_embedding_dim
        self.layer = nn.Linear(1, number_embedding_dim, device=device)
        self.activation_func = nn.LeakyReLU()
        self.device = device

    def forward(self, x: float) -> torch.FloatTensor:
        x = torch.tensor([x], dtype=torch.float32, device=self.device)
        x = self.layer(x)
        x = self.activation_func(x)
        return x


class PointEncoder(nn.Module):
    def __init__(self, point_embedding_dim, device):
        super().__init__()
        assert point_embedding_dim % 4 == 0
        self.point_embedding_dim = point_embedding_dim
        self.sin_layer_x = nn.Linear(1, point_embedding_dim // 4, device=device)
        self.cos_layer_x = nn.Linear(1, point_embedding_dim // 4, device=device)
        self.sin_layer_y = nn.Linear(1, point_embedding_dim // 4, device=device)
        self.cos_layer_y = nn.Linear(1, point_embedding_dim // 4, device=device)
        self.device = device

        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, p: Point) -> torch.FloatTensor:
        # with torch.no_grad():
        x = torch.tensor([p.x], dtype=torch.float32, device=self.device)
        y = torch.tensor([p.y], dtype=torch.float32, device=self.device)
        return torch.cat([
            torch.sin(self.sin_layer_x(x)),
            torch.cos(self.cos_layer_x(x)),
            torch.sin(self.sin_layer_y(y)),
            torch.cos(self.cos_layer_y(y)),
        ], dim=-1)


class CourierEncoder(nn.Module):
    def __init__(self, courier_embedding_dim, point_embedding_dim, number_embedding_dim, device) -> None:
        super().__init__()
        self.courier_embedding_dim = courier_embedding_dim
        self.coord_encoder = PointEncoder(point_embedding_dim=point_embedding_dim, device=device)
        self.time_encoder = NumberEncoder(number_embedding_dim=number_embedding_dim, device=device)
        self.mlp = nn.Sequential(
            nn.Linear(point_embedding_dim + number_embedding_dim, courier_embedding_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(courier_embedding_dim, courier_embedding_dim, device=device),
            nn.LeakyReLU(),
        )

    def forward(self, crr: Courier, dttm: datetime) -> torch.FloatTensor:
        coords_emb = self.coord_encoder(crr.position)
        time_emb = self.time_encoder((dttm - crr.start_dttm).total_seconds())
        x = torch.cat([coords_emb, time_emb], dim=-1)
        x = self.mlp(x)
        return x


class RouteEncoder(nn.Module):
    def __init__(self, route_embedding_dim, point_embedding_dim,
                 number_embedding_dim, max_num_points_in_route, device) -> None:
        super().__init__()
        self.route_embedding_dim = route_embedding_dim
        self.max_num_points_in_route = max_num_points_in_route
        self.coord_encoder = PointEncoder(point_embedding_dim=point_embedding_dim, device=device)
        self.point_type_encoder = NumberEncoder(number_embedding_dim=number_embedding_dim, device=device)

        raw_emb_dim = (point_embedding_dim + number_embedding_dim) * max_num_points_in_route 
        self.mlp = nn.Sequential(
            nn.Linear(raw_emb_dim, route_embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(route_embedding_dim, route_embedding_dim),
            nn.LeakyReLU(),
        )

    def forward(self, route: Route) -> torch.FloatTensor:
        assert len(route.route_points) <= self.max_num_points_in_route
        embeddings_list = []
        for point in route.route_points:
            embeddings_list.append(self.coord_encoder(point.point))
            embeddings_list.append(self.point_type_encoder(point.point_type.value))
        zeroes_left = (self.coord_encoder.point_embedding_dim + self.point_type_encoder.number_embedding_dim) \
            * (self.max_num_points_in_route - len(route.route_points))
        embeddings_list.append(torch.zeros(size=(zeroes_left,)))
        x = torch.cat(embeddings_list, dim=-1)
        x = self.mlp(x)
        return x


class CourierOrderEncoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 courier_embedding_dim,
                 route_embedding_dim,
                 point_embedding_dim,
                 number_embedding_dim,
                 max_num_points_in_route,
                 device,
                 ) -> None:
        super().__init__()
        self.courier_encoder = CourierEncoder(courier_embedding_dim, point_embedding_dim, number_embedding_dim, device)
        self.route_encoder = RouteEncoder(route_embedding_dim, point_embedding_dim, number_embedding_dim,
                                          max_num_points_in_route, device)
        self.time_encoder = NumberEncoder(number_embedding_dim, device)
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(courier_embedding_dim + route_embedding_dim, embedding_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim, device=device),
            nn.LeakyReLU(),
        )

    def forward(self, item: Order | Courier, dttm: datetime) -> torch.FloatTensor:
        if isinstance(item, Courier):
            courier = item
            route = Route([])
        elif isinstance(item, Order):
            courier = item.courier
            route = item.route
        else:
            raise RuntimeError('item should be either courier or order')
        crr_emb = self.courier_encoder(courier, dttm)
        rt_emb = self.route_encoder(route)
        both_emb = torch.cat([crr_emb, rt_emb], dim=-1)
        emb = self.mlp(both_emb)
        return emb


class ClaimEncoder(nn.Module):
    def __init__(self, claim_embedding_dim, point_embedding_dim, number_embedding_dim, device) -> None:
        super().__init__()
        self.claim_embedding_dim = claim_embedding_dim
        self.source_coord_encoder = PointEncoder(point_embedding_dim=point_embedding_dim, device=device)
        self.destination_coord_encoder = PointEncoder(point_embedding_dim=point_embedding_dim, device=device)
        self.time_encoder = NumberEncoder(number_embedding_dim=number_embedding_dim, device=device)
        self.source_waiting_encoder = NumberEncoder(number_embedding_dim=number_embedding_dim, device=device)
        self.destination_waiting_encoder = NumberEncoder(number_embedding_dim=number_embedding_dim, device=device)
        self.mlp = nn.Sequential(
            nn.Linear(point_embedding_dim * 2 + number_embedding_dim * 3, claim_embedding_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(claim_embedding_dim, claim_embedding_dim, device=device),
            nn.LeakyReLU(),
        )

    def forward(self, claim: Claim, dttm: datetime) -> torch.FloatTensor:
        embeddings_list = []
        embeddings_list.append(self.source_coord_encoder(claim.source_point))
        embeddings_list.append(self.destination_coord_encoder(claim.destination_point))
        embeddings_list.append(self.time_encoder((dttm - claim.creation_dttm).total_seconds()))
        embeddings_list.append(self.source_waiting_encoder(claim.waiting_on_point_source.total_seconds()))
        embeddings_list.append(self.destination_waiting_encoder(claim.waiting_on_point_destination.total_seconds()))
        x = torch.cat(embeddings_list, dim=-1)
        emb = self.mlp(x)
        return emb


class GambleEncoder(nn.Module):
    def __init__(self,
                 courier_order_embedding_dim,
                 claim_embedding_dim,
                 courier_embedding_dim,
                 route_embedding_dim,
                 point_embedding_dim,
                 number_embedding_dim,
                 max_num_points_in_route,
                 device,
                 ) -> None:
        super().__init__()
        self.claim_encoder = ClaimEncoder(claim_embedding_dim, point_embedding_dim, number_embedding_dim, device)
        self.courier_order_encoder = CourierOrderEncoder(courier_order_embedding_dim,
                                                         courier_embedding_dim, route_embedding_dim,
                                                         point_embedding_dim,
                                                         number_embedding_dim, max_num_points_in_route, device)

    def forward(self, gamble: Gamble) -> dict[str, list[torch.FloatTensor]]:
        return {
            'couriers': [self.courier_order_encoder(courier, gamble.dttm_start) for courier in gamble.couriers],
            'orders': [self.courier_order_encoder(order, gamble.dttm_start) for order in gamble.orders],
            'claims': [self.claim_encoder(claim, gamble.dttm_start) for claim in gamble.claims],
        }
