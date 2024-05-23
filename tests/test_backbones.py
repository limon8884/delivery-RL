import torch
from torch import nn
import numpy as np
from datetime import datetime, timedelta
from torch.nn.utils.rnn import pad_sequence

from src.networks.encoders import GambleEncoder
# from src.networks.backbones import TransformerBackbone
from src.objects import Point, Courier, Route, Order, Claim, Gamble
from src.utils import get_random_point


BASE_DTTM = datetime.min


def make_courier(i):
    return Courier(i, get_random_point(), BASE_DTTM - timedelta(seconds=i), BASE_DTTM, 'auto')


def make_claim(i):
    return Claim(i, get_random_point(), get_random_point(), BASE_DTTM - timedelta(seconds=2 * i), BASE_DTTM,
                 timedelta(seconds=3), timedelta(seconds=5))


def make_route(i):
    return Route.from_points(
        points=[get_random_point() for _ in range(6)],
        claim_ids=list(range(i)),
        point_types=[Route.PointType.SOURCE] * 2 + [Route.PointType.DESTINATION] * 4,
    )


def make_gamble(n_crrs, n_clms, n_ords):
    return Gamble(
        couriers=[make_courier(i) for i in range(n_crrs)],
        orders=[
            Order(i, BASE_DTTM - timedelta(seconds=i), courier=make_courier(i), route=make_route(i),
                  claims=[make_claim(j) for j in range(4)])
            for i in range(n_ords)
        ],
        claims=[make_claim(i) for i in range(n_clms)],
        dttm_start=BASE_DTTM,
        dttm_end=BASE_DTTM + timedelta(seconds=30),
    )


# def test_transformer_backbone():
#     courier_order_embedding_dim = 68
#     claim_embedding_dim = 36
#     nhead = 4
#     model = TransformerBackbone(
#         claim_embedding_dim=claim_embedding_dim,
#         courier_order_embedding_dim=courier_order_embedding_dim,
#         nhead=nhead,
#         hidden_dim=17,
#         dim_feedforward=137,
#         num_encoder_layers=2,
#         num_decoder_layers=2,
#         device=None
#     )

#     co_embs_list = [
#         torch.randn(6, courier_order_embedding_dim),  # 2 c, 3 o
#         torch.randn(2, courier_order_embedding_dim),  # 1 c, 0 o
#     ]
#     co_embs = pad_sequence(co_embs_list, batch_first=True)

#     clm_embs_list = [
#         torch.randn(5, claim_embedding_dim),
#         torch.randn(3, claim_embedding_dim),
#     ]
#     clm_embs = pad_sequence(clm_embs_list, batch_first=True)

#     co_masks_list = [
#         torch.tensor([False, False, True, False, True, False]),
#         torch.tensor([False, False])
#     ]
#     co_masks = pad_sequence(co_masks_list, batch_first=True, padding_value=True)
#     co_masks = (co_masks.unsqueeze(-1) | co_masks.unsqueeze(-2))
#     co_masks = co_masks * -1e9
#     bs = co_masks.size(0)
#     seq_len = co_masks.size(1)
#     co_masks = co_masks.unsqueeze(1).repeat(1, nhead, 1, 1).view(bs * nhead, seq_len, seq_len)

#     clm_padding_mask = torch.tensor([
#         [False] * 5 + [True] * 0,
#         [False] * 3 + [True] * 2
#     ])
#     co_padding_mask = torch.tensor([
#         [False] * 6 + [True] * 0,
#         [False] * 2 + [True] * 4
#     ])

#     policy, value = model(co_embs, clm_embs, co_masks, clm_padding_mask, co_padding_mask)

#     assert policy.shape == (2, 5, 6), policy.shape
#     assert value.shape == (2,), value.shape
