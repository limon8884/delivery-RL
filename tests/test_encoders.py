# import torch
# import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta

from src.networks.encoders import (
    CoordEncoder,
    NumberEncoder,
    ItemEncoder,
    GambleEncoder,
)
from src.objects import Point, Courier, Route, Order, Claim, Gamble
from src.utils import get_random_point


BASE_DTTM = datetime.utcnow()


def test_point_encoder():
    enc = CoordEncoder(coords_np_dim=10, coords_embedding_dim=64, device=None)
    coords = np.random.randn(3, 10)
    assert enc(coords).shape == (3, 64)


def test_number_encoder():
    enc = NumberEncoder(numbers_np_dim=2, number_embedding_dim=8, device=None)
    nums = np.random.randn(3, 2)
    assert enc(nums).shape == (3, 8)


def test_courier_encoder():
    enc = ItemEncoder(feature_types=Courier.numpy_feature_types(),
                      item_embedding_dim=32, point_embedding_dim=16, number_embedding_dim=8, device=None)
    crr = Courier(
        id=0,
        position=Point(0.5, -1.5),
        start_dttm=BASE_DTTM,
        end_dttm=BASE_DTTM + timedelta(days=1),
        courier_type='auto',
    )
    emb = enc(crr.to_numpy().reshape(1, -1))
    assert emb.shape == (1, 32)


TEST_ROUTE = Route.from_points(
    [
        Point(0.0, 0.0),
        Point(1.0, 0.0),
        Point(0.0, 0.0),
        Point(-10.0, -10.0),
        Point(7.1, 5.0),
        Point(1.0, 1.0)
    ],
    [1, 2, 1, 3, 4, 3],
    [
        Route.PointType.SOURCE,
        Route.PointType.DESTINATION,
        Route.PointType.DESTINATION,
        Route.PointType.SOURCE,
        Route.PointType.DESTINATION,
        Route.PointType.DESTINATION,
    ]
)


def test_order_encoder():
    enc = ItemEncoder(
        feature_types=Order.numpy_feature_types(max_num_points_in_route=8),
        item_embedding_dim=64,
        point_embedding_dim=32,
        number_embedding_dim=4,
        device=None,
    )
    
    courier = Courier(
        id=0,
        position=Point(0.0, 1.0),
        start_dttm=BASE_DTTM,
        end_dttm=BASE_DTTM + timedelta(seconds=10),
        courier_type='auto',
    )
    order = Order(
        id=0,
        creation_dttm=BASE_DTTM - timedelta(seconds=10),
        courier=courier,
        route=TEST_ROUTE,
        claims=[],
    )
    ord_emb = enc(order.to_numpy(max_num_points_in_route=8).reshape(1, -1))
    assert ord_emb.shape == (1, 64)


def test_claim_encoder():
    enc = ItemEncoder(
        Claim.numpy_feature_types(),
        item_embedding_dim=64,
        point_embedding_dim=32,
        number_embedding_dim=4,
        device=None,
    )
    claim = Claim(
        id=0,
        source_point=Point(0.0, 1.0),
        destination_point=Point(-0.5, 1.5),
        creation_dttm=BASE_DTTM - timedelta(seconds=10),
        cancell_if_not_assigned_dttm=BASE_DTTM + timedelta(seconds=10),
        waiting_on_point_source=timedelta(seconds=3),
        waiting_on_point_destination=timedelta(seconds=0),
    )

    emb = enc(claim.to_numpy().reshape(1, -1))
    assert emb.shape == (1, 64)


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


def test_gamble_encoder():
    enc = GambleEncoder(
        order_embedding_dim=64,
        claim_embedding_dim=32,
        courier_embedding_dim=32,
        point_embedding_dim=8,
        number_embedding_dim=4,
        max_num_points_in_route=10,
        device=None,
        )
    gamble = Gamble(
        couriers=[make_courier(i) for i in range(5)],
        orders=[
            Order(i, BASE_DTTM - timedelta(seconds=i), courier=make_courier(i), route=make_route(i),
                  claims=[make_claim(j) for j in range(4)])
            for i in range(3)
        ],
        claims=[make_claim(i) for i in range(4)],
        dttm_start=BASE_DTTM,
        dttm_end=BASE_DTTM + timedelta(seconds=30),
    )

    d = {
        'crr': np.stack([crr.to_numpy() for crr in gamble.couriers], axis=0),
        'clm': np.stack([clm.to_numpy() for clm in gamble.claims], axis=0),
        'ord': np.stack([ord.to_numpy(max_num_points_in_route=10) for ord in gamble.orders], axis=0),
    }
    emb_dict = enc(d)
    assert isinstance(emb_dict, dict)
    assert emb_dict['crr'].shape == (5, 32)
    assert emb_dict['clm'].shape == (4, 32)
    assert emb_dict['ord'].shape == (3, 64)
