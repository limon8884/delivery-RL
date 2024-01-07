import torch
import torch.nn as nn
import random
from datetime import datetime, timedelta

from src_new.networks.encoders import (
    PointEncoder,
    NumberEncoder,
    CourierEncoder,
    RouteEncoder,
    CourierOrderEncoder,
    ClaimEncoder,
    GambleEncoder,
)
from src_new.objects import Point, Courier, Route, Order, Claim, Gamble
from src_new.utils import get_random_point


BASE_DTTM = datetime.utcnow()


def test_point_encoder():
    enc = PointEncoder(point_embedding_dim=64, device=None)
    for _ in range(10):
        p = get_random_point()
        v = enc(p)
        assert v.shape == (64,)


def test_number_encoder():
    enc = NumberEncoder(number_embedding_dim=8, device=None)
    for _ in range(10):
        x = random.random()
        v = enc(x)
        assert v.shape == (8,)
        x = random.randint(-10, 10)
        v = enc(x)
        assert v.shape == (8,)


def test_courier_encoder():
    enc = CourierEncoder(courier_embedding_dim=128, point_embedding_dim=64, number_embedding_dim=8, device=None)
    crr = Courier(
        id=0,
        position=Point(0.5, -1.5),
        start_dttm=BASE_DTTM,
        end_dttm=BASE_DTTM + timedelta(days=1),
        courier_type='auto',
    )
    emb = enc(crr, BASE_DTTM + timedelta(seconds=10))
    assert emb.shape == (128,)


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


def test_route_encoder():
    enc = RouteEncoder(route_embedding_dim=64, point_embedding_dim=32,
                       number_embedding_dim=4, max_num_points_in_route=8, device=None)
    emb = enc(TEST_ROUTE)
    assert emb.shape == (64, )


def test_courierOrder_encoder():
    enc = CourierOrderEncoder(
        embedding_dim=64,
        courier_embedding_dim=32,
        route_embedding_dim=64,
        point_embedding_dim=32,
        number_embedding_dim=4,
        max_num_points_in_route=8,
        device=None,
    )
    courier = Courier(
        id=0,
        position=Point(0.0, 1.0),
        start_dttm=BASE_DTTM,
        end_dttm=BASE_DTTM + timedelta(seconds=10),
        courier_type='auto',
    )
    crr_emb = enc(courier, BASE_DTTM)
    assert crr_emb.shape == (64,)

    order = Order(
        id=0,
        creation_dttm=BASE_DTTM - timedelta(seconds=10),
        courier=courier,
        route=TEST_ROUTE,
        claims=[],
    )
    ord_emb = enc(order, BASE_DTTM)
    assert ord_emb.shape == (64,)


def test_claim_encoder():
    enc = ClaimEncoder(64, 32, 4, device=None)
    claim = Claim(
        id=0,
        source_point=Point(0.0, 1.0),
        destination_point=Point(-0.5, 1.5),
        creation_dttm=BASE_DTTM - timedelta(seconds=10),
        cancell_if_not_assigned_dttm=BASE_DTTM + timedelta(seconds=10),
        waiting_on_point_source=timedelta(seconds=3),
        waiting_on_point_destination=timedelta(seconds=0),
    )

    emb = enc(claim, BASE_DTTM)
    assert emb.shape == (64,)


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
    enc = GambleEncoder(64, 32, 32, 64, 16, 4, 8, device=None)
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
    emb_dict = enc(gamble)
    assert isinstance(emb_dict, dict)
    assert isinstance(emb_dict['couriers'], list) and len(emb_dict['couriers']) == 5
    assert isinstance(emb_dict['orders'], list) and len(emb_dict['orders']) == 3
    assert isinstance(emb_dict['claims'], list) and len(emb_dict['claims']) == 4
    assert emb_dict['couriers'][0].shape == (64,)
    assert emb_dict['orders'][0].shape == (64,)
    assert emb_dict['claims'][0].shape == (32,)
