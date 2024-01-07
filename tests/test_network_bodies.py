
from datetime import datetime, timedelta

from src_new.objects import Point, Courier, Route, Order, Claim, Gamble
from src_new.utils import get_random_point
from src_new.networks.encoders import GambleEncoder
from src_new.networks.bodies import SimpleSequentialMLP


BASE_DTTM = datetime.utcnow()


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


def test_simple_sequential_mlp():
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
    net = SimpleSequentialMLP(32, 64, device=None)
    emb_dict = enc(gamble)
    emb_crr_ord = emb_dict['couriers'] + emb_dict['orders']
    for claim_emb in emb_dict['claims']:
        probs = net(claim_emb, emb_crr_ord)
        assert probs.shape == (len(emb_crr_ord) + 1,)
