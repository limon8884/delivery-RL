import pytest
from datetime import datetime, timedelta
import random
import numpy as np

from src.dispatchs.scorers import DistanceScorer
from src.dispatchs.hungarian_dispatch import HungarianDispatch
from src.dispatchs.greedy_dispatch import GreedyDispatch
from src.dispatchs.random_dispatch import RandomDispatch
from src.dispatchs.neural_sequantial_dispatch import NeuralSequantialDispatch
from src.networks.encoders import GambleEncoder
# from src.networks.networks import SimpleSequentialMLP
from src.objects import (
    Gamble,
    Courier,
    Claim,
    Point,
    Order,
    Route,
)
from src.utils import get_random_point


BASE_DTTM = datetime.min


def make_gamble_from_coords(couriers_coords: list[Point], claims_coords: list[Point]) -> Gamble:
    return Gamble(
        couriers=[
            Courier(
                id=i,
                position=p,
                start_dttm=BASE_DTTM,
                end_dttm=BASE_DTTM,
                courier_type='auto'
            )
            for i, p in enumerate(couriers_coords)
        ],
        claims=[
            Claim(
                id=i,
                source_point=p,
                destination_point=Point(0, 0),
                creation_dttm=BASE_DTTM,
                cancell_if_not_assigned_dttm=BASE_DTTM,
                waiting_on_point_source=timedelta(seconds=0),
                waiting_on_point_destination=timedelta(seconds=0)
            )
            for i, p in enumerate(claims_coords)
        ],
        orders=[],
        dttm_start=BASE_DTTM,
        dttm_end=BASE_DTTM
    )


def test_distance_scorer_shapes():
    for _ in range(10):
        n_couriers = random.randint(0, 3)
        n_claims = random.randint(0, 3)
        couriers_coords = [get_random_point() for _ in range(n_couriers)]
        claims_coords = [get_random_point() for _ in range(n_claims)]
        gamble = make_gamble_from_coords(couriers_coords, claims_coords)
        scores = DistanceScorer().score(gamble)
        assert scores.shape == (n_couriers, n_claims + 1)


TEST_GAMBLES = [
    (0, make_gamble_from_coords(
        [
            Point(0.0, 0.0),
            Point(1.0, 1.0),
        ],
        [
            Point(0.7, 0.7),
            Point(0.6, 0.6),
        ]
    )),
    (1, make_gamble_from_coords(
        [
            Point(0.0, 1.0),
            Point(1.0, 1.0),
        ],
        [
            Point(0.0, 0.0),
            Point(0.5, 0.0),
            Point(1.0, 0.0),
        ]
    )),
    (2, make_gamble_from_coords(
        [],
        [
            Point(0.0, 0.0),
            Point(1.0, 0.0),
        ]
    )),
    (3, make_gamble_from_coords(
        [
            Point(0.0, 1.0),
            Point(1.0, 1.0),
        ],
        []
    )),
]


EXPECTED_DISTANCE_SCORER = [
    np.exp(-2**0.5 * np.array([[0.7, 0.6, np.inf], [0.3, 0.4, np.inf]])),
    np.exp(-np.array([[1.0, 1.25**0.5, 2**0.5, np.inf], [2**0.5, 1.25**0.5, 1.0, np.inf]])),
    np.array([]),
    np.zeros((2, 1)),
]


@pytest.mark.parametrize(['idx', 'gamble'], TEST_GAMBLES)
def test_distance_scorer(idx: int, gamble: Gamble):
    # gamble = make_gamble_from_coords(couriers_coords, claims_coords)
    scores = DistanceScorer().score(gamble)
    if len(gamble.couriers) == 0:
        assert len(scores) == 0
        return
    assert np.isclose(scores, EXPECTED_DISTANCE_SCORER[idx]).all()


EXPECTED_HUNGARIAN_DISPATCH = [
    [(0, 1), (1, 0)],
    [(0, 0), (1, 2)],
    [],
    [],
]


@pytest.mark.parametrize(['idx', 'gamble'], TEST_GAMBLES)
def test_hungarian_dispatch(idx: int, gamble: Gamble):
    dsp = HungarianDispatch(DistanceScorer())
    assert sorted(dsp(gamble).ids) == sorted(EXPECTED_HUNGARIAN_DISPATCH[idx])


EXPECTED_GREEDY_DISPATCH = [
    [(0, 1), (1, 0)],
    [(0, 0), (1, 2)],
    [],
    [],
]


@pytest.mark.parametrize(['idx', 'gamble'], TEST_GAMBLES)
def test_greedy_dispatch(idx: int, gamble: Gamble):
    dsp = GreedyDispatch(DistanceScorer())
    assert sorted(dsp(gamble).ids) == sorted(EXPECTED_GREEDY_DISPATCH[idx])


@pytest.mark.parametrize(['idx', 'gamble'], TEST_GAMBLES)
def test_random_dispatch(idx: int, gamble: Gamble):
    dsp = RandomDispatch()
    assert len(dsp(gamble).ids) == min(len(gamble.claims), len(gamble.couriers))


# @pytest.mark.parametrize(['idx', 'gamble'], TEST_GAMBLES)
# def test_network_dispatch(idx: int, gamble: Gamble):
#     ord_clm = Claim(10, Point(2.0, 2.0), Point(3.0, 3.0), 
#                     BASE_DTTM, BASE_DTTM + timedelta(days=1), timedelta(seconds=0), timedelta(seconds=0))
#     gamble.orders = [
#         Order(
#             id=0,
#             creation_dttm=BASE_DTTM,
#             courier=Courier(10, Point(2.0, 2.0), BASE_DTTM, BASE_DTTM + timedelta(days=1), 'auto'),
#             route=Route.from_claim(ord_clm),
#             claims=[ord_clm],
#         )
#     ]
#     encoder = GambleEncoder(
#         courier_order_embedding_dim=64,
#         claim_embedding_dim=32,
#         courier_embedding_dim=32,
#         route_embedding_dim=128,
#         point_embedding_dim=32,
#         number_embedding_dim=8,
#         max_num_points_in_route=8,
#         device=None,
#     )
#     net = SimpleSequentialMLP(
#         claim_embedding_dim=32,
#         courier_order_embedding_dim=64,
#         device=None,
#     )
#     dsp = NeuralSequantialDispatch(encoder=encoder, network=net, max_num_points_in_route=4)
#     dsp(gamble).ids
