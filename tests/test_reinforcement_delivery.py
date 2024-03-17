import torch
import numpy as np
import json
from datetime import datetime, timedelta

from src.objects import Assignment, Gamble, Point, Courier, Claim
from src.reinforcement.delivery import (
    DeliveryState,
    DeliveryAction,
    DeliveryEnvironment,
    DeliveryActorCritic,
)
from src.simulator.simulator import Simulator, DataReader
from src.router_makers import BaseRouteMaker
from src.networks.encoders import GambleEncoder


BASE_DTTM = datetime.min


TEST_DATA_CLAIMS = [
    {
        'created_dttm': BASE_DTTM + timedelta(seconds=10),
        'cancelled_dttm': BASE_DTTM + timedelta(seconds=40),
        'source_point_lat': 0.0,
        'source_point_lon': 0.2,
        'destination_point_lat': 0.0,
        'destination_point_lon': 1.0,
        'waiting_on_point_source': timedelta(seconds=0),
        'waiting_on_point_destination': timedelta(seconds=0),
    },
    {
        'created_dttm': BASE_DTTM + timedelta(seconds=0),
        'cancelled_dttm': BASE_DTTM + timedelta(seconds=40),
        'source_point_lat': 1.0,
        'source_point_lon': 0.2,
        'destination_point_lat': 1.0,
        'destination_point_lon': 1.0,
        'waiting_on_point_source': timedelta(seconds=0),
        'waiting_on_point_destination': timedelta(seconds=0),
    },
    {
        'created_dttm': BASE_DTTM + timedelta(seconds=70),
        'cancelled_dttm': BASE_DTTM + timedelta(seconds=100),
        'source_point_lat': 0.5,
        'source_point_lon': 0.0,
        'destination_point_lat': 0.5,
        'destination_point_lon': 1.0,
        'waiting_on_point_source': timedelta(seconds=0),
        'waiting_on_point_destination': timedelta(seconds=0),
    },
    {
        'created_dttm': BASE_DTTM + timedelta(seconds=130),
        'cancelled_dttm': BASE_DTTM + timedelta(seconds=200),
        'source_point_lat': 0.0,
        'source_point_lon': 0.8,
        'destination_point_lat': 0.0,
        'destination_point_lon': 0.0,
        'waiting_on_point_source': timedelta(seconds=0),
        'waiting_on_point_destination': timedelta(seconds=0),
    },
]

TEST_DATA_COURIERS = [
    {
        'start_dttm': BASE_DTTM + timedelta(seconds=0),
        'end_dttm': BASE_DTTM + timedelta(seconds=300),
        'start_position_lat': 1.0,
        'start_position_lon': 0.0
    },
    {
        'start_dttm': BASE_DTTM + timedelta(seconds=10),
        'end_dttm': BASE_DTTM + timedelta(seconds=300),
        'start_position_lat': 0.5,
        'start_position_lon': 1.0
    },
    {
        'start_dttm': BASE_DTTM + timedelta(seconds=0),
        'end_dttm': BASE_DTTM + timedelta(seconds=300),
        'start_position_lat': 0.0,
        'start_position_lon': 0.0
    },
]


def test_simulator(tmp_path):
    # fake_sim = TestSimulator(TEST_GAMBLES, TEST_ASSIGNMENTS)
    config_path = tmp_path / 'tmp.json'
    with open(config_path, 'w') as f:
        config = {
            'gamble_duration_interval_sec': 30,
            'courier_speed': 0.02
        }
        json.dump(config, f)
    reader = DataReader.from_list(TEST_DATA_COURIERS, TEST_DATA_CLAIMS, logger=None)
    route_maker = BaseRouteMaker(max_points_lenght=0, cutoff_radius=0.0)  # empty route_maker
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=config_path, logger=None)
    sim.next(Assignment([]))
    g0 = sim.get_state()
    assert len(g0.claims) == 2 and len(g0.couriers) == 3 and len(g0.orders) == 0
    assert g0.claims[0].id == 0 and g0.claims[1].id == 1, (g0.claims[0].id, g0.claims[1].id)
    assert g0.claims[0].source_point == Point(0.0, 0.2), g0.claims[0].source_point
    sim.next(Assignment([(0, 1), (2, 0)]))
    g1 = sim.get_state()
    assert len(g1.claims) == 0 and len(g1.couriers) == 1 and len(g1.orders) == 2
    sim.next(Assignment([]))
    g2 = sim.get_state()
    assert len(g2.claims) == 1 and len(g2.couriers) == 3 and len(g2.orders) == 0
    sim.next(Assignment([(1, 2)]))
    g3 = sim.get_state()
    assert len(g3.claims) == 0 and len(g3.couriers) == 2 and len(g3.orders) == 1
    sim.next(Assignment([]))
    g4 = sim.get_state()
    assert len(g4.claims) == 1 and len(g4.couriers) == 2 and len(g4.orders) == 1
    sim.next(Assignment([(2, 3)]))
    g5 = sim.get_state()
    assert len(g5.claims) == 0 and len(g5.couriers) == 1 and len(g5.orders) == 2


def test_delivery_environment(tmp_path):
    config_path = tmp_path / 'tmp.json'
    with open(config_path, 'w') as f:
        config = {
            'gamble_duration_interval_sec': 30,
            'courier_speed': 0.02
        }
        json.dump(config, f)
    reader = DataReader.from_list(TEST_DATA_COURIERS, TEST_DATA_CLAIMS, logger=None)
    route_maker = BaseRouteMaker(max_points_lenght=0, cutoff_radius=0.0)  # empty route_maker
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=config_path, logger=None)
    env = DeliveryEnvironment(simulator=sim, max_num_points_in_route=4, num_gambles=6, device=None)
    state1 = env.reset()
    assert env._iter == 1, env._iter
    assert np.isclose(state1.claim_emb, [0.0, 0.2, 0.0, 1.0, 0.0, 20.0]).all(), (state1.claim_emb)
    assert state1.couriers_embs.shape == (3, 4)
    assert np.isclose(state1.couriers_embs, [[1.0, 0.0, 0.0, 30.0], [0.5, 1.0, 0.0, 20.0], [0.0, 0.0, 0.0, 30.0]]).all()
    assert state1.orders_embs is None, state1.orders_embs

    state2, reward, done, info = env.step(DeliveryAction(2))
    assert env._iter == 1, env._iter
    assert not done
    assert np.isclose(state2.claim_emb, [1.0, 0.2, 1.0, 1.0, 0.0, 30.0]).all(), (state2.claim_emb)
    assert np.isclose(state2.couriers_embs, [[1.0, 0.0, 0.0, 30.0], [0.5, 1.0, 0.0, 20.0], [0.0, 0.0, 0.0, 30.0]]).all()
    assert state2.orders_embs is None, state2.orders_embs

    state3, reward, done, info = env.step(DeliveryAction(0))
    assert env._iter == 3, env._iter
    assert not done
    assert np.isclose(state3.claim_emb, [0.5, 0.0, 0.5, 1.0, 0.0, 20.0]).all(), (state3.claim_emb)
    assert state3.orders_embs is None, state3.orders_embs
    assert np.isclose(state3.couriers_embs, [[0.5, 1.0, 0.0, 80.0], [0.0, 1.0, 0.0, 90.0], [1.0, 1.0, 0.0, 90.0]]).all()

    state4, reward, done, info = env.step(DeliveryAction(0))
    assert env._iter == 5, env._iter
    assert not done
    assert np.isclose(state4.claim_emb, [0.0, 0.8, 0.0, 0.0, 0.0, 20.0]).all(), (state4.claim_emb)
    assert np.isclose(state4.couriers_embs, [[0.0, 1.0, 0.0, 150.0], [1.0, 1.0, 0.0, 150.0]]).all()
    assert len(state4.orders_embs) == 1
    assert np.isclose(state4.orders_embs[0], [0.5, 0.2, 1.0, 140.0, 1.0, 60.0, 0.5, 1.0] + [0.0] * 6).all()

    state5, reward, done, info = env.step(DeliveryAction(2))
    assert done, env._iter


def test_delivery_actor_critic_shape(tmp_path):
    config_path = tmp_path / 'tmp.json'
    with open(config_path, 'w') as f:
        config = {
            'gamble_duration_interval_sec': 30,
            'courier_speed': 0.02
        }
        json.dump(config, f)
    reader = DataReader.from_list(TEST_DATA_COURIERS, TEST_DATA_CLAIMS, logger=None)
    route_maker = BaseRouteMaker(max_points_lenght=0, cutoff_radius=0.0)  # empty route_maker
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=config_path, logger=None)
    env = DeliveryEnvironment(simulator=sim, max_num_points_in_route=4, num_gambles=6, device=None)
    gamble_encoder = GambleEncoder(
        order_embedding_dim=32,
        claim_embedding_dim=16,
        courier_embedding_dim=32,
        point_embedding_dim=16,
        number_embedding_dim=4,
        courier_order_embedding_dim=32,
        max_num_points_in_route=4,
        device=None,
    )
    ac = DeliveryActorCritic(gamble_encoder, clm_emb_size=16, temperature=1.0, device=None)
    state1 = env.reset()  # (3, 0)
    policy_half_tens, value_half_tens, claim_emb = ac._make_three_tensors_from_state(state1)
    assert policy_half_tens.shape == (4, 16), policy_half_tens.shape
    assert value_half_tens.shape == (4, 16), value_half_tens.shape
    assert claim_emb.shape == (16,)

    state2, reward, done, info = env.step(DeliveryAction(2))  # (3, 0)
    state3, reward, done, info = env.step(DeliveryAction(0))  # (3, 0)
    state4, reward, done, info = env.step(DeliveryAction(0))  # (2, 1)
    policy_half_tens, value_half_tens, claim_emb = ac._make_three_tensors_from_state(state4)
    assert policy_half_tens.shape == (4, 16), policy_half_tens.shape
    assert value_half_tens.shape == (4, 16), value_half_tens.shape
    assert claim_emb.shape == (16,)


class FakeGambleEncoder(GambleEncoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.claim_embedding_dim = kwargs['claim_embedding_dim']

    def forward(self, embs_dict: dict[str, np.ndarray | None]) -> dict[str, torch.FloatTensor | None]:
        assert embs_dict['clm'].shape[0] == 1
        n_crr = len(embs_dict['crr']) if embs_dict['crr'] is not None else None
        n_ord = len(embs_dict['ord']) if embs_dict['ord'] is not None else None
        return {
            'clm': torch.arange(self.claim_embedding_dim, dtype=torch.float).reshape(1, -1),
            'crr': torch.arange(2 * self.claim_embedding_dim,
                                dtype=torch.float).repeat(n_crr, 1) if n_crr is not None else None,
            'ord': torch.arange(2 * self.claim_embedding_dim,
                                dtype=torch.float).repeat(n_ord, 1) if n_ord is not None else None,
        }


def test_delivery_actor_critic():
    gamble_encoder = FakeGambleEncoder(
        order_embedding_dim=32,
        claim_embedding_dim=16,
        courier_embedding_dim=32,
        point_embedding_dim=16,
        number_embedding_dim=4,
        courier_order_embedding_dim=32,
        max_num_points_in_route=4,
        device=None,
    )
    ac = DeliveryActorCritic(gamble_encoder, clm_emb_size=16, temperature=1.0, device=None)
    state1 = DeliveryState(
        claim_emb=np.zeros((1, 10)),
        couriers_embs=np.zeros((5, 8)),
        orders_embs=None,
    )
    state2 = DeliveryState(
        claim_emb=np.zeros((1, 10)),
        couriers_embs=np.zeros((2, 8)),
        orders_embs=np.zeros((1, 6)),
    )
    pol_tens, val_tens = ac._make_padded_policy_value_tensors([state1, state2])
    assert pol_tens.shape == (2, 6)
    crr_ord_val = (torch.arange(16, dtype=torch.float)**2).sum()
    fake_crr_val = torch.arange(16, dtype=torch.float).sum()
    pad_val = -1e11
    assert torch.isclose(pol_tens[0, :], torch.tensor([crr_ord_val] * 5 + [fake_crr_val])).all()
    assert torch.isclose(pol_tens[1, :], torch.tensor([crr_ord_val] * 3 + [fake_crr_val] + [pad_val] * 2)).all()

    crr_ord_val = (torch.arange(16, dtype=torch.float) * torch.arange(16, 32)).sum()
    assert val_tens.shape == (2,)
    assert torch.isclose(val_tens[0], torch.tensor(crr_ord_val * 5 / 6 + fake_crr_val / 6)).all()
    assert torch.isclose(val_tens[1], torch.tensor(crr_ord_val * 3 / 4 + fake_crr_val / 4)).all()
