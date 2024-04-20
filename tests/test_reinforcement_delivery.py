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
    DeliveryRewarder,
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
    reader = DataReader.from_list(TEST_DATA_COURIERS, TEST_DATA_CLAIMS, db_logger=None)
    route_maker = BaseRouteMaker(max_points_lenght=0, cutoff_radius=0.0)  # empty route_maker
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=config_path, db_logger=None)
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
    reader = DataReader.from_list(TEST_DATA_COURIERS, TEST_DATA_CLAIMS, db_logger=None)
    route_maker = BaseRouteMaker(max_points_lenght=0, cutoff_radius=0.0)  # empty route_maker
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=config_path, db_logger=None)
    rewarder = DeliveryRewarder(coef_reward_assigned=0.1, coef_reward_cancelled=1.0, coef_reward_distance=0.0,
                                coef_reward_completed=0.0)
    env = DeliveryEnvironment(simulator=sim, rewarder=rewarder, max_num_points_in_route=4, use_dist=False,
                              num_gambles_in_day=6, device=None)
    state1 = env.reset()
    assert env._iter == 1, env._iter
    assert np.isclose(state1.claim_emb, [0.0, 0.2, 0.0, 1.0, 0.0, 20.0]).all(), (state1.claim_emb)
    assert state1.couriers_embs.shape == (3, 4)
    assert np.isclose(state1.couriers_embs, [[1.0, 0.0, 0.0, 30.0], [0.5, 1.0, 0.0, 20.0], [0.0, 0.0, 0.0, 30.0]]).all()
    assert state1.orders_embs is None, state1.orders_embs
    assert state1.prev_idxs == []

    state2, reward, done, info = env.step(DeliveryAction(2))
    assert env._iter == 1, env._iter
    assert not done
    assert np.isclose(state2.claim_emb, [1.0, 0.2, 1.0, 1.0, 0.0, 30.0]).all(), (state2.claim_emb)
    assert np.isclose(state2.couriers_embs, [[1.0, 0.0, 0.0, 30.0], [0.5, 1.0, 0.0, 20.0], [0.0, 0.0, 0.0, 30.0]]).all()
    assert state2.orders_embs is None, state2.orders_embs
    assert state2.prev_idxs == [2]

    state3, reward, done, info = env.step(DeliveryAction(0))
    assert env._iter == 3, env._iter
    assert not done
    assert np.isclose(state3.claim_emb, [0.5, 0.0, 0.5, 1.0, 0.0, 20.0]).all(), (state3.claim_emb)
    assert state3.orders_embs is None, state3.orders_embs
    assert np.isclose(state3.couriers_embs, [[0.5, 1.0, 0.0, 80.0], [0.0, 1.0, 0.0, 90.0], [1.0, 1.0, 0.0, 90.0]]).all()
    assert state3.prev_idxs == []

    state4, reward, done, info = env.step(DeliveryAction(0))
    assert env._iter == 5, env._iter
    assert not done
    assert np.isclose(state4.claim_emb, [0.0, 0.8, 0.0, 0.0, 0.0, 20.0]).all(), (state4.claim_emb)
    assert np.isclose(state4.couriers_embs, [[0.0, 1.0, 0.0, 150.0], [1.0, 1.0, 0.0, 150.0]]).all()
    assert len(state4.orders_embs) == 1
    assert np.isclose(state4.orders_embs[0], [0.5, 0.2, 1.0, 140.0, 0.5, 1.0] + [0.0] * 6 + [1.0, 60.0]).all()
    assert state4.prev_idxs == []

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
    reader = DataReader.from_list(TEST_DATA_COURIERS, TEST_DATA_CLAIMS, db_logger=None)
    route_maker = BaseRouteMaker(max_points_lenght=0, cutoff_radius=0.0)  # empty route_maker
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=config_path, db_logger=None)
    rewarder = DeliveryRewarder(coef_reward_assigned=0.1, coef_reward_cancelled=1.0, coef_reward_distance=0.0,
                                coef_reward_completed=0.0)
    env = DeliveryEnvironment(simulator=sim, rewarder=rewarder, max_num_points_in_route=4, use_dist=False,
                              num_gambles_in_day=6, device=None)
    gamble_encoder = GambleEncoder(
        route_embedding_dim=128,
        claim_embedding_dim=16,
        point_embedding_dim=16,
        cat_points_embedding_dim=8,
        courier_order_embedding_dim=32,
        max_num_points_in_route=4,
        num_layers=2,
        gamble_features_embedding_dim=8,
        use_dist=False,
        device=None,
        use_pretrained_encoders=False,
    )
    ac = DeliveryActorCritic(gamble_encoder, clm_emb_size=16, co_emb_size=32, gmb_emb_size=8, temperature=1.0,
                             device=None)
    state1 = env.reset()  # (3, 0)
    co_embs, claim_emb, gamble_emb = ac._make_embeddings_tensors_from_state(state1)
    add_emb = ac._make_additional_features_from_state(state1)
    assert co_embs.shape == (4, 32), co_embs.shape
    assert claim_emb.shape == (1, 16)
    assert gamble_emb.shape == (1, 8)
    assert add_emb.shape == (4, 2)
    assert add_emb.isclose(torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=torch.float)).all()

    state2, reward, done, info = env.step(DeliveryAction(2))  # (3, 0)
    add_emb = ac._make_additional_features_from_state(state2)
    assert add_emb.shape == (4, 2)
    assert add_emb.isclose(torch.tensor([[0, 1], [0, 1], [1, 1], [0, 1]], dtype=torch.float)).all()
    state3, reward, done, info = env.step(DeliveryAction(0))  # (3, 0)
    state4, reward, done, info = env.step(DeliveryAction(0))  # (2, 1)
    co_embs, claim_emb, gamble_emb = ac._make_embeddings_tensors_from_state(state4)
    assert co_embs.shape == (4, 32), co_embs.shape
    assert claim_emb.shape == (1, 16)
    assert gamble_emb.shape == (1, 8)


def test_delivery_actor_critic():
    gamble_encoder = GambleEncoder(
        route_embedding_dim=128,
        claim_embedding_dim=16,
        point_embedding_dim=16,
        cat_points_embedding_dim=8,
        courier_order_embedding_dim=32,
        max_num_points_in_route=4,
        use_dist=False,
        num_layers=2,
        gamble_features_embedding_dim=8,
        device=None,
        use_pretrained_encoders=False,
    )
    ac = DeliveryActorCritic(gamble_encoder, clm_emb_size=16, co_emb_size=32, gmb_emb_size=8, temperature=1.0,
                             device=None)
    state1 = DeliveryState(
        claim_emb=np.zeros((6,)),
        couriers_embs=np.zeros((5, 4)),
        orders_embs=None,
        prev_idxs=[2, 3],
        orders_full_masks=[],
        claim_to_couries_dists=np.array(list(range(5))),
        gamble_features=np.zeros((5,)),
        claim_idx=0
    )
    state2 = DeliveryState(
        claim_emb=np.zeros((6,)),
        couriers_embs=np.zeros((2, 4)),
        orders_embs=np.zeros((1, 14)),
        prev_idxs=[],
        orders_full_masks=[False],
        claim_to_couries_dists=np.array(list(range(3))),
        gamble_features=np.zeros((5,)),
        claim_idx=1
    )
    pol_tens, val_tens = ac._make_padded_policy_value_tensors([state1, state2])
    assert pol_tens.shape == (2, 6)
    assert val_tens.shape == (2,)
