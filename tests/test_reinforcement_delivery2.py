import torch
import numpy as np
import json
from datetime import datetime, timedelta

from src.objects import Assignment, Gamble, Point, Courier, Claim
from src.reinforcement.delivery2 import (
    DeliveryState2,
    DeliveryAction2,
    DeliveryEnvironment2,
    DeliveryActorCritic2,
    DeliveryRewarder2,
)
from src.simulator.simulator import Simulator, DataReader
from src.router_makers import BaseRouteMaker
from src.networks.encoders import GambleEncoder
from src.networks.backbones import TransformerBackbone


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


def test_delivery_environment2(tmp_path):
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
    rewarder = DeliveryRewarder2(coef_reward_assigned=0.1, coef_reward_cancelled=1.0, coef_reward_distance=0.0,
                                 coef_reward_completed=0.0)
    env = DeliveryEnvironment2(simulator=sim, rewarder=rewarder, max_num_points_in_route=4,
                               num_gambles_in_day=6, device=None)
    state1 = env.reset()
    assert env._iter == 1, env._iter
    assert state1.claim_embs.shape == (2, 6)
    assert np.isclose(state1.claim_embs,
                      [[0.0, 0.2, 0.0, 1.0, 0.0, 20.0], [1.0, 0.2, 1.0, 1.0, 0.0, 30.0]]).all(), (state1.claim_embs)
    assert state1.couriers_embs.shape == (3, 4)
    assert np.isclose(state1.couriers_embs, [[1.0, 0.0, 0.0, 30.0], [0.5, 1.0, 0.0, 20.0], [0.0, 0.0, 0.0, 30.0]]).all()
    assert state1.orders_embs is None, state1.orders_embs

    state2, reward, done, info = env.step(DeliveryAction2([2, 0]))
    assert env._iter == 3, env._iter
    assert not done
    assert state2.claim_embs.shape == (1, 6)
    assert np.isclose(state2.claim_embs, [[0.5, 0.0, 0.5, 1.0, 0.0, 20.0]]).all(), (state2.claim_embs)
    assert state2.orders_embs is None, state2.orders_embs
    assert np.isclose(state2.couriers_embs, [[0.5, 1.0, 0.0, 80.0], [0.0, 1.0, 0.0, 90.0], [1.0, 1.0, 0.0, 90.0]]).all()

    state4, reward, done, info = env.step(DeliveryAction2([0]))
    assert env._iter == 5, env._iter
    assert not done
    assert state4.claim_embs.shape == (1, 6)
    assert np.isclose(state4.claim_embs, [[0.0, 0.8, 0.0, 0.0, 0.0, 20.0]]).all(), (state4.claim_embs)
    assert np.isclose(state4.couriers_embs, [[0.0, 1.0, 0.0, 150.0], [1.0, 1.0, 0.0, 150.0]]).all()
    assert len(state4.orders_embs) == 1
    assert np.isclose(state4.orders_embs[0], [0.5, 0.2, 1.0, 140.0, 1.0, 60.0, 0.5, 1.0] + [0.0] * 6).all()

    state5, reward, done, info = env.step(DeliveryAction2([2]))
    assert done, env._iter


def test_delivery2_actor_critic_shape(tmp_path):
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
    rewarder = DeliveryRewarder2(coef_reward_assigned=0.1, coef_reward_cancelled=1.0, coef_reward_distance=0.0,
                                 coef_reward_completed=0.0)
    env = DeliveryEnvironment2(simulator=sim, rewarder=rewarder, max_num_points_in_route=4,
                               num_gambles_in_day=6, device=None)
    gamble_encoder = GambleEncoder(
        route_embedding_dim=128,
        claim_embedding_dim=16,
        point_embedding_dim=16,
        cat_points_embedding_dim=8,
        courier_order_embedding_dim=32,
        max_num_points_in_route=4,
        device=None,
        use_pretrained_encoders=False,
    )
    model = TransformerBackbone(
        claim_embedding_dim=16,
        courier_order_embedding_dim=32,
        nhead=4,
        hidden_dim=15,
        dim_feedforward=37,
        num_encoder_layers=1,
        num_decoder_layers=2,
        device=None,
    )
    ac = DeliveryActorCritic2(gamble_encoder, backbone=model, clm_emb_size=16,
                              crr_ord_emb_size=32, temperature=1.0, device=None)
    state1 = env.reset()  # (2, 3, 0)
    co_embs, claim_embs = ac._make_three_tensors_from_state(state1)
    assert co_embs.shape == (4, 32), co_embs.shape
    assert claim_embs.shape == (2, 16)

    state2, reward, done, info = env.step(DeliveryAction2([2, 0]))  # (1, 3, 0)
    state4, reward, done, info = env.step(DeliveryAction2([0]))  # (1, 2, 1)
    co_embs, claim_emb = ac._make_three_tensors_from_state(state4)
    assert co_embs.shape == (4, 32), co_embs.shape
    assert claim_emb.shape == (1, 16)


def test_delivery2_actor_critic(tmp_path):
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
    rewarder = DeliveryRewarder2(coef_reward_assigned=0.1, coef_reward_cancelled=1.0, coef_reward_distance=0.0,
                                 coef_reward_completed=0.0)
    env = DeliveryEnvironment2(simulator=sim, rewarder=rewarder, max_num_points_in_route=4,
                               num_gambles_in_day=6, device=None)
    gamble_encoder = GambleEncoder(
        route_embedding_dim=128,
        claim_embedding_dim=16,
        point_embedding_dim=16,
        cat_points_embedding_dim=8,
        courier_order_embedding_dim=32,
        max_num_points_in_route=4,
        device=None,
        use_pretrained_encoders=False,
    )
    model = TransformerBackbone(
        claim_embedding_dim=16,
        courier_order_embedding_dim=32,
        nhead=4,
        hidden_dim=15,
        dim_feedforward=37,
        num_encoder_layers=1,
        num_decoder_layers=2,
        device=None,
    )
    ac = DeliveryActorCritic2(gamble_encoder, backbone=model, clm_emb_size=16,
                              crr_ord_emb_size=32, temperature=1.0, device=None)

    state1 = DeliveryState2(
        claim_embs=np.zeros((1, 6)),
        couriers_embs=np.zeros((5, 4)),
        orders_embs=None,
        orders_full_masks=[],
        claims_to_couries_dists=np.array([])
    )
    state2 = DeliveryState2(
        claim_embs=np.zeros((3, 6)),
        couriers_embs=np.zeros((2, 4)),
        orders_embs=np.zeros((1, 14)),
        orders_full_masks=[False],
        claims_to_couries_dists=np.array([])
    )

    states = [state1, state2]
    ac(states)
    policy = ac.get_log_probs_tensor()
    value = ac.get_values_tensor()
    
    assert value.shape == (2,), value
    assert policy.shape == (2, 3, 6), policy
    assert isinstance(ac.get_actions_list(), list), ac.get_actions_list()
    assert isinstance(ac.get_log_probs_list(), list), ac.get_log_probs_list()
    assert isinstance(ac.get_actions_list(), list), ac.get_actions_list()

    assert (policy[0, 1:, :]).isclose(torch.ones((2, 6)) * -np.log(6)).all()
    assert (policy[1, :, 4:]).isclose(torch.ones((3, 2)) * -1e9).all()
