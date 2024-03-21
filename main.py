import json
import torch
from pathlib import Path

from src.simulator.simulator import Simulator
from src.simulator.data_reader import DataReader
from src.router_makers import AppendRouteMaker
from src.database.database import Database, Metric, DatabaseLogger
from src.dispatchs.hungarian_dispatch import HungarianDispatch, BaseDispatch
from src.dispatchs.greedy_dispatch import GreedyDispatch
from src.dispatchs.random_dispatch import RandomDispatch
from src.dispatchs.scorers import DistanceScorer
from src.dispatchs.neural_sequantial_dispatch import NeuralSequantialDispatch
from src.networks.encoders import GambleEncoder
from src.reinforcement.delivery import DeliveryActorCritic
from src.evaluation import evaluate


def main():
    db = Database(Path('history.db'))
    db.clear()

    # print('Hungarian')
    # res = evaluate(
    #     dispatch=HungarianDispatch(DistanceScorer()),
    #     run_id=0,
    #     simulator_cfg_path='configs/simulator.json',
    #     sampler_mode='distr_sampler',
    #     max_num_points_in_route=8,
    #     history_db_path='history.db',
    #     eval_num_simulator_steps=200,
    # )
    # print(res)

    # print('Greedy')
    # res = evaluate(
    #     dispatch=GreedyDispatch(DistanceScorer()),
    #     run_id=1,
    #     simulator_cfg_path='configs/simulator.json',
    #     sampler_mode='distr_sampler',
    #     max_num_points_in_route=8,
    #     history_db_path='history.db',
    #     eval_num_simulator_steps=200,
    # )
    # print(res)

    print('Random')
    res = evaluate(
        dispatch=RandomDispatch(),
        run_id=2,
        simulator_cfg_path='configs/simulator.json',
        sampler_mode='distr_sampler',
        max_num_points_in_route=4,
        history_db_path='history.db',
        eval_num_simulator_steps=200,
    )
    print(res)

    print('Neural')
    max_num_points_in_route = 4
    with open('configs/network.json') as f:
        net_cfg = json.load(f)['encoder']
    encoder = GambleEncoder(
        order_embedding_dim=net_cfg['order_embedding_dim'],
        claim_embedding_dim=net_cfg['claim_embedding_dim'],
        courier_embedding_dim=net_cfg['courier_embedding_dim'],
        route_embedding_dim=net_cfg['route_embedding_dim'],
        point_embedding_dim=net_cfg['point_embedding_dim'],
        number_embedding_dim=net_cfg['number_embedding_dim'],
        max_num_points_in_route=max_num_points_in_route,
        device=None,
    )
    ac = DeliveryActorCritic(gamble_encoder=encoder, clm_emb_size=net_cfg['claim_embedding_dim'], device=None,
                             temperature=1.0)
    ac.load_state_dict(torch.load('checkpoints/6313c9d40bce480f8b1416a0f0976544.pt', map_location='cpu'))
    dsp = NeuralSequantialDispatch(actor_critic=ac, max_num_points_in_route=max_num_points_in_route)
    res = evaluate(
        dispatch=dsp,
        run_id=3,
        simulator_cfg_path='configs/simulator.json',
        sampler_mode='distr_sampler',
        max_num_points_in_route=4,
        history_db_path='history.db',
        eval_num_simulator_steps=200,
    )
    print(res)


def debug():
    db_path = Path('history.db')
    db = Database(db_path)
    res = db.select('select * from claims;')
    print(res)


if __name__ == '__main__':
    main()
    # debug()
