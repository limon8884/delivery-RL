import json
import torch
import numpy as np
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
    sample_mode = 'dummy_sampler'
    model_size = 'medium'
    max_num_points_in_route = 2
    eval_num_simulator_steps = 200
    n_runs = 3

    print('Hungarian')
    res = evaluate(
        dispatch=HungarianDispatch(DistanceScorer()),
        run_id=0,
        simulator_cfg_path='configs/simulator.json',
        sampler_mode=sample_mode,
        max_num_points_in_route=max_num_points_in_route,
        history_db_path='history.db',
        eval_num_simulator_steps=eval_num_simulator_steps,
    )
    print(res)

    print('Greedy')
    res = evaluate(
        dispatch=GreedyDispatch(DistanceScorer()),
        run_id=1,
        simulator_cfg_path='configs/simulator.json',
        sampler_mode=sample_mode,
        max_num_points_in_route=max_num_points_in_route,
        history_db_path='history.db',
        eval_num_simulator_steps=eval_num_simulator_steps,
    )
    print(res)

    print('Random')
    ress = []
    for run in range(n_runs):
        ress = []
        db.clear()
        res = evaluate(
            dispatch=RandomDispatch(),
            run_id=run,
            simulator_cfg_path='configs/simulator.json',
            sampler_mode=sample_mode,
            max_num_points_in_route=max_num_points_in_route,
            history_db_path='history.db',
            eval_num_simulator_steps=eval_num_simulator_steps,
        )
        # ress.append(res['CR'])
        ress.append(res)
    # print(np.mean(ress))
    print(ress)

    print('Neural')
    ress = []
    for run in range(n_runs):
        with open('configs/network.json') as f:
            net_cfg = json.load(f)['encoder'][model_size]
        encoder = GambleEncoder(
            order_embedding_dim=net_cfg['order_embedding_dim'],
            claim_embedding_dim=net_cfg['claim_embedding_dim'],
            courier_embedding_dim=net_cfg['courier_embedding_dim'],
            route_embedding_dim=net_cfg['route_embedding_dim'],
            point_embedding_dim=net_cfg['point_embedding_dim'],
            number_embedding_dim=net_cfg['number_embedding_dim'],
            max_num_points_in_route=max_num_points_in_route,
            dropout=0.2,
            device=None,
        )
        ac = DeliveryActorCritic(gamble_encoder=encoder, clm_emb_size=net_cfg['claim_embedding_dim'], device=None,
                                 temperature=1.0)
        # ac.load_state_dict(torch.load('checkpoints/f091c4ff33874a6bbf160403623507eb.pt', map_location='cpu'))
        dsp = NeuralSequantialDispatch(actor_critic=ac, max_num_points_in_route=max_num_points_in_route)
        db.clear()
        res = evaluate(
            dispatch=dsp,
            run_id=run,
            simulator_cfg_path='configs/simulator.json',
            sampler_mode=sample_mode,
            max_num_points_in_route=max_num_points_in_route,
            history_db_path='history.db',
            eval_num_simulator_steps=eval_num_simulator_steps,
        )
        # ress.append(res['CR'])
        ress.append(res)
    # print(np.mean(ress))
    print(ress)


def debug():
    with open('configs/network.json') as f:
        net_cfg = json.load(f)['encoder']
    encoder = GambleEncoder(
        order_embedding_dim=net_cfg['order_embedding_dim'],
        claim_embedding_dim=net_cfg['claim_embedding_dim'],
        courier_embedding_dim=net_cfg['courier_embedding_dim'],
        route_embedding_dim=net_cfg['route_embedding_dim'],
        point_embedding_dim=net_cfg['point_embedding_dim'],
        number_embedding_dim=net_cfg['number_embedding_dim'],
        max_num_points_in_route=2,
        device=None,
    )
    param_size = 0
    for param in encoder.parameters():
        param_size += param.nelement()
    buffer_size = 0
    for buffer in encoder.buffers():
        buffer_size += buffer.nelement()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}M'.format(size_all_mb))


if __name__ == '__main__':
    main()
    # debug()
