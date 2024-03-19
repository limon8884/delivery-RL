import json
import torch
from pathlib import Path

from src.simulator.simulator import Simulator
from src.simulator.data_reader import DataReader
from src.router_makers import AppendRouteMaker
from src.database.database import Database, Metric, Logger
from src.dispatchs.hungarian_dispatch import HungarianDispatch, BaseDispatch
from src.dispatchs.greedy_dispatch import GreedyDispatch
from src.dispatchs.random_dispatch import RandomDispatch
from src.dispatchs.scorers import DistanceScorer
from src.dispatchs.neural_sequantial_dispatch import NeuralSequantialDispatch
from src.networks.encoders import GambleEncoder
from src.reinforcement.delivery import DeliveryActorCritic
from src.evaluation import evaluate


def run_dsp(dsp: BaseDispatch, config_path: Path, db_path: Path, run_id: int, max_num_points_in_route: int) -> None:
    logger = Logger(run_id=run_id)
    reader = DataReader.from_config(config_path=config_path, sampler_mode='dummy_sampler', logger=logger)
    route_maker = AppendRouteMaker(max_points_lenght=max_num_points_in_route, cutoff_radius=0.0)
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=config_path, logger=logger)
    try:
        sim.run(dsp, num_iters=288)
    finally:
        db = Database(db_path)
        db.export_from_logger(logger)
        # cr, ctd = db.get_metric(Metric.CR, run_id), db.get_metric(Metric.CTD, run_id)
        print(f'CR: {db.get_metric(Metric.CR, run_id)}\n \
            CTD: {db.get_metric(Metric.CTD, run_id)}\n \
            \n')


def main():
    simulator_config_path = Path('configs/simulator.json')
    network_config_path = Path('configs/network.json')
    db_path = Path('history.db')
    db = Database(db_path)
    db.clear()

    print('Hungarian')
    res = evaluate(
        dispatch=HungarianDispatch(DistanceScorer()),
        run_id=0,
        simulator_cfg_path='configs/simulator.json',
        sampler_mode='distr_sampler',
        max_num_points_in_route=8,
        history_db_path='history.db',
        eval_num_simulator_steps=200,
    )
    print(res)

    print('Greedy')
    res = evaluate(
        dispatch=GreedyDispatch(DistanceScorer()),
        run_id=1,
        simulator_cfg_path='configs/simulator.json',
        sampler_mode='distr_sampler',
        max_num_points_in_route=8,
        history_db_path='history.db',
        eval_num_simulator_steps=200,
    )
    print(res)

    print('Greedy')
    res = evaluate(
        dispatch=RandomDispatch(),
        run_id=1,
        simulator_cfg_path='configs/simulator.json',
        sampler_mode='distr_sampler',
        max_num_points_in_route=8,
        history_db_path='history.db',
        eval_num_simulator_steps=200,
    )
    print(res)

    # print('Neural')
    # max_num_points_in_route = 8
    # with open(network_config_path) as f:
    #     net_cfg = json.load(f)['encoder']
    # encoder = GambleEncoder(
    #     order_embedding_dim=net_cfg['order_embedding_dim'],
    #     claim_embedding_dim=net_cfg['claim_embedding_dim'],
    #     courier_embedding_dim=net_cfg['courier_embedding_dim'],
    #     route_embedding_dim=net_cfg['route_embedding_dim'],
    #     point_embedding_dim=net_cfg['point_embedding_dim'],
    #     number_embedding_dim=net_cfg['number_embedding_dim'],
    #     max_num_points_in_route=max_num_points_in_route,
    #     device=None,
    # )
    # ac = DeliveryActorCritic(gamble_encoder=encoder, clm_emb_size=net_cfg['claim_embedding_dim'], device=None,
    #                          temperature=1.0)
    # ac.load_state_dict(torch.load('checkpoints/234.pt', map_location='cpu'))
    # run_dsp(NeuralSequantialDispatch(actor_critic=ac, max_num_points_in_route=max_num_points_in_route),
    #         simulator_config_path, db_path, run_id=3,
    #         max_num_points_in_route=max_num_points_in_route)


def debug():
    db_path = Path('history.db')
    db = Database(db_path)
    res = db.select('select * from claims;')
    print(res)


if __name__ == '__main__':
    main()
    # debug()
